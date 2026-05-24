"""The base MNEMOSYNE agent.

An agent is a transformer with first-class introspection. Concretely,
an Agent instance bundles:

* a **HookedTransformer** — the language-modeling backbone
* a **TopKSAE per introspectable layer** — the agent's dictionary of
  features it can name and reason about
* an **AgentMemory** — working / episodic / semantic stores
* a **communication identity** — registered with a CommunicationChannel
* an **introspection API** — methods like ``.why_did_i_say(token)`` that
  return causal attributions over the agent's own features

The design goal: an agent should be able to answer, in addition to
"what is the answer to X?", these questions:

  - "Which features of mine were active when I answered X?"
  - "Which features caused me to choose X over Y?"
  - "If feature 42 had not fired, what would I have said?"
  - "Have I seen something like this before?"
  - "What concept does this input most match?"

These aren't post-hoc explanations bolted on after the fact. They are
the agent's normal operating mode, exposed through the same interface
as text generation. The intent is to make introspection cheap enough
that the agent can use it *during reasoning* rather than only after.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mnemosyne.arch.tokenizer import Tokenizer
from mnemosyne.arch.transformer import (
    HookContext, HookedTransformer, TransformerConfig, hooks,
)
from mnemosyne.causal.interventions import (
    Counterfactual, FeatureAttribution, feature_attribution,
    find_counterfactual,
)
from mnemosyne.communication.channel import CommunicationChannel, Message
from mnemosyne.interp.sae import SAEConfig, TopKSAE
from mnemosyne.memory.hierarchical import AgentMemory, Episode


@dataclass
class AgentConfig:
    """Configuration for a single agent."""
    name: str
    role: str
    transformer_cfg: TransformerConfig = field(default_factory=TransformerConfig)
    sae_cfg: SAEConfig = field(default_factory=SAEConfig)
    introspection_sites: tuple[str, ...] = ("block_2.resid_post",)
    working_memory_capacity: int = 8
    max_episodes: int = 4096
    max_concepts: int = 64

    def __post_init__(self) -> None:
        # Keep SAE input dim aligned with the transformer's hidden dim.
        if self.sae_cfg.d_model != self.transformer_cfg.hidden_dim:
            self.sae_cfg = SAEConfig(
                d_model=self.transformer_cfg.hidden_dim,
                n_features=self.sae_cfg.n_features,
                k=self.sae_cfg.k,
                dead_threshold=self.sae_cfg.dead_threshold,
                aux_k=self.sae_cfg.aux_k,
            )


@dataclass
class IntrospectionReport:
    """Structured output of an introspection call."""
    target_token: int
    site: str
    top_features: list[FeatureAttribution]
    counterfactual: Optional[Counterfactual]
    most_similar_episode_id: Optional[int]
    matched_concept_label: Optional[str]
    summary: str = ""

    def __repr__(self) -> str:
        return (f"IntrospectionReport(token={self.target_token}, "
                f"site={self.site!r}, "
                f"{len(self.top_features)} features, "
                f"counterfactual={'yes' if self.counterfactual else 'no'})")


class Agent(nn.Module):
    """A single MNEMOSYNE agent.

    Sub-classed by :mod:`mnemosyne.agents.specialists` for role-specific
    behaviors (Proposer, Critic, Verifier, Synthesizer, Metacognitor).
    The base class implements the general capabilities common to all
    of them: forward pass, introspection, memory access, communication.
    """

    def __init__(self, cfg: AgentConfig, tokenizer: Tokenizer,
                 channel: Optional[CommunicationChannel] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer

        # Backbone.
        # Make sure transformer's vocab matches the tokenizer.
        if cfg.transformer_cfg.vocab_size != tokenizer.vocab_size:
            cfg.transformer_cfg = TransformerConfig(
                **{**cfg.transformer_cfg.__dict__,
                   "vocab_size": tokenizer.vocab_size},
            )
        self.transformer = HookedTransformer(cfg.transformer_cfg)

        # One SAE per introspectable site. Stored as a ModuleDict for
        # easy state-dict handling.
        self.saes = nn.ModuleDict({
            self._safe_key(site): TopKSAE(cfg.sae_cfg)
            for site in cfg.introspection_sites
        })

        # Memory.
        self.memory = AgentMemory.build(
            hidden_dim=cfg.transformer_cfg.hidden_dim,
            n_features=cfg.sae_cfg.n_features,
            working_capacity=cfg.working_memory_capacity,
            max_episodes=cfg.max_episodes,
            max_concepts=cfg.max_concepts,
        )

        # Communication.
        self.channel = channel
        if channel is not None:
            channel.register(cfg.name)

        # A learned projection from final hidden state into the latent
        # communication bandwidth. Initialized to identity for stability.
        self.send_head = nn.Linear(cfg.transformer_cfg.hidden_dim,
                                     cfg.transformer_cfg.hidden_dim, bias=False)
        nn.init.eye_(self.send_head.weight)
        self.recv_head = nn.Linear(cfg.transformer_cfg.hidden_dim,
                                     cfg.transformer_cfg.hidden_dim, bias=False)
        nn.init.eye_(self.recv_head.weight)

    # ──────────────────────────────────────────────────────────────────
    # Naming helpers
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _safe_key(site: str) -> str:
        """ModuleDict keys can't contain '.', so replace with '__'."""
        return site.replace(".", "__")

    @staticmethod
    def _from_safe_key(key: str) -> str:
        return key.replace("__", ".")

    # ──────────────────────────────────────────────────────────────────
    # Forward pass with introspection-friendly capture
    # ──────────────────────────────────────────────────────────────────
    def encode(self, text: str) -> torch.Tensor:
        ids = self.tokenizer.encode(text)
        return torch.tensor([ids], dtype=torch.long, device=self._device())

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, token_ids: torch.Tensor
                  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run the transformer; capture activations at all introspection sites."""
        sites = list(self.cfg.introspection_sites)
        logits, captured = self.transformer.run_with_capture(token_ids, sites=sites)
        return logits, captured

    def reply(self, text: str, max_new_tokens: int = 32,
                temperature: float = 0.0, log_to_working: bool = True
                ) -> tuple[str, torch.Tensor, dict[str, torch.Tensor]]:
        """Generate a response in token space.

        Returns ``(text, final_logits, final_captured_activations)``.
        Greedy if ``temperature == 0``, otherwise temperature sampling.
        """
        ids = self.encode(text)
        # Truncate so we leave room for new tokens within max_seq_len.
        max_input = self.cfg.transformer_cfg.max_seq_len - max_new_tokens - 4
        if ids.shape[1] > max_input:
            ids = ids[:, -max_input:]
        for _ in range(max_new_tokens):
            logits, _ = self.transformer(ids)
            next_logits = logits[0, -1] / max(temperature, 1e-6)
            if temperature <= 0:
                next_id = int(next_logits.argmax().item())
            else:
                probs = F.softmax(next_logits, dim=-1)
                next_id = int(torch.multinomial(probs, 1).item())
            ids = torch.cat([ids, torch.tensor([[next_id]], device=ids.device)],
                              dim=-1)
            if next_id == self.tokenizer.special_id("<eos>"):
                break
        # Final capture for introspection.
        final_logits, captured = self.forward(ids)
        out_text = self.tokenizer.decode(ids[0].tolist())
        if log_to_working:
            site = self.cfg.introspection_sites[0]
            hidden = captured[site][0, -1]
            sae = self.saes[self._safe_key(site)]
            z = sae.encode(hidden.unsqueeze(0)).squeeze(0)
            self.memory.working.push(hidden, sparse_code=z,
                                      meta={"text_in": text, "text_out": out_text})
        return out_text, final_logits, captured

    # ──────────────────────────────────────────────────────────────────
    # Introspection
    # ──────────────────────────────────────────────────────────────────
    def introspect(self, token_ids: torch.Tensor,
                   site: Optional[str] = None,
                   target_token: Optional[int] = None,
                   compute_counterfactual: bool = True,
                   top_n: int = 6) -> IntrospectionReport:
        """The canonical "why did I say that?" call.

        Runs feature attribution at ``site`` (default: the first
        introspection site), optionally searches for a counterfactual,
        and looks up the most similar episodic memory.
        """
        if site is None:
            site = self.cfg.introspection_sites[0]
        sae = self.saes[self._safe_key(site)]
        logits, _ = self.forward(token_ids)
        if target_token is None:
            target_token = int(logits[0, -1].argmax().item())

        top = feature_attribution(self.transformer, sae, site,
                                    token_ids, target_token=target_token, top_n=top_n)
        cf = None
        if compute_counterfactual:
            cf = find_counterfactual(self.transformer, sae, site, token_ids)

        # Episodic recall.
        sites_captured = self.transformer.run_with_capture(token_ids, sites=[site])[1]
        query_key = sites_captured[site][0, -1]  # last-token hidden
        matches = self.memory.episodic.retrieve(query_key, k=1,
                                                  similarity_threshold=0.3)
        ep_idx = None
        if matches:
            ep = matches[0][0]
            ep_idx = self.memory.episodic._episodes.index(ep)

        # Semantic concept match.
        z_last = sae.encode(query_key.unsqueeze(0)).squeeze(0)
        concept_match = self.memory.semantic.assign(z_last)
        concept_label = None
        if concept_match is not None:
            concept_label = self.memory.semantic._concepts[concept_match[0]].label

        summary = self._format_introspection_summary(top, cf, concept_label)
        return IntrospectionReport(
            target_token=target_token, site=site,
            top_features=top, counterfactual=cf,
            most_similar_episode_id=ep_idx,
            matched_concept_label=concept_label,
            summary=summary,
        )

    def _format_introspection_summary(
        self,
        top: list[FeatureAttribution],
        cf: Optional[Counterfactual],
        concept_label: Optional[str],
    ) -> str:
        lines = [f"agent={self.cfg.name!r} role={self.cfg.role!r} introspection:"]
        if top:
            lines.append("  top features causing this answer:")
            for f in top[:5]:
                lines.append(f"    feature_{f.feature_idx:>3d}  "
                              f"act={f.activation:+.3f}  Δlogit={f.delta_logit:+.3f}")
        if cf:
            lines.append(
                f"  counterfactual: ablating features {cf.ablated_features} "
                f"would have flipped my answer "
                f"({cf.original_token} → {cf.counterfactual_token})"
            )
        else:
            lines.append("  counterfactual: no flip found within budget — "
                          "answer was robustly determined")
        if concept_label:
            lines.append(f"  closest semantic concept: {concept_label!r}")
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────
    # Memory operations
    # ──────────────────────────────────────────────────────────────────
    def remember(self, input_text: str, output_text: str,
                  outcome: dict, captured: dict[str, torch.Tensor]) -> int:
        """Store this interaction as an episode."""
        # Use the first introspection site's last-token activation as the key.
        site = self.cfg.introspection_sites[0]
        hidden = captured[site][0, -1].detach().clone()
        sae = self.saes[self._safe_key(site)]
        feat_sig = sae.encode(hidden.unsqueeze(0)).squeeze(0).detach().clone()
        ep = Episode(
            key=hidden,
            input_text=input_text,
            output_text=output_text,
            feature_signature=feat_sig,
            outcome=outcome,
        )
        return self.memory.episodic.store(ep)

    def recall(self, text: str, k: int = 4) -> list[tuple[Episode, float]]:
        """Find episodes most similar to a given input."""
        ids = self.encode(text)
        site = self.cfg.introspection_sites[0]
        _, captured = self.forward(ids)
        query = captured[site][0, -1]
        return self.memory.episodic.retrieve(query, k=k)

    # ──────────────────────────────────────────────────────────────────
    # Communication
    # ──────────────────────────────────────────────────────────────────
    def speak(self, recipient: str, text: str,
                latent: Optional[torch.Tensor] = None,
                metadata: Optional[dict] = None) -> Optional[Message]:
        if self.channel is None:
            return None
        return self.channel.send(self.cfg.name, recipient,
                                  token_text=text, latent=latent,
                                  metadata=metadata)

    def listen(self) -> list[Message]:
        if self.channel is None:
            return []
        return self.channel.inbox(self.cfg.name)

    # ──────────────────────────────────────────────────────────────────
    # Self-modeling: a small head that predicts the agent's own next action
    # ──────────────────────────────────────────────────────────────────
    def predict_own_output(self, token_ids: torch.Tensor) -> torch.Tensor:
        """The agent's prediction of what it will output, computed from the
        same final hidden state as the LM head. This is the trivial
        self-model — every agent has it for free, and it serves as the
        baseline against which more sophisticated self-models compare."""
        logits, _ = self.transformer(token_ids)
        return logits[0, -1]
