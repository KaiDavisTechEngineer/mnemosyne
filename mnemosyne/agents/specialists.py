"""The five specialist agents in a MNEMOSYNE society.

A society is organized around five complementary roles, modeled on a
common decomposition of expert problem-solving:

1. **Proposer** — generates candidate answers. Optimized for
   exploration: high temperature, broad attention, willing to take
   risks. The proposer's introspection focuses on *which features
   prompted this candidate*.

2. **Critic** — finds flaws in proposals. Optimized for negative
   space: trained on identifying counterexamples, missing cases,
   logical gaps. The critic's introspection focuses on *which features
   triggered objection*.

3. **Verifier** — formally checks proposals when checkable.
   Wraps the Z3 SMT verifier from PROOFCAST as a tool. The verifier
   is mostly deterministic; its "model" is just enough neural
   machinery to translate proposals into SMT and decide when to call.

4. **Synthesizer** — combines proposer ideas with critic objections to
   produce refined answers. Optimized for integration: features that
   light up on synthesizer tend to correspond to "this resolves a
   tension."

5. **Metacognitor** — observes the other four and maintains a
   learned model of *their* behavior. Predicts what each agent will
   say next, scores how well that prediction matches reality, and
   updates trust accordingly.

Each subclass is intentionally short: they share the heavy lifting
(transformer, SAE, memory, communication) from the base Agent class
and just specialize the prompting, temperature, and the introspection
heuristics.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mnemosyne.agents.base import Agent, AgentConfig, IntrospectionReport
from mnemosyne.arch.tokenizer import Tokenizer
from mnemosyne.communication.channel import CommunicationChannel, Message


# ─────────────────────────────────────────────────────────────────────
# Proposer
# ─────────────────────────────────────────────────────────────────────
class Proposer(Agent):
    """Generates candidate solutions. Tuned for exploration."""

    def __init__(self, cfg: AgentConfig, tokenizer: Tokenizer,
                 channel: Optional[CommunicationChannel] = None) -> None:
        cfg.role = "proposer"
        super().__init__(cfg, tokenizer, channel)
        # Sampling temperature for proposal generation. Higher than the
        # other agents because we want diverse proposals.
        self.temperature = 1.0

    def propose(self, question: str, k: int = 1) -> list[str]:
        """Generate k candidate proposals."""
        out: list[str] = []
        prompt = f"{self.tokenizer.role_token('proposer')}<msg>{question}</msg>"
        for _ in range(k):
            text, _, _ = self.reply(prompt, max_new_tokens=24,
                                      temperature=self.temperature)
            out.append(text)
        if self.channel:
            for o in out:
                self.speak("all", o, metadata={"phase": "propose"})
        return out


# ─────────────────────────────────────────────────────────────────────
# Critic
# ─────────────────────────────────────────────────────────────────────
class Critic(Agent):
    """Finds flaws in proposals. Tuned for skeptical analysis."""

    def __init__(self, cfg: AgentConfig, tokenizer: Tokenizer,
                 channel: Optional[CommunicationChannel] = None) -> None:
        cfg.role = "critic"
        super().__init__(cfg, tokenizer, channel)
        # Lower temperature — critic should be precise about objections.
        self.temperature = 0.2

    def critique(self, question: str, proposal: str) -> str:
        prompt = (
            f"{self.tokenizer.role_token('critic')}<msg>question: {question} | "
            f"proposal: {proposal}</msg>"
        )
        text, _, _ = self.reply(prompt, max_new_tokens=24,
                                  temperature=self.temperature)
        if self.channel:
            self.speak("all", text, metadata={"phase": "critique"})
        return text

    def find_disagreement_features(self, question: str, proposal: str,
                                     ) -> IntrospectionReport:
        """Return an introspection report localized to the features that
        fired most when reading the proposal. Useful for the
        metacognitor to learn what triggers critic objections."""
        prompt = (
            f"{self.tokenizer.role_token('critic')}<msg>{question} → {proposal}</msg>"
        )
        ids = self.encode(prompt)
        return self.introspect(ids, compute_counterfactual=False, top_n=6)


# ─────────────────────────────────────────────────────────────────────
# Verifier
# ─────────────────────────────────────────────────────────────────────
class Verifier(Agent):
    """Formal verifier. Mostly mechanical — the neural net only decides
    *what to check*, not the answer."""

    def __init__(self, cfg: AgentConfig, tokenizer: Tokenizer,
                 channel: Optional[CommunicationChannel] = None) -> None:
        cfg.role = "verifier"
        super().__init__(cfg, tokenizer, channel)
        self.temperature = 0.0  # deterministic

    def verify(self, question: str, proposal: str,
                verification_fn=None) -> dict:
        """Run formal verification on a proposal.

        ``verification_fn`` is a user-supplied callable that takes
        ``(question, proposal)`` and returns a dict
        ``{"verdict": "ok"|"fail"|"unknown", "reason": str, ...}``.
        If not supplied, we use a placeholder that always returns
        "unknown" — real applications would plug in their own (e.g.,
        Z3 for logical claims, a test runner for code, a calculator
        for arithmetic).
        """
        if verification_fn is None:
            verdict = {"verdict": "unknown",
                        "reason": "no verification function configured"}
        else:
            verdict = verification_fn(question, proposal)
        msg = (f"verdict={verdict['verdict']}"
               + (f" reason={verdict.get('reason','')}"
                  if verdict.get('reason') else ""))
        if self.channel:
            self.speak("all", msg, metadata={"phase": "verify", **verdict})
        return verdict


# ─────────────────────────────────────────────────────────────────────
# Synthesizer
# ─────────────────────────────────────────────────────────────────────
class Synthesizer(Agent):
    """Combines proposals + critiques + verifications into a final answer."""

    def __init__(self, cfg: AgentConfig, tokenizer: Tokenizer,
                 channel: Optional[CommunicationChannel] = None) -> None:
        cfg.role = "synthesizer"
        super().__init__(cfg, tokenizer, channel)
        self.temperature = 0.5  # moderate

    def synthesize(self, question: str, proposals: list[str],
                    critiques: list[str], verifications: list[dict],
                    ) -> str:
        ctx_parts = [f"question: {question}"]
        for i, p in enumerate(proposals):
            ctx_parts.append(f"proposal_{i}: {p}")
        for i, c in enumerate(critiques):
            ctx_parts.append(f"critique_{i}: {c}")
        for i, v in enumerate(verifications):
            ctx_parts.append(f"verify_{i}: {v.get('verdict','?')}")
        prompt = (
            f"{self.tokenizer.role_token('synthesizer')}"
            f"<msg>{' | '.join(ctx_parts)}</msg>"
        )
        text, _, _ = self.reply(prompt, max_new_tokens=32,
                                  temperature=self.temperature)
        if self.channel:
            self.speak("all", text, metadata={"phase": "synthesize"})
        return text


# ─────────────────────────────────────────────────────────────────────
# Metacognitor — the agent that models the other agents
# ─────────────────────────────────────────────────────────────────────
class AgentModel(nn.Module):
    """A small neural model that the metacognitor uses to *predict* what
    another agent will say next.

    The metacognitor maintains one of these per modeled agent.
    Architecturally it's a tiny transformer that consumes the recent
    message history and outputs a distribution over likely next
    messages — concretely, over a few discrete "behavioral modes."
    """
    def __init__(self, vocab_size: int, hidden_dim: int = 64,
                  n_modes: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.mode_head = nn.Linear(hidden_dim, n_modes)
        self.confidence_head = nn.Linear(hidden_dim, 1)

    def forward(self, token_ids: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mode_logits, predicted_confidence)."""
        x = self.embed(token_ids)
        _, h = self.gru(x)
        h = h.squeeze(0)
        return self.mode_head(h), torch.sigmoid(self.confidence_head(h)).squeeze(-1)


class Metacognitor(Agent):
    """Observes the other agents and maintains a learned model of each.

    The metacognitor is unique among MNEMOSYNE agents in that its
    *output* is not a candidate answer — it's a *prediction*. After
    each round, it ranks the other agents by how reliable their
    contributions have been, suggests trust updates, and (optionally)
    intervenes by addressing one specific agent ("Critic, you were
    wrong on the last three rounds — consider feature 17 more
    carefully").
    """

    def __init__(self, cfg: AgentConfig, tokenizer: Tokenizer,
                 channel: Optional[CommunicationChannel] = None,
                 modeled_agents: Optional[list[str]] = None) -> None:
        cfg.role = "metacognitor"
        super().__init__(cfg, tokenizer, channel)
        self.temperature = 0.3
        self.modeled_agents = list(modeled_agents or [])
        # One small predictive model per agent observed.
        self.agent_models = nn.ModuleDict({
            name: AgentModel(tokenizer.vocab_size, hidden_dim=64, n_modes=8)
            for name in self.modeled_agents
        })

    def add_modeled_agent(self, agent_name: str) -> None:
        if agent_name in self.agent_models:
            return
        self.agent_models[agent_name] = AgentModel(
            self.tokenizer.vocab_size, hidden_dim=64, n_modes=8,
        )
        self.modeled_agents.append(agent_name)

    def predict_agent_mode(self, agent_name: str,
                              recent_messages: list[Message]) -> tuple[int, float]:
        """Predict which behavioral mode the named agent will be in next.

        ``recent_messages`` are the messages from that agent in the
        recent past (in order). The metacognitor's prediction is a
        mode index in [0, 8) and a confidence in [0, 1].
        """
        if agent_name not in self.agent_models:
            self.add_modeled_agent(agent_name)
        model = self.agent_models[agent_name]
        if not recent_messages:
            # No history — uniform prior.
            return random.randrange(8), 0.5
        joined = "\n".join(m.token_text for m in recent_messages[-8:])
        ids = self.encode(joined)
        with torch.no_grad():
            mode_logits, conf = model(ids)
        return int(mode_logits.argmax().item()), float(conf.item())

    def assess_round(self, channel: CommunicationChannel,
                      verifier_verdicts: list[dict]) -> dict[str, float]:
        """After a round, score each agent by alignment with verifier outcomes.

        Returns a dict mapping ``agent_name -> trust_delta``. The
        orchestrator applies these deltas to the channel's trust map.

        Heuristic (this is intentionally simple — the interesting work
        is the *interface*, not the scoring rule, since the rule is
        easy to swap):

        * If the verifier said "ok", agents that proposed/synthesized
          gain trust.
        * If the verifier said "fail", proposers/synthesizers lose
          trust and critics gain.
        * Always increase trust for the verifier itself (it's the
          ground-truth source for this round).
        """
        deltas: dict[str, float] = {}
        any_ok = any(v.get("verdict") == "ok" for v in verifier_verdicts)
        any_fail = any(v.get("verdict") == "fail" for v in verifier_verdicts)
        for msg in channel.transcript:
            if msg.round_idx != channel._round:
                continue
            role = msg.metadata.get("phase")
            if role == "verify":
                deltas[msg.sender] = deltas.get(msg.sender, 0.0) + 0.05
            elif role in ("propose", "synthesize"):
                d = 0.03 if any_ok else (-0.03 if any_fail else 0.0)
                deltas[msg.sender] = deltas.get(msg.sender, 0.0) + d
            elif role == "critique":
                d = 0.03 if any_fail else (-0.01 if any_ok else 0.0)
                deltas[msg.sender] = deltas.get(msg.sender, 0.0) + d
        return deltas
