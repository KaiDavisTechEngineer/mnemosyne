"""Self-model: the agent's learned model of itself.

A core thesis of MNEMOSYNE is that an agent should *predict its own
outputs* and *measure its surprise* when those predictions miss. The
surprise signal is what we call **calibration error** — it tells the
agent (and the metacognitor) when the agent is operating outside the
regime where its self-model is reliable.

Concretely, the self-model adds a small auxiliary head to each agent
that, given the agent's hidden state at position ``t``, predicts the
distribution over the agent's *own* output at position ``t+1``. During
training we minimize KL divergence between the self-model's prediction
and the actual LM head's softmax. After training, the gap between
prediction and reality is the agent's *online surprise signal*.

Why this matters
----------------
1. **Detection of distribution shift.** If the self-model was trained
   on inputs of family A and now sees family B, the prediction error
   rises sharply. The agent can flag "I'm uncertain about my own
   reasoning here" without needing an external benchmark.

2. **Metacognitor input.** The metacognitor reads each agent's
   self-surprise as a feature of its predict-the-agent task. An agent
   that is currently uncertain about itself is one to discount.

3. **Self-improvement signal.** Episodes with high surprise are the
   most informative to consolidate into semantic memory — they're the
   ones the agent has not yet "internalized." This biases the
   consolidation routine toward growing the agent's known repertoire.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SelfModelConfig:
    hidden_dim: int
    vocab_size: int
    inner_dim: int = 64


class SelfModel(nn.Module):
    """A small auxiliary head that predicts the agent's own next-token
    distribution from its hidden state.

    Architecture: hidden → inner_dim (with SiLU) → vocab logits. Much
    smaller than the main LM head; the self-model is meant to be a
    *summary* of the LM head, not a re-implementation of it. The gap
    between the summary and the full LM head is the surprise signal.
    """

    def __init__(self, cfg: SelfModelConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(cfg.hidden_dim, cfg.inner_dim)
        self.head = nn.Linear(cfg.inner_dim, cfg.vocab_size)
        self.cfg = cfg

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Predict next-token logits from hidden state (B, T, D) → (B, T, V)."""
        return self.head(F.silu(self.proj(hidden)))

    def surprise(self, hidden: torch.Tensor, true_logits: torch.Tensor
                  ) -> torch.Tensor:
        """Per-position symmetric KL between self-model and true LM head.

        Returns a tensor of shape ``(B, T)`` with one surprise scalar
        per token position. Higher = more surprised.
        """
        my_logp = F.log_softmax(self.forward(hidden), dim=-1)
        true_logp = F.log_softmax(true_logits, dim=-1)
        true_p = true_logp.exp()
        my_p = my_logp.exp()
        kl_fwd = (true_p * (true_logp - my_logp)).sum(dim=-1)
        kl_bwd = (my_p * (my_logp - true_logp)).sum(dim=-1)
        return 0.5 * (kl_fwd + kl_bwd)

    def calibration_loss(self, hidden: torch.Tensor, true_logits: torch.Tensor
                          ) -> torch.Tensor:
        """Training loss: KL(true || self-model) averaged over positions.

        This is the loss we'd minimize during a brief 'self-calibration'
        phase — the agent learns to imitate its own outputs from a
        compressed summary."""
        my_logp = F.log_softmax(self.forward(hidden), dim=-1)
        true_p = F.softmax(true_logits, dim=-1).detach()
        return -(true_p * my_logp).sum(dim=-1).mean()


def attach_self_model(agent_module: nn.Module,
                       hidden_dim: int,
                       vocab_size: int) -> SelfModel:
    """Attach a SelfModel as a submodule of ``agent_module``.

    Convenience helper used during agent construction or post-hoc when
    upgrading older agents."""
    self_model = SelfModel(SelfModelConfig(hidden_dim=hidden_dim,
                                            vocab_size=vocab_size))
    agent_module.add_module("self_model", self_model)
    return self_model
