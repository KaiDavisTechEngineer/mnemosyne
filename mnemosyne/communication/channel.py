"""Inter-agent communication.

Agents exchange information through two parallel channels:

1. **Token channel** — natural-language-ish messages framed by ``<msg>``
   envelopes. Standard, inspectable, but lossy.

2. **Latent channel** — direct transmission of hidden-state vectors.
   When agent A wants to share a thought with agent B, A's final-layer
   hidden state can be projected through a learned ``send_head`` and
   incorporated into B's context via B's ``recv_head``. This is
   higher-bandwidth than tokens — the agents are literally sharing
   continuous distributed representations — but at the cost of
   interpretability.

Why two channels?
-----------------
The token channel is what humans (and the metacognitive agent) can
inspect. The latent channel is what makes multi-agent reasoning
*efficient*: agent B doesn't have to re-parse a sentence A would
otherwise have had to serialize and then decode. Together they give
us the speed of latent communication with an inspectable audit trail.

This dual-channel design is the protocol-level analog of Coconut's
latent-vs-token reasoning — applied to multi-agent communication
rather than single-agent chain-of-thought.

Protocol design choices:

* Latent vectors travel through a learned linear projection between
  agents (the ``BridgeHead`` below). Different agents can have
  different hidden_dim values; the bridge handles the resizing.
* Every latent transmission is *also* logged as a token-channel
  summary (a compressed natural-language description). This keeps
  inspection possible.
* Receivers don't trust transmissions blindly. Each agent has a
  ``trust`` scalar per-sender that scales incoming messages.
* All messages carry a ``round`` counter so debates can be replayed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class Message:
    """One message in the conversation."""
    sender: str
    recipient: str        # "all" for broadcasts
    round_idx: int
    token_text: str
    latent: Optional[torch.Tensor] = None
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class BridgeHead(nn.Module):
    """Bidirectional projection between two agents' hidden spaces.

    If both agents share the same hidden_dim, the bridge is just a
    learned linear layer (initialized to identity so untrained bridges
    don't destroy information). For mismatched dims, the bridge does
    a learned dimensionality conversion.
    """
    def __init__(self, send_dim: int, recv_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(send_dim, recv_dim, bias=False)
        # Initialize as close to identity as possible.
        if send_dim == recv_dim:
            nn.init.eye_(self.proj.weight)
        else:
            nn.init.normal_(self.proj.weight, std=1.0 / send_dim ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class CommunicationChannel:
    """The shared bus over which agents exchange messages.

    Each agent registers with the channel and obtains a sender id.
    They send messages via ``send()`` and pull pending messages via
    ``inbox()``. The channel maintains the full transcript so any
    observer (the metacognitive agent, a human reviewer) can replay it.
    """

    def __init__(self) -> None:
        self.transcript: list[Message] = []
        self._inbox: dict[str, list[Message]] = {}
        self._registered: set[str] = set()
        self._trust: dict[tuple[str, str], float] = {}  # (recipient, sender) -> trust
        self._round = 0

    def register(self, agent_id: str) -> None:
        self._registered.add(agent_id)
        self._inbox.setdefault(agent_id, [])

    def send(self, sender: str, recipient: str, token_text: str,
              latent: Optional[torch.Tensor] = None,
              metadata: Optional[dict] = None) -> Message:
        if sender not in self._registered:
            raise ValueError(f"agent {sender!r} not registered with the channel")
        msg = Message(
            sender=sender, recipient=recipient, round_idx=self._round,
            token_text=token_text, latent=latent,
            metadata=metadata or {},
        )
        self.transcript.append(msg)
        # Route to inbox(es).
        if recipient == "all":
            for aid in self._registered:
                if aid != sender:
                    self._inbox[aid].append(msg)
        else:
            if recipient not in self._inbox:
                self._inbox[recipient] = []
            self._inbox[recipient].append(msg)
        return msg

    def inbox(self, agent_id: str, clear: bool = True) -> list[Message]:
        msgs = list(self._inbox.get(agent_id, []))
        if clear:
            self._inbox[agent_id] = []
        return msgs

    def advance_round(self) -> int:
        """Bump the round counter — call once per turn of the orchestrator."""
        self._round += 1
        return self._round

    def trust(self, recipient: str, sender: str) -> float:
        return self._trust.get((recipient, sender), 1.0)

    def set_trust(self, recipient: str, sender: str, value: float) -> None:
        self._trust[(recipient, sender)] = max(0.0, min(1.0, value))

    def update_trust(self, recipient: str, sender: str, delta: float) -> None:
        new = self.trust(recipient, sender) + delta
        self.set_trust(recipient, sender, new)

    # ──────────────────────────────────────────────────────────────────
    # Introspection helpers
    # ──────────────────────────────────────────────────────────────────
    def role_distribution(self) -> dict[str, int]:
        """Count messages per sender — useful for diagnosing dominance
        patterns in multi-agent debates."""
        out: dict[str, int] = {}
        for m in self.transcript:
            out[m.sender] = out.get(m.sender, 0) + 1
        return out

    def transcript_text(self, max_rows: int = 50) -> str:
        """Render the transcript in human-readable form."""
        lines = []
        for m in self.transcript[-max_rows:]:
            tag = f"[r{m.round_idx}] {m.sender} → {m.recipient}"
            lines.append(f"{tag}: {m.token_text}")
        return "\n".join(lines)
