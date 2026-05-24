"""The MNEMOSYNE society orchestrator.

The orchestrator runs the multi-agent reasoning protocol. A single
*round* consists of:

  1. **Propose** — each Proposer emits one (or more) candidate answers.
  2. **Critique** — each Critic reviews each proposal.
  3. **Verify** — each Verifier formally checks each proposal (when a
     verification function is available for the domain).
  4. **Synthesize** — the Synthesizer combines proposals + critiques +
     verifications into a refined answer.
  5. **Meta-assess** — the Metacognitor observes the round and updates
     trust scores in the communication channel.

After the synthesizer's answer, the orchestrator decides whether to:
  * **Halt** — answer is ready (verifier passed, or rounds budget hit).
  * **Iterate** — feed the synthesized answer back as a new proposal,
    run another round.

The orchestrator also drives **memory consolidation**: after every N
rounds it asks each agent to run a consolidation pass, moving stable
episodic patterns into semantic memory.

Design notes
------------
* The protocol is deliberately simple. The interesting research
  questions concern (a) how the agents' learned strategies adapt over
  many problems, (b) which interventions the metacognitor learns to
  make, (c) how trust dynamics evolve. The orchestrator is the *stage*;
  the drama happens in the agents and their interactions.

* The orchestrator is purely Python control flow — no neural nets of
  its own. All learning lives in the agents. This keeps the protocol
  transparent and reproducible.

* The return value of a debate is a structured ``DebateResult`` with
  the full transcript, final answer, every agent's introspection
  report, and the metacognitor's assessment. Downstream code (eval,
  visualization, distillation) consumes this object directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from mnemosyne.agents.base import Agent, IntrospectionReport
from mnemosyne.agents.specialists import (
    Critic, Metacognitor, Proposer, Synthesizer, Verifier,
)
from mnemosyne.communication.channel import CommunicationChannel


@dataclass
class DebateConfig:
    """How the orchestrator runs a debate."""
    max_rounds: int = 3
    proposals_per_round: int = 2
    halt_on_verified_ok: bool = True
    consolidate_every: int = 5    # rounds between memory consolidations
    enable_introspection: bool = True


@dataclass
class RoundOutcome:
    """One round of the debate."""
    round_idx: int
    question: str
    proposals: list[str]
    critiques: list[str]
    verifications: list[dict]
    synthesis: str
    metacog_assessment: dict[str, float]
    introspections: dict[str, IntrospectionReport] = field(default_factory=dict)


@dataclass
class DebateResult:
    """The complete record of a multi-round debate."""
    question: str
    final_answer: str
    rounds: list[RoundOutcome]
    transcript: list           # the full channel transcript
    halted_reason: str
    succeeded: bool

    def __repr__(self) -> str:
        return (f"DebateResult(rounds={len(self.rounds)}, "
                f"succeeded={self.succeeded}, "
                f"final_answer={self.final_answer!r}, "
                f"halted={self.halted_reason!r})")

    def render(self) -> str:
        """Human-readable summary of the debate."""
        lines = [f"╔═══ MNEMOSYNE debate ═══════════════════════════════════════════"]
        lines.append(f"║ Question: {self.question}")
        lines.append(f"║ Rounds: {len(self.rounds)}, halted: {self.halted_reason}")
        lines.append(f"║ Final answer: {self.final_answer}")
        lines.append(f"║ Succeeded: {self.succeeded}")
        for r in self.rounds:
            lines.append(f"╟── Round {r.round_idx} " + "─" * 47)
            for i, p in enumerate(r.proposals):
                lines.append(f"║   proposer:    [{i}] {p[:90]}")
            for i, c in enumerate(r.critiques):
                lines.append(f"║   critic:      [{i}] {c[:90]}")
            for i, v in enumerate(r.verifications):
                lines.append(f"║   verifier:    [{i}] {v.get('verdict','?')} "
                              f"{v.get('reason','')[:60]}")
            lines.append(f"║   synthesizer: {r.synthesis[:90]}")
            if r.metacog_assessment:
                deltas = ", ".join(f"{k}={v:+.2f}" for k, v in r.metacog_assessment.items())
                lines.append(f"║   metacog Δ:   {deltas}")
        lines.append(f"╚══════════════════════════════════════════════════════════════")
        return "\n".join(lines)


class Society:
    """A society of MNEMOSYNE agents running a multi-round reasoning debate."""

    def __init__(self,
                  proposer: Proposer,
                  critic: Critic,
                  verifier: Verifier,
                  synthesizer: Synthesizer,
                  metacognitor: Metacognitor,
                  channel: CommunicationChannel,
                  cfg: Optional[DebateConfig] = None) -> None:
        self.proposer = proposer
        self.critic = critic
        self.verifier = verifier
        self.synthesizer = synthesizer
        self.metacognitor = metacognitor
        self.channel = channel
        self.cfg = cfg or DebateConfig()
        self.total_rounds_seen = 0

        # Ensure metacognitor knows about all the agents.
        for name in (proposer.cfg.name, critic.cfg.name,
                     verifier.cfg.name, synthesizer.cfg.name):
            metacognitor.add_modeled_agent(name)

    def debate(self,
                question: str,
                verification_fn: Optional[Callable] = None,
                ) -> DebateResult:
        """Run a multi-round debate on ``question``. Returns the full result."""
        rounds: list[RoundOutcome] = []
        final_answer = ""
        halted_reason = "max_rounds"
        succeeded = False
        current_question = question

        for round_idx in range(self.cfg.max_rounds):
            self.channel.advance_round()

            # ── 1. Propose ────────────────────────────────────────
            proposals = self.proposer.propose(
                current_question, k=self.cfg.proposals_per_round,
            )

            # ── 2. Critique ───────────────────────────────────────
            critiques: list[str] = []
            for p in proposals:
                critiques.append(self.critic.critique(current_question, p))

            # ── 3. Verify ─────────────────────────────────────────
            verifications: list[dict] = []
            for p in proposals:
                v = self.verifier.verify(current_question, p,
                                           verification_fn=verification_fn)
                verifications.append(v)

            # ── 4. Synthesize ─────────────────────────────────────
            synthesis = self.synthesizer.synthesize(
                current_question, proposals, critiques, verifications,
            )

            # ── 5. Meta-assess ────────────────────────────────────
            deltas = self.metacognitor.assess_round(self.channel, verifications)
            for agent_name, delta in deltas.items():
                # Update trust from the synthesizer's perspective.
                self.channel.update_trust(
                    self.synthesizer.cfg.name, agent_name, delta,
                )

            # ── 6. Introspect (optional) ──────────────────────────
            introspections: dict[str, IntrospectionReport] = {}
            if self.cfg.enable_introspection:
                # Each role-active agent introspects on its own output.
                if proposals:
                    ids = self.proposer.encode(proposals[0])
                    introspections[self.proposer.cfg.name] = (
                        self.proposer.introspect(ids, compute_counterfactual=False, top_n=4)
                    )
                if critiques:
                    ids = self.critic.encode(critiques[0])
                    introspections[self.critic.cfg.name] = (
                        self.critic.introspect(ids, compute_counterfactual=False, top_n=4)
                    )
                if synthesis:
                    ids = self.synthesizer.encode(synthesis)
                    introspections[self.synthesizer.cfg.name] = (
                        self.synthesizer.introspect(ids, compute_counterfactual=False, top_n=4)
                    )

            rounds.append(RoundOutcome(
                round_idx=round_idx,
                question=current_question,
                proposals=proposals,
                critiques=critiques,
                verifications=verifications,
                synthesis=synthesis,
                metacog_assessment=deltas,
                introspections=introspections,
            ))
            final_answer = synthesis
            self.total_rounds_seen += 1

            # ── 7. Halt? ──────────────────────────────────────────
            any_ok = any(v.get("verdict") == "ok" for v in verifications)
            if self.cfg.halt_on_verified_ok and any_ok:
                halted_reason = "verified_ok"
                succeeded = True
                break
            # Otherwise: iterate.
            current_question = (
                f"{question}\nprevious synthesis: {synthesis}\n"
                f"refine the answer."
            )

        # Consolidate memory every N rounds.
        if (self.total_rounds_seen > 0
                and self.total_rounds_seen % self.cfg.consolidate_every == 0):
            for agent in (self.proposer, self.critic, self.verifier,
                           self.synthesizer, self.metacognitor):
                agent.memory.consolidate(n_clusters=4, n_iters=10,
                                          min_cluster_size=2)

        return DebateResult(
            question=question,
            final_answer=final_answer,
            rounds=rounds,
            transcript=list(self.channel.transcript),
            halted_reason=halted_reason,
            succeeded=succeeded,
        )

    # ──────────────────────────────────────────────────────────────────
    # Recording outcomes back into episodic memory
    # ──────────────────────────────────────────────────────────────────
    def remember_debate(self, result: DebateResult) -> None:
        """Store the debate as an episode in each agent's memory."""
        outcome = {"succeeded": result.succeeded,
                    "n_rounds": len(result.rounds),
                    "halted_reason": result.halted_reason}
        for agent in (self.proposer, self.critic, self.verifier,
                       self.synthesizer, self.metacognitor):
            ids = agent.encode(result.question)
            _, captured = agent.forward(ids)
            agent.remember(
                input_text=result.question,
                output_text=result.final_answer,
                outcome=outcome,
                captured=captured,
            )
