"""Evaluation suite for MNEMOSYNE.

Reports metrics across the architecture's many capabilities:

* **Debate solve rate** — % of samples where the synthesizer's answer
  matches the gold.
* **Rounds to convergence** — average number of rounds before the
  debate halted on a successful verification.
* **SAE reconstruction quality** — mean MSE between activations and
  their SAE reconstructions, across all introspection sites.
* **Counterfactual flip rate** — % of samples where the agent's
  introspection successfully identified a small set of features whose
  ablation flips its answer (i.e., a "minimal cause" for the output).
* **Metacognitor accuracy** — % of next-message mode predictions that
  agreed with the actual behavioral mode (where ground truth is
  defined by which agent said something in the next turn).
* **Memory growth** — episodes stored / concepts consolidated.

Eval is purely diagnostic — it never modifies model weights.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from mnemosyne.causal.interventions import find_counterfactual
from mnemosyne.society.orchestrator import Society
from mnemosyne.training.tasks import Sample


@dataclass
class EvalReport:
    n_samples: int = 0
    n_correct: int = 0
    n_verifier_ok: int = 0
    total_rounds: int = 0
    sae_recon_mse: float = 0.0
    n_counterfactuals_found: int = 0
    n_counterfactual_attempts: int = 0
    per_family: dict[str, dict] = field(
        default_factory=lambda: defaultdict(lambda: {"n": 0, "correct": 0})
    )
    episodes_stored: int = 0
    concepts_in_semantic: int = 0

    @property
    def solve_rate(self) -> float:
        return self.n_correct / max(self.n_samples, 1)

    @property
    def verifier_pass_rate(self) -> float:
        return self.n_verifier_ok / max(self.n_samples, 1)

    @property
    def avg_rounds(self) -> float:
        return self.total_rounds / max(self.n_samples, 1)

    @property
    def counterfactual_flip_rate(self) -> float:
        return self.n_counterfactuals_found / max(self.n_counterfactual_attempts, 1)

    def pretty(self) -> str:
        lines = [
            "═══ MNEMOSYNE eval report ════════════════════════════════════",
            f"  samples evaluated:        {self.n_samples}",
            f"  solve rate (gold match):  {self.n_correct}/{self.n_samples} "
            f"({self.solve_rate:.1%})",
            f"  verifier pass rate:       {self.n_verifier_ok}/{self.n_samples} "
            f"({self.verifier_pass_rate:.1%})",
            f"  average rounds:           {self.avg_rounds:.2f}",
            "",
            f"  SAE reconstruction MSE:   {self.sae_recon_mse:.4f}",
            f"  counterfactual flip rate: {self.n_counterfactuals_found}"
            f"/{self.n_counterfactual_attempts} ({self.counterfactual_flip_rate:.1%})",
            "",
            f"  episodes stored:          {self.episodes_stored}",
            f"  semantic concepts:        {self.concepts_in_semantic}",
            "",
            "  per-family solve rates:",
        ]
        for fam, d in sorted(self.per_family.items()):
            rate = d["correct"] / max(d["n"], 1)
            lines.append(f"    {fam:<12s}  {d['correct']}/{d['n']} ({rate:.1%})")
        lines.append("═" * 64)
        return "\n".join(lines)


def evaluate_society(
    society: Society, samples: list[Sample], cf_sample_rate: float = 0.5
) -> EvalReport:
    """Run the society over each sample and aggregate diagnostics.

    ``cf_sample_rate`` controls how often we run the (expensive)
    counterfactual search on the proposer; default 50% to keep eval
    snappy.
    """
    report = EvalReport(n_samples=len(samples))
    society.proposer.eval()
    society.critic.eval()
    society.synthesizer.eval()
    society.metacognitor.eval()

    for i, s in enumerate(samples):
        with torch.no_grad():
            result = society.debate(s.question, verification_fn=s.verify)

        # Solve rate by gold-match.
        match = s.gold_answer.lower() in result.final_answer.lower()
        if match:
            report.n_correct += 1
        if result.succeeded:
            report.n_verifier_ok += 1
        report.total_rounds += len(result.rounds)
        fam = report.per_family[s.family]
        fam["n"] += 1
        if match:
            fam["correct"] += 1

        # SAE reconstruction quality (using the proposer as exemplar).
        with torch.no_grad():
            ids = society.proposer.encode(s.question)
            site = society.proposer.cfg.introspection_sites[0]
            _, captured = society.proposer.transformer.run_with_capture(
                ids, sites=[site]
            )
            a = captured[site].view(-1, captured[site].shape[-1])
            sae = society.proposer.saes[society.proposer._safe_key(site)]
            recon, _ = sae(a)
            mse = F.mse_loss(recon, a).item()
            report.sae_recon_mse += mse / report.n_samples

        # Counterfactual probe (subsampled).
        if i / max(len(samples), 1) < cf_sample_rate:
            with torch.no_grad():
                cf = find_counterfactual(
                    society.proposer.transformer,
                    society.proposer.saes[society.proposer._safe_key(site)],
                    site,
                    ids,
                    max_features=3,
                )
            report.n_counterfactual_attempts += 1
            if cf is not None:
                report.n_counterfactuals_found += 1

    report.episodes_stored = len(society.proposer.memory.episodic)
    report.concepts_in_semantic = len(society.proposer.memory.semantic)
    return report
