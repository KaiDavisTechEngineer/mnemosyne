"""Causal interventions on transformer activations.

This module implements the operations that turn a hook-instrumented
transformer + sparse autoencoder into a causally-introspectable system.
Three core capabilities:

1. **Activation patching** (Heimersheim & Nanda, "Causal scrubbing";
   Zhang & Nanda 2024). Run two forward passes, one "clean" and one
   "corrupted"; copy activations from clean → corrupted at a specific
   site; measure how much the output recovers. The recovery is a
   causal measure of how much that site contributes to the difference.

2. **Attribution patching** (Syed et al. 2023 — sometimes called
   *AtP*; later refined as ``AtP_star`` by Kramár et al. 2024). A
   first-order Taylor expansion of activation patching that needs
   only one extra backward pass instead of one forward pass per site.
   Trades exactness for tractability — useful when there are
   thousands of features to score.

3. **Counterfactual reasoning**. Given an input where the agent's
   output was Y, find the smallest set of features whose ablation
   flips the output to something else. This is the operational
   definition of "what mattered for my answer."

Together these let the agent answer the canonical introspection
question: *"Which features of mine caused me to say what I said?"*
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from mnemosyne.arch.transformer import (
    HookContext,
    HookedTransformer,
    hooks,
)
from mnemosyne.interp.sae import TopKSAE


# ─────────────────────────────────────────────────────────────────────
# Activation patching
# ─────────────────────────────────────────────────────────────────────
def activation_patch(
    model: HookedTransformer,
    clean_ids: torch.Tensor,
    corrupted_ids: torch.Tensor,
    patch_site: str,
    metric: str = "logit_diff",
    target_token: Optional[int] = None,
) -> float:
    """How much does patching ``patch_site`` from clean → corrupted recover
    the clean output?

    Concretely:
      1. Run a clean forward pass; capture the activation at ``patch_site``.
      2. Run a corrupted forward pass with that activation patched in.
      3. Compare the patched output to the clean and corrupted baselines.

    Returns a recovery score in [0, 1]: 0 means the patch had no effect,
    1 means the patched output matches the clean output exactly.
    """
    # 1. Clean forward — capture target activation.
    clean_logits, captured = model.run_with_capture(clean_ids, sites=[patch_site])
    clean_act = captured[patch_site]

    # 2. Corrupted forward — patch in the clean activation.
    if clean_ids.shape != corrupted_ids.shape:
        raise ValueError("clean and corrupted inputs must have the same shape")
    patched_logits, _ = model.run_with_intervention(
        corrupted_ids, {patch_site: clean_act}
    )

    # 3. Baseline: corrupted alone.
    _, corrupted_logits = model(corrupted_ids)

    # Compute the metric — default is logit-difference of the target token
    # at the last position.
    if target_token is None:
        target_token = int(clean_logits[0, -1].argmax().item())

    def m(logits: torch.Tensor) -> float:
        return float(logits[0, -1, target_token].item())

    clean_m = m(clean_logits)
    corrupted_m = m(corrupted_logits)
    patched_m = m(patched_logits)

    denom = clean_m - corrupted_m
    if abs(denom) < 1e-8:
        return 0.0
    return float((patched_m - corrupted_m) / denom)


# ─────────────────────────────────────────────────────────────────────
# Attribution patching (first-order approximation)
# ─────────────────────────────────────────────────────────────────────
def attribution_patch(
    model: HookedTransformer,
    clean_ids: torch.Tensor,
    corrupted_ids: torch.Tensor,
    sites: list[str],
    target_token: int,
) -> dict[str, float]:
    """Attribution-patching scores for many sites in one pass.

    Algorithm (Syed et al. 2023): the recovery from patching site s is
    approximately

        recovery(s) ≈ (a_clean(s) - a_corr(s)) · ∇_{a(s)} L | a_corr

    where L is the target metric (logit of ``target_token`` at the last
    position) and the gradient is taken at the corrupted activations.
    One backward pass through the corrupted forward gives us all the
    site gradients at once.
    """
    # Capture clean activations at every site.
    _, clean_acts = model.run_with_capture(clean_ids, sites=sites)
    # Forward + backward through corrupted, capturing both activations and
    # their gradients.
    grads: dict[str, torch.Tensor] = {}
    corr_acts: dict[str, torch.Tensor] = {}

    def grad_hook_factory(name: str):
        def hook(act: torch.Tensor, site: str) -> torch.Tensor:
            # We want gradients with respect to the activation as it leaves
            # this site. Detach + recompose so backward picks it up.
            act = act.detach().requires_grad_(True)
            corr_acts[name] = act
            return act

        return hook

    ctx = HookContext(replace={s: grad_hook_factory(s) for s in sites}, capture=[])
    model.zero_grad(set_to_none=True)
    with hooks(ctx):
        _, logits = model(corrupted_ids)
    target = logits[0, -1, target_token]
    target.backward(retain_graph=False)

    for s in sites:
        if s in corr_acts and corr_acts[s].grad is not None:
            grads[s] = corr_acts[s].grad.detach()
    # Compute first-order attribution.
    out: dict[str, float] = {}
    for s in sites:
        if s not in clean_acts or s not in grads or s not in corr_acts:
            out[s] = 0.0
            continue
        delta = clean_acts[s] - corr_acts[s].detach()
        attr = (delta * grads[s]).sum().item()
        out[s] = float(attr)
    return out


# ─────────────────────────────────────────────────────────────────────
# Feature-level attribution using SAEs
# ─────────────────────────────────────────────────────────────────────
@dataclass
class FeatureAttribution:
    """Per-feature contribution to an output."""

    site: str
    feature_idx: int
    delta_logit: float
    activation: float

    def __repr__(self) -> str:
        return (
            f"FeatureAttribution(site={self.site!r}, "
            f"feature={self.feature_idx}, "
            f"Δlogit={self.delta_logit:+.4f}, "
            f"act={self.activation:.4f})"
        )


def feature_attribution(
    model: HookedTransformer,
    sae: TopKSAE,
    site: str,
    input_ids: torch.Tensor,
    target_token: int,
    top_n: int = 8,
) -> list[FeatureAttribution]:
    """Ablate each active SAE feature one at a time and record logit changes.

    Algorithm:
      1. Run the model, capture the activation at ``site``.
      2. Encode through the SAE to get the sparse code.
      3. For each active feature, replace the activation with the
         SAE's reconstruction *minus* that feature's contribution, run
         the model again, record the change in the target-token logit.

    Returns the top-N features by absolute Δlogit, sorted descending.
    """
    # Baseline forward — capture site, get baseline logit.
    base_logits, captured = model.run_with_capture(input_ids, sites=[site])
    base_act = captured[site]
    baseline_logit = float(base_logits[0, -1, target_token].item())

    # Encode through SAE.
    z = sae.encode(base_act)  # (B, T, n_features)
    # Take the last-token feature activations (introspection focuses on the
    # token that mattered for the decision).
    z_last = z[0, -1]  # (n_features,)
    active = (z_last > 0).nonzero(as_tuple=False).squeeze(-1).tolist()

    attributions: list[FeatureAttribution] = []
    for f in active:
        # Ablate feature f at the last token position.
        z_abl = z.clone()
        z_abl[0, -1, f] = 0.0
        a_abl = sae.decode(z_abl)
        logits_abl, _ = model.run_with_intervention(input_ids, {site: a_abl})
        delta = baseline_logit - float(logits_abl[0, -1, target_token].item())
        attributions.append(
            FeatureAttribution(
                site=site,
                feature_idx=f,
                delta_logit=delta,
                activation=float(z_last[f].item()),
            )
        )
    # Sort by absolute Δlogit, descending.
    attributions.sort(key=lambda a: abs(a.delta_logit), reverse=True)
    return attributions[:top_n]


# ─────────────────────────────────────────────────────────────────────
# Counterfactual generation
# ─────────────────────────────────────────────────────────────────────
@dataclass
class Counterfactual:
    """A minimal feature ablation that changes the agent's answer."""

    site: str
    ablated_features: list[int]
    original_token: int
    counterfactual_token: int
    confidence_drop: float

    def __repr__(self) -> str:
        return (
            f"Counterfactual(site={self.site!r}, "
            f"ablated={self.ablated_features}, "
            f"{self.original_token} → {self.counterfactual_token}, "
            f"confidence_drop={self.confidence_drop:+.4f})"
        )


def find_counterfactual(
    model: HookedTransformer,
    sae: TopKSAE,
    site: str,
    input_ids: torch.Tensor,
    max_features: int = 4,
) -> Optional[Counterfactual]:
    """Search for the smallest set of feature ablations that changes the
    model's argmax at the last token.

    Greedy algorithm: rank features by attribution magnitude, then
    accumulate ablations until either the argmax flips or we exceed
    ``max_features``. The result is a feature-level *counterfactual
    explanation*: "if these K features had not fired, you would have
    said X instead of Y."

    Returns None if no flip was found within the budget.
    """
    base_logits, captured = model.run_with_capture(input_ids, sites=[site])
    base_act = captured[site]
    orig_probs = F.softmax(base_logits[0, -1], dim=-1)
    orig_token = int(orig_probs.argmax().item())
    orig_conf = float(orig_probs[orig_token].item())

    # Rank features by attribution magnitude.
    attrs = feature_attribution(
        model, sae, site, input_ids, orig_token, top_n=max(16, max_features * 4)
    )

    z_base = sae.encode(base_act).clone()
    ablated: list[int] = []
    for attr in attrs:
        # Only ablate features that had positive Δlogit (i.e., features
        # that were *supporting* the current answer).
        if attr.delta_logit <= 0:
            continue
        z_test = z_base.clone()
        for f in ablated + [attr.feature_idx]:
            z_test[0, -1, f] = 0.0
        a_test = sae.decode(z_test)
        logits_test, _ = model.run_with_intervention(input_ids, {site: a_test})
        new_probs = F.softmax(logits_test[0, -1], dim=-1)
        new_token = int(new_probs.argmax().item())
        if new_token != orig_token:
            ablated.append(attr.feature_idx)
            return Counterfactual(
                site=site,
                ablated_features=ablated,
                original_token=orig_token,
                counterfactual_token=new_token,
                confidence_drop=orig_conf - float(new_probs[orig_token].item()),
            )
        ablated.append(attr.feature_idx)
        if len(ablated) >= max_features:
            return None
    return None


# ─────────────────────────────────────────────────────────────────────
# Convenience wrappers used by the introspection layer
# ─────────────────────────────────────────────────────────────────────
def ablate_feature(
    model: HookedTransformer,
    sae: TopKSAE,
    site: str,
    input_ids: torch.Tensor,
    feature_idx: int,
) -> torch.Tensor:
    """Run a forward pass with feature ``feature_idx`` clamped to zero at
    the site. Returns the resulting logits."""
    _, captured = model.run_with_capture(input_ids, sites=[site])
    a = captured[site]
    z = sae.encode(a)
    z[..., feature_idx] = 0.0
    a_abl = sae.decode(z)
    logits, _ = model.run_with_intervention(input_ids, {site: a_abl})
    return logits


def causal_distance(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    """Symmetric KL distance between two output distributions. A coarse
    summary of how different two outputs are."""
    p = F.softmax(logits_a[0, -1], dim=-1)
    q = F.softmax(logits_b[0, -1], dim=-1)
    return float(
        0.5
        * (
            F.kl_div(q.log(), p, reduction="sum")
            + F.kl_div(p.log(), q, reduction="sum")
        ).item()
    )
