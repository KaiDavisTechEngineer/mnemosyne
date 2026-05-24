"""TopK Sparse Autoencoder (TopK-SAE).

This implements the technique from OpenAI's "Scaling and Evaluating
Sparse Autoencoders" (Gao et al. 2024) and the parallel Anthropic line
of work on dictionary learning (Bricken et al. 2023, "Towards
Monosemanticity").

The objective
-------------
An SAE learns a *dictionary* of feature directions in activation
space. Given an activation vector ``a ∈ R^d``, the encoder produces a
sparse code ``z ∈ R^k`` (k > d) and the decoder reconstructs:

    encode:  z = TopK(W_enc · a + b_enc)
    decode:  â = W_dec · z + b_dec
    loss:    ‖a − â‖² + (auxiliary terms)

Where ``TopK`` keeps only the K largest entries and zeros the rest.
The hyperparameter K controls sparsity *exactly* — there is no
softness threshold to tune, no L1 penalty to balance, no Lagrange
multiplier. Each feature index in the dictionary corresponds to a
learned direction; when active, that direction is participating in
representing the input.

Why this matters for MNEMOSYNE
------------------------------
The SAE turns the agent's hidden states into an interpretable
*alphabet of concepts*. The agent can refer to feature 42 by name
(via the ``<feature:42>`` token from the tokenizer), reason about
which features fire on which inputs, and — crucially — *intervene
on its own features*: clamp feature K to zero and observe what its
output would have been.

This is what we mean by "the agent has a causal model of itself."

Implementation notes
--------------------
* We use the **TopK** activation rather than ReLU + L1, following Gao
  et al. (2024). Exact sparsity, no penalty balance.
* The decoder weight is **unit-normalized per column** after each
  optimizer step. This prevents the SAE from cheating by scaling up
  features arbitrarily; it forces each feature direction to be a
  proper basis vector.
* We track **dead features** — features that haven't activated in
  N batches — and periodically reset them with the largest
  reconstruction error in the current batch (Gao et al.'s
  "auxiliary reconstruction loss" trick).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEConfig:
    """Hyperparameters."""

    d_model: int = 64  # input/output dimensionality (= activation dim)
    n_features: int = 256  # dictionary size; should be 4-16× d_model
    k: int = 8  # number of active features per token (sparsity)
    dead_threshold: int = 500  # batches without activation → "dead"
    aux_k: int = 16  # K for the auxiliary loss path


class TopKSAE(nn.Module):
    """A TopK Sparse Autoencoder, instrumented for MNEMOSYNE's introspection."""

    def __init__(self, cfg: SAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.W_enc = nn.Parameter(
            torch.randn(cfg.d_model, cfg.n_features) * (1.0 / cfg.d_model**0.5)
        )
        self.b_enc = nn.Parameter(torch.zeros(cfg.n_features))
        # Initialize decoder as the transpose of encoder, a common SAE practice
        # that gives a warm start and avoids early instability.
        self.W_dec = nn.Parameter(self.W_enc.detach().T.clone())
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_model))
        # Track activation counts for dead-feature detection.
        self.register_buffer(
            "activation_counts", torch.zeros(cfg.n_features, dtype=torch.long)
        )
        self.register_buffer("step_count", torch.zeros((), dtype=torch.long))

    # ──────────────────────────────────────────────────────────────────
    # Encoding / decoding
    # ──────────────────────────────────────────────────────────────────
    def encode_pre(self, a: torch.Tensor) -> torch.Tensor:
        """Pre-activation: W_enc · (a - b_dec) + b_enc.

        Subtracting ``b_dec`` (the decoder bias = expected mean activation)
        is the "geometric median" centering trick from Bricken et al.;
        it gives the encoder a stable reference frame.
        """
        return (a - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, a: torch.Tensor) -> torch.Tensor:
        """Encode activation ``a`` of shape (..., d_model) to a sparse code
        of shape (..., n_features) keeping only the top-K entries per
        sample positive.
        """
        pre = self.encode_pre(a)
        return _topk_relu(pre, k=self.cfg.k)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct activation from sparse code."""
        return z @ self.W_dec + self.b_dec

    def forward(self, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full encode → decode pass. Returns (reconstruction, sparse_code)."""
        z = self.encode(a)
        recon = self.decode(z)
        return recon, z

    # ──────────────────────────────────────────────────────────────────
    # Training: loss + dead-feature resurrection
    # ──────────────────────────────────────────────────────────────────
    def loss(self, a: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Compute training loss and diagnostics for a batch of activations.

        Returns ``(scalar loss, info dict)`` where info contains the
        component losses, average L0, fraction of dead features, etc.
        """
        # Main reconstruction.
        pre = self.encode_pre(a)
        z = _topk_relu(pre, k=self.cfg.k)
        recon = self.decode(z)
        recon_loss = F.mse_loss(recon, a)

        # Update activation counts (without gradients).
        with torch.no_grad():
            active_mask = (z != 0).any(dim=tuple(range(z.dim() - 1)))
            # active_mask is shape (n_features,)
            self.activation_counts += active_mask.long()
            self.step_count += 1

        # Auxiliary reconstruction loss using dead features only. This is
        # the Gao et al. (2024) "auxk" trick — gives gradient to features
        # that haven't been activating, so they can recover.
        dead = self._dead_features()
        if dead.any():
            residual = (a - recon).detach()
            # Score features by their pre-activations, masked to dead-only.
            scores = pre.clone()
            scores[..., ~dead] = float("-inf")
            z_aux = _topk_relu(scores, k=min(self.cfg.aux_k, int(dead.sum())))
            recon_aux = z_aux @ self.W_dec
            aux_loss = F.mse_loss(recon_aux, residual)
            aux_coef = 1.0 / 32.0  # Gao et al.'s magnitude
            total = recon_loss + aux_coef * aux_loss
        else:
            aux_loss = torch.zeros((), device=a.device)
            total = recon_loss

        info = {
            "recon_loss": float(recon_loss.item()),
            "aux_loss": float(aux_loss.item()),
            "L0": float((z != 0).float().sum(dim=-1).mean().item()),
            "dead_frac": float(dead.float().mean().item()),
            "active_frac": float((self.activation_counts > 0).float().mean().item()),
        }
        return total, info

    def _dead_features(self) -> torch.Tensor:
        """Boolean mask of features that have not activated recently.

        A feature is "dead" if it has not activated in the last
        ``dead_threshold`` batches.
        """
        if self.step_count < self.cfg.dead_threshold:
            return torch.zeros(
                self.cfg.n_features,
                dtype=torch.bool,
                device=self.activation_counts.device,
            )
        return self.activation_counts == 0

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Project decoder columns onto the unit sphere.

        Called after each optimizer step. This is what keeps the SAE
        from cheating: each feature direction must be a unit vector,
        so the sparse code is the only place where magnitude can live.
        """
        norms = self.W_dec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.W_dec.data = self.W_dec.data / norms

    @torch.no_grad()
    def reset_dead_features(
        self, activations: torch.Tensor, count_per_step: int = 4
    ) -> int:
        """Resurrect dead features by initializing them to high-error inputs.

        Gao et al. 2024 §3.4 — for each dead feature, set its encoder
        direction to a normalized example activation with high
        reconstruction error. Returns the number of features reset.
        """
        dead = self._dead_features()
        if not dead.any():
            return 0
        # Pick the worst-reconstructed activations in this batch.
        a_flat = activations.view(-1, activations.shape[-1])
        with torch.no_grad():
            recon, _ = self.forward(a_flat)
            err = (a_flat - recon).norm(dim=-1)
            worst = err.topk(min(count_per_step, a_flat.shape[0])).indices
        # Re-init dead features to worst examples (normalized).
        dead_idx = dead.nonzero(as_tuple=False).squeeze(-1)
        n_reset = min(len(dead_idx), len(worst))
        for d_i, w_i in zip(dead_idx[:n_reset].tolist(), worst[:n_reset].tolist()):
            direction = a_flat[w_i] - self.b_dec
            direction = direction / (direction.norm() + 1e-8)
            self.W_enc.data[:, d_i] = direction * 0.1
            self.W_dec.data[d_i, :] = direction
            self.b_enc.data[d_i] = 0.0
            self.activation_counts[d_i] = 0
        return n_reset

    # ──────────────────────────────────────────────────────────────────
    # Inspection helpers used by the introspection layer
    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def top_features(self, a: torch.Tensor, n: int = 5) -> list[tuple[int, float]]:
        """Return the n highest-magnitude active features in ``a``."""
        z = self.encode(a)
        z_flat = z.view(-1, z.shape[-1]).mean(dim=0)  # average over batch & seq
        vals, idx = z_flat.topk(min(n, self.cfg.n_features))
        return list(zip(idx.tolist(), vals.tolist()))

    @torch.no_grad()
    def ablate_feature(self, a: torch.Tensor, feature_idx: int) -> torch.Tensor:
        """Return the reconstruction of ``a`` with one feature clamped off.

        This is the building block of counterfactual self-inquiry: what
        would the reconstructed activation look like if feature K had
        not fired? The agent uses this to ask "what would I have said
        without this feature?"
        """
        z = self.encode(a)
        z = z.clone()
        z[..., feature_idx] = 0.0
        return self.decode(z)


# ─────────────────────────────────────────────────────────────────────
# TopK activation primitive (used as both ReLU and sparsity step)
# ─────────────────────────────────────────────────────────────────────
def _topk_relu(x: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k largest positive entries of x along the last dim, zero rest.

    Sub-zero entries are zeroed first (acts like a positivity gate).
    Differentiable through the surviving entries (the gating is a
    straight-through estimator: the gradient flows through the kept
    entries as if no operation happened).
    """
    if k >= x.shape[-1]:
        return F.relu(x)
    pos = F.relu(x)
    values, indices = pos.topk(k, dim=-1)
    out = torch.zeros_like(x)
    out.scatter_(-1, indices, values)
    return out
