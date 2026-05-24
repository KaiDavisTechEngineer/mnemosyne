"""MNEMOSYNE's hook-instrumented transformer.

The transformer in this file is designed differently from the usual
PyTorch transformer. Every sublayer exposes a hook point — a named
location at which activations can be:

  * captured (read out for later analysis or feature extraction)
  * replaced (clamped to a fixed tensor, ablated, or patched from another
    forward pass)
  * routed through a sparse autoencoder for interpretability

This is the foundation MNEMOSYNE's self-modeling layer is built on.
Without first-class hooks, causal interventions and circuit tracing
become brittle monkey-patches; with them, the agent can introspect on
its own reasoning in a principled way.

The modern primitives are unchanged from a standard Llama-style stack
(RMSNorm + RoPE + SwiGLU + GQA + KV cache + tied embeddings), but the
forward pass is built around a ``HookContext`` that records every
activation site by name.

Naming convention for hook sites
--------------------------------
At layer ``i`` of an ``N``-layer stack, the following hooks fire in
order during a forward pass::

    embed              once before the stack
    block_{i}.resid_pre    residual stream entering the block
    block_{i}.attn_norm    output of pre-attention RMSNorm
    block_{i}.attn_q       Q projection
    block_{i}.attn_k       K projection
    block_{i}.attn_v       V projection
    block_{i}.attn_out     post-attention projection
    block_{i}.resid_mid    residual stream after attention residual
    block_{i}.mlp_norm     output of pre-MLP RMSNorm
    block_{i}.mlp_pre      SwiGLU input (gate * up)
    block_{i}.mlp_out      down-projection output
    block_{i}.resid_post   residual stream after MLP residual
    final_norm             output of the final RMSNorm
    logits                 next-token logits

Why this matters
----------------
Anthropic's mechanistic interpretability work (Olsson et al., Conmy
et al., Marks & Tegmark) builds attribution graphs over exactly these
locations. By making them first-class in the model definition, every
subsequent analysis — sparse autoencoders, activation patching,
attribution patching, causal scrubbing — composes cleanly without
modifying model internals.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────
# Hook infrastructure
# ─────────────────────────────────────────────────────────────────────
HookFn = Callable[[torch.Tensor, str], torch.Tensor]
"""A hook receives (activation, site_name) and returns a possibly-modified
activation. Returning the input unchanged is the no-op."""


@dataclass
class HookContext:
    """Per-forward-pass hook state.

    Two collections:
      * ``capture`` is a list of site-name patterns whose activations
        should be stored on the way through. Matching uses simple prefix
        rules so ``"block_3."`` captures everything in block 3.
      * ``replace`` is a dict from site name to either a tensor (used
        directly as the new activation) or a callable that transforms
        the activation. Replacements override captures.

    After the forward pass, ``captured`` holds the recorded activations.
    """

    capture: list[str] = field(default_factory=list)
    replace: dict[str, torch.Tensor | HookFn] = field(default_factory=dict)
    captured: dict[str, torch.Tensor] = field(default_factory=dict)

    def should_capture(self, site: str) -> bool:
        return any(site == p or site.startswith(p) for p in self.capture)

    def fire(self, activation: torch.Tensor, site: str) -> torch.Tensor:
        """Run any hooks registered for this site. Returns the (possibly
        replaced) activation that should continue through the network."""
        if self.should_capture(site):
            self.captured[site] = activation.detach().clone()
        if site in self.replace:
            r = self.replace[site]
            if callable(r):
                return r(activation, site)
            return r
        return activation

    def clear(self) -> None:
        self.captured.clear()


# A module-level current-hook-context. We use contextvar-style scoping
# so multiple forward passes don't trample each other when running
# concurrently (e.g. when computing counterfactuals).
_HOOK_STACK: list[HookContext] = [HookContext()]


@contextmanager
def hooks(ctx: HookContext):
    """Activate ``ctx`` as the hook state for forward passes inside this
    context. Use as a context manager::

        ctx = HookContext(capture=["block_3.resid_post"])
        with hooks(ctx):
            logits = model(input_ids)
        print(ctx.captured["block_3.resid_post"].shape)
    """
    _HOOK_STACK.append(ctx)
    try:
        yield ctx
    finally:
        _HOOK_STACK.pop()


def _hook(x: torch.Tensor, site: str) -> torch.Tensor:
    return _HOOK_STACK[-1].fire(x, site)


# ─────────────────────────────────────────────────────────────────────
# Modern primitives
# ─────────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (Zhang & Sennrich 2019).

    Used by Llama, Mistral, every modern open model since 2023. Simpler
    than LayerNorm (no centering, no bias) and empirically just as
    effective. The single learnable scale parameter is initialized to
    one.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


def precompute_rope(
    head_dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute (cos, sin) RoPE tables of shape (max_seq_len, head_dim/2)."""
    assert head_dim % 2 == 0
    half = head_dim // 2
    freqs = 1.0 / (
        base ** (torch.arange(0, half, dtype=torch.float32, device=device) / half)
    )
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
    return angles.cos(), angles.sin()


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_offset: int = 0
) -> torch.Tensor:
    """Half-rotation RoPE applied to (B, H, T, D)."""
    T = x.shape[-2]
    cos_slice = cos[position_offset : position_offset + T].view(1, 1, T, -1)
    sin_slice = sin[position_offset : position_offset + T].view(1, 1, T, -1)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat(
        [x1 * cos_slice - x2 * sin_slice, x1 * sin_slice + x2 * cos_slice], dim=-1
    )


@dataclass
class TransformerConfig:
    """Hyperparameters. The defaults are sized so that a 4-agent society
    fits comfortably in CPU memory on a MacBook."""

    vocab_size: int = 256
    hidden_dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 2  # grouped-query attention: 4 Q heads share 2 KV heads
    mlp_mult: float = 8 / 3
    max_seq_len: int = 1024
    rope_base: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        assert self.hidden_dim % self.n_heads == 0
        return self.hidden_dim // self.n_heads


class GroupedQueryAttention(nn.Module):
    """Causal multi-head self-attention with grouped-query reduction.

    Each query head still has its own parameters, but K and V are shared
    across groups of query heads (n_heads / n_kv_heads heads per group).
    Cuts KV memory and bandwidth at inference time — the technique
    behind every fast modern open model.
    """

    def __init__(self, cfg: TransformerConfig, block_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.idx = block_idx
        self.n_q_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        assert self.n_q_heads % self.n_kv_heads == 0, (
            "n_heads must be divisible by n_kv_heads"
        )
        self.head_dim = cfg.head_dim
        self.heads_per_kv = self.n_q_heads // self.n_kv_heads

        q_proj_dim = self.n_q_heads * self.head_dim
        kv_proj_dim = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(cfg.hidden_dim, q_proj_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_dim, kv_proj_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_dim, kv_proj_dim, bias=False)
        self.o_proj = nn.Linear(q_proj_dim, cfg.hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_offset: int = 0,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = _hook(q, f"block_{self.idx}.attn_q")
        k = _hook(k, f"block_{self.idx}.attn_k")
        v = _hook(v, f"block_{self.idx}.attn_v")

        # (B, H, T, D) shapes
        q = q.view(B, T, self.n_q_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin, position_offset)
        k = apply_rope(k, cos, sin, position_offset)

        # Repeat KV heads to match Q heads for grouped attention.
        k = k.repeat_interleave(self.heads_per_kv, dim=1)
        v = v.repeat_interleave(self.heads_per_kv, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.o_proj(out)
        return _hook(out, f"block_{self.idx}.attn_out")


class SwiGLU(nn.Module):
    """SwiGLU MLP (Shazeer 2020). Three projections: gate, up, down."""

    def __init__(self, cfg: TransformerConfig, block_idx: int) -> None:
        super().__init__()
        self.idx = block_idx
        inner = int(cfg.mlp_mult * cfg.hidden_dim)
        inner = ((inner + 7) // 8) * 8  # align to multiple of 8
        self.gate_proj = nn.Linear(cfg.hidden_dim, inner, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_dim, inner, bias=False)
        self.down_proj = nn.Linear(inner, cfg.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre = F.silu(self.gate_proj(x)) * self.up_proj(x)
        pre = _hook(pre, f"block_{self.idx}.mlp_pre")
        out = self.down_proj(pre)
        return _hook(out, f"block_{self.idx}.mlp_out")


class Block(nn.Module):
    """One transformer block. Pre-norm structure with hook points at
    every residual stream position."""

    def __init__(self, cfg: TransformerConfig, idx: int) -> None:
        super().__init__()
        self.idx = idx
        self.attn_norm = RMSNorm(cfg.hidden_dim)
        self.attn = GroupedQueryAttention(cfg, idx)
        self.mlp_norm = RMSNorm(cfg.hidden_dim)
        self.mlp = SwiGLU(cfg, idx)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_offset: int = 0,
    ) -> torch.Tensor:
        x = _hook(x, f"block_{self.idx}.resid_pre")
        n = self.attn_norm(x)
        n = _hook(n, f"block_{self.idx}.attn_norm")
        x = x + self.attn(n, cos, sin, position_offset)
        x = _hook(x, f"block_{self.idx}.resid_mid")
        n = self.mlp_norm(x)
        n = _hook(n, f"block_{self.idx}.mlp_norm")
        x = x + self.mlp(n)
        return _hook(x, f"block_{self.idx}.resid_post")


class HookedTransformer(nn.Module):
    """The hook-instrumented transformer.

    Use either the bare ``forward`` (returns hidden, logits) or wrap a
    call in ``hooks(ctx)`` to capture / intervene on intermediate
    activations. The model never holds onto hook state between calls —
    a fresh forward pass starts with the current top-of-stack
    ``HookContext`` (or the empty default).
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.hidden_dim)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self, token_ids: torch.Tensor, position_offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(token_ids)
        x = _hook(x, "embed")
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin, position_offset)
        x = self.final_norm(x)
        x = _hook(x, "final_norm")
        logits = self.lm_head(x)
        logits = _hook(logits, "logits")
        return x, logits

    # ──────────────────────────────────────────────────────────────────
    # Convenience: run with hook context and return captured activations
    # ──────────────────────────────────────────────────────────────────
    def run_with_capture(
        self, token_ids: torch.Tensor, sites: list[str]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass while recording activations at the requested sites.

        Returns ``(logits, captured)`` where ``captured`` is a dict
        keyed by site name.
        """
        ctx = HookContext(capture=sites)
        with hooks(ctx):
            _, logits = self.forward(token_ids)
        return logits, ctx.captured

    def run_with_intervention(
        self, token_ids: torch.Tensor, interventions: dict[str, torch.Tensor | HookFn]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass with one or more activations clamped / replaced.

        ``interventions`` is a dict from site name to either a tensor
        (replaces the activation directly) or a callable that takes the
        current activation and the site name and returns the
        intervention. Useful for ablation studies and counterfactual
        reasoning."""
        ctx = HookContext(replace=interventions, capture=list(interventions.keys()))
        with hooks(ctx):
            _, logits = self.forward(token_ids)
        return logits, ctx.captured

    def site_names(self) -> list[str]:
        """Every hook site the model exposes, in dependency order. Useful
        for iterating over all activations during analysis."""
        names = ["embed"]
        for i in range(self.cfg.n_layers):
            for suffix in (
                "resid_pre",
                "attn_norm",
                "attn_q",
                "attn_k",
                "attn_v",
                "attn_out",
                "resid_mid",
                "mlp_norm",
                "mlp_pre",
                "mlp_out",
                "resid_post",
            ):
                names.append(f"block_{i}.{suffix}")
        names += ["final_norm", "logits"]
        return names
