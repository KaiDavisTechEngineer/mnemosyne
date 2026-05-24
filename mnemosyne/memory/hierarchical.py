"""Hierarchical memory for MNEMOSYNE agents.

The memory architecture is modeled on the working / episodic / semantic
distinction from cognitive science (Tulving 1972, more recent
neuroscience), adapted for transformer agents:

* **Working memory** is a small fixed-capacity FIFO buffer of the
  most recent hidden-state vectors plus a sparse code from each. It is
  read every forward pass — the agent literally sees its own recent
  thoughts as part of its context.

* **Episodic memory** stores entire reasoning episodes (input → trace
  → outcome) as compact key-value pairs. Keys are pooled hidden-state
  summaries; values are the full episode payloads. Retrieved by
  cosine similarity at inference time.

* **Semantic memory** stores distilled *concepts* — clusters of
  features that have co-fired across many episodes and acquired stable
  meaning. These are the agent's "learned vocabulary of what its
  inputs are about."

A **consolidation** routine runs offline (think of it as the agent's
sleep): it compresses episodic memories into semantic concepts by
clustering recurring feature patterns. This mirrors hippocampal →
neocortical consolidation in mammals.

Why this matters
----------------
Single-context-window agents have no real continuity across tasks.
Hierarchical memory gives MNEMOSYNE agents a *biography*: a learned
sense of "what I've seen before, what worked, what didn't" that
informs new decisions without forcing a 1M-token context window.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import torch


# ─────────────────────────────────────────────────────────────────────
# Working memory: tiny, fast, FIFO
# ─────────────────────────────────────────────────────────────────────
@dataclass
class WorkingMemory:
    """A small FIFO buffer of recent (hidden_state, sparse_code) pairs.

    The agent reads working memory every forward pass — these are the
    "thoughts I had a moment ago."
    """

    capacity: int = 8
    items: deque = field(default_factory=lambda: deque(maxlen=8))

    def __post_init__(self) -> None:
        if self.items.maxlen != self.capacity:
            self.items = deque(self.items, maxlen=self.capacity)

    def push(
        self,
        hidden: torch.Tensor,
        sparse_code: Optional[torch.Tensor] = None,
        meta: Optional[dict] = None,
    ) -> None:
        self.items.append(
            {
                "hidden": hidden.detach().clone(),
                "sparse_code": sparse_code.detach().clone()
                if sparse_code is not None
                else None,
                "meta": meta or {},
                "timestamp": time.time(),
            }
        )

    def hiddens(self) -> torch.Tensor:
        """Return all working-memory hidden states stacked, or empty if none."""
        if not self.items:
            return torch.zeros(0)
        return torch.stack([it["hidden"] for it in self.items], dim=0)

    def clear(self) -> None:
        self.items.clear()

    def __len__(self) -> int:
        return len(self.items)


# ─────────────────────────────────────────────────────────────────────
# Episodic memory: vector store of past reasoning episodes
# ─────────────────────────────────────────────────────────────────────
@dataclass
class Episode:
    """A complete reasoning episode the agent has lived through."""

    key: torch.Tensor  # pooled summary used for retrieval
    input_text: str
    output_text: str
    feature_signature: Optional[torch.Tensor]  # which SAE features fired most
    outcome: dict[str, Any]  # outcome metrics ("correct", "verifier_passed", etc.)
    timestamp: float = field(default_factory=time.time)
    retrieval_count: int = 0


class EpisodicMemory:
    """A simple cosine-similarity vector store of episodes.

    Production systems would use FAISS, HNSW, or scaNN — at MNEMOSYNE's
    scale (hundreds to thousands of episodes), a plain torch tensor
    indexed by argsort is fast enough and avoids an external
    dependency. The interface is designed so a richer index can be
    dropped in later.
    """

    def __init__(self, dim: int, max_episodes: int = 4096) -> None:
        self.dim = dim
        self.max_episodes = max_episodes
        self._episodes: list[Episode] = []
        self._key_matrix: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return len(self._episodes)

    def store(self, episode: Episode) -> int:
        """Add an episode. If at capacity, evict the least-recently-retrieved."""
        if len(self._episodes) >= self.max_episodes:
            # Find least-recently-retrieved episode.
            evict_idx = min(
                range(len(self._episodes)),
                key=lambda i: (
                    self._episodes[i].retrieval_count,
                    self._episodes[i].timestamp,
                ),
            )
            self._episodes.pop(evict_idx)
        self._episodes.append(episode)
        self._rebuild_index()
        return len(self._episodes) - 1

    def retrieve(
        self, query: torch.Tensor, k: int = 4, similarity_threshold: float = 0.0
    ) -> list[tuple[Episode, float]]:
        """Return the top-k most similar episodes by cosine similarity.

        Only returns episodes whose similarity exceeds the threshold —
        this prevents hallucinated retrievals when nothing relevant is
        stored yet.
        """
        if not self._episodes or self._key_matrix is None:
            return []
        q = query.detach().view(-1)
        q = q / (q.norm() + 1e-8)
        keys = self._key_matrix  # (N, D)
        # Cosine similarity (keys already normalized when stored).
        sims = keys @ q  # (N,)
        k_eff = min(k, len(self._episodes))
        vals, idx = sims.topk(k_eff)
        out: list[tuple[Episode, float]] = []
        for v, i in zip(vals.tolist(), idx.tolist()):
            if v < similarity_threshold:
                break
            self._episodes[i].retrieval_count += 1
            out.append((self._episodes[i], v))
        return out

    def _rebuild_index(self) -> None:
        keys = torch.stack([e.key for e in self._episodes], dim=0)
        keys = keys / (keys.norm(dim=-1, keepdim=True) + 1e-8)
        self._key_matrix = keys


# ─────────────────────────────────────────────────────────────────────
# Semantic memory: distilled concepts
# ─────────────────────────────────────────────────────────────────────
@dataclass
class Concept:
    """A stable cluster of features that have learned a shared meaning.

    Each concept has a *centroid* in feature-activation space; episodes
    whose feature signatures lie near the centroid count as instances
    of that concept. Concepts have human-readable labels that the
    agent (or a human operator) can edit.
    """

    centroid: torch.Tensor  # (n_features,) — normalized
    member_indices: list[int]  # indices into episodic memory
    label: str = ""
    confidence: float = 0.0  # 0..1; coherence of the cluster
    creation_time: float = field(default_factory=time.time)


class SemanticMemory:
    """Concept store with k-means-style updates during consolidation."""

    def __init__(self, n_features: int, max_concepts: int = 64) -> None:
        self.n_features = n_features
        self.max_concepts = max_concepts
        self._concepts: list[Concept] = []

    def __len__(self) -> int:
        return len(self._concepts)

    def concepts(self) -> list[Concept]:
        return list(self._concepts)

    def assign(self, feature_sig: torch.Tensor) -> Optional[tuple[int, float]]:
        """Find the concept whose centroid is closest to ``feature_sig``.

        Returns ``(concept_idx, similarity)`` or None if no concepts exist.
        """
        if not self._concepts:
            return None
        f = feature_sig.detach().view(-1)
        f = f / (f.norm() + 1e-8)
        centroids = torch.stack([c.centroid for c in self._concepts], dim=0)
        sims = centroids @ f
        idx = int(sims.argmax().item())
        return idx, float(sims[idx].item())

    def add_concept(
        self,
        centroid: torch.Tensor,
        members: list[int],
        label: str = "",
        confidence: float = 0.0,
    ) -> int:
        if len(self._concepts) >= self.max_concepts:
            # Evict the lowest-confidence concept.
            evict_idx = min(
                range(len(self._concepts)), key=lambda i: self._concepts[i].confidence
            )
            self._concepts.pop(evict_idx)
        c = centroid.detach().clone()
        c = c / (c.norm() + 1e-8)
        self._concepts.append(
            Concept(
                centroid=c,
                member_indices=list(members),
                label=label,
                confidence=confidence,
            )
        )
        return len(self._concepts) - 1

    def relabel(self, concept_idx: int, label: str) -> None:
        self._concepts[concept_idx].label = label


# ─────────────────────────────────────────────────────────────────────
# Consolidation: sleep-like compression of episodic → semantic
# ─────────────────────────────────────────────────────────────────────
def consolidate(
    episodic: EpisodicMemory,
    semantic: SemanticMemory,
    n_clusters: int = 8,
    n_iters: int = 20,
    min_cluster_size: int = 3,
) -> int:
    """Run k-means on episode feature signatures to produce new concepts.

    Returns the number of concepts added. Existing concepts are not
    modified — consolidation is additive. This is deliberate: the agent
    should not forget concepts already in semantic memory just because
    the recent episode distribution has shifted.

    The clusters are filtered: only clusters with at least
    ``min_cluster_size`` members are retained. Tiny clusters are
    noise and should not become concepts.
    """
    if len(episodic) < min_cluster_size * 2:
        return 0
    # Build a matrix of feature signatures.
    sigs = []
    valid_indices = []
    for i, ep in enumerate(episodic._episodes):
        if ep.feature_signature is None:
            continue
        sigs.append(ep.feature_signature.detach().view(-1))
        valid_indices.append(i)
    if len(sigs) < min_cluster_size * 2:
        return 0
    X = torch.stack(sigs, dim=0)
    X = X / (X.norm(dim=-1, keepdim=True) + 1e-8)

    # K-means with cosine distance.
    k = min(n_clusters, X.shape[0] // min_cluster_size)
    if k < 1:
        return 0
    # Initialize centroids by sampling.
    init_idx = torch.randperm(X.shape[0])[:k]
    centroids = X[init_idx].clone()

    for _ in range(n_iters):
        # Assign each point to nearest centroid.
        sims = X @ centroids.T  # (N, k)
        assign = sims.argmax(dim=-1)
        # Recompute centroids.
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(k, dtype=torch.long)
        for ci in range(k):
            mask = assign == ci
            if mask.any():
                new_centroids[ci] = X[mask].mean(dim=0)
                counts[ci] = int(mask.sum().item())
            else:
                # Re-initialize empty cluster to a random point.
                new_centroids[ci] = X[torch.randint(0, X.shape[0], (1,)).item()]
        new_centroids = new_centroids / (
            new_centroids.norm(dim=-1, keepdim=True) + 1e-8
        )
        if torch.allclose(new_centroids, centroids, atol=1e-4):
            centroids = new_centroids
            break
        centroids = new_centroids

    # Final assignment.
    sims = X @ centroids.T
    assign = sims.argmax(dim=-1)
    n_added = 0
    for ci in range(k):
        mask = assign == ci
        if mask.sum().item() < min_cluster_size:
            continue
        members = [valid_indices[i] for i, m in enumerate(mask.tolist()) if m]
        # Confidence: within-cluster similarity.
        in_cluster = X[mask]
        conf = float((in_cluster @ centroids[ci]).mean().item())
        semantic.add_concept(
            centroids[ci], members, label=f"concept_{len(semantic)}", confidence=conf
        )
        n_added += 1
    return n_added


# ─────────────────────────────────────────────────────────────────────
# Top-level memory bundle used by agents
# ─────────────────────────────────────────────────────────────────────
@dataclass
class AgentMemory:
    """The complete memory stack an agent carries."""

    working: WorkingMemory
    episodic: EpisodicMemory
    semantic: SemanticMemory

    @classmethod
    def build(
        cls,
        hidden_dim: int,
        n_features: int,
        working_capacity: int = 8,
        max_episodes: int = 4096,
        max_concepts: int = 64,
    ) -> "AgentMemory":
        return cls(
            working=WorkingMemory(capacity=working_capacity),
            episodic=EpisodicMemory(dim=hidden_dim, max_episodes=max_episodes),
            semantic=SemanticMemory(n_features=n_features, max_concepts=max_concepts),
        )

    def consolidate(self, **kwargs) -> int:
        """Run a consolidation pass. Returns concepts added."""
        return consolidate(self.episodic, self.semantic, **kwargs)
