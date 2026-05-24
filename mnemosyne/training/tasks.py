"""Synthetic reasoning tasks used to train MNEMOSYNE agents.

We deliberately keep the tasks small and structured: transitive
ordering, boolean implication chains, simple comparisons. These map
cleanly onto well-defined symbolic targets so we can compute a
ground-truth reward signal without needing a teacher LM.

A task is a callable that produces ``Sample`` objects::

    sample = task()
    sample.question        # natural-language input
    sample.gold_answer     # the correct response
    sample.verify(proposal) # returns {"verdict": "ok"|"fail"|"unknown", ...}

The same ``verify`` callable can be passed to the verifier agent as
the ``verification_fn``, closing the loop: the same reward function
that scores training trajectories grades the multi-agent debate.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

NAMES = [
    "alice",
    "bob",
    "carol",
    "dave",
    "eve",
    "frank",
    "grace",
    "henry",
    "iris",
    "jack",
    "kate",
    "leo",
    "mia",
    "nick",
    "olivia",
    "pete",
]


@dataclass
class Sample:
    """One training sample."""

    question: str
    gold_answer: str
    family: str
    difficulty: int = 1
    verify: Callable[[str, str], dict] = field(
        default=lambda q, p: {"verdict": "unknown"}
    )
    metadata: dict = field(default_factory=dict)


def _make_verifier(
    gold_answer: str, accepted_keywords: list[str]
) -> Callable[[str, str], dict]:
    """Return a verification function for a sample.

    Accepts as 'ok' any proposal whose lowercased text contains the
    gold answer or one of the accepted keywords. Otherwise 'fail'.
    The signature is ``(question, proposal) -> verdict_dict`` to match
    the Verifier agent's expectation.
    """
    gold_lower = gold_answer.lower().strip()
    kws = [k.lower() for k in accepted_keywords] + [gold_lower]

    def verify(question: str, proposal: str) -> dict:
        p = proposal.lower()
        if any(kw in p for kw in kws if kw):
            return {"verdict": "ok", "reason": f"matched gold {gold_answer!r}"}
        return {"verdict": "fail", "reason": f"expected {gold_answer!r}"}

    return verify


# ─────────────────────────────────────────────────────────────────────
# Family 1: transitive ordering
# ─────────────────────────────────────────────────────────────────────
def task_ordering(rng: random.Random, n: int = 4) -> Sample:
    """Generate "X taller than Y; Y taller than Z; who is tallest?" """
    names = rng.sample(NAMES, n)
    # Hidden total order. names[order[0]] is shortest, names[order[-1]] tallest.
    order = list(range(n))
    rng.shuffle(order)
    sorted_names = [names[order[i]] for i in range(n)]

    # Emit adjacent comparisons in random order.
    parts = []
    for i in range(n - 1):
        parts.append(f"{sorted_names[i + 1]} is taller than {sorted_names[i]}")
    rng.shuffle(parts)

    tallest = sorted_names[-1]
    shortest = sorted_names[0]
    q_type = rng.choice(["tallest", "shortest"])
    if q_type == "tallest":
        question = "; ".join(parts) + ". Who is tallest?"
        gold = tallest
    else:
        question = "; ".join(parts) + ". Who is shortest?"
        gold = shortest

    return Sample(
        question=question,
        gold_answer=gold,
        family="ordering",
        difficulty=min(5, max(1, n - 2)),
        verify=_make_verifier(gold, [gold]),
        metadata={"names": names, "sorted": sorted_names},
    )


# ─────────────────────────────────────────────────────────────────────
# Family 2: boolean implication
# ─────────────────────────────────────────────────────────────────────
def task_boolean(rng: random.Random, n: int = 3) -> Sample:
    """Generate "if A then B; if B then C; A is true. Is C true?" """
    names = rng.sample(NAMES, min(n, len(NAMES)))
    parts = [f"{names[0]} is true"]
    for i in range(len(names) - 1):
        parts.append(f"if {names[i]} is true then {names[i + 1]} is true")
    rng.shuffle(parts[1:])
    target = names[-1]
    question = "; ".join(parts) + f". Is {target} true?"
    gold = "yes"
    return Sample(
        question=question,
        gold_answer=gold,
        family="boolean",
        difficulty=min(5, max(1, n - 1)),
        verify=_make_verifier(gold, ["yes", "true", target]),
        metadata={"chain": names},
    )


# ─────────────────────────────────────────────────────────────────────
# Family 3: numeric comparison
# ─────────────────────────────────────────────────────────────────────
def task_comparison(rng: random.Random, n: int = 3) -> Sample:
    """Generate "X is N; Y > X; Z > Y; is Z > N?" — chained comparisons."""
    names = rng.sample(NAMES, n)
    base = rng.randint(20, 50)
    parts = [f"{names[0]} is {base}"]
    for i in range(n - 1):
        parts.append(f"{names[i + 1]} is greater than {names[i]}")
    rng.shuffle(parts[1:])
    question = "; ".join(parts) + f". Is {names[-1]} greater than {base}?"
    gold = "yes"
    return Sample(
        question=question,
        gold_answer=gold,
        family="comparison",
        difficulty=min(5, max(1, n - 1)),
        verify=_make_verifier(gold, ["yes", "greater", names[-1]]),
        metadata={"base": base, "names": names},
    )


# ─────────────────────────────────────────────────────────────────────
# Top-level mixed sampler
# ─────────────────────────────────────────────────────────────────────
def sample_task(rng: random.Random, family: str = "mix", difficulty: int = 2) -> Sample:
    """Sample one task. ``family`` is "ordering" | "boolean" | "comparison"
    | "mix"; ``difficulty`` controls the size."""
    if family == "mix":
        family = rng.choice(["ordering", "boolean", "comparison"])
    n = difficulty + 2
    if family == "ordering":
        return task_ordering(rng, n=n)
    if family == "boolean":
        return task_boolean(rng, n=min(n, 5))
    if family == "comparison":
        return task_comparison(rng, n=min(n, 5))
    raise ValueError(f"unknown family: {family}")


def sample_dataset(
    n: int,
    seed: int = 0,
    family: str = "mix",
    difficulty_range: tuple[int, int] = (1, 3),
) -> list[Sample]:
    """Generate a fixed dataset."""
    rng = random.Random(seed)
    out: list[Sample] = []
    for _ in range(n):
        d = rng.randint(*difficulty_range)
        out.append(sample_task(rng, family=family, difficulty=d))
    return out
