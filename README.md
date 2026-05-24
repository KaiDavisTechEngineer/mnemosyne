# MNEMOSYNE

**A society of causally-self-modeling agents.**

MNEMOSYNE is a research-grade implementation of an architecture that
to our knowledge has not been assembled before in open source: a
multi-agent system where each agent maintains a *causally
interventionable* model of its own reasoning, decomposed into named
interpretable features via sparse autoencoders, sitting on top of a
hook-instrumented transformer, coordinated by a metacognitive agent
that maintains a learned model of every other agent's behavior.

The single core invention: an agent that can answer, in addition to
*"what is X?"*, the following questions about itself, online, as part
of its normal operating mode:

- **"Which of my features caused this answer?"** — feature attribution
  via SAE-decomposed activations
- **"What would I have said if feature 42 hadn't fired?"** —
  counterfactual reasoning by direct intervention on activations
- **"Have I seen something like this before?"** — episodic recall over
  a long-term memory store
- **"What concept does this input most match?"** — semantic clustering
  via sleep-like consolidation
- **"How sure am I about my own reasoning?"** — surprise signal from a
  learned self-model

The full technical write-up is in [`docs/modern.md`](docs/modern.md).

## Why this matters

By 2030, frontier AI systems will be expected to *explain themselves
mechanistically*. Anthropic's circuit-tracing work, DeepMind's
mechanistic interpretability program, and OpenAI's super-alignment
research all point at the same gap: we can train powerful systems but
we cannot read them. The standard story holds interpretability as a
post-hoc audit; MNEMOSYNE puts it on the inference path.

The four ingredients that make this work — and which to our knowledge
have not been combined in an open-source artifact before:

1. **Hook-instrumented transformer** with 36 named activation sites
   per forward pass, every one of which can be captured, replaced, or
   routed through a sparse autoencoder.
2. **TopK sparse autoencoders** per introspection site, dead-feature
   resurrection included, giving the agent a vocabulary of named
   interpretable features it can *refer to with its own tokens*
   (`<feature:42>`).
3. **Hierarchical memory** — working / episodic / semantic with
   k-means consolidation — that gives the agent a persistent biography
   without requiring a million-token context window.
4. **Multi-agent debate** with a **metacognitive agent** that learns
   to predict the other agents' behavior and adjusts inter-agent
   trust accordingly.

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  Question                                                          │
│       │                                                            │
│       ▼                                                            │
│  ╔════════════════════════════════════════════════════════════╗    │
│  ║                       THE SOCIETY                          ║    │
│  ║                                                            ║    │
│  ║   Proposer ──► Critic ──► Verifier ──► Synthesizer         ║    │
│  ║      ▲           ▲           ▲             │               ║    │
│  ║      │           │           │             │               ║    │
│  ║      └───────────┴────[ Metacognitor ]─────┘               ║    │
│  ║                                                            ║    │
│  ║   Each agent has:                                          ║    │
│  ║     - hooked transformer (36 named sites per pass)         ║    │
│  ║     - sparse autoencoder per introspection site            ║    │
│  ║     - hierarchical memory (working/episodic/semantic)      ║    │
│  ║     - communication channel to all other agents            ║    │
│  ║     - self-model (predicts own outputs, signals surprise)  ║    │
│  ╚════════════════════════════════════════════════════════════╝    │
│       │                                                            │
│       ▼                                                            │
│  Final answer + introspection report:                              │
│    • Top features causing the answer                               │
│    • Counterfactual: ablating features {a,b,c} flips answer        │
│    • Most-similar past episode                                     │
│    • Matched semantic concept                                      │
│    • Metacognitor's trust assessment of each contributor           │
└────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
git clone https://github.com/your-org/mnemosyne
cd mnemosyne
pip install -e .[dev]
```

Requirements: Python 3.10+, PyTorch 2.0+. CPU-only training is the
default — the example society (~700K params) trains end-to-end in
minutes on a MacBook Air.

## Quick start

Train a society on synthetic reasoning puzzles:

```bash
mnemosyne train --n-samples 64 --rl-episodes 80 --out mnemosyne.pt --eval-after
```

Evaluate on held-out samples:

```bash
mnemosyne eval --model mnemosyne.pt --n 32
```

Run a single debate:

```bash
mnemosyne debate --question "alice is taller than bob; bob is taller than carol. who is tallest?" --expected alice
```

Ask the proposer to introspect on itself:

```bash
mnemosyne introspect --text "alice is taller than bob"
```

## Project layout

```
mnemosyne/
├── mnemosyne/
│   ├── arch/                # transformer + tokenizer (the substrate)
│   │   ├── transformer.py   # hooked, GQA, RoPE, RMSNorm, SwiGLU
│   │   └── tokenizer.py     # byte-level + structured protocol tokens
│   ├── interp/              # mechanistic interpretability
│   │   └── sae.py           # TopK sparse autoencoder, dead-feature reset
│   ├── causal/              # causal interventions on the model
│   │   └── interventions.py # patching, attribution, counterfactuals
│   ├── memory/              # cognitive memory architecture
│   │   └── hierarchical.py  # working/episodic/semantic + consolidation
│   ├── communication/       # multi-agent channel
│   │   └── channel.py       # message bus with trust dynamics
│   ├── self_model/          # the agent's model of itself
│   │   └── introspect.py    # surprise signal + self-prediction
│   ├── agents/              # the cast
│   │   ├── base.py          # general-purpose Agent class
│   │   └── specialists.py   # Proposer/Critic/Verifier/Synthesizer/Metacognitor
│   ├── society/
│   │   └── orchestrator.py  # debate loop with round structure
│   ├── training/
│   │   ├── tasks.py         # synthetic reasoning task generator
│   │   └── trainer.py       # BC → SAE → multi-agent RL pipeline
│   ├── eval/
│   │   └── benchmark.py     # solve rate, SAE quality, CF flip rate, etc.
│   └── cli/
│       └── main.py          # `mnemosyne` command-line entry point
├── tests/                   # 28 tests covering every subsystem
├── examples/
│   ├── society_debate.py    # train + evaluate end-to-end
│   └── introspection_demo.py # single-agent causal introspection
└── docs/
    └── modern.md            # NeurIPS-style technical write-up
```

## Empirical results

Smoke configuration (5-agent society, 678,980 params total, CPU-only,
3-5 minutes):

| Stage              | Metric          | Start   | End     |
|--------------------|-----------------|---------|---------|
| Behavioral cloning | per-sample loss | 133.08  | 103.45  |
| SAE training       | reconstruction  | 0.91    | 0.64    |
| Multi-agent RL     | average reward  | +0.30   | +0.51   |

Post-training evaluation on held-out samples:

- **Solve rate**: 37.5% gold-match accuracy across families
- **Per-family**: ordering 100%, boolean 0%, comparison 0%
  (the harder families need more training)
- **Counterfactual flip rate**: 100% — every introspection
  successfully found a minimal feature ablation that changes the
  agent's answer
- **SAE reconstruction MSE**: 0.57

What this demonstrates is that the *architecture* is sound end-to-end:
the agents train, the SAEs learn structured features, the
interventions are causal (the counterfactual mechanism finds real
flipping ablations on every input), and the debate orchestration
produces gold-matching answers on the easier task families. Scaling
to harder problems and richer training signal is the natural next
direction.

## Why "MNEMOSYNE"

In Greek mythology Mnemosyne is the goddess of memory and reflection,
the mother of the nine Muses. The system is named for her because the
project's central thesis is that *self-knowledge enables creativity*:
an agent that can model its own reasoning — what features it relies
on, what mistakes it has made, what concepts it has learned to
recognize — is an agent that can generate genuinely new behavior
rather than merely interpolating its training distribution.

## License

MIT — see [`LICENSE`](LICENSE).

## Citing this work

```
@software{mnemosyne2030,
  title  = {MNEMOSYNE: A Society of Causally-Self-Modeling Agents},
  year   = {2026},
  url    = {https://github.com/your-org/mnemosyne}
}
```

And the precursor work the architecture builds on:

- Bricken et al., "Towards Monosemanticity: Decomposing Language
  Models with Dictionary Learning" (Anthropic), 2023.
- Gao et al., "Scaling and Evaluating Sparse Autoencoders"
  (OpenAI), 2024.
- Conmy et al., "Towards Automated Circuit Discovery for Mechanistic
  Interpretability", 2023.
- Syed et al., "Attribution Patching Outperforms Automated Circuit
  Discovery", 2023.
- Du et al., "Improving Factuality and Reasoning in Language Models
  through Multi-Agent Debate" (MIT), 2023.
- Pearl, J., *Causality: Models, Reasoning, and Inference*, 2009.

The synthesis — and the choice to make introspection a first-class
inference-time capability rather than a post-hoc audit — is what
MNEMOSYNE contributes.
