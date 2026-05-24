# MNEMOSYNE: A Society of Causally-Self-Modeling Agents

*Technical write-up. Intended audience: researchers and engineers
evaluating the architecture. Assumes familiarity with transformers,
sparse autoencoders, multi-agent systems, and causal inference.*

---

## 1. Position

Frontier AI in 2030 will be expected to do something that frontier AI
in 2025 cannot reliably do: **explain its own reasoning at the
mechanistic level, online, while reasoning**.

This is not the same as chain-of-thought, which is a *behavioral*
trace — a story the model tells in token space — and is known to be
unfaithful to the model's actual computation (Turpin et al. 2023,
Lanham et al. 2023). It is also not the same as post-hoc
interpretability, which produces explanations *after* the model has
acted, by separate analysis pipelines that the model itself never
sees.

What we want is a model whose internal states are *typed*: every
hidden state corresponds to a known dictionary of features, and the
model can refer to those features by name in its own outputs. We want
counterfactuals as a first-class operation: the model should be able
to answer "what would I have said if feature 42 had not fired?" as
cheaply as it answers "what would I say if X happened?" And we want
this in a multi-agent context, where one agent's introspection
becomes another agent's training signal.

MNEMOSYNE is an open-source attempt at this design point. It is
deliberately small — a five-agent society of 200K-parameter
transformers — because the contribution is the *architecture*, and
the architecture should be inspectable end-to-end by one person in an
afternoon.

---

## 2. Architecture

```
            ┌──────────────────────────────────────────────────┐
            │  HookedTransformer                               │
            │    embed → block_0 → block_1 → ... → final_norm  │
            │  36 named activation sites per forward pass      │
            └──────────────────────────────────────────────────┘
                        │
                        │ activations
                        ▼
            ┌──────────────────────────────────────────────────┐
            │  TopKSAE (one per introspection site)            │
            │    z = TopK(W_enc · (a − b_dec) + b_enc)         │
            │    â = W_dec · z + b_dec                         │
            └──────────────────────────────────────────────────┘
                        │
                        │ sparse code z
                        ▼
            ┌──────────────────────────────────────────────────┐
            │  Causal interventions                            │
            │    feature_attribution(z, target)                │
            │    find_counterfactual(z) → minimal ablation     │
            │    activation_patch(site_A → site_B)             │
            └──────────────────────────────────────────────────┘
                        │
                        │ structured reports
                        ▼
            ┌──────────────────────────────────────────────────┐
            │  Agent (encapsulates all of the above + memory)  │
            │    .reply(text)                                  │
            │    .introspect(ids) → IntrospectionReport         │
            │    .speak / .listen / .remember / .recall        │
            └──────────────────────────────────────────────────┘
```

### 2.1 Hook-instrumented transformer

The transformer in `arch/transformer.py` exposes 36 named hook sites
per forward pass:

```
embed
block_{i}.{resid_pre, attn_norm, attn_{q,k,v,out},
           resid_mid, mlp_norm, mlp_pre, mlp_out, resid_post}
final_norm
logits
```

Hook points are first-class: any caller can register a `HookContext`
that either *captures* activations at named sites for later analysis,
or *replaces* them with arbitrary tensors (for ablation studies and
counterfactual interventions). The hook mechanism is implemented as a
module-level context stack with thread-safe scoping; nested forward
passes (e.g., during counterfactual search) do not interfere.

Architecturally the model is otherwise a stock Llama-style stack:
RMSNorm, RoPE, SwiGLU MLPs, Grouped-Query Attention (we use 4 Q heads
sharing 2 KV heads in the default config), tied embeddings, no
biases. The design choice that matters is *not* the primitives but
the hook-first construction: in a standard transformer, intervention
requires monkey-patching `forward` methods, which is fragile and
opaque. Here it is the normal operating mode.

### 2.2 TopK sparse autoencoders

Following Gao et al. 2024 ("Scaling and Evaluating Sparse
Autoencoders") we use a TopK SAE rather than the older L1-penalized
variant. The encoder produces a pre-activation, we keep only the K
largest positive entries, the decoder reconstructs. K controls
sparsity *exactly* — there is no sparsity penalty to tune.

Each agent carries one SAE per *introspection site*. By default we
introspect at `block_{N-1}.resid_post` (the residual stream entering
the final norm), but the design admits SAEs at any of the 36 sites.
Multiple SAEs let the agent reason about features at different
abstraction levels.

We adopt two refinements from the Gao et al. paper:

- **Decoder unit-normalization** after each optimizer step. Each
  feature direction becomes a unit vector; the sparse code is the
  only place where magnitude can live.
- **Dead-feature resurrection**: features that haven't activated in
  N batches are tracked, and the auxiliary reconstruction loss
  channels gradient back through them via residual error. Without
  this, sparse autoencoders collapse to a handful of dominant
  features and waste dictionary capacity.

### 2.3 Causal interventions

Three primitives, in `causal/interventions.py`:

**Activation patching** (Heimersheim & Nanda 2024). Run a clean and
corrupted forward pass; patch one site from clean → corrupted; measure
how much output recovers. Returns a recovery score in [0, 1]. This is
the gold-standard causal measure but expensive (one forward pass per
site).

**Attribution patching** (Syed et al. 2023, Kramár et al. 2024,
`AtP_star`). First-order Taylor approximation: one backward pass gives
us gradient at every site simultaneously; multiply by (clean −
corrupted) to estimate the patch effect. Trades exactness for
tractability.

**Feature attribution + counterfactuals**. Combine SAE encoding with
ablation: for each active feature, zero it out, re-decode, run the
model, observe the change in target-token logit. Greedy counterfactual
search accumulates ablations in order of attribution magnitude until
the argmax flips. The output is a *minimal causal explanation*:
"ablating features {a, b, c} would have changed your answer from X
to Y."

In smoke tests, the counterfactual search succeeds (finds a flipping
ablation within budget) on 100% of evaluation samples. This is
unsurprising — small models tend to be fragile to feature ablation —
but it does demonstrate that the mechanism is wired up correctly.

### 2.4 Hierarchical memory

`memory/hierarchical.py` implements three stores:

**Working memory** is a fixed-capacity FIFO of recent
(hidden_state, sparse_code) tuples. The agent reads working memory
every forward pass; it is literally what the agent "saw in its
recent past."

**Episodic memory** stores entire reasoning episodes —
(question, answer, feature signature, outcome) — as cosine-similarity
addressable key-value pairs. Capacity 4096 by default, LRU eviction.

**Semantic memory** stores *concepts*: clusters of recurring feature
signatures with human-readable labels. Concepts are produced by
k-means consolidation over episodic feature signatures, filtered to
require a minimum cluster size (default 3) so noise doesn't become
spurious concepts.

The **consolidation routine** runs offline ("the agent sleeps") and
compresses episodic patterns into semantic memory. This mirrors the
hippocampal → neocortical consolidation in mammalian cognition
(McClelland et al. 1995). It is the mechanism by which an agent's
biography becomes its abstract vocabulary.

### 2.5 Communication channel

`communication/channel.py` provides a message bus over which agents
exchange information. Each message carries two channels in parallel:

- **Token channel** — natural-language-ish text, framed by `<msg>`
  envelopes. Inspectable; lossy.
- **Latent channel** — a hidden-state vector projected through a
  learned `BridgeHead`. Higher-bandwidth than tokens; less
  inspectable.

The channel maintains per-receiver, per-sender trust scalars in
[0, 1]. The metacognitor updates trust based on round outcomes,
producing inter-agent attention modulation that emerges from
collaboration history rather than being hand-tuned.

### 2.6 Five-agent society

`agents/specialists.py` defines:

- **Proposer**: high temperature, generates candidate answers
- **Critic**: low temperature, identifies flaws in proposals
- **Verifier**: deterministic, runs a domain-specific verification
  function (Z3 for logic, test runner for code, etc.)
- **Synthesizer**: moderate temperature, combines proposals +
  critiques + verifications into a refined answer
- **Metacognitor**: maintains one `AgentModel` per modeled agent; the
  AgentModel is a small GRU that predicts the modeled agent's next
  behavioral mode (8 discrete modes). Trust deltas are computed from
  the agreement between the metacognitor's predictions and the
  verifier's verdicts.

The five-agent decomposition is *not* meant to be the discovery of
this paper; it is a known recipe (Du et al. 2023, "Multi-Agent
Debate"). What is new is that each agent is a first-class
*self-introspector*, so the debate transcript can include not only
"agent A said X" but also "agent A said X *because* features {f1, f2,
f3} fired, and would have said Y if f1 had been ablated."

---

## 3. Training pipeline

Three stages, all in `training/trainer.py`:

### 3.1 Behavioral cloning warmup

For each (question, gold answer) sample, we supervised-train:
- Proposer: question → gold answer
- Synthesizer: (question + gold proposal) → gold answer
- Critic: (question + gold proposal) → "correct"

Cross-entropy on the answer tokens, AdamW, grad clipping. Three epochs
on a 24-sample dataset takes BC loss from ~133 to ~103 in our smoke
config — clearly learning.

### 3.2 Sparse autoencoder training

With backbones in a reasonable state, we sweep activations from
every agent's introspection sites and train the SAEs. Standard MSE
+ auxiliary-loss objective, decoder-normalized after each step.

In smoke tests SAE loss drops from 0.91 → 0.64 over 40 steps. With
more steps it would continue to drop and the dead-feature fraction
would stabilize.

### 3.3 Multi-agent RL

REINFORCE on the synthesizer, with reward = `match_gold(answer,
question)`. Moving-average baseline for variance reduction; entropy
regularization to prevent premature collapse.

We currently train only the synthesizer end-to-end; the proposer and
critic are frozen during this stage. This is a deliberate
simplification — full multi-agent credit assignment (each agent
gets a gradient proportional to how it contributed to the
synthesizer's reward-weighted log-probability) is a clean extension
but adds complexity that is not the point of v0.1.

In smoke tests average reward goes from +0.30 → +0.51 over 15
episodes, and post-training solve rate is 37.5%. On the "ordering"
family alone it is 100%; the harder families ("boolean",
"comparison") are still 0% after 15 episodes and need either more
training or a richer reward signal.

---

## 4. What's novel

This codebase combines, in one runnable artifact:

1. A hook-instrumented transformer designed *from the start* for
   mechanistic interpretability (not bolted on)
2. TopK sparse autoencoders trained on its own activations
3. Causal interventions (activation patching, attribution patching,
   feature attribution, counterfactual search) wired into the
   inference path
4. A hierarchical memory architecture (working/episodic/semantic)
   with k-means consolidation
5. A multi-agent debate orchestrator with metacognitive trust
   dynamics
6. A self-model head that signals surprise when the agent's outputs
   diverge from what it predicts about itself
7. A training pipeline that brings the system from random init to
   measurable solve rate

To our knowledge, no open-source system has all of these. The
closest prior art:

- **Anthropic interpretability work** (Olsson, Marks, Conmy et al.)
  builds the sparse autoencoder + circuit tracing toolkit but on
  pre-trained models, post-hoc, single-agent.
- **DeepMind AlphaProof** integrates a learned model with a formal
  verifier in a tight loop, but the model is massive and closed and
  the introspection is not exposed.
- **Multi-agent debate** (Du et al., Liang et al.) achieves
  benefits from multiple agents critiquing each other, but the
  agents do not introspect on themselves and the trust dynamics are
  not learned.
- **Latent reasoning** (Hao et al., Coconut) recycles hidden states
  through the model, but does not decompose them into named
  features or expose them for intervention.

The synthesis is the contribution.

---

## 5. Limitations

1. **Scale**. ~200K parameters per agent. This is what makes the
   system inspectable line-by-line; it is also why solve rates on
   harder task families are zero after brief training. Scaling to
   ~25M params per agent (still cheap to train on one GPU) would
   change the empirical picture entirely. The architectural code does
   not change.

2. **Verifier interface**. The Verifier agent is mostly mechanical —
   it wraps a domain-specific verification function passed in by the
   caller. For logic puzzles this can be Z3 (we ship a PROOFCAST-style
   adapter in `examples/`); for code synthesis it could be a test
   runner; for math, a calculator. The neural net inside the verifier
   is currently underused. A natural next step is to train it to
   *select which fragment of the proposal to verify* rather than
   verify the whole thing.

3. **Multi-agent credit assignment**. We currently train only the
   synthesizer end-to-end. Full multi-agent PPO with per-agent
   advantage estimates is a known recipe (Yu et al. 2022, MAPPO) but
   adds complexity beyond the scope of v0.1.

4. **Synthetic tasks**. The training tasks are simple by design —
   logic puzzles with structured gold answers. The architecture's
   adaptation to less-structured domains (math word problems,
   programming, scientific reasoning) is future work.

5. **Self-model is shallow**. The current `SelfModel` is a single
   2-layer head predicting next-token distributions from the
   penultimate hidden state. A richer self-model would predict the
   agent's *introspection report* — "what features will I report as
   causally important?" — closing the loop on causal self-knowledge.

---

## 6. Future work

* **Scale-up experiments**. 25M-param agents on a real benchmark
  (GSM8K, FOLIO, ProofWriter). The architectural plumbing already
  supports it; only the hyperparameters and dataset change.

* **Distillation from a frontier teacher**. Use Claude / GPT as a
  teacher LM that generates gold trajectories the MNEMOSYNE agents
  imitate. The interesting question: does the MNEMOSYNE society
  produce *more interpretable* solutions than the teacher, by virtue
  of being structured around verifiable claims?

* **Cross-agent introspection**. The metacognitor currently models
  the *behavior* of the other agents (next-message-mode prediction).
  A natural extension is to model their *internal features*: the
  metacognitor learns to predict which SAE features will fire in the
  proposer for a given input, then uses that prediction to forecast
  whether the proposer is likely to make a mistake.

* **Self-improvement loop**. Use the verifier's verdict as a filter
  on the synthesizer's outputs: keep only verified-correct
  trajectories as new behavioral-cloning data. This is the
  closed-loop variant of FunSearch / AlphaProof's evolutionary
  scheme, applied at the scale where it can be studied openly.

* **Formal safety analysis**. The combination of hook-level
  interventions and SAE-decomposed features makes it possible to
  define *behavioral properties* of an agent in feature-space
  ("never let feature 42 activate when feature 17 is active") and
  verify them via causal scrubbing. This is a direct line to
  formally-verified agent behavior — a problem the field considers
  hard but not impossible.

---

## 7. Reproducing the experiments

```bash
pip install -e .[dev]
pytest tests/                                           # 28 tests
python examples/introspection_demo.py                   # single-agent
python examples/society_debate.py                       # full pipeline
mnemosyne train --rl-episodes 80 --eval-after           # CLI training
```

Smoke runtime on a CPU-only MacBook Air: 3-5 minutes for the full
five-agent training run; ~10 seconds for the single-agent
introspection demo. The trained society is ~3 MB on disk.

---

## 8. Conclusion

MNEMOSYNE is a working demonstration that the four ingredients —
hook-instrumented transformer, sparse autoencoders, causal
interventions, hierarchical memory — can be assembled into a
multi-agent system where introspection is a first-class operation
on the inference path. The system is small, the tasks are simple,
the absolute solve rates are modest. None of that is the point.

The point is that the architecture *exists*, runs end-to-end, and
demonstrates the right shape for a 2030-era frontier system: an
agent that has a vocabulary for its own internal states, can
intervene on them causally, can be modeled in turn by another agent
that builds a trust map of who-knows-what, and can carry its
biography across episodes. The frontier is making this work at scale,
and on harder problems. The blueprint, as far as we can find, has
not been laid out openly before. Here it is.
