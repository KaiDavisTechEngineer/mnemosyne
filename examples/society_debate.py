"""End-to-end example: train MNEMOSYNE on synthetic puzzles, evaluate,
then introspect.

Runtime: ~3-5 minutes on a CPU-only laptop.

What it demonstrates:
  1. Building a five-agent society from scratch
  2. Training the agents via three-stage curriculum (BC → SAE → RL)
  3. Evaluating on held-out samples with rich diagnostics
  4. Running one debate end-to-end and printing the transcript
  5. Asking the proposer to introspect on its own answer — top features,
     counterfactual, episodic match
"""

from __future__ import annotations

import random

import torch

from mnemosyne import (
    AgentConfig,
    Critic,
    CommunicationChannel,
    DebateConfig,
    Metacognitor,
    Proposer,
    SAEConfig,
    Society,
    Synthesizer,
    Tokenizer,
    TrainConfig,
    TransformerConfig,
    Verifier,
    evaluate_society,
    sample_dataset,
    train_all,
)


def main() -> None:
    torch.manual_seed(0)
    random.seed(0)
    tok = Tokenizer.build()
    channel = CommunicationChannel()

    HIDDEN = 48
    N_LAYERS = 2
    site = f"block_{N_LAYERS - 1}.resid_post"

    def cfg(name: str) -> AgentConfig:
        return AgentConfig(
            name=name,
            role=name,
            transformer_cfg=TransformerConfig(
                vocab_size=tok.vocab_size,
                hidden_dim=HIDDEN,
                n_layers=N_LAYERS,
                n_heads=4,
                n_kv_heads=2,
                max_seq_len=1024,
            ),
            sae_cfg=SAEConfig(d_model=HIDDEN, n_features=64, k=4),
            introspection_sites=(site,),
        )

    proposer = Proposer(cfg("proposer"), tok, channel)
    critic = Critic(cfg("critic"), tok, channel)
    verifier = Verifier(cfg("verifier"), tok, channel)
    synth = Synthesizer(cfg("synth"), tok, channel)
    meta = Metacognitor(
        cfg("meta"),
        tok,
        channel,
        modeled_agents=["proposer", "critic", "verifier", "synth"],
    )

    n_params = sum(
        sum(p.numel() for p in a.parameters())
        for a in (proposer, critic, verifier, synth, meta)
    )
    print(f"society total params: {n_params:,}")

    society = Society(
        proposer,
        critic,
        verifier,
        synth,
        meta,
        channel,
        DebateConfig(max_rounds=2, proposals_per_round=1, enable_introspection=False),
    )

    train = sample_dataset(n=24, seed=1, difficulty_range=(1, 2))
    eval_set = sample_dataset(n=12, seed=99, difficulty_range=(1, 2))
    print(f"train: {len(train)} samples | eval: {len(eval_set)} samples")
    print(f"example question: {train[0].question[:90]}")
    print(f"   gold: {train[0].gold_answer}")
    print()

    cfg_train = TrainConfig(
        bc_epochs=3,
        sae_steps=120,
        rl_episodes=40,
        log_every=10,
    )
    train_all(society, train, cfg_train)

    print()
    print(evaluate_society(society, eval_set, cf_sample_rate=0.5).pretty())

    # Pick one eval sample for a deep-dive debate.
    print()
    print("=" * 64)
    print(" One debate in detail ")
    print("=" * 64)
    s = eval_set[0]
    result = society.debate(s.question, verification_fn=s.verify)
    print(result.render())

    # Introspect on the proposer's view of this question.
    print()
    print("=" * 64)
    print(" Proposer introspection on the same question ")
    print("=" * 64)
    ids = proposer.encode(s.question)
    report = proposer.introspect(ids, compute_counterfactual=True, top_n=5)
    print(report.summary)


if __name__ == "__main__":
    main()
