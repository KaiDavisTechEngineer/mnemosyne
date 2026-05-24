"""Causal introspection on a single agent.

Demonstrates the core MNEMOSYNE invention without any training overhead:

  1. Build one agent with a hooked transformer + sparse autoencoder.
  2. Run a forward pass on a question.
  3. Ask: "which features caused this answer?" (feature attribution)
  4. Ask: "what is the minimum feature ablation that would have flipped
     my answer?" (counterfactual)
  5. Visualize the feature names that the agent cares about.

Because the model is randomly initialized, the answers themselves are
nonsense — but the *introspection mechanism* still works correctly,
which is what this example illustrates.
"""
from __future__ import annotations

import torch

from mnemosyne import (
    Agent, AgentConfig, CommunicationChannel, SAEConfig,
    Tokenizer, TransformerConfig,
)


def main() -> None:
    torch.manual_seed(0)
    tok = Tokenizer.build()
    channel = CommunicationChannel()

    agent = Agent(
        AgentConfig(
            name="solo",
            role="proposer",
            transformer_cfg=TransformerConfig(
                vocab_size=tok.vocab_size, hidden_dim=64, n_layers=3,
                n_heads=4, n_kv_heads=2, max_seq_len=1024,
            ),
            sae_cfg=SAEConfig(d_model=64, n_features=128, k=6),
            introspection_sites=("block_1.resid_post", "block_2.resid_post"),
        ),
        tok, channel,
    )

    n_params = sum(p.numel() for p in agent.parameters())
    print(f"agent params: {n_params:,}")
    print(f"introspectable sites: {[agent._from_safe_key(k) for k in agent.saes]}")
    print()

    question = "alice is taller than bob; bob is taller than carol. who is tallest?"
    ids = agent.encode(question)
    print(f"question: {question}")
    print(f"encoded: {ids.shape[1]} tokens")
    print()

    # Run introspection at the deepest site (block_2.resid_post).
    report = agent.introspect(
        ids, site="block_2.resid_post",
        top_n=8, compute_counterfactual=True,
    )

    print(report.summary)
    print()

    # Show the agent's *vocabulary* for its own features. With 128
    # features, the proposer can refer to feature N via the
    # <feature:N> special token. Here are the top 8 it currently uses
    # to answer this question.
    print("agent's named features causing this answer:")
    for f in report.top_features:
        token_name = tok.feature_token(f.feature_idx)
        print(f"  {token_name:<16s}  act={f.activation:+.3f}  Δlogit={f.delta_logit:+.3f}")

    if report.counterfactual:
        cf = report.counterfactual
        print()
        print(f"counterfactual found:")
        print(f"  ablating {[tok.feature_token(f) for f in cf.ablated_features]}")
        print(f"  would have flipped my answer from token {cf.original_token} "
              f"to token {cf.counterfactual_token}")
        print(f"  confidence dropped by {cf.confidence_drop:.3f}")


if __name__ == "__main__":
    main()
