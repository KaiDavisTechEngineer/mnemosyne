"""``mnemosyne`` command-line interface.

Subcommands::

    mnemosyne train        # build a society from scratch and train it
    mnemosyne eval         # evaluate a trained society on held-out samples
    mnemosyne debate       # run one debate on a hand-written question
    mnemosyne introspect   # ask a trained proposer 'why did you say that?'
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

from mnemosyne.agents.base import AgentConfig
from mnemosyne.agents.specialists import (
    Critic,
    Metacognitor,
    Proposer,
    Synthesizer,
    Verifier,
)
from mnemosyne.arch.tokenizer import Tokenizer
from mnemosyne.arch.transformer import TransformerConfig
from mnemosyne.communication.channel import CommunicationChannel
from mnemosyne.eval.benchmark import evaluate_society
from mnemosyne.interp.sae import SAEConfig
from mnemosyne.society.orchestrator import DebateConfig, Society
from mnemosyne.training.tasks import sample_dataset
from mnemosyne.training.trainer import TrainConfig, train_all


def _build_society(
    tok: Tokenizer,
    hidden_dim: int = 48,
    n_layers: int = 2,
    n_features: int = 64,
    k: int = 4,
) -> Society:
    channel = CommunicationChannel()

    def cfg(name: str) -> AgentConfig:
        return AgentConfig(
            name=name,
            role=name,
            transformer_cfg=TransformerConfig(
                vocab_size=tok.vocab_size,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                n_heads=4,
                n_kv_heads=2,
                max_seq_len=1024,
            ),
            sae_cfg=SAEConfig(d_model=hidden_dim, n_features=n_features, k=k),
            introspection_sites=(f"block_{n_layers - 1}.resid_post",),
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
    return Society(
        proposer,
        critic,
        verifier,
        synth,
        meta,
        channel,
        cfg=DebateConfig(
            max_rounds=2, proposals_per_round=1, enable_introspection=False
        ),
    )


def _save_society(society: Society, path: Path) -> None:
    state = {
        "proposer": society.proposer.state_dict(),
        "critic": society.critic.state_dict(),
        "verifier": society.verifier.state_dict(),
        "synth": society.synthesizer.state_dict(),
        "meta": society.metacognitor.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _load_society(society: Society, path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    society.proposer.load_state_dict(state["proposer"])
    society.critic.load_state_dict(state["critic"])
    society.verifier.load_state_dict(state["verifier"])
    society.synthesizer.load_state_dict(state["synth"])
    society.metacognitor.load_state_dict(state["meta"])


def cmd_train(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    tok = Tokenizer.build()
    society = _build_society(tok, hidden_dim=args.hidden_dim, n_layers=args.n_layers)
    n_params = sum(
        sum(p.numel() for p in a.parameters())
        for a in (
            society.proposer,
            society.critic,
            society.verifier,
            society.synthesizer,
            society.metacognitor,
        )
    )
    print(f"society params: {n_params:,}")
    train = sample_dataset(
        n=args.n_samples, seed=args.seed, difficulty_range=(1, args.max_difficulty)
    )
    config = TrainConfig(
        bc_epochs=args.bc_epochs,
        sae_steps=args.sae_steps,
        rl_episodes=args.rl_episodes,
        log_every=args.log_every,
    )
    history = train_all(society, train, config)
    if args.out:
        _save_society(society, Path(args.out))
        print(f"\nsaved society to {args.out}")
    if args.eval_after:
        eval_set = sample_dataset(
            n=args.eval_n,
            seed=args.seed + 9999,
            difficulty_range=(1, args.max_difficulty),
        )
        print()
        print(evaluate_society(society, eval_set).pretty())
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    tok = Tokenizer.build()
    society = _build_society(tok, hidden_dim=args.hidden_dim, n_layers=args.n_layers)
    _load_society(society, Path(args.model))
    eval_set = sample_dataset(
        n=args.n, seed=args.seed, difficulty_range=(1, args.max_difficulty)
    )
    print(evaluate_society(society, eval_set).pretty())
    return 0


def cmd_debate(args: argparse.Namespace) -> int:
    tok = Tokenizer.build()
    society = _build_society(tok, hidden_dim=args.hidden_dim, n_layers=args.n_layers)
    if args.model:
        _load_society(society, Path(args.model))

    def verify_fn(q, p):
        if args.expected and args.expected.lower() in p.lower():
            return {"verdict": "ok", "reason": "matched expected"}
        return {"verdict": "unknown", "reason": "no oracle"}

    result = society.debate(args.question, verification_fn=verify_fn)
    print(result.render())
    return 0


def cmd_introspect(args: argparse.Namespace) -> int:
    tok = Tokenizer.build()
    society = _build_society(tok, hidden_dim=args.hidden_dim, n_layers=args.n_layers)
    if args.model:
        _load_society(society, Path(args.model))

    ids = society.proposer.encode(args.text)
    report = society.proposer.introspect(
        ids, compute_counterfactual=True, top_n=args.top_n
    )
    print(report.summary)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="mnemosyne",
        description="A society of causally-self-modeling agents",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = lambda p: (
        p.add_argument("--hidden-dim", type=int, default=48),
        p.add_argument("--n-layers", type=int, default=2),
    )

    t = sub.add_parser("train", help="train a new society from scratch")
    common(t)
    t.add_argument("--n-samples", type=int, default=64)
    t.add_argument("--bc-epochs", type=int, default=3)
    t.add_argument("--sae-steps", type=int, default=200)
    t.add_argument("--rl-episodes", type=int, default=80)
    t.add_argument("--max-difficulty", type=int, default=3)
    t.add_argument("--log-every", type=int, default=10)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", type=str, default="mnemosyne.pt")
    t.add_argument("--eval-after", action="store_true")
    t.add_argument("--eval-n", type=int, default=16)

    e = sub.add_parser("eval", help="evaluate a trained society")
    common(e)
    e.add_argument("--model", type=str, required=True)
    e.add_argument("--n", type=int, default=32)
    e.add_argument("--max-difficulty", type=int, default=3)
    e.add_argument("--seed", type=int, default=12345)

    d = sub.add_parser("debate", help="run a single debate")
    common(d)
    d.add_argument("--model", type=str, default=None)
    d.add_argument("--question", type=str, required=True)
    d.add_argument(
        "--expected",
        type=str,
        default=None,
        help="optional expected answer to ground the verifier",
    )

    i = sub.add_parser("introspect", help="show why the proposer said what it said")
    common(i)
    i.add_argument("--model", type=str, default=None)
    i.add_argument("--text", type=str, required=True)
    i.add_argument("--top-n", type=int, default=6)

    args = ap.parse_args(argv)
    return {
        "train": cmd_train,
        "eval": cmd_eval,
        "debate": cmd_debate,
        "introspect": cmd_introspect,
    }[args.cmd](args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
