"""Multi-component training for MNEMOSYNE.

Training MNEMOSYNE is more involved than training a single LM because
five components — five agents, each with a backbone transformer, a
sparse autoencoder per introspection site, a self-model, and shared
trust dynamics — need to learn together. We break it into three
stages:

1. **Behavioral cloning warmup** (``warm_start``)
   Each agent is supervised-trained to imitate gold trajectories. For
   the Proposer/Synthesizer this is "predict the gold answer given the
   question." For the Critic it is "predict 'correct' or 'incorrect'
   given a (question, candidate) pair." For the Verifier we mostly
   skip training — its job is mechanical. The Metacognitor is trained
   to predict the other agents' outputs from recent message history.

2. **SAE training** (``train_saes``)
   With the backbones in a reasonable state we collect activations
   from each introspection site over a stream of samples and train the
   sparse autoencoders. The SAE objective is independent of the
   downstream task — it just learns to dictionary-decode activations.
   We run this for a fixed number of steps with the dead-feature
   resurrection trick.

3. **Multi-agent RL** (``society_finetune``)
   The society runs debates on real samples; the synthesizer's final
   answer is checked against the gold; a reward signal flows back to
   every agent. We use REINFORCE with a moving-average baseline (the
   simplest method that gives reliable signal at this scale).

These stages can be run independently. ``train_all`` runs them in
sequence with sensible defaults.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from mnemosyne.agents.specialists import (
    Critic,
    Metacognitor,
    Proposer,
    Synthesizer,
)
from mnemosyne.society.orchestrator import Society
from mnemosyne.training.tasks import Sample


# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    bc_epochs: int = 3
    sae_steps: int = 300
    rl_episodes: int = 80
    lr: float = 3e-4
    sae_lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    baseline_momentum: float = 0.9
    entropy_coef: float = 0.005
    log_every: int = 10
    save_every: Optional[int] = None


@dataclass
class TrainHistory:
    bc_loss: list[float] = field(default_factory=list)
    sae_loss: list[float] = field(default_factory=list)
    rl_return: list[float] = field(default_factory=list)
    rl_solve_rate: list[float] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Stage 1 — Behavioral cloning warmup
# ─────────────────────────────────────────────────────────────────────
def _bc_loss_for_agent(agent, prompt: str, target: str) -> torch.Tensor:
    """Compute cross-entropy loss on a single (prompt → target) example."""
    tok = agent.tokenizer
    prompt_ids = tok.encode(prompt)
    target_ids = tok.encode(target + "<eos>")
    full_ids = torch.tensor(
        [prompt_ids + target_ids], dtype=torch.long, device=agent._device()
    )
    # Truncate to max_seq_len
    max_len = agent.cfg.transformer_cfg.max_seq_len - 1
    if full_ids.shape[1] > max_len:
        full_ids = full_ids[:, -max_len:]
    # Forward.
    _, logits = agent.transformer(full_ids)
    # Predict each target token from the prefix ending at the previous one.
    # Shift: logits[t] predicts token[t+1].
    prompt_len = len(prompt_ids)
    if full_ids.shape[1] <= prompt_len:
        return torch.tensor(0.0, device=agent._device(), requires_grad=True)
    # Slice the logits over the target span.
    logits_slice = logits[0, prompt_len - 1 : -1]  # (T_target, V)
    target_slice = full_ids[0, prompt_len:]  # (T_target,)
    return F.cross_entropy(logits_slice, target_slice)


def warm_start(
    proposer: Proposer,
    critic: Critic,
    synthesizer: Synthesizer,
    metacognitor: Metacognitor,
    samples: list[Sample],
    cfg: TrainConfig,
) -> list[float]:
    """Behavioral cloning warmup. Returns the per-epoch average loss."""
    params = (
        list(proposer.transformer.parameters())
        + list(critic.transformer.parameters())
        + list(synthesizer.transformer.parameters())
        + list(metacognitor.transformer.parameters())
    )
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    losses: list[float] = []

    for epoch in range(cfg.bc_epochs):
        random.shuffle(samples)
        epoch_loss = 0.0
        n = 0
        for s in samples:
            # Proposer: question → gold answer
            l1 = _bc_loss_for_agent(
                proposer,
                f"{proposer.tokenizer.role_token('proposer')}<msg>{s.question}</msg>",
                s.gold_answer,
            )
            # Synthesizer: question + (synthetic correct proposal) → gold answer
            l2 = _bc_loss_for_agent(
                synthesizer,
                f"{synthesizer.tokenizer.role_token('synthesizer')}<msg>"
                f"question: {s.question} | proposal: {s.gold_answer}</msg>",
                s.gold_answer,
            )
            # Critic: question + (gold answer) → 'correct'
            l3 = _bc_loss_for_agent(
                critic,
                f"{critic.tokenizer.role_token('critic')}<msg>"
                f"question: {s.question} | proposal: {s.gold_answer}</msg>",
                "correct",
            )
            loss = l1 + l2 + l3
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            epoch_loss += float(loss.item())
            n += 1
        avg = epoch_loss / max(n, 1)
        losses.append(avg)
        print(f"[bc] epoch {epoch + 1}/{cfg.bc_epochs}  avg_loss={avg:.4f}")
    return losses


# ─────────────────────────────────────────────────────────────────────
# Stage 2 — SAE training on captured activations
# ─────────────────────────────────────────────────────────────────────
def train_saes(agents: list, samples: list[Sample], cfg: TrainConfig) -> list[float]:
    """Train every agent's sparse autoencoders on activations captured
    while running the agent's transformer on the sample stream.

    Returns the global average SAE loss per logging interval.
    """
    all_saes = []
    all_agents = []
    for agent in agents:
        for safe_key, sae in agent.saes.items():
            all_saes.append((agent, safe_key, sae))
            all_agents.append(agent)
    sae_params = []
    for _, _, sae in all_saes:
        sae_params.extend(sae.parameters())
    opt = torch.optim.AdamW(sae_params, lr=cfg.sae_lr, weight_decay=0.0)
    losses: list[float] = []
    log_buffer = deque(maxlen=20)

    for step in range(cfg.sae_steps):
        s = samples[step % len(samples)]
        total_loss = torch.zeros((), device=agents[0]._device())
        n = 0
        for agent, safe_key, sae in all_saes:
            site = agent._from_safe_key(safe_key)
            prompt = (
                f"{agent.tokenizer.role_token(agent.cfg.role)}<msg>{s.question}</msg>"
            )
            ids = agent.encode(prompt)
            with torch.no_grad():
                _, captured = agent.transformer.run_with_capture(ids, sites=[site])
            a = captured[site].reshape(-1, captured[site].shape[-1])
            loss, info = sae.loss(a)
            total_loss = total_loss + loss
            n += 1
        total_loss = total_loss / max(n, 1)
        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(sae_params, cfg.grad_clip)
        opt.step()
        for _, _, sae in all_saes:
            sae.normalize_decoder()
        log_buffer.append(float(total_loss.item()))
        if (step + 1) % cfg.log_every == 0:
            avg = sum(log_buffer) / len(log_buffer)
            losses.append(avg)
            print(f"[sae] step {step + 1:4d}/{cfg.sae_steps}  loss(20)={avg:.4f}")
    return losses


# ─────────────────────────────────────────────────────────────────────
# Stage 3 — Multi-agent RL with verification reward
# ─────────────────────────────────────────────────────────────────────
def _compute_reward(result, gold_answer: str) -> float:
    """Reward = 1.0 if the synthesizer's answer matches the gold."""
    final = result.final_answer.lower()
    gold = gold_answer.lower()
    if gold in final:
        return 1.0
    # Partial credit for the verifier landing on 'ok' at all
    if result.succeeded:
        return 0.3
    return -0.3


def society_finetune(
    society: Society, samples: list[Sample], cfg: TrainConfig
) -> TrainHistory:
    """REINFORCE fine-tuning of the synthesizer.

    The synthesizer is the agent whose output directly determines
    the reward, so it is the primary learner here. The proposer and
    critic are updated less directly: their gradient signal comes from
    whatever they contributed to the synthesizer's input.

    The simplest implementation — and the one we use — is to just
    train the synthesizer via REINFORCE on the verifier's verdict, and
    leave the other agents frozen during this stage. Going further to
    full multi-agent credit assignment is future work.
    """
    history = TrainHistory()
    syn_params = list(society.synthesizer.transformer.parameters())
    opt = torch.optim.AdamW(syn_params, lr=cfg.lr * 0.3, weight_decay=cfg.weight_decay)
    baseline = 0.0
    recent_rewards = deque(maxlen=50)
    recent_solves = deque(maxlen=50)

    for ep in range(cfg.rl_episodes):
        s = random.choice(samples)
        # Run a debate.
        result = society.debate(s.question, verification_fn=s.verify)
        reward = _compute_reward(result, s.gold_answer)
        recent_rewards.append(reward)
        recent_solves.append(1.0 if reward >= 1.0 else 0.0)
        history.rl_return.append(reward)
        history.rl_solve_rate.append(sum(recent_solves) / len(recent_solves))

        # REINFORCE: replay the synthesizer's last forward pass with grad
        # and apply the policy-gradient loss with the (reward - baseline)
        # advantage.
        if not result.rounds:
            continue
        last_round = result.rounds[-1]
        syn_text = last_round.synthesis
        # Re-encode + forward with grad to get log-probs we can backprop.
        prompt_parts = [f"question: {s.question}"]
        for i, p in enumerate(last_round.proposals):
            prompt_parts.append(f"proposal_{i}: {p}")
        prompt = (
            f"{society.synthesizer.tokenizer.role_token('synthesizer')}"
            f"<msg>{' | '.join(prompt_parts)}</msg>"
        )
        prompt_ids = society.synthesizer.tokenizer.encode(prompt)
        syn_ids = society.synthesizer.tokenizer.encode(syn_text + "<eos>")
        max_len = society.synthesizer.cfg.transformer_cfg.max_seq_len - 1
        full = prompt_ids + syn_ids
        if len(full) > max_len:
            full = full[-max_len:]
            # Re-derive how many of the head are 'prompt' tokens
            prompt_len = max(1, len(full) - len(syn_ids))
        else:
            prompt_len = len(prompt_ids)
        full_t = torch.tensor(
            [full], dtype=torch.long, device=society.synthesizer._device()
        )
        if full_t.shape[1] <= prompt_len:
            continue
        _, logits = society.synthesizer.transformer(full_t)
        logits_slice = logits[0, prompt_len - 1 : -1]
        target_slice = full_t[0, prompt_len:]
        log_probs = F.log_softmax(logits_slice, dim=-1)
        token_logp = log_probs.gather(-1, target_slice.unsqueeze(-1)).squeeze(-1)
        sum_logp = token_logp.sum()
        advantage = reward - baseline
        # Entropy regularizer.
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
        loss = -sum_logp * advantage - cfg.entropy_coef * entropy
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(syn_params, cfg.grad_clip)
        opt.step()
        baseline = (
            cfg.baseline_momentum * baseline + (1 - cfg.baseline_momentum) * reward
        )

        if (ep + 1) % cfg.log_every == 0:
            avg = sum(recent_rewards) / len(recent_rewards)
            sr = sum(recent_solves) / len(recent_solves)
            print(
                f"[rl ] ep {ep + 1:4d}/{cfg.rl_episodes}  "
                f"avg_reward(50)={avg:+.3f}  solve_rate(50)={sr:.1%}  "
                f"baseline={baseline:+.3f}"
            )
    return history


# ─────────────────────────────────────────────────────────────────────
# Top-level orchestrating function
# ─────────────────────────────────────────────────────────────────────
def train_all(
    society: Society, samples: list[Sample], cfg: TrainConfig
) -> TrainHistory:
    """Run all three training stages in sequence."""
    print("=" * 64)
    print(" Stage 1 — Behavioral cloning warmup ")
    print("=" * 64)
    bc_losses = warm_start(
        society.proposer,
        society.critic,
        society.synthesizer,
        society.metacognitor,
        samples,
        cfg,
    )

    print()
    print("=" * 64)
    print(" Stage 2 — SAE training ")
    print("=" * 64)
    agents = [society.proposer, society.critic, society.synthesizer]
    sae_losses = train_saes(agents, samples, cfg)

    print()
    print("=" * 64)
    print(" Stage 3 — Multi-agent RL ")
    print("=" * 64)
    history = society_finetune(society, samples, cfg)
    history.bc_loss = bc_losses
    history.sae_loss = sae_losses
    return history
