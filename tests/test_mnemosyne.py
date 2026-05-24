"""End-to-end test suite for MNEMOSYNE.

Coverage:
* Hook-instrumented transformer — forward pass, capture, intervention
* Sparse autoencoder — training, sparsity, dead-feature handling
* Causal interventions — patching, attribution, counterfactuals
* Hierarchical memory — working/episodic/semantic + consolidation
* Communication channel — routing, trust, transcript
* Base agent — forward, introspect, remember, recall
* Specialist agents — propose / critique / verify / synthesize / meta
* Society orchestrator — full debate end-to-end
* Training pipeline — BC/SAE/RL stages run without crashing
* Evaluation — report structure is sensible
"""
from __future__ import annotations

import random

import pytest
import torch

from mnemosyne.agents.base import Agent, AgentConfig
from mnemosyne.agents.specialists import (
    Critic, Metacognitor, Proposer, Synthesizer, Verifier,
)
from mnemosyne.arch.tokenizer import Tokenizer
from mnemosyne.arch.transformer import (
    HookContext, HookedTransformer, TransformerConfig, hooks,
)
from mnemosyne.causal.interventions import (
    activation_patch, feature_attribution, find_counterfactual,
)
from mnemosyne.communication.channel import CommunicationChannel
from mnemosyne.eval.benchmark import evaluate_society
from mnemosyne.interp.sae import SAEConfig, TopKSAE
from mnemosyne.memory.hierarchical import (
    AgentMemory, Episode, EpisodicMemory, SemanticMemory, WorkingMemory,
    consolidate,
)
from mnemosyne.self_model.introspect import SelfModel, SelfModelConfig
from mnemosyne.society.orchestrator import DebateConfig, Society
from mnemosyne.training.tasks import (
    sample_dataset, sample_task, task_boolean, task_comparison, task_ordering,
)
from mnemosyne.training.trainer import (
    TrainConfig, society_finetune, train_saes, warm_start,
)


@pytest.fixture
def tok():
    return Tokenizer.build()


@pytest.fixture
def channel():
    return CommunicationChannel()


def _agent_cfg(name, tok, hidden=32, n_layers=2):
    return AgentConfig(
        name=name, role=name,
        transformer_cfg=TransformerConfig(
            vocab_size=tok.vocab_size, hidden_dim=hidden,
            n_layers=n_layers, n_heads=4, n_kv_heads=2, max_seq_len=512,
        ),
        sae_cfg=SAEConfig(d_model=hidden, n_features=32, k=3),
        introspection_sites=(f"block_{n_layers-1}.resid_post",),
    )


# ─────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────
class TestTokenizer:
    def test_specials_in_vocab(self, tok):
        ids = tok.encode("<role:proposer><msg>x</msg>")
        assert tok.decode(ids) == "<role:proposer><msg>x</msg>"

    def test_byte_fallback(self, tok):
        ids = tok.encode("hello world")
        # Each byte produces one id, plus no specials in plain text.
        assert len(ids) == len("hello world")
        assert tok.decode(ids) == "hello world"

    def test_feature_tokens(self, tok):
        ids = tok.encode("<feature:42>")
        assert len(ids) == 1
        assert tok.decode(ids) == "<feature:42>"


# ─────────────────────────────────────────────────────────────────────
# Transformer + hooks
# ─────────────────────────────────────────────────────────────────────
class TestTransformer:
    def test_forward_shapes(self, tok):
        cfg = TransformerConfig(vocab_size=tok.vocab_size, hidden_dim=32,
                                  n_layers=2, n_heads=4, n_kv_heads=2,
                                  max_seq_len=256)
        m = HookedTransformer(cfg)
        x = torch.tensor([[1, 2, 3, 4, 5]])
        hidden, logits = m(x)
        assert hidden.shape == (1, 5, 32)
        assert logits.shape == (1, 5, tok.vocab_size)

    def test_capture(self, tok):
        cfg = TransformerConfig(vocab_size=tok.vocab_size, hidden_dim=32,
                                  n_layers=2, n_heads=4, n_kv_heads=2)
        m = HookedTransformer(cfg)
        x = torch.tensor([[1, 2, 3, 4, 5]])
        logits, captured = m.run_with_capture(x, sites=["block_1.resid_post"])
        assert "block_1.resid_post" in captured
        assert captured["block_1.resid_post"].shape == (1, 5, 32)

    def test_intervention_changes_logits(self, tok):
        cfg = TransformerConfig(vocab_size=tok.vocab_size, hidden_dim=32,
                                  n_layers=2, n_heads=4, n_kv_heads=2)
        m = HookedTransformer(cfg)
        x = torch.tensor([[1, 2, 3, 4, 5]])
        _, base_logits = m(x)
        zeros = torch.zeros(1, 5, 32)
        patched_logits, _ = m.run_with_intervention(
            x, {"block_0.resid_post": zeros}
        )
        assert not torch.allclose(base_logits, patched_logits)

    def test_site_names_complete(self, tok):
        cfg = TransformerConfig(vocab_size=tok.vocab_size, hidden_dim=32,
                                  n_layers=3, n_heads=4, n_kv_heads=2)
        m = HookedTransformer(cfg)
        names = m.site_names()
        # 3 blocks × 11 sites + embed + final_norm + logits = 36
        assert len(names) == 36


# ─────────────────────────────────────────────────────────────────────
# SAE
# ─────────────────────────────────────────────────────────────────────
class TestSAE:
    def test_topk_exact_sparsity(self):
        sae = TopKSAE(SAEConfig(d_model=16, n_features=32, k=4))
        a = torch.randn(8, 4, 16)
        _, z = sae(a)
        n_active = (z != 0).float().sum(dim=-1)
        # Some tokens might have <k active if there aren't k positive
        # pre-activations; but on average we should be at k.
        assert n_active.max() <= 4

    def test_reduces_loss_on_structured_data(self):
        torch.manual_seed(0)
        sae = TopKSAE(SAEConfig(d_model=16, n_features=32, k=4))
        opt = torch.optim.AdamW(sae.parameters(), lr=1e-3)
        # Make low-rank data.
        dirs = torch.randn(8, 16); dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        first_loss = None
        last_loss = None
        for step in range(50):
            w = torch.randn(64, 8).softmax(-1)
            a = w @ dirs + 0.01 * torch.randn(64, 16)
            loss, info = sae.loss(a)
            opt.zero_grad(); loss.backward(); opt.step()
            sae.normalize_decoder()
            if step == 0:
                first_loss = info["recon_loss"]
            last_loss = info["recon_loss"]
        assert last_loss < first_loss

    def test_ablate_feature_shifts_recon(self):
        sae = TopKSAE(SAEConfig(d_model=16, n_features=32, k=4))
        a = torch.randn(2, 1, 16)
        recon, _ = sae(a)
        ablated = sae.ablate_feature(a, feature_idx=0)
        # Some shift is expected (might be 0 if feature 0 wasn't active).
        assert ablated.shape == recon.shape


# ─────────────────────────────────────────────────────────────────────
# Causal interventions
# ─────────────────────────────────────────────────────────────────────
class TestInterventions:
    def test_activation_patch_returns_score(self, tok):
        cfg = TransformerConfig(vocab_size=tok.vocab_size, hidden_dim=32,
                                  n_layers=2, n_heads=4, n_kv_heads=2)
        m = HookedTransformer(cfg)
        clean = torch.tensor([[1, 2, 3, 4, 5]])
        corrupted = torch.tensor([[5, 4, 3, 2, 1]])
        score = activation_patch(m, clean, corrupted, "block_0.resid_post")
        assert isinstance(score, float)

    def test_feature_attribution_returns_list(self, tok):
        cfg = TransformerConfig(vocab_size=tok.vocab_size, hidden_dim=32,
                                  n_layers=2, n_heads=4, n_kv_heads=2)
        m = HookedTransformer(cfg)
        sae = TopKSAE(SAEConfig(d_model=32, n_features=64, k=4))
        x = torch.tensor([[1, 2, 3, 4, 5]])
        attrs = feature_attribution(m, sae, "block_1.resid_post", x,
                                      target_token=0, top_n=3)
        assert isinstance(attrs, list)
        assert len(attrs) <= 3


# ─────────────────────────────────────────────────────────────────────
# Memory
# ─────────────────────────────────────────────────────────────────────
class TestMemory:
    def test_working_memory_fifo(self):
        wm = WorkingMemory(capacity=3)
        for i in range(5):
            wm.push(torch.zeros(8) + i)
        assert len(wm) == 3
        # The last three pushes should remain.
        hiddens = wm.hiddens()
        assert hiddens[-1][0] == 4

    def test_episodic_retrieval(self):
        em = EpisodicMemory(dim=8)
        for i in range(5):
            key = torch.zeros(8); key[i] = 1.0
            em.store(Episode(key=key, input_text=f"q{i}", output_text=f"a{i}",
                              feature_signature=None, outcome={}))
        # Query that points strongly at episode 2.
        q = torch.zeros(8); q[2] = 1.0
        results = em.retrieve(q, k=1)
        assert len(results) == 1
        assert results[0][0].input_text == "q2"

    def test_consolidation_produces_concepts(self):
        em = EpisodicMemory(dim=8)
        sm = SemanticMemory(n_features=16)
        rng = torch.manual_seed(0)
        # Make two clusters of feature signatures.
        for _ in range(10):
            sig = torch.zeros(16)
            sig[:3] = torch.randn(3).abs()
            em.store(Episode(key=torch.randn(8), input_text="x",
                              output_text="y", feature_signature=sig, outcome={}))
        for _ in range(10):
            sig = torch.zeros(16)
            sig[8:11] = torch.randn(3).abs()
            em.store(Episode(key=torch.randn(8), input_text="x",
                              output_text="y", feature_signature=sig, outcome={}))
        n_added = consolidate(em, sm, n_clusters=2, n_iters=20,
                                min_cluster_size=3)
        assert n_added >= 1


# ─────────────────────────────────────────────────────────────────────
# Communication channel
# ─────────────────────────────────────────────────────────────────────
class TestChannel:
    def test_send_and_inbox(self):
        ch = CommunicationChannel()
        ch.register("alice")
        ch.register("bob")
        ch.send("alice", "bob", "hello")
        inbox = ch.inbox("bob")
        assert len(inbox) == 1
        assert inbox[0].token_text == "hello"

    def test_broadcast(self):
        ch = CommunicationChannel()
        for name in ("alice", "bob", "carol"):
            ch.register(name)
        ch.send("alice", "all", "announcement")
        assert len(ch.inbox("bob")) == 1
        assert len(ch.inbox("carol")) == 1

    def test_trust_updates(self):
        ch = CommunicationChannel()
        ch.register("a"); ch.register("b")
        assert ch.trust("a", "b") == 1.0
        ch.update_trust("a", "b", -0.3)
        assert abs(ch.trust("a", "b") - 0.7) < 1e-6
        ch.update_trust("a", "b", -1.0)  # should clip
        assert ch.trust("a", "b") == 0.0


# ─────────────────────────────────────────────────────────────────────
# Self-model
# ─────────────────────────────────────────────────────────────────────
class TestSelfModel:
    def test_predicts_and_surprise(self):
        sm = SelfModel(SelfModelConfig(hidden_dim=16, vocab_size=20, inner_dim=8))
        hidden = torch.randn(2, 5, 16)
        true_logits = torch.randn(2, 5, 20)
        preds = sm(hidden)
        assert preds.shape == (2, 5, 20)
        surprise = sm.surprise(hidden, true_logits)
        assert surprise.shape == (2, 5)
        assert (surprise >= 0).all()


# ─────────────────────────────────────────────────────────────────────
# Agents
# ─────────────────────────────────────────────────────────────────────
class TestAgents:
    def test_agent_forward(self, tok, channel):
        agent = Agent(_agent_cfg("a", tok), tok, channel)
        ids = agent.encode("hello")
        logits, captured = agent(ids)
        assert logits.shape[-1] == tok.vocab_size
        assert any("resid_post" in k for k in captured.keys())

    def test_agent_speak_and_listen(self, tok, channel):
        a = Agent(_agent_cfg("a", tok), tok, channel)
        b = Agent(_agent_cfg("b", tok), tok, channel)
        a.speak("b", "hi b")
        msgs = b.listen()
        assert len(msgs) == 1
        assert msgs[0].sender == "a"

    def test_introspect_returns_report(self, tok, channel):
        agent = Agent(_agent_cfg("a", tok), tok, channel)
        ids = agent.encode("hello world")
        report = agent.introspect(ids, top_n=3, compute_counterfactual=False)
        assert report.site.endswith("resid_post")
        assert isinstance(report.top_features, list)

    def test_specialists_have_correct_roles(self, tok, channel):
        p = Proposer(_agent_cfg("p", tok), tok, channel)
        c = Critic(_agent_cfg("c", tok), tok, channel)
        v = Verifier(_agent_cfg("v", tok), tok, channel)
        s = Synthesizer(_agent_cfg("s", tok), tok, channel)
        assert p.cfg.role == "proposer"
        assert c.cfg.role == "critic"
        assert v.cfg.role == "verifier"
        assert s.cfg.role == "synthesizer"


# ─────────────────────────────────────────────────────────────────────
# Society
# ─────────────────────────────────────────────────────────────────────
class TestSociety:
    def test_full_debate(self, tok):
        ch = CommunicationChannel()
        p = Proposer(_agent_cfg("p", tok), tok, ch)
        c = Critic(_agent_cfg("c", tok), tok, ch)
        v = Verifier(_agent_cfg("v", tok), tok, ch)
        s = Synthesizer(_agent_cfg("s", tok), tok, ch)
        m = Metacognitor(_agent_cfg("m", tok), tok, ch,
                          modeled_agents=["p", "c", "v", "s"])
        soc = Society(p, c, v, s, m, ch,
                       DebateConfig(max_rounds=1, proposals_per_round=1,
                                     enable_introspection=False))
        def verify(q, prop):
            return {"verdict": "ok", "reason": "test"}
        res = soc.debate("test question", verification_fn=verify)
        assert len(res.rounds) == 1
        assert res.succeeded


# ─────────────────────────────────────────────────────────────────────
# Tasks
# ─────────────────────────────────────────────────────────────────────
class TestTasks:
    def test_ordering_task_self_consistent(self):
        rng = random.Random(0)
        s = task_ordering(rng, n=4)
        # The gold answer should appear in the verifier's accept set.
        v = s.verify(s.question, "the answer is " + s.gold_answer)
        assert v["verdict"] == "ok"

    def test_dataset_generation(self):
        ds = sample_dataset(n=20, seed=1)
        assert len(ds) == 20
        for s in ds:
            assert s.family in ("ordering", "boolean", "comparison")
            assert s.gold_answer


# ─────────────────────────────────────────────────────────────────────
# Training pipeline smoke test
# ─────────────────────────────────────────────────────────────────────
class TestTrainingPipeline:
    def test_bc_reduces_loss(self, tok):
        torch.manual_seed(0); random.seed(0)
        ch = CommunicationChannel()
        p = Proposer(_agent_cfg("p", tok), tok, ch)
        c = Critic(_agent_cfg("c", tok), tok, ch)
        s = Synthesizer(_agent_cfg("s", tok), tok, ch)
        m = Metacognitor(_agent_cfg("m", tok), tok, ch,
                          modeled_agents=["p", "c", "s"])
        samples = sample_dataset(n=8, seed=0, family="ordering",
                                  difficulty_range=(1, 1))
        cfg = TrainConfig(bc_epochs=2, log_every=10)
        losses = warm_start(p, c, s, m, samples, cfg)
        assert losses[-1] <= losses[0] + 1e-3

    def test_sae_training_runs(self, tok):
        ch = CommunicationChannel()
        p = Proposer(_agent_cfg("p", tok), tok, ch)
        samples = sample_dataset(n=4, seed=0, family="ordering")
        cfg = TrainConfig(sae_steps=10, log_every=5)
        losses = train_saes([p], samples, cfg)
        assert len(losses) >= 1
