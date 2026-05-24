"""MNEMOSYNE — A society of causally-self-modeling agents.

Top-level public API. Importing this module pulls in the foundational
abstractions; submodules are imported lazily where heavy.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Architecture
from mnemosyne.arch.tokenizer import Tokenizer
from mnemosyne.arch.transformer import (
    HookContext,
    HookedTransformer,
    TransformerConfig,
    hooks,
)

# Interpretability
from mnemosyne.interp.sae import SAEConfig, TopKSAE

# Causal interventions
from mnemosyne.causal.interventions import (
    Counterfactual,
    FeatureAttribution,
    activation_patch,
    attribution_patch,
    feature_attribution,
    find_counterfactual,
)

# Memory
from mnemosyne.memory.hierarchical import (
    AgentMemory,
    Concept,
    Episode,
    EpisodicMemory,
    SemanticMemory,
    WorkingMemory,
    consolidate,
)

# Communication
from mnemosyne.communication.channel import (
    BridgeHead,
    CommunicationChannel,
    Message,
)

# Self-model
from mnemosyne.self_model.introspect import (
    SelfModel,
    SelfModelConfig,
    attach_self_model,
)

# Agents
from mnemosyne.agents.base import Agent, AgentConfig, IntrospectionReport
from mnemosyne.agents.specialists import (
    AgentModel,
    Critic,
    Metacognitor,
    Proposer,
    Synthesizer,
    Verifier,
)

# Society
from mnemosyne.society.orchestrator import (
    DebateConfig,
    DebateResult,
    RoundOutcome,
    Society,
)

# Training
from mnemosyne.training.tasks import (
    Sample,
    sample_dataset,
    sample_task,
    task_boolean,
    task_comparison,
    task_ordering,
)
from mnemosyne.training.trainer import (
    TrainConfig,
    TrainHistory,
    society_finetune,
    train_all,
    train_saes,
    warm_start,
)

# Evaluation
from mnemosyne.eval.benchmark import EvalReport, evaluate_society

__all__ = [
    "__version__",
    # arch
    "Tokenizer",
    "HookContext",
    "HookedTransformer",
    "TransformerConfig",
    "hooks",
    # interp
    "SAEConfig",
    "TopKSAE",
    # causal
    "Counterfactual",
    "FeatureAttribution",
    "activation_patch",
    "attribution_patch",
    "feature_attribution",
    "find_counterfactual",
    # memory
    "AgentMemory",
    "Concept",
    "Episode",
    "EpisodicMemory",
    "SemanticMemory",
    "WorkingMemory",
    "consolidate",
    # communication
    "BridgeHead",
    "CommunicationChannel",
    "Message",
    # self-model
    "SelfModel",
    "SelfModelConfig",
    "attach_self_model",
    # agents
    "Agent",
    "AgentConfig",
    "IntrospectionReport",
    "AgentModel",
    "Critic",
    "Metacognitor",
    "Proposer",
    "Synthesizer",
    "Verifier",
    # society
    "DebateConfig",
    "DebateResult",
    "RoundOutcome",
    "Society",
    # training
    "Sample",
    "sample_dataset",
    "sample_task",
    "task_boolean",
    "task_comparison",
    "task_ordering",
    "TrainConfig",
    "TrainHistory",
    "society_finetune",
    "train_all",
    "train_saes",
    "warm_start",
    # evaluation
    "EvalReport",
    "evaluate_society",
]
