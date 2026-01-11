"""
Training infrastructure for reasoning models.
"""

from .trainer import (
    TrainingConfig,
    Trainer,
)

from .distillation import (
    DistillationConfig,
    DistillationLoss,
    DistillationTrainer,
    HiddenStateProjector,
    AttentionTransfer,
    distill_reasoning_model,
)

__all__ = [
    # Training
    "TrainingConfig",
    "Trainer",
    # Distillation
    "DistillationConfig",
    "DistillationLoss",
    "DistillationTrainer",
    "HiddenStateProjector",
    "AttentionTransfer",
    "distill_reasoning_model",
]
