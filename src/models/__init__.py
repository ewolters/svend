"""
Custom transformer models for reasoning tasks.

This package provides:
- TransformerConfig: Configuration dataclass for model architecture
- ReasoningTransformer: Core transformer model
- Pre-defined configs for 500M, 1B, 3B, 7B, 13B scales
- Knowledge distillation utilities
"""

from .config import (
    TransformerConfig,
    create_500m_config,
    create_1b_config,
    create_3b_config,
    create_7b_config,
    create_13b_config,
    get_config,
    MODEL_CONFIGS,
    # Ensemble specialists
    create_router_config,
    create_language_specialist_config,
    create_reasoning_specialist_config,
    create_verifier_specialist_config,
    EnsembleConfig,
    create_ensemble_config,
)

from .transformer import (
    ReasoningTransformer,
    ReasoningTransformerForCausalLM,
    create_model,
)

from .layers import (
    RMSNorm,
    RotaryEmbedding,
    GroupedQueryAttention,
    SwiGLU,
    TransformerBlock,
)

__all__ = [
    # Config
    "TransformerConfig",
    "create_500m_config",
    "create_1b_config",
    "create_3b_config",
    "create_7b_config",
    "create_13b_config",
    "get_config",
    "MODEL_CONFIGS",
    # Ensemble specialists
    "create_router_config",
    "create_language_specialist_config",
    "create_reasoning_specialist_config",
    "create_verifier_specialist_config",
    "EnsembleConfig",
    "create_ensemble_config",
    # Models
    "ReasoningTransformer",
    "ReasoningTransformerForCausalLM",
    "create_model",
    # Layers
    "RMSNorm",
    "RotaryEmbedding",
    "GroupedQueryAttention",
    "SwiGLU",
    "TransformerBlock",
]
