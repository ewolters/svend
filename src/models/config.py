"""
Model configuration system for Svend reasoning models.

Optimized for tool-augmented reasoning with:
- Larger context windows (8K-32K)
- Tool calling support
- Verification capabilities
- Server deployment

svend.ai
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
import json
from pathlib import Path


@dataclass
class TransformerConfig:
    """
    Configuration for Svend transformer models.

    Architecture choices optimized for reasoning:
    - RoPE positional embeddings (better length generalization)
    - SwiGLU activation (improved gradient flow)
    - RMSNorm (faster, stable training)
    - Grouped Query Attention (memory efficient)
    - Pre-norm architecture (stable deep networks)
    - Tool calling tokens for external system integration
    """

    # Model identity
    name: str = "svend-reasoner"
    version: str = "0.1.0"
    model_type: Literal["reasoner", "verifier", "router", "language", "reasoning"] = "reasoner"

    # Core dimensions
    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 5504
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4  # GQA

    # Sequence and embeddings - LARGER for reasoning chains
    max_position_embeddings: int = 8192  # 8K context default
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None  # For extended context

    # Normalization and activation
    rms_norm_eps: float = 1e-6
    hidden_act: Literal["silu", "gelu", "swiglu"] = "swiglu"

    # Regularization
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # Initialization
    initializer_range: float = 0.02

    # Efficiency options
    use_cache: bool = True
    tie_word_embeddings: bool = False

    # Training specific
    gradient_checkpointing: bool = False

    # Tool calling configuration
    tool_calling: bool = True
    num_tool_tokens: int = 64  # Special tokens for tool interactions

    # Computed
    head_dim: int = field(init=False)

    def __post_init__(self):
        self.head_dim = self.hidden_size // self.num_attention_heads

        assert self.num_attention_heads % self.num_key_value_heads == 0, \
            f"num_attention_heads ({self.num_attention_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})"

        if self.hidden_act == "swiglu" and self.intermediate_size == 0:
            self.intermediate_size = int(2.7 * self.hidden_size)
            self.intermediate_size = ((self.intermediate_size + 255) // 256) * 256

    @property
    def total_vocab_size(self) -> int:
        """Total vocabulary including tool tokens."""
        if self.tool_calling:
            return self.vocab_size + self.num_tool_tokens
        return self.vocab_size

    def num_parameters(self, include_embeddings: bool = True) -> int:
        """Calculate total parameter count."""
        params = 0

        if include_embeddings:
            vocab = self.total_vocab_size
            params += vocab * self.hidden_size
            if not self.tie_word_embeddings:
                params += vocab * self.hidden_size

        per_layer = 0
        per_layer += self.hidden_size * self.head_dim * self.num_attention_heads  # Q
        per_layer += self.hidden_size * self.head_dim * self.num_key_value_heads  # K
        per_layer += self.hidden_size * self.head_dim * self.num_key_value_heads  # V
        per_layer += self.head_dim * self.num_attention_heads * self.hidden_size  # O

        if self.hidden_act == "swiglu":
            per_layer += self.hidden_size * self.intermediate_size  # gate
            per_layer += self.hidden_size * self.intermediate_size  # up
            per_layer += self.intermediate_size * self.hidden_size  # down
        else:
            per_layer += self.hidden_size * self.intermediate_size * 2

        per_layer += 2 * self.hidden_size  # RMSNorm
        params += per_layer * self.num_hidden_layers
        params += self.hidden_size  # Final norm

        return params

    def memory_footprint(self, dtype_bytes: int = 2, batch_size: int = 1, seq_len: int = 2048) -> dict:
        """Estimate memory usage in bytes."""
        params = self.num_parameters()

        weight_memory = params * dtype_bytes
        optimizer_memory = params * 4 * 3
        gradient_memory = params * dtype_bytes

        activation_per_layer = (
            batch_size * seq_len * self.hidden_size * dtype_bytes * 4 +
            batch_size * self.num_attention_heads * seq_len * seq_len * dtype_bytes
        )
        if self.gradient_checkpointing:
            activation_memory = activation_per_layer * 2
        else:
            activation_memory = activation_per_layer * self.num_hidden_layers

        return {
            "weights_gb": weight_memory / 1e9,
            "optimizer_gb": optimizer_memory / 1e9,
            "gradients_gb": gradient_memory / 1e9,
            "activations_gb": activation_memory / 1e9,
            "total_training_gb": (weight_memory + optimizer_memory + gradient_memory + activation_memory) / 1e9,
            "inference_gb": weight_memory / 1e9,
        }

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TransformerConfig":
        with open(path) as f:
            data = json.load(f)
            # Handle backwards compatibility
            if 'model_type' not in data:
                data['model_type'] = 'reasoner'
            if 'tool_calling' not in data:
                data['tool_calling'] = False
            if 'num_tool_tokens' not in data:
                data['num_tool_tokens'] = 64
            return cls(**data)


# =============================================================================
# SVEND Model Configurations - Server Deployment
# =============================================================================

def create_3b_verifier_config() -> TransformerConfig:
    """
    3B parameter verifier model.

    Fast model that checks reasoning chains for errors.
    Runs alongside the main reasoner for verification loops.
    """
    return TransformerConfig(
        name="svend-verifier-3b",
        model_type="verifier",
        vocab_size=32000,
        hidden_size=2560,
        intermediate_size=6912,
        num_hidden_layers=32,
        num_attention_heads=20,
        num_key_value_heads=4,
        max_position_embeddings=8192,
        hidden_act="swiglu",
        tool_calling=False,  # Verifier doesn't call tools
    )


def create_7b_reasoner_config() -> TransformerConfig:
    """
    7B parameter main reasoning model.

    Primary model for complex reasoning with tool calling.
    Fits comfortably on A100 80GB for training.
    """
    return TransformerConfig(
        name="svend-reasoner-7b",
        model_type="reasoner",
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=8192,
        hidden_act="swiglu",
        tool_calling=True,
        gradient_checkpointing=True,
    )


def create_13b_reasoner_config() -> TransformerConfig:
    """
    13B parameter advanced reasoning model.

    Maximum capability for complex multi-step reasoning.
    Requires A100 80GB with gradient checkpointing + bf16.
    """
    return TransformerConfig(
        name="svend-reasoner-13b",
        model_type="reasoner",
        vocab_size=32000,
        hidden_size=5120,
        intermediate_size=13824,
        num_hidden_layers=40,
        num_attention_heads=40,
        num_key_value_heads=8,
        max_position_embeddings=8192,
        hidden_act="swiglu",
        tool_calling=True,
        gradient_checkpointing=True,
    )


def create_13b_extended_config() -> TransformerConfig:
    """
    13B with 32K context for very long reasoning chains.

    Uses RoPE scaling for extended context.
    For complex multi-document reasoning.
    """
    config = create_13b_reasoner_config()
    config.name = "svend-reasoner-13b-32k"
    config.max_position_embeddings = 32768
    config.rope_scaling = {
        "type": "dynamic",
        "factor": 4.0,
    }
    return config


# Smaller models for testing and edge deployment
def create_500m_config() -> TransformerConfig:
    """Small model for testing pipeline."""
    return TransformerConfig(
        name="svend-test-500m",
        model_type="reasoner",
        vocab_size=32000,
        hidden_size=1024,
        intermediate_size=2816,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=4,
        max_position_embeddings=4096,
        hidden_act="swiglu",
        tool_calling=True,
    )


def create_1b_config() -> TransformerConfig:
    """1B model - good for fast iteration."""
    return TransformerConfig(
        name="svend-reasoner-1b",
        model_type="reasoner",
        vocab_size=32000,
        hidden_size=1536,
        intermediate_size=4096,
        num_hidden_layers=28,
        num_attention_heads=12,
        num_key_value_heads=4,
        max_position_embeddings=8192,
        hidden_act="swiglu",
        tool_calling=True,
    )


def create_1_5b_reasoning_config() -> TransformerConfig:
    """
    1.5B Reasoning specialist - sweet spot for tool-augmented reasoning.

    ~27GB training on A100, leaves plenty of headroom for larger batches.
    Good balance of capability vs iteration speed.
    """
    return TransformerConfig(
        name="svend-reasoning-1.5b",
        model_type="reasoning",
        vocab_size=32000,
        hidden_size=2048,
        intermediate_size=5504,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=4,
        max_position_embeddings=8192,
        hidden_act="swiglu",
        tool_calling=True,
    )


def create_2b_reasoning_config() -> TransformerConfig:
    """
    2B Reasoning specialist - higher capability, still fits A100 80GB.

    ~33GB training. Recommended for production reasoning specialist.
    """
    return TransformerConfig(
        name="svend-reasoning-2b",
        model_type="reasoning",
        vocab_size=32000,
        hidden_size=2304,
        intermediate_size=6144,
        num_hidden_layers=28,
        num_attention_heads=18,
        num_key_value_heads=6,
        max_position_embeddings=8192,
        hidden_act="swiglu",
        tool_calling=True,
    )


def create_3b_config() -> TransformerConfig:
    """3B reasoner - solid mid-range option."""
    return TransformerConfig(
        name="svend-reasoner-3b",
        model_type="reasoner",
        vocab_size=32000,
        hidden_size=2560,
        intermediate_size=6912,
        num_hidden_layers=32,
        num_attention_heads=20,
        num_key_value_heads=4,
        max_position_embeddings=8192,
        hidden_act="swiglu",
        tool_calling=True,
    )


def create_7b_config() -> TransformerConfig:
    """Alias for primary 7B reasoner."""
    return create_7b_reasoner_config()


def create_13b_config() -> TransformerConfig:
    """Alias for 13B reasoner."""
    return create_13b_reasoner_config()


# =============================================================================
# SPECIALIST Model Configurations - Multi-Model Ensemble
# =============================================================================

def create_router_config() -> TransformerConfig:
    """
    125M Router model - classifies intent and routes to specialists.

    Very fast, runs on every request to determine which specialist(s) to invoke.
    Trained on intent classification + domain detection.
    """
    return TransformerConfig(
        name="svend-router-125m",
        model_type="router",
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,
        max_position_embeddings=2048,  # Short context - just needs to see the prompt
        hidden_act="swiglu",
        tool_calling=False,
    )


def create_language_specialist_config() -> TransformerConfig:
    """
    500M Language specialist - prompt interpretation, synthesis, output formatting.

    Handles:
    - Understanding user intent
    - Synthesizing outputs from other specialists
    - Generating natural language responses
    - General conversation
    """
    return TransformerConfig(
        name="svend-language-500m",
        model_type="language",
        vocab_size=32000,
        hidden_size=1024,
        intermediate_size=2816,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=4,
        max_position_embeddings=4096,
        hidden_act="swiglu",
        tool_calling=False,  # Language model doesn't call tools directly
    )


def create_reasoning_specialist_config() -> TransformerConfig:
    """
    500M Reasoning specialist - math, logic, chain-of-thought, tool orchestration.

    Handles:
    - Step-by-step reasoning chains
    - Mathematical problem solving
    - Logical deduction
    - Calling external tools (SymPy, Z3, code sandbox)
    """
    return TransformerConfig(
        name="svend-reasoning-500m",
        model_type="reasoning",
        vocab_size=32000,
        hidden_size=1024,
        intermediate_size=2816,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=4,
        max_position_embeddings=8192,  # Longer context for reasoning chains
        hidden_act="swiglu",
        tool_calling=True,
    )


def create_verifier_specialist_config() -> TransformerConfig:
    """
    250M Verifier specialist - checks answers, catches errors.

    Handles:
    - Validating reasoning steps
    - Checking mathematical correctness
    - Flagging potential errors
    - Confidence scoring
    """
    return TransformerConfig(
        name="svend-verifier-250m",
        model_type="verifier",
        vocab_size=32000,
        hidden_size=896,
        intermediate_size=2432,
        num_hidden_layers=16,
        num_attention_heads=14,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        hidden_act="swiglu",
        tool_calling=False,
    )


# =============================================================================
# Ensemble Configuration
# =============================================================================

@dataclass
class EnsembleConfig:
    """Configuration for the full Svend ensemble system."""

    router: TransformerConfig = None
    language: TransformerConfig = None
    reasoning: TransformerConfig = None
    verifier: TransformerConfig = None

    def __post_init__(self):
        if self.router is None:
            self.router = create_router_config()
        if self.language is None:
            self.language = create_language_specialist_config()
        if self.reasoning is None:
            self.reasoning = create_reasoning_specialist_config()
        if self.verifier is None:
            self.verifier = create_verifier_specialist_config()

    def total_parameters(self) -> int:
        """Total parameters across all models."""
        return (
            self.router.num_parameters() +
            self.language.num_parameters() +
            self.reasoning.num_parameters() +
            self.verifier.num_parameters()
        )

    def total_memory(self) -> dict:
        """Combined memory footprint."""
        models = [self.router, self.language, self.reasoning, self.verifier]
        total = {"weights_gb": 0, "inference_gb": 0, "total_training_gb": 0}
        for m in models:
            mem = m.memory_footprint()
            total["weights_gb"] += mem["weights_gb"]
            total["inference_gb"] += mem["inference_gb"]
            total["total_training_gb"] += mem["total_training_gb"]
        return total

    def print_summary(self):
        """Print ensemble summary."""
        print("\n" + "=" * 70)
        print("SVEND Ensemble Configuration")
        print("=" * 70)

        models = [
            ("Router", self.router),
            ("Language", self.language),
            ("Reasoning", self.reasoning),
            ("Verifier", self.verifier),
        ]

        for name, config in models:
            params = config.num_parameters()
            mem = config.memory_footprint()
            print(f"\n{name}: {config.name}")
            print(f"  Parameters: {params / 1e6:.0f}M")
            print(f"  Hidden: {config.hidden_size}, Layers: {config.num_hidden_layers}")
            print(f"  Context: {config.max_position_embeddings}, Tools: {config.tool_calling}")
            print(f"  Training memory: {mem['total_training_gb']:.1f} GB")

        total = self.total_parameters()
        total_mem = self.total_memory()
        print(f"\n{'=' * 70}")
        print(f"TOTAL: {total / 1e6:.0f}M parameters")
        print(f"Combined inference memory: {total_mem['inference_gb']:.1f} GB")
        print(f"Combined training memory: {total_mem['total_training_gb']:.1f} GB")
        print("=" * 70)


def create_ensemble_config() -> EnsembleConfig:
    """Create the default Svend ensemble configuration."""
    return EnsembleConfig()


MODEL_CONFIGS = {
    # Single model configs (legacy + scaled)
    "500m": create_500m_config,
    "1b": create_1b_config,
    "1.5b": create_1_5b_reasoning_config,
    "2b": create_2b_reasoning_config,
    "3b": create_3b_config,
    "3b-verifier": create_3b_verifier_config,
    "7b": create_7b_reasoner_config,
    "13b": create_13b_reasoner_config,
    "13b-32k": create_13b_extended_config,
    # Specialist configs
    "router": create_router_config,
    "language": create_language_specialist_config,
    "reasoning": create_reasoning_specialist_config,
    "reasoning-1.5b": create_1_5b_reasoning_config,
    "reasoning-2b": create_2b_reasoning_config,
    "verifier": create_verifier_specialist_config,
}


def get_config(size: str) -> TransformerConfig:
    if size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model size: {size}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[size]()


def print_config_comparison():
    """Print a comparison table of all configurations."""
    print("\n" + "=" * 100)
    print("SVEND Model Configuration Comparison")
    print("=" * 100)

    headers = ["Size", "Type", "Params", "Hidden", "Layers", "Heads", "Context", "Tools", "Train GB"]
    row_format = "{:<12} {:<10} {:<10} {:<8} {:<8} {:<8} {:<10} {:<8} {:<10}"

    print(row_format.format(*headers))
    print("-" * 100)

    for name, config_fn in MODEL_CONFIGS.items():
        config = config_fn()
        params = config.num_parameters()
        memory = config.memory_footprint()

        print(row_format.format(
            name,
            config.model_type[:8],
            f"{params / 1e9:.2f}B",
            config.hidden_size,
            config.num_hidden_layers,
            config.num_attention_heads,
            f"{config.max_position_embeddings // 1024}K",
            "Yes" if config.tool_calling else "No",
            f"{memory['total_training_gb']:.1f}",
        ))

    print("=" * 100)


if __name__ == "__main__":
    print_config_comparison()
