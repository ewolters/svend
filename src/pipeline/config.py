"""
Pipeline Configuration System

Defines model scales and training configurations for progressive scaling.
Start small, validate, scale up - never skip validation steps.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Literal
from enum import Enum
from pathlib import Path
import json
import yaml


class ModelScale(Enum):
    """
    Progressive model scales for development.

    Strategy: Validate at each scale before moving up.
    Each scale should pass all tests before proceeding.
    """
    TINY = "tiny"         # 50M  - Pipeline validation (CPU/6GB GPU)
    SMALL = "small"       # 125M - Quick experiments (8GB GPU)
    MEDIUM = "medium"     # 500M - Serious testing (12GB GPU)
    LARGE = "large"       # 1B   - Pre-production validation (24GB GPU)
    XL = "xl"             # 3B   - Production candidate (40GB GPU)
    XXL = "xxl"           # 7B   - Full scale (80GB GPU)
    FLAGSHIP = "flagship" # 13B  - Maximum capability (80GB GPU, gradient checkpointing)


@dataclass
class ModelConfig:
    """Configuration for a specific model scale."""

    name: str
    scale: ModelScale

    # Architecture
    vocab_size: int = 32000
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 4  # GQA ratio
    max_position_embeddings: int = 4096

    # Training defaults for this scale
    default_batch_size: int = 8
    default_grad_accum: int = 4
    default_learning_rate: float = 1e-4
    recommended_samples: int = 10000

    # Memory estimates (GB)
    training_memory_gb: float = 8.0
    inference_memory_gb: float = 2.0

    # Capabilities
    supports_tools: bool = True
    supports_verification: bool = False  # Verifier models only

    def num_parameters(self) -> int:
        """Estimate parameter count."""
        # Embeddings
        params = self.vocab_size * self.hidden_size * 2  # in + out

        # Per layer
        per_layer = 0
        head_dim = self.hidden_size // self.num_attention_heads
        per_layer += self.hidden_size * head_dim * self.num_attention_heads  # Q
        per_layer += self.hidden_size * head_dim * self.num_key_value_heads  # K
        per_layer += self.hidden_size * head_dim * self.num_key_value_heads  # V
        per_layer += head_dim * self.num_attention_heads * self.hidden_size  # O
        per_layer += self.hidden_size * self.intermediate_size * 3  # SwiGLU
        per_layer += self.hidden_size * 2  # norms

        params += per_layer * self.num_hidden_layers
        params += self.hidden_size  # final norm

        return params

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "scale": self.scale.value,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "num_parameters": self.num_parameters(),
        }


# Pre-defined scale configurations
SCALE_CONFIGS: Dict[ModelScale, ModelConfig] = {
    ModelScale.TINY: ModelConfig(
        name="svend-tiny-50m",
        scale=ModelScale.TINY,
        hidden_size=512,
        intermediate_size=1408,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        default_batch_size=16,
        default_grad_accum=2,
        default_learning_rate=3e-4,
        recommended_samples=1000,
        training_memory_gb=4.0,
        inference_memory_gb=0.5,
    ),
    ModelScale.SMALL: ModelConfig(
        name="svend-small-125m",
        scale=ModelScale.SMALL,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,
        max_position_embeddings=4096,
        default_batch_size=8,
        default_grad_accum=4,
        default_learning_rate=2e-4,
        recommended_samples=5000,
        training_memory_gb=8.0,
        inference_memory_gb=1.0,
    ),
    ModelScale.MEDIUM: ModelConfig(
        name="svend-medium-500m",
        scale=ModelScale.MEDIUM,
        hidden_size=1024,
        intermediate_size=2816,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=4,
        max_position_embeddings=4096,
        default_batch_size=4,
        default_grad_accum=8,
        default_learning_rate=1e-4,
        recommended_samples=20000,
        training_memory_gb=16.0,
        inference_memory_gb=2.0,
    ),
    ModelScale.LARGE: ModelConfig(
        name="svend-large-1b",
        scale=ModelScale.LARGE,
        hidden_size=1536,
        intermediate_size=4096,
        num_hidden_layers=28,
        num_attention_heads=12,
        num_key_value_heads=4,
        max_position_embeddings=8192,
        default_batch_size=2,
        default_grad_accum=16,
        default_learning_rate=1e-4,
        recommended_samples=50000,
        training_memory_gb=28.0,
        inference_memory_gb=4.0,
    ),
    ModelScale.XL: ModelConfig(
        name="svend-xl-3b",
        scale=ModelScale.XL,
        hidden_size=2560,
        intermediate_size=6912,
        num_hidden_layers=32,
        num_attention_heads=20,
        num_key_value_heads=4,
        max_position_embeddings=8192,
        default_batch_size=1,
        default_grad_accum=32,
        default_learning_rate=5e-5,
        recommended_samples=100000,
        training_memory_gb=45.0,
        inference_memory_gb=8.0,
    ),
    ModelScale.XXL: ModelConfig(
        name="svend-xxl-7b",
        scale=ModelScale.XXL,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=8192,
        default_batch_size=1,
        default_grad_accum=32,
        default_learning_rate=3e-5,
        recommended_samples=150000,
        training_memory_gb=65.0,
        inference_memory_gb=16.0,
    ),
    ModelScale.FLAGSHIP: ModelConfig(
        name="svend-flagship-13b",
        scale=ModelScale.FLAGSHIP,
        hidden_size=5120,
        intermediate_size=13824,
        num_hidden_layers=40,
        num_attention_heads=40,
        num_key_value_heads=8,
        max_position_embeddings=8192,
        default_batch_size=1,
        default_grad_accum=64,
        default_learning_rate=2e-5,
        recommended_samples=200000,
        training_memory_gb=80.0,
        inference_memory_gb=28.0,
    ),
}


def get_model_config(scale: ModelScale) -> ModelConfig:
    """Get the pre-defined config for a scale."""
    return SCALE_CONFIGS[scale]


@dataclass
class DataConfig:
    """Configuration for training data."""

    # Data sources
    sources: List[str] = field(default_factory=lambda: [
        "openmath",
        "slimorca",
        "cot_collection",
    ])

    # Mixing ratios (must sum to 1.0)
    mix_ratios: Dict[str, float] = field(default_factory=lambda: {
        "reasoning": 0.4,
        "tool_use": 0.3,
        "code": 0.15,
        "general": 0.15,
    })

    # Processing
    max_seq_length: int = 4096
    num_workers: int = 4

    # Validation split
    val_ratio: float = 0.05

    # Synthetic data
    synthetic_path: Optional[str] = None
    synthetic_ratio: float = 0.0  # Fraction of synthetic data


@dataclass
class TrainingConfig:
    """Configuration for a training run."""

    # Experiment identity
    experiment_name: str = "svend-experiment"
    run_id: Optional[str] = None  # Auto-generated if None

    # Model
    model_scale: ModelScale = ModelScale.TINY

    # Training parameters
    num_epochs: int = 3
    max_steps: Optional[int] = None
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0

    # Batch settings
    batch_size: int = 4
    gradient_accumulation_steps: int = 8

    # Precision
    mixed_precision: bool = True
    bf16: bool = True

    # Optimization
    gradient_checkpointing: bool = True
    fused_adam: bool = True

    # Scheduler
    lr_scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    min_lr_ratio: float = 0.1

    # Data
    data: DataConfig = field(default_factory=DataConfig)

    # Checkpointing
    output_dir: str = "checkpoints"
    save_steps: int = 500
    save_total_limit: int = 3
    resume_from: Optional[str] = None

    # Logging
    logging_steps: int = 10
    eval_steps: int = 500
    use_wandb: bool = True
    wandb_project: str = "svend"

    # Validation gates (must pass to continue training)
    validation_gates: Dict[str, float] = field(default_factory=lambda: {
        "min_accuracy": 0.0,  # Set per experiment
        "max_loss": 10.0,
        "min_tool_accuracy": 0.0,
    })

    def __post_init__(self):
        import uuid
        from datetime import datetime

        if self.run_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            short_id = str(uuid.uuid4())[:8]
            self.run_id = f"{self.experiment_name}_{timestamp}_{short_id}"

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    def apply_scale_defaults(self):
        """Apply defaults from model scale config."""
        scale_config = get_model_config(self.model_scale)
        self.batch_size = scale_config.default_batch_size
        self.gradient_accumulation_steps = scale_config.default_grad_accum
        self.learning_rate = scale_config.default_learning_rate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "run_id": self.run_id,
            "model_scale": self.model_scale.value,
            "num_epochs": self.num_epochs,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "effective_batch_size": self.effective_batch_size,
            "mixed_precision": self.mixed_precision,
            "bf16": self.bf16,
        }

    def save(self, path: str):
        """Save config to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = {
            "experiment_name": self.experiment_name,
            "run_id": self.run_id,
            "model_scale": self.model_scale.value,
            "num_epochs": self.num_epochs,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "mixed_precision": self.mixed_precision,
            "bf16": self.bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "lr_scheduler": self.lr_scheduler,
            "output_dir": self.output_dir,
            "save_steps": self.save_steps,
            "use_wandb": self.use_wandb,
            "wandb_project": self.wandb_project,
            "data": {
                "sources": self.data.sources,
                "mix_ratios": self.data.mix_ratios,
                "max_seq_length": self.data.max_seq_length,
                "val_ratio": self.data.val_ratio,
            },
            "validation_gates": self.validation_gates,
        }

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        data["model_scale"] = ModelScale(data["model_scale"])

        if "data" in data:
            data["data"] = DataConfig(**data["data"])

        return cls(**data)


@dataclass
class PipelineConfig:
    """
    Master configuration for the full training pipeline.

    Orchestrates:
    - Model configuration at specified scale
    - Training configuration
    - Evaluation configuration
    - Deployment configuration
    """

    # Core configs
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Pipeline behavior
    auto_scale: bool = False  # Automatically try next scale on success
    strict_validation: bool = True  # Fail on validation gate failures

    # Evaluation settings
    eval_benchmarks: List[str] = field(default_factory=lambda: [
        "gsm8k_sample",  # Small sample for quick validation
        "tool_accuracy",
    ])
    full_eval_benchmarks: List[str] = field(default_factory=lambda: [
        "gsm8k",
        "math",
        "humaneval",
        "tool_accuracy",
        "safety",
    ])

    # Paths
    base_dir: str = "."
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    eval_dir: str = "evaluations"
    log_dir: str = "logs"

    def get_model_config(self) -> ModelConfig:
        """Get model config for current scale."""
        return get_model_config(self.training.model_scale)

    def next_scale(self) -> Optional[ModelScale]:
        """Get the next scale up, if any."""
        scales = list(ModelScale)
        current_idx = scales.index(self.training.model_scale)
        if current_idx < len(scales) - 1:
            return scales[current_idx + 1]
        return None


def create_quick_test_config() -> PipelineConfig:
    """Create a config for quick pipeline testing."""
    training = TrainingConfig(
        experiment_name="quick-test",
        model_scale=ModelScale.TINY,
        num_epochs=1,
        max_steps=100,
        learning_rate=3e-4,
        batch_size=4,
        gradient_accumulation_steps=2,
        save_steps=50,
        logging_steps=10,
        eval_steps=50,
        use_wandb=False,
    )
    training.data.max_seq_length = 512

    return PipelineConfig(
        training=training,
        strict_validation=False,
        eval_benchmarks=["tool_accuracy_sample"],
    )


def create_scale_test_config(scale: ModelScale) -> PipelineConfig:
    """Create a config for testing a specific scale."""
    training = TrainingConfig(
        experiment_name=f"scale-test-{scale.value}",
        model_scale=scale,
        num_epochs=1,
        use_wandb=True,
    )
    training.apply_scale_defaults()

    # Reduce steps for testing
    scale_config = get_model_config(scale)
    training.max_steps = min(1000, scale_config.recommended_samples // training.effective_batch_size)

    return PipelineConfig(training=training)


def create_production_config(scale: ModelScale = ModelScale.FLAGSHIP) -> PipelineConfig:
    """Create a config for production training."""
    training = TrainingConfig(
        experiment_name=f"svend-production-{scale.value}",
        model_scale=scale,
        num_epochs=3,
        use_wandb=True,
        wandb_project="svend-production",
    )
    training.apply_scale_defaults()

    # Production validation gates
    training.validation_gates = {
        "min_accuracy": 0.3,
        "max_loss": 3.0,
        "min_tool_accuracy": 0.5,
    }

    return PipelineConfig(
        training=training,
        strict_validation=True,
        eval_benchmarks=["gsm8k_sample", "tool_accuracy", "safety_sample"],
    )


def print_scale_comparison():
    """Print a comparison of all model scales."""
    print("\n" + "=" * 100)
    print("SVEND Model Scale Comparison")
    print("=" * 100)

    headers = ["Scale", "Name", "Params", "Hidden", "Layers", "Heads", "Context", "Train GB", "Samples"]
    row_format = "{:<10} {:<20} {:<10} {:<8} {:<8} {:<8} {:<10} {:<10} {:<10}"

    print(row_format.format(*headers))
    print("-" * 100)

    for scale in ModelScale:
        config = get_model_config(scale)
        params = config.num_parameters()

        print(row_format.format(
            scale.value,
            config.name,
            f"{params / 1e6:.0f}M" if params < 1e9 else f"{params / 1e9:.1f}B",
            config.hidden_size,
            config.num_hidden_layers,
            config.num_attention_heads,
            f"{config.max_position_embeddings // 1024}K",
            f"{config.training_memory_gb:.0f}",
            f"{config.recommended_samples // 1000}K",
        ))

    print("=" * 100)


if __name__ == "__main__":
    print_scale_comparison()
