# Svend Training Pipeline Guide

This guide covers the training pipeline infrastructure for building Svend reasoning models. The pipeline is designed for iterative development: **test small, validate, scale up**.

## Table of Contents

1. [Philosophy](#philosophy)
2. [Quick Start](#quick-start)
3. [Architecture Overview](#architecture-overview)
4. [Model Scales](#model-scales)
5. [Configuration System](#configuration-system)
6. [Validation Gates](#validation-gates)
7. [Checkpoint Management](#checkpoint-management)
8. [Running Training](#running-training)
9. [Progressive Scaling](#progressive-scaling)
10. [Colab Integration](#colab-integration)
11. [Troubleshooting](#troubleshooting)

---

## Philosophy

The Svend training pipeline follows these principles:

1. **No vibe coding** - Every component is tested and validated
2. **Progressive scaling** - Start with tiny models, validate, scale up
3. **Fail fast** - Catch problems early with validation gates
4. **Reproducibility** - All configs and checkpoints are tracked
5. **Colab-friendly** - Designed for Google Colab A100 training

### The Scale-Up Process

```
TINY (50M)     →  Validate pipeline, test configs
    ↓
SMALL (125M)   →  Quick experiments, hyperparameter search
    ↓
MEDIUM (500M)  →  Serious testing, architecture validation
    ↓
LARGE (1B)     →  Pre-production validation
    ↓
XL (3B)        →  Production candidate
    ↓
XXL (7B)       →  Full scale validation
    ↓
FLAGSHIP (13B) →  Maximum capability
```

Never skip validation between scales. A model that fails at 500M will fail at 13B - find the problem early.

---

## Quick Start

### 1. Validate Infrastructure

Before any training, verify everything works:

```python
from src.pipeline import PipelineRunner, PipelineConfig
from src.pipeline.config import create_quick_test_config

# Create minimal test config
config = create_quick_test_config()
runner = PipelineRunner(config)

# Run infrastructure checks
if runner.validate_infrastructure():
    print("Ready for training!")
else:
    print("Fix infrastructure issues first")
```

### 2. Test at Tiny Scale

```python
from src.pipeline.config import ModelScale, create_scale_test_config

# Create config for tiny model
config = create_scale_test_config(ModelScale.TINY)

# Run short training
runner = PipelineRunner(config)
result = runner.run()

if result.success:
    print(f"Tiny model trained successfully! Loss: {result.final_loss:.4f}")
```

### 3. Progressive Training

```python
from src.pipeline import PipelineRunner, PipelineConfig
from src.pipeline.config import ModelScale

config = PipelineConfig()
config.training.experiment_name = "svend-v1"

runner = PipelineRunner(config)

# Train from tiny to large, validating at each step
results = runner.run_progressive(
    start_scale=ModelScale.TINY,
    end_scale=ModelScale.LARGE,
    validation_between_scales=True,
)
```

---

## Architecture Overview

```
src/pipeline/
├── __init__.py       # Public API exports
├── config.py         # Configuration system
├── runner.py         # Pipeline orchestration
├── validation.py     # Validation gates
└── checkpoints.py    # Checkpoint management
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `PipelineConfig` | Master configuration for training runs |
| `ModelConfig` | Per-scale model architecture settings |
| `TrainingConfig` | Hyperparameters and training settings |
| `PipelineRunner` | Orchestrates training with validation |
| `PipelineValidator` | Enforces quality gates |
| `CheckpointManager` | Handles checkpoint save/load |

---

## Model Scales

Seven predefined scales, each with optimized defaults:

| Scale | Parameters | Hidden | Layers | GPU Memory | Use Case |
|-------|------------|--------|--------|------------|----------|
| TINY | 50M | 512 | 8 | 4 GB | Pipeline testing |
| SMALL | 125M | 768 | 12 | 8 GB | Quick experiments |
| MEDIUM | 500M | 1024 | 24 | 16 GB | Serious testing |
| LARGE | 1B | 1536 | 28 | 28 GB | Pre-production |
| XL | 3B | 2560 | 32 | 45 GB | Production candidate |
| XXL | 7B | 4096 | 32 | 65 GB | Full scale |
| FLAGSHIP | 13B | 5120 | 40 | 80 GB | Maximum capability |

### Getting Scale Configuration

```python
from src.pipeline.config import ModelScale, get_model_config

config = get_model_config(ModelScale.MEDIUM)
print(f"Name: {config.name}")
print(f"Parameters: {config.num_parameters() / 1e6:.0f}M")
print(f"Hidden size: {config.hidden_size}")
print(f"Recommended samples: {config.recommended_samples}")
```

### Print All Scales

```python
from src.pipeline.config import print_scale_comparison
print_scale_comparison()
```

Output:
```
====================================================================================================
SVEND Model Scale Comparison
====================================================================================================
Scale      Name                 Params     Hidden   Layers   Heads    Context    Train GB   Samples
----------------------------------------------------------------------------------------------------
tiny       svend-tiny-50m       52M        512      8        8        2K         4          1K
small      svend-small-125m     125M       768      12       12       4K         8          5K
medium     svend-medium-500m    503M       1024     24       16       4K         16         20K
large      svend-large-1b       1.0B       1536     28       12       8K         28         50K
xl         svend-xl-3b          2.9B       2560     32       20       8K         45         100K
xxl        svend-xxl-7b         6.9B       4096     32       32       8K         65         150K
flagship   svend-flagship-13b   13.0B      5120     40       40       8K         80         200K
====================================================================================================
```

---

## Configuration System

### PipelineConfig

Master configuration that controls the entire training run:

```python
from src.pipeline.config import PipelineConfig, TrainingConfig, DataConfig, ModelScale

# Create custom configuration
training = TrainingConfig(
    experiment_name="my-experiment",
    model_scale=ModelScale.MEDIUM,
    num_epochs=3,
    learning_rate=1e-4,
    batch_size=4,
    gradient_accumulation_steps=8,
    use_wandb=True,
    wandb_project="svend",
)

# Data configuration
training.data = DataConfig(
    sources=["openmath", "slimorca"],
    mix_ratios={
        "reasoning": 0.5,
        "tool_use": 0.3,
        "general": 0.2,
    },
    max_seq_length=4096,
)

# Validation gates
training.validation_gates = {
    "min_accuracy": 0.2,
    "max_loss": 5.0,
    "min_tool_accuracy": 0.5,
}

config = PipelineConfig(
    training=training,
    strict_validation=True,
    eval_benchmarks=["gsm8k_sample", "tool_accuracy"],
)
```

### Using Scale Defaults

Each scale has optimized defaults. Apply them automatically:

```python
training = TrainingConfig(
    experiment_name="auto-config",
    model_scale=ModelScale.LARGE,
)
training.apply_scale_defaults()  # Sets batch_size, learning_rate, etc.
```

### Saving and Loading Configs

```python
# Save configuration
config.training.save("configs/my_experiment.yaml")

# Load configuration
training = TrainingConfig.load("configs/my_experiment.yaml")
```

---

## Validation Gates

Validation gates prevent wasted compute by catching problems early.

### Gate Levels

| Level | When Used | Strictness |
|-------|-----------|------------|
| QUICK | Every 100 steps | Very lenient, catches crashes |
| STANDARD | At checkpoints | Moderate, catches training issues |
| FULL | Before scale-up | Strict, ensures quality |
| PRODUCTION | Before deployment | Very strict, production-ready |

### Built-in Gates

**Quick Gates** (during training):
- `loss_reasonable`: Loss < 100 (catches explosions)
- `no_nan`: No NaN values in model

**Standard Gates** (at checkpoints):
- `loss_decreasing`: Loss < 10
- `gradient_norm`: Gradients < 100
- `perplexity`: Perplexity < 1000

**Full Gates** (before scale-up):
- `min_accuracy`: Accuracy > 10%
- `tool_accuracy`: Tool use > 30%

**Production Gates** (before deployment):
- `gsm8k_accuracy`: GSM8K > 50%
- `tool_accuracy`: Tool use > 85%
- `safety_accuracy`: Safety > 95%
- `latency_p99`: P99 latency < 5s

### Custom Gates

```python
from src.pipeline.validation import ValidationGate, PipelineValidator

custom_gates = [
    ValidationGate(
        name="chemistry_accuracy",
        metric_name="chemistry_accuracy",
        threshold=0.6,
        comparison=">=",
        required=True,
        description="Accuracy on chemistry reasoning tasks",
    ),
]

validator = PipelineValidator(
    strict=True,
    custom_gates=custom_gates,
)
```

### Manual Validation

```python
# Quick check during training
result = validator.validate_quick(model, loss=0.5, step=1000)
print(result.summary())

# Full validation before scale-up
result = validator.validate_full(model, eval_dataloader)
if result.passed:
    print("Ready for next scale!")
else:
    print(f"Failed: {result.failures}")
```

---

## Checkpoint Management

### Basic Usage

```python
from src.pipeline.checkpoints import CheckpointManager

manager = CheckpointManager(
    output_dir="checkpoints/my-run",
    save_total_limit=3,      # Keep 3 most recent
    best_metric="loss",      # Track best by loss
    best_metric_mode="min",  # Lower is better
)

# Save checkpoint
checkpoint_id = manager.save(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    step=1000,
    metrics={"loss": 0.5, "accuracy": 0.7},
)

# Load latest checkpoint
step, epoch, extra_state = manager.load_latest(model, optimizer, scheduler)

# Load best checkpoint
step, epoch, extra_state = manager.load_best(model)

# Print summary
manager.print_summary()
```

### Checkpoint Contents

Each checkpoint directory contains:
```
checkpoints/my-run/step_00001000/
├── model.pt           # Model weights
├── optimizer.pt       # Optimizer state
├── scheduler.pt       # Learning rate scheduler
├── scaler.pt          # AMP gradient scaler
├── extra_state.pt     # Custom state
└── metadata.json      # Full metadata
```

### Google Drive Integration (Colab)

```python
from src.pipeline.checkpoints import DriveCheckpointManager

manager = DriveCheckpointManager(
    drive_path="/content/drive/MyDrive/svend-checkpoints/my-run",
    local_cache="/content/checkpoints",
)

# Automatically mounts Drive and persists checkpoints
manager.save(model=model, step=1000, metrics={"loss": 0.5})
```

---

## Running Training

### Single Scale Training

```python
from src.pipeline import PipelineRunner, PipelineConfig
from src.pipeline.config import ModelScale, TrainingConfig

training = TrainingConfig(
    experiment_name="svend-medium-test",
    model_scale=ModelScale.MEDIUM,
    num_epochs=2,
)
training.apply_scale_defaults()

config = PipelineConfig(training=training)
runner = PipelineRunner(config)

result = runner.run()

print(f"Success: {result.success}")
print(f"Final loss: {result.final_loss:.4f}")
print(f"Training time: {result.training_time_seconds / 3600:.2f} hours")
```

### Custom Model Factory

If you need custom model architecture:

```python
def custom_model_factory(scale: ModelScale) -> nn.Module:
    from src.models.transformer import ReasoningTransformer
    from src.models.config import TransformerConfig
    from src.pipeline.config import get_model_config

    scale_config = get_model_config(scale)

    # Customize architecture
    model_config = TransformerConfig(
        name=f"custom-{scale.value}",
        hidden_size=scale_config.hidden_size,
        num_hidden_layers=scale_config.num_hidden_layers,
        # Custom modifications
        attention_dropout=0.1,
        tool_calling=True,
    )

    return ReasoningTransformer(model_config)

runner = PipelineRunner(config, model_factory=custom_model_factory)
```

---

## Progressive Scaling

The pipeline supports automatic progression through scales:

```python
from src.pipeline import PipelineRunner, PipelineConfig
from src.pipeline.config import ModelScale

config = PipelineConfig()
config.training.experiment_name = "svend-progressive"

runner = PipelineRunner(config)

# Train from tiny to xl, validating at each step
results = runner.run_progressive(
    start_scale=ModelScale.TINY,
    end_scale=ModelScale.XL,
    validation_between_scales=True,
)

# Check results for each scale
for result in results:
    status = "PASS" if result.success else "FAIL"
    print(f"{result.scale.value}: {status} (loss={result.final_loss:.4f})")
```

### Scale-Up Criteria

Before proceeding to the next scale, the model must:
1. Complete training without crashes
2. Pass all required validation gates
3. Show decreasing loss trend

If validation fails at any scale, progression stops.

---

## Colab Integration

### Setup Notebook

```python
# Cell 1: Mount Drive and install dependencies
from google.colab import drive
drive.mount('/content/drive')

!pip install wandb transformers datasets accelerate
!git clone https://github.com/YOUR_REPO/reasoning-lab.git
%cd reasoning-lab
!pip install -r requirements.txt

# Cell 2: Check GPU
import torch
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Training with Drive Checkpoints

```python
from src.pipeline import PipelineRunner, PipelineConfig
from src.pipeline.config import ModelScale, TrainingConfig
from src.pipeline.checkpoints import DriveCheckpointManager

# Configure for Colab
training = TrainingConfig(
    experiment_name="svend-colab",
    model_scale=ModelScale.XXL,  # 7B for A100 80GB
    num_epochs=2,
    save_steps=500,  # Frequent saves for Colab timeouts
    use_wandb=True,
    wandb_project="svend-production",
)
training.apply_scale_defaults()

config = PipelineConfig(
    training=training,
    checkpoint_dir="/content/drive/MyDrive/svend-checkpoints",
)

runner = PipelineRunner(config)

# Validate before training
if runner.validate_infrastructure():
    result = runner.run()
```

### Resume After Timeout

```python
# After Colab reconnection
from src.pipeline.checkpoints import DriveCheckpointManager

# Load latest checkpoint
manager = DriveCheckpointManager(
    drive_path="/content/drive/MyDrive/svend-checkpoints/svend-colab",
)

step, epoch, _ = manager.load_latest(model, optimizer, scheduler)
print(f"Resuming from step {step}")

# Continue training...
```

---

## Troubleshooting

### Common Issues

**Out of Memory**
```python
# Reduce batch size
training.batch_size = 1
training.gradient_accumulation_steps = 32

# Enable gradient checkpointing
training.gradient_checkpointing = True

# Use a smaller scale
training.model_scale = ModelScale.LARGE  # instead of XL
```

**Loss Not Decreasing**
```python
# Check learning rate (scale-dependent)
training.apply_scale_defaults()

# Or manual adjustment
training.learning_rate = 5e-5  # Lower for larger models
```

**Validation Failing**
```python
# Check validation history
runner.validator.print_summary()

# Loosen gates for debugging
config.strict_validation = False
training.validation_gates["min_accuracy"] = 0.05
```

**Checkpoint Not Loading**
```python
# Check checkpoint contents
manager.print_summary()

# List all checkpoints
for cp in manager.list_checkpoints():
    print(f"{cp.checkpoint_id}: step={cp.step}, loss={cp.metrics.get('loss')}")
```

### Debug Mode

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with smaller data
training.max_steps = 100
training.data.sources = ["slimorca"]  # Single source

# Disable wandb
training.use_wandb = False
```

---

## Next Steps

After completing pipeline training:

1. **Evaluate**: Run full benchmark suite (GSM8K, MATH, HumanEval)
2. **Safety**: Train and integrate safety classifier
3. **Tools**: Add chemistry/physics tools
4. **Deploy**: Set up inference server on svend.ai

See also:
- `docs/SVEND_PRODUCTION_PLAN.md` - Full production roadmap
- `docs/TRAINING_STRATEGY.md` - Training data and hyperparameters
- `src/evaluation/` - Benchmark implementations
