#!/usr/bin/env python3
"""
Train Svend Ensemble Models.

Trains the 4-model specialist ensemble:
- Router (125M): Intent classification, routes to specialists
- Language (500M): Prompt interpretation, synthesis, output formatting
- Reasoning (500M): Math, logic, chain-of-thought, tool orchestration
- Verifier (250M): Checks answers, catches errors

Usage:
    # Train all models sequentially
    python scripts/train_ensemble.py --all --epochs 3

    # Train specific model
    python scripts/train_ensemble.py --model router --epochs 3
    python scripts/train_ensemble.py --model language --epochs 3
    python scripts/train_ensemble.py --model reasoning --epochs 3
    python scripts/train_ensemble.py --model verifier --epochs 3

    # Quick test run
    python scripts/train_ensemble.py --all --samples 1000 --epochs 1 --no-wandb
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from src.models import (
    create_model,
    get_config,
    TransformerConfig,
    MODEL_CONFIGS,
)
from src.models.config import (
    create_router_config,
    create_language_specialist_config,
    create_reasoning_specialist_config,
    create_verifier_specialist_config,
    EnsembleConfig,
    create_ensemble_config,
)
from src.data import (
    create_tokenizer,
    create_dataloaders,
)
from src.training import TrainingConfig, Trainer


# =============================================================================
# Specialist Model Definitions
# =============================================================================

SPECIALIST_MODELS = {
    "router": {
        "config_fn": create_router_config,
        "description": "Intent classification and routing",
        "data_focus": ["intent", "classification"],
        "recommended_epochs": 5,
        "recommended_lr": 5e-4,
    },
    "language": {
        "config_fn": create_language_specialist_config,
        "description": "Prompt interpretation, synthesis, output",
        "data_focus": ["general", "conversation", "synthesis"],
        "recommended_epochs": 3,
        "recommended_lr": 1e-4,
    },
    "reasoning": {
        "config_fn": create_reasoning_specialist_config,
        "description": "Math, logic, chain-of-thought, tools",
        "data_focus": ["math", "logic", "reasoning", "tools"],
        "recommended_epochs": 3,
        "recommended_lr": 1e-4,
    },
    "verifier": {
        "config_fn": create_verifier_specialist_config,
        "description": "Answer validation, error detection",
        "data_focus": ["verification", "validation"],
        "recommended_epochs": 5,
        "recommended_lr": 3e-4,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Svend Ensemble Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all 4 specialists
  python scripts/train_ensemble.py --all --epochs 3

  # Train just the reasoning specialist
  python scripts/train_ensemble.py --model reasoning --epochs 3

  # Quick pipeline test
  python scripts/train_ensemble.py --all --samples 500 --epochs 1 --no-wandb

  # Resume training
  python scripts/train_ensemble.py --model reasoning --resume checkpoints/reasoning/latest
        """
    )

    # Model selection
    parser.add_argument("--model", type=str, choices=list(SPECIALIST_MODELS.keys()),
                        help="Train a specific specialist model")
    parser.add_argument("--all", action="store_true",
                        help="Train all specialist models sequentially")

    # Data
    parser.add_argument("--samples", type=int, default=None,
                        help="Limit training samples (default: all)")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Maximum sequence length (default: 2048)")

    # Training
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=None,
                        help="Peak learning rate (uses model default if not specified)")
    parser.add_argument("--warmup-ratio", type=float, default=0.05,
                        help="Warmup ratio of total steps")

    # Efficiency
    parser.add_argument("--no-gradient-checkpointing", action="store_true",
                        help="Disable gradient checkpointing")
    parser.add_argument("--no-bf16", action="store_true",
                        help="Disable bfloat16 (uses fp16 instead)")

    # Output
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Base output directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")

    # Logging
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="svend-ensemble",
                        help="WandB project name")

    return parser.parse_args()


def print_banner():
    print("""
+---------------------------------------------------------------+
|                                                               |
|   SVEND ENSEMBLE - Multi-Model Specialist Training            |
|                                              svend.ai         |
|                                                               |
|   Router (125M) -> Language (500M) + Reasoning (500M)         |
|                           |                                   |
|                    Verifier (250M)                            |
|                                                               |
+---------------------------------------------------------------+
    """)


def print_ensemble_summary():
    """Print summary of all ensemble models."""
    ensemble = create_ensemble_config()
    ensemble.print_summary()


def check_hardware() -> str:
    """Check and report hardware capabilities."""
    print("\n" + "=" * 60)
    print("Hardware Check")
    print("=" * 60)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name}")
        print(f"  Memory: {gpu_memory:.1f} GB")

        if "A100" in gpu_name:
            print("  [OK] A100 detected - can train all specialists easily!")
            return "a100"
        elif gpu_memory > 20:
            print("  [OK] Large GPU - can train all specialists")
            return "large-gpu"
        elif gpu_memory > 10:
            print("  [OK] Medium GPU - can train specialists individually")
            return "medium-gpu"
        else:
            print("  [WARN] Smaller GPU - may need smaller batch sizes")
            return "small-gpu"
    else:
        print("  [X] No GPU detected - training will be very slow")
        return "cpu"


def get_data_for_specialist(specialist: str, tokenizer, max_seq_length: int, num_samples: Optional[int] = None):
    """Get appropriate training data for a specialist model."""
    from src.data import DatasetConfig, create_combined_dataset, ReasoningDataset

    # Different data focus per specialist
    data_focus = SPECIALIST_MODELS[specialist]["data_focus"]

    print(f"\n  Loading data for {specialist} specialist...")
    print(f"  Focus areas: {data_focus}")

    # For now, use the combined dataset for all
    # In production, would filter by domain
    config = DatasetConfig(max_seq_length=max_seq_length)

    try:
        dataset = create_combined_dataset(config)

        if num_samples and len(dataset) > num_samples:
            dataset = dataset.select(range(num_samples))

        # Split
        train_size = int(0.95 * len(dataset))
        train_data = dataset.select(range(train_size))
        val_data = dataset.select(range(train_size, len(dataset)))

        print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

        train_dataset = ReasoningDataset(train_data, tokenizer, max_length=max_seq_length)
        val_dataset = ReasoningDataset(val_data, tokenizer, max_length=max_seq_length)

        return train_dataset, val_dataset

    except Exception as e:
        print(f"  [WARN] Could not load full dataset: {e}")
        print("  Using synthetic placeholder data for testing...")

        # Create minimal synthetic data for testing
        from torch.utils.data import Dataset

        class SyntheticDataset(Dataset):
            def __init__(self, size: int, seq_len: int, vocab_size: int):
                self.size = size
                self.seq_len = seq_len
                self.vocab_size = vocab_size

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.randint(0, self.vocab_size, (self.seq_len,)),
                    "attention_mask": torch.ones(self.seq_len),
                    "labels": torch.randint(0, self.vocab_size, (self.seq_len,)),
                }

        train_dataset = SyntheticDataset(num_samples or 1000, max_seq_length, 32000)
        val_dataset = SyntheticDataset(100, max_seq_length, 32000)

        return train_dataset, val_dataset


def train_specialist(
    specialist: str,
    args,
    hardware: str,
) -> Dict[str, Any]:
    """Train a single specialist model."""

    spec_info = SPECIALIST_MODELS[specialist]
    config_fn = spec_info["config_fn"]

    print(f"\n{'=' * 60}")
    print(f"Training: {specialist.upper()} Specialist")
    print(f"{'=' * 60}")
    print(f"  Description: {spec_info['description']}")

    # Create model config
    config = config_fn()
    config.gradient_checkpointing = not args.no_gradient_checkpointing

    print(f"\n  Model: {config.name}")
    print(f"  Parameters: {config.num_parameters() / 1e6:.0f}M")
    print(f"  Context: {config.max_position_embeddings}")
    print(f"  Tool calling: {config.tool_calling}")

    mem = config.memory_footprint()
    print(f"  Estimated training memory: {mem['total_training_gb']:.1f} GB")

    # Create model
    print("\n  Creating model...")
    model = create_model(config)
    print(f"  Created with {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    # Create tokenizer
    print("\n  Loading tokenizer...")
    tokenizer = create_tokenizer(
        base_tokenizer="mistralai/Mistral-7B-v0.1",
        vocab_size=config.vocab_size,
        add_reasoning_tokens=True,
    )
    print(f"  Vocabulary: {len(tokenizer)} tokens")

    # Resize embeddings if needed
    if len(tokenizer) > config.vocab_size:
        print(f"  Resizing embeddings: {config.vocab_size} -> {len(tokenizer)}")
        model.embed_tokens = torch.nn.Embedding(len(tokenizer), config.hidden_size)

    # Load data
    train_dataset, val_dataset = get_data_for_specialist(
        specialist, tokenizer, args.max_seq_length, args.samples
    )

    # Create dataloaders
    dataloaders = create_dataloaders(
        train_dataset, val_dataset,
        batch_size=args.batch_size,
        num_workers=2,
    )

    # Training config
    lr = args.lr or spec_info["recommended_lr"]
    epochs = args.epochs

    output_dir = Path(args.output_dir) / specialist
    output_dir.mkdir(parents=True, exist_ok=True)

    training_config = TrainingConfig(
        num_epochs=epochs,
        learning_rate=lr,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        mixed_precision=True,
        bf16=not args.no_bf16,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        output_dir=str(output_dir),
        save_steps=500,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=f"{specialist}-{datetime.now().strftime('%Y%m%d-%H%M')}",
    )

    print(f"\n  Training config:")
    print(f"    Epochs: {epochs}")
    print(f"    Learning rate: {lr}")
    print(f"    Batch size: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"    Output: {output_dir}")

    # Create trainer
    trainer = Trainer(
        model=model,
        config=training_config,
        train_dataloader=dataloaders["train"],
        eval_dataloader=dataloaders.get("val"),
    )

    # Train!
    print(f"\n  Starting training...")
    results = trainer.train()

    print(f"\n  Training complete!")
    print(f"    Total steps: {results.get('total_steps', 'N/A')}")
    print(f"    Final loss: {results.get('final_loss', 'N/A')}")
    print(f"    Checkpoint: {output_dir}")

    # Save final model
    final_path = output_dir / "final"
    final_path.mkdir(exist_ok=True)
    torch.save(model.state_dict(), final_path / "model.pt")
    config.save(str(final_path / "config.json"))
    tokenizer.save_pretrained(str(final_path))

    print(f"  Final model saved to: {final_path}")

    return {
        "specialist": specialist,
        "results": results,
        "output_dir": str(output_dir),
    }


def main():
    args = parse_args()
    print_banner()

    if not args.model and not args.all:
        print("Error: Specify --model <name> or --all")
        print("\nAvailable specialists:")
        for name, info in SPECIALIST_MODELS.items():
            config = info["config_fn"]()
            print(f"  {name:12s} ({config.num_parameters()/1e6:.0f}M) - {info['description']}")
        print("\nUse --all to train all specialists sequentially")
        sys.exit(1)

    # Show ensemble summary
    print_ensemble_summary()

    # Check hardware
    hardware = check_hardware()

    # Determine which models to train
    if args.all:
        models_to_train = list(SPECIALIST_MODELS.keys())
        print(f"\n[INFO] Training all {len(models_to_train)} specialists: {models_to_train}")
    else:
        models_to_train = [args.model]

    # Train each model
    all_results = []
    for specialist in models_to_train:
        results = train_specialist(specialist, args, hardware)
        all_results.append(results)

    # Summary
    print("\n" + "=" * 60)
    print("ENSEMBLE TRAINING COMPLETE")
    print("=" * 60)

    for result in all_results:
        spec = result["specialist"]
        print(f"\n  {spec.upper()}:")
        print(f"    Output: {result['output_dir']}")
        if "results" in result and result["results"]:
            r = result["results"]
            print(f"    Steps: {r.get('total_steps', 'N/A')}")
            print(f"    Loss: {r.get('final_loss', 'N/A')}")

    print("\n[DONE] All specialists trained!")
    print("\nNext steps:")
    print("  1. Evaluate each specialist: python scripts/evaluate_models.py --ensemble")
    print("  2. Run inference: python -m src.server.api --ensemble checkpoints/")


if __name__ == "__main__":
    main()
