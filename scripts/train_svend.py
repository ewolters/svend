#!/usr/bin/env python3
"""
Train Svend Reasoning Models.

Trains both the main reasoner (7B/13B) and optional verifier (3B).
Designed for A100 80GB - can train 13B from scratch.

Usage:
    # Train 13B reasoner (recommended for production)
    python scripts/train_svend.py --model-size 13b --epochs 3

    # Train 7B reasoner (faster, still good)
    python scripts/train_svend.py --model-size 7b --epochs 3

    # Train with verifier
    python scripts/train_svend.py --model-size 13b --train-verifier

    # Quick test run
    python scripts/train_svend.py --model-size 1b --samples 1000 --epochs 1
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from src.models import (
    create_model,
    get_config,
    TransformerConfig,
    MODEL_CONFIGS,
)
from src.data import (
    DatasetConfig,
    create_combined_dataset,
    ReasoningDataset,
    create_dataloaders,
    create_tokenizer,
)
from src.training import TrainingConfig, Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Svend Reasoning Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full 13B training on A100 80GB
  python scripts/train_svend.py --model-size 13b --epochs 3

  # Quick iteration with 7B
  python scripts/train_svend.py --model-size 7b --samples 50000 --epochs 1

  # Test the pipeline
  python scripts/train_svend.py --model-size 500m --samples 1000 --epochs 1 --no-wandb
        """
    )

    # Model
    parser.add_argument("--model-size", type=str, default="13b",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model size to train (default: 13b)")
    parser.add_argument("--train-verifier", action="store_true",
                        help="Also train a 3B verifier model")

    # Data
    parser.add_argument("--samples", type=int, default=None,
                        help="Limit training samples (default: all ~150k)")
    parser.add_argument("--max-seq-length", type=int, default=4096,
                        help="Maximum sequence length (default: 4096)")

    # Training
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Per-device batch size (adjust based on GPU)")
    parser.add_argument("--grad-accum", type=int, default=32,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Peak learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.03,
                        help="Warmup ratio of total steps")
    parser.add_argument("--weight-decay", type=float, default=0.1,
                        help="Weight decay")

    # Efficiency
    parser.add_argument("--no-gradient-checkpointing", action="store_true",
                        help="Disable gradient checkpointing (uses more memory)")
    parser.add_argument("--no-bf16", action="store_true",
                        help="Disable bfloat16 (uses fp16 instead)")

    # Output
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: checkpoints/svend-{size}-{date})")
    parser.add_argument("--save-steps", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--save-total-limit", type=int, default=3,
                        help="Keep only N most recent checkpoints")

    # Logging
    parser.add_argument("--wandb-project", type=str, default="svend",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="Custom run name for wandb")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")

    return parser.parse_args()


def print_banner():
    print("""
+---------------------------------------------------------------+
|                                                               |
|   SVEND - Tool-Augmented Reasoning System                     |
|                                              svend.ai         |
|                                                               |
+---------------------------------------------------------------+
    """)


def check_hardware():
    """Check and report hardware capabilities."""
    print("\n" + "="*60)
    print("Hardware Check")
    print("="*60)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name}")
        print(f"  Memory: {gpu_memory:.1f} GB")

        if "A100" in gpu_name:
            if gpu_memory > 70:
                print("  [OK] A100 80GB detected - can train 13B from scratch!")
                return "a100-80gb"
            else:
                print("  [OK] A100 40GB detected - can train 7B comfortably")
                return "a100-40gb"
        elif gpu_memory > 20:
            print("  [OK] Large GPU - can train up to 7B with checkpointing")
            return "large-gpu"
        else:
            print("  [WARN] Smaller GPU - recommend 3B or smaller")
            return "small-gpu"
    else:
        print("  [X] No GPU detected - training will be extremely slow")
        return "cpu"


def get_recommended_settings(hardware: str, model_size: str) -> dict:
    """Get recommended training settings based on hardware and model."""
    settings = {
        "a100-80gb": {
            "13b": {"batch_size": 2, "grad_accum": 16, "seq_len": 4096},
            "7b": {"batch_size": 4, "grad_accum": 8, "seq_len": 4096},
            "3b": {"batch_size": 8, "grad_accum": 4, "seq_len": 4096},
        },
        "a100-40gb": {
            "13b": {"batch_size": 1, "grad_accum": 32, "seq_len": 2048},
            "7b": {"batch_size": 2, "grad_accum": 16, "seq_len": 4096},
            "3b": {"batch_size": 4, "grad_accum": 8, "seq_len": 4096},
        },
        "large-gpu": {
            "7b": {"batch_size": 1, "grad_accum": 32, "seq_len": 2048},
            "3b": {"batch_size": 2, "grad_accum": 16, "seq_len": 4096},
            "1b": {"batch_size": 4, "grad_accum": 8, "seq_len": 4096},
        },
    }

    hw_settings = settings.get(hardware, {})
    return hw_settings.get(model_size, {"batch_size": 1, "grad_accum": 32, "seq_len": 2048})


def main():
    args = parse_args()

    print_banner()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Check hardware
    hardware = check_hardware()

    # Get recommended settings
    recommended = get_recommended_settings(hardware, args.model_size)
    if args.batch_size == 1 and recommended.get("batch_size", 1) > 1:
        print(f"\n  Tip: Your hardware can handle batch_size={recommended['batch_size']}")

    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d")
        args.output_dir = f"checkpoints/svend-{args.model_size}-{timestamp}"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Create model configuration
    print("\n" + "="*60)
    print("1. Model Configuration")
    print("="*60)

    config = get_config(args.model_size)
    config.gradient_checkpointing = not args.no_gradient_checkpointing

    print(f"  Model: {config.name}")
    print(f"  Type: {config.model_type}")
    print(f"  Parameters: {config.num_parameters() / 1e9:.2f}B")
    print(f"  Context: {config.max_position_embeddings // 1024}K tokens")
    print(f"  Tool calling: {config.tool_calling}")

    memory = config.memory_footprint(
        dtype_bytes=2,
        batch_size=args.batch_size,
        seq_len=args.max_seq_length,
    )
    print(f"  Estimated training memory: {memory['total_training_gb']:.1f} GB")

    # Create model
    print("\n" + "="*60)
    print("2. Initialize Model")
    print("="*60)

    model = create_model(config)
    print(f"  Created model with {model.num_parameters() / 1e9:.2f}B parameters")
    print(f"  Gradient checkpointing: {config.gradient_checkpointing}")

    # Load tokenizer
    print("\n" + "="*60)
    print("3. Tokenizer")
    print("="*60)

    tokenizer = create_tokenizer(
        base_tokenizer="mistralai/Mistral-7B-v0.1",
        vocab_size=config.vocab_size,
        add_reasoning_tokens=True,
    )
    print(f"  Base vocabulary: {config.vocab_size}")
    print(f"  With special tokens: {len(tokenizer)}")
    print(f"  Tool tokens: {config.num_tool_tokens}")

    # Resize embeddings if needed
    total_vocab = config.total_vocab_size
    if len(tokenizer) > config.vocab_size:
        total_vocab = len(tokenizer)
    if total_vocab != config.vocab_size:
        print(f"  Resizing embeddings: {config.vocab_size} -> {total_vocab}")
        model.embed_tokens = torch.nn.Embedding(total_vocab, config.hidden_size)
        if model.lm_head is not None:
            model.lm_head = torch.nn.Linear(config.hidden_size, total_vocab, bias=False)

    # Load data
    print("\n" + "="*60)
    print("4. Training Data")
    print("="*60)

    data_config = DatasetConfig(
        max_seq_length=args.max_seq_length,
        seed=args.seed,
    )

    if args.samples:
        samples_per_source = args.samples // len(data_config.sources)
        for source in data_config.sources:
            source["sample_size"] = min(
                source.get("sample_size", samples_per_source),
                samples_per_source
            )
        print(f"  Limiting to ~{args.samples} samples")

    dataset = create_combined_dataset(data_config)

    train_size = int(0.98 * len(dataset))
    train_data = dataset.select(range(train_size))
    val_data = dataset.select(range(train_size, len(dataset)))

    print(f"  Total examples: {len(dataset)}")
    print(f"  Train: {len(train_data)}")
    print(f"  Validation: {len(val_data)}")

    train_dataset = ReasoningDataset(train_data, tokenizer, max_length=args.max_seq_length)
    val_dataset = ReasoningDataset(val_data, tokenizer, max_length=args.max_seq_length)

    dataloaders = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
    )

    # Training configuration
    print("\n" + "="*60)
    print("5. Training Configuration")
    print("="*60)

    # Calculate warmup steps
    steps_per_epoch = len(dataloaders["train"]) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    training_config = TrainingConfig(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=warmup_steps,
        mixed_precision=True,
        bf16=not args.no_bf16,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name or f"svend-{args.model_size}",
        resume_from=args.resume,
    )

    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.grad_accum}")
    print(f"  Effective batch size: {training_config.effective_batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Total steps: {total_steps}")
    print(f"  Precision: {'bf16' if not args.no_bf16 else 'fp16'}")
    print(f"  Output: {args.output_dir}")

    # Create trainer
    trainer = Trainer(
        model=model,
        config=training_config,
        train_dataloader=dataloaders["train"],
        eval_dataloader=dataloaders.get("val"),
    )

    # Train
    print("\n" + "="*60)
    print("6. Training")
    print("="*60)

    results = trainer.train()

    # Summary
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"  Total steps: {results['total_steps']}")
    print(f"  Final loss: {results['final_loss']:.4f}")
    print(f"  Training time: {results['training_time']/3600:.2f} hours")
    print(f"  Checkpoints: {args.output_dir}")

    # Save config
    config.save(f"{args.output_dir}/final/config.json")
    print(f"  Config saved to: {args.output_dir}/final/config.json")

    # Optionally train verifier
    if args.train_verifier:
        print("\n" + "="*60)
        print("7. Training Verifier (3B)")
        print("="*60)
        # Would train verifier here
        print("  Verifier training not yet implemented")
        print("  Use --model-size 3b-verifier separately for now")

    print("\n[DONE] Your Svend model is ready.")
    print(f"  To serve: python -m src.server.api --model-path {args.output_dir}/final")


if __name__ == "__main__":
    main()
