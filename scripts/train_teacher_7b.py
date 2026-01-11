#!/usr/bin/env python3
"""
Train the 7B teacher model for reasoning.

This is designed to run on A100 80GB Colab.
Expected training time: ~24-48 hours for full training.

Usage:
    python scripts/train_teacher_7b.py --config configs/teacher_7b.yaml

For Colab:
    !python scripts/train_teacher_7b.py --epochs 1 --samples 50000
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoTokenizer

from src.models import create_model, create_7b_config, TransformerConfig
from src.data import (
    DatasetConfig,
    create_combined_dataset,
    ReasoningDataset,
    create_dataloaders,
    create_tokenizer,
)
from src.training import TrainingConfig, Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train 7B reasoning teacher model")

    # Model
    parser.add_argument("--model-size", type=str, default="7b",
                        choices=["3b", "7b", "13b"],
                        help="Model size to train")

    # Data
    parser.add_argument("--samples", type=int, default=None,
                        help="Number of training samples (None = full dataset)")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Maximum sequence length")

    # Training
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=16,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="Warmup steps")

    # Efficiency
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True,
                        help="Enable gradient checkpointing")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use bfloat16 precision")

    # Output
    parser.add_argument("--output-dir", type=str, default="checkpoints/teacher_7b",
                        help="Output directory for checkpoints")
    parser.add_argument("--save-steps", type=int, default=500,
                        help="Save checkpoint every N steps")

    # Logging
    parser.add_argument("--wandb-project", type=str, default="reasoning-teacher",
                        help="Weights & Biases project name")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("="*60)
    print("Training 7B Reasoning Teacher Model")
    print("="*60)

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("WARNING: No GPU detected. Training will be very slow.")

    # Create model configuration
    print("\n1. Creating model configuration...")
    if args.model_size == "7b":
        config = create_7b_config()
    elif args.model_size == "3b":
        from src.models import create_3b_config
        config = create_3b_config()
    else:
        from src.models import create_13b_config
        config = create_13b_config()

    config.gradient_checkpointing = args.gradient_checkpointing

    print(f"   Model: {config.name}")
    print(f"   Parameters: {config.num_parameters() / 1e9:.2f}B")
    memory = config.memory_footprint(dtype_bytes=2, batch_size=args.batch_size)
    print(f"   Estimated training memory: {memory['total_training_gb']:.1f} GB")

    # Create model
    print("\n2. Initializing model...")
    model = create_model(config)
    print(f"   Created model with {model.num_parameters() / 1e9:.2f}B parameters")

    # Load tokenizer
    print("\n3. Loading tokenizer...")
    tokenizer = create_tokenizer(
        base_tokenizer="meta-llama/Llama-2-7b-hf",
        vocab_size=config.vocab_size,
        add_reasoning_tokens=True,
    )
    print(f"   Vocabulary size: {len(tokenizer)}")

    # Resize embeddings if needed
    if len(tokenizer) != config.vocab_size:
        print(f"   Resizing embeddings: {config.vocab_size} -> {len(tokenizer)}")
        model.embed_tokens = torch.nn.Embedding(len(tokenizer), config.hidden_size)
        if model.lm_head is not None:
            model.lm_head = torch.nn.Linear(config.hidden_size, len(tokenizer), bias=False)

    # Load data
    print("\n4. Loading training data...")
    data_config = DatasetConfig(
        max_seq_length=args.max_seq_length,
        seed=args.seed,
    )

    # Limit samples if specified
    if args.samples:
        for source in data_config.sources:
            source["sample_size"] = min(
                source.get("sample_size", args.samples),
                args.samples // len(data_config.sources)
            )

    dataset = create_combined_dataset(data_config)
    print(f"   Total examples: {len(dataset)}")

    # Create train/val split
    train_size = int(0.98 * len(dataset))
    train_data = dataset.select(range(train_size))
    val_data = dataset.select(range(train_size, len(dataset)))

    print(f"   Train: {len(train_data)}, Val: {len(val_data)}")

    # Create datasets
    train_dataset = ReasoningDataset(
        train_data,
        tokenizer,
        max_length=args.max_seq_length,
    )
    val_dataset = ReasoningDataset(
        val_data,
        tokenizer,
        max_length=args.max_seq_length,
    )

    # Create dataloaders
    dataloaders = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
    )

    # Training configuration
    print("\n5. Setting up training...")
    training_config = TrainingConfig(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=args.warmup_steps,
        mixed_precision=True,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        resume_from=args.resume,
    )

    print(f"   Effective batch size: {training_config.effective_batch_size}")
    print(f"   Learning rate: {training_config.learning_rate}")
    print(f"   Epochs: {training_config.num_epochs}")

    # Create trainer
    trainer = Trainer(
        model=model,
        config=training_config,
        train_dataloader=dataloaders["train"],
        eval_dataloader=dataloaders.get("val"),
    )

    # Train
    print("\n6. Starting training...")
    results = trainer.train()

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Total steps: {results['total_steps']}")
    print(f"Final loss: {results['final_loss']:.4f}")
    print(f"Training time: {results['training_time']/3600:.2f} hours")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
