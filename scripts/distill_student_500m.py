#!/usr/bin/env python3
"""
Distill the 7B teacher to a 500M student model.

This runs after the teacher is trained.
Can run on A100 or even smaller GPUs (24GB+).

Usage:
    python scripts/distill_student_500m.py \
        --teacher-path checkpoints/teacher_7b/final \
        --output-dir checkpoints/student_500m
"""

import os
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoTokenizer

from src.models import (
    create_model,
    create_7b_config,
    create_500m_config,
    ReasoningTransformer,
)
from src.data import (
    DatasetConfig,
    create_combined_dataset,
    ReasoningDataset,
    create_dataloaders,
    create_tokenizer,
)
from src.training import (
    DistillationConfig,
    DistillationTrainer,
    distill_reasoning_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Distill 7B teacher to 500M student")

    # Models
    parser.add_argument("--teacher-path", type=str, required=True,
                        help="Path to trained teacher checkpoint")
    parser.add_argument("--student-size", type=str, default="500m",
                        choices=["500m", "1b"],
                        help="Student model size")

    # Distillation settings
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Distillation temperature")
    parser.add_argument("--alpha-ce", type=float, default=0.5,
                        help="Weight for cross-entropy loss with labels")
    parser.add_argument("--alpha-kl", type=float, default=0.5,
                        help="Weight for KL divergence with teacher")
    parser.add_argument("--alpha-hidden", type=float, default=0.1,
                        help="Weight for hidden state matching")

    # Training
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of distillation epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--samples", type=int, default=None,
                        help="Number of training samples")

    # Output
    parser.add_argument("--output-dir", type=str, default="checkpoints/student_500m",
                        help="Output directory")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    return parser.parse_args()


def load_teacher(path: str, config) -> ReasoningTransformer:
    """Load trained teacher model."""
    model = create_model(config)

    checkpoint_path = Path(path) / "model.pt"
    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"Loaded teacher from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")

    return model


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    print("="*60)
    print("Knowledge Distillation: 7B -> 500M")
    print("="*60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create configurations
    print("\n1. Creating model configurations...")
    teacher_config = create_7b_config()
    if args.student_size == "500m":
        student_config = create_500m_config()
    else:
        from src.models import create_1b_config
        student_config = create_1b_config()

    print(f"   Teacher: {teacher_config.num_parameters() / 1e9:.2f}B parameters")
    print(f"   Student: {student_config.num_parameters() / 1e9:.2f}B parameters")
    print(f"   Compression: {teacher_config.num_parameters() / student_config.num_parameters():.1f}x")

    # Load teacher
    print("\n2. Loading teacher model...")
    teacher = load_teacher(args.teacher_path, teacher_config)
    teacher.to(device)
    teacher.eval()

    # Create student
    print("\n3. Creating student model...")
    student = create_model(student_config)
    student.to(device)
    print(f"   Initialized with {student.num_parameters() / 1e9:.2f}B parameters")

    # Load tokenizer
    print("\n4. Loading tokenizer...")
    tokenizer = create_tokenizer(
        base_tokenizer="meta-llama/Llama-2-7b-hf",
        vocab_size=teacher_config.vocab_size,
    )

    # Load data
    print("\n5. Loading training data...")
    data_config = DatasetConfig(seed=args.seed)
    if args.samples:
        for source in data_config.sources:
            source["sample_size"] = min(
                source.get("sample_size", args.samples),
                args.samples // len(data_config.sources)
            )

    dataset = create_combined_dataset(data_config)
    train_size = int(0.98 * len(dataset))
    train_data = dataset.select(range(train_size))
    val_data = dataset.select(range(train_size, len(dataset)))

    print(f"   Train: {len(train_data)}, Val: {len(val_data)}")

    train_dataset = ReasoningDataset(train_data, tokenizer)
    val_dataset = ReasoningDataset(val_data, tokenizer)

    dataloaders = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
    )

    # Distillation configuration
    print("\n6. Setting up distillation...")
    distill_config = DistillationConfig(
        temperature=args.temperature,
        alpha_ce=args.alpha_ce,
        alpha_kl=args.alpha_kl,
        alpha_hidden=args.alpha_hidden,
        freeze_teacher=True,
    )

    print(f"   Temperature: {distill_config.temperature}")
    print(f"   Loss weights - CE: {distill_config.alpha_ce}, KL: {distill_config.alpha_kl}, Hidden: {distill_config.alpha_hidden}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=0.1,
    )

    # Create trainer
    trainer = DistillationTrainer(
        teacher=teacher,
        student=student,
        config=distill_config,
        train_dataloader=dataloaders["train"],
        eval_dataloader=dataloaders.get("val"),
        optimizer=optimizer,
        device=device,
    )

    # Train
    print("\n7. Starting distillation...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        losses = trainer.train_epoch()
        print(f"  Train losses: {losses}")

        if dataloaders.get("val"):
            eval_metrics = trainer.evaluate()
            print(f"  Eval metrics: {eval_metrics}")

    # Save student
    print("\n8. Saving student model...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(student.state_dict(), output_dir / "model.pt")
    student_config.save(output_dir / "config.json")

    print(f"   Saved to {output_dir}")

    print("\n" + "="*60)
    print("Distillation Complete!")
    print("="*60)
    print(f"Student model: {student_config.num_parameters() / 1e6:.0f}M parameters")
    print(f"Saved to: {args.output_dir}")

    # Quick inference test
    print("\nQuick inference test:")
    student.eval()
    test_prompt = "What is 2 + 2? Think step by step."
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = student.generate(
            inputs.input_ids,
            max_new_tokens=100,
            temperature=0.7,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {test_prompt}")
    print(f"Response: {response[len(test_prompt):]}")


if __name__ == "__main__":
    main()
