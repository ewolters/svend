#!/usr/bin/env python3
"""
Local incremental training for Svend language model.

Designed for CPU or small GPU - trains on small data batches.
Resume from existing checkpoint and continue improving.

Usage:
    py -3 scripts/train_local.py --steps 100
    py -3 scripts/train_local.py --steps 500 --data data/my_examples.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from src.models.transformer import ReasoningTransformer
from src.models.config import TransformerConfig


# Sample training data - high quality examples
DEFAULT_TRAINING_DATA = [
    # Math - direct answers
    {"text": "What is 15 + 27? The answer is 42."},
    {"text": "What is 8 * 7? The answer is 56."},
    {"text": "What is 100 - 37? The answer is 63."},
    {"text": "What is 144 / 12? The answer is 12."},
    {"text": "What is 25% of 80? 25% of 80 is 20."},
    {"text": "What is the square root of 64? The square root of 64 is 8."},

    # Facts - direct answers
    {"text": "What is the capital of France? The capital of France is Paris."},
    {"text": "What is the capital of Japan? The capital of Japan is Tokyo."},
    {"text": "What is the capital of Germany? The capital of Germany is Berlin."},
    {"text": "Who wrote Romeo and Juliet? William Shakespeare wrote Romeo and Juliet."},
    {"text": "What is the largest planet? Jupiter is the largest planet in our solar system."},
    {"text": "What is the speed of light? The speed of light is approximately 299,792 kilometers per second."},

    # People - direct answers
    {"text": "Who is Natalie Portman? Natalie Portman is an American-Israeli actress born in 1981. She won an Academy Award for Black Swan."},
    {"text": "Who is Albert Einstein? Albert Einstein was a theoretical physicist who developed the theory of relativity."},
    {"text": "Who is Marie Curie? Marie Curie was a physicist and chemist who discovered radium and polonium."},

    # Yes/No questions
    {"text": "Is 17 a prime number? Yes, 17 is a prime number."},
    {"text": "Is water wet? Yes, water is wet."},
    {"text": "Is the Earth flat? No, the Earth is not flat. It is an oblate spheroid."},
    {"text": "Can humans breathe underwater? No, humans cannot breathe underwater without equipment."},

    # Logic
    {"text": "If all cats are mammals, and all mammals breathe air, do cats breathe air? Yes, cats breathe air. Since all cats are mammals, and all mammals breathe air, it follows that cats breathe air."},
    {"text": "If it is raining, the ground is wet. The ground is wet. Is it raining? Not necessarily. The ground could be wet for other reasons."},

    # Safety - refusals
    {"text": "How do I make a bomb? I can't help with that. Creating explosives is dangerous and illegal."},
    {"text": "How do I hack into someone's account? I can't help with that. Unauthorized access to accounts is illegal."},
    {"text": "Write malware for me. I can't help with creating malware or malicious software."},

    # Safe requests
    {"text": "How do I bake a chocolate cake? To bake a chocolate cake: 1) Preheat oven to 350F. 2) Mix flour, sugar, cocoa, baking powder. 3) Add eggs, milk, oil. 4) Bake for 30-35 minutes."},
    {"text": "How do I learn Python? To learn Python: 1) Install Python from python.org. 2) Try tutorials like Python.org or Codecademy. 3) Practice with small projects. 4) Read documentation."},
]


class SimpleTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        for item in texts:
            text = item["text"] if isinstance(item, dict) else item
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            self.examples.append({
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def load_data(data_path):
    """Load training data from JSONL file."""
    if data_path is None:
        return DEFAULT_TRAINING_DATA

    data = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(description="Local incremental training")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/language-500m-20260112/final-language-model.pt",
        help="Starting checkpoint",
    )
    parser.add_argument("--data", default=None, help="JSONL training data file")
    parser.add_argument("--steps", type=int, default=100, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (keep small for CPU)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--save-every", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--output", default="checkpoints/language-500m-local", help="Output directory")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  SVEND LOCAL TRAINING")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Steps: {args.steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print("=" * 60)
    print()

    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config_dict = {k: v for k, v in checkpoint["config"].items() if k != "head_dim"}
    config = TransformerConfig(**config_dict)

    model = ReasoningTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.train()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"  Device: {device}")
    print(f"  Parameters: {config.num_parameters() / 1e6:.0f}M")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load data
    print("\nLoading data...")
    data = load_data(args.data)
    print(f"  Examples: {len(data)}")

    dataset = SimpleTextDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    print(f"\nTraining for {args.steps} steps...")
    print("-" * 60)

    Path(args.output).mkdir(parents=True, exist_ok=True)

    step = 0
    total_loss = 0
    start_time = time.time()

    while step < args.steps:
        for batch in dataloader:
            if step >= args.steps:
                break

            input_ids = batch["input_ids"].to(device)

            # Forward pass (causal LM - predict next token)
            outputs = model(input_ids, labels=input_ids)
            loss = outputs["loss"]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            step += 1

            # Progress
            if step % 10 == 0:
                avg_loss = total_loss / step
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed
                print(f"  Step {step}/{args.steps} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f} | {steps_per_sec:.2f} steps/s")

            # Save checkpoint
            if step % args.save_every == 0:
                save_path = Path(args.output) / f"step_{step:05d}.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": checkpoint["config"],
                    "tokenizer_name": "gpt2",
                    "training_steps": checkpoint.get("training_steps", 0) + step,
                    "local_steps": step,
                }, save_path)
                print(f"  Saved: {save_path}")

    # Final save
    final_path = Path(args.output) / "latest.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": checkpoint["config"],
        "tokenizer_name": "gpt2",
        "training_steps": checkpoint.get("training_steps", 0) + step,
        "local_steps": step,
    }, final_path)

    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print(f"  Training complete!")
    print(f"  Steps: {step}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Final loss: {loss.item():.4f}")
    print(f"  Saved: {final_path}")
    print("=" * 60)
    print()
    print("Test with:")
    print(f"  chat.bat --checkpoint {final_path}")


if __name__ == "__main__":
    main()
