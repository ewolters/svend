#!/usr/bin/env python3
"""
Interactive chat with Svend language model.

Usage:
    py -3 scripts/chat_interactive.py
    py -3 scripts/chat_interactive.py --checkpoint path/to/model.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Interactive chat with Svend=374M 150K model")
    parser.add_argument(
        "--checkpoint", "-c",
        default="checkpoints/final-language-model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument("--max-tokens", type=int, default=150, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  SVEND LANGUAGE MODEL - Interactive Chat")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print("  Commands: 'quit' to exit, 'temp X' to change temperature")
    print("=" * 60)
    print()

    # Load model
    print("Loading model...")
    import torch
    from transformers import AutoTokenizer
    from src.models.transformer import ReasoningTransformer
    from src.models.config import TransformerConfig

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config_dict = {k: v for k, v in checkpoint["config"].items() if k != "head_dim"}
    config = TransformerConfig(**config_dict)

    model = ReasoningTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.cuda()
        print(f"  Model loaded on GPU (CUDA)")
    else:
        print(f"  Model loaded on CPU")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    params = config.num_parameters()
    print(f"  Parameters: {params / 1e6:.0f}M")
    print(f"  Temperature: {args.temperature}")
    print()
    print("Ready! Type your prompt and press Enter.")
    print("-" * 60)

    temperature = args.temperature

    while True:
        try:
            prompt = input("\nYou: ").strip()

            if not prompt:
                continue

            # Commands
            if prompt.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if prompt.lower().startswith("temp "):
                try:
                    temperature = float(prompt.split()[1])
                    print(f"Temperature set to {temperature}")
                    continue
                except (IndexError, ValueError):
                    print("Usage: temp 0.7")
                    continue

            # Generate
            inputs = tokenizer(prompt, return_tensors="pt")
            if device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=args.max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    top_k=50,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )

            print(f"\nSvend: {response.strip()}")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
