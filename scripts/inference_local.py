#!/usr/bin/env python3
"""
Local inference script for the 500M student model.

Designed to run on the HP tower or any machine with 8GB+ RAM.
Can run on CPU with reasonable speed for a 500M model.

Usage:
    python scripts/inference_local.py --model-path checkpoints/student_500m

Interactive mode:
    python scripts/inference_local.py --model-path checkpoints/student_500m --interactive
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from src.models import create_model, create_500m_config, TransformerConfig
from src.data import create_tokenizer


def load_model(path: str, device: torch.device):
    """Load the student model."""
    path = Path(path)

    # Load config
    config_path = path / "config.json"
    if config_path.exists():
        config = TransformerConfig.load(str(config_path))
    else:
        config = create_500m_config()

    # Create and load model
    model = create_model(config)

    model_path = path / "model.pt"
    if model_path.exists():
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")

    model.to(device)
    model.eval()

    return model, config


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: torch.device = None,
) -> str:
    """Generate a response to a prompt."""
    device = device or next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Return only the new tokens
    return response[len(prompt):].strip()


def interactive_mode(model, tokenizer, device, args):
    """Run interactive chat mode."""
    print("\n" + "="*60)
    print("Interactive Reasoning Model")
    print("="*60)
    print("Commands:")
    print("  /quit - Exit")
    print("  /temp <value> - Set temperature (current: {:.1f})".format(args.temperature))
    print("  /tokens <value> - Set max tokens (current: {})".format(args.max_tokens))
    print("  /clear - Clear screen")
    print("="*60 + "\n")

    temperature = args.temperature
    max_tokens = args.max_tokens

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                parts = user_input.split()
                cmd = parts[0].lower()

                if cmd == "/quit":
                    print("Goodbye!")
                    break
                elif cmd == "/temp" and len(parts) > 1:
                    temperature = float(parts[1])
                    print(f"Temperature set to {temperature}")
                    continue
                elif cmd == "/tokens" and len(parts) > 1:
                    max_tokens = int(parts[1])
                    print(f"Max tokens set to {max_tokens}")
                    continue
                elif cmd == "/clear":
                    print("\033[2J\033[H")
                    continue
                else:
                    print("Unknown command")
                    continue

            # Add reasoning prompt if not present
            if "step by step" not in user_input.lower() and "think" not in user_input.lower():
                prompt = f"{user_input}\n\nThink step by step."
            else:
                prompt = user_input

            print("\nAssistant: ", end="", flush=True)

            response = generate_response(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                device=device,
            )

            print(response)
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description="Local inference with reasoning model")

    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt to process")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p sampling")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Maximum tokens to generate")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device to use")

    args = parser.parse_args()

    # Select device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model_path}...")
    model, config = load_model(args.model_path, device)
    print(f"Loaded {config.num_parameters() / 1e6:.0f}M parameter model")

    # Load tokenizer
    tokenizer = create_tokenizer(
        base_tokenizer="meta-llama/Llama-2-7b-hf",
        vocab_size=config.vocab_size,
    )

    if args.interactive:
        interactive_mode(model, tokenizer, device, args)

    elif args.prompt:
        print(f"\nPrompt: {args.prompt}")
        print("-" * 40)

        response = generate_response(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )

        print(f"Response:\n{response}")

    else:
        # Run default test prompts
        test_prompts = [
            "What is 25 * 4? Think step by step.",
            "If it's raining, the ground is wet. The ground is wet. Is it definitely raining?",
            "Write a Python function to check if a number is prime.",
        ]

        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            print("-" * 40)

            response = generate_response(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                device=device,
            )

            print(f"Response:\n{response}\n")


if __name__ == "__main__":
    main()
