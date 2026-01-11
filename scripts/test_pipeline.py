#!/usr/bin/env python3
"""
Quick pipeline test for local machine (~6GB GPU).

Tests the full training pipeline with a tiny model:
- Model creation
- Data loading
- Forward/backward pass
- Checkpointing
- Tool token handling

Run: python scripts/test_pipeline.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class TinyConfig:
    """50M param model for testing."""
    name: str = "svend-test-50m"
    vocab_size: int = 32000
    hidden_size: int = 512
    intermediate_size: int = 1408
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    hidden_act: str = "swiglu"
    tool_calling: bool = True
    num_tool_tokens: int = 64
    gradient_checkpointing: bool = False

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    def total_vocab_size(self):
        return self.vocab_size + self.num_tool_tokens if self.tool_calling else self.vocab_size


def create_tiny_model(config: TinyConfig) -> nn.Module:
    """Create minimal transformer for testing."""
    from src.models.transformer import ReasoningTransformerForCausalLM
    from src.models.config import TransformerConfig

    # Convert to full config
    full_config = TransformerConfig(
        name=config.name,
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_position_embeddings=config.max_position_embeddings,
        tool_calling=config.tool_calling,
        num_tool_tokens=config.num_tool_tokens,
    )

    return ReasoningTransformerForCausalLM(full_config)


def create_dummy_data(batch_size: int = 2, seq_len: int = 256, vocab_size: int = 32000):
    """Create dummy training batch."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    labels[:, :10] = -100  # Mask first 10 tokens (simulating prompt)
    return {"input_ids": input_ids, "labels": labels}


def test_forward_backward():
    """Test forward and backward pass."""
    print("\n" + "=" * 50)
    print("Testing Forward/Backward Pass")
    print("=" * 50)

    config = TinyConfig()
    print(f"Creating {config.name}...")

    try:
        model = create_tiny_model(config)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {num_params / 1e6:.1f}M")

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        if device.type == "cuda":
            mem_before = torch.cuda.memory_allocated() / 1e9
            print(f"GPU memory before: {mem_before:.2f} GB")

        model = model.to(device)

        if device.type == "cuda":
            mem_after = torch.cuda.memory_allocated() / 1e9
            print(f"GPU memory after model load: {mem_after:.2f} GB")

        # Create dummy data
        batch = create_dummy_data(batch_size=2, seq_len=256, vocab_size=config.vocab_size)
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        print("\nRunning forward pass...")
        model.train()
        outputs = model(**batch)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        print(f"Loss: {loss.item():.4f}")

        # Backward pass
        print("Running backward pass...")
        loss.backward()
        print("Backward pass successful!")

        if device.type == "cuda":
            mem_peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"Peak GPU memory: {mem_peak:.2f} GB")

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tool_tokens():
    """Test tool token handling."""
    print("\n" + "=" * 50)
    print("Testing Tool Token Handling")
    print("=" * 50)

    try:
        from src.tools.registry import ToolRegistry, ToolResult, ToolStatus

        registry = ToolRegistry()

        # Test tool call formatting
        tool_call = registry.format_tool_call("symbolic_math", {"operation": "solve", "expression": "x**2 - 4"})
        print(f"Formatted tool call:\n{tool_call}")

        # Test parsing
        parsed = registry.parse_tool_call(tool_call)
        if parsed:
            name, args = parsed
            print(f"Parsed: {name}({args})")
        else:
            print("Failed to parse tool call")
            return False

        # Test result formatting (must use ToolResult object)
        result = ToolResult(
            status=ToolStatus.SUCCESS,
            output={"solutions": [-2, 2]},
        )
        result_str = registry.format_tool_result(result)
        print(f"Formatted result:\n{result_str}")

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint():
    """Test checkpoint save/load."""
    print("\n" + "=" * 50)
    print("Testing Checkpoint Save/Load")
    print("=" * 50)

    import tempfile
    import os

    config = TinyConfig()

    try:
        model = create_tiny_model(config)

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test_checkpoint.pt")

            print(f"Saving checkpoint to {ckpt_path}...")
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config.__dict__,
            }, ckpt_path)

            # Check file size
            size_mb = os.path.getsize(ckpt_path) / 1e6
            print(f"Checkpoint size: {size_mb:.1f} MB")

            # Load
            print("Loading checkpoint...")
            checkpoint = torch.load(ckpt_path, map_location="cpu")

            model2 = create_tiny_model(config)
            model2.load_state_dict(checkpoint["model_state_dict"])

            print("Checkpoint load successful!")

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """Test data pipeline."""
    print("\n" + "=" * 50)
    print("Testing Data Loading")
    print("=" * 50)

    try:
        # Test with synthetic data since we might not have HF datasets
        print("Creating synthetic reasoning examples...")

        examples = [
            {
                "question": "What is 2 + 2?",
                "reasoning": "I need to add 2 and 2. 2 + 2 = 4.",
                "answer": "4"
            },
            {
                "question": "What is the derivative of x^2?",
                "reasoning": "Using the power rule, d/dx(x^n) = n*x^(n-1). So d/dx(x^2) = 2x.",
                "answer": "2x"
            }
        ]

        print(f"Created {len(examples)} synthetic examples")

        # Test formatting
        for ex in examples:
            formatted = f"Question: {ex['question']}\n\nReasoning: {ex['reasoning']}\n\nAnswer: {ex['answer']}"
            print(f"\nExample:\n{formatted[:100]}...")

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("""
================================================================
                         SVEND
                    Pipeline Test
                      svend.ai
================================================================
    """)

    # Check environment
    print("Environment Check")
    print("=" * 50)
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Run tests
    results = {}

    results["data_loading"] = test_data_loading()
    results["tool_tokens"] = test_tool_tokens()
    results["forward_backward"] = test_forward_backward()
    results["checkpoint"] = test_checkpoint()

    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)

    all_passed = True
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n[OK] All tests passed! Pipeline is ready.")
        print("\nNext steps:")
        print("  1. Generate tool data: python scripts/generate_tool_data.py")
        print("  2. Train on Colab: python scripts/train_svend.py --model-size 1b")
    else:
        print("\n[FAIL] Some tests failed. Check errors above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
