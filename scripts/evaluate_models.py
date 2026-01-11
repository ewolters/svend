#!/usr/bin/env python3
"""
Evaluate and compare teacher/student models.

Usage:
    # Evaluate single model
    python scripts/evaluate_models.py --model-path checkpoints/student_500m

    # Compare teacher and student
    python scripts/evaluate_models.py \
        --teacher-path checkpoints/teacher_7b/final \
        --student-path checkpoints/student_500m \
        --compare
"""

import os
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from src.models import (
    create_model,
    create_7b_config,
    create_500m_config,
    TransformerConfig,
)
from src.data import create_tokenizer
from src.evaluation import (
    BenchmarkConfig,
    ModelEvaluator,
    compare_models,
    InferenceBenchmark,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate reasoning models")

    # Models
    parser.add_argument("--model-path", type=str,
                        help="Path to model checkpoint")
    parser.add_argument("--model-size", type=str, default="500m",
                        choices=["500m", "1b", "3b", "7b"],
                        help="Model size (for loading config)")

    # Comparison mode
    parser.add_argument("--compare", action="store_true",
                        help="Compare teacher and student")
    parser.add_argument("--teacher-path", type=str,
                        help="Path to teacher checkpoint")
    parser.add_argument("--student-path", type=str,
                        help="Path to student checkpoint")

    # Benchmarks
    parser.add_argument("--benchmarks", type=str, nargs="+",
                        default=["gsm8k", "reasoning_custom"],
                        help="Benchmarks to run")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples per benchmark")

    # Output
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                        help="Output directory")

    return parser.parse_args()


def load_model(path: str, size: str):
    """Load a model from checkpoint."""
    config_creators = {
        "500m": lambda: __import__("src.models", fromlist=["create_500m_config"]).create_500m_config(),
        "1b": lambda: __import__("src.models", fromlist=["create_1b_config"]).create_1b_config(),
        "3b": lambda: __import__("src.models", fromlist=["create_3b_config"]).create_3b_config(),
        "7b": lambda: __import__("src.models", fromlist=["create_7b_config"]).create_7b_config(),
    }

    config = config_creators[size]()
    model = create_model(config)

    checkpoint_path = Path(path)
    if (checkpoint_path / "model.pt").exists():
        state_dict = torch.load(checkpoint_path / "model.pt", map_location="cpu")
        model.load_state_dict(state_dict)
    elif checkpoint_path.suffix == ".pt":
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {path}")

    return model, config


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer
    tokenizer = create_tokenizer(
        base_tokenizer="meta-llama/Llama-2-7b-hf",
        vocab_size=32000,
    )

    # Benchmark config
    bench_config = BenchmarkConfig(
        benchmarks=args.benchmarks,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
    )

    if args.compare:
        # Comparison mode
        if not args.teacher_path or not args.student_path:
            print("Error: --teacher-path and --student-path required for comparison")
            sys.exit(1)

        print("Loading teacher model...")
        teacher, _ = load_model(args.teacher_path, "7b")

        print("Loading student model...")
        student, _ = load_model(args.student_path, "500m")

        print("\nComparing models...")
        results = compare_models(teacher, student, tokenizer, bench_config)

        # Print key retention metrics
        print("\n" + "="*60)
        print("Distillation Quality Summary")
        print("="*60)

        for key, vals in results["comparison"].items():
            if isinstance(vals, dict) and "retention" in vals:
                retention = vals["retention"]
                status = "Excellent" if retention > 0.9 else "Good" if retention > 0.7 else "Needs improvement"
                print(f"{key}: {retention:.1%} retention ({status})")

    else:
        # Single model evaluation
        if not args.model_path:
            print("Error: --model-path required for single model evaluation")
            sys.exit(1)

        print(f"Loading model from {args.model_path}...")
        model, config = load_model(args.model_path, args.model_size)

        print(f"Model size: {config.num_parameters() / 1e9:.2f}B parameters")

        evaluator = ModelEvaluator(model, tokenizer, bench_config, device)
        results = evaluator.run_all_benchmarks()

        # Print summary
        print("\n" + "="*60)
        print("Evaluation Summary")
        print("="*60)

        for key, value in results["metrics"].items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            elif isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\nDetailed results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
