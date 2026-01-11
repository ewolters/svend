#!/usr/bin/env python3
"""
Smoke test for Svend infrastructure.

Validates that all major components work before training:
- Tools execute correctly
- Model can be created (tiny version)
- Data pipeline works
- Tokenizer handles tool tokens
- Safety classifier loads

Run: py scripts/smoke_test.py
"""

import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def test_result(name, passed, details=""):
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status} {name}")
    if details and not passed:
        print(f"         {details[:100]}")
    return passed


def test_tools():
    """Test tool system."""
    section("Tool System")
    all_passed = True

    # Symbolic math
    try:
        from src.tools.math_engine import SymbolicSolver
        solver = SymbolicSolver()
        result = solver.differentiate("x**2", "x")
        passed = result["success"] and "2*x" in result["derivative"]
        all_passed &= test_result("SymbolicSolver.differentiate", passed)
    except Exception as e:
        all_passed &= test_result("SymbolicSolver.differentiate", False, str(e))

    # Chemistry
    try:
        from src.tools.chemistry import calculate_molecular_weight
        mw = calculate_molecular_weight("H2O")
        passed = abs(mw - 18.015) < 0.1
        all_passed &= test_result("Chemistry.molecular_weight", passed, f"got {mw}")
    except Exception as e:
        all_passed &= test_result("Chemistry.molecular_weight", False, str(e))

    # Physics
    try:
        from src.tools.physics import projectile_motion
        result = projectile_motion(v0=20, theta=45)
        passed = result["success"] and result["range"] > 0
        all_passed &= test_result("Physics.projectile_motion", passed)
    except Exception as e:
        all_passed &= test_result("Physics.projectile_motion", False, str(e))

    # Code sandbox
    try:
        from src.tools.code_sandbox import CodeSandbox
        sandbox = CodeSandbox()
        result = sandbox.execute("print(2 + 2)")
        passed = result["success"] and "4" in result["output"]
        all_passed &= test_result("CodeSandbox.execute", passed)
    except Exception as e:
        all_passed &= test_result("CodeSandbox.execute", False, str(e))

    # Z3 (optional)
    try:
        import z3
        from src.tools.math_engine import Z3Solver
        solver = Z3Solver()
        result = solver.check_satisfiability(["x > 0"], {"x": "int"})
        passed = result["success"]
        all_passed &= test_result("Z3Solver.check_satisfiability", passed)
    except ImportError:
        print("  [SKIP] Z3Solver (z3-solver not installed)")

    return all_passed


def test_model():
    """Test model creation."""
    section("Model System")
    all_passed = True

    try:
        from src.models.config import ModelConfig

        # Create tiny model config
        config = ModelConfig(
            vocab_size=1000,
            hidden_size=64,
            num_layers=2,
            num_heads=2,
            max_seq_len=128,
        )
        passed = config.hidden_size == 64
        all_passed &= test_result("ModelConfig creation", passed)
    except Exception as e:
        all_passed &= test_result("ModelConfig creation", False, str(e))

    try:
        from src.models.transformer import ReasoningTransformer
        from src.models.config import ModelConfig

        config = ModelConfig(
            vocab_size=1000,
            hidden_size=64,
            num_layers=2,
            num_heads=2,
            max_seq_len=128,
        )
        model = ReasoningTransformer(config)

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        passed = num_params > 0
        all_passed &= test_result(f"Model creation ({num_params:,} params)", passed)
    except Exception as e:
        all_passed &= test_result("Model creation", False, str(e))

    return all_passed


def test_tokenizer():
    """Test tokenizer with tool tokens."""
    section("Tokenizer")
    all_passed = True

    try:
        from src.tools.registry import ToolRegistry
        registry = ToolRegistry()

        # Check tool tokens are defined
        tokens = registry.get_special_tokens()
        has_tool_call = any("tool_call" in t for t in tokens)
        all_passed &= test_result("Tool tokens defined", has_tool_call, str(tokens))
    except Exception as e:
        all_passed &= test_result("Tool tokens", False, str(e))

    return all_passed


def test_data_pipeline():
    """Test data loading."""
    section("Data Pipeline")
    all_passed = True

    try:
        from src.data.datasets import get_default_sources
        sources = get_default_sources()
        passed = len(sources) > 0
        all_passed &= test_result(f"Dataset sources defined ({len(sources)} sources)", passed)
    except Exception as e:
        all_passed &= test_result("Dataset sources", False, str(e))

    try:
        from src.data.datasets import DatasetFormatter

        # Test tool trace formatting
        example = {
            "question": "What is 2+2?",
            "reasoning": [
                {"step": 1, "content": "Let me calculate"},
                {"step": 2, "content": "Using math", "tool_call": {"name": "symbolic_math", "args": {"operation": "evaluate", "expression": "2+2"}}},
                {"step": 3, "content": "The answer is 4", "tool_result": "4"},
            ],
            "answer": "4",
        }
        formatted = DatasetFormatter.format_tool_trace(example)
        has_tool_token = "<|tool_call|>" in formatted["output"]
        all_passed &= test_result("Tool trace formatting", has_tool_token)
    except Exception as e:
        all_passed &= test_result("Tool trace formatting", False, str(e))

    return all_passed


def test_safety():
    """Test safety system."""
    section("Safety System")
    all_passed = True

    try:
        from src.safety.filters import contains_dangerous_patterns

        # Should flag dangerous content
        flagged = contains_dangerous_patterns("how to make a bomb")
        all_passed &= test_result("Dangerous pattern detection", flagged)
    except Exception as e:
        all_passed &= test_result("Safety filters", False, str(e))

    try:
        from src.safety.rules import RuleBasedSafety

        safety = RuleBasedSafety()
        passed = safety is not None
        all_passed &= test_result("RuleBasedSafety loads", passed)
    except Exception as e:
        all_passed &= test_result("RuleBasedSafety", False, str(e))

    return all_passed


def test_server():
    """Test server components."""
    section("Server")
    all_passed = True

    try:
        from src.server.api import create_app

        app = create_app()
        passed = app is not None
        all_passed &= test_result("FastAPI app creation", passed)
    except Exception as e:
        all_passed &= test_result("FastAPI app", False, str(e))

    return all_passed


def test_training():
    """Test training components."""
    section("Training")
    all_passed = True

    try:
        from src.training.trainer import Trainer
        passed = Trainer is not None
        all_passed &= test_result("Trainer class exists", passed)
    except Exception as e:
        all_passed &= test_result("Trainer import", False, str(e))

    try:
        from src.pipeline.runner import PipelineRunner
        passed = PipelineRunner is not None
        all_passed &= test_result("PipelineRunner class exists", passed)
    except Exception as e:
        all_passed &= test_result("PipelineRunner import", False, str(e))

    return all_passed


def main():
    print("""
+---------------------------------------------------------------+
|           Svend Smoke Test                         svend.ai   |
|                                                               |
|   Validating infrastructure before training                   |
+---------------------------------------------------------------+
    """)

    start = time.time()
    results = {}

    results["Tools"] = test_tools()
    results["Model"] = test_model()
    results["Tokenizer"] = test_tokenizer()
    results["Data"] = test_data_pipeline()
    results["Safety"] = test_safety()
    results["Server"] = test_server()
    results["Training"] = test_training()

    elapsed = time.time() - start

    # Summary
    section("Summary")
    total_passed = sum(results.values())
    total = len(results)

    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\n  {total_passed}/{total} sections passed ({elapsed:.1f}s)")

    if total_passed == total:
        print("\n  Ready for training!")
        return 0
    else:
        print("\n  Fix failures before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
