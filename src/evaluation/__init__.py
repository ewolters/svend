"""
Svend Evaluation Framework

Comprehensive evaluation for reasoning models.

Components:
- benchmarks: Standard benchmarks (GSM8K, MATH, custom reasoning)
- harness: Evaluation suite with validation gates
- metrics: Accuracy, F1, pass@k, coherence metrics

Usage:
    # Quick evaluation for validation gates
    from src.evaluation import quick_eval
    metrics = quick_eval(model, tokenizer)

    # Full evaluation suite
    from src.evaluation import EvaluationSuite
    suite = EvaluationSuite()
    suite.add_full_benchmarks()
    results = suite.run(model, tokenizer)
"""

from .benchmarks import (
    BenchmarkConfig,
    BenchmarkBase,
    GSM8KBenchmark,
    MATHBenchmark,
    ReasoningBenchmark,
    InferenceBenchmark,
    ModelEvaluator,
    compare_models,
)

from .harness import (
    EvaluationSuite,
    EvaluationResult,
    SuiteResult,
    BenchmarkRunner,
    BaseBenchmark,
    ToolAccuracyBenchmark,
    SafetyBenchmark,
    quick_eval,
    full_eval,
)

from .metrics import (
    accuracy,
    exact_match,
    f1_score,
    pass_at_k,
    reasoning_coherence,
    tool_precision_recall,
    numeric_accuracy,
    MetricAggregator,
)

__all__ = [
    # Benchmarks
    "BenchmarkConfig",
    "BenchmarkBase",
    "GSM8KBenchmark",
    "MATHBenchmark",
    "ReasoningBenchmark",
    "InferenceBenchmark",
    "ModelEvaluator",
    "compare_models",
    # Harness
    "EvaluationSuite",
    "EvaluationResult",
    "SuiteResult",
    "BenchmarkRunner",
    "BaseBenchmark",
    "ToolAccuracyBenchmark",
    "SafetyBenchmark",
    "quick_eval",
    "full_eval",
    # Metrics
    "accuracy",
    "exact_match",
    "f1_score",
    "pass_at_k",
    "reasoning_coherence",
    "tool_precision_recall",
    "numeric_accuracy",
    "MetricAggregator",
]
