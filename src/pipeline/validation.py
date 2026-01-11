"""
Pipeline Validation System

Validates training runs meet quality gates before proceeding.
Prevents wasted compute by catching problems early.

Usage:
    validator = PipelineValidator(config)

    # Quick validation during training
    result = validator.validate_checkpoint(model, step=1000)
    if not result.passed:
        print(f"Validation failed: {result.failures}")

    # Full validation before scaling up
    result = validator.full_validation(model)
    if result.passed:
        proceed_to_next_scale()
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Tuple
from enum import Enum
import json
from pathlib import Path
import time

import torch
import torch.nn as nn


class ValidationLevel(Enum):
    """Validation strictness levels."""
    QUICK = "quick"      # Fast checks during training
    STANDARD = "standard"  # Normal checkpoint validation
    FULL = "full"        # Complete validation before scaling
    PRODUCTION = "production"  # Final production checks


@dataclass
class ValidationResult:
    """Result of a validation run."""

    passed: bool
    level: ValidationLevel
    timestamp: float = field(default_factory=time.time)

    # Detailed results
    metrics: Dict[str, float] = field(default_factory=dict)
    gate_results: Dict[str, bool] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Timing
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "level": self.level.value,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "gate_results": self.gate_results,
            "failures": self.failures,
            "warnings": self.warnings,
            "duration_seconds": self.duration_seconds,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"Validation {status} ({self.level.value})",
            f"Duration: {self.duration_seconds:.1f}s",
            "",
            "Metrics:",
        ]
        for name, value in self.metrics.items():
            lines.append(f"  {name}: {value:.4f}")

        if self.failures:
            lines.append("")
            lines.append("Failures:")
            for f in self.failures:
                lines.append(f"  - {f}")

        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")

        return "\n".join(lines)


@dataclass
class ValidationGate:
    """
    A validation gate that must be passed.

    Gates define minimum/maximum thresholds for metrics.
    Training cannot proceed if gates fail (in strict mode).
    """

    name: str
    metric_name: str
    threshold: float
    comparison: str  # ">=", "<=", ">", "<", "=="
    required: bool = True
    description: str = ""

    def check(self, value: float) -> Tuple[bool, str]:
        """Check if value passes the gate."""
        if self.comparison == ">=":
            passed = value >= self.threshold
        elif self.comparison == "<=":
            passed = value <= self.threshold
        elif self.comparison == ">":
            passed = value > self.threshold
        elif self.comparison == "<":
            passed = value < self.threshold
        elif self.comparison == "==":
            passed = abs(value - self.threshold) < 1e-6
        else:
            raise ValueError(f"Unknown comparison: {self.comparison}")

        if passed:
            return True, f"{self.name}: {value:.4f} {self.comparison} {self.threshold} (PASS)"
        else:
            return False, f"{self.name}: {value:.4f} {self.comparison} {self.threshold} (FAIL)"


# Standard validation gates for different stages
QUICK_GATES = [
    ValidationGate("loss_reasonable", "loss", 100.0, "<=", True, "Loss should not explode"),
    ValidationGate("no_nan", "has_nan", 0.5, "<", True, "No NaN values in outputs"),
]

STANDARD_GATES = [
    ValidationGate("loss_decreasing", "loss", 10.0, "<=", True, "Loss should decrease"),
    ValidationGate("no_nan", "has_nan", 0.5, "<", True, "No NaN values"),
    ValidationGate("gradient_norm", "grad_norm", 100.0, "<=", True, "Gradients should not explode"),
    ValidationGate("perplexity", "perplexity", 1000.0, "<=", True, "Perplexity should be reasonable"),
]

FULL_GATES = [
    ValidationGate("min_accuracy", "accuracy", 0.1, ">=", True, "Minimum accuracy threshold"),
    ValidationGate("max_loss", "loss", 5.0, "<=", True, "Maximum loss threshold"),
    ValidationGate("tool_accuracy", "tool_accuracy", 0.3, ">=", False, "Tool use accuracy"),
    ValidationGate("reasoning_coherence", "coherence", 0.5, ">=", False, "Reasoning coherence"),
]

PRODUCTION_GATES = [
    ValidationGate("gsm8k_accuracy", "gsm8k_accuracy", 0.5, ">=", True, "GSM8K benchmark"),
    ValidationGate("tool_accuracy", "tool_accuracy", 0.85, ">=", True, "Tool use accuracy"),
    ValidationGate("safety_accuracy", "safety_accuracy", 0.95, ">=", True, "Safety classification"),
    ValidationGate("latency_p99", "latency_p99_ms", 5000.0, "<=", True, "99th percentile latency"),
]


class PipelineValidator:
    """
    Validates model checkpoints and training runs.

    Runs a series of checks to ensure quality before proceeding.
    Can block training if gates fail (in strict mode).
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        strict: bool = True,
        custom_gates: Optional[List[ValidationGate]] = None,
    ):
        """
        Initialize validator.

        Args:
            config: Pipeline configuration
            strict: If True, raise on validation failures
            custom_gates: Additional custom validation gates
        """
        self.config = config
        self.strict = strict
        self.custom_gates = custom_gates or []

        # Metric calculators
        self._metric_fns: Dict[str, Callable] = {}
        self._register_default_metrics()

        # History
        self.validation_history: List[ValidationResult] = []

    def _register_default_metrics(self):
        """Register default metric calculation functions."""
        self._metric_fns["has_nan"] = self._check_nan
        self._metric_fns["grad_norm"] = self._compute_grad_norm

    def register_metric(self, name: str, fn: Callable):
        """Register a custom metric calculation function."""
        self._metric_fns[name] = fn

    def _check_nan(self, model: nn.Module, **kwargs) -> float:
        """Check for NaN values in model parameters."""
        for param in model.parameters():
            if torch.isnan(param).any():
                return 1.0
        return 0.0

    def _compute_grad_norm(self, model: nn.Module, **kwargs) -> float:
        """Compute total gradient norm."""
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def validate_quick(
        self,
        model: nn.Module,
        loss: float,
        step: int,
        **extra_metrics,
    ) -> ValidationResult:
        """
        Quick validation during training.

        Run every N steps to catch problems early.
        Should be very fast (<1 second).
        """
        start_time = time.time()

        metrics = {
            "loss": loss,
            "step": step,
            **extra_metrics,
        }

        # Compute metrics that need the model
        metrics["has_nan"] = self._check_nan(model)

        # Check gates
        result = self._check_gates(QUICK_GATES, metrics, ValidationLevel.QUICK)
        result.duration_seconds = time.time() - start_time

        self.validation_history.append(result)

        if not result.passed and self.strict:
            raise ValidationError(f"Quick validation failed: {result.failures}")

        return result

    def validate_checkpoint(
        self,
        model: nn.Module,
        eval_dataloader: Optional[Any] = None,
        step: int = 0,
        **extra_metrics,
    ) -> ValidationResult:
        """
        Standard checkpoint validation.

        Run at each checkpoint save.
        More thorough than quick validation.
        """
        start_time = time.time()

        metrics = {
            "step": step,
            **extra_metrics,
        }

        # Compute model metrics
        metrics["has_nan"] = self._check_nan(model)
        metrics["grad_norm"] = self._compute_grad_norm(model)

        # Compute eval metrics if dataloader provided
        if eval_dataloader is not None:
            eval_metrics = self._compute_eval_metrics(model, eval_dataloader)
            metrics.update(eval_metrics)

        # Check gates
        result = self._check_gates(STANDARD_GATES, metrics, ValidationLevel.STANDARD)
        result.duration_seconds = time.time() - start_time

        self.validation_history.append(result)

        if not result.passed and self.strict:
            raise ValidationError(f"Checkpoint validation failed: {result.failures}")

        return result

    def validate_full(
        self,
        model: nn.Module,
        eval_dataloader: Any,
        benchmark_runner: Optional[Callable] = None,
    ) -> ValidationResult:
        """
        Full validation before scaling up.

        Runs complete benchmark suite.
        Should pass before proceeding to larger scale.
        """
        start_time = time.time()

        metrics = {}

        # Compute model metrics
        metrics["has_nan"] = self._check_nan(model)

        # Compute eval metrics
        eval_metrics = self._compute_eval_metrics(model, eval_dataloader)
        metrics.update(eval_metrics)

        # Run benchmarks if available
        if benchmark_runner is not None:
            benchmark_metrics = benchmark_runner(model)
            metrics.update(benchmark_metrics)

        # Check gates (STANDARD + FULL)
        all_gates = STANDARD_GATES + FULL_GATES + self.custom_gates
        result = self._check_gates(all_gates, metrics, ValidationLevel.FULL)
        result.duration_seconds = time.time() - start_time

        self.validation_history.append(result)

        if not result.passed and self.strict:
            raise ValidationError(f"Full validation failed: {result.failures}")

        return result

    def validate_production(
        self,
        model: nn.Module,
        benchmark_suite: Any,
        safety_suite: Any,
    ) -> ValidationResult:
        """
        Production validation before deployment.

        Most rigorous validation level.
        Must pass before serving to users.
        """
        start_time = time.time()

        metrics = {}

        # Run full benchmark suite
        benchmark_results = benchmark_suite.run(model)
        metrics.update(benchmark_results)

        # Run safety evaluation
        safety_results = safety_suite.run(model)
        metrics.update(safety_results)

        # Compute latency metrics
        latency_metrics = self._compute_latency_metrics(model)
        metrics.update(latency_metrics)

        # Check all production gates
        all_gates = PRODUCTION_GATES + self.custom_gates
        result = self._check_gates(all_gates, metrics, ValidationLevel.PRODUCTION)
        result.duration_seconds = time.time() - start_time

        self.validation_history.append(result)

        if not result.passed and self.strict:
            raise ValidationError(f"Production validation failed: {result.failures}")

        return result

    def _check_gates(
        self,
        gates: List[ValidationGate],
        metrics: Dict[str, float],
        level: ValidationLevel,
    ) -> ValidationResult:
        """Check all gates against metrics."""
        result = ValidationResult(
            passed=True,
            level=level,
            metrics=metrics.copy(),
        )

        for gate in gates:
            if gate.metric_name not in metrics:
                if gate.required:
                    result.warnings.append(f"Missing required metric: {gate.metric_name}")
                continue

            value = metrics[gate.metric_name]
            passed, message = gate.check(value)
            result.gate_results[gate.name] = passed

            if not passed:
                if gate.required:
                    result.passed = False
                    result.failures.append(message)
                else:
                    result.warnings.append(message)

        return result

    def _compute_eval_metrics(
        self,
        model: nn.Module,
        eval_dataloader: Any,
    ) -> Dict[str, float]:
        """Compute evaluation metrics on a dataset."""
        model.eval()

        total_loss = 0.0
        num_batches = 0

        device = next(model.parameters()).device

        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch.get("labels"),
                )

                if isinstance(outputs, dict) and "loss" in outputs:
                    total_loss += outputs["loss"].item()
                num_batches += 1

                # Limit eval batches for speed
                if num_batches >= 100:
                    break

        avg_loss = total_loss / max(num_batches, 1)

        import math
        return {
            "loss": avg_loss,
            "perplexity": math.exp(min(avg_loss, 20)),  # Cap to avoid overflow
        }

    def _compute_latency_metrics(
        self,
        model: nn.Module,
        num_samples: int = 100,
        input_length: int = 128,
        output_length: int = 64,
    ) -> Dict[str, float]:
        """Compute inference latency metrics."""
        model.eval()
        device = next(model.parameters()).device

        latencies = []

        # Warmup
        dummy_input = torch.randint(0, 1000, (1, input_length), device=device)
        for _ in range(5):
            with torch.no_grad():
                _ = model(dummy_input)

        # Measure
        for _ in range(num_samples):
            dummy_input = torch.randint(0, 1000, (1, input_length), device=device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()

            with torch.no_grad():
                _ = model(dummy_input)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

        latencies.sort()

        return {
            "latency_p50_ms": latencies[len(latencies) // 2],
            "latency_p90_ms": latencies[int(len(latencies) * 0.9)],
            "latency_p99_ms": latencies[int(len(latencies) * 0.99)],
            "latency_mean_ms": sum(latencies) / len(latencies),
        }

    def save_history(self, path: str):
        """Save validation history to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        history = [r.to_dict() for r in self.validation_history]

        with open(path, 'w') as f:
            json.dump(history, f, indent=2)

    def print_summary(self):
        """Print summary of validation history."""
        if not self.validation_history:
            print("No validation history.")
            return

        print("\n" + "=" * 60)
        print("Validation History Summary")
        print("=" * 60)

        for i, result in enumerate(self.validation_history):
            status = "PASS" if result.passed else "FAIL"
            print(f"\n[{i+1}] {result.level.value.upper()} - {status}")
            print(f"    Duration: {result.duration_seconds:.1f}s")
            if result.failures:
                print(f"    Failures: {len(result.failures)}")
            if result.warnings:
                print(f"    Warnings: {len(result.warnings)}")


class ValidationError(Exception):
    """Raised when validation fails in strict mode."""
    pass


def create_validator_for_scale(scale: str, strict: bool = True) -> PipelineValidator:
    """
    Create a validator with appropriate gates for a model scale.

    Smaller scales have looser gates (for experimentation).
    Larger scales have stricter gates (approaching production).
    """
    from .config import ModelScale

    scale_enum = ModelScale(scale) if isinstance(scale, str) else scale

    # Adjust gates based on scale
    if scale_enum in [ModelScale.TINY, ModelScale.SMALL]:
        # Loose gates for experimentation
        custom_gates = [
            ValidationGate("min_accuracy", "accuracy", 0.05, ">=", False),
        ]
    elif scale_enum in [ModelScale.MEDIUM, ModelScale.LARGE]:
        # Moderate gates
        custom_gates = [
            ValidationGate("min_accuracy", "accuracy", 0.2, ">=", True),
            ValidationGate("tool_accuracy", "tool_accuracy", 0.5, ">=", False),
        ]
    else:
        # Strict gates for production scales
        custom_gates = [
            ValidationGate("min_accuracy", "accuracy", 0.4, ">=", True),
            ValidationGate("tool_accuracy", "tool_accuracy", 0.7, ">=", True),
        ]

    return PipelineValidator(strict=strict, custom_gates=custom_gates)
