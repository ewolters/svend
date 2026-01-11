"""
Evaluation Metrics

Common metrics for evaluating reasoning models.

Metrics:
- accuracy: Simple correctness
- exact_match: Exact string match
- f1_score: Token-level F1
- pass_at_k: Code execution pass rate
- reasoning_coherence: Chain coherence score
- tool_precision: Tool selection precision/recall
"""

from typing import List, Dict, Any, Optional, Callable
import re
from collections import Counter
import math


def accuracy(predictions: List[Any], targets: List[Any]) -> float:
    """
    Simple accuracy metric.

    Args:
        predictions: List of predicted values
        targets: List of target values

    Returns:
        Fraction of correct predictions
    """
    if not predictions:
        return 0.0

    correct = sum(1 for p, t in zip(predictions, targets) if p == t)
    return correct / len(predictions)


def exact_match(prediction: str, target: str, normalize: bool = True) -> bool:
    """
    Check for exact string match.

    Args:
        prediction: Predicted string
        target: Target string
        normalize: Whether to normalize strings before comparison

    Returns:
        True if strings match
    """
    if normalize:
        prediction = _normalize_text(prediction)
        target = _normalize_text(target)

    return prediction == target


def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def f1_score(prediction: str, target: str) -> float:
    """
    Token-level F1 score.

    Args:
        prediction: Predicted text
        target: Target text

    Returns:
        F1 score between 0 and 1
    """
    pred_tokens = _normalize_text(prediction).split()
    target_tokens = _normalize_text(target).split()

    if not pred_tokens or not target_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    target_counter = Counter(target_tokens)

    common = pred_counter & target_counter
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(target_tokens)

    return 2 * precision * recall / (precision + recall)


def pass_at_k(
    results: List[List[bool]],
    k: int = 1,
) -> float:
    """
    Pass@k metric for code generation.

    For each problem, we have n samples. Pass@k is the probability
    that at least one of k samples passes.

    Args:
        results: List of lists, where results[i] contains pass/fail for problem i
        k: Number of samples to consider

    Returns:
        Pass@k score
    """
    if not results:
        return 0.0

    def pass_at_k_single(n: int, c: int, k: int) -> float:
        """
        Calculate pass@k for single problem.

        n: total samples
        c: correct samples
        k: k value
        """
        if n - c < k:
            return 1.0
        return 1.0 - math.prod([(n - c - i) / (n - i) for i in range(k)])

    scores = []
    for problem_results in results:
        n = len(problem_results)
        c = sum(problem_results)
        if n >= k:
            scores.append(pass_at_k_single(n, c, k))

    return sum(scores) / len(scores) if scores else 0.0


def reasoning_coherence(
    steps: List[str],
    check_logical_flow: bool = True,
    check_conclusion: bool = True,
) -> Dict[str, float]:
    """
    Evaluate coherence of a reasoning chain.

    Args:
        steps: List of reasoning steps
        check_logical_flow: Whether to check step-to-step coherence
        check_conclusion: Whether to verify conclusion follows from steps

    Returns:
        Dict with coherence metrics
    """
    if not steps:
        return {
            "coherence_score": 0.0,
            "num_steps": 0,
            "has_conclusion": False,
        }

    metrics = {
        "num_steps": len(steps),
        "avg_step_length": sum(len(s) for s in steps) / len(steps),
    }

    # Check for logical connectors
    connector_patterns = [
        r"\btherefore\b", r"\bthus\b", r"\bhence\b",
        r"\bbecause\b", r"\bsince\b", r"\bso\b",
        r"\bif\b.*\bthen\b", r"\bfollows that\b",
    ]

    connector_count = 0
    for step in steps:
        for pattern in connector_patterns:
            if re.search(pattern, step.lower()):
                connector_count += 1
                break

    metrics["connector_ratio"] = connector_count / len(steps) if steps else 0

    # Check for conclusion indicators
    conclusion_patterns = [
        r"\btherefore\b", r"\bthus\b", r"\bconclusion\b",
        r"\banswer is\b", r"\bwe get\b", r"\bresult is\b",
    ]

    has_conclusion = any(
        any(re.search(p, step.lower()) for p in conclusion_patterns)
        for step in steps[-2:]  # Check last 2 steps
    )
    metrics["has_conclusion"] = has_conclusion

    # Calculate overall coherence score
    score = 0.0
    score += min(metrics["connector_ratio"] * 2, 0.5)  # Up to 0.5 for connectors
    score += 0.3 if has_conclusion else 0.0  # 0.3 for conclusion
    score += min(len(steps) / 5, 0.2)  # Up to 0.2 for having multiple steps

    metrics["coherence_score"] = score

    return metrics


def tool_precision_recall(
    predictions: List[Optional[str]],
    targets: List[Optional[str]],
) -> Dict[str, float]:
    """
    Calculate precision and recall for tool selection.

    Args:
        predictions: List of predicted tool names (None if no tool)
        targets: List of target tool names (None if no tool)

    Returns:
        Dict with precision, recall, and F1
    """
    # True positives: predicted tool and it's correct
    tp = sum(1 for p, t in zip(predictions, targets)
             if p is not None and p == t)

    # False positives: predicted tool but wrong or shouldn't have
    fp = sum(1 for p, t in zip(predictions, targets)
             if p is not None and p != t)

    # False negatives: should have predicted tool but didn't
    fn = sum(1 for p, t in zip(predictions, targets)
             if p is None and t is not None)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def numeric_accuracy(
    predictions: List[float],
    targets: List[float],
    tolerance: float = 1e-6,
    relative: bool = False,
) -> float:
    """
    Accuracy for numeric predictions with tolerance.

    Args:
        predictions: Predicted values
        targets: Target values
        tolerance: Acceptable error
        relative: If True, use relative tolerance

    Returns:
        Fraction of predictions within tolerance
    """
    if not predictions:
        return 0.0

    correct = 0
    for pred, target in zip(predictions, targets):
        if pred is None or target is None:
            continue

        if relative:
            error = abs(pred - target) / max(abs(target), 1e-10)
        else:
            error = abs(pred - target)

        if error <= tolerance:
            correct += 1

    return correct / len(predictions)


def perplexity(logits: "torch.Tensor", labels: "torch.Tensor") -> float:
    """
    Calculate perplexity from logits and labels.

    Args:
        logits: Model logits [batch, seq_len, vocab]
        labels: Target labels [batch, seq_len]

    Returns:
        Perplexity score
    """
    import torch
    import torch.nn.functional as F

    # Flatten
    logits_flat = logits.view(-1, logits.size(-1))
    labels_flat = labels.view(-1)

    # Calculate cross entropy
    loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100)

    return math.exp(loss.item())


class MetricAggregator:
    """
    Aggregates metrics across multiple evaluations.

    Useful for tracking metrics over training.
    """

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}

    def add(self, name: str, value: float):
        """Add a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def add_batch(self, metrics: Dict[str, float]):
        """Add multiple metrics at once."""
        for name, value in metrics.items():
            self.add(name, value)

    def mean(self, name: str) -> float:
        """Get mean of a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        return sum(self.metrics[name]) / len(self.metrics[name])

    def std(self, name: str) -> float:
        """Get standard deviation of a metric."""
        if name not in self.metrics or len(self.metrics[name]) < 2:
            return 0.0

        values = self.metrics[name]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance ** 0.5

    def latest(self, name: str) -> Optional[float]:
        """Get latest value of a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return None
        return self.metrics[name][-1]

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        return {
            name: {
                "mean": self.mean(name),
                "std": self.std(name),
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
                "count": len(values),
            }
            for name, values in self.metrics.items()
        }

    def reset(self):
        """Clear all metrics."""
        self.metrics = {}
