"""
Scaling Diagnostics - Determine When to Add Parameters

Measures where the model is bottlenecked:
- Tool routing accuracy (can it pick the right tool?)
- Tool formatting accuracy (can it format calls correctly?)
- Reasoning depth (can it chain multiple steps?)
- Knowledge gaps (does it lack facts, or lack reasoning?)
- Attention saturation (is context window the limit?)

Key insight: If tool routing is good but answers are wrong,
you need better tools or data, not more params.
If tool routing itself fails, you might need more params.

Philosophy:
- Maximize tool capability before scaling params
- Scale params only when tool routing fundamentally fails
- Prefer data/tool improvements over param increases
- Stay lean, win on tooling first
"""

from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import re
import time


class BottleneckType(Enum):
    """Types of model bottlenecks."""
    TOOL_ROUTING = "tool_routing"        # Can't pick right tool
    TOOL_FORMATTING = "tool_formatting"  # Picks tool but malformed call
    REASONING_DEPTH = "reasoning_depth"  # Fails on multi-step
    KNOWLEDGE_GAP = "knowledge_gap"      # Missing facts (need data/tools)
    ATTENTION = "attention"              # Context too long
    PARAMETER_BOUND = "parameter_bound"  # Fundamental capacity limit
    NONE = "none"                        # No bottleneck detected


class ScalingRecommendation(Enum):
    """What to do next."""
    ADD_TOOL_DATA = "add_tool_data"          # More tool-use examples
    ADD_DOMAIN_DATA = "add_domain_data"      # More domain knowledge
    IMPROVE_TOOLS = "improve_tools"          # Better tool implementations
    ADD_COT_DATA = "add_cot_data"            # More chain-of-thought
    EXTEND_CONTEXT = "extend_context"        # Longer context window
    SCALE_PARAMS = "scale_params"            # Add parameters
    NO_ACTION = "no_action"                  # Model is performing well


@dataclass
class DiagnosticResult:
    """Result from a diagnostic test."""
    test_name: str
    passed: bool
    bottleneck: BottleneckType
    details: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""


@dataclass
class ScalingReport:
    """Full diagnostic report with scaling recommendations."""
    model_name: str
    model_params: int
    timestamp: str = ""
    results: List[DiagnosticResult] = field(default_factory=list)

    # Summary metrics
    tool_routing_accuracy: float = 0.0
    tool_formatting_accuracy: float = 0.0
    reasoning_depth_score: float = 0.0
    knowledge_score: float = 0.0
    attention_score: float = 0.0

    # Bottleneck analysis
    primary_bottleneck: BottleneckType = BottleneckType.NONE
    bottleneck_counts: Dict[str, int] = field(default_factory=dict)

    # Recommendations
    recommendations: List[ScalingRecommendation] = field(default_factory=list)
    should_scale_params: bool = False
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_params": self.model_params,
            "timestamp": self.timestamp,
            "metrics": {
                "tool_routing_accuracy": self.tool_routing_accuracy,
                "tool_formatting_accuracy": self.tool_formatting_accuracy,
                "reasoning_depth_score": self.reasoning_depth_score,
                "knowledge_score": self.knowledge_score,
                "attention_score": self.attention_score,
            },
            "primary_bottleneck": self.primary_bottleneck.value,
            "bottleneck_counts": self.bottleneck_counts,
            "recommendations": [r.value for r in self.recommendations],
            "should_scale_params": self.should_scale_params,
            "explanation": self.explanation,
            "results": [
                {
                    "test": r.test_name,
                    "passed": r.passed,
                    "bottleneck": r.bottleneck.value,
                    "details": r.details,
                    "recommendation": r.recommendation,
                }
                for r in self.results
            ],
        }


class ScalingDiagnostics:
    """
    Diagnostic suite to determine if/when to scale parameters.

    Run this after training to determine next steps:
    1. If tool routing fails → add tool-use data
    2. If formatting fails → add syntax examples
    3. If reasoning depth fails → add CoT data, then maybe scale
    4. If knowledge gaps → add domain data or tools
    5. If attention limited → extend context
    6. If all else fails → scale params
    """

    # Test cases for tool routing
    ROUTING_TESTS = [
        # (prompt, expected_tool, difficulty)
        ("What is 7 * 8?", "calculator", "easy"),
        ("What is 234 + 567?", "calculator", "easy"),
        ("Solve x^2 - 5x + 6 = 0", "symbolic_math", "easy"),
        ("What is the derivative of x^3?", "symbolic_math", "easy"),
        ("What is the molecular weight of H2O?", "chemistry", "easy"),
        ("Balance H2 + O2 -> H2O", "chemistry", "easy"),
        ("Convert 100 km/h to m/s", "unit_converter", "easy"),

        # Medium - requires understanding
        ("If I drop a ball from 10m, how long until it hits the ground?", "physics", "medium"),
        ("Is there a solution where x > 5 AND x < 3?", "logic_solver", "medium"),
        ("Plot sin(x) from 0 to 2*pi", "plotter", "medium"),
        ("What's the mean of [1,2,3,4,5,100]?", "statistics", "medium"),
        ("Find the eigenvalues of [[1,2],[3,4]]", "numerical", "medium"),

        # Hard - ambiguous or multi-tool
        ("What's the integral of x^2 from 0 to 1, and verify numerically?", "symbolic_math", "hard"),
        ("Solve 3x + 2y = 7, x - y = 1 for x and y", "symbolic_math", "hard"),
        ("What's the pH of 0.01M HCl?", "chemistry", "hard"),
    ]

    # Test cases for reasoning depth
    DEPTH_TESTS = [
        # (prompt, expected_steps, expected_answer)
        ("What is 15% of 80?", 1, 12),
        ("A shirt costs $40 with 25% off. What's the sale price?", 2, 30),
        ("Train goes 60mph for 2hr, then 80mph for 1hr. Average speed?", 3, 66.67),
        ("Mix 50mL of 2M HCl with 50mL of 1M NaOH. Final HCl concentration?", 4, 0.5),
    ]

    def __init__(self, model, tokenizer, tool_registry=None):
        """
        Args:
            model: The model to diagnose
            tokenizer: Tokenizer for the model
            tool_registry: Optional tool registry
        """
        self.model = model
        self.tokenizer = tokenizer
        self.tool_registry = tool_registry
        self.results: List[DiagnosticResult] = []

    def run_full_diagnostic(self) -> ScalingReport:
        """Run all diagnostic tests and generate scaling report."""
        from datetime import datetime

        # Get model info
        model_params = sum(p.numel() for p in self.model.parameters())
        model_name = getattr(self.model, 'name', getattr(self.model, '__class__', 'unknown'))

        report = ScalingReport(
            model_name=str(model_name),
            model_params=model_params,
            timestamp=datetime.now().isoformat(),
        )

        print(f"\n{'='*60}")
        print(f"SCALING DIAGNOSTICS: {model_name}")
        print(f"Parameters: {model_params:,}")
        print(f"{'='*60}\n")

        # Run test suites
        print("Testing tool routing...")
        routing_results = self._test_tool_routing()
        report.results.extend(routing_results)

        print("Testing tool formatting...")
        format_results = self._test_tool_formatting()
        report.results.extend(format_results)

        print("Testing reasoning depth...")
        depth_results = self._test_reasoning_depth()
        report.results.extend(depth_results)

        print("Testing knowledge vs reasoning...")
        knowledge_results = self._test_knowledge_vs_reasoning()
        report.results.extend(knowledge_results)

        print("Testing attention limits...")
        attention_results = self._test_attention_limits()
        report.results.extend(attention_results)

        # Calculate scores
        report.tool_routing_accuracy = self._calc_accuracy(routing_results)
        report.tool_formatting_accuracy = self._calc_accuracy(format_results)
        report.reasoning_depth_score = self._calc_accuracy(depth_results)
        report.knowledge_score = self._calc_accuracy(knowledge_results)
        report.attention_score = self._calc_accuracy(attention_results)

        # Analyze bottlenecks
        report.bottleneck_counts = self._count_bottlenecks(report.results)
        report.primary_bottleneck = self._find_primary_bottleneck(report.bottleneck_counts)

        # Generate recommendations
        report.recommendations, report.should_scale_params, report.explanation = \
            self._generate_recommendations(report)

        return report

    def _calc_accuracy(self, results: List[DiagnosticResult]) -> float:
        """Calculate pass rate for a set of results."""
        if not results:
            return 0.0
        return sum(1 for r in results if r.passed) / len(results)

    def _count_bottlenecks(self, results: List[DiagnosticResult]) -> Dict[str, int]:
        """Count bottlenecks by type."""
        counts = {}
        for r in results:
            bt = r.bottleneck.value
            counts[bt] = counts.get(bt, 0) + 1
        return counts

    def _find_primary_bottleneck(self, counts: Dict[str, int]) -> BottleneckType:
        """Find the most common bottleneck type."""
        non_none = {k: v for k, v in counts.items() if k != "none" and v > 0}
        if not non_none:
            return BottleneckType.NONE
        primary = max(non_none.keys(), key=lambda x: non_none[x])
        return BottleneckType(primary)

    # ==================== TEST SUITES ====================

    def _test_tool_routing(self) -> List[DiagnosticResult]:
        """Test if model can select the correct tool."""
        results = []

        for prompt, expected_tool, difficulty in self.ROUTING_TESTS:
            response = self._generate(prompt, max_tokens=200)
            tool_call = self._extract_tool_call(response)

            if tool_call is None:
                result = DiagnosticResult(
                    test_name=f"routing:{expected_tool}:{difficulty}",
                    passed=False,
                    bottleneck=BottleneckType.TOOL_ROUTING,
                    details={"expected": expected_tool, "actual": None, "difficulty": difficulty},
                    recommendation="Model didn't attempt tool call.",
                )
            else:
                actual = tool_call.get("tool_name", "")
                passed = actual == expected_tool
                result = DiagnosticResult(
                    test_name=f"routing:{expected_tool}:{difficulty}",
                    passed=passed,
                    bottleneck=BottleneckType.NONE if passed else BottleneckType.TOOL_ROUTING,
                    details={"expected": expected_tool, "actual": actual, "difficulty": difficulty},
                    recommendation="" if passed else f"Expected {expected_tool}, got {actual}",
                )

            results.append(result)

        return results

    def _test_tool_formatting(self) -> List[DiagnosticResult]:
        """Test if model can format tool calls correctly."""
        results = []

        format_tests = [
            ("Calculate sqrt(144)", "calculator", ["expression"]),
            ("Solve 2x + 3 = 7 for x", "symbolic_math", ["expression", "operation"]),
            ("Convert 32 F to C", "unit_converter", ["value", "from_unit", "to_unit"]),
        ]

        for prompt, expected_tool, required_args in format_tests:
            response = self._generate(prompt, max_tokens=300)
            tool_call = self._extract_tool_call(response)

            if tool_call is None:
                result = DiagnosticResult(
                    test_name=f"format:{expected_tool}",
                    passed=False,
                    bottleneck=BottleneckType.TOOL_FORMATTING,
                    details={"error": "No tool call found"},
                )
            elif tool_call.get("args") is None:
                result = DiagnosticResult(
                    test_name=f"format:{expected_tool}",
                    passed=False,
                    bottleneck=BottleneckType.TOOL_FORMATTING,
                    details={"error": "Invalid JSON args"},
                )
            else:
                args = tool_call.get("args", {})
                # Check if we got valid JSON with some args
                has_args = len(args) > 0
                result = DiagnosticResult(
                    test_name=f"format:{expected_tool}",
                    passed=has_args,
                    bottleneck=BottleneckType.NONE if has_args else BottleneckType.TOOL_FORMATTING,
                    details={"provided_args": list(args.keys())},
                )

            results.append(result)

        return results

    def _test_reasoning_depth(self) -> List[DiagnosticResult]:
        """Test multi-step reasoning capability."""
        results = []

        for prompt, expected_steps, expected_answer in self.DEPTH_TESTS:
            response = self._generate(prompt, max_tokens=500)

            # Extract answer
            numbers = re.findall(r'-?\d+\.?\d*', response)
            got_answer = None
            tolerance = abs(expected_answer) * 0.1 + 0.01

            for num in reversed(numbers):
                try:
                    val = float(num)
                    if abs(val - expected_answer) < tolerance:
                        got_answer = val
                        break
                except:
                    continue

            correct = got_answer is not None

            # Count reasoning steps
            think_count = len(re.findall(r'<\|think\|>', response))
            step_count = len(re.findall(r'(?:step\s*\d|^\d+[\.\)])', response, re.MULTILINE | re.IGNORECASE))
            actual_steps = max(think_count, step_count, 1)

            if correct:
                bottleneck = BottleneckType.NONE
            elif actual_steps < expected_steps:
                bottleneck = BottleneckType.REASONING_DEPTH
            else:
                bottleneck = BottleneckType.KNOWLEDGE_GAP

            results.append(DiagnosticResult(
                test_name=f"depth:{expected_steps}_steps",
                passed=correct,
                bottleneck=bottleneck,
                details={
                    "expected_steps": expected_steps,
                    "actual_steps": actual_steps,
                    "expected_answer": expected_answer,
                    "got_answer": got_answer,
                },
            ))

        return results

    def _test_knowledge_vs_reasoning(self) -> List[DiagnosticResult]:
        """Distinguish knowledge gaps from reasoning failures."""
        results = []

        # Test with and without facts provided
        knowledge_tests = [
            {
                "name": "kinetic_energy",
                "without": "What's the kinetic energy of a 2kg object moving at 3 m/s?",
                "with": "KE = (1/2)mv². What's the KE of a 2kg object at 3 m/s?",
                "answer": 9,
            },
            {
                "name": "ph_calculation",
                "without": "What's the pH of a 0.001 M HCl solution?",
                "with": "For strong acids, pH = -log10([H+]). HCl fully dissociates. What's pH of 0.001 M HCl?",
                "answer": 3,
            },
        ]

        for test in knowledge_tests:
            # Without facts
            resp_without = self._generate(test["without"], max_tokens=300)
            correct_without = self._check_answer(resp_without, test["answer"])

            # With facts
            resp_with = self._generate(test["with"], max_tokens=300)
            correct_with = self._check_answer(resp_with, test["answer"])

            # Diagnose
            if correct_with and not correct_without:
                # Reasoning OK, missing knowledge
                results.append(DiagnosticResult(
                    test_name=f"knowledge:{test['name']}",
                    passed=False,
                    bottleneck=BottleneckType.KNOWLEDGE_GAP,
                    details={"with_facts": correct_with, "without_facts": correct_without},
                    recommendation="Add domain data or tools for facts.",
                ))
            elif not correct_with:
                # Fails even with facts - reasoning issue
                results.append(DiagnosticResult(
                    test_name=f"knowledge:{test['name']}",
                    passed=False,
                    bottleneck=BottleneckType.REASONING_DEPTH,
                    details={"with_facts": correct_with, "without_facts": correct_without},
                    recommendation="Reasoning failure even with facts provided.",
                ))
            else:
                results.append(DiagnosticResult(
                    test_name=f"knowledge:{test['name']}",
                    passed=True,
                    bottleneck=BottleneckType.NONE,
                    details={"with_facts": correct_with, "without_facts": correct_without},
                ))

        return results

    def _test_attention_limits(self) -> List[DiagnosticResult]:
        """Test if model struggles with long contexts."""
        results = []

        # Simple arithmetic with increasing context padding
        context_lengths = [100, 500, 1000, 2000]

        for length in context_lengths:
            padding = "The quick brown fox jumps over the lazy dog. " * (length // 45)
            prompt = f"{padding}\n\nQuestion: What is 17 + 25?\nAnswer:"

            start = time.time()
            response = self._generate(prompt, max_tokens=50)
            latency = (time.time() - start) * 1000

            correct = "42" in response

            results.append(DiagnosticResult(
                test_name=f"attention:{length}_tokens",
                passed=correct,
                bottleneck=BottleneckType.NONE if correct else BottleneckType.ATTENTION,
                details={"context_length": length, "latency_ms": latency},
            ))

            if not correct:
                break  # Don't test longer if already failing

        return results

    def _check_answer(self, response: str, expected: float) -> bool:
        """Check if response contains the expected answer."""
        tolerance = abs(expected) * 0.1 + 0.01
        numbers = re.findall(r'-?\d+\.?\d*', response)
        for num in numbers:
            try:
                if abs(float(num) - expected) < tolerance:
                    return True
            except:
                continue
        return False

    # ==================== RECOMMENDATIONS ====================

    def _generate_recommendations(
        self,
        report: ScalingReport,
    ) -> Tuple[List[ScalingRecommendation], bool, str]:
        """Generate scaling recommendations based on results."""
        recs = []
        should_scale = False
        explanations = []

        # Check each metric
        if report.tool_routing_accuracy < 0.6:
            recs.append(ScalingRecommendation.ADD_TOOL_DATA)
            explanations.append(f"Tool routing at {report.tool_routing_accuracy:.0%} - add more tool-use examples")

        if report.tool_formatting_accuracy < 0.7:
            recs.append(ScalingRecommendation.ADD_TOOL_DATA)
            explanations.append(f"Tool formatting at {report.tool_formatting_accuracy:.0%} - add JSON syntax examples")

        if report.reasoning_depth_score < 0.5:
            recs.append(ScalingRecommendation.ADD_COT_DATA)
            explanations.append(f"Reasoning depth at {report.reasoning_depth_score:.0%} - add chain-of-thought examples")

        if report.knowledge_score < 0.5:
            # Knowledge gaps can be fixed with data or tools
            knowledge_bottlenecks = report.bottleneck_counts.get("knowledge_gap", 0)
            reasoning_bottlenecks = report.bottleneck_counts.get("reasoning_depth", 0)

            if knowledge_bottlenecks > reasoning_bottlenecks:
                recs.append(ScalingRecommendation.ADD_DOMAIN_DATA)
                recs.append(ScalingRecommendation.IMPROVE_TOOLS)
                explanations.append("Knowledge gaps detected - add domain data or improve tools")
            else:
                recs.append(ScalingRecommendation.ADD_COT_DATA)
                explanations.append("Reasoning failures even with facts - add more CoT data")

        if report.attention_score < 0.8:
            recs.append(ScalingRecommendation.EXTEND_CONTEXT)
            explanations.append(f"Attention issues at longer contexts")

        # Check if all else is good but still failing
        all_scores = [
            report.tool_routing_accuracy,
            report.tool_formatting_accuracy,
            report.reasoning_depth_score,
        ]

        if all(s >= 0.7 for s in all_scores):
            if report.knowledge_score >= 0.7 and report.attention_score >= 0.8:
                recs = [ScalingRecommendation.NO_ACTION]
                explanations = ["Model performing well across all diagnostics"]
            else:
                # Good at tools and reasoning but still gaps
                recs.append(ScalingRecommendation.ADD_DOMAIN_DATA)

        # Only recommend scaling params as last resort
        if report.tool_routing_accuracy >= 0.8 and report.tool_formatting_accuracy >= 0.8:
            if report.reasoning_depth_score < 0.4:
                # Good at tools but can't reason - might need params
                recs.append(ScalingRecommendation.SCALE_PARAMS)
                should_scale = True
                explanations.append("Tool use OK but reasoning limited - consider scaling params")

        # Remove duplicates while preserving order
        seen = set()
        unique_recs = []
        for r in recs:
            if r not in seen:
                seen.add(r)
                unique_recs.append(r)

        explanation = " | ".join(explanations) if explanations else "No significant issues found"

        return unique_recs, should_scale, explanation

    # ==================== HELPERS ====================

    def _generate(self, prompt: str, max_tokens: int = 200) -> str:
        """Generate response from model."""
        import torch

        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        if hasattr(self.model, 'device'):
            inputs = inputs.to(self.model.device)
        elif next(self.model.parameters(), None) is not None:
            inputs = inputs.to(next(self.model.parameters()).device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        if prompt in response:
            response = response[response.index(prompt) + len(prompt):]

        return response.strip()

    def _extract_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract tool call from generated text."""
        # Try structured format
        pattern = r'<\|tool_call\|>.*?<\|tool_name\|>(\w+)<\|tool_args\|>(.+?)<\|/tool_call\|>'
        match = re.search(pattern, text, re.DOTALL)

        if match:
            tool_name = match.group(1)
            try:
                args = json.loads(match.group(2))
            except json.JSONDecodeError:
                args = None
            return {"tool_name": tool_name, "args": args}

        # Try simpler patterns
        simple_patterns = [
            r'Tool:\s*(\w+)\s*\nArgs:\s*(\{.+?\})',
            r'```tool\s*(\w+)\s*(\{.+?\})\s*```',
            r'(\w+)\s*\(\s*(.+?)\s*\)',
        ]

        for pattern in simple_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                tool_name = match.group(1)
                try:
                    args = json.loads(match.group(2))
                except:
                    args = {}
                return {"tool_name": tool_name, "args": args}

        return None


def run_scaling_diagnostics(
    model,
    tokenizer,
    tool_registry=None,
    output_path: Optional[str] = None,
) -> ScalingReport:
    """
    Run scaling diagnostics on a model.

    Args:
        model: Model to diagnose
        tokenizer: Tokenizer
        tool_registry: Optional tool registry
        output_path: Optional path to save JSON report

    Returns:
        ScalingReport with recommendations
    """
    diagnostics = ScalingDiagnostics(model, tokenizer, tool_registry)
    report = diagnostics.run_full_diagnostic()

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nReport saved to {output_path}")

    return report


def print_scaling_report(report: ScalingReport):
    """Print human-readable scaling report."""
    print("\n" + "="*60)
    print("SCALING DIAGNOSTICS REPORT")
    print("="*60)

    print(f"\nModel: {report.model_name}")
    print(f"Parameters: {report.model_params:,}")

    print("\n--- METRICS ---")
    metrics = [
        ("Tool Routing", report.tool_routing_accuracy),
        ("Tool Formatting", report.tool_formatting_accuracy),
        ("Reasoning Depth", report.reasoning_depth_score),
        ("Knowledge", report.knowledge_score),
        ("Attention", report.attention_score),
    ]

    for name, score in metrics:
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        status = "✓" if score >= 0.8 else "⚠" if score >= 0.5 else "✗"
        print(f"  {name:18} {bar} {score*100:5.1f}% {status}")

    print(f"\n--- PRIMARY BOTTLENECK ---")
    print(f"  {report.primary_bottleneck.value}")

    print(f"\n--- RECOMMENDATIONS ---")
    for rec in report.recommendations:
        symbol = {
            ScalingRecommendation.ADD_TOOL_DATA: "📚",
            ScalingRecommendation.ADD_DOMAIN_DATA: "📖",
            ScalingRecommendation.IMPROVE_TOOLS: "🔧",
            ScalingRecommendation.ADD_COT_DATA: "🧠",
            ScalingRecommendation.EXTEND_CONTEXT: "📏",
            ScalingRecommendation.SCALE_PARAMS: "⬆️",
            ScalingRecommendation.NO_ACTION: "✓",
        }.get(rec, "•")
        print(f"  {symbol} {rec.value}")

    print(f"\n--- VERDICT ---")
    if report.should_scale_params:
        print("  ⚠️  SCALING PARAMS MAY BE WARRANTED")
        print("  Model is good at tools but limited in reasoning capacity.")
    else:
        print("  ✓ DO NOT SCALE PARAMS YET")
        print("  Focus on data/tools improvements first.")

    print(f"\n  {report.explanation}")
    print("\n" + "="*60)
