"""
Counterfactual Tool - Sensitivity Analysis and What-If Reasoning

LLMs love one story. Reality usually has several.

This tool:
- Varies inputs to test robustness
- Perturbs assumptions to find sensitivity
- Reruns reasoning paths with different parameters
- Reports "conclusion is sensitive to X but robust to Y"

That's real reasoning. Almost no models do this honestly.
"""

from typing import Optional, Dict, Any, List, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from copy import deepcopy
import math

from .registry import Tool, ToolParameter, ToolResult, ToolStatus, ToolRegistry


class Sensitivity(Enum):
    """How sensitive is the conclusion to a parameter?"""
    ROBUST = "robust"  # Conclusion unchanged across all variations
    SENSITIVE = "sensitive"  # Conclusion changes with parameter
    THRESHOLD = "threshold"  # Conclusion flips at specific value
    UNKNOWN = "unknown"  # Could not determine


@dataclass
class Variation:
    """A single parameter variation."""
    parameter: str
    original_value: Any
    varied_value: Any
    original_result: Any
    varied_result: Any
    conclusion_changed: bool


@dataclass
class SensitivityReport:
    """Report on parameter sensitivity."""
    parameter: str
    sensitivity: Sensitivity
    threshold: Optional[float] = None  # Value where conclusion flips
    variations_tested: int = 0
    conclusion_changed_count: int = 0
    details: List[Variation] = field(default_factory=list)


class CounterfactualAnalyzer:
    """
    Analyzes how conclusions change under different assumptions.

    Core principle: A robust conclusion should survive reasonable
    perturbations. If it doesn't, that's important to know.
    """

    @staticmethod
    def perturb_numeric(
        value: float,
        perturbations: List[float] = None,
        relative: bool = True
    ) -> List[float]:
        """Generate perturbations of a numeric value."""
        if perturbations is None:
            # Default: ±10%, ±25%, ±50%, 2x, 0.5x
            perturbations = [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0, -0.5]

        if relative:
            if value == 0:
                # Can't do relative perturbation of zero
                return [0.1, -0.1, 1, -1, 10, -10]
            return [value * (1 + p) for p in perturbations]
        else:
            return [value + p for p in perturbations]

    @staticmethod
    def perturb_integer(
        value: int,
        range_frac: float = 0.5
    ) -> List[int]:
        """Generate perturbations of an integer value."""
        delta = max(1, int(abs(value) * range_frac))
        return list(range(value - delta, value + delta + 1))

    @staticmethod
    def perturb_boolean(value: bool) -> List[bool]:
        """Flip boolean."""
        return [not value]

    def analyze_function(
        self,
        func: Callable,
        base_inputs: Dict[str, Any],
        vary_params: List[str] = None,
        result_extractor: Callable = None,
        comparison: str = "equality"  # equality, sign, threshold, direction
    ) -> Dict[str, SensitivityReport]:
        """
        Analyze sensitivity of a function to its inputs.

        Args:
            func: Function to analyze
            base_inputs: Baseline input values
            vary_params: Parameters to vary (default: all numeric)
            result_extractor: Function to extract comparable value from result
            comparison: How to compare results

        Returns:
            Dict mapping parameter names to sensitivity reports
        """
        if vary_params is None:
            # Vary all numeric parameters
            vary_params = [
                k for k, v in base_inputs.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]

        if result_extractor is None:
            result_extractor = lambda x: x

        # Get baseline result
        base_result = func(**base_inputs)
        base_value = result_extractor(base_result)

        reports = {}

        for param in vary_params:
            original = base_inputs[param]

            # Generate perturbations based on type
            if isinstance(original, bool):
                perturbations = self.perturb_boolean(original)
            elif isinstance(original, int):
                perturbations = self.perturb_integer(original)
            elif isinstance(original, float):
                perturbations = self.perturb_numeric(original)
            else:
                continue  # Can't perturb this type

            variations = []
            changed_count = 0

            for new_value in perturbations:
                if new_value == original:
                    continue

                # Run with varied parameter
                varied_inputs = deepcopy(base_inputs)
                varied_inputs[param] = new_value

                try:
                    varied_result = func(**varied_inputs)
                    varied_value = result_extractor(varied_result)

                    # Compare results
                    if comparison == "equality":
                        changed = varied_value != base_value
                    elif comparison == "sign":
                        changed = (varied_value > 0) != (base_value > 0) if base_value != 0 else varied_value != 0
                    elif comparison == "direction":
                        # For monotonic relationships
                        changed = (varied_value > base_value) if new_value > original else (varied_value < base_value)
                        changed = not changed  # We want to know if it's NOT monotonic
                    else:
                        changed = varied_value != base_value

                    if changed:
                        changed_count += 1

                    variations.append(Variation(
                        parameter=param,
                        original_value=original,
                        varied_value=new_value,
                        original_result=base_value,
                        varied_result=varied_value,
                        conclusion_changed=changed
                    ))

                except Exception:
                    # Variation caused error - note it but continue
                    pass

            # Determine sensitivity
            if not variations:
                sensitivity = Sensitivity.UNKNOWN
            elif changed_count == 0:
                sensitivity = Sensitivity.ROBUST
            elif changed_count == len(variations):
                sensitivity = Sensitivity.SENSITIVE
            else:
                # Find threshold if possible
                sensitivity = Sensitivity.THRESHOLD

            # Try to find threshold
            threshold = None
            if sensitivity == Sensitivity.THRESHOLD:
                # Find where it flips
                sorted_vars = sorted(variations, key=lambda v: v.varied_value)
                for i in range(len(sorted_vars) - 1):
                    if sorted_vars[i].conclusion_changed != sorted_vars[i+1].conclusion_changed:
                        threshold = (sorted_vars[i].varied_value + sorted_vars[i+1].varied_value) / 2
                        break

            reports[param] = SensitivityReport(
                parameter=param,
                sensitivity=sensitivity,
                threshold=threshold,
                variations_tested=len(variations),
                conclusion_changed_count=changed_count,
                details=variations
            )

        return reports

    def compare_scenarios(
        self,
        scenarios: Dict[str, Dict[str, Any]],
        func: Callable,
        result_extractor: Callable = None
    ) -> Dict[str, Tuple[Any, Any]]:
        """
        Compare results across named scenarios.

        Args:
            scenarios: Dict of scenario_name -> inputs
            func: Function to evaluate
            result_extractor: Function to extract comparable value

        Returns:
            Dict of scenario_name -> (inputs, result)
        """
        if result_extractor is None:
            result_extractor = lambda x: x

        results = {}
        for name, inputs in scenarios.items():
            try:
                result = func(**inputs)
                extracted = result_extractor(result)
                results[name] = (inputs, extracted)
            except Exception as e:
                results[name] = (inputs, f"ERROR: {e}")

        return results

    def find_boundary(
        self,
        func: Callable,
        base_inputs: Dict[str, Any],
        param: str,
        low: float,
        high: float,
        target_result: Any,
        result_extractor: Callable = None,
        tolerance: float = 0.001,
        max_iterations: int = 50
    ) -> Optional[float]:
        """
        Binary search for parameter value that produces target result.

        Useful for finding "at what point does X become true?"
        """
        if result_extractor is None:
            result_extractor = lambda x: x

        for _ in range(max_iterations):
            mid = (low + high) / 2

            inputs = deepcopy(base_inputs)
            inputs[param] = mid

            try:
                result = func(**inputs)
                value = result_extractor(result)

                if value == target_result:
                    return mid

                # Assume monotonic relationship
                inputs_low = deepcopy(base_inputs)
                inputs_low[param] = low
                low_result = result_extractor(func(**inputs_low))

                if low_result == target_result:
                    high = mid
                else:
                    low = mid

                if high - low < tolerance:
                    return mid

            except Exception:
                return None

        return (low + high) / 2


# Built-in analysis functions for common scenarios

def analyze_breakeven(
    revenue_per_unit: float,
    cost_per_unit: float,
    fixed_costs: float,
    units: float
) -> Dict[str, Any]:
    """Analyze profitability."""
    total_revenue = revenue_per_unit * units
    total_cost = cost_per_unit * units + fixed_costs
    profit = total_revenue - total_cost
    breakeven_units = fixed_costs / (revenue_per_unit - cost_per_unit) if revenue_per_unit > cost_per_unit else float('inf')

    return {
        "profit": profit,
        "profitable": profit > 0,
        "breakeven_units": breakeven_units,
        "margin_per_unit": revenue_per_unit - cost_per_unit,
        "total_revenue": total_revenue,
        "total_cost": total_cost
    }


def analyze_investment(
    initial: float,
    rate: float,
    years: int,
    inflation: float = 0.0
) -> Dict[str, Any]:
    """Analyze investment growth."""
    nominal_value = initial * ((1 + rate) ** years)
    real_rate = (1 + rate) / (1 + inflation) - 1
    real_value = initial * ((1 + real_rate) ** years)

    return {
        "nominal_value": nominal_value,
        "real_value": real_value,
        "nominal_gain": nominal_value - initial,
        "real_gain": real_value - initial,
        "effective_rate": real_rate,
        "beats_inflation": rate > inflation
    }


def analyze_decision(
    option_a_value: float,
    option_a_probability: float,
    option_b_value: float,
    option_b_probability: float
) -> Dict[str, Any]:
    """Analyze expected value decision."""
    ev_a = option_a_value * option_a_probability
    ev_b = option_b_value * option_b_probability

    return {
        "ev_a": ev_a,
        "ev_b": ev_b,
        "better_option": "A" if ev_a > ev_b else "B" if ev_b > ev_a else "TIE",
        "ev_difference": abs(ev_a - ev_b),
        "relative_advantage": ev_a / ev_b if ev_b != 0 else float('inf')
    }


# Tool implementation

def counterfactual_tool(
    operation: str,
    # For sensitivity analysis
    analysis_type: Optional[str] = None,
    base_inputs: Optional[Dict[str, Any]] = None,
    vary_params: Optional[List[str]] = None,
    result_key: Optional[str] = None,
    # For scenario comparison
    scenarios: Optional[Dict[str, Dict[str, Any]]] = None,
    # For boundary finding
    param: Optional[str] = None,
    low: Optional[float] = None,
    high: Optional[float] = None,
    target: Optional[Any] = None,
    # For custom function
    expression: Optional[str] = None,
) -> ToolResult:
    """Execute counterfactual analysis."""
    try:
        analyzer = CounterfactualAnalyzer()

        if operation == "sensitivity":
            # Analyze sensitivity of a built-in function
            if not analysis_type or not base_inputs:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output=None,
                    error="Need analysis_type and base_inputs"
                )

            # Select function
            if analysis_type == "breakeven":
                func = analyze_breakeven
                if result_key is None:
                    result_key = "profitable"
            elif analysis_type == "investment":
                func = analyze_investment
                if result_key is None:
                    result_key = "beats_inflation"
            elif analysis_type == "decision":
                func = analyze_decision
                if result_key is None:
                    result_key = "better_option"
            else:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output=None,
                    error=f"Unknown analysis_type: {analysis_type}. Valid: breakeven, investment, decision"
                )

            # Extract result value
            def extractor(result):
                if result_key and isinstance(result, dict):
                    return result.get(result_key)
                return result

            # Run analysis
            reports = analyzer.analyze_function(
                func,
                base_inputs,
                vary_params,
                extractor
            )

            # Format output
            output_lines = [f"Sensitivity Analysis ({analysis_type})", "=" * 40]

            # Get baseline
            baseline = func(**base_inputs)
            output_lines.append(f"Baseline: {baseline}")
            output_lines.append("")

            robust_params = []
            sensitive_params = []
            threshold_params = []

            for param_name, report in reports.items():
                if report.sensitivity == Sensitivity.ROBUST:
                    robust_params.append(param_name)
                elif report.sensitivity == Sensitivity.SENSITIVE:
                    sensitive_params.append(param_name)
                elif report.sensitivity == Sensitivity.THRESHOLD:
                    threshold_params.append((param_name, report.threshold))

            if robust_params:
                output_lines.append(f"ROBUST to: {', '.join(robust_params)}")
            if sensitive_params:
                output_lines.append(f"SENSITIVE to: {', '.join(sensitive_params)}")
            if threshold_params:
                for p, t in threshold_params:
                    output_lines.append(f"THRESHOLD at {p} = {t:.4g}")

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output="\n".join(output_lines),
                metadata={
                    "baseline": baseline,
                    "robust": robust_params,
                    "sensitive": sensitive_params,
                    "thresholds": {p: t for p, t in threshold_params},
                    "reports": {k: {
                        "sensitivity": v.sensitivity.value,
                        "threshold": v.threshold,
                        "changed_count": v.conclusion_changed_count,
                        "tested_count": v.variations_tested
                    } for k, v in reports.items()}
                }
            )

        elif operation == "scenarios":
            # Compare named scenarios
            if not scenarios or not analysis_type:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output=None,
                    error="Need scenarios and analysis_type"
                )

            if analysis_type == "breakeven":
                func = analyze_breakeven
            elif analysis_type == "investment":
                func = analyze_investment
            elif analysis_type == "decision":
                func = analyze_decision
            else:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output=None,
                    error=f"Unknown analysis_type: {analysis_type}"
                )

            results = analyzer.compare_scenarios(scenarios, func)

            output_lines = ["Scenario Comparison", "=" * 40]
            for name, (inputs, result) in results.items():
                output_lines.append(f"\n{name}:")
                output_lines.append(f"  Inputs: {inputs}")
                output_lines.append(f"  Result: {result}")

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output="\n".join(output_lines),
                metadata={"scenarios": {k: {"inputs": v[0], "result": v[1]} for k, v in results.items()}}
            )

        elif operation == "boundary":
            # Find parameter boundary
            if not analysis_type or not base_inputs or not param or low is None or high is None or target is None:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output=None,
                    error="Need analysis_type, base_inputs, param, low, high, target"
                )

            if analysis_type == "breakeven":
                func = analyze_breakeven
            elif analysis_type == "investment":
                func = analyze_investment
            elif analysis_type == "decision":
                func = analyze_decision
            else:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output=None,
                    error=f"Unknown analysis_type: {analysis_type}"
                )

            def extractor(result):
                if result_key and isinstance(result, dict):
                    return result.get(result_key)
                return result

            boundary = analyzer.find_boundary(
                func, base_inputs, param, low, high, target, extractor
            )

            if boundary is not None:
                # Verify
                test_inputs = deepcopy(base_inputs)
                test_inputs[param] = boundary
                test_result = func(**test_inputs)

                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=f"Boundary found: {param} = {boundary:.6g}\nAt this value: {test_result}",
                    metadata={"param": param, "boundary": boundary, "result_at_boundary": test_result}
                )
            else:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=f"No boundary found for {param} in range [{low}, {high}]",
                    metadata={"param": param, "boundary": None}
                )

        elif operation == "what_if":
            # Simple what-if with custom expression
            if not expression or not base_inputs:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output=None,
                    error="Need expression and base_inputs"
                )

            # Safe evaluation
            safe_globals = {
                "__builtins__": {},
                "abs": abs, "min": min, "max": max, "sum": sum,
                "round": round, "len": len,
                "sqrt": math.sqrt, "log": math.log, "exp": math.exp,
                "sin": math.sin, "cos": math.cos, "tan": math.tan,
                "pi": math.pi, "e": math.e
            }

            try:
                result = eval(expression, safe_globals, base_inputs)
            except Exception as e:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    output=None,
                    error=f"Expression error: {e}"
                )

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"With {base_inputs}:\n{expression} = {result}",
                metadata={"inputs": base_inputs, "expression": expression, "result": result}
            )

        else:
            return ToolResult(
                status=ToolStatus.INVALID_INPUT,
                output=None,
                error=f"Unknown operation: {operation}. Valid: sensitivity, scenarios, boundary, what_if"
            )

    except Exception as e:
        return ToolResult(status=ToolStatus.ERROR, output=None, error=str(e))


def create_counterfactual_tool() -> Tool:
    """Create counterfactual analysis tool."""
    return Tool(
        name="counterfactual",
        description="Sensitivity analysis and what-if reasoning. Tests how conclusions change under different assumptions. Reports 'robust to X, sensitive to Y, threshold at Z'.",
        parameters=[
            ToolParameter(
                name="operation",
                description="sensitivity (vary params), scenarios (compare named), boundary (find threshold), what_if (simple eval)",
                type="string",
                required=True,
                enum=["sensitivity", "scenarios", "boundary", "what_if"]
            ),
            ToolParameter(
                name="analysis_type",
                description="Built-in analysis: breakeven, investment, decision",
                type="string",
                required=False,
                enum=["breakeven", "investment", "decision"]
            ),
            ToolParameter(
                name="base_inputs",
                description="Input values: {param: value, ...}",
                type="object",
                required=False,
            ),
            ToolParameter(
                name="vary_params",
                description="Which parameters to vary (default: all numeric)",
                type="array",
                required=False,
            ),
            ToolParameter(
                name="result_key",
                description="Which result field to track for sensitivity",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="scenarios",
                description="For scenarios: {name: {inputs}, ...}",
                type="object",
                required=False,
            ),
            ToolParameter(
                name="param",
                description="For boundary: parameter to search",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="low",
                description="For boundary: search range low",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="high",
                description="For boundary: search range high",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="target",
                description="For boundary: target result value",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="expression",
                description="For what_if: expression to evaluate",
                type="string",
                required=False,
            ),
        ],
        execute_fn=counterfactual_tool,
        timeout_ms=30000,
    )


def register_counterfactual_tools(registry: ToolRegistry) -> None:
    """Register counterfactual analysis tools."""
    registry.register(create_counterfactual_tool())
