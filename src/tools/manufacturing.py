"""
Manufacturing Engineering Specialist Tool

Provides manufacturing analysis and metrics:
- Process capability (Cp, Cpk, Pp, Ppk)
- Overall Equipment Effectiveness (OEE)
- Tolerance stackup analysis
- Statistical Process Control (SPC)
- Lean metrics (cycle time, takt time)
- Cost estimation

This is epistemic scaffolding - encodes how manufacturing engineers think.
"""

from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math

from .registry import Tool, ToolParameter, ToolResult, ToolStatus, ToolRegistry


class CapabilityRating(Enum):
    """Process capability classification."""
    EXCELLENT = "excellent"     # Cpk >= 2.0
    GOOD = "good"               # 1.67 <= Cpk < 2.0
    ACCEPTABLE = "acceptable"   # 1.33 <= Cpk < 1.67
    MARGINAL = "marginal"       # 1.0 <= Cpk < 1.33
    INADEQUATE = "inadequate"   # Cpk < 1.0


class OEECategory(Enum):
    """OEE performance classification."""
    WORLD_CLASS = "world_class"   # >= 85%
    GOOD = "good"                  # 65-85%
    TYPICAL = "typical"            # 40-65%
    LOW = "low"                    # < 40%


@dataclass
class ToleranceStack:
    """A dimension in a tolerance stack."""
    name: str
    nominal: float
    plus_tol: float
    minus_tol: float

    @property
    def bilateral_tol(self) -> float:
        """Convert to symmetric bilateral tolerance."""
        return max(abs(self.plus_tol), abs(self.minus_tol))

    @property
    def mean(self) -> float:
        """Statistical mean (shifted by asymmetric tolerance)."""
        return self.nominal + (self.plus_tol + self.minus_tol) / 2


class ManufacturingAnalyzer:
    """
    Manufacturing engineering analysis engine.

    Encodes how manufacturing engineers approach problems:
    1. Define specifications (tolerances, targets)
    2. Measure process performance
    3. Calculate capability indices
    4. Identify improvement opportunities
    5. Control and monitor
    """

    def process_capability(
        self,
        data: List[float] = None,
        mean: float = None,
        std_dev: float = None,
        usl: float = None,
        lsl: float = None,
        sample_size: int = None
    ) -> Dict[str, Any]:
        """
        Calculate process capability indices.

        Cp = (USL - LSL) / (6 * sigma)  -- Potential capability
        Cpk = min(Cpu, Cpl)             -- Actual capability (accounts for centering)
        Pp, Ppk use overall std dev (long-term)
        """
        # Calculate stats from data if provided
        if data:
            n = len(data)
            mean = sum(data) / n
            std_dev = math.sqrt(sum((x - mean) ** 2 for x in data) / (n - 1))
            sample_size = n
        elif mean is None or std_dev is None:
            return {"error": "Need either data array or mean and std_dev"}

        if usl is None and lsl is None:
            return {"error": "Need at least one specification limit (USL or LSL)"}

        results = {
            "mean": round(mean, 4),
            "std_dev": round(std_dev, 6),
            "sample_size": sample_size
        }

        # Two-sided specification
        if usl is not None and lsl is not None:
            spec_range = usl - lsl
            results["usl"] = usl
            results["lsl"] = lsl
            results["spec_range"] = spec_range

            # Cp - potential capability (assumes centered)
            cp = spec_range / (6 * std_dev) if std_dev > 0 else float('inf')
            results["Cp"] = round(cp, 3)

            # Cpk - actual capability (accounts for centering)
            cpu = (usl - mean) / (3 * std_dev) if std_dev > 0 else float('inf')
            cpl = (mean - lsl) / (3 * std_dev) if std_dev > 0 else float('inf')
            cpk = min(cpu, cpl)
            results["Cpu"] = round(cpu, 3)
            results["Cpl"] = round(cpl, 3)
            results["Cpk"] = round(cpk, 3)

            # Centering ratio
            results["k_factor"] = round(abs(2 * (mean - (usl + lsl) / 2) / spec_range), 3)

        # One-sided upper spec
        elif usl is not None:
            results["usl"] = usl
            cpu = (usl - mean) / (3 * std_dev) if std_dev > 0 else float('inf')
            results["Cpu"] = round(cpu, 3)
            results["Cpk"] = round(cpu, 3)
            cpk = cpu

        # One-sided lower spec
        else:
            results["lsl"] = lsl
            cpl = (mean - lsl) / (3 * std_dev) if std_dev > 0 else float('inf')
            results["Cpl"] = round(cpl, 3)
            results["Cpk"] = round(cpl, 3)
            cpk = cpl

        # Rating
        if cpk >= 2.0:
            rating = CapabilityRating.EXCELLENT
            sigma_level = "6+ sigma"
        elif cpk >= 1.67:
            rating = CapabilityRating.GOOD
            sigma_level = "5 sigma"
        elif cpk >= 1.33:
            rating = CapabilityRating.ACCEPTABLE
            sigma_level = "4 sigma"
        elif cpk >= 1.0:
            rating = CapabilityRating.MARGINAL
            sigma_level = "3 sigma"
        else:
            rating = CapabilityRating.INADEQUATE
            sigma_level = "< 3 sigma"

        results["rating"] = rating.value
        results["sigma_level"] = sigma_level

        # Expected defect rate (approximate, for normal distribution)
        # Using Cpk directly: PPM ~ 2 * Phi(-3*Cpk) * 1e6
        # Simplified approximation
        if cpk >= 1.0:
            # Very approximate PPM for Cpk values
            ppm_approx = {2.0: 0.002, 1.67: 0.3, 1.33: 32, 1.0: 2700}
            results["estimated_ppm"] = ppm_approx.get(round(cpk, 1), f"~{10**(4-3*cpk):.0f}")
        else:
            results["estimated_ppm"] = "> 66,800"

        return results

    def oee(
        self,
        planned_time: float,
        actual_runtime: float,
        ideal_cycle_time: float,
        total_count: int,
        good_count: int
    ) -> Dict[str, Any]:
        """
        Calculate Overall Equipment Effectiveness.

        OEE = Availability x Performance x Quality

        Availability = Actual Runtime / Planned Production Time
        Performance = (Ideal Cycle Time x Total Count) / Actual Runtime
        Quality = Good Count / Total Count
        """
        # Availability
        availability = actual_runtime / planned_time if planned_time > 0 else 0

        # Performance (can exceed 100% if faster than ideal)
        ideal_time_needed = ideal_cycle_time * total_count
        performance = ideal_time_needed / actual_runtime if actual_runtime > 0 else 0
        performance = min(performance, 1.0)  # Cap at 100%

        # Quality (first pass yield)
        quality = good_count / total_count if total_count > 0 else 0

        # OEE
        oee = availability * performance * quality

        # Classification
        if oee >= 0.85:
            category = OEECategory.WORLD_CLASS
        elif oee >= 0.65:
            category = OEECategory.GOOD
        elif oee >= 0.40:
            category = OEECategory.TYPICAL
        else:
            category = OEECategory.LOW

        # Loss analysis
        availability_loss = planned_time - actual_runtime
        performance_loss = (1 - performance) * actual_runtime
        quality_loss = (total_count - good_count) * ideal_cycle_time

        return {
            "oee": round(oee * 100, 1),
            "availability": round(availability * 100, 1),
            "performance": round(performance * 100, 1),
            "quality": round(quality * 100, 1),
            "category": category.value,
            "losses": {
                "availability_loss_time": round(availability_loss, 2),
                "performance_loss_time": round(performance_loss, 2),
                "quality_loss_units": total_count - good_count
            },
            "production": {
                "planned_time": planned_time,
                "actual_runtime": actual_runtime,
                "total_count": total_count,
                "good_count": good_count,
                "reject_count": total_count - good_count
            },
            "improvement_priority": self._oee_improvement_priority(availability, performance, quality)
        }

    def _oee_improvement_priority(self, a: float, p: float, q: float) -> str:
        """Identify which OEE factor to improve first."""
        factors = [("availability", a), ("performance", p), ("quality", q)]
        factors.sort(key=lambda x: x[1])
        return f"Focus on {factors[0][0]} ({factors[0][1]*100:.1f}%) - biggest opportunity"

    def tolerance_stackup(
        self,
        dimensions: List[Dict[str, float]],  # [{name, nominal, plus_tol, minus_tol}]
        method: str = "worst_case"  # worst_case, rss, monte_carlo
    ) -> Dict[str, Any]:
        """
        Perform tolerance stackup analysis.

        Methods:
        - worst_case: All tolerances at max simultaneously (conservative)
        - rss: Root Sum Square (statistical, assumes normal distribution)
        """
        dims = [
            ToleranceStack(
                name=d["name"],
                nominal=d["nominal"],
                plus_tol=d.get("plus_tol", d.get("tolerance", 0)),
                minus_tol=d.get("minus_tol", -d.get("tolerance", 0))
            )
            for d in dimensions
        ]

        # Calculate nominal assembly dimension
        nominal_total = sum(d.nominal for d in dims)

        if method == "worst_case":
            # Maximum and minimum possible
            max_total = sum(d.nominal + d.plus_tol for d in dims)
            min_total = sum(d.nominal + d.minus_tol for d in dims)
            range_total = max_total - min_total

            return {
                "method": "worst_case",
                "nominal": round(nominal_total, 4),
                "maximum": round(max_total, 4),
                "minimum": round(min_total, 4),
                "total_tolerance": round(range_total, 4),
                "bilateral_tolerance": round(range_total / 2, 4),
                "dimensions": [d.__dict__ for d in dims],
                "interpretation": "100% of assemblies will be within these limits (if all parts are in spec)"
            }

        elif method == "rss":
            # Root Sum Square (statistical)
            # Assumes tolerances are 3-sigma, independent, normally distributed
            variance_sum = sum((d.bilateral_tol / 3) ** 2 for d in dims)
            rss_sigma = math.sqrt(variance_sum)
            rss_tolerance = 3 * rss_sigma  # 3-sigma limits

            # Compare to worst case
            wc_tolerance = sum(d.bilateral_tol for d in dims)
            reduction = (wc_tolerance - rss_tolerance) / wc_tolerance * 100

            return {
                "method": "rss",
                "nominal": round(nominal_total, 4),
                "maximum": round(nominal_total + rss_tolerance, 4),
                "minimum": round(nominal_total - rss_tolerance, 4),
                "total_tolerance": round(2 * rss_tolerance, 4),
                "bilateral_tolerance": round(rss_tolerance, 4),
                "worst_case_tolerance": round(wc_tolerance, 4),
                "reduction_vs_wc_pct": round(reduction, 1),
                "dimensions": [d.__dict__ for d in dims],
                "interpretation": f"99.73% (3-sigma) of assemblies within limits. {reduction:.1f}% tighter than worst-case."
            }

        else:
            return {"error": f"Unknown method: {method}. Use worst_case or rss"}

    def cycle_time_analysis(
        self,
        operations: List[Dict[str, float]],  # [{name, time}]
        demand_rate: float = None,  # units per hour
        available_time: float = None  # hours per shift
    ) -> Dict[str, Any]:
        """
        Analyze cycle time and identify bottlenecks.

        Calculates takt time if demand given.
        """
        total_cycle = sum(op["time"] for op in operations)

        # Find bottleneck
        bottleneck = max(operations, key=lambda x: x["time"])

        # Takt time (if demand provided)
        takt_time = None
        capacity = None
        if demand_rate and available_time:
            takt_time = (available_time * 60) / demand_rate  # minutes per unit
            capacity = available_time * 60 / bottleneck["time"]

        # Balance efficiency
        n_stations = len(operations)
        if n_stations > 0 and bottleneck["time"] > 0:
            balance_efficiency = total_cycle / (n_stations * bottleneck["time"]) * 100
        else:
            balance_efficiency = 0

        result = {
            "total_cycle_time": round(total_cycle, 2),
            "operations": operations,
            "bottleneck": {
                "operation": bottleneck["name"],
                "time": bottleneck["time"]
            },
            "balance_efficiency": round(balance_efficiency, 1),
            "n_stations": n_stations
        }

        if takt_time:
            result["takt_time"] = round(takt_time, 2)
            result["capacity_per_shift"] = round(capacity, 0)
            result["meets_demand"] = bottleneck["time"] <= takt_time
            if bottleneck["time"] > takt_time:
                result["bottleneck_gap"] = round(bottleneck["time"] - takt_time, 2)
                result["recommendation"] = f"Bottleneck exceeds takt by {result['bottleneck_gap']} min - need to reduce {bottleneck['name']} time or add parallel capacity"
            else:
                result["recommendation"] = "Cycle time meets takt time requirement"

        return result

    def spc_control_limits(
        self,
        data: List[float],
        subgroup_size: int = 5
    ) -> Dict[str, Any]:
        """
        Calculate SPC control limits for X-bar and R charts.
        """
        n = len(data)
        if n < subgroup_size * 2:
            return {"error": f"Need at least {subgroup_size * 2} data points"}

        # Create subgroups
        n_subgroups = n // subgroup_size
        subgroups = [
            data[i*subgroup_size:(i+1)*subgroup_size]
            for i in range(n_subgroups)
        ]

        # Calculate X-bar and R for each subgroup
        x_bars = [sum(sg) / len(sg) for sg in subgroups]
        ranges = [max(sg) - min(sg) for sg in subgroups]

        x_bar_bar = sum(x_bars) / len(x_bars)
        r_bar = sum(ranges) / len(ranges)

        # Control chart constants (for subgroup size 5)
        A2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
        D3 = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
        D4 = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}

        a2 = A2.get(subgroup_size, 0.577)
        d3 = D3.get(subgroup_size, 0)
        d4 = D4.get(subgroup_size, 2.114)

        # X-bar chart limits
        x_bar_ucl = x_bar_bar + a2 * r_bar
        x_bar_lcl = x_bar_bar - a2 * r_bar

        # R chart limits
        r_ucl = d4 * r_bar
        r_lcl = d3 * r_bar

        # Check for out of control points
        ooc_x = [i for i, x in enumerate(x_bars) if x > x_bar_ucl or x < x_bar_lcl]
        ooc_r = [i for i, r in enumerate(ranges) if r > r_ucl or r < r_lcl]

        return {
            "x_bar_chart": {
                "center_line": round(x_bar_bar, 4),
                "ucl": round(x_bar_ucl, 4),
                "lcl": round(x_bar_lcl, 4),
                "values": [round(x, 4) for x in x_bars],
                "out_of_control": ooc_x
            },
            "r_chart": {
                "center_line": round(r_bar, 4),
                "ucl": round(r_ucl, 4),
                "lcl": round(r_lcl, 4),
                "values": [round(r, 4) for r in ranges],
                "out_of_control": ooc_r
            },
            "subgroup_size": subgroup_size,
            "n_subgroups": n_subgroups,
            "process_status": "IN CONTROL" if not ooc_x and not ooc_r else "OUT OF CONTROL",
            "issues": ooc_x + ooc_r
        }


# Tool implementation for registry

def manufacturing_tool(
    operation: str,
    # Process capability
    data: Optional[List[float]] = None,
    mean: Optional[float] = None,
    std_dev: Optional[float] = None,
    usl: Optional[float] = None,
    lsl: Optional[float] = None,
    # OEE
    planned_time: Optional[float] = None,
    actual_runtime: Optional[float] = None,
    ideal_cycle_time: Optional[float] = None,
    total_count: Optional[int] = None,
    good_count: Optional[int] = None,
    # Tolerance stackup
    dimensions: Optional[List[Dict[str, float]]] = None,
    method: Optional[str] = None,
    # Cycle time
    operations: Optional[List[Dict[str, float]]] = None,
    demand_rate: Optional[float] = None,
    available_time: Optional[float] = None,
    # SPC
    subgroup_size: Optional[int] = None,
) -> ToolResult:
    """Execute manufacturing analysis."""
    try:
        mfg = ManufacturingAnalyzer()

        if operation == "capability":
            result = mfg.process_capability(
                data=data, mean=mean, std_dev=std_dev,
                usl=usl, lsl=lsl
            )

            if "error" in result:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=result["error"])

            output = f"Process Capability Analysis\n"
            output += "=" * 40 + "\n\n"
            output += f"Process: mean = {result['mean']}, std_dev = {result['std_dev']}\n"
            if 'spec_range' in result:
                output += f"Specification: LSL = {result.get('lsl')}, USL = {result.get('usl')}\n\n"
            else:
                output += f"Specification: {'USL = ' + str(result.get('usl')) if 'usl' in result else 'LSL = ' + str(result.get('lsl'))}\n\n"

            if 'Cp' in result:
                output += f"Cp  = {result['Cp']} (potential capability)\n"
            output += f"Cpk = {result['Cpk']} (actual capability)\n\n"
            output += f"Rating: {result['rating'].upper()} ({result['sigma_level']})\n"
            output += f"Estimated defect rate: {result['estimated_ppm']} PPM"

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=output,
                metadata=result
            )

        elif operation == "oee":
            if any(x is None for x in [planned_time, actual_runtime, ideal_cycle_time, total_count, good_count]):
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output="Need planned_time, actual_runtime, ideal_cycle_time, total_count, good_count"
                )

            result = mfg.oee(planned_time, actual_runtime, ideal_cycle_time, total_count, good_count)

            output = f"Overall Equipment Effectiveness (OEE)\n"
            output += "=" * 40 + "\n\n"
            output += f"OEE = {result['oee']}% ({result['category'].upper()})\n\n"
            output += f"  Availability: {result['availability']}%\n"
            output += f"  Performance:  {result['performance']}%\n"
            output += f"  Quality:      {result['quality']}%\n\n"
            output += f"Production: {result['production']['good_count']} good / {result['production']['total_count']} total\n"
            output += f"Rejects: {result['losses']['quality_loss_units']}\n\n"
            output += result['improvement_priority']

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=output,
                metadata=result
            )

        elif operation == "tolerance_stackup":
            if not dimensions:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output="Need dimensions list: [{name, nominal, plus_tol, minus_tol}]"
                )

            result = mfg.tolerance_stackup(dimensions, method or "worst_case")

            if "error" in result:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=result["error"])

            output = f"Tolerance Stackup Analysis ({result['method'].upper()})\n"
            output += "=" * 40 + "\n\n"
            output += f"Nominal assembly: {result['nominal']}\n"
            output += f"Maximum: {result['maximum']}\n"
            output += f"Minimum: {result['minimum']}\n"
            output += f"Total tolerance: +/- {result['bilateral_tolerance']}\n\n"
            if 'reduction_vs_wc_pct' in result:
                output += f"RSS vs Worst-Case: {result['reduction_vs_wc_pct']}% tighter\n"
            output += result['interpretation']

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=output,
                metadata=result
            )

        elif operation == "cycle_time":
            if not operations:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output="Need operations list: [{name, time}]"
                )

            result = mfg.cycle_time_analysis(operations, demand_rate, available_time)

            output = f"Cycle Time Analysis\n"
            output += "=" * 40 + "\n\n"
            output += f"Total cycle time: {result['total_cycle_time']} min\n"
            output += f"Bottleneck: {result['bottleneck']['operation']} ({result['bottleneck']['time']} min)\n"
            output += f"Balance efficiency: {result['balance_efficiency']}%\n"
            if 'takt_time' in result:
                output += f"\nTakt time: {result['takt_time']} min/unit\n"
                output += f"Capacity: {result['capacity_per_shift']} units/shift\n"
                output += f"Meets demand: {'YES' if result['meets_demand'] else 'NO'}\n"
            if 'recommendation' in result:
                output += f"\n{result['recommendation']}"

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=output,
                metadata=result
            )

        elif operation == "spc":
            if not data:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output="Need data array for SPC control limits"
                )

            result = mfg.spc_control_limits(data, subgroup_size or 5)

            if "error" in result:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=result["error"])

            output = f"SPC Control Limits (subgroup size = {result['subgroup_size']})\n"
            output += "=" * 40 + "\n\n"
            output += f"X-bar Chart:\n"
            output += f"  UCL = {result['x_bar_chart']['ucl']}\n"
            output += f"  CL  = {result['x_bar_chart']['center_line']}\n"
            output += f"  LCL = {result['x_bar_chart']['lcl']}\n\n"
            output += f"R Chart:\n"
            output += f"  UCL = {result['r_chart']['ucl']}\n"
            output += f"  CL  = {result['r_chart']['center_line']}\n"
            output += f"  LCL = {result['r_chart']['lcl']}\n\n"
            output += f"Status: {result['process_status']}"
            if result['issues']:
                output += f"\nOut-of-control points: {result['issues']}"

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=output,
                metadata=result
            )

        else:
            return ToolResult(
                status=ToolStatus.INVALID_INPUT,
                output=f"Unknown operation: {operation}. Valid: capability, oee, tolerance_stackup, cycle_time, spc"
            )

    except Exception as e:
        return ToolResult(
            status=ToolStatus.ERROR,
            output=None,
            error=str(e)
        )


def create_manufacturing_tool() -> Tool:
    """Create manufacturing engineering tool."""
    return Tool(
        name="manufacturing",
        description="Manufacturing analysis: process capability (Cp/Cpk), OEE, tolerance stackup (worst-case, RSS), cycle time/takt time, SPC control limits",
        parameters=[
            ToolParameter(
                name="operation",
                description="capability, oee, tolerance_stackup, cycle_time, spc",
                type="string",
                required=True,
                enum=["capability", "oee", "tolerance_stackup", "cycle_time", "spc"]
            ),
            ToolParameter(
                name="data",
                description="Measurement data array for capability or SPC",
                type="array",
                required=False,
            ),
            ToolParameter(
                name="mean",
                description="Process mean (alternative to data)",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="std_dev",
                description="Process standard deviation",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="usl",
                description="Upper specification limit",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="lsl",
                description="Lower specification limit",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="planned_time",
                description="Planned production time (OEE)",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="actual_runtime",
                description="Actual runtime (OEE)",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="ideal_cycle_time",
                description="Ideal cycle time per unit (OEE)",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="total_count",
                description="Total units produced (OEE)",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="good_count",
                description="Good units produced (OEE)",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="dimensions",
                description="Dimensions for tolerance stackup: [{name, nominal, plus_tol, minus_tol}]",
                type="array",
                required=False,
            ),
            ToolParameter(
                name="method",
                description="Stackup method: worst_case or rss",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="operations",
                description="Operations for cycle time: [{name, time}]",
                type="array",
                required=False,
            ),
            ToolParameter(
                name="demand_rate",
                description="Required units per hour",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="available_time",
                description="Available production hours per shift",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="subgroup_size",
                description="Subgroup size for SPC (default 5)",
                type="number",
                required=False,
            ),
        ],
        execute_fn=manufacturing_tool,
        timeout_ms=10000,
    )


def register_manufacturing_tools(registry: ToolRegistry) -> None:
    """Register manufacturing engineering tools."""
    registry.register(create_manufacturing_tool())
