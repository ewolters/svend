"""
Reliability Engineering Specialist Tool

Epistemic scaffolding for reliability and maintainability analysis.
Encodes how reliability engineers think about failure modes.

Operations:
- weibull: Weibull distribution analysis (β, η, MTTF)
- exponential: Constant failure rate analysis (λ, MTBF)
- availability: System availability calculation
- fault_tree: Fault tree analysis (AND/OR gates)
- redundancy: Redundancy analysis (parallel, k-of-n)
- bathtub: Life cycle phase determination
- reliability_block: Series/parallel system reliability
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import math

from .registry import Tool, ToolParameter, ToolResult, ToolStatus, ToolRegistry


class FailureDistribution(Enum):
    """Types of failure distributions."""
    EXPONENTIAL = "exponential"  # Constant failure rate (β=1)
    WEIBULL = "weibull"          # General case
    NORMAL = "normal"            # Wear-out failures
    LOGNORMAL = "lognormal"      # Fatigue, corrosion


class GateType(Enum):
    """Fault tree gate types."""
    AND = "and"     # All inputs must fail
    OR = "or"       # Any input causes failure
    KOFN = "k_of_n" # K of N must fail


@dataclass
class WeibullParams:
    """Weibull distribution parameters."""
    beta: float   # Shape parameter (β)
    eta: float    # Scale parameter (η) - characteristic life
    gamma: float = 0.0  # Location parameter (γ) - failure-free period

    @property
    def mttf(self) -> float:
        """Mean Time To Failure."""
        # MTTF = γ + η * Γ(1 + 1/β)
        return self.gamma + self.eta * math.gamma(1 + 1/self.beta)

    def reliability(self, t: float) -> float:
        """R(t) = exp(-((t-γ)/η)^β) for t > γ."""
        if t <= self.gamma:
            return 1.0
        return math.exp(-((t - self.gamma) / self.eta) ** self.beta)

    def failure_rate(self, t: float) -> float:
        """h(t) = (β/η) * ((t-γ)/η)^(β-1) for t > γ."""
        if t <= self.gamma:
            return 0.0
        return (self.beta / self.eta) * ((t - self.gamma) / self.eta) ** (self.beta - 1)

    def percentile_life(self, p: float) -> float:
        """Time at which p fraction have failed (B-life)."""
        # B_p = γ + η * (-ln(1-p))^(1/β)
        return self.gamma + self.eta * (-math.log(1 - p)) ** (1 / self.beta)


@dataclass
class FaultTreeNode:
    """Node in a fault tree."""
    name: str
    gate_type: Optional[GateType] = None
    probability: Optional[float] = None  # For basic events
    children: Optional[List['FaultTreeNode']] = None
    k: Optional[int] = None  # For k-of-n gates

    def evaluate(self) -> float:
        """Calculate probability of this event occurring."""
        if self.probability is not None:
            return self.probability

        if not self.children:
            return 0.0

        child_probs = [c.evaluate() for c in self.children]

        if self.gate_type == GateType.AND:
            # All must fail: P = P1 * P2 * ... * Pn
            result = 1.0
            for p in child_probs:
                result *= p
            return result

        elif self.gate_type == GateType.OR:
            # Any can fail: P = 1 - (1-P1)(1-P2)...(1-Pn)
            result = 1.0
            for p in child_probs:
                result *= (1 - p)
            return 1 - result

        elif self.gate_type == GateType.KOFN:
            # K of N must fail - combinatorial
            n = len(child_probs)
            k = self.k or 1
            return self._k_of_n_probability(child_probs, k)

        return 0.0

    def _k_of_n_probability(self, probs: List[float], k: int) -> float:
        """Calculate P(at least k of n fail) - assuming identical components."""
        n = len(probs)
        if k > n:
            return 0.0
        if k <= 0:
            return 1.0

        # For identical components, use binomial
        if len(set(round(p, 10) for p in probs)) == 1:
            p = probs[0]
            total = 0.0
            for i in range(k, n + 1):
                comb = math.comb(n, i)
                total += comb * (p ** i) * ((1 - p) ** (n - i))
            return total

        # For non-identical, use recursive formula
        return self._k_of_n_recursive(probs, k)

    def _k_of_n_recursive(self, probs: List[float], k: int) -> float:
        """Recursive k-of-n for non-identical components."""
        n = len(probs)
        if k <= 0:
            return 1.0
        if k > n:
            return 0.0
        if n == 1:
            return probs[0] if k == 1 else 0.0

        p_n = probs[-1]
        rest = probs[:-1]

        # P(k of n fail) = p_n * P(k-1 of n-1 fail) + (1-p_n) * P(k of n-1 fail)
        return p_n * self._k_of_n_recursive(rest, k - 1) + \
               (1 - p_n) * self._k_of_n_recursive(rest, k)


class ReliabilityAnalyzer:
    """
    Reliability engineering calculations.

    Provides analysis for:
    - Weibull life data analysis
    - Exponential reliability (constant failure rate)
    - System availability
    - Fault tree analysis
    - Redundancy effectiveness
    """

    def weibull_analysis(
        self,
        beta: float,
        eta: float,
        gamma: float = 0.0,
        time_points: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Analyze Weibull distribution.

        Args:
            beta: Shape parameter (β)
                  β < 1: Infant mortality (decreasing failure rate)
                  β = 1: Random failures (constant rate, exponential)
                  β > 1: Wear-out (increasing failure rate)
            eta: Scale parameter (η) - characteristic life (63.2% fail by η)
            gamma: Location parameter - failure-free period
            time_points: Optional specific times to evaluate

        Returns:
            MTTF, B-lives, reliability curve, interpretation
        """
        params = WeibullParams(beta=beta, eta=eta, gamma=gamma)

        # Determine failure mode
        if beta < 0.95:
            mode = "INFANT_MORTALITY"
            interpretation = f"Decreasing failure rate (β={beta:.2f}<1). Early failures dominate. Consider burn-in testing."
        elif beta <= 1.05:
            mode = "RANDOM"
            interpretation = f"Constant failure rate (β≈1). Random failures. Time-based PM not effective."
        elif beta <= 2.5:
            mode = "EARLY_WEAR"
            interpretation = f"Increasing failure rate (β={beta:.2f}). Gradual wear-out. Age-based replacement may help."
        else:
            mode = "RAPID_WEAR"
            interpretation = f"Strongly increasing failure rate (β={beta:.2f}). Predictable wear-out. PM highly effective."

        # Calculate B-lives (percentile lives)
        b_lives = {
            "B1": params.percentile_life(0.01),   # 1% failed
            "B10": params.percentile_life(0.10),  # 10% failed (common design life)
            "B50": params.percentile_life(0.50),  # Median life
        }

        # Default time points if not provided
        if time_points is None:
            max_t = eta * 2.5
            time_points = [max_t * i / 10 for i in range(11)]

        # Reliability curve
        reliability_curve = []
        for t in time_points:
            reliability_curve.append({
                "time": t,
                "reliability": params.reliability(t),
                "failure_rate": params.failure_rate(t),
                "cumulative_failure": 1 - params.reliability(t)
            })

        return {
            "parameters": {
                "beta": beta,
                "eta": eta,
                "gamma": gamma
            },
            "mttf": params.mttf,
            "characteristic_life": eta,
            "b_lives": b_lives,
            "failure_mode": mode,
            "interpretation": interpretation,
            "reliability_curve": reliability_curve
        }

    def exponential_analysis(
        self,
        failure_rate: Optional[float] = None,
        mtbf: Optional[float] = None,
        mission_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Analyze exponential (constant failure rate) reliability.

        Args:
            failure_rate: λ (failures per unit time)
            mtbf: Mean Time Between Failures (= 1/λ)
            mission_time: Time period to analyze

        Returns:
            MTBF, reliability for mission, failure probability
        """
        if failure_rate is None and mtbf is None:
            return {"error": "Must provide either failure_rate or mtbf"}

        if failure_rate is None:
            failure_rate = 1 / mtbf
        else:
            mtbf = 1 / failure_rate

        result = {
            "failure_rate_lambda": failure_rate,
            "mtbf": mtbf,
            "distribution": "exponential",
            "note": "Constant failure rate - memoryless property applies"
        }

        if mission_time is not None:
            reliability = math.exp(-failure_rate * mission_time)
            result["mission_analysis"] = {
                "mission_time": mission_time,
                "reliability": reliability,
                "failure_probability": 1 - reliability,
                "expected_failures": failure_rate * mission_time
            }

        return result

    def availability(
        self,
        mtbf: float,
        mttr: float,
        include_logistics: bool = False,
        mldt: float = 0.0
    ) -> Dict[str, Any]:
        """
        Calculate system availability.

        Args:
            mtbf: Mean Time Between Failures
            mttr: Mean Time To Repair
            include_logistics: Include logistics delay time
            mldt: Mean Logistics Delay Time

        Returns:
            Inherent, achieved, and operational availability
        """
        # Inherent availability (ideal conditions)
        a_inherent = mtbf / (mtbf + mttr)

        # Achieved availability (includes preventive maintenance)
        # For simplicity, assume PM adds 10% to downtime
        pm_time = mttr * 0.1
        a_achieved = mtbf / (mtbf + mttr + pm_time)

        # Operational availability (includes logistics)
        total_downtime = mttr + mldt if include_logistics else mttr
        a_operational = mtbf / (mtbf + total_downtime)

        return {
            "mtbf": mtbf,
            "mttr": mttr,
            "mldt": mldt if include_logistics else None,
            "availability": {
                "inherent": a_inherent,
                "achieved": a_achieved,
                "operational": a_operational if include_logistics else None
            },
            "downtime_percent": (1 - a_inherent) * 100,
            "uptime_ratio": f"{mtbf:.1f}:{mttr:.1f}"
        }

    def fault_tree(
        self,
        tree_structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze fault tree.

        Args:
            tree_structure: Nested dict describing fault tree
                {
                    "name": "System Failure",
                    "gate": "or",
                    "children": [
                        {"name": "Component A", "probability": 0.01},
                        {"name": "Subsystem B", "gate": "and", "children": [...]}
                    ]
                }

        Returns:
            Top event probability, cut sets, importance measures
        """
        def build_tree(node_dict: Dict) -> FaultTreeNode:
            """Recursively build fault tree from dict."""
            if "probability" in node_dict:
                return FaultTreeNode(
                    name=node_dict["name"],
                    probability=node_dict["probability"]
                )

            gate_str = node_dict.get("gate", "or").lower()
            if gate_str == "and":
                gate = GateType.AND
            elif gate_str == "or":
                gate = GateType.OR
            elif gate_str.startswith("k_of_n") or "of" in gate_str:
                gate = GateType.KOFN
            else:
                gate = GateType.OR

            children = [build_tree(c) for c in node_dict.get("children", [])]

            return FaultTreeNode(
                name=node_dict["name"],
                gate_type=gate,
                children=children,
                k=node_dict.get("k")
            )

        root = build_tree(tree_structure)
        top_probability = root.evaluate()

        # Find minimal cut sets (simplified - just list basic events)
        basic_events = []
        def collect_events(node: FaultTreeNode):
            if node.probability is not None:
                basic_events.append({
                    "name": node.name,
                    "probability": node.probability
                })
            if node.children:
                for c in node.children:
                    collect_events(c)

        collect_events(root)

        # Fussell-Vesely importance (approximate)
        importance = []
        for event in basic_events:
            # FV = (P_top with event) - (P_top without event) / P_top
            # Simplified: just use contribution ratio
            contribution = event["probability"] / top_probability if top_probability > 0 else 0
            importance.append({
                "event": event["name"],
                "probability": event["probability"],
                "contribution_ratio": min(contribution, 1.0)
            })

        importance.sort(key=lambda x: x["probability"], reverse=True)

        return {
            "top_event": tree_structure["name"],
            "top_event_probability": top_probability,
            "basic_events_count": len(basic_events),
            "basic_events": basic_events,
            "importance_ranking": importance,
            "interpretation": self._interpret_fault_tree(top_probability, importance)
        }

    def _interpret_fault_tree(
        self,
        top_prob: float,
        importance: List[Dict]
    ) -> str:
        """Generate interpretation of fault tree results."""
        if top_prob < 0.001:
            severity = "LOW"
            action = "System reliability is acceptable"
        elif top_prob < 0.01:
            severity = "MODERATE"
            action = "Consider improving highest-importance components"
        elif top_prob < 0.1:
            severity = "HIGH"
            action = "Reliability improvement required"
        else:
            severity = "CRITICAL"
            action = "Major redesign needed"

        top_contributors = [e["event"] for e in importance[:3]]

        return f"{severity} risk (P={top_prob:.4f}). {action}. Top contributors: {', '.join(top_contributors)}"

    def redundancy(
        self,
        component_reliability: float,
        n_parallel: int = 2,
        k_required: int = 1,
        standby: bool = False,
        switching_reliability: float = 1.0
    ) -> Dict[str, Any]:
        """
        Analyze redundancy effectiveness.

        Args:
            component_reliability: R of each identical component
            n_parallel: Total number of parallel components
            k_required: Minimum needed to function (k-of-n)
            standby: If True, cold standby; if False, active parallel
            switching_reliability: For standby, P(switch works)

        Returns:
            System reliability, improvement factor, cost analysis
        """
        r = component_reliability
        n = n_parallel
        k = k_required

        if standby:
            # Cold standby: components don't age while on standby
            # R_sys = R1 * (1 + λt + (λt)²/2! + ... for n-1 spares)
            # Simplified: exponential assumption
            if r > 0.99:
                # High reliability: limited improvement from standby
                r_system = r * switching_reliability * (1 + (1-r) * (n-1))
            else:
                # Use actual standby calculation
                r_system = 0
                for i in range(n):
                    # Probability of exactly i failures before mission end, still working
                    term = r * ((-math.log(r)) ** i) / math.factorial(i) * (switching_reliability ** i)
                    r_system += term
                r_system = min(r_system, 1.0)
        else:
            # Active parallel k-of-n
            # R_sys = Σ C(n,i) * R^i * (1-R)^(n-i) for i = k to n
            r_system = 0.0
            for i in range(k, n + 1):
                comb = math.comb(n, i)
                r_system += comb * (r ** i) * ((1 - r) ** (n - i))

        # Single component failure probability
        f_single = 1 - r
        f_system = 1 - r_system

        # Improvement factor
        improvement = r_system / r if r > 0 else float('inf')
        failure_reduction = f_single / f_system if f_system > 0 else float('inf')

        return {
            "configuration": {
                "n_components": n,
                "k_required": k,
                "type": "standby" if standby else "active_parallel",
                "notation": f"{k}-of-{n}" if not standby else f"{n} standby"
            },
            "component_reliability": r,
            "system_reliability": r_system,
            "improvement_factor": improvement,
            "failure_reduction": failure_reduction,
            "unreliability": {
                "single": f_single,
                "system": f_system
            },
            "interpretation": self._interpret_redundancy(r, r_system, n, k, standby)
        }

    def _interpret_redundancy(
        self,
        r_comp: float,
        r_sys: float,
        n: int,
        k: int,
        standby: bool
    ) -> str:
        """Generate interpretation of redundancy analysis."""
        improvement = (r_sys - r_comp) / (1 - r_comp) * 100 if r_comp < 1 else 0

        if standby:
            config = f"{n} standby components"
        else:
            config = f"{k}-of-{n} active parallel"

        if improvement > 90:
            return f"{config} achieves {improvement:.1f}% reduction in unreliability. Excellent redundancy."
        elif improvement > 50:
            return f"{config} achieves {improvement:.1f}% reduction in unreliability. Good improvement."
        else:
            return f"{config} achieves {improvement:.1f}% reduction in unreliability. Limited benefit - consider more redundancy or improving component reliability."

    def bathtub_phase(
        self,
        failure_rate_data: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Determine bathtub curve phase from failure rate data.

        Args:
            failure_rate_data: List of {"time": t, "failure_rate": λ}

        Returns:
            Phase identification, transition points, recommendations
        """
        if len(failure_rate_data) < 3:
            return {"error": "Need at least 3 data points"}

        # Sort by time
        data = sorted(failure_rate_data, key=lambda x: x["time"])

        # Calculate trend (slope) for each segment
        phases = []
        for i in range(len(data) - 1):
            t1, λ1 = data[i]["time"], data[i]["failure_rate"]
            t2, λ2 = data[i+1]["time"], data[i+1]["failure_rate"]

            if t2 - t1 == 0:
                slope = 0
            else:
                slope = (λ2 - λ1) / (t2 - t1)

            # Normalize slope relative to failure rate
            avg_λ = (λ1 + λ2) / 2
            norm_slope = slope / avg_λ if avg_λ > 0 else 0

            if norm_slope < -0.1:
                phase = "INFANT_MORTALITY"
            elif norm_slope > 0.1:
                phase = "WEAR_OUT"
            else:
                phase = "USEFUL_LIFE"

            phases.append({
                "time_start": t1,
                "time_end": t2,
                "slope": slope,
                "phase": phase
            })

        # Find current phase (most recent)
        current_phase = phases[-1]["phase"]

        # Find transitions
        transitions = []
        for i in range(len(phases) - 1):
            if phases[i]["phase"] != phases[i+1]["phase"]:
                transitions.append({
                    "from": phases[i]["phase"],
                    "to": phases[i+1]["phase"],
                    "time": phases[i]["time_end"]
                })

        # Recommendations
        recommendations = {
            "INFANT_MORTALITY": "Focus on quality control, burn-in testing, and early-life screening",
            "USEFUL_LIFE": "Constant failure rate - predictive maintenance less effective. Focus on redundancy.",
            "WEAR_OUT": "Age-based replacement effective. Consider condition monitoring."
        }

        return {
            "current_phase": current_phase,
            "phase_segments": phases,
            "transitions": transitions,
            "recommendation": recommendations[current_phase],
            "data_points": len(data)
        }

    def reliability_block(
        self,
        blocks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate reliability block diagram.

        Args:
            blocks: List of blocks with structure:
                {"name": "A", "reliability": 0.99}  # Single component
                {"name": "subsystem", "type": "series", "components": [...]}
                {"name": "subsystem", "type": "parallel", "components": [...]}

        Returns:
            System reliability, critical components
        """
        def calc_block_reliability(block: Dict) -> float:
            """Recursively calculate block reliability."""
            if "reliability" in block:
                return block["reliability"]

            components = block.get("components", [])
            if not components:
                return 1.0

            comp_reliabilities = [calc_block_reliability(c) for c in components]

            block_type = block.get("type", "series").lower()

            if block_type == "series":
                # R_series = R1 * R2 * ... * Rn
                result = 1.0
                for r in comp_reliabilities:
                    result *= r
                return result
            elif block_type == "parallel":
                # R_parallel = 1 - (1-R1)(1-R2)...(1-Rn)
                result = 1.0
                for r in comp_reliabilities:
                    result *= (1 - r)
                return 1 - result
            else:
                return 1.0

        # Calculate overall system (series of all top-level blocks)
        if len(blocks) == 1:
            system_reliability = calc_block_reliability(blocks[0])
        else:
            # Multiple top-level blocks in series
            system_reliability = 1.0
            for block in blocks:
                system_reliability *= calc_block_reliability(block)

        # Find critical components (lowest reliability in series paths)
        def find_critical(block: Dict, path_reliability: float = 1.0) -> List[Dict]:
            critical = []
            if "reliability" in block:
                return [{
                    "name": block["name"],
                    "reliability": block["reliability"],
                    "impact": path_reliability * (1 - block["reliability"])
                }]

            components = block.get("components", [])
            block_type = block.get("type", "series").lower()

            for c in components:
                critical.extend(find_critical(c, path_reliability))

            return critical

        critical_components = []
        for block in blocks:
            critical_components.extend(find_critical(block))

        critical_components.sort(key=lambda x: x["reliability"])

        return {
            "system_reliability": system_reliability,
            "system_unreliability": 1 - system_reliability,
            "block_count": len(blocks),
            "critical_components": critical_components[:5],  # Top 5 most critical
            "interpretation": f"System R = {system_reliability:.6f}. Most critical: {critical_components[0]['name'] if critical_components else 'N/A'}"
        }


def create_reliability_tool() -> Tool:
    """Create the reliability engineering tool."""
    analyzer = ReliabilityAnalyzer()

    def execute(operation: str, **kwargs) -> ToolResult:
        """Execute a reliability operation."""
        operations = {
            "weibull": analyzer.weibull_analysis,
            "exponential": analyzer.exponential_analysis,
            "availability": analyzer.availability,
            "fault_tree": analyzer.fault_tree,
            "redundancy": analyzer.redundancy,
            "bathtub": analyzer.bathtub_phase,
            "reliability_block": analyzer.reliability_block
        }

        if operation not in operations:
            return ToolResult(
                status=ToolStatus.ERROR,
                output=None,
                error=f"Unknown operation: {operation}. Available: {list(operations.keys())}"
            )

        try:
            result = operations[operation](**kwargs)
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=result
            )
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                output=None,
                error=str(e)
            )

    return Tool(
        name="reliability",
        description="""Reliability engineering analysis tool.

Operations:
- weibull: Weibull distribution analysis
  Args: beta (float), eta (float), gamma (float, default=0), time_points (list, optional)
  Returns: MTTF, B-lives, reliability curve, failure mode interpretation

- exponential: Constant failure rate analysis
  Args: failure_rate (float) OR mtbf (float), mission_time (float, optional)
  Returns: MTBF, mission reliability

- availability: System availability calculation
  Args: mtbf (float), mttr (float), include_logistics (bool), mldt (float)
  Returns: Inherent, achieved, operational availability

- fault_tree: Fault tree analysis
  Args: tree_structure (dict with name, gate, children, probability)
  Returns: Top event probability, importance ranking

- redundancy: Redundancy effectiveness analysis
  Args: component_reliability (float), n_parallel (int), k_required (int), standby (bool)
  Returns: System reliability, improvement factor

- bathtub: Life cycle phase determination
  Args: failure_rate_data (list of {time, failure_rate})
  Returns: Current phase, transitions, recommendations

- reliability_block: Series/parallel system reliability
  Args: blocks (list of block dicts with reliability, type, components)
  Returns: System reliability, critical components""",
        parameters=[
            ToolParameter(
                name="operation",
                type="string",
                description="Operation to perform",
                required=True
            )
        ],
        execute_fn=execute
    )


def register_reliability_tools(registry: ToolRegistry) -> None:
    """Register reliability engineering tools."""
    registry.register(create_reliability_tool())
