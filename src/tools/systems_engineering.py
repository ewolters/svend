"""
Systems Engineering Specialist Tool

Provides systems engineering analysis and frameworks:
- Requirements decomposition and traceability
- Interface analysis (N-squared diagrams)
- Trade studies and decision matrices
- V-model verification planning
- FMEA (Failure Modes and Effects Analysis)
- Technical Performance Measures (TPMs)

This is epistemic scaffolding - encodes how systems engineers think.
"""

from typing import Optional, Dict, Any, List, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import math

from .registry import Tool, ToolParameter, ToolResult, ToolStatus, ToolRegistry


class RequirementType(Enum):
    """Types of requirements."""
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    INTERFACE = "interface"
    CONSTRAINT = "constraint"
    DERIVED = "derived"


class VerificationMethod(Enum):
    """How requirements are verified (IADT)."""
    INSPECTION = "inspection"
    ANALYSIS = "analysis"
    DEMONSTRATION = "demonstration"
    TEST = "test"


class RiskSeverity(Enum):
    """FMEA severity levels."""
    CATASTROPHIC = 10
    CRITICAL = 8
    MAJOR = 6
    MINOR = 4
    NEGLIGIBLE = 2


class RiskOccurrence(Enum):
    """FMEA occurrence levels."""
    CERTAIN = 10
    LIKELY = 8
    MODERATE = 6
    UNLIKELY = 4
    RARE = 2


class RiskDetection(Enum):
    """FMEA detection levels."""
    UNDETECTABLE = 10
    VERY_LOW = 8
    LOW = 6
    MODERATE = 4
    HIGH = 2


@dataclass
class Requirement:
    """A system requirement."""
    id: str
    text: str
    type: RequirementType
    parent_id: Optional[str] = None
    verification: VerificationMethod = VerificationMethod.TEST
    priority: int = 2  # 1=critical, 2=essential, 3=desirable
    status: str = "draft"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "type": self.type.value,
            "parent_id": self.parent_id,
            "verification": self.verification.value,
            "priority": self.priority,
            "status": self.status
        }


@dataclass
class TradeStudyOption:
    """An option in a trade study."""
    name: str
    scores: Dict[str, float]  # criterion -> score (0-10)
    cost: float = 0
    risk: float = 0

    @property
    def weighted_score(self) -> float:
        return sum(self.scores.values())


@dataclass
class FMEAItem:
    """A failure mode in FMEA."""
    component: str
    failure_mode: str
    cause: str
    effect: str
    severity: int  # 1-10
    occurrence: int  # 1-10
    detection: int  # 1-10

    @property
    def rpn(self) -> int:
        """Risk Priority Number = S x O x D."""
        return self.severity * self.occurrence * self.detection

    @property
    def criticality(self) -> str:
        rpn = self.rpn
        if rpn >= 200:
            return "CRITICAL"
        elif rpn >= 100:
            return "HIGH"
        elif rpn >= 50:
            return "MEDIUM"
        else:
            return "LOW"


class SystemsEngineer:
    """
    Systems engineering analysis engine.

    Encodes how systems engineers approach problems:
    1. Define requirements (what the system must do)
    2. Decompose into subsystems
    3. Analyze interfaces
    4. Evaluate trades
    5. Verify/validate
    """

    def decompose_requirements(
        self,
        system_requirement: str,
        req_type: str = "functional"
    ) -> Dict[str, Any]:
        """
        Suggest a decomposition structure for a high-level requirement.

        This provides templates for how to break down requirements.
        """
        templates = {
            "functional": [
                "The system shall provide {capability}",
                "The system shall accept {input}",
                "The system shall produce {output}",
                "The system shall interface with {external_system}",
                "The system shall store {data}",
                "The system shall process {information}",
            ],
            "performance": [
                "The system shall achieve {metric} of at least {value} {unit}",
                "The system shall complete {operation} within {time}",
                "The system shall support {quantity} simultaneous {things}",
                "The system shall operate at {rate} {unit}",
            ],
            "interface": [
                "The system shall send {data} to {destination}",
                "The system shall receive {data} from {source}",
                "The system shall use {protocol} for {communication}",
            ],
            "constraint": [
                "The system shall not exceed {limit} {unit}",
                "The system shall operate within {range}",
                "The system shall comply with {standard}",
                "The system shall be compatible with {existing_system}",
            ]
        }

        verification_hints = {
            "functional": VerificationMethod.DEMONSTRATION,
            "performance": VerificationMethod.TEST,
            "interface": VerificationMethod.TEST,
            "constraint": VerificationMethod.ANALYSIS,
        }

        return {
            "original": system_requirement,
            "decomposition_templates": templates.get(req_type, templates["functional"]),
            "suggested_verification": verification_hints.get(req_type, VerificationMethod.TEST).value,
            "guidance": f"Break '{system_requirement}' into 3-7 child requirements using these patterns",
            "traceability": "Each child requirement should trace to this parent"
        }

    def create_n_squared(
        self,
        subsystems: List[str],
        interfaces: List[Tuple[str, str, str]]  # (from, to, description)
    ) -> Dict[str, Any]:
        """
        Create N-squared interface diagram data.

        N-squared diagrams show all interfaces between N subsystems.
        """
        n = len(subsystems)
        matrix = [[None for _ in range(n)] for _ in range(n)]

        # Index lookup
        idx = {s: i for i, s in enumerate(subsystems)}

        # Fill matrix
        interface_count = 0
        for from_sys, to_sys, desc in interfaces:
            if from_sys in idx and to_sys in idx:
                i, j = idx[from_sys], idx[to_sys]
                matrix[i][j] = desc
                interface_count += 1

        # Calculate complexity metrics
        max_interfaces = n * (n - 1)  # Excluding diagonal
        complexity = interface_count / max_interfaces if max_interfaces > 0 else 0

        # Find highly coupled subsystems
        coupling = {}
        for sys in subsystems:
            i = idx[sys]
            outgoing = sum(1 for j in range(n) if matrix[i][j] is not None)
            incoming = sum(1 for j in range(n) if matrix[j][i] is not None)
            coupling[sys] = {"outgoing": outgoing, "incoming": incoming, "total": outgoing + incoming}

        most_coupled = max(coupling.items(), key=lambda x: x[1]["total"])

        return {
            "subsystems": subsystems,
            "matrix": matrix,
            "interface_count": interface_count,
            "max_possible": max_interfaces,
            "complexity_ratio": round(complexity, 3),
            "coupling_analysis": coupling,
            "most_coupled": {"subsystem": most_coupled[0], "interfaces": most_coupled[1]["total"]},
            "recommendation": "High coupling suggests need for interface control documents (ICDs)" if complexity > 0.5 else "Moderate coupling - manageable interfaces"
        }

    def trade_study(
        self,
        criteria: Dict[str, float],  # criterion -> weight
        options: List[Dict[str, Any]]  # [{name, scores: {criterion: score}}]
    ) -> Dict[str, Any]:
        """
        Perform a weighted trade study.

        Criteria weights should sum to 1.0 (normalized if not).
        Scores should be 0-10 scale.
        """
        # Normalize weights
        total_weight = sum(criteria.values())
        weights = {k: v / total_weight for k, v in criteria.items()}

        # Calculate weighted scores
        results = []
        for opt in options:
            weighted = 0
            score_breakdown = {}
            for crit, weight in weights.items():
                score = opt.get("scores", {}).get(crit, 5)  # Default 5 if missing
                contribution = score * weight
                weighted += contribution
                score_breakdown[crit] = {
                    "raw_score": score,
                    "weight": round(weight, 3),
                    "weighted": round(contribution, 3)
                }

            results.append({
                "name": opt["name"],
                "weighted_score": round(weighted, 3),
                "breakdown": score_breakdown,
                "cost": opt.get("cost", 0),
                "risk": opt.get("risk", 0)
            })

        # Sort by weighted score
        results.sort(key=lambda x: x["weighted_score"], reverse=True)

        # Sensitivity analysis - which criterion most affects the winner?
        winner = results[0]["name"]
        sensitivity = {}
        for crit in criteria:
            # What if this criterion had 0 weight?
            alt_weights = {k: v for k, v in weights.items() if k != crit}
            alt_total = sum(alt_weights.values())
            if alt_total > 0:
                alt_weights = {k: v / alt_total for k, v in alt_weights.items()}

                alt_scores = []
                for opt in options:
                    alt_weighted = sum(
                        opt.get("scores", {}).get(c, 5) * w
                        for c, w in alt_weights.items()
                    )
                    alt_scores.append((opt["name"], alt_weighted))

                alt_winner = max(alt_scores, key=lambda x: x[1])[0]
                sensitivity[crit] = "Winner changes" if alt_winner != winner else "Winner unchanged"

        return {
            "criteria": weights,
            "results": results,
            "winner": winner,
            "margin": round(results[0]["weighted_score"] - results[1]["weighted_score"], 3) if len(results) > 1 else 0,
            "sensitivity": sensitivity,
            "recommendation": f"Select '{winner}' with score {results[0]['weighted_score']:.2f}/10"
        }

    def fmea_analysis(
        self,
        failure_modes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform FMEA (Failure Modes and Effects Analysis).

        Input: [{component, failure_mode, cause, effect, severity, occurrence, detection}]
        """
        items = []
        for fm in failure_modes:
            item = FMEAItem(
                component=fm.get("component", "Unknown"),
                failure_mode=fm.get("failure_mode", "Unknown"),
                cause=fm.get("cause", "Unknown"),
                effect=fm.get("effect", "Unknown"),
                severity=fm.get("severity", 5),
                occurrence=fm.get("occurrence", 5),
                detection=fm.get("detection", 5)
            )
            items.append(item)

        # Sort by RPN
        items.sort(key=lambda x: x.rpn, reverse=True)

        results = []
        for item in items:
            results.append({
                "component": item.component,
                "failure_mode": item.failure_mode,
                "cause": item.cause,
                "effect": item.effect,
                "severity": item.severity,
                "occurrence": item.occurrence,
                "detection": item.detection,
                "rpn": item.rpn,
                "criticality": item.criticality
            })

        # Summary statistics
        rpns = [item.rpn for item in items]
        critical_count = sum(1 for item in items if item.criticality == "CRITICAL")
        high_count = sum(1 for item in items if item.criticality == "HIGH")

        return {
            "failure_modes": results,
            "summary": {
                "total_items": len(items),
                "critical_count": critical_count,
                "high_count": high_count,
                "max_rpn": max(rpns) if rpns else 0,
                "avg_rpn": round(sum(rpns) / len(rpns), 1) if rpns else 0,
            },
            "top_risks": results[:3],
            "recommendation": f"Address {critical_count} critical and {high_count} high-priority failure modes first"
        }

    def verification_matrix(
        self,
        requirements: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Create a verification cross-reference matrix.

        Maps requirements to verification methods (IADT).
        """
        matrix = []
        method_counts = {"inspection": 0, "analysis": 0, "demonstration": 0, "test": 0}

        for req in requirements:
            method = req.get("verification", "test").lower()
            if method in method_counts:
                method_counts[method] += 1

            matrix.append({
                "id": req.get("id", "REQ-X"),
                "requirement": req.get("text", "")[:50] + "...",
                "type": req.get("type", "functional"),
                "verification": method,
                "priority": req.get("priority", 2)
            })

        # Analysis
        total = len(requirements)
        test_heavy = method_counts["test"] / total if total > 0 else 0

        return {
            "matrix": matrix,
            "method_distribution": method_counts,
            "total_requirements": total,
            "analysis": {
                "test_percentage": round(test_heavy * 100, 1),
                "recommendation": "Consider analysis for derived requirements to reduce test burden" if test_heavy > 0.7 else "Good verification method distribution"
            }
        }

    def tpm_tracker(
        self,
        measures: List[Dict[str, Any]]  # [{name, target, current, unit, trend}]
    ) -> Dict[str, Any]:
        """
        Track Technical Performance Measures (TPMs).

        TPMs track how well the system is progressing toward key targets.
        """
        results = []
        at_risk = []

        for tpm in measures:
            name = tpm.get("name", "Unknown")
            target = tpm.get("target", 0)
            current = tpm.get("current", 0)
            unit = tpm.get("unit", "")
            trend = tpm.get("trend", "stable")  # improving, degrading, stable

            # Calculate margin
            if target != 0:
                margin_pct = ((target - current) / abs(target)) * 100
                # Positive margin = haven't reached target
                # Negative margin = exceeded target
            else:
                margin_pct = 0

            status = "GREEN"
            if abs(margin_pct) > 20:
                status = "RED"
            elif abs(margin_pct) > 10:
                status = "YELLOW"

            if status != "GREEN" or trend == "degrading":
                at_risk.append(name)

            results.append({
                "name": name,
                "target": target,
                "current": current,
                "unit": unit,
                "margin_pct": round(margin_pct, 1),
                "trend": trend,
                "status": status
            })

        return {
            "measures": results,
            "summary": {
                "total": len(measures),
                "green": sum(1 for r in results if r["status"] == "GREEN"),
                "yellow": sum(1 for r in results if r["status"] == "YELLOW"),
                "red": sum(1 for r in results if r["status"] == "RED"),
            },
            "at_risk": at_risk,
            "recommendation": f"Monitor {len(at_risk)} at-risk TPMs closely" if at_risk else "All TPMs on track"
        }


# Tool implementation for registry

def systems_engineering_tool(
    operation: str,
    # Decomposition
    requirement: Optional[str] = None,
    req_type: Optional[str] = None,
    # N-squared
    subsystems: Optional[List[str]] = None,
    interfaces: Optional[List[List[str]]] = None,  # [[from, to, desc], ...]
    # Trade study
    criteria: Optional[Dict[str, float]] = None,
    options: Optional[List[Dict[str, Any]]] = None,
    # FMEA
    failure_modes: Optional[List[Dict[str, Any]]] = None,
    # Verification
    requirements: Optional[List[Dict[str, str]]] = None,
    # TPM
    measures: Optional[List[Dict[str, Any]]] = None,
) -> ToolResult:
    """Execute systems engineering analysis."""
    try:
        se = SystemsEngineer()

        if operation == "decompose":
            if not requirement:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output="Need requirement text for decomposition"
                )

            result = se.decompose_requirements(requirement, req_type or "functional")

            output = f"Requirement Decomposition for: \"{requirement}\"\n"
            output += f"Type: {req_type or 'functional'}\n\n"
            output += "Decomposition Templates:\n"
            for i, template in enumerate(result["decomposition_templates"], 1):
                output += f"  {i}. {template}\n"
            output += f"\nSuggested Verification: {result['suggested_verification']}"

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=output,
                metadata=result
            )

        elif operation == "n_squared":
            if not subsystems or not interfaces:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output="Need subsystems and interfaces for N-squared diagram"
                )

            # Convert interfaces to tuples
            interface_tuples = [(i[0], i[1], i[2] if len(i) > 2 else "") for i in interfaces]
            result = se.create_n_squared(subsystems, interface_tuples)

            output = f"N-Squared Interface Analysis\n"
            output += f"Subsystems: {subsystems}\n"
            output += f"Interface count: {result['interface_count']} / {result['max_possible']} possible\n"
            output += f"Complexity ratio: {result['complexity_ratio']}\n"
            output += f"Most coupled: {result['most_coupled']['subsystem']} ({result['most_coupled']['interfaces']} interfaces)\n"
            output += f"\n{result['recommendation']}"

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=output,
                metadata=result
            )

        elif operation == "trade_study":
            if not criteria or not options:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output="Need criteria (with weights) and options for trade study"
                )

            result = se.trade_study(criteria, options)

            output = f"Trade Study Results\n"
            output += "=" * 40 + "\n\n"
            output += f"Winner: {result['winner']} (score: {result['results'][0]['weighted_score']:.2f}/10)\n"
            output += f"Margin over 2nd place: {result['margin']:.2f}\n\n"
            output += "Rankings:\n"
            for i, r in enumerate(result["results"], 1):
                output += f"  {i}. {r['name']}: {r['weighted_score']:.2f}\n"
            output += f"\nSensitivity: {result['sensitivity']}"

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=output,
                metadata=result
            )

        elif operation == "fmea":
            if not failure_modes:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output="Need failure_modes list for FMEA"
                )

            result = se.fmea_analysis(failure_modes)

            output = f"FMEA Analysis\n"
            output += "=" * 40 + "\n\n"
            output += f"Total failure modes: {result['summary']['total_items']}\n"
            output += f"Critical (RPN >= 200): {result['summary']['critical_count']}\n"
            output += f"High (RPN >= 100): {result['summary']['high_count']}\n"
            output += f"Max RPN: {result['summary']['max_rpn']}, Avg: {result['summary']['avg_rpn']}\n\n"
            output += "Top Risks:\n"
            for i, risk in enumerate(result["top_risks"], 1):
                output += f"  {i}. {risk['component']} - {risk['failure_mode']} (RPN: {risk['rpn']}, {risk['criticality']})\n"
            output += f"\n{result['recommendation']}"

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=output,
                metadata=result
            )

        elif operation == "verification":
            if not requirements:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output="Need requirements list for verification matrix"
                )

            result = se.verification_matrix(requirements)

            output = f"Verification Cross-Reference Matrix\n"
            output += "=" * 40 + "\n\n"
            output += f"Total requirements: {result['total_requirements']}\n\n"
            output += "Method Distribution:\n"
            for method, count in result["method_distribution"].items():
                pct = count / result["total_requirements"] * 100 if result["total_requirements"] > 0 else 0
                output += f"  {method.capitalize()}: {count} ({pct:.0f}%)\n"
            output += f"\n{result['analysis']['recommendation']}"

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=output,
                metadata=result
            )

        elif operation == "tpm":
            if not measures:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output="Need measures list for TPM tracking"
                )

            result = se.tpm_tracker(measures)

            output = f"Technical Performance Measures\n"
            output += "=" * 40 + "\n\n"
            output += f"Status: {result['summary']['green']} GREEN, {result['summary']['yellow']} YELLOW, {result['summary']['red']} RED\n\n"
            for m in result["measures"]:
                status_icon = {"GREEN": "[OK]", "YELLOW": "[!]", "RED": "[X]"}[m["status"]]
                output += f"{status_icon} {m['name']}: {m['current']} / {m['target']} {m['unit']} ({m['margin_pct']:+.1f}% margin, {m['trend']})\n"
            output += f"\n{result['recommendation']}"

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=output,
                metadata=result
            )

        else:
            return ToolResult(
                status=ToolStatus.INVALID_INPUT,
                output=f"Unknown operation: {operation}. Valid: decompose, n_squared, trade_study, fmea, verification, tpm"
            )

    except Exception as e:
        return ToolResult(
            status=ToolStatus.ERROR,
            output=None,
            error=str(e)
        )


def create_systems_engineering_tool() -> Tool:
    """Create systems engineering tool."""
    return Tool(
        name="systems_engineering",
        description="Systems engineering analysis: requirements decomposition, N-squared interface diagrams, trade studies, FMEA, verification matrices, TPM tracking",
        parameters=[
            ToolParameter(
                name="operation",
                description="decompose, n_squared, trade_study, fmea, verification, tpm",
                type="string",
                required=True,
                enum=["decompose", "n_squared", "trade_study", "fmea", "verification", "tpm"]
            ),
            ToolParameter(
                name="requirement",
                description="High-level requirement text for decomposition",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="req_type",
                description="Requirement type: functional, performance, interface, constraint",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="subsystems",
                description="List of subsystem names for N-squared",
                type="array",
                required=False,
            ),
            ToolParameter(
                name="interfaces",
                description="List of [from, to, description] for N-squared",
                type="array",
                required=False,
            ),
            ToolParameter(
                name="criteria",
                description="Trade study criteria with weights: {criterion: weight}",
                type="object",
                required=False,
            ),
            ToolParameter(
                name="options",
                description="Trade study options: [{name, scores: {criterion: score}}]",
                type="array",
                required=False,
            ),
            ToolParameter(
                name="failure_modes",
                description="FMEA items: [{component, failure_mode, cause, effect, severity, occurrence, detection}]",
                type="array",
                required=False,
            ),
            ToolParameter(
                name="requirements",
                description="Requirements for verification: [{id, text, type, verification}]",
                type="array",
                required=False,
            ),
            ToolParameter(
                name="measures",
                description="TPM items: [{name, target, current, unit, trend}]",
                type="array",
                required=False,
            ),
        ],
        execute_fn=systems_engineering_tool,
        timeout_ms=10000,
    )


def register_systems_engineering_tools(registry: ToolRegistry) -> None:
    """Register systems engineering tools."""
    registry.register(create_systems_engineering_tool())
