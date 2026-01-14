"""
Project Management Specialist Tool

Epistemic scaffolding for project planning and control.
Encodes how project managers think about schedules, resources, and value.

Operations:
- critical_path: CPM analysis (ES, EF, LS, LF, float)
- earned_value: EVM metrics (PV, EV, AC, SPI, CPI, EAC)
- resource_level: Resource leveling and smoothing
- schedule_risk: Monte Carlo schedule simulation
- pert: PERT estimates (optimistic, most likely, pessimistic)
- dependency_graph: Task dependency visualization
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math
import random
from collections import defaultdict

from .registry import Tool, ToolParameter, ToolResult, ToolStatus, ToolRegistry


class TaskStatus(Enum):
    """Task completion status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"


@dataclass
class Task:
    """Project task with scheduling info."""
    id: str
    name: str
    duration: float
    predecessors: List[str] = field(default_factory=list)
    resources: Dict[str, float] = field(default_factory=dict)  # resource -> units

    # Calculated fields
    es: float = 0.0  # Early Start
    ef: float = 0.0  # Early Finish
    ls: float = 0.0  # Late Start
    lf: float = 0.0  # Late Finish
    total_float: float = 0.0
    free_float: float = 0.0
    is_critical: bool = False

    # PERT fields
    optimistic: Optional[float] = None
    most_likely: Optional[float] = None
    pessimistic: Optional[float] = None


@dataclass
class EarnedValueData:
    """Earned Value Management data point."""
    period: int
    planned_value: float
    earned_value: float
    actual_cost: float


class ProjectAnalyzer:
    """
    Project management calculations.

    Provides analysis for:
    - Critical Path Method (CPM)
    - Earned Value Management (EVM)
    - Resource leveling
    - Schedule risk (Monte Carlo)
    - PERT estimates
    """

    def critical_path(
        self,
        tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate Critical Path Method.

        Args:
            tasks: List of task dicts with:
                - id: Unique identifier
                - name: Task name
                - duration: Task duration
                - predecessors: List of predecessor task IDs

        Returns:
            Critical path, float times, project duration
        """
        # Build task objects
        task_map: Dict[str, Task] = {}
        for t in tasks:
            task_map[t["id"]] = Task(
                id=t["id"],
                name=t.get("name", t["id"]),
                duration=t["duration"],
                predecessors=t.get("predecessors", [])
            )

        # Build successor map
        successors: Dict[str, List[str]] = defaultdict(list)
        for task in task_map.values():
            for pred in task.predecessors:
                successors[pred].append(task.id)

        # Forward pass - calculate ES and EF
        def forward_pass():
            # Topological sort
            in_degree = {t.id: len(t.predecessors) for t in task_map.values()}
            queue = [t.id for t in task_map.values() if in_degree[t.id] == 0]
            order = []

            while queue:
                current = queue.pop(0)
                order.append(current)
                task = task_map[current]

                # ES = max(EF of all predecessors)
                if task.predecessors:
                    task.es = max(task_map[p].ef for p in task.predecessors)
                else:
                    task.es = 0

                task.ef = task.es + task.duration

                # Update successors
                for succ in successors[current]:
                    in_degree[succ] -= 1
                    if in_degree[succ] == 0:
                        queue.append(succ)

            return order

        order = forward_pass()

        # Project duration
        project_duration = max(t.ef for t in task_map.values())

        # Backward pass - calculate LS and LF
        def backward_pass():
            # Process in reverse topological order
            for task_id in reversed(order):
                task = task_map[task_id]

                # LF = min(LS of all successors), or project_duration if no successors
                succ_list = successors[task_id]
                if succ_list:
                    task.lf = min(task_map[s].ls for s in succ_list)
                else:
                    task.lf = project_duration

                task.ls = task.lf - task.duration

                # Calculate float
                task.total_float = task.ls - task.es
                task.is_critical = abs(task.total_float) < 0.001

        backward_pass()

        # Calculate free float
        for task in task_map.values():
            succ_list = successors[task.id]
            if succ_list:
                min_succ_es = min(task_map[s].es for s in succ_list)
                task.free_float = min_succ_es - task.ef
            else:
                task.free_float = task.total_float

        # Find critical path
        critical_tasks = [t for t in task_map.values() if t.is_critical]
        critical_path = self._trace_critical_path(critical_tasks, successors, task_map)

        # Build result
        task_results = []
        for task in task_map.values():
            task_results.append({
                "id": task.id,
                "name": task.name,
                "duration": task.duration,
                "es": task.es,
                "ef": task.ef,
                "ls": task.ls,
                "lf": task.lf,
                "total_float": task.total_float,
                "free_float": task.free_float,
                "is_critical": task.is_critical
            })

        return {
            "project_duration": project_duration,
            "critical_path": critical_path,
            "critical_path_length": len(critical_path),
            "tasks": task_results,
            "float_summary": {
                "zero_float_tasks": len([t for t in task_results if t["total_float"] == 0]),
                "total_tasks": len(task_results)
            }
        }

    def _trace_critical_path(
        self,
        critical_tasks: List[Task],
        successors: Dict[str, List[str]],
        task_map: Dict[str, Task]
    ) -> List[str]:
        """Trace the critical path through the network."""
        # Find starting critical tasks (no critical predecessors)
        critical_ids = {t.id for t in critical_tasks}

        path = []
        # Start from tasks with no predecessors or no critical predecessors
        start_tasks = [t for t in critical_tasks if not any(p in critical_ids for p in t.predecessors)]

        if not start_tasks:
            return [t.id for t in critical_tasks]

        # Trace from first start task
        current = start_tasks[0]
        while current:
            path.append(current.id)
            # Find critical successor
            crit_succs = [s for s in successors[current.id] if s in critical_ids]
            if crit_succs:
                current = task_map[crit_succs[0]]
            else:
                current = None

        return path

    def earned_value(
        self,
        budget_at_completion: float,
        planned_percent: float,
        earned_percent: float,
        actual_cost: float,
        time_elapsed: float,
        total_duration: float
    ) -> Dict[str, Any]:
        """
        Calculate Earned Value metrics.

        Args:
            budget_at_completion: Total project budget (BAC)
            planned_percent: Planned % complete at this point
            earned_percent: Actual % of work completed
            actual_cost: Actual cost spent so far
            time_elapsed: Time elapsed
            total_duration: Total planned duration

        Returns:
            EVM metrics, variances, forecasts
        """
        bac = budget_at_completion
        pv = bac * (planned_percent / 100)
        ev = bac * (earned_percent / 100)
        ac = actual_cost

        # Variances
        sv = ev - pv  # Schedule Variance
        cv = ev - ac  # Cost Variance

        # Performance Indices
        spi = ev / pv if pv > 0 else 0  # Schedule Performance Index
        cpi = ev / ac if ac > 0 else 0  # Cost Performance Index

        # Forecasts
        # EAC - Estimate at Completion
        eac_typical = bac / cpi if cpi > 0 else float('inf')  # Typical variance continues
        eac_atypical = ac + (bac - ev)  # Atypical, rest at budget rate

        # ETC - Estimate to Complete
        etc = eac_typical - ac

        # VAC - Variance at Completion
        vac = bac - eac_typical

        # TCPI - To Complete Performance Index
        tcpi_bac = (bac - ev) / (bac - ac) if (bac - ac) > 0 else float('inf')
        tcpi_eac = (bac - ev) / (eac_typical - ac) if (eac_typical - ac) > 0 else float('inf')

        # Schedule forecasting
        time_percent = time_elapsed / total_duration if total_duration > 0 else 0
        estimated_completion = total_duration / spi if spi > 0 else float('inf')

        # Status interpretation
        status = []
        if sv >= 0:
            status.append("AHEAD of schedule" if sv > 0 else "ON schedule")
        else:
            status.append("BEHIND schedule")

        if cv >= 0:
            status.append("UNDER budget" if cv > 0 else "ON budget")
        else:
            status.append("OVER budget")

        return {
            "inputs": {
                "bac": bac,
                "planned_percent": planned_percent,
                "earned_percent": earned_percent,
                "actual_cost": ac
            },
            "core_metrics": {
                "pv": pv,  # Planned Value (BCWS)
                "ev": ev,  # Earned Value (BCWP)
                "ac": ac   # Actual Cost (ACWP)
            },
            "variances": {
                "sv": sv,
                "sv_percent": (sv / pv * 100) if pv > 0 else 0,
                "cv": cv,
                "cv_percent": (cv / ev * 100) if ev > 0 else 0
            },
            "indices": {
                "spi": spi,
                "cpi": cpi,
                "csi": spi * cpi  # Cost-Schedule Index
            },
            "forecasts": {
                "eac_typical": eac_typical,
                "eac_atypical": eac_atypical,
                "etc": etc,
                "vac": vac,
                "estimated_completion_time": estimated_completion
            },
            "tcpi": {
                "to_bac": tcpi_bac,
                "to_eac": tcpi_eac
            },
            "status": status,
            "interpretation": self._interpret_evm(spi, cpi, sv, cv)
        }

    def _interpret_evm(
        self,
        spi: float,
        cpi: float,
        sv: float,
        cv: float
    ) -> str:
        """Generate interpretation of EVM results."""
        # Quadrant analysis
        if spi >= 1.0 and cpi >= 1.0:
            quadrant = "GREEN - Ahead of schedule and under budget"
            action = "Continue current performance"
        elif spi >= 1.0 and cpi < 1.0:
            quadrant = "YELLOW - Ahead of schedule but over budget"
            action = "Review cost efficiency"
        elif spi < 1.0 and cpi >= 1.0:
            quadrant = "YELLOW - Behind schedule but under budget"
            action = "Consider accelerating with available budget"
        else:
            quadrant = "RED - Behind schedule and over budget"
            action = "Immediate corrective action required"

        return f"{quadrant}. SPI={spi:.2f}, CPI={cpi:.2f}. {action}"

    def resource_leveling(
        self,
        tasks: List[Dict[str, Any]],
        resource_limits: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Perform resource leveling.

        Args:
            tasks: Tasks with resources (from critical_path format + resources)
            resource_limits: Maximum available units per resource

        Returns:
            Leveled schedule, resource utilization
        """
        # First get CPM results
        cpm_result = self.critical_path(tasks)

        # Build resource timeline
        # Simple approach: delay non-critical tasks if resources exceeded
        project_duration = int(cpm_result["project_duration"]) + 1

        # Resource usage per time period
        resource_usage: Dict[str, List[float]] = {
            r: [0.0] * project_duration for r in resource_limits
        }

        # Task scheduling (use ES initially)
        scheduled_tasks = []
        for task in cpm_result["tasks"]:
            original = next(t for t in tasks if t["id"] == task["id"])
            resources = original.get("resources", {})

            scheduled_tasks.append({
                "id": task["id"],
                "name": task["name"],
                "duration": task["duration"],
                "original_es": task["es"],
                "scheduled_start": task["es"],  # May be adjusted
                "resources": resources,
                "total_float": task["total_float"],
                "is_critical": task["is_critical"]
            })

        # Sort by float (critical tasks first, then by float)
        scheduled_tasks.sort(key=lambda t: (not t["is_critical"], t["total_float"]))

        # Level resources
        conflicts = []
        for task in scheduled_tasks:
            start = int(task["scheduled_start"])
            duration = int(task["duration"])

            # Check resource availability
            while True:
                conflict = False
                for r, amount in task["resources"].items():
                    if r not in resource_limits:
                        continue
                    limit = resource_limits[r]

                    for t in range(start, start + duration):
                        if t < project_duration and resource_usage[r][t] + amount > limit:
                            conflict = True
                            conflicts.append({
                                "task": task["id"],
                                "resource": r,
                                "time": t,
                                "needed": resource_usage[r][t] + amount,
                                "limit": limit
                            })
                            break
                    if conflict:
                        break

                if conflict and task["total_float"] > 0:
                    # Delay task if it has float
                    start += 1
                    task["scheduled_start"] = start
                    task["total_float"] -= 1
                else:
                    break

            # Allocate resources
            for t in range(start, min(start + duration, project_duration)):
                for r, amount in task["resources"].items():
                    if r in resource_usage:
                        resource_usage[r][t] += amount

        # Calculate utilization
        utilization = {}
        for r, usage in resource_usage.items():
            limit = resource_limits[r]
            non_zero = [u for u in usage if u > 0]
            if non_zero:
                avg = sum(non_zero) / len(non_zero)
                peak = max(usage)
                utilization[r] = {
                    "average": avg / limit * 100,
                    "peak": peak / limit * 100,
                    "peak_value": peak,
                    "limit": limit
                }

        # New project duration
        new_duration = max(
            t["scheduled_start"] + t["duration"]
            for t in scheduled_tasks
        )

        return {
            "original_duration": cpm_result["project_duration"],
            "leveled_duration": new_duration,
            "extension": new_duration - cpm_result["project_duration"],
            "scheduled_tasks": scheduled_tasks,
            "resource_utilization": utilization,
            "conflicts_resolved": len(set(c["task"] for c in conflicts)),
            "interpretation": f"Resource leveling extended project by {new_duration - cpm_result['project_duration']:.1f} time units"
        }

    def schedule_risk(
        self,
        tasks: List[Dict[str, Any]],
        iterations: int = 1000,
        confidence_levels: List[float] = [0.5, 0.8, 0.9, 0.95]
    ) -> Dict[str, Any]:
        """
        Monte Carlo schedule risk analysis.

        Args:
            tasks: Tasks with PERT estimates (optimistic, most_likely, pessimistic)
            iterations: Number of simulation runs
            confidence_levels: Percentiles to calculate

        Returns:
            Duration distribution, confidence levels, risk metrics
        """
        durations = []

        for _ in range(iterations):
            # Sample durations from PERT distribution
            sampled_tasks = []
            for t in tasks:
                if all(k in t for k in ["optimistic", "most_likely", "pessimistic"]):
                    o, m, p = t["optimistic"], t["most_likely"], t["pessimistic"]
                    # PERT beta distribution approximation
                    mean = (o + 4*m + p) / 6
                    std = (p - o) / 6
                    # Sample from triangular as approximation
                    sampled = random.triangular(o, p, m)
                else:
                    sampled = t["duration"]

                sampled_tasks.append({
                    "id": t["id"],
                    "name": t.get("name", t["id"]),
                    "duration": sampled,
                    "predecessors": t.get("predecessors", [])
                })

            # Run CPM on sampled tasks
            result = self.critical_path(sampled_tasks)
            durations.append(result["project_duration"])

        # Statistics
        durations.sort()
        mean_duration = sum(durations) / len(durations)
        std_duration = math.sqrt(sum((d - mean_duration)**2 for d in durations) / len(durations))

        # Percentiles
        percentiles = {}
        for level in confidence_levels:
            idx = int(level * len(durations))
            percentiles[f"P{int(level*100)}"] = durations[min(idx, len(durations)-1)]

        # Deterministic comparison
        deterministic = self.critical_path(tasks)["project_duration"]

        # Probability of meeting deterministic deadline
        prob_on_time = sum(1 for d in durations if d <= deterministic) / len(durations)

        return {
            "iterations": iterations,
            "deterministic_duration": deterministic,
            "statistics": {
                "mean": mean_duration,
                "std_dev": std_duration,
                "min": min(durations),
                "max": max(durations),
                "range": max(durations) - min(durations)
            },
            "percentiles": percentiles,
            "probability_on_time": prob_on_time,
            "contingency_recommendation": {
                "P80_contingency": percentiles.get("P80", mean_duration) - deterministic,
                "P90_contingency": percentiles.get("P90", mean_duration) - deterministic
            },
            "interpretation": f"P80 duration: {percentiles.get('P80', 0):.1f}. {prob_on_time*100:.1f}% chance of meeting deterministic estimate."
        }

    def pert_estimate(
        self,
        optimistic: float,
        most_likely: float,
        pessimistic: float
    ) -> Dict[str, Any]:
        """
        Calculate PERT estimate.

        Args:
            optimistic: Best-case duration (a)
            most_likely: Most likely duration (m)
            pessimistic: Worst-case duration (b)

        Returns:
            Expected value, standard deviation, confidence intervals
        """
        o, m, p = optimistic, most_likely, pessimistic

        # PERT formulas
        te = (o + 4*m + p) / 6  # Expected time
        sigma = (p - o) / 6     # Standard deviation

        # Variance
        variance = sigma ** 2

        # Confidence intervals (assuming normal distribution)
        intervals = {
            "68%": (te - sigma, te + sigma),
            "95%": (te - 2*sigma, te + 2*sigma),
            "99.7%": (te - 3*sigma, te + 3*sigma)
        }

        # Skewness check
        if m < (o + p) / 2:
            skew = "LEFT (optimistic bias)"
        elif m > (o + p) / 2:
            skew = "RIGHT (pessimistic bias)"
        else:
            skew = "SYMMETRIC"

        return {
            "inputs": {
                "optimistic": o,
                "most_likely": m,
                "pessimistic": p
            },
            "expected_duration": te,
            "standard_deviation": sigma,
            "variance": variance,
            "confidence_intervals": intervals,
            "range": p - o,
            "skewness": skew,
            "interpretation": f"Expected: {te:.2f} ± {sigma:.2f}. 95% likely between {te-2*sigma:.2f} and {te+2*sigma:.2f}"
        }

    def dependency_analysis(
        self,
        tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze task dependencies.

        Args:
            tasks: List of tasks with predecessors

        Returns:
            Dependency metrics, chains, potential bottlenecks
        """
        # Build dependency graph
        task_map = {t["id"]: t for t in tasks}
        successors: Dict[str, List[str]] = defaultdict(list)
        predecessors: Dict[str, List[str]] = defaultdict(list)

        for t in tasks:
            for pred in t.get("predecessors", []):
                successors[pred].append(t["id"])
                predecessors[t["id"]].append(pred)

        # Calculate metrics
        metrics = []
        for t in tasks:
            t_id = t["id"]
            in_degree = len(predecessors[t_id])
            out_degree = len(successors[t_id])

            metrics.append({
                "id": t_id,
                "name": t.get("name", t_id),
                "predecessors": in_degree,
                "successors": out_degree,
                "bottleneck_score": out_degree  # More successors = more impact if delayed
            })

        # Find longest dependency chain
        def find_longest_chain(task_id: str, visited: set) -> List[str]:
            if task_id in visited:
                return []
            visited.add(task_id)

            succ_chains = []
            for succ in successors[task_id]:
                chain = find_longest_chain(succ, visited.copy())
                succ_chains.append(chain)

            best_chain = max(succ_chains, key=len) if succ_chains else []
            return [task_id] + best_chain

        # Find chains from each starting task
        start_tasks = [t["id"] for t in tasks if not predecessors[t["id"]]]
        chains = []
        for start in start_tasks:
            chain = find_longest_chain(start, set())
            chains.append(chain)

        longest_chain = max(chains, key=len) if chains else []

        # Identify bottlenecks (high out-degree)
        metrics.sort(key=lambda x: x["bottleneck_score"], reverse=True)
        bottlenecks = [m for m in metrics if m["bottleneck_score"] >= 2][:5]

        # Parallelism potential
        max_parallel = max(len([t for t in tasks if t.get("predecessors", []) == preds])
                          for preds in [t.get("predecessors", []) for t in tasks]) if tasks else 1

        return {
            "task_count": len(tasks),
            "dependency_edges": sum(len(t.get("predecessors", [])) for t in tasks),
            "start_tasks": start_tasks,
            "end_tasks": [t["id"] for t in tasks if not successors[t["id"]]],
            "longest_chain": longest_chain,
            "chain_length": len(longest_chain),
            "bottleneck_tasks": bottlenecks,
            "parallelism_potential": max_parallel,
            "task_metrics": sorted(metrics, key=lambda x: x["id"]),
            "interpretation": f"Longest chain: {len(longest_chain)} tasks. {len(bottlenecks)} potential bottlenecks."
        }


def create_project_management_tool() -> Tool:
    """Create the project management tool."""
    analyzer = ProjectAnalyzer()

    def execute(operation: str, **kwargs) -> ToolResult:
        """Execute a project management operation."""
        operations = {
            "critical_path": analyzer.critical_path,
            "earned_value": analyzer.earned_value,
            "resource_level": analyzer.resource_leveling,
            "schedule_risk": analyzer.schedule_risk,
            "pert": analyzer.pert_estimate,
            "dependency_analysis": analyzer.dependency_analysis
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
        name="project_management",
        description="""Project management analysis tool.

Operations:
- critical_path: CPM analysis
  Args: tasks (list of {id, name, duration, predecessors})
  Returns: Critical path, ES/EF/LS/LF/float for each task

- earned_value: EVM metrics
  Args: budget_at_completion, planned_percent, earned_percent, actual_cost, time_elapsed, total_duration
  Returns: PV, EV, AC, SV, CV, SPI, CPI, EAC, ETC

- resource_level: Resource leveling
  Args: tasks (with resources), resource_limits (dict)
  Returns: Leveled schedule, utilization

- schedule_risk: Monte Carlo simulation
  Args: tasks (with optimistic, most_likely, pessimistic), iterations, confidence_levels
  Returns: Duration distribution, percentiles

- pert: PERT three-point estimate
  Args: optimistic, most_likely, pessimistic
  Returns: Expected value, standard deviation, confidence intervals

- dependency_analysis: Task dependency analysis
  Args: tasks (list of {id, name, predecessors})
  Returns: Chains, bottlenecks, parallelism potential""",
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


def register_project_management_tools(registry: ToolRegistry) -> None:
    """Register project management tools."""
    registry.register(create_project_management_tool())
