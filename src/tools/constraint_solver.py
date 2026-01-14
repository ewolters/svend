"""
Constraint Solver - SAT/UNSAT with Explanations

Kills hallucinations where model says "here's a plan" when the plan is IMPOSSIBLE.

Returns:
- SATISFIABLE + solution
- UNSATISFIABLE + minimal conflicting constraints
- UNKNOWN (timeout/complexity)

This forces intellectual honesty about feasibility.
"""

from typing import Optional, Dict, Any, List, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from itertools import combinations, product
import re

from .registry import Tool, ToolParameter, ToolResult, ToolStatus, ToolRegistry


class SatisfiabilityStatus(Enum):
    """Result of constraint checking."""
    SAT = "satisfiable"
    UNSAT = "unsatisfiable"
    UNKNOWN = "unknown"  # Timeout or too complex


@dataclass
class ConstraintResult:
    """Result of constraint solving."""
    status: SatisfiabilityStatus
    solution: Optional[Dict[str, Any]] = None  # Variable assignments if SAT
    conflicts: Optional[List[str]] = None  # Conflicting constraints if UNSAT
    explanation: Optional[str] = None


class ConstraintSolver:
    """
    Lightweight constraint solver for common constraint types.

    For heavy-duty SAT/SMT, we have Z3 in math_engine.py.
    This is for quick feasibility checks that don't need full SMT.
    """

    def __init__(self):
        self.variables: Dict[str, Any] = {}
        self.domains: Dict[str, Set] = {}
        self.constraints: List[Tuple[str, callable, str]] = []  # (name, check_fn, description)

    def add_variable(self, name: str, domain: Set[Any]):
        """Add a variable with its domain."""
        self.variables[name] = None
        self.domains[name] = set(domain)

    def add_int_variable(self, name: str, low: int, high: int):
        """Add an integer variable with range."""
        self.add_variable(name, set(range(low, high + 1)))

    def add_bool_variable(self, name: str):
        """Add a boolean variable."""
        self.add_variable(name, {True, False})

    def add_constraint(self, name: str, check_fn: callable, description: str = ""):
        """Add a constraint (function that takes variable dict, returns bool)."""
        self.constraints.append((name, check_fn, description or name))

    def add_equality(self, var1: str, var2: str):
        """Add constraint: var1 == var2."""
        self.add_constraint(
            f"{var1}=={var2}",
            lambda v, v1=var1, v2=var2: v.get(v1) == v.get(v2),
            f"{var1} must equal {var2}"
        )

    def add_inequality(self, var1: str, var2: str):
        """Add constraint: var1 != var2."""
        self.add_constraint(
            f"{var1}!={var2}",
            lambda v, v1=var1, v2=var2: v.get(v1) != v.get(v2),
            f"{var1} must not equal {var2}"
        )

    def add_less_than(self, var1: str, var2: str):
        """Add constraint: var1 < var2."""
        self.add_constraint(
            f"{var1}<{var2}",
            lambda v, v1=var1, v2=var2: v.get(v1) is not None and v.get(v2) is not None and v.get(v1) < v.get(v2),
            f"{var1} must be less than {var2}"
        )

    def add_all_different(self, variables: List[str]):
        """Add constraint: all variables must have different values."""
        name = f"alldiff({','.join(variables)})"
        self.add_constraint(
            name,
            lambda v, vs=variables: len(set(v.get(var) for var in vs)) == len(vs),
            f"All of {variables} must be different"
        )

    def add_sum_constraint(self, variables: List[str], target: int, op: str = "=="):
        """Add constraint on sum of variables."""
        name = f"sum({','.join(variables)}){op}{target}"

        if op == "==":
            check = lambda v, vs=variables, t=target: sum(v.get(var, 0) for var in vs) == t
        elif op == "<=":
            check = lambda v, vs=variables, t=target: sum(v.get(var, 0) for var in vs) <= t
        elif op == ">=":
            check = lambda v, vs=variables, t=target: sum(v.get(var, 0) for var in vs) >= t
        elif op == "<":
            check = lambda v, vs=variables, t=target: sum(v.get(var, 0) for var in vs) < t
        elif op == ">":
            check = lambda v, vs=variables, t=target: sum(v.get(var, 0) for var in vs) > t
        else:
            raise ValueError(f"Unknown operator: {op}")

        self.add_constraint(name, check, f"Sum of {variables} {op} {target}")

    def check_assignment(self, assignment: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check if an assignment satisfies all constraints."""
        violations = []
        for name, check_fn, desc in self.constraints:
            try:
                if not check_fn(assignment):
                    violations.append(desc)
            except:
                violations.append(f"{desc} (error during check)")
        return len(violations) == 0, violations

    def solve_brute_force(self, max_iterations: int = 100000) -> ConstraintResult:
        """
        Solve by enumerating all possible assignments.

        Only practical for small domains.
        """
        if not self.variables:
            return ConstraintResult(SatisfiabilityStatus.SAT, {}, None, "No variables")

        var_names = list(self.variables.keys())
        domains = [list(self.domains[v]) for v in var_names]

        # Check total search space
        total = 1
        for d in domains:
            total *= len(d)
            if total > max_iterations:
                return ConstraintResult(
                    SatisfiabilityStatus.UNKNOWN,
                    None,
                    None,
                    f"Search space too large ({total} > {max_iterations})"
                )

        # Enumerate
        iterations = 0
        for values in product(*domains):
            iterations += 1
            if iterations > max_iterations:
                return ConstraintResult(
                    SatisfiabilityStatus.UNKNOWN,
                    None,
                    None,
                    f"Exceeded max iterations ({max_iterations})"
                )

            assignment = dict(zip(var_names, values))
            satisfied, violations = self.check_assignment(assignment)

            if satisfied:
                return ConstraintResult(
                    SatisfiabilityStatus.SAT,
                    assignment,
                    None,
                    f"Found solution in {iterations} iterations"
                )

        # No solution found - find minimal conflict set
        conflicts = self._find_minimal_conflicts()

        return ConstraintResult(
            SatisfiabilityStatus.UNSAT,
            None,
            conflicts,
            f"No solution exists. Checked {iterations} assignments."
        )

    def solve_backtracking(self, max_iterations: int = 100000) -> ConstraintResult:
        """Solve using backtracking with constraint propagation."""
        if not self.variables:
            return ConstraintResult(SatisfiabilityStatus.SAT, {}, None, "No variables")

        var_names = list(self.variables.keys())
        domains = {v: set(self.domains[v]) for v in var_names}

        iterations = [0]  # Use list for closure

        def backtrack(assignment: Dict[str, Any], remaining: List[str]) -> Optional[Dict[str, Any]]:
            iterations[0] += 1
            if iterations[0] > max_iterations:
                return None

            if not remaining:
                # All variables assigned - do final check
                satisfied, _ = self.check_assignment(assignment)
                return assignment if satisfied else None

            var = remaining[0]
            rest = remaining[1:]

            for value in domains[var]:
                assignment[var] = value

                # For partial assignments, skip constraint check
                # (constraints may reference unassigned variables)
                if rest:
                    # Not all assigned yet - just recurse
                    result = backtrack(assignment.copy(), rest)
                    if result is not None:
                        return result
                else:
                    # All assigned - check constraints
                    satisfied, _ = self.check_assignment(assignment)
                    if satisfied:
                        return assignment.copy()

            del assignment[var]
            return None

        result = backtrack({}, var_names)

        if result is not None:
            return ConstraintResult(
                SatisfiabilityStatus.SAT,
                result,
                None,
                f"Found solution in {iterations[0]} iterations"
            )

        if iterations[0] >= max_iterations:
            return ConstraintResult(
                SatisfiabilityStatus.UNKNOWN,
                None,
                None,
                f"Exceeded max iterations ({max_iterations})"
            )

        conflicts = self._find_minimal_conflicts()
        return ConstraintResult(
            SatisfiabilityStatus.UNSAT,
            None,
            conflicts,
            f"No solution exists. Searched {iterations[0]} nodes."
        )

    def _find_minimal_conflicts(self) -> List[str]:
        """Find a minimal set of conflicting constraints."""
        # Try removing constraints one at a time to find which are essential
        if len(self.constraints) <= 3:
            return [desc for _, _, desc in self.constraints]

        # Check pairs of constraints
        conflicts = []
        for i, (name1, _, desc1) in enumerate(self.constraints):
            for j, (name2, _, desc2) in enumerate(self.constraints):
                if i < j:
                    # Check if these two alone conflict
                    temp_solver = ConstraintSolver()
                    for v, d in self.domains.items():
                        temp_solver.add_variable(v, d)
                    temp_solver.constraints = [self.constraints[i], self.constraints[j]]

                    result = temp_solver.solve_brute_force(max_iterations=1000)
                    if result.status == SatisfiabilityStatus.UNSAT:
                        conflicts.append(f"{desc1} AND {desc2}")

        if conflicts:
            return conflicts[:5]  # Return first few conflicts

        return [desc for _, _, desc in self.constraints]


class SchedulingChecker:
    """
    Specialized constraint checker for scheduling problems.

    Checks:
    - Resource conflicts (two tasks same resource same time)
    - Precedence (task A before task B)
    - Duration feasibility
    - Deadline violations
    """

    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.resources: Dict[str, int] = {}  # resource -> capacity
        self.precedences: List[Tuple[str, str]] = []  # (before, after)

    def add_task(
        self,
        name: str,
        duration: int,
        resource: Optional[str] = None,
        earliest_start: int = 0,
        deadline: Optional[int] = None
    ):
        """Add a task."""
        self.tasks[name] = {
            "duration": duration,
            "resource": resource,
            "earliest_start": earliest_start,
            "deadline": deadline,
            "start": None  # To be assigned
        }

    def add_resource(self, name: str, capacity: int = 1):
        """Add a resource with capacity."""
        self.resources[name] = capacity

    def add_precedence(self, before: str, after: str):
        """Task 'before' must complete before 'after' starts."""
        self.precedences.append((before, after))

    def check_schedule(self, schedule: Dict[str, int]) -> Tuple[bool, List[str]]:
        """
        Check if a schedule (task -> start_time) is feasible.

        Returns (feasible, violations).
        """
        violations = []

        for task_name, start_time in schedule.items():
            if task_name not in self.tasks:
                violations.append(f"Unknown task: {task_name}")
                continue

            task = self.tasks[task_name]
            end_time = start_time + task["duration"]

            # Check earliest start
            if start_time < task["earliest_start"]:
                violations.append(f"{task_name} starts at {start_time} but earliest is {task['earliest_start']}")

            # Check deadline
            if task["deadline"] is not None and end_time > task["deadline"]:
                violations.append(f"{task_name} ends at {end_time} but deadline is {task['deadline']}")

        # Check precedences
        for before, after in self.precedences:
            if before in schedule and after in schedule:
                before_end = schedule[before] + self.tasks[before]["duration"]
                after_start = schedule[after]
                if before_end > after_start:
                    violations.append(f"{before} (ends {before_end}) must complete before {after} (starts {after_start})")

        # Check resource conflicts
        for resource, capacity in self.resources.items():
            # Find tasks using this resource
            resource_tasks = [
                (name, schedule.get(name), self.tasks[name]["duration"])
                for name, task in self.tasks.items()
                if task["resource"] == resource and name in schedule
            ]

            # Check each time point
            if resource_tasks:
                max_time = max(start + dur for _, start, dur in resource_tasks if start is not None)
                for t in range(max_time + 1):
                    active = sum(
                        1 for name, start, dur in resource_tasks
                        if start is not None and start <= t < start + dur
                    )
                    if active > capacity:
                        violations.append(f"Resource {resource} overloaded at time {t} ({active} > {capacity})")
                        break  # One violation per resource is enough

        return len(violations) == 0, violations

    def find_conflicts(self) -> List[str]:
        """Find inherent conflicts in the problem definition."""
        conflicts = []

        # Check if any task's deadline is before it could possibly finish
        for name, task in self.tasks.items():
            if task["deadline"] is not None:
                min_finish = task["earliest_start"] + task["duration"]
                if min_finish > task["deadline"]:
                    conflicts.append(
                        f"{name}: cannot finish by deadline {task['deadline']} "
                        f"(earliest finish is {min_finish})"
                    )

        # Check precedence cycles
        # Build graph and check for cycles
        from collections import defaultdict, deque

        graph = defaultdict(list)
        in_degree = defaultdict(int)
        all_tasks = set(self.tasks.keys())

        for before, after in self.precedences:
            graph[before].append(after)
            in_degree[after] += 1
            all_tasks.add(before)
            all_tasks.add(after)

        # Kahn's algorithm for cycle detection
        queue = deque([t for t in all_tasks if in_degree[t] == 0])
        visited = 0

        while queue:
            node = queue.popleft()
            visited += 1
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited < len(all_tasks):
            conflicts.append("Precedence constraints contain a cycle - impossible to satisfy")

        return conflicts


# Tool implementation

def constraint_tool(
    operation: str,
    variables: Optional[Dict[str, Any]] = None,
    constraints: Optional[List[Dict[str, Any]]] = None,
    assignment: Optional[Dict[str, Any]] = None,
    # For scheduling
    tasks: Optional[List[Dict[str, Any]]] = None,
    resources: Optional[Dict[str, int]] = None,
    precedences: Optional[List[List[str]]] = None,
    schedule: Optional[Dict[str, int]] = None,
    max_iterations: int = 10000,
) -> ToolResult:
    """Execute constraint checking."""
    try:
        if operation == "check_feasibility":
            # Check if constraints are satisfiable
            if not variables or not constraints:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output=None,
                    error="Need variables and constraints"
                )

            solver = ConstraintSolver()

            # Add variables
            for var_name, domain_spec in variables.items():
                if isinstance(domain_spec, list):
                    solver.add_variable(var_name, set(domain_spec))
                elif isinstance(domain_spec, dict):
                    if "range" in domain_spec:
                        low, high = domain_spec["range"]
                        solver.add_int_variable(var_name, low, high)
                    elif "values" in domain_spec:
                        solver.add_variable(var_name, set(domain_spec["values"]))
                elif domain_spec == "bool":
                    solver.add_bool_variable(var_name)

            # Add constraints
            for c in constraints:
                c_type = c.get("type")

                if c_type == "equal":
                    solver.add_equality(c["var1"], c["var2"])
                elif c_type == "not_equal":
                    solver.add_inequality(c["var1"], c["var2"])
                elif c_type == "less_than":
                    solver.add_less_than(c["var1"], c["var2"])
                elif c_type == "all_different":
                    solver.add_all_different(c["variables"])
                elif c_type == "sum":
                    solver.add_sum_constraint(
                        c["variables"],
                        c["target"],
                        c.get("op", "==")
                    )
                elif c_type == "custom":
                    # Custom constraint with expression
                    expr = c["expression"]
                    desc = c.get("description", expr)
                    # Simple expression evaluation (safe subset)
                    solver.add_constraint(
                        expr,
                        lambda v, e=expr: eval(e, {"__builtins__": {}}, v),
                        desc
                    )

            # Solve
            result = solver.solve_backtracking(max_iterations)

            if result.status == SatisfiabilityStatus.SAT:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=f"SATISFIABLE\nSolution: {result.solution}\n{result.explanation}",
                    metadata={
                        "satisfiable": True,
                        "solution": result.solution,
                        "explanation": result.explanation
                    }
                )
            elif result.status == SatisfiabilityStatus.UNSAT:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=f"UNSATISFIABLE\nConflicts: {result.conflicts}\n{result.explanation}",
                    metadata={
                        "satisfiable": False,
                        "conflicts": result.conflicts,
                        "explanation": result.explanation
                    }
                )
            else:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=f"UNKNOWN (complexity exceeded)\n{result.explanation}",
                    metadata={
                        "satisfiable": None,
                        "explanation": result.explanation
                    }
                )

        elif operation == "check_assignment":
            # Check if a specific assignment satisfies constraints
            if not variables or not constraints or not assignment:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output=None,
                    error="Need variables, constraints, and assignment"
                )

            solver = ConstraintSolver()

            for var_name, domain_spec in variables.items():
                if isinstance(domain_spec, list):
                    solver.add_variable(var_name, set(domain_spec))
                elif isinstance(domain_spec, dict):
                    if "range" in domain_spec:
                        low, high = domain_spec["range"]
                        solver.add_int_variable(var_name, low, high)
                    elif "values" in domain_spec:
                        solver.add_variable(var_name, set(domain_spec["values"]))
                elif domain_spec == "bool":
                    solver.add_bool_variable(var_name)

            for c in constraints:
                c_type = c.get("type")
                if c_type == "equal":
                    solver.add_equality(c["var1"], c["var2"])
                elif c_type == "not_equal":
                    solver.add_inequality(c["var1"], c["var2"])
                elif c_type == "less_than":
                    solver.add_less_than(c["var1"], c["var2"])
                elif c_type == "all_different":
                    solver.add_all_different(c["variables"])
                elif c_type == "sum":
                    solver.add_sum_constraint(c["variables"], c["target"], c.get("op", "=="))

            satisfied, violations = solver.check_assignment(assignment)

            if satisfied:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=f"Assignment SATISFIES all constraints",
                    metadata={"valid": True, "violations": []}
                )
            else:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=f"Assignment VIOLATES constraints:\n" + "\n".join(f"- {v}" for v in violations),
                    metadata={"valid": False, "violations": violations}
                )

        elif operation == "check_schedule":
            # Check schedule feasibility
            if not tasks:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output=None,
                    error="Need tasks"
                )

            checker = SchedulingChecker()

            for task in tasks:
                checker.add_task(
                    task["name"],
                    task["duration"],
                    task.get("resource"),
                    task.get("earliest_start", 0),
                    task.get("deadline")
                )

            for res, cap in (resources or {}).items():
                checker.add_resource(res, cap)

            for prec in (precedences or []):
                checker.add_precedence(prec[0], prec[1])

            # First check for inherent conflicts
            inherent_conflicts = checker.find_conflicts()
            if inherent_conflicts:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=f"IMPOSSIBLE - Inherent conflicts:\n" + "\n".join(f"- {c}" for c in inherent_conflicts),
                    metadata={
                        "feasible": False,
                        "inherent_conflicts": inherent_conflicts
                    }
                )

            # If schedule provided, check it
            if schedule:
                feasible, violations = checker.check_schedule(schedule)
                if feasible:
                    return ToolResult(
                        status=ToolStatus.SUCCESS,
                        output="Schedule is FEASIBLE",
                        metadata={"feasible": True, "violations": []}
                    )
                else:
                    return ToolResult(
                        status=ToolStatus.SUCCESS,
                        output=f"Schedule is INFEASIBLE:\n" + "\n".join(f"- {v}" for v in violations),
                        metadata={"feasible": False, "violations": violations}
                    )

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output="Problem definition appears consistent (no inherent conflicts detected)",
                metadata={"feasible": None, "note": "No schedule provided to check"}
            )

        else:
            return ToolResult(
                status=ToolStatus.INVALID_INPUT,
                output=None,
                error=f"Unknown operation: {operation}. Valid: check_feasibility, check_assignment, check_schedule"
            )

    except Exception as e:
        return ToolResult(status=ToolStatus.ERROR, output=None, error=str(e))


def create_constraint_tool() -> Tool:
    """Create constraint solver tool."""
    return Tool(
        name="constraint",
        description="Check constraint satisfaction. Returns SATISFIABLE (with solution), UNSATISFIABLE (with conflicts), or UNKNOWN. Use to verify if a plan/assignment is POSSIBLE before proposing it.",
        parameters=[
            ToolParameter(
                name="operation",
                description="check_feasibility (solve), check_assignment (verify), check_schedule (scheduling)",
                type="string",
                required=True,
                enum=["check_feasibility", "check_assignment", "check_schedule"]
            ),
            ToolParameter(
                name="variables",
                description="Variable definitions: {name: [domain] or {range: [low, high]} or 'bool'}",
                type="object",
                required=False,
            ),
            ToolParameter(
                name="constraints",
                description="List of constraints: [{type: equal/not_equal/less_than/all_different/sum, ...}]",
                type="array",
                required=False,
            ),
            ToolParameter(
                name="assignment",
                description="Variable assignment to check: {var: value, ...}",
                type="object",
                required=False,
            ),
            ToolParameter(
                name="tasks",
                description="For scheduling: [{name, duration, resource?, earliest_start?, deadline?}]",
                type="array",
                required=False,
            ),
            ToolParameter(
                name="resources",
                description="For scheduling: {resource_name: capacity}",
                type="object",
                required=False,
            ),
            ToolParameter(
                name="precedences",
                description="For scheduling: [[before, after], ...]",
                type="array",
                required=False,
            ),
            ToolParameter(
                name="schedule",
                description="Schedule to check: {task_name: start_time, ...}",
                type="object",
                required=False,
            ),
            ToolParameter(
                name="max_iterations",
                description="Max search iterations (default: 10000)",
                type="number",
                required=False,
            ),
        ],
        execute_fn=constraint_tool,
        timeout_ms=30000,
    )


def register_constraint_tools(registry: ToolRegistry) -> None:
    """Register constraint solver tools."""
    registry.register(create_constraint_tool())
