"""
Logic puzzles tool for Svend reasoning system.

Handles classic logic puzzle types:
- Knights and Knaves (truth-tellers and liars)
- Zebra/Einstein puzzles (constraint grids)
- Syllogisms
- Propositional logic
- River crossing puzzles
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum
from itertools import permutations, product
import re


class PersonType(Enum):
    KNIGHT = "knight"  # Always tells truth
    KNAVE = "knave"    # Always lies


@dataclass
class LogicResult:
    """Result of logic puzzle solution."""
    solution: Any
    is_unique: bool
    num_solutions: int
    steps: List[str]
    valid: bool = True


class LogicPuzzleEngine:
    """
    Logic puzzle solver engine.

    Provides systematic approaches to classic logic puzzles.
    """

    def solve_knights_knaves(
        self,
        people: List[str],
        statements: List[Dict[str, Any]]
    ) -> LogicResult:
        """
        Solve a Knights and Knaves puzzle.

        Knights always tell the truth, Knaves always lie.

        Args:
            people: List of person names
            statements: List of statements, each dict with:
                - speaker: who made the statement
                - claim: what they claimed
                - about: who the claim is about (optional, defaults to self)
                - claim_type: "is_knight", "is_knave", "same_type", "different_type"

        Returns:
            LogicResult with solution assignments
        """
        steps = []
        n = len(people)

        steps.append(f"People: {', '.join(people)}")
        steps.append(f"Statements to analyze: {len(statements)}")
        steps.append("")

        # Try all possible assignments
        valid_solutions = []

        for assignment in product([PersonType.KNIGHT, PersonType.KNAVE], repeat=n):
            person_types = dict(zip(people, assignment))

            steps.append(f"Testing: {', '.join(f'{p}={t.value}' for p, t in person_types.items())}")

            all_consistent = True

            for stmt in statements:
                speaker = stmt["speaker"]
                claim_type = stmt["claim_type"]
                about = stmt.get("about", speaker)

                # Evaluate if the claim is true
                if claim_type == "is_knight":
                    claim_true = person_types[about] == PersonType.KNIGHT
                elif claim_type == "is_knave":
                    claim_true = person_types[about] == PersonType.KNAVE
                elif claim_type == "same_type":
                    claim_true = person_types[speaker] == person_types[about]
                elif claim_type == "different_type":
                    claim_true = person_types[speaker] != person_types[about]
                elif claim_type == "at_least_one_knave":
                    targets = stmt.get("targets", people)
                    claim_true = any(person_types[p] == PersonType.KNAVE for p in targets)
                elif claim_type == "all_knights":
                    targets = stmt.get("targets", people)
                    claim_true = all(person_types[p] == PersonType.KNIGHT for p in targets)
                elif claim_type == "exactly_n_knights":
                    targets = stmt.get("targets", people)
                    n_claimed = stmt.get("n", 1)
                    actual = sum(1 for p in targets if person_types[p] == PersonType.KNIGHT)
                    claim_true = actual == n_claimed
                else:
                    raise ValueError(f"Unknown claim type: {claim_type}")

                # Check consistency: Knight tells truth, Knave lies
                speaker_is_knight = person_types[speaker] == PersonType.KNIGHT

                if speaker_is_knight and not claim_true:
                    steps.append(f"  ✗ {speaker} (knight) made false claim")
                    all_consistent = False
                    break
                elif not speaker_is_knight and claim_true:
                    steps.append(f"  ✗ {speaker} (knave) made true claim")
                    all_consistent = False
                    break

            if all_consistent:
                steps.append(f"  ✓ Consistent!")
                valid_solutions.append(person_types)
            steps.append("")

        # Summarize
        if len(valid_solutions) == 0:
            steps.append("No valid solution exists - puzzle is contradictory")
            return LogicResult(
                solution=None,
                is_unique=False,
                num_solutions=0,
                steps=steps,
                valid=False
            )
        elif len(valid_solutions) == 1:
            sol = valid_solutions[0]
            steps.append(f"Unique solution found:")
            for p, t in sol.items():
                steps.append(f"  {p} is a {t.value}")
            return LogicResult(
                solution={p: t.value for p, t in sol.items()},
                is_unique=True,
                num_solutions=1,
                steps=steps
            )
        else:
            steps.append(f"Multiple solutions ({len(valid_solutions)}):")
            for i, sol in enumerate(valid_solutions):
                steps.append(f"  Solution {i+1}: {', '.join(f'{p}={t.value}' for p, t in sol.items())}")
            return LogicResult(
                solution=[{p: t.value for p, t in sol.items()} for sol in valid_solutions],
                is_unique=False,
                num_solutions=len(valid_solutions),
                steps=steps
            )

    def solve_zebra_puzzle(
        self,
        categories: Dict[str, List[str]],
        constraints: List[Dict[str, Any]]
    ) -> LogicResult:
        """
        Solve a Zebra/Einstein-style logic grid puzzle.

        Args:
            categories: Dict mapping category names to list of options
                e.g., {"nationality": ["Brit", "Swede"], "pet": ["dog", "cat"]}
            constraints: List of constraint dicts with:
                - type: "same", "different", "adjacent", "left_of", "immediate_left"
                - items: list of (category, value) pairs that must satisfy constraint

        Returns:
            LogicResult with solution grid
        """
        steps = []

        # Get the primary category (usually position or first listed)
        primary = list(categories.keys())[0]
        positions = categories[primary]
        n = len(positions)

        other_cats = {k: v for k, v in categories.items() if k != primary}

        steps.append(f"Positions: {positions}")
        steps.append(f"Categories to assign: {list(other_cats.keys())}")
        steps.append(f"Constraints: {len(constraints)}")
        steps.append("")

        # Generate all possible assignments for each category
        # Each assignment maps position -> value for that category

        def check_constraints(assignment: Dict[str, Dict[int, str]]) -> bool:
            """Check if current assignment satisfies all constraints."""
            for c in constraints:
                ctype = c["type"]
                items = c["items"]

                # Find positions of each item
                positions_found = []
                for cat, val in items:
                    if cat == primary:
                        # It's a position reference
                        try:
                            pos = categories[primary].index(val)
                            positions_found.append(pos)
                        except ValueError:
                            return False
                    else:
                        # Find which position has this value
                        found = False
                        for pos, assigned_val in assignment.get(cat, {}).items():
                            if assigned_val == val:
                                positions_found.append(pos)
                                found = True
                                break
                        if not found:
                            # Value not yet assigned, can't check
                            return True  # Assume ok for partial assignments

                if len(positions_found) < 2:
                    continue  # Not enough info to check

                p1, p2 = positions_found[0], positions_found[1]

                if ctype == "same":
                    if p1 != p2:
                        return False
                elif ctype == "different":
                    if p1 == p2:
                        return False
                elif ctype == "adjacent":
                    if abs(p1 - p2) != 1:
                        return False
                elif ctype == "left_of":
                    if p1 >= p2:
                        return False
                elif ctype == "right_of":
                    if p1 <= p2:
                        return False
                elif ctype == "immediate_left":
                    if p1 != p2 - 1:
                        return False
                elif ctype == "immediate_right":
                    if p1 != p2 + 1:
                        return False

            return True

        # Try all permutations
        valid_solutions = []
        cat_names = list(other_cats.keys())

        # Generate all possible complete assignments
        all_perms = [list(permutations(range(n))) for _ in cat_names]

        for perm_combo in product(*all_perms):
            assignment = {}
            for i, cat in enumerate(cat_names):
                assignment[cat] = {pos: other_cats[cat][perm_combo[i][pos]] for pos in range(n)}

            if check_constraints(assignment):
                valid_solutions.append(assignment)

        # Format results
        if len(valid_solutions) == 0:
            steps.append("No valid solution exists")
            return LogicResult(
                solution=None,
                is_unique=False,
                num_solutions=0,
                steps=steps,
                valid=False
            )

        # Format solution as grid
        def format_solution(assignment):
            grid = {}
            for pos in range(n):
                pos_name = positions[pos]
                grid[pos_name] = {cat: assignment[cat][pos] for cat in cat_names}
            return grid

        if len(valid_solutions) == 1:
            sol = format_solution(valid_solutions[0])
            steps.append("Unique solution found:")
            for pos, attrs in sol.items():
                steps.append(f"  {pos}: {attrs}")
            return LogicResult(
                solution=sol,
                is_unique=True,
                num_solutions=1,
                steps=steps
            )
        else:
            steps.append(f"Multiple solutions ({len(valid_solutions)})")
            return LogicResult(
                solution=[format_solution(s) for s in valid_solutions[:5]],  # Limit output
                is_unique=False,
                num_solutions=len(valid_solutions),
                steps=steps
            )

    def check_syllogism(
        self,
        premise1: str,
        premise2: str,
        conclusion: str
    ) -> LogicResult:
        """
        Check if a syllogism is valid.

        Handles categorical syllogisms in standard form:
        - All A are B
        - No A are B
        - Some A are B
        - Some A are not B

        Args:
            premise1: First premise
            premise2: Second premise
            conclusion: Proposed conclusion

        Returns:
            LogicResult indicating validity
        """
        steps = []

        def parse_statement(stmt: str) -> Tuple[str, str, str, bool]:
            """Parse statement into (quantifier, subject, predicate, affirmative)."""
            stmt = stmt.lower().strip()

            patterns = [
                (r"all (\w+) are (\w+)", "all", True),
                (r"no (\w+) are (\w+)", "no", False),
                (r"some (\w+) are (\w+)", "some", True),
                (r"some (\w+) are not (\w+)", "some-not", False),
            ]

            for pattern, quant, affirm in patterns:
                match = re.match(pattern, stmt)
                if match:
                    return (quant, match.group(1), match.group(2), affirm)

            raise ValueError(f"Cannot parse statement: {stmt}")

        try:
            p1 = parse_statement(premise1)
            p2 = parse_statement(premise2)
            conc = parse_statement(conclusion)
        except ValueError as e:
            steps.append(f"Parse error: {e}")
            return LogicResult(
                solution={"valid": False, "reason": str(e)},
                is_unique=True,
                num_solutions=1,
                steps=steps,
                valid=False
            )

        steps.append(f"Premise 1: {p1}")
        steps.append(f"Premise 2: {p2}")
        steps.append(f"Conclusion: {conc}")

        # Find the middle term (appears in both premises but not conclusion)
        terms_p1 = {p1[1], p1[2]}
        terms_p2 = {p2[1], p2[2]}
        terms_conc = {conc[1], conc[2]}

        middle_terms = (terms_p1 & terms_p2) - terms_conc

        if len(middle_terms) != 1:
            steps.append("Invalid syllogism structure: cannot identify unique middle term")
            return LogicResult(
                solution={"valid": False, "reason": "No unique middle term"},
                is_unique=True,
                num_solutions=1,
                steps=steps
            )

        middle = middle_terms.pop()
        steps.append(f"Middle term: {middle}")

        # Check basic syllogism rules
        rules_violated = []

        # Rule 1: Middle term must be distributed at least once
        def is_distributed(quant: str, term: str, subject: str, predicate: str) -> bool:
            if term == subject:
                return quant in ["all", "no"]
            else:  # predicate
                return quant in ["no", "some-not"]

        mid_dist_p1 = is_distributed(p1[0], middle, p1[1], p1[2])
        mid_dist_p2 = is_distributed(p2[0], middle, p2[1], p2[2])

        if not mid_dist_p1 and not mid_dist_p2:
            rules_violated.append("Middle term not distributed (undistributed middle fallacy)")

        # Rule 2: No term may be distributed in conclusion if not distributed in premise
        conc_subject = conc[1]
        conc_predicate = conc[2]

        # Check subject distribution
        subj_dist_conc = is_distributed(conc[0], conc_subject, conc[1], conc[2])
        subj_dist_prem = False
        for p in [p1, p2]:
            if conc_subject in [p[1], p[2]]:
                if is_distributed(p[0], conc_subject, p[1], p[2]):
                    subj_dist_prem = True

        if subj_dist_conc and not subj_dist_prem:
            rules_violated.append("Illicit major/minor: term distributed in conclusion but not premise")

        # Rule 3: Two negative premises = no valid conclusion
        neg_premises = sum(1 for p in [p1, p2] if not p[3])
        if neg_premises == 2:
            rules_violated.append("Two negative premises (exclusive premises fallacy)")

        # Rule 4: Negative premise requires negative conclusion
        if neg_premises == 1 and conc[3]:
            rules_violated.append("Negative premise with affirmative conclusion")

        # Rule 5: Two particular premises = no valid conclusion
        particular = sum(1 for p in [p1, p2] if p[0] in ["some", "some-not"])
        if particular == 2:
            rules_violated.append("Two particular premises")

        if rules_violated:
            steps.append("Rules violated:")
            for r in rules_violated:
                steps.append(f"  - {r}")
            return LogicResult(
                solution={"valid": False, "violations": rules_violated},
                is_unique=True,
                num_solutions=1,
                steps=steps
            )

        steps.append("All syllogism rules satisfied")
        steps.append("The argument is VALID")

        return LogicResult(
            solution={"valid": True},
            is_unique=True,
            num_solutions=1,
            steps=steps
        )

    def solve_river_crossing(
        self,
        actors: List[str],
        boat_capacity: int,
        forbidden_pairs: List[Tuple[str, str]],
        must_row: Optional[List[str]] = None
    ) -> LogicResult:
        """
        Solve a river crossing puzzle.

        Args:
            actors: List of actors that need to cross
            boat_capacity: Max actors per trip (including rower)
            forbidden_pairs: Pairs that cannot be left alone together
            must_row: Actors capable of rowing (default: all)

        Returns:
            LogicResult with sequence of moves
        """
        steps = []

        if must_row is None:
            must_row = actors.copy()

        steps.append(f"Actors: {actors}")
        steps.append(f"Boat capacity: {boat_capacity}")
        steps.append(f"Forbidden pairs: {forbidden_pairs}")
        steps.append(f"Can row: {must_row}")
        steps.append("")

        # State: (frozenset on left bank, frozenset on right bank, boat_position)
        # boat_position: 'left' or 'right'

        initial = (frozenset(actors), frozenset(), 'left')
        goal = (frozenset(), frozenset(actors), 'right')

        def is_safe(bank: frozenset) -> bool:
            """Check if a bank configuration is safe."""
            for a, b in forbidden_pairs:
                if a in bank and b in bank:
                    # Check if someone who can supervise is present
                    # For classic puzzles, this is usually the farmer
                    return False
            return True

        def get_moves(state):
            """Generate valid moves from current state."""
            left, right, boat_pos = state
            current_bank = left if boat_pos == 'left' else right
            other_bank = right if boat_pos == 'left' else left
            new_boat_pos = 'right' if boat_pos == 'left' else 'left'

            moves = []

            # Try all combinations of passengers
            rowers_available = [a for a in current_bank if a in must_row]

            for rower in rowers_available:
                # Rower alone
                new_current = current_bank - {rower}
                new_other = other_bank | {rower}

                if is_safe(new_current):
                    if boat_pos == 'left':
                        moves.append(((new_current, new_other, new_boat_pos), [rower]))
                    else:
                        moves.append(((new_other, new_current, new_boat_pos), [rower]))

                # Rower with passengers
                if boat_capacity > 1:
                    passengers = current_bank - {rower}
                    for p in passengers:
                        new_current = current_bank - {rower, p}
                        new_other = other_bank | {rower, p}

                        if is_safe(new_current):
                            if boat_pos == 'left':
                                moves.append(((new_current, new_other, new_boat_pos), [rower, p]))
                            else:
                                moves.append(((new_other, new_current, new_boat_pos), [rower, p]))

            return moves

        # BFS to find shortest solution
        from collections import deque

        queue = deque([(initial, [])])
        visited = {initial}

        while queue:
            state, path = queue.popleft()

            if state == goal:
                steps.append(f"Solution found in {len(path)} crossings:")
                for i, (move, direction) in enumerate(path):
                    steps.append(f"  {i+1}. {move} goes {direction}")

                return LogicResult(
                    solution=path,
                    is_unique=False,  # May be other solutions
                    num_solutions=1,  # Found at least one
                    steps=steps
                )

            for new_state, passengers in get_moves(state):
                if new_state not in visited:
                    visited.add(new_state)
                    direction = "right →" if state[2] == 'left' else "← left"
                    queue.append((new_state, path + [(passengers, direction)]))

        steps.append("No solution exists!")
        return LogicResult(
            solution=None,
            is_unique=False,
            num_solutions=0,
            steps=steps,
            valid=False
        )

    def propositional_truth_table(
        self,
        expression: str,
        variables: List[str]
    ) -> LogicResult:
        """
        Generate truth table for a propositional logic expression.

        Args:
            expression: Logical expression using & (and), | (or), ~ (not), -> (implies), <-> (iff)
            variables: List of variable names

        Returns:
            LogicResult with truth table
        """
        steps = []

        # Convert expression to Python-evaluable form
        expr = expression.replace('->', ' <= ').replace('<->', ' == ')  # Hacky but works
        expr = expr.replace('~', ' not ')

        steps.append(f"Expression: {expression}")
        steps.append(f"Variables: {variables}")
        steps.append("")

        # Generate all combinations
        table = []
        header = variables + [expression]
        table.append(header)

        for values in product([False, True], repeat=len(variables)):
            env = dict(zip(variables, values))
            try:
                result = eval(expr, {"__builtins__": {}}, env)
            except Exception as e:
                steps.append(f"Error evaluating: {e}")
                return LogicResult(
                    solution=None,
                    is_unique=False,
                    num_solutions=0,
                    steps=steps,
                    valid=False
                )

            row = [str(v)[0] for v in values] + [str(result)[0]]
            table.append(row)

        # Format table
        steps.append("Truth Table:")
        col_widths = [max(len(str(row[i])) for row in table) for i in range(len(header))]

        for row in table:
            formatted = " | ".join(str(cell).center(col_widths[i]) for i, cell in enumerate(row))
            steps.append(f"  {formatted}")
            if row == header:
                steps.append("  " + "-+-".join("-" * w for w in col_widths))

        # Check properties
        results = [row[-1] for row in table[1:]]

        is_tautology = all(r == 'T' for r in results)
        is_contradiction = all(r == 'F' for r in results)
        is_contingent = not is_tautology and not is_contradiction

        steps.append("")
        if is_tautology:
            steps.append("This is a TAUTOLOGY (always true)")
        elif is_contradiction:
            steps.append("This is a CONTRADICTION (always false)")
        else:
            steps.append("This is CONTINGENT (sometimes true, sometimes false)")

        return LogicResult(
            solution={
                "table": table,
                "is_tautology": is_tautology,
                "is_contradiction": is_contradiction,
                "is_contingent": is_contingent
            },
            is_unique=True,
            num_solutions=1,
            steps=steps
        )


# Tool interface for Svend
def logic_tool(operation: str, **kwargs) -> Dict[str, Any]:
    """
    Main entry point for logic puzzles tool.

    Operations:
        - knights_knaves: Solve knights and knaves puzzle
        - zebra: Solve zebra/Einstein puzzle
        - syllogism: Check syllogism validity
        - river_crossing: Solve river crossing puzzle
        - truth_table: Generate propositional logic truth table
    """
    engine = LogicPuzzleEngine()

    if operation == "knights_knaves":
        result = engine.solve_knights_knaves(
            kwargs["people"],
            kwargs["statements"]
        )

    elif operation == "zebra":
        result = engine.solve_zebra_puzzle(
            kwargs["categories"],
            kwargs["constraints"]
        )

    elif operation == "syllogism":
        result = engine.check_syllogism(
            kwargs["premise1"],
            kwargs["premise2"],
            kwargs["conclusion"]
        )

    elif operation == "river_crossing":
        result = engine.solve_river_crossing(
            kwargs["actors"],
            kwargs["boat_capacity"],
            kwargs["forbidden_pairs"],
            kwargs.get("must_row")
        )

    elif operation == "truth_table":
        result = engine.propositional_truth_table(
            kwargs["expression"],
            kwargs["variables"]
        )

    else:
        raise ValueError(f"Unknown operation: {operation}")

    return {
        "solution": result.solution,
        "is_unique": result.is_unique,
        "num_solutions": result.num_solutions,
        "valid": result.valid,
        "steps": result.steps
    }


def register_logic_tools(registry) -> None:
    """Register logic puzzle tools with the registry."""
    from .registry import Tool, ToolParameter, ToolResult, ToolStatus

    def _logic_execute(**kwargs) -> ToolResult:
        try:
            result = logic_tool(**kwargs)
            return ToolResult(status=ToolStatus.SUCCESS, output=result)
        except Exception as e:
            return ToolResult(status=ToolStatus.ERROR, output=None, error=str(e))

    registry.register(Tool(
        name="logic_puzzle",
        description="Logic puzzle solver: knights/knaves, zebra puzzles, syllogisms, river crossing, truth tables",
        parameters=[
            ToolParameter(name="operation", type="string", description="Operation: knights_knaves, zebra, syllogism, river_crossing, truth_table", required=True),
            ToolParameter(name="people", type="array", description="List of people (for knights_knaves)", required=False),
            ToolParameter(name="statements", type="array", description="List of statement dicts with speaker, claim_type, about (for knights_knaves)", required=False),
            ToolParameter(name="categories", type="object", description="Dict of category -> options (for zebra)", required=False),
            ToolParameter(name="constraints", type="array", description="List of constraint dicts (for zebra)", required=False),
            ToolParameter(name="premise1", type="string", description="First premise (for syllogism)", required=False),
            ToolParameter(name="premise2", type="string", description="Second premise (for syllogism)", required=False),
            ToolParameter(name="conclusion", type="string", description="Conclusion to check (for syllogism)", required=False),
            ToolParameter(name="actors", type="array", description="List of actors (for river_crossing)", required=False),
            ToolParameter(name="boat_capacity", type="integer", description="Boat capacity (for river_crossing)", required=False),
            ToolParameter(name="forbidden_pairs", type="array", description="Pairs that can't be alone (for river_crossing)", required=False),
            ToolParameter(name="must_row", type="array", description="Actors who can row (for river_crossing)", required=False),
            ToolParameter(name="expression", type="string", description="Logical expression (for truth_table)", required=False),
            ToolParameter(name="variables", type="array", description="Variable names (for truth_table)", required=False),
        ],
        execute_fn=_logic_execute
    ))
