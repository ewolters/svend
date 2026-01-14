#!/usr/bin/env python3
"""
Test V0.2 tools: state_machine, constraint, enumerate.

These are the "epistemic honesty" tools that eliminate error classes.
"""

import sys
sys.path.insert(0, '.')

from src.tools import create_v0_registry


def test_state_machine():
    """Test state machine simulation."""
    print("\n=== State Machine ===")
    registry = create_v0_registry()
    tool = registry.get("state_machine")

    # Simple state machine: counter with threshold
    initial = {"name": "low", "properties": {"count": 0}}
    transitions = [
        {"name": "increment", "from": "low", "to": "low", "guard": {"count": {"lt": 10}}, "effects": {"count": {"add": 1}}},
        {"name": "increment", "from": "low", "to": "high", "guard": {"count": {"gte": 10}}, "effects": {"count": {"add": 1}}},
        {"name": "reset", "from": "high", "to": "low", "effects": {"count": {"set": 0}}},
    ]

    # Query available actions
    result = tool.execute(operation="available_actions", initial_state=initial, transitions=transitions)
    print(f"Available from low (count=0): {result.output}")
    assert "increment" in result.output

    # What if we increment 10 times?
    actions = ["increment"] * 11
    result = tool.execute(operation="what_if", initial_state=initial, transitions=transitions, actions=actions)
    print(f"After 11 increments: {result.output}")
    assert "high" in result.output.lower()

    # Resource simulation
    result = tool.execute(
        operation="resource_sim",
        initial_resources={"gold": 100, "wood": 50},
        resource_actions=[
            {"type": "remove", "resource": "gold", "amount": 30},
            {"type": "add", "resource": "wood", "amount": 20},
            {"type": "transfer", "from": "gold", "to": "soldiers", "amount": 50, "ratio": 0.1}
        ]
    )
    print(f"Resource sim: {result.output}")
    assert "succeeded" in result.output.lower()

    print("[OK] State machine tests passed")


def test_constraint_solver():
    """Test constraint feasibility checking."""
    print("\n=== Constraint Solver ===")
    registry = create_v0_registry()
    tool = registry.get("constraint")

    # Satisfiable problem: find x,y where x < y
    result = tool.execute(
        operation="check_feasibility",
        variables={"x": {"range": [1, 5]}, "y": {"range": [1, 5]}},
        constraints=[
            {"type": "less_than", "var1": "x", "var2": "y"}
        ]
    )
    print(f"Satisfiable (x < y): {result.output}")
    assert "SATISFIABLE" in result.output

    # Unsatisfiable problem: x < y AND y < x
    result = tool.execute(
        operation="check_feasibility",
        variables={"x": {"range": [1, 5]}, "y": {"range": [1, 5]}},
        constraints=[
            {"type": "less_than", "var1": "x", "var2": "y"},
            {"type": "less_than", "var1": "y", "var2": "x"}
        ]
    )
    print(f"Unsatisfiable (x<y AND y<x): {result.output}")
    assert "UNSATISFIABLE" in result.output

    # Check assignment
    result = tool.execute(
        operation="check_assignment",
        variables={"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]},
        constraints=[{"type": "all_different", "variables": ["a", "b", "c"]}],
        assignment={"a": 1, "b": 2, "c": 3}
    )
    print(f"Valid assignment: {result.output}")
    assert "SATISFIES" in result.output

    # Invalid assignment
    result = tool.execute(
        operation="check_assignment",
        variables={"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]},
        constraints=[{"type": "all_different", "variables": ["a", "b", "c"]}],
        assignment={"a": 1, "b": 1, "c": 3}
    )
    print(f"Invalid assignment: {result.output}")
    assert "VIOLATES" in result.output

    # Scheduling check - impossible deadline
    result = tool.execute(
        operation="check_schedule",
        tasks=[
            {"name": "A", "duration": 10, "earliest_start": 0, "deadline": 5}  # Impossible!
        ]
    )
    print(f"Impossible schedule: {result.output}")
    assert "IMPOSSIBLE" in result.output

    print("[OK] Constraint solver tests passed")


def test_enumerator():
    """Test exhaustive enumeration."""
    print("\n=== Enumerator ===")
    registry = create_v0_registry()
    tool = registry.get("enumerate")

    # Find all primes between 1 and 20
    result = tool.execute(
        operation="search",
        space_type="integers",
        low=1,
        high=20,
        condition_type="prime",
        find_all=True
    )
    print(f"Primes 1-20: {result.output}")
    assert "VERIFIED" in result.output
    assert result.metadata["satisfying_count"] == 8  # 2,3,5,7,11,13,17,19

    # None satisfy: find perfect square that's also prime > 2
    result = tool.execute(
        operation="search",
        space_type="integers",
        low=4,
        high=100,
        condition_type="custom",
        condition="(x**0.5 == int(x**0.5)) and all(x % i != 0 for i in range(2, int(x**0.5)+1))",
        find_all=True
    )
    print(f"Perfect squares that are prime (4-100): {result.output}")
    assert "NONE" in result.output or result.metadata["satisfying_count"] == 0

    # Count search space
    result = tool.execute(
        operation="count",
        space_type="combinations",
        items=[1, 2, 3, 4, 5],
        size=3
    )
    print(f"C(5,3) count: {result.output}")
    assert "10" in result.output

    # All pairs of [1,2,3] that sum to even
    result = tool.execute(
        operation="search",
        space_type="pairs",
        items=[1, 2, 3, 4],
        condition_type="custom",
        condition="(x[0] + x[1]) % 2 == 0",
        find_all=True
    )
    print(f"Pairs summing to even: {result.output}")
    # (1,3), (2,4), (1,3), (2,4) - wait, pairs are distinct so: (1,3), (2,4)
    assert "VERIFIED" in result.output

    # Too large search space
    result = tool.execute(
        operation="search",
        space_type="tuples",
        items=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        size=8,
        condition_type="prime",
        max_check=1000
    )
    print(f"Large space: {result.output}")
    assert "CANNOT_VERIFY" in result.output or "exceeds" in result.output.lower()

    print("[OK] Enumerator tests passed")


def main():
    print("=" * 60)
    print("Testing V0.2 Tools - Epistemic Honesty")
    print("=" * 60)

    try:
        test_state_machine()
        test_constraint_solver()
        test_enumerator()

        print("\n" + "=" * 60)
        print("All V0.2 tests passed!")
        print("=" * 60)

        # Summary
        registry = create_v0_registry()
        print(f"\nTotal tools: {len(registry.list_tools())}")

        v02_tools = ["state_machine", "constraint", "enumerate"]
        print(f"V0.2 tools: {', '.join(v02_tools)}")

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
