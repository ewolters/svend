#!/usr/bin/env python3
"""
Quick test for new V0.1 tools: combinatorics, graph, geometry, sequence, finance.
"""

import sys
sys.path.insert(0, '.')

from src.tools import create_v0_registry


def test_combinatorics():
    """Test combinatorics tool."""
    print("\n=== Combinatorics ===")
    registry = create_v0_registry()
    tool = registry.get("combinatorics")

    # Permutations
    result = tool.execute(operation="permutations", n=5, r=3)
    print(f"P(5,3): {result.output}")
    assert "60" in result.output

    # Combinations
    result = tool.execute(operation="combinations", n=10, r=3)
    print(f"C(10,3): {result.output}")
    assert "120" in result.output

    # Catalan
    result = tool.execute(operation="catalan", n=5)
    print(f"Catalan(5): {result.output}")
    assert "42" in result.output

    # Bell
    result = tool.execute(operation="bell", n=5)
    print(f"Bell(5): {result.output}")
    assert "52" in result.output

    # Integer partitions
    result = tool.execute(operation="partitions", n=5)
    print(f"Partitions(5): {result.output}")

    print("[OK] Combinatorics tests passed")


def test_graph():
    """Test graph algorithms tool."""
    print("\n=== Graph Algorithms ===")
    registry = create_v0_registry()
    tool = registry.get("graph")

    # Simple graph
    edges = [["A", "B", 1], ["B", "C", 2], ["A", "C", 4], ["C", "D", 1]]

    # BFS
    result = tool.execute(operation="bfs", edges=edges, start="A")
    print(f"BFS from A: {result.output}")

    # Shortest path
    result = tool.execute(operation="shortest_path", edges=edges, start="A", end="D")
    print(f"Shortest A to D: {result.output}")
    assert "A -> B -> C -> D" in result.output or "distance: 4" in result.output

    # Connected components
    result = tool.execute(operation="connected_components", edges=edges)
    print(f"Components: {result.output}")

    # Cycle detection
    result = tool.execute(operation="has_cycle", edges=edges)
    print(f"Has cycle: {result.output}")

    # MST
    result = tool.execute(operation="mst", edges=edges)
    print(f"MST: {result.output}")

    print("[OK] Graph tests passed")


def test_geometry():
    """Test geometry tool."""
    print("\n=== Geometry ===")
    registry = create_v0_registry()
    tool = registry.get("geometry")

    # Distance
    result = tool.execute(operation="distance", points=[[0, 0], [3, 4]])
    print(f"Distance (0,0) to (3,4): {result.output}")
    assert "5" in result.output

    # Midpoint
    result = tool.execute(operation="midpoint", points=[[0, 0], [10, 10]])
    print(f"Midpoint: {result.output}")

    # Line from points
    result = tool.execute(operation="line_from_points", points=[[0, 0], [1, 1]])
    print(f"Line through (0,0),(1,1): {result.output}")

    # Triangle
    result = tool.execute(operation="triangle", points=[[0, 0], [4, 0], [2, 3]])
    print(f"Triangle: {result.output}")

    # Circle from 3 points
    result = tool.execute(operation="circle_from_points", points=[[0, 0], [4, 0], [2, 2]])
    print(f"Circumcircle: {result.output}")

    print("[OK] Geometry tests passed")


def test_sequence():
    """Test sequence analyzer tool."""
    print("\n=== Sequence Analyzer ===")
    registry = create_v0_registry()
    tool = registry.get("sequence")

    # Arithmetic sequence
    result = tool.execute(operation="analyze", sequence=[2, 5, 8, 11, 14])
    print(f"Arithmetic [2,5,8,11,14]: {result.output}")

    # Geometric sequence
    result = tool.execute(operation="analyze", sequence=[3, 6, 12, 24, 48])
    print(f"Geometric [3,6,12,24,48]: {result.output}")

    # Fibonacci-like
    result = tool.execute(operation="analyze", sequence=[1, 1, 2, 3, 5, 8, 13])
    print(f"Fibonacci: {result.output}")

    # Polynomial (squares)
    result = tool.execute(operation="analyze", sequence=[1, 4, 9, 16, 25])
    print(f"Squares [1,4,9,16,25]: {result.output}")

    # Generate
    result = tool.execute(operation="generate", sequence_type="prime", n=10)
    print(f"First 10 primes: {result.output}")

    print("[OK] Sequence tests passed")


def test_finance():
    """Test finance tool."""
    print("\n=== Finance ===")
    registry = create_v0_registry()
    tool = registry.get("finance")

    # Future value
    result = tool.execute(operation="future_value", present_value=1000, rate=0.05, periods=10)
    print(f"FV of $1000 at 5% for 10yr: {result.output}")

    # Loan payment
    result = tool.execute(operation="loan_payment", principal=200000, rate=0.005, periods=360)
    print(f"$200K mortgage at 6%/yr (0.5%/mo) 30yr: {result.output}")

    # NPV
    result = tool.execute(operation="npv", rate=0.10, cash_flows=[-1000, 300, 400, 500, 600])
    print(f"NPV at 10%: {result.output}")

    # IRR
    result = tool.execute(operation="irr", cash_flows=[-1000, 300, 400, 500, 600])
    print(f"IRR: {result.output}")

    # Bond price
    result = tool.execute(
        operation="bond_price",
        face_value=1000,
        coupon_rate=0.06,
        years=10,
        yield_to_maturity=0.08
    )
    print(f"Bond price (6% coupon, 8% YTM): {result.output}")

    # Rule of 72
    result = tool.execute(operation="rule_of_72", rate=0.07)
    print(f"Rule of 72 at 7%: {result.output}")

    print("[OK] Finance tests passed")


def main():
    print("=" * 60)
    print("Testing New V0.1 Tools")
    print("=" * 60)

    try:
        test_combinatorics()
        test_graph()
        test_geometry()
        test_sequence()
        test_finance()

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)

        # List all registered tools
        registry = create_v0_registry()
        print(f"\nTotal tools registered: {len(registry.list_tools())}")
        print("Tools:", ", ".join(sorted(registry.list_tools())))

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
