#!/usr/bin/env python3
"""
Unit tests for the tool system.

Run with: py -m pytest tests/test_tools.py -v
Or just: py tests/test_tools.py
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestSymbolicMath:
    """Tests for symbolic math operations."""

    def test_differentiate(self):
        from src.tools.math_engine import SymbolicSolver
        solver = SymbolicSolver()

        result = solver.differentiate("x**3 + 2*x", "x")
        assert result["success"]
        assert "3*x**2 + 2" in result["derivative"]

    def test_integrate(self):
        from src.tools.math_engine import SymbolicSolver
        solver = SymbolicSolver()

        result = solver.integrate("x**2", "x")
        assert result["success"]
        assert "x**3/3" in result["antiderivative"]

    def test_solve_equation(self):
        from src.tools.math_engine import SymbolicSolver
        solver = SymbolicSolver()

        result = solver.solve_equation("x**2 - 4", "x")
        assert result["success"]
        solutions = result["solutions"]
        assert "-2" in solutions or "2" in solutions

    def test_simplify(self):
        from src.tools.math_engine import SymbolicSolver
        solver = SymbolicSolver()

        result = solver.simplify("(x**2 - 1)/(x - 1)")
        assert result["success"]
        assert "x + 1" in result["simplified"]

    def test_limit(self):
        from src.tools.math_engine import SymbolicSolver
        solver = SymbolicSolver()

        result = solver.limit("sin(x)/x", "x", "0")
        assert result["success"]
        assert result["limit"] == "1"

    def test_series(self):
        from src.tools.math_engine import SymbolicSolver
        solver = SymbolicSolver()

        result = solver.series("exp(x)", "x", "0", 4)
        assert result["success"]
        assert "1 + x" in result["series"]

    def test_factor(self):
        from src.tools.math_engine import SymbolicSolver
        solver = SymbolicSolver()

        result = solver.factor("x**2 - 9")
        assert result["success"]
        assert "(x - 3)" in result["factored"]
        assert "(x + 3)" in result["factored"]

    def test_expand(self):
        from src.tools.math_engine import SymbolicSolver
        solver = SymbolicSolver()

        result = solver.expand("(x + 1)**2")
        assert result["success"]
        assert "x**2" in result["expanded"]

    def test_matrix_determinant(self):
        from src.tools.math_engine import SymbolicSolver
        solver = SymbolicSolver()

        result = solver.matrix_operations("determinant", [[1, 2], [3, 4]])
        assert result["success"]
        assert result["result"] == "-2"

    def test_matrix_inverse(self):
        from src.tools.math_engine import SymbolicSolver
        solver = SymbolicSolver()

        result = solver.matrix_operations("inverse", [[1, 2], [3, 4]])
        assert result["success"]
        assert result["result"] is not None


class TestChemistry:
    """Tests for chemistry tools."""

    def test_molecular_weight(self):
        from src.tools.chemistry import calculate_molecular_weight

        # Water: 2*1.008 + 16 = 18.016
        mw = calculate_molecular_weight("H2O")
        assert abs(mw - 18.015) < 0.1

        # NaCl: 22.99 + 35.45 = 58.44
        mw = calculate_molecular_weight("NaCl")
        assert abs(mw - 58.44) < 0.1

    def test_parse_formula(self):
        from src.tools.chemistry import parse_formula

        elements = parse_formula("H2O")
        assert elements == {"H": 2, "O": 1}

        elements = parse_formula("Ca(OH)2")
        assert elements == {"Ca": 1, "O": 2, "H": 2}

    def test_balance_equation(self):
        from src.tools.chemistry import balance_equation

        result = balance_equation("H2 + O2 -> H2O")
        assert result["success"]
        assert "2H2" in result["balanced_equation"] or "2 H2" in result["balanced_equation"].replace("H2O", "")

    def test_ph_strong_acid(self):
        from src.tools.chemistry import ph_calculation

        result = ph_calculation(0.01, "acid", strong=True)
        assert result["success"]
        assert abs(result["pH"] - 2.0) < 0.1

    def test_ph_strong_base(self):
        from src.tools.chemistry import ph_calculation

        result = ph_calculation(0.01, "base", strong=True)
        assert result["success"]
        assert abs(result["pH"] - 12.0) < 0.1

    def test_dilution(self):
        from src.tools.chemistry import dilution

        # C1V1 = C2V2: 2.0 * 50 = 0.5 * V2 => V2 = 200
        result = dilution(C1=2.0, V1=50, C2=0.5)
        assert result["success"]
        assert abs(result["V2"] - 200) < 0.1

    def test_percent_composition(self):
        from src.tools.chemistry import percent_composition

        result = percent_composition("H2O")
        assert result["success"]
        # Hydrogen should be about 11%, Oxygen about 89%
        assert result["composition"]["H"]["percent"] < 15
        assert result["composition"]["O"]["percent"] > 80


class TestPhysics:
    """Tests for physics tools."""

    def test_unit_convert_length(self):
        from src.tools.physics import convert_units

        result = convert_units(1000, "m", "km")
        assert result["success"]
        assert abs(result["result"]["value"] - 1.0) < 0.001

    def test_unit_convert_temperature(self):
        from src.tools.physics import convert_units

        result = convert_units(100, "C", "F")
        assert result["success"]
        assert abs(result["result"]["value"] - 212) < 0.1

    def test_kinematics_final_velocity(self):
        from src.tools.physics import kinematics

        # v = v0 + at: 0 + 2*5 = 10
        result = kinematics({"v0": 0, "a": 2, "t": 5}, "v")
        assert result["success"]
        assert abs(result["v"] - 10) < 0.1

    def test_projectile(self):
        from src.tools.physics import projectile_motion

        result = projectile_motion(v0=20, theta=45)
        assert result["success"]
        assert result["range"] > 0
        assert result["max_height"] > 0

    def test_kinetic_energy(self):
        from src.tools.physics import energy_work

        # KE = 0.5 * m * v^2 = 0.5 * 2 * 10^2 = 100
        result = energy_work("kinetic_energy", m=2, v=10)
        assert result["success"]
        assert abs(result["KE"] - 100) < 0.1

    def test_ohms_law(self):
        from src.tools.physics import electricity

        # V = IR: I = V/R = 12/4 = 3
        result = electricity("ohms_law", V=12, R=4)
        assert result["success"]
        assert abs(result["I"] - 3) < 0.1

    def test_series_resistance(self):
        from src.tools.physics import electricity

        result = electricity("series_resistance", resistances=[10, 20, 30])
        assert result["success"]
        assert abs(result["R_total"] - 60) < 0.1

    def test_thin_lens(self):
        from src.tools.physics import optics

        # 1/f = 1/do + 1/di: 1/0.1 = 1/0.3 + 1/di => di = 0.15
        result = optics("thin_lens", f=0.1, do=0.3)
        assert result["success"]
        assert result["di"] > 0

    def test_shm_spring(self):
        from src.tools.physics import simple_harmonic_motion

        result = simple_harmonic_motion(type="spring", m=1, k=100)
        assert result["success"]
        assert result["period"] > 0
        assert result["frequency"] > 0


class TestLogicSolver:
    """Tests for Z3 logic solver."""

    @staticmethod
    def z3_available():
        try:
            import z3
            return True
        except ImportError:
            return False

    def test_check_sat_satisfiable(self):
        if not self.z3_available():
            print("  [SKIP] Z3 not installed")
            return

        from src.tools.math_engine import Z3Solver
        solver = Z3Solver()

        result = solver.check_satisfiability(
            ["x > 0", "x < 10"],
            {"x": "int"}
        )
        assert result["success"]
        assert result["satisfiable"] == True

    def test_check_sat_unsatisfiable(self):
        if not self.z3_available():
            print("  [SKIP] Z3 not installed")
            return

        from src.tools.math_engine import Z3Solver
        solver = Z3Solver()

        result = solver.check_satisfiability(
            ["x > 10", "x < 5"],
            {"x": "int"}
        )
        assert result["success"]
        assert result["satisfiable"] == False

    def test_prove_valid(self):
        if not self.z3_available():
            print("  [SKIP] Z3 not installed")
            return

        from src.tools.math_engine import Z3Solver
        solver = Z3Solver()

        result = solver.prove(
            ["x > 0", "y > 0"],
            "x + y > 0",
            {"x": "real", "y": "real"}
        )
        assert result["success"]
        assert result["proven"] == True


def run_tests():
    """Run all tests manually if pytest not available."""
    import traceback

    test_classes = [
        TestSymbolicMath,
        TestChemistry,
        TestPhysics,
        TestLogicSolver,
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {test_class.__name__}")
        print('='*60)

        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith("test_")]

        for method_name in methods:
            total += 1
            try:
                method = getattr(instance, method_name)
                method()
                print(f"  [PASS] {method_name}")
                passed += 1
            except AssertionError as e:
                print(f"  [FAIL] {method_name}: {e}")
                failed += 1
                errors.append((test_class.__name__, method_name, str(e)))
            except Exception as e:
                print(f"  [ERROR] {method_name}: {e}")
                failed += 1
                errors.append((test_class.__name__, method_name, traceback.format_exc()))

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print('='*60)

    if errors:
        print("\nFailed tests:")
        for cls, method, error in errors:
            print(f"\n  {cls}.{method}:")
            print(f"    {error[:200]}")

    return failed == 0


if __name__ == "__main__":
    # Try pytest first, fall back to manual runner
    try:
        import pytest
        sys.exit(pytest.main([__file__, "-v"]))
    except ImportError:
        print("pytest not installed, running manual test runner")
        success = run_tests()
        sys.exit(0 if success else 1)
