#!/usr/bin/env python3
"""
Generate synthetic tool-calling training data using Claude API.

This script generates training examples that teach a model to:
1. Identify when tool use is appropriate
2. Format tool calls correctly
3. Interpret tool results
4. Integrate results into reasoning

Usage:
    python scripts/generate_tool_data.py --num-examples 1000 --output data/tool_traces.jsonl
    python scripts/generate_tool_data.py --domain math --num-examples 500
    python scripts/generate_tool_data.py --dry-run  # Preview problems

Requires: ANTHROPIC_API_KEY environment variable
"""

import os
import sys
import json
import argparse
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env file if it exists
env_path = project_root / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value


# =============================================================================
# Tool Definitions (matching the enriched tools)
# =============================================================================

TOOL_DEFINITIONS = {
    "symbolic_math": {
        "description": "Symbolic mathematics using SymPy",
        "operations": [
            "simplify",        # simplify expressions
            "solve",           # solve equations
            "differentiate",   # d/dx
            "integrate",       # (definite and indefinite)
            "evaluate",        # substitute values
            "limit",           # compute limits (including at infinity)
            "series",          # Taylor/Maclaurin series
            "factor",          # factor polynomials
            "expand",          # expand polynomials
            "matrix",          # determinant, inverse, eigenvalues, rref, multiply
        ],
        "examples": [
            {"operation": "differentiate", "expression": "x**3 + 2*x", "variable": "x"},
            {"operation": "limit", "expression": "sin(x)/x", "variable": "x", "point": "0"},
            {"operation": "matrix", "matrix": "[[1,2],[3,4]]", "values": "determinant"},
        ],
    },
    # V0.2 Epistemic Tools - Error Class Elimination
    "state_machine": {
        "description": "Discrete state transitions and resource simulations - eliminates 'what happens next' hallucinations",
        "operations": [
            "what_if",          # simulate sequence of actions
            "available_actions", # get valid actions from current state
            "resource_sim",     # simulate resource flows
        ],
        "examples": [
            {"operation": "what_if", "initial_state": {"name": "idle", "properties": {"count": 0}},
             "transitions": [{"name": "start", "from": "idle", "to": "running"}], "actions": ["start"]},
            {"operation": "resource_sim", "initial_resources": {"gold": 100},
             "resource_actions": [{"type": "remove", "resource": "gold", "amount": 30}]},
        ],
    },
    "constraint": {
        "description": "SAT/UNSAT feasibility checking - prevents proposing impossible plans",
        "operations": [
            "check_feasibility", # check if constraints can be satisfied
            "check_assignment",  # validate specific assignment
            "check_schedule",    # check scheduling feasibility
        ],
        "examples": [
            {"operation": "check_feasibility",
             "variables": {"x": {"range": [1, 10]}, "y": {"range": [1, 10]}},
             "constraints": [{"type": "less_than", "var1": "x", "var2": "y"}]},
            {"operation": "check_schedule",
             "tasks": [{"name": "A", "duration": 5, "earliest_start": 0, "deadline": 10}]},
        ],
    },
    "enumerate": {
        "description": "Bounded exhaustive search - eliminates 'seems unlikely' without verification",
        "operations": [
            "search",  # find items satisfying condition
            "count",   # count search space size
        ],
        "examples": [
            {"operation": "search", "space_type": "integers", "low": 1, "high": 20,
             "condition_type": "prime", "find_all": True},
            {"operation": "count", "space_type": "combinations", "items": [1,2,3,4,5], "size": 3},
        ],
    },
    "counterfactual": {
        "description": "Sensitivity analysis - reports 'robust to X, sensitive to Y'",
        "operations": [
            "sensitivity",  # analyze parameter sensitivity
            "scenarios",    # compare named scenarios
            "boundary",     # find threshold values
            "what_if",      # evaluate expressions with varied inputs
        ],
        "examples": [
            {"operation": "sensitivity", "analysis_type": "breakeven",
             "base_inputs": {"revenue_per_unit": 100, "cost_per_unit": 60, "fixed_costs": 10000, "units": 300}},
            {"operation": "scenarios", "analysis_type": "investment",
             "scenarios": {"conservative": {"initial": 1000, "rate": 0.05, "years": 10},
                          "aggressive": {"initial": 1000, "rate": 0.12, "years": 10}}},
        ],
    },
    "execute_python": {
        "description": "Sandboxed Python execution for numerical computation",
        "operations": ["execute"],
        "examples": [
            {"code": "import math\nresult = sum(1/math.factorial(n) for n in range(10))\nprint(f'e ≈ {result}')"},
            {"code": "primes = [n for n in range(2, 100) if all(n % i != 0 for i in range(2, int(n**0.5)+1))]\nprint(primes[:10])"},
        ],
    },
    "logic_solver": {
        "description": "Z3-based SAT/SMT solver for logical reasoning",
        "operations": [
            "check_sat",       # is formula satisfiable?
            "prove",           # prove theorem (shows unsatisfiability of negation)
        ],
        "examples": [
            {"operation": "check_sat", "constraints": '["x > 0", "x < 10", "x*x == 25"]', "variables": '{"x": "int"}'},
            {"operation": "prove", "constraints": '["x > 0", "y > 0"]', "variables": '{"x": "real", "y": "real"}', "conclusion": "x + y > 0"},
        ],
    },
    "chemistry": {
        "description": "Chemistry calculations and analysis",
        "operations": [
            "molecular_weight",    # compute molar mass from formula
            "parse_formula",       # parse formula to element counts
            "balance_equation",    # balance chemical equations
            "stoichiometry",       # mole ratios and yields
            "ph",                  # pH from concentration (strong/weak acids/bases with Ka/Kb)
            "molar_conversion",    # grams ↔ moles ↔ particles
            "dilution",            # C1V1 = C2V2
            "concentration",       # molarity and molality
            "percent_composition", # percent by mass of each element
            "limiting_reagent",    # find limiting reagent and theoretical yield
        ],
        "examples": [
            {"operation": "molecular_weight", "formula": "H2SO4"},
            {"operation": "dilution", "C1": 2.0, "V1": 50, "C2": 0.5},
            {"operation": "ph", "concentration": 0.1, "type": "acid", "strong": False, "Ka": 1.8e-5},
        ],
    },
    "physics": {
        "description": "Physics calculations for mechanics, thermodynamics, E&M, optics",
        "operations": [
            "unit_convert",        # convert between physical units
            "kinematics",          # v = v0 + at, x = v0t + ½at², etc.
            "ideal_gas",           # PV = nRT
            "waves",               # v = fλ, E = hf, Doppler
            "constant",            # get physical constants (c, G, h, e, k_B, etc.)
            "energy",              # KE = ½mv², PE = mgh, W = Fd, P = W/t
            "electricity",         # Ohm's law, power, series/parallel, Coulomb's law
            "optics",              # thin lens equation, magnification, Snell's law
            "projectile",          # range, max height, time of flight
            "shm",                 # simple harmonic motion (spring, pendulum)
        ],
        "examples": [
            {"operation": "electricity", "params": {"operation": "ohms_law", "V": 12, "R": 4}},
            {"operation": "projectile", "v0": 20, "theta": 45},
            {"operation": "optics", "params": {"operation": "thin_lens", "f": 0.1, "do": 0.3}},
        ],
    },
}


# =============================================================================
# Problem Templates by Domain
# =============================================================================

class Domain(Enum):
    CALCULUS = "calculus"
    ALGEBRA = "algebra"
    LINEAR_ALGEBRA = "linear_algebra"
    PHYSICS_MECHANICS = "physics_mechanics"
    PHYSICS_EM_OPTICS = "physics_em_optics"
    CHEMISTRY_FUNDAMENTALS = "chemistry_fundamentals"
    CHEMISTRY_SOLUTIONS = "chemistry_solutions"
    LOGIC = "logic"
    NUMERICAL = "numerical"
    MULTI_TOOL = "multi_tool"
    # V0.2 Epistemic domains - Error class elimination
    STATE_REASONING = "state_reasoning"
    CONSTRAINT_REASONING = "constraint_reasoning"
    EXHAUSTIVE_REASONING = "exhaustive_reasoning"
    SENSITIVITY_ANALYSIS = "sensitivity_analysis"
    # Meta-cognitive domains
    FAILURE_RECOGNITION = "failure_recognition"
    CLARIFICATION = "clarification"


# Problem templates with expected tool usage
PROBLEM_TEMPLATES = {
    Domain.CALCULUS: [
        {
            "template": "Find the derivative of f(x) = {expr}",
            "tool": "symbolic_math",
            "operation": "differentiate",
            "vars": {"expr": ["x**3 + 2*x**2 - 5*x + 3", "sin(x)*cos(x)", "exp(x)*ln(x)", "x**4 - 16*x**2", "sqrt(x**2 + 1)", "tan(x)", "x*exp(-x**2)", "ln(x**2 + 1)"]},
            "difficulty": "easy",
        },
        {
            "template": "Compute the integral of {expr} with respect to x",
            "tool": "symbolic_math",
            "operation": "integrate",
            "vars": {"expr": ["x**2", "sin(x)", "exp(-x)", "1/(x+1)", "x*exp(x)", "cos(x)**2", "1/sqrt(1-x**2)", "x**3 + 3*x"]},
            "difficulty": "medium",
        },
        {
            "template": "Evaluate the definite integral from 0 to {upper} of {expr} dx",
            "tool": "symbolic_math",
            "operation": "integrate",
            "vars": {"expr": ["x**2", "sin(x)", "exp(-x)", "x"], "upper": ["1", "pi", "2", "pi/2"]},
            "difficulty": "medium",
        },
        {
            "template": "Find the limit as x approaches {point} of {expr}",
            "tool": "symbolic_math",
            "operation": "limit",
            "vars": {"expr": ["sin(x)/x", "(exp(x)-1)/x", "(1+1/x)**x", "x*ln(x)", "(1-cos(x))/x**2"], "point": ["0", "0", "oo", "0", "0"]},
            "difficulty": "medium",
        },
        {
            "template": "Find the Taylor series of {expr} around x=0 up to order {n}",
            "tool": "symbolic_math",
            "operation": "series",
            "vars": {"expr": ["exp(x)", "sin(x)", "cos(x)", "ln(1+x)", "1/(1-x)"], "n": ["4", "5", "4", "4", "5"]},
            "difficulty": "hard",
        },
        {
            "template": "Find the second derivative of f(x) = {expr}",
            "tool": "symbolic_math",
            "operation": "differentiate",
            "vars": {"expr": ["x**4 - 2*x**2 + 1", "sin(2*x)", "exp(-x**2)", "ln(x**2)", "x*sin(x)"]},
            "difficulty": "medium",
        },
    ],
    Domain.ALGEBRA: [
        {
            "template": "Solve the equation {expr} = 0",
            "tool": "symbolic_math",
            "operation": "solve",
            "vars": {"expr": ["x**2 - 5*x + 6", "x**3 - 8", "x**2 + 4*x + 4", "2*x**2 - 7*x + 3", "x**4 - 1"]},
            "difficulty": "easy",
        },
        {
            "template": "Simplify the expression {expr}",
            "tool": "symbolic_math",
            "operation": "simplify",
            "vars": {"expr": ["(x**2-1)/(x-1)", "sin(x)**2 + cos(x)**2", "(x+1)**2 - x**2 - 1", "exp(ln(x**2))", "(a+b)**2 - a**2 - b**2"]},
            "difficulty": "easy",
        },
        {
            "template": "Factor the polynomial {expr}",
            "tool": "symbolic_math",
            "operation": "factor",
            "vars": {"expr": ["x**2 - 9", "x**3 - 27", "x**4 - 1", "x**2 + 5*x + 6", "2*x**2 - 8"]},
            "difficulty": "easy",
        },
        {
            "template": "Expand {expr}",
            "tool": "symbolic_math",
            "operation": "expand",
            "vars": {"expr": ["(x+1)**3", "(x-2)*(x+2)", "(a+b)**4", "(x**2+1)*(x**2-1)", "(2*x+3)**2"]},
            "difficulty": "easy",
        },
        {
            "template": "Evaluate {expr} when {var}={val}",
            "tool": "symbolic_math",
            "operation": "evaluate",
            "vars": {"expr": ["x**2 + 3*x + 2", "sin(x) + cos(x)", "x**3 - 2*x + 1"], "var": ["x", "x", "x"], "val": ["5", "pi/4", "3"]},
            "difficulty": "easy",
        },
    ],
    Domain.LINEAR_ALGEBRA: [
        {
            "template": "Find the determinant of the matrix {matrix}",
            "tool": "symbolic_math",
            "operation": "matrix",
            "vars": {"matrix": ["[[1,2],[3,4]]", "[[1,0,2],[0,1,3],[2,3,1]]", "[[2,1],[1,2]]", "[[1,2,3],[4,5,6],[7,8,9]]"]},
            "matrix_op": "determinant",
            "difficulty": "medium",
        },
        {
            "template": "Find the inverse of the matrix {matrix}",
            "tool": "symbolic_math",
            "operation": "matrix",
            "vars": {"matrix": ["[[1,2],[3,4]]", "[[2,1],[1,1]]", "[[1,0],[0,1]]"]},
            "matrix_op": "inverse",
            "difficulty": "medium",
        },
        {
            "template": "Find the eigenvalues of the matrix {matrix}",
            "tool": "symbolic_math",
            "operation": "matrix",
            "vars": {"matrix": ["[[2,1],[1,2]]", "[[3,1],[0,2]]", "[[0,1],[-1,0]]"]},
            "matrix_op": "eigenvalues",
            "difficulty": "hard",
        },
        {
            "template": "Reduce the matrix {matrix} to row echelon form",
            "tool": "symbolic_math",
            "operation": "matrix",
            "vars": {"matrix": ["[[1,2,3],[4,5,6],[7,8,9]]", "[[1,2],[3,4],[5,6]]"]},
            "matrix_op": "rref",
            "difficulty": "medium",
        },
    ],
    Domain.PHYSICS_MECHANICS: [
        {
            "template": "A car starts from rest and accelerates at {a} m/s^2 for {t} seconds. What is its final velocity?",
            "tool": "physics",
            "operation": "kinematics",
            "vars": {"a": ["2", "3", "5", "9.8"], "t": ["5", "10", "8", "3"]},
            "difficulty": "easy",
        },
        {
            "template": "A ball is thrown upward with an initial velocity of {v0} m/s. What is the maximum height reached?",
            "tool": "physics",
            "operation": "kinematics",
            "vars": {"v0": ["10", "20", "15", "25"]},
            "difficulty": "medium",
        },
        {
            "template": "A projectile is launched at {v0} m/s at an angle of {theta} degrees above horizontal. Find the range and maximum height.",
            "tool": "physics",
            "operation": "projectile",
            "vars": {"v0": ["20", "30", "50", "100"], "theta": ["30", "45", "60", "37"]},
            "difficulty": "medium",
        },
        {
            "template": "What is the kinetic energy of a {m} kg object moving at {v} m/s?",
            "tool": "physics",
            "operation": "energy",
            "vars": {"m": ["2", "5", "10", "0.5"], "v": ["10", "5", "3", "20"]},
            "difficulty": "easy",
        },
        {
            "template": "A spring with spring constant k = {k} N/m is compressed by {x} m. What is the potential energy stored?",
            "tool": "physics",
            "operation": "energy",
            "vars": {"k": ["100", "200", "50", "500"], "x": ["0.1", "0.05", "0.2", "0.02"]},
            "difficulty": "easy",
        },
        {
            "template": "A mass-spring system has mass {m} kg and spring constant {k} N/m. What is the period of oscillation?",
            "tool": "physics",
            "operation": "shm",
            "vars": {"m": ["1", "2", "0.5", "4"], "k": ["100", "50", "200", "25"]},
            "difficulty": "medium",
        },
        {
            "template": "A simple pendulum has length {L} m. What is its period?",
            "tool": "physics",
            "operation": "shm",
            "vars": {"L": ["1", "0.5", "2", "0.25"]},
            "difficulty": "easy",
        },
    ],
    Domain.PHYSICS_EM_OPTICS: [
        {
            "template": "A circuit has a {V} V battery and a {R} ohm resistor. What is the current?",
            "tool": "physics",
            "operation": "electricity",
            "vars": {"V": ["12", "9", "24", "6"], "R": ["4", "3", "6", "2"]},
            "difficulty": "easy",
        },
        {
            "template": "Three resistors with values {r1}, {r2}, and {r3} ohms are connected in series. What is the total resistance?",
            "tool": "physics",
            "operation": "electricity",
            "vars": {"r1": ["10", "100", "5"], "r2": ["20", "200", "10"], "r3": ["30", "300", "15"]},
            "difficulty": "easy",
        },
        {
            "template": "Three resistors with values {r1}, {r2}, and {r3} ohms are connected in parallel. What is the total resistance?",
            "tool": "physics",
            "operation": "electricity",
            "vars": {"r1": ["10", "100", "6"], "r2": ["20", "200", "3"], "r3": ["30", "300", "2"]},
            "difficulty": "medium",
        },
        {
            "template": "Two charges of {q1} C and {q2} C are separated by {r} m. What is the electric force between them?",
            "tool": "physics",
            "operation": "electricity",
            "vars": {"q1": ["1e-6", "2e-6", "1e-9"], "q2": ["1e-6", "-2e-6", "1e-9"], "r": ["0.1", "0.5", "0.01"]},
            "difficulty": "medium",
        },
        {
            "template": "A converging lens has focal length {f} m. An object is placed {do} m from the lens. Where is the image formed?",
            "tool": "physics",
            "operation": "optics",
            "vars": {"f": ["0.1", "0.2", "0.15"], "do": ["0.3", "0.5", "0.2"]},
            "difficulty": "medium",
        },
        {
            "template": "Light travels from a medium with index {n1} to one with index {n2}. If the incident angle is {theta} degrees, what is the refracted angle?",
            "tool": "physics",
            "operation": "optics",
            "vars": {"n1": ["1.0", "1.5", "1.33"], "n2": ["1.5", "1.0", "1.0"], "theta": ["30", "45", "60"]},
            "difficulty": "medium",
        },
        {
            "template": "What is the wavelength of a photon with energy {E} eV?",
            "tool": "physics",
            "operation": "waves",
            "vars": {"E": ["2", "3", "1.5", "5"]},
            "difficulty": "medium",
        },
    ],
    Domain.CHEMISTRY_FUNDAMENTALS: [
        {
            "template": "What is the molecular weight of {formula}?",
            "tool": "chemistry",
            "operation": "molecular_weight",
            "vars": {"formula": ["H2O", "NaCl", "C6H12O6", "H2SO4", "Ca(OH)2", "Fe2O3", "C2H5OH", "NH4NO3"]},
            "difficulty": "easy",
        },
        {
            "template": "Balance the chemical equation: {equation}",
            "tool": "chemistry",
            "operation": "balance_equation",
            "vars": {"equation": ["H2 + O2 -> H2O", "Fe + O2 -> Fe2O3", "CH4 + O2 -> CO2 + H2O", "N2 + H2 -> NH3", "C3H8 + O2 -> CO2 + H2O"]},
            "difficulty": "medium",
        },
        {
            "template": "If {amount} grams of {reactant} react according to {equation}, how many grams of {product} are produced?",
            "tool": "chemistry",
            "operation": "stoichiometry",
            "vars": {"amount": ["10", "20", "5"], "reactant": ["H2", "CH4", "N2"], "equation": ["H2 + O2 -> H2O", "CH4 + O2 -> CO2 + H2O", "N2 + H2 -> NH3"], "product": ["H2O", "CO2", "NH3"]},
            "difficulty": "hard",
        },
        {
            "template": "What is the percent composition by mass of each element in {formula}?",
            "tool": "chemistry",
            "operation": "percent_composition",
            "vars": {"formula": ["H2O", "CO2", "NaCl", "C6H12O6", "NH3"]},
            "difficulty": "medium",
        },
        {
            "template": "Convert {amount} grams of {formula} to moles",
            "tool": "chemistry",
            "operation": "molar_conversion",
            "vars": {"amount": ["18", "44", "58.5", "180"], "formula": ["H2O", "CO2", "NaCl", "C6H12O6"]},
            "difficulty": "easy",
        },
    ],
    Domain.CHEMISTRY_SOLUTIONS: [
        {
            "template": "What is the pH of a {conc} M solution of a strong {acid_base}?",
            "tool": "chemistry",
            "operation": "ph",
            "vars": {"conc": ["0.1", "0.01", "0.001", "1.0"], "acid_base": ["acid", "base", "acid", "acid"]},
            "difficulty": "easy",
        },
        {
            "template": "A weak acid with Ka = {Ka} has concentration {conc} M. What is the pH?",
            "tool": "chemistry",
            "operation": "ph",
            "vars": {"Ka": ["1.8e-5", "6.8e-4", "1.0e-5"], "conc": ["0.1", "0.05", "0.2"]},
            "difficulty": "medium",
        },
        {
            "template": "You have {V1} mL of a {C1} M solution. What volume of water must be added to make a {C2} M solution?",
            "tool": "chemistry",
            "operation": "dilution",
            "vars": {"V1": ["50", "100", "25"], "C1": ["2.0", "1.0", "4.0"], "C2": ["0.5", "0.25", "1.0"]},
            "difficulty": "easy",
        },
        {
            "template": "How many grams of {formula} are needed to prepare {V} L of a {M} M solution?",
            "tool": "chemistry",
            "operation": "concentration",
            "vars": {"formula": ["NaCl", "NaOH", "H2SO4"], "V": ["1", "0.5", "2"], "M": ["0.1", "0.5", "0.25"]},
            "difficulty": "medium",
        },
        {
            "template": "In the reaction {equation}, if you have {amount1} g of {r1} and {amount2} g of {r2}, which is the limiting reagent?",
            "tool": "chemistry",
            "operation": "limiting_reagent",
            "vars": {"equation": ["H2 + O2 -> H2O", "N2 + H2 -> NH3"], "amount1": ["2", "28"], "amount2": ["16", "3"], "r1": ["H2", "N2"], "r2": ["O2", "H2"]},
            "difficulty": "hard",
        },
    ],
    Domain.LOGIC: [
        {
            "template": "Prove that if x > 0 and y > 0, then {conclusion}",
            "tool": "logic_solver",
            "operation": "prove",
            "vars": {"conclusion": ["x + y > 0", "x * y > 0", "x + y > x", "x + y > y"]},
            "difficulty": "medium",
        },
        {
            "template": "Is there an integer x such that {constraint1} and {constraint2}?",
            "tool": "logic_solver",
            "operation": "check_sat",
            "vars": {"constraint1": ["x > 5", "x**2 < 100", "x > 0"], "constraint2": ["x < 10", "x > 50", "x**2 == 16"]},
            "difficulty": "medium",
        },
        {
            "template": "Find values of x and y that satisfy: {c1}, {c2}, {c3}",
            "tool": "logic_solver",
            "operation": "check_sat",
            "vars": {"c1": ["x + y == 10", "x > 0", "2*x + y == 15"], "c2": ["x - y == 2", "y > 0", "x + 2*y == 10"], "c3": ["x > 0", "x + y < 20", "x > y"]},
            "difficulty": "hard",
        },
    ],
    Domain.NUMERICAL: [
        {
            "template": "Calculate the first {n} terms of the Fibonacci sequence",
            "tool": "execute_python",
            "operation": "execute",
            "vars": {"n": ["10", "15", "20", "8"]},
            "difficulty": "easy",
        },
        {
            "template": "Find all prime numbers less than {n}",
            "tool": "execute_python",
            "operation": "execute",
            "vars": {"n": ["50", "100", "30", "200"]},
            "difficulty": "easy",
        },
        {
            "template": "Calculate {n}! (factorial)",
            "tool": "execute_python",
            "operation": "execute",
            "vars": {"n": ["10", "15", "20", "12"]},
            "difficulty": "easy",
        },
        {
            "template": "Compute the sum 1 + 1/2 + 1/3 + ... + 1/{n}",
            "tool": "execute_python",
            "operation": "execute",
            "vars": {"n": ["100", "1000", "50"]},
            "difficulty": "easy",
        },
        {
            "template": "Find the GCD of {a} and {b}",
            "tool": "execute_python",
            "operation": "execute",
            "vars": {"a": ["48", "1071", "252"], "b": ["18", "462", "105"]},
            "difficulty": "easy",
        },
        {
            "template": "Compute e^x for x = {x} using the series expansion",
            "tool": "execute_python",
            "operation": "execute",
            "vars": {"x": ["1", "2", "0.5", "-1"]},
            "difficulty": "medium",
        },
        {
            "template": "Use Newton's method to find the square root of {n}",
            "tool": "execute_python",
            "operation": "execute",
            "vars": {"n": ["2", "10", "50", "7"]},
            "difficulty": "medium",
        },
    ],
    Domain.MULTI_TOOL: [
        {
            "template": "What is the derivative of {expr}, and what is its value at x = {val}?",
            "tools": ["symbolic_math", "symbolic_math"],
            "operations": ["differentiate", "evaluate"],
            "vars": {"expr": ["x**3 + 2*x", "sin(x)", "exp(x)"], "val": ["2", "0", "1"]},
            "difficulty": "medium",
        },
        {
            "template": "Solve the equation {expr} = 0 and verify by substituting the solution back",
            "tools": ["symbolic_math", "symbolic_math"],
            "operations": ["solve", "evaluate"],
            "vars": {"expr": ["x**2 - 4", "x**2 - 5*x + 6"]},
            "difficulty": "medium",
        },
        {
            "template": "Calculate the molecular weight of {formula} and determine how many moles are in {grams} grams",
            "tools": ["chemistry", "chemistry"],
            "operations": ["molecular_weight", "molar_conversion"],
            "vars": {"formula": ["NaCl", "H2SO4", "C6H12O6"], "grams": ["58.5", "98", "180"]},
            "difficulty": "medium",
        },
        {
            "template": "A projectile is launched at {v0} m/s at {theta} degrees. What is the kinetic energy at the highest point?",
            "tools": ["physics", "physics"],
            "operations": ["projectile", "energy"],
            "vars": {"v0": ["20", "30", "50"], "theta": ["45", "30", "60"]},
            "difficulty": "hard",
        },
    ],
    # ==========================================================================
    # V0.2 EPISTEMIC TOOLS - Error Class Elimination
    # ==========================================================================
    Domain.STATE_REASONING: [
        # State machine - "What happens next" problems
        {
            "template": "A traffic light starts at {state}. After {n} cycles (green->yellow->red->green), what state is it in?",
            "tool": "state_machine",
            "operation": "what_if",
            "vars": {"state": ["green", "red", "yellow"], "n": ["5", "7", "12", "20"]},
            "difficulty": "easy",
        },
        {
            "template": "A vending machine has {amount} items. If {sold} are sold and {restocked} are restocked, how many remain?",
            "tool": "state_machine",
            "operation": "resource_sim",
            "vars": {"amount": ["50", "100", "25"], "sold": ["15", "30", "10"], "restocked": ["20", "0", "5"]},
            "difficulty": "easy",
        },
        {
            "template": "A game character starts with {hp} HP, {gold} gold. They take {damage} damage, collect {loot} gold, and use a potion healing {heal} HP. What are their final stats?",
            "tool": "state_machine",
            "operation": "resource_sim",
            "vars": {"hp": ["100", "50", "80"], "gold": ["0", "50", "100"], "damage": ["30", "20", "45"], "loot": ["25", "10", "50"], "heal": ["20", "15", "30"]},
            "difficulty": "medium",
        },
        {
            "template": "A door starts {state}. The sequence of actions is: {actions}. What is the final state?",
            "tool": "state_machine",
            "operation": "what_if",
            "vars": {"state": ["closed", "open", "locked"], "actions": ["open,close,lock", "unlock,open,close", "open,close,open,close"]},
            "difficulty": "easy",
        },
        {
            "template": "A factory machine can be: idle, running, or broken. It starts {state}. After {actions}, what state is it in?",
            "tool": "state_machine",
            "operation": "what_if",
            "vars": {"state": ["idle", "running"], "actions": ["start,break,repair,start", "start,stop,start,break", "start,stop"]},
            "difficulty": "medium",
        },
    ],
    Domain.CONSTRAINT_REASONING: [
        # Constraint satisfaction - "Is this plan possible?"
        {
            "template": "Can you schedule {n} tasks where each takes {duration} hours in a {hours}-hour day, if they can't overlap?",
            "tool": "constraint",
            "operation": "check_schedule",
            "vars": {"n": ["3", "4", "5"], "duration": ["2", "3", "4"], "hours": ["8", "10", "12"]},
            "difficulty": "easy",
        },
        {
            "template": "Is there an integer between {low} and {high} that is divisible by both {a} and {b}?",
            "tool": "constraint",
            "operation": "check_feasibility",
            "vars": {"low": ["1", "10", "100"], "high": ["50", "100", "200"], "a": ["3", "7", "11"], "b": ["5", "9", "13"]},
            "difficulty": "medium",
        },
        {
            "template": "Three friends must sit in a row. {c1}. {c2}. Is this arrangement possible?",
            "tool": "constraint",
            "operation": "check_feasibility",
            "vars": {"c1": ["Alice must sit next to Bob", "Alice cannot sit next to Carol", "Bob must be on the left"], "c2": ["Carol must be in the middle", "Bob cannot be on the end", "Alice must be on the right"]},
            "difficulty": "medium",
        },
        {
            "template": "A project has tasks A, B, C. A takes {a}h, B takes {b}h, C takes {c}h. B requires A to finish first. C requires B. Can it be done in {deadline} hours?",
            "tool": "constraint",
            "operation": "check_schedule",
            "vars": {"a": ["2", "3", "4"], "b": ["3", "2", "5"], "c": ["2", "4", "3"], "deadline": ["6", "8", "10"]},
            "difficulty": "medium",
        },
        {
            "template": "Can 4 people share {items} items such that everyone gets at least {min} and no one gets more than {max}?",
            "tool": "constraint",
            "operation": "check_feasibility",
            "vars": {"items": ["12", "20", "8"], "min": ["2", "3", "1"], "max": ["5", "7", "3"]},
            "difficulty": "easy",
        },
    ],
    Domain.EXHAUSTIVE_REASONING: [
        # Enumeration - "Are there any?" / "How many?" problems
        {
            "template": "How many prime numbers are there between {low} and {high}?",
            "tool": "enumerate",
            "operation": "search",
            "vars": {"low": ["1", "50", "100"], "high": ["50", "100", "150"]},
            "difficulty": "easy",
        },
        {
            "template": "How many ways can you choose {k} items from a set of {n}?",
            "tool": "enumerate",
            "operation": "count",
            "vars": {"k": ["2", "3", "4"], "n": ["5", "6", "8"]},
            "difficulty": "easy",
        },
        {
            "template": "Are there any perfect squares between {low} and {high} that are also divisible by {d}?",
            "tool": "enumerate",
            "operation": "search",
            "vars": {"low": ["1", "100", "50"], "high": ["100", "500", "200"], "d": ["3", "7", "5"]},
            "difficulty": "medium",
        },
        {
            "template": "List all pairs (a,b) where a and b are single digits and a + b = {sum}",
            "tool": "enumerate",
            "operation": "search",
            "vars": {"sum": ["10", "12", "15", "8"]},
            "difficulty": "easy",
        },
        {
            "template": "How many 3-digit numbers have all distinct digits?",
            "tool": "enumerate",
            "operation": "count",
            "vars": {},
            "difficulty": "medium",
        },
        {
            "template": "Find all integers n where 1 <= n <= {max} and n^2 ends in {digit}",
            "tool": "enumerate",
            "operation": "search",
            "vars": {"max": ["50", "100", "30"], "digit": ["1", "4", "9", "6"]},
            "difficulty": "medium",
        },
    ],
    Domain.SENSITIVITY_ANALYSIS: [
        # Counterfactual - "How robust is this conclusion?"
        {
            "template": "A business sells widgets for ${price} each, costs ${cost} per unit, fixed costs ${fixed}. At {units} units sold, is it profitable? How sensitive is this to price changes?",
            "tool": "counterfactual",
            "operation": "sensitivity",
            "vars": {"price": ["100", "50", "200"], "cost": ["60", "30", "120"], "fixed": ["10000", "5000", "20000"], "units": ["300", "500", "200"]},
            "difficulty": "medium",
        },
        {
            "template": "An investment of ${initial} at {rate}% annual return for {years} years. Compare conservative ({low_rate}%) vs aggressive ({high_rate}%) strategies.",
            "tool": "counterfactual",
            "operation": "scenarios",
            "vars": {"initial": ["1000", "10000", "5000"], "rate": ["7", "5", "10"], "years": ["10", "20", "5"], "low_rate": ["4", "3", "5"], "high_rate": ["12", "15", "10"]},
            "difficulty": "medium",
        },
        {
            "template": "A decision: Option A gives ${a_val} with {a_prob}% probability. Option B gives ${b_val} with {b_prob}% probability. Which is better, and at what probability would you switch?",
            "tool": "counterfactual",
            "operation": "boundary",
            "vars": {"a_val": ["1000", "500", "2000"], "a_prob": ["80", "60", "90"], "b_val": ["2000", "1500", "5000"], "b_prob": ["40", "50", "30"]},
            "difficulty": "hard",
        },
        {
            "template": "If production costs increase by {pct}%, at what price point does a ${price} product become unprofitable (margin < {min_margin}%)?",
            "tool": "counterfactual",
            "operation": "boundary",
            "vars": {"pct": ["10", "20", "15"], "price": ["100", "50", "200"], "min_margin": ["10", "15", "20"]},
            "difficulty": "hard",
        },
        {
            "template": "A factory can produce {units} units. Best case: ${best} profit/unit. Worst case: ${worst} profit/unit. What's the range of outcomes?",
            "tool": "counterfactual",
            "operation": "scenarios",
            "vars": {"units": ["1000", "500", "2000"], "best": ["50", "100", "30"], "worst": ["10", "20", "5"]},
            "difficulty": "easy",
        },
    ],
    # ==========================================================================
    # FAILURE RECOGNITION - Model learns to identify unsolvable/error cases
    # ==========================================================================
    Domain.FAILURE_RECOGNITION: [
        # Insufficient information
        {
            "template": "What is the velocity of the object?",
            "tool": "none",
            "operation": "recognize_missing_info",
            "vars": {},
            "difficulty": "easy",
            "expected_behavior": "recognize_insufficient_info",
        },
        {
            "template": "Calculate the area of the triangle.",
            "tool": "none",
            "operation": "recognize_missing_info",
            "vars": {},
            "difficulty": "easy",
            "expected_behavior": "recognize_insufficient_info",
        },
        {
            "template": "A car travels for {t} hours. How far did it go?",
            "tool": "none",
            "operation": "recognize_missing_info",
            "vars": {"t": ["2", "5", "3"]},
            "difficulty": "easy",
            "expected_behavior": "recognize_insufficient_info",
        },
        # Mathematically undefined/impossible
        {
            "template": "Find the real solutions to x^2 + {a} = 0",
            "tool": "symbolic_math",
            "operation": "solve",
            "vars": {"a": ["1", "4", "9", "16"]},
            "difficulty": "medium",
            "expected_behavior": "recognize_no_real_solutions",
        },
        {
            "template": "Compute ln({x})",
            "tool": "symbolic_math",
            "operation": "evaluate",
            "vars": {"x": ["-1", "-5", "0"]},
            "difficulty": "medium",
            "expected_behavior": "recognize_undefined",
        },
        {
            "template": "Find the inverse of the matrix [[1,2],[2,4]]",
            "tool": "symbolic_math",
            "operation": "matrix",
            "vars": {},
            "difficulty": "medium",
            "expected_behavior": "recognize_singular_matrix",
        },
        # Tool errors - learn to handle gracefully
        {
            "template": "Solve the equation {expr} for z",
            "tool": "symbolic_math",
            "operation": "solve",
            "vars": {"expr": ["x + y = 5", "a*b = c", "sin(theta) = 0.5"]},
            "difficulty": "medium",
            "expected_behavior": "recognize_wrong_variable",
        },
        # Division by zero scenarios
        {
            "template": "Evaluate {expr} at x = {val}",
            "tool": "symbolic_math",
            "operation": "evaluate",
            "vars": {"expr": ["1/x", "1/(x-2)", "x/(x^2-1)"], "val": ["0", "2", "1"]},
            "difficulty": "medium",
            "expected_behavior": "recognize_undefined",
        },
        # Contradictory constraints
        {
            "template": "Find x where x > 10 and x < 5",
            "tool": "logic_solver",
            "operation": "check_sat",
            "vars": {},
            "difficulty": "easy",
            "expected_behavior": "recognize_contradiction",
        },
        # Beyond scope
        {
            "template": "Prove the Riemann hypothesis",
            "tool": "none",
            "operation": "recognize_beyond_scope",
            "vars": {},
            "difficulty": "hard",
            "expected_behavior": "recognize_beyond_scope",
        },
        {
            "template": "Write a poem about calculus",
            "tool": "none",
            "operation": "recognize_wrong_domain",
            "vars": {},
            "difficulty": "easy",
            "expected_behavior": "recognize_wrong_domain",
        },
    ],
    # ==========================================================================
    # CLARIFICATION - Model learns to ask for missing information
    # ==========================================================================
    Domain.CLARIFICATION: [
        # Ambiguous units
        {
            "template": "Convert {value} to meters",
            "tool": "physics",
            "operation": "unit_convert",
            "vars": {"value": ["100", "50", "25"]},
            "difficulty": "easy",
            "expected_behavior": "ask_source_units",
            "clarification_needed": "What units is the original value in?",
        },
        {
            "template": "What is the temperature in Fahrenheit?",
            "tool": "physics",
            "operation": "unit_convert",
            "vars": {},
            "difficulty": "easy",
            "expected_behavior": "ask_source_value",
            "clarification_needed": "What is the temperature you want to convert?",
        },
        # Ambiguous chemical notation
        {
            "template": "Balance the equation: {reactants} -> products",
            "tool": "chemistry",
            "operation": "balance_equation",
            "vars": {"reactants": ["H2 + O2", "Fe + O2", "C + O2"]},
            "difficulty": "medium",
            "expected_behavior": "ask_products",
            "clarification_needed": "What are the products of this reaction?",
        },
        {
            "template": "Calculate the pH",
            "tool": "chemistry",
            "operation": "ph",
            "vars": {},
            "difficulty": "easy",
            "expected_behavior": "ask_concentration",
            "clarification_needed": "What is the concentration and type of solution?",
        },
        # Ambiguous physics problems
        {
            "template": "Calculate the force",
            "tool": "physics",
            "operation": "kinematics",
            "vars": {},
            "difficulty": "easy",
            "expected_behavior": "ask_mass_acceleration",
            "clarification_needed": "What is the mass and acceleration?",
        },
        {
            "template": "A ball is thrown. Where does it land?",
            "tool": "physics",
            "operation": "projectile",
            "vars": {},
            "difficulty": "medium",
            "expected_behavior": "ask_initial_conditions",
            "clarification_needed": "What is the initial velocity and angle?",
        },
        {
            "template": "Find the current in the circuit",
            "tool": "physics",
            "operation": "electricity",
            "vars": {},
            "difficulty": "medium",
            "expected_behavior": "ask_circuit_parameters",
            "clarification_needed": "What is the voltage and resistance?",
        },
        # Ambiguous math problems
        {
            "template": "Solve for x",
            "tool": "symbolic_math",
            "operation": "solve",
            "vars": {},
            "difficulty": "easy",
            "expected_behavior": "ask_equation",
            "clarification_needed": "What equation should I solve?",
        },
        {
            "template": "Find the derivative",
            "tool": "symbolic_math",
            "operation": "differentiate",
            "vars": {},
            "difficulty": "easy",
            "expected_behavior": "ask_function",
            "clarification_needed": "What function should I differentiate?",
        },
        {
            "template": "Integrate from {a} to {b}",
            "tool": "symbolic_math",
            "operation": "integrate",
            "vars": {"a": ["0", "1"], "b": ["1", "10"]},
            "difficulty": "easy",
            "expected_behavior": "ask_integrand",
            "clarification_needed": "What function should I integrate?",
        },
        # Multiple interpretations
        {
            "template": "What is the solution to x^2 = {val}?",
            "tool": "symbolic_math",
            "operation": "solve",
            "vars": {"val": ["4", "9", "16"]},
            "difficulty": "medium",
            "expected_behavior": "ask_domain",
            "clarification_needed": "Do you want real solutions only, or complex solutions as well?",
        },
        {
            "template": "Simplify {expr}",
            "tool": "symbolic_math",
            "operation": "simplify",
            "vars": {"expr": ["x^2 - 1", "sin(x)^2 + cos(x)^2"]},
            "difficulty": "medium",
            "expected_behavior": "ask_simplification_goal",
            "clarification_needed": "What form would you like the answer in (factored, expanded, etc.)?",
        },
        # Context-dependent
        {
            "template": "How much energy is needed?",
            "tool": "physics",
            "operation": "energy",
            "vars": {},
            "difficulty": "medium",
            "expected_behavior": "ask_context",
            "clarification_needed": "For what process? (e.g., heating, lifting, accelerating)",
        },
        {
            "template": "What concentration should I use?",
            "tool": "chemistry",
            "operation": "dilution",
            "vars": {},
            "difficulty": "medium",
            "expected_behavior": "ask_purpose",
            "clarification_needed": "What is the target concentration and starting solution?",
        },
    ],
}

# Domain distribution for generation
DOMAIN_DISTRIBUTION = {
    # Core math/science domains
    Domain.CALCULUS: 2000,
    Domain.ALGEBRA: 1500,
    Domain.LINEAR_ALGEBRA: 500,
    Domain.PHYSICS_MECHANICS: 1500,
    Domain.PHYSICS_EM_OPTICS: 1000,
    Domain.CHEMISTRY_FUNDAMENTALS: 1000,
    Domain.CHEMISTRY_SOLUTIONS: 1000,
    Domain.LOGIC: 1000,
    Domain.NUMERICAL: 1500,
    Domain.MULTI_TOOL: 1000,
    # V0.2 Epistemic tools - Error class elimination
    Domain.STATE_REASONING: 800,        # "What happens next" - state machines
    Domain.CONSTRAINT_REASONING: 800,   # "Is this plan possible" - SAT/UNSAT
    Domain.EXHAUSTIVE_REASONING: 800,   # "Are there any" - enumeration
    Domain.SENSITIVITY_ANALYSIS: 800,   # "How robust" - counterfactuals
    # Meta-cognitive - critical for robust behavior
    Domain.FAILURE_RECOGNITION: 1000,   # Learn to recognize unsolvable problems
    Domain.CLARIFICATION: 1000,         # Learn to ask for missing info
}


# =============================================================================
# Generation System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are generating training data for a multi-model reasoning ensemble.

The ensemble has 4 specialists:
- ROUTER: Classifies intent and decides which specialist(s) to use
- LANGUAGE: Interprets prompts, synthesizes outputs, formats responses
- REASONING: Handles math, logic, chain-of-thought, tool orchestration
- VERIFIER: Checks answers, catches errors, validates reasoning

Given a problem, generate a detailed reasoning trace that:
1. Shows step-by-step thinking with specialist assignments
2. Calls tools when computation is needed (not for trivial steps)
3. Incorporates tool results into reasoning
4. Arrives at a final answer OR recognizes when this isn't possible

IMPORTANT: The model should learn:
- WHEN to call tools (non-trivial computation, symbolic math, precise calculations)
- When to reason directly (simple arithmetic, conceptual understanding)
- When to RECOGNIZE FAILURE (impossible problems, undefined operations, contradictions)
- When to ASK FOR CLARIFICATION (missing information, ambiguous questions)

Format your response as JSON with this EXACT structure:
{
  "question": "the original question exactly as given",
  "domain": "math|physics|chemistry|logic|code",
  "routing": {
    "primary_specialist": "reasoning|language",
    "needs_verification": true,
    "confidence": 0.95
  },
  "reasoning": [
    {
      "step": 1,
      "specialist": "language",
      "content": "First, I need to understand what the problem is asking..."
    },
    {
      "step": 2,
      "specialist": "reasoning",
      "content": "Let me compute this using symbolic math.",
      "tool_call": {
        "name": "symbolic_math",
        "args": {"operation": "differentiate", "expression": "x**3 + 2*x", "variable": "x"}
      }
    },
    {
      "step": 3,
      "specialist": "reasoning",
      "content": "The derivative is 3x^2 + 2. Now I can interpret this result...",
      "tool_result": "3*x**2 + 2"
    },
    {
      "step": 4,
      "specialist": "language",
      "content": "Therefore, the final answer is..."
    },
    {
      "step": 5,
      "specialist": "verifier",
      "content": "Let me verify: d/dx(x^3) = 3x^2, d/dx(2x) = 2. Sum = 3x^2 + 2. Correct!",
      "verification": {"valid": true, "confidence": 0.98}
    }
  ],
  "answer": "the final answer",
  "tools_used": ["symbolic_math"]
}

Available tools:

1. symbolic_math (SymPy)
   Operations: simplify, solve, differentiate, integrate, evaluate, limit, series, factor, expand, matrix
   Args: operation, expression, variable, values (for evaluate), point (for limit), order (for series), matrix/matrix2/values (for matrix ops)

2. execute_python
   For numerical computation, verification, or complex logic
   Args: code (Python code string)

3. logic_solver (Z3)
   Operations: check_sat, prove
   Args: operation, constraints (JSON array), variables (JSON object mapping name to type), conclusion (for prove)

4. chemistry
   Operations: molecular_weight, parse_formula, balance_equation, stoichiometry, ph, molar_conversion, dilution, concentration, percent_composition, limiting_reagent
   Args vary by operation (see examples)

5. physics
   Operations: unit_convert, kinematics, ideal_gas, waves, constant, energy, electricity, optics, projectile, shm
   Args vary by operation (see examples)

6. state_machine (V0.2 - Epistemic)
   Eliminates "what happens next" hallucinations by simulating discrete state transitions
   Operations: what_if (simulate actions), available_actions, resource_sim
   Args: initial_state, transitions, actions, initial_resources, resource_actions

7. constraint (V0.2 - Epistemic)
   Prevents proposing impossible plans by checking SAT/UNSAT
   Operations: check_feasibility, check_assignment, check_schedule
   Args: variables (with ranges), constraints, assignment, tasks

8. enumerate (V0.2 - Epistemic)
   Eliminates "seems unlikely" without verification via bounded exhaustive search
   Operations: search (find satisfying items), count (count search space)
   Args: space_type, low, high, condition_type, find_all, items, size

9. counterfactual (V0.2 - Epistemic)
   Sensitivity analysis - reports "robust to X, sensitive to Y, threshold at Z"
   Operations: sensitivity, scenarios, boundary, what_if
   Args: analysis_type, base_inputs, vary_params, scenarios, param, low, high, target

Guidelines:
- Use 3-6 reasoning steps for most problems
- Tool calls should have COMPLETE, VALID arguments
- tool_result appears in the NEXT step after tool_call
- Final answer should be clear and specific
- For multi-step problems, multiple tool calls are fine
- Include units in physics/chemistry answers

SPECIAL CASES:

For FAILURE RECOGNITION problems (impossible/undefined/missing info):
{
  "question": "Find the real solutions to x^2 + 4 = 0",
  "domain": "math",
  "routing": {"primary_specialist": "reasoning", "needs_verification": true, "confidence": 0.9},
  "reasoning": [
    {"step": 1, "specialist": "language", "content": "The user is asking to solve a quadratic equation for real solutions."},
    {"step": 2, "specialist": "reasoning", "content": "I need to solve x^2 + 4 = 0, which means x^2 = -4"},
    {"step": 3, "specialist": "reasoning", "content": "For real numbers, x^2 is always >= 0, so x^2 cannot equal -4"},
    {"step": 4, "specialist": "verifier", "content": "Confirmed: no real number squared gives a negative. Complex solutions exist (x = +/- 2i) but are not real.", "verification": {"valid": true, "confidence": 0.99}}
  ],
  "answer": "No real solutions exist. The equation x^2 + 4 = 0 has no real solutions because x^2 cannot be negative for real x.",
  "tools_used": [],
  "outcome": "failure_recognized"
}

For CLARIFICATION problems (ambiguous/incomplete questions):
{
  "question": "Convert 100 to meters",
  "domain": "physics",
  "routing": {"primary_specialist": "language", "needs_verification": false, "confidence": 0.7},
  "reasoning": [
    {"step": 1, "specialist": "language", "content": "The user wants to convert 100 to meters, but hasn't specified what unit 100 is in."},
    {"step": 2, "specialist": "language", "content": "I cannot perform this conversion without knowing the source units. I need to ask for clarification."}
  ],
  "answer": null,
  "clarification_request": "What units is 100 in? For example, is it 100 centimeters, 100 feet, 100 inches, etc.?",
  "tools_used": [],
  "outcome": "clarification_needed"
}
"""


# =============================================================================
# Anthropic Client and Generation
# =============================================================================

def create_anthropic_client():
    """Create Anthropic client."""
    try:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY environment variable not set")
            print("Set it with: export ANTHROPIC_API_KEY=your-key-here")
            sys.exit(1)
        return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        print("ERROR: anthropic package not installed")
        print("Install with: pip install anthropic")
        sys.exit(1)


def generate_problem(domain: Domain, template_idx: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
    """Generate a problem from a template."""
    templates = PROBLEM_TEMPLATES[domain]

    if template_idx is not None:
        template_info = templates[template_idx % len(templates)]
    else:
        template_info = random.choice(templates)

    template = template_info["template"]
    vars_dict = template_info["vars"]

    # Select random values for each variable
    selected_vars = {}
    for var_name, var_options in vars_dict.items():
        if isinstance(var_options, list):
            selected_vars[var_name] = random.choice(var_options)
        else:
            selected_vars[var_name] = var_options

    # Format the problem
    problem = template.format(**selected_vars)

    metadata = {
        "domain": domain.value,
        "tool": template_info.get("tool") or template_info.get("tools", []),
        "operation": template_info.get("operation") or template_info.get("operations", []),
        "difficulty": template_info.get("difficulty", "medium"),
        "vars": selected_vars,
    }

    return problem, metadata


def generate_trace(client, problem: str, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Generate a reasoning trace for a problem using Claude."""

    user_prompt = f"""Generate a training example for this problem:

Problem: {problem}
Domain: {metadata['domain']}
Expected tool(s): {metadata['tool']}
Difficulty: {metadata['difficulty']}

Output valid JSON with: question, domain, reasoning (array of steps), answer, tools_used"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}]
        )

        content = response.content[0].text

        # Extract JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]

        # Parse JSON
        example = json.loads(content.strip())

        # Add metadata
        example["_metadata"] = metadata
        example["_generation_model"] = "claude-sonnet-4-20250514"

        return example

    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"  Generation error: {e}")
        return None


# =============================================================================
# Tool Execution and Validation
# =============================================================================

def execute_tool_call(tool_call: Dict[str, Any]) -> Optional[str]:
    """Execute a tool call and return the result."""
    tool_name = tool_call.get("name")
    args = tool_call.get("args", {})

    try:
        if tool_name == "symbolic_math":
            from src.tools.math_engine import SymbolicSolver
            solver = SymbolicSolver()
            operation = args.get("operation")

            if operation == "differentiate":
                result = solver.differentiate(args["expression"], args.get("variable", "x"))
            elif operation == "integrate":
                result = solver.integrate(args["expression"], args.get("variable", "x"))
            elif operation == "solve":
                result = solver.solve_equation(args["expression"], args.get("variable", "x"))
            elif operation == "simplify":
                result = solver.simplify(args["expression"])
            elif operation == "evaluate":
                values = args.get("values", {})
                if isinstance(values, str):
                    values = json.loads(values)
                result = solver.evaluate(args["expression"], **values)
            elif operation == "limit":
                result = solver.limit(args["expression"], args.get("variable", "x"), args.get("point", "0"))
            elif operation == "series":
                result = solver.series(args["expression"], args.get("variable", "x"), args.get("point", "0"), args.get("order", 6))
            elif operation == "factor":
                result = solver.factor(args["expression"])
            elif operation == "expand":
                result = solver.expand(args["expression"])
            elif operation == "matrix":
                matrix = args.get("matrix")
                if isinstance(matrix, str):
                    matrix = json.loads(matrix)
                matrix2 = args.get("matrix2")
                if matrix2 and isinstance(matrix2, str):
                    matrix2 = json.loads(matrix2)
                mat_op = args.get("values", "determinant")
                result = solver.matrix_operations(mat_op, matrix, matrix2)
            else:
                return None

            if result.get("success"):
                # Return the main result value
                for key in ["derivative", "antiderivative", "solutions", "simplified", "result", "limit", "series", "factored", "expanded"]:
                    if key in result:
                        return str(result[key])
            return None

        elif tool_name == "chemistry":
            from src.tools.chemistry import ChemistryTool
            tool = ChemistryTool()
            result = tool.execute(args.get("operation"), **{k: v for k, v in args.items() if k != "operation"})
            if result.success:
                return json.dumps(result.data, default=str)
            return None

        elif tool_name == "physics":
            from src.tools.physics import PhysicsTool
            tool = PhysicsTool()
            result = tool.execute(args.get("operation"), **{k: v for k, v in args.items() if k != "operation"})
            if result.success:
                return json.dumps(result.data, default=str)
            return None

        elif tool_name == "execute_python":
            from src.tools.code_sandbox import CodeSandbox
            sandbox = CodeSandbox()
            result = sandbox.execute(args.get("code", ""))
            if result.get("success"):
                return result.get("output", "")
            return None

        elif tool_name == "logic_solver":
            from src.tools.math_engine import Z3Solver
            solver = Z3Solver()
            operation = args.get("operation")

            constraints = args.get("constraints", [])
            if isinstance(constraints, str):
                constraints = json.loads(constraints)

            variables = args.get("variables", {})
            if isinstance(variables, str):
                variables = json.loads(variables)

            if operation == "check_sat":
                result = solver.check_satisfiability(constraints, variables)
            elif operation == "prove":
                result = solver.prove(constraints, args.get("conclusion", ""), variables)
            else:
                return None

            if result.get("success"):
                return json.dumps(result, default=str)
            return None

    except Exception as e:
        print(f"  Tool execution error ({tool_name}): {e}")
        return None

    return None


def validate_example(example: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate an example and return (is_valid, list of issues)."""
    issues = []

    # Check outcome type - special handling for failure/clarification
    outcome = example.get("outcome", "success")

    if outcome == "clarification_needed":
        # Clarification examples need: question, reasoning, clarification_request
        required = ["question", "domain", "reasoning", "clarification_request"]
        for field in required:
            if field not in example:
                issues.append(f"Missing required field for clarification: {field}")
        # Answer can be null for clarification
        if issues:
            return False, issues
        # Basic structure check
        reasoning = example.get("reasoning", [])
        if not isinstance(reasoning, list) or len(reasoning) < 1:
            issues.append("Reasoning must have at least 1 step")
        return len(issues) == 0, issues

    if outcome == "failure_recognized":
        # Failure examples need: question, reasoning, answer explaining the failure
        required = ["question", "domain", "reasoning", "answer"]
        for field in required:
            if field not in example:
                issues.append(f"Missing required field for failure: {field}")
        if issues:
            return False, issues
        reasoning = example.get("reasoning", [])
        if not isinstance(reasoning, list) or len(reasoning) < 1:
            issues.append("Reasoning must have at least 1 step")
        # tools_used can be empty for failures
        return len(issues) == 0, issues

    # Standard success case - original validation
    required = ["question", "domain", "reasoning", "answer", "tools_used"]
    for field in required:
        if field not in example:
            issues.append(f"Missing required field: {field}")

    if issues:
        return False, issues

    # Check reasoning structure
    reasoning = example.get("reasoning", [])
    if not isinstance(reasoning, list) or len(reasoning) < 2:
        issues.append("Reasoning must be a list with at least 2 steps")

    # Check for tool calls and results
    has_tool_call = False
    tool_results_match = True

    for i, step in enumerate(reasoning):
        if not isinstance(step, dict):
            issues.append(f"Step {i+1} is not a dict")
            continue

        if "step" not in step or "content" not in step:
            issues.append(f"Step {i+1} missing 'step' or 'content'")

        if "tool_call" in step:
            has_tool_call = True
            tool_call = step["tool_call"]

            # Validate tool call structure
            if not isinstance(tool_call, dict):
                issues.append(f"Step {i+1} tool_call is not a dict")
            elif "name" not in tool_call or "args" not in tool_call:
                issues.append(f"Step {i+1} tool_call missing 'name' or 'args'")
            else:
                # Optionally execute and validate result
                expected_result = execute_tool_call(tool_call)

                # Check if next step has matching tool_result
                if i + 1 < len(reasoning):
                    next_step = reasoning[i + 1]
                    if "tool_result" not in next_step:
                        issues.append(f"Step {i+2} should have tool_result after tool_call")

    if not has_tool_call:
        issues.append("No tool calls found in reasoning")

    # Check tools_used matches actual tools called
    declared_tools = set(example.get("tools_used", []))
    actual_tools = set()
    for step in reasoning:
        if "tool_call" in step and isinstance(step["tool_call"], dict):
            actual_tools.add(step["tool_call"].get("name", ""))

    if declared_tools != actual_tools:
        issues.append(f"tools_used mismatch: declared={declared_tools}, actual={actual_tools}")

    return len(issues) == 0, issues


# =============================================================================
# Main Generation Loop
# =============================================================================

@dataclass
class GenerationStats:
    total_attempted: int = 0
    successful: int = 0
    failed: int = 0
    validation_failed: int = 0
    by_domain: Dict[str, int] = field(default_factory=dict)
    total_time_ms: float = 0.0

    def report(self) -> str:
        lines = [
            "+---------------------------------------------------------------+",
            "|                    Generation Summary                         |",
            "+---------------------------------------------------------------+",
            f"  Total attempted:     {self.total_attempted}",
            f"  Successful:          {self.successful}",
            f"  Failed (generation): {self.failed}",
            f"  Failed (validation): {self.validation_failed}",
            f"  Success rate:        {100 * self.successful / max(self.total_attempted, 1):.1f}%",
            f"  Avg time per example:{self.total_time_ms / max(self.total_attempted, 1):.0f}ms",
            "",
            "  By domain:",
        ]
        for domain, count in sorted(self.by_domain.items()):
            lines.append(f"    {domain:25s} {count:5d}")
        return "\n".join(lines)


def generate_batch(
    client,
    num_examples: int,
    output_path: str,
    domains: Optional[List[Domain]] = None,
    validate: bool = True,
    resume: bool = False,
) -> GenerationStats:
    """Generate a batch of examples."""

    stats = GenerationStats()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Determine which domains to use
    if domains is None:
        domains = list(Domain)

    # Calculate examples per domain proportionally
    total_weight = sum(DOMAIN_DISTRIBUTION.get(d, 1000) for d in domains)
    examples_per_domain = {
        d: max(1, int(num_examples * DOMAIN_DISTRIBUTION.get(d, 1000) / total_weight))
        for d in domains
    }

    # Adjust to hit exact target
    current_total = sum(examples_per_domain.values())
    if current_total < num_examples:
        # Add extras to largest domain
        largest = max(examples_per_domain.keys(), key=lambda d: examples_per_domain[d])
        examples_per_domain[largest] += num_examples - current_total

    print(f"Generating {num_examples} examples across {len(domains)} domains...")
    print(f"Distribution: {dict((d.value, c) for d, c in examples_per_domain.items())}")
    print()

    # Resume support
    existing_ids = set()
    mode = "a" if resume else "w"
    if resume and Path(output_path).exists():
        with open(output_path, "r") as f:
            for line in f:
                try:
                    ex = json.loads(line)
                    if "_id" in ex:
                        existing_ids.add(ex["_id"])
                except:
                    pass
        print(f"Resuming: found {len(existing_ids)} existing examples")

    with open(output_path, mode) as f:
        for domain in domains:
            domain_count = examples_per_domain[domain]
            generated = 0

            print(f"\n[{domain.value}] Generating {domain_count} examples...")

            for i in range(domain_count):
                # Generate unique ID
                problem, metadata = generate_problem(domain, i)
                example_id = hashlib.md5(f"{domain.value}:{problem}:{i}".encode()).hexdigest()[:12]

                if example_id in existing_ids:
                    continue

                stats.total_attempted += 1
                start_time = time.perf_counter()

                # Generate trace
                example = generate_trace(client, problem, metadata)

                elapsed_ms = (time.perf_counter() - start_time) * 1000
                stats.total_time_ms += elapsed_ms

                if example is None:
                    stats.failed += 1
                    print(f"  [{generated+1}/{domain_count}] FAILED: {problem[:50]}...")
                    continue

                # Validate
                if validate:
                    is_valid, issues = validate_example(example)
                    if not is_valid:
                        stats.validation_failed += 1
                        print(f"  [{generated+1}/{domain_count}] INVALID: {issues[0]}")
                        continue

                # Add ID and save
                example["_id"] = example_id
                f.write(json.dumps(example) + "\n")
                f.flush()

                stats.successful += 1
                stats.by_domain[domain.value] = stats.by_domain.get(domain.value, 0) + 1
                generated += 1

                print(f"  [{generated}/{domain_count}] OK {problem[:60]}...")

                # Rate limiting
                time.sleep(0.5)

    return stats


def dry_run_examples():
    """Print example problems without API calls."""
    print("\n=== DRY RUN: Example Problems ===\n")

    for domain in Domain:
        print(f"\n{domain.value.upper()}")
        print("-" * 40)

        templates = PROBLEM_TEMPLATES.get(domain, [])
        for i, template_info in enumerate(templates[:3]):
            problem, metadata = generate_problem(domain, i)
            print(f"  - {problem}")
            print(f"    Tool: {metadata['tool']}, Op: {metadata['operation']}")

    print("\n\n=== Tool Definitions ===\n")
    for tool_name, tool_def in TOOL_DEFINITIONS.items():
        print(f"{tool_name}")
        print(f"  Description: {tool_def['description']}")
        print(f"  Operations: {', '.join(tool_def['operations'])}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate tool-calling training data using Claude API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_tool_data.py --num-examples 100 --output data/tool_traces.jsonl
  python scripts/generate_tool_data.py --domain calculus --num-examples 500
  python scripts/generate_tool_data.py --dry-run
  python scripts/generate_tool_data.py --num-examples 1000 --resume
        """
    )
    parser.add_argument("--num-examples", type=int, default=100,
                        help="Number of examples to generate (default: 100)")
    parser.add_argument("--output", type=str, default="data/tool_traces.jsonl",
                        help="Output file path (default: data/tool_traces.jsonl)")
    parser.add_argument("--domain", type=str, choices=[d.value for d in Domain],
                        help="Generate only for specific domain")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print example problems without API calls")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip validation of generated examples")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")

    args = parser.parse_args()

    print("""
+---------------------------------------------------------------+
|           Svend Tool-Trace Generator               svend.ai   |
|                                                               |
|   Generating synthetic training data for tool-augmented       |
|   reasoning models.                                           |
+---------------------------------------------------------------+
    """)

    if args.dry_run:
        dry_run_examples()
        return

    client = create_anthropic_client()

    # Determine domains
    domains = None
    if args.domain:
        domains = [Domain(args.domain)]

    # Generate
    stats = generate_batch(
        client=client,
        num_examples=args.num_examples,
        output_path=args.output,
        domains=domains,
        validate=not args.no_validate,
        resume=args.resume,
    )

    print("\n")
    print(stats.report())
    print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()
