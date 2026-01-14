"""
Svend Tool System - V1 Toolkit

External tools that augment the reasoning model's capabilities.

Tier 1: Core Reasoning
- calculator: Verified arithmetic
- execute_python: Sandboxed code execution
- symbolic_math: SymPy - algebra, calculus, equations
- logic_solver: Z3 - SAT/SMT, constraints
- unit_converter: Dimensional analysis

Tier 2: Domain Computation
- chemistry: Molecular weight, stoichiometry, pH
- physics: Kinematics, thermodynamics, circuits
- numerical: NumPy/SciPy - linear algebra, optimization, ODEs
- statistics: Distributions, hypothesis tests, regression
- plotter: Data visualization
- combinatorics: Permutations, combinations, partitions, Stirling/Bell/Catalan
- graph: BFS, DFS, shortest path, MST, cycle detection, topological sort
- geometry: Points, lines, circles, triangles, polygons, intersections
- sequence: Pattern recognition, next term prediction, formula detection
- finance: TVM, annuities, NPV/IRR, bonds, depreciation

Tier 3: Epistemic Honesty (V0.2)
- state_machine: Discrete state transitions, resource flows, "what happens next"
- constraint: SAT/UNSAT feasibility, scheduling conflicts, plan validation
- enumerate: Bounded exhaustive search, counterexamples, verified verdicts
- counterfactual: Sensitivity analysis, what-if reasoning, robustness testing

Tier 4: Domain Specialists (V1.0)
- controls: Transfer functions, stability analysis, PID tuning
- systems_engineering: Requirements, interfaces, trade studies, FMEA
- manufacturing: Process capability (Cp/Cpk), OEE, tolerance stackup, SPC
- reliability: Weibull analysis, MTBF, fault trees, redundancy
- project_management: Critical path, earned value, resource leveling, PERT

Tier 5: Decision Sciences (V1.1)
- economics: Supply/demand, elasticity, PV/FV, marginal analysis, GDP
- logic_puzzles: Knights/knaves, zebra puzzles, syllogisms, river crossing
- game_theory: Nash equilibrium, minimax, dominated strategies, auctions

Tier 6: External Data
- wolfram: WolframAlpha verified facts
- pubchem: Chemical compound data
- web_search: Current information

Tier 7: Utilities
- latex_render: Mathematical typesetting
"""

# Core registry and execution
from .registry import ToolRegistry, Tool, ToolResult, ToolStatus, ToolParameter
from .executor import ToolExecutor

# Existing tools
from .code_sandbox import CodeSandbox
from .math_engine import MathEngine, SymbolicSolver, Z3Solver
from .orchestrator import ReasoningOrchestrator

# New V0 tools
from .calculator import Calculator
from .unit_converter import UnitConverter
from .numerical import NumericalEngine
from .statistics_tool import StatisticsEngine
from .plotter import Plotter
from .latex_render import LaTeXRenderer

# Extended V0.1 tools
from .graph_tools import Combinatorics, Graph
from .geometry import Point, Line, Circle, Triangle, Polygon
from .sequence_analyzer import SequenceAnalyzer
from .finance import FinanceCalculator

# V0.2 tools - epistemic honesty / error class elimination
from .state_machine import StateMachine, ResourceSimulator
from .constraint_solver import ConstraintSolver, SchedulingChecker
from .enumerator import Enumerator
from .counterfactual import CounterfactualAnalyzer

# V1.0 Domain Specialists
from .controls import ControlsAnalyzer
from .systems_engineering import SystemsEngineer
from .manufacturing import ManufacturingAnalyzer
from .reliability import ReliabilityAnalyzer
from .project_management import ProjectAnalyzer
from .specialist_plots import SpecialistPlotter

# V1.1 Decision Sciences
from .economics import EconomicsEngine, economics_tool
from .logic_puzzles import LogicPuzzleEngine, logic_tool
from .game_theory import GameTheoryEngine, game_theory_tool


def create_v0_registry() -> ToolRegistry:
    """
    Create a registry with all V0 tools.

    This is the standard toolkit for Svend training and inference.
    """
    registry = ToolRegistry()

    # Tier 1: Core Reasoning
    from .calculator import register_calculator_tools
    from .code_sandbox import register_code_tools
    from .math_engine import register_math_tools
    from .unit_converter import register_unit_converter_tools

    register_calculator_tools(registry)
    register_code_tools(registry)
    register_math_tools(registry)
    register_unit_converter_tools(registry)

    # Tier 2: Domain Specialists
    from .chemistry import ChemistryTool
    from .physics import PhysicsTool
    from .numerical import register_numerical_tools
    from .statistics_tool import register_statistics_tools
    from .plotter import register_plotter_tools
    from .graph_tools import register_graph_tools
    from .geometry import register_geometry_tools
    from .sequence_analyzer import register_sequence_tools
    from .finance import register_finance_tools

    # Chemistry and Physics use the specialist pattern
    # They need to be adapted or we register them differently
    register_numerical_tools(registry)
    register_statistics_tools(registry)
    register_plotter_tools(registry)

    # Extended V0.1 tools
    register_graph_tools(registry)
    register_geometry_tools(registry)
    register_sequence_tools(registry)
    register_finance_tools(registry)

    # V0.2 tools - epistemic honesty / error class elimination
    from .state_machine import register_state_machine_tools
    from .constraint_solver import register_constraint_tools
    from .enumerator import register_enumerator_tools

    register_state_machine_tools(registry)
    register_constraint_tools(registry)
    register_enumerator_tools(registry)

    from .counterfactual import register_counterfactual_tools
    register_counterfactual_tools(registry)

    # V1.0 Domain Specialists
    from .controls import register_controls_tools
    from .systems_engineering import register_systems_engineering_tools
    from .manufacturing import register_manufacturing_tools
    from .reliability import register_reliability_tools
    from .project_management import register_project_management_tools

    register_controls_tools(registry)
    register_systems_engineering_tools(registry)
    register_manufacturing_tools(registry)
    register_reliability_tools(registry)
    register_project_management_tools(registry)

    from .specialist_plots import register_specialist_plot_tools
    register_specialist_plot_tools(registry)

    # V1.1 Decision Sciences
    from .economics import register_economics_tools
    from .logic_puzzles import register_logic_tools
    from .game_theory import register_game_theory_tools

    register_economics_tools(registry)
    register_logic_tools(registry)
    register_game_theory_tools(registry)

    # Tier 6: External Data (optional - require API keys or network)
    # Only register if dependencies available
    try:
        from .external_apis import register_external_tools
        register_external_tools(registry)
    except ImportError:
        pass  # External tools optional

    # Tier 4: Utilities
    from .latex_render import register_latex_tools
    register_latex_tools(registry)

    return registry


def create_minimal_registry() -> ToolRegistry:
    """
    Create a minimal registry for testing or resource-constrained environments.

    Includes only: calculator, execute_python, symbolic_math
    """
    registry = ToolRegistry()

    from .calculator import register_calculator_tools
    from .code_sandbox import register_code_tools
    from .math_engine import register_math_tools

    register_calculator_tools(registry)
    register_code_tools(registry)
    register_math_tools(registry)

    return registry


__all__ = [
    # Registry and execution
    "ToolRegistry",
    "Tool",
    "ToolResult",
    "ToolStatus",
    "ToolParameter",
    "ToolExecutor",

    # Existing infrastructure
    "CodeSandbox",
    "MathEngine",
    "SymbolicSolver",
    "Z3Solver",
    "ReasoningOrchestrator",

    # New V0 tool engines
    "Calculator",
    "UnitConverter",
    "NumericalEngine",
    "StatisticsEngine",
    "Plotter",
    "LaTeXRenderer",

    # Extended V0.1 tool engines
    "Combinatorics",
    "Graph",
    "Point",
    "Line",
    "Circle",
    "Triangle",
    "Polygon",
    "SequenceAnalyzer",
    "FinanceCalculator",

    # V0.2 tool engines - epistemic honesty
    "StateMachine",
    "ResourceSimulator",
    "ConstraintSolver",
    "SchedulingChecker",
    "Enumerator",
    "CounterfactualAnalyzer",

    # V1.0 Domain Specialists
    "ControlsAnalyzer",
    "SystemsEngineer",
    "ManufacturingAnalyzer",
    "ReliabilityAnalyzer",
    "ProjectAnalyzer",
    "SpecialistPlotter",

    # V1.1 Decision Sciences
    "EconomicsEngine",
    "economics_tool",
    "LogicPuzzleEngine",
    "logic_tool",
    "GameTheoryEngine",
    "game_theory_tool",

    # Factory functions
    "create_v0_registry",
    "create_minimal_registry",
]
