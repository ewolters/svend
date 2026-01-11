"""
Mathematical Reasoning Tools

Provides symbolic mathematics and formal logic capabilities:
- SymPy: Symbolic algebra, calculus, equation solving
- Z3: SAT/SMT solving for formal logic and constraints
"""

from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
import json

from .registry import Tool, ToolParameter, ToolResult, ToolStatus, ToolRegistry


class SymbolicSolver:
    """
    Symbolic mathematics using SymPy.

    Capabilities:
    - Algebraic simplification
    - Equation solving
    - Calculus (derivatives, integrals)
    - Series expansion
    - Matrix operations
    """

    def __init__(self):
        self._sympy = None

    @property
    def sympy(self):
        """Lazy import of SymPy."""
        if self._sympy is None:
            try:
                import sympy
                self._sympy = sympy
            except ImportError:
                raise ImportError("SymPy is required for symbolic math. Install with: pip install sympy")
        return self._sympy

    def simplify(self, expression: str) -> Dict[str, Any]:
        """Simplify a mathematical expression."""
        try:
            expr = self.sympy.sympify(expression)
            simplified = self.sympy.simplify(expr)
            return {
                "success": True,
                "input": expression,
                "simplified": str(simplified),
                "latex": self.sympy.latex(simplified),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def solve_equation(
        self,
        equation: str,
        variable: str = "x",
    ) -> Dict[str, Any]:
        """Solve an equation for a variable."""
        try:
            sym = self.sympy
            var = sym.Symbol(variable)

            # Parse equation (handle both "expr = 0" and "lhs = rhs" formats)
            if "=" in equation and "==" not in equation:
                lhs, rhs = equation.split("=")
                expr = sym.sympify(lhs) - sym.sympify(rhs)
            else:
                expr = sym.sympify(equation.replace("==", "-"))

            solutions = sym.solve(expr, var)

            return {
                "success": True,
                "equation": equation,
                "variable": variable,
                "solutions": [str(s) for s in solutions],
                "num_solutions": len(solutions),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def solve_system(
        self,
        equations: List[str],
        variables: List[str],
    ) -> Dict[str, Any]:
        """Solve a system of equations."""
        try:
            sym = self.sympy
            vars_sym = [sym.Symbol(v) for v in variables]

            # Parse equations
            eqs = []
            for eq in equations:
                if "=" in eq and "==" not in eq:
                    lhs, rhs = eq.split("=")
                    eqs.append(sym.Eq(sym.sympify(lhs), sym.sympify(rhs)))
                else:
                    eqs.append(sym.sympify(eq.replace("==", "-")))

            solutions = sym.solve(eqs, vars_sym)

            # Format solutions
            if isinstance(solutions, dict):
                formatted = {str(k): str(v) for k, v in solutions.items()}
            elif isinstance(solutions, list):
                formatted = [
                    {str(variables[i]): str(sol[i]) for i in range(len(variables))}
                    for sol in solutions
                ]
            else:
                formatted = str(solutions)

            return {
                "success": True,
                "equations": equations,
                "variables": variables,
                "solutions": formatted,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def differentiate(
        self,
        expression: str,
        variable: str = "x",
        order: int = 1,
    ) -> Dict[str, Any]:
        """Compute derivative of an expression."""
        try:
            sym = self.sympy
            var = sym.Symbol(variable)
            expr = sym.sympify(expression)

            derivative = sym.diff(expr, var, order)

            return {
                "success": True,
                "input": expression,
                "variable": variable,
                "order": order,
                "derivative": str(derivative),
                "latex": sym.latex(derivative),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def integrate(
        self,
        expression: str,
        variable: str = "x",
        lower: Optional[str] = None,
        upper: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute integral (definite or indefinite)."""
        try:
            sym = self.sympy
            var = sym.Symbol(variable)
            expr = sym.sympify(expression)

            if lower is not None and upper is not None:
                # Definite integral
                result = sym.integrate(expr, (var, sym.sympify(lower), sym.sympify(upper)))
                return {
                    "success": True,
                    "input": expression,
                    "variable": variable,
                    "bounds": [lower, upper],
                    "definite": True,
                    "result": str(result),
                    "numeric": float(result) if result.is_number else None,
                }
            else:
                # Indefinite integral
                result = sym.integrate(expr, var)
                return {
                    "success": True,
                    "input": expression,
                    "variable": variable,
                    "definite": False,
                    "antiderivative": str(result) + " + C",
                    "latex": sym.latex(result) + " + C",
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def evaluate(self, expression: str, **values) -> Dict[str, Any]:
        """Evaluate expression with given values."""
        try:
            sym = self.sympy
            expr = sym.sympify(expression)

            # Substitute values
            for var, val in values.items():
                expr = expr.subs(sym.Symbol(var), val)

            # Try to get numeric result
            result = expr.evalf() if hasattr(expr, 'evalf') else expr

            return {
                "success": True,
                "input": expression,
                "values": values,
                "result": str(result),
                "numeric": float(result) if result.is_number else None,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def limit(
        self,
        expression: str,
        variable: str = "x",
        point: str = "oo",
        direction: str = "+",
    ) -> Dict[str, Any]:
        """Compute the limit of an expression."""
        try:
            sym = self.sympy
            var = sym.Symbol(variable)
            expr = sym.sympify(expression)

            # Parse point (handle infinity)
            if point in ["oo", "inf", "infinity"]:
                pt = sym.oo
            elif point in ["-oo", "-inf", "-infinity"]:
                pt = -sym.oo
            else:
                pt = sym.sympify(point)

            # Compute limit
            if direction == "+" or pt in [sym.oo, -sym.oo]:
                result = sym.limit(expr, var, pt, "+")
            elif direction == "-":
                result = sym.limit(expr, var, pt, "-")
            else:
                result = sym.limit(expr, var, pt)

            return {
                "success": True,
                "input": expression,
                "variable": variable,
                "point": str(pt),
                "direction": direction,
                "limit": str(result),
                "latex": sym.latex(result),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def series(
        self,
        expression: str,
        variable: str = "x",
        point: str = "0",
        order: int = 6,
    ) -> Dict[str, Any]:
        """Compute Taylor/Maclaurin series expansion."""
        try:
            sym = self.sympy
            var = sym.Symbol(variable)
            expr = sym.sympify(expression)
            pt = sym.sympify(point)

            # Compute series
            series_result = sym.series(expr, var, pt, order)

            # Get coefficients
            coeffs = {}
            for i in range(order):
                coeff = series_result.coeff(var - pt, i)
                if coeff != 0:
                    coeffs[f"a_{i}"] = str(coeff)

            return {
                "success": True,
                "input": expression,
                "variable": variable,
                "point": point,
                "order": order,
                "series": str(series_result),
                "coefficients": coeffs,
                "latex": sym.latex(series_result),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def factor(self, expression: str) -> Dict[str, Any]:
        """Factor a polynomial expression."""
        try:
            sym = self.sympy
            expr = sym.sympify(expression)
            factored = sym.factor(expr)

            return {
                "success": True,
                "input": expression,
                "factored": str(factored),
                "latex": sym.latex(factored),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def expand(self, expression: str) -> Dict[str, Any]:
        """Expand a polynomial expression."""
        try:
            sym = self.sympy
            expr = sym.sympify(expression)
            expanded = sym.expand(expr)

            return {
                "success": True,
                "input": expression,
                "expanded": str(expanded),
                "latex": sym.latex(expanded),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def matrix_operations(
        self,
        operation: str,
        matrix: list,
        matrix2: list = None,
    ) -> Dict[str, Any]:
        """Perform matrix operations."""
        try:
            sym = self.sympy
            M1 = sym.Matrix(matrix)

            if operation == "determinant":
                result = M1.det()
                return {
                    "success": True,
                    "operation": "determinant",
                    "result": str(result),
                }

            elif operation == "inverse":
                result = M1.inv()
                return {
                    "success": True,
                    "operation": "inverse",
                    "result": result.tolist(),
                }

            elif operation == "eigenvalues":
                eigenvals = M1.eigenvals()
                return {
                    "success": True,
                    "operation": "eigenvalues",
                    "eigenvalues": {str(k): v for k, v in eigenvals.items()},
                }

            elif operation == "transpose":
                result = M1.T
                return {
                    "success": True,
                    "operation": "transpose",
                    "result": result.tolist(),
                }

            elif operation == "multiply" and matrix2 is not None:
                M2 = sym.Matrix(matrix2)
                result = M1 * M2
                return {
                    "success": True,
                    "operation": "multiply",
                    "result": result.tolist(),
                }

            elif operation == "rref":
                result, pivots = M1.rref()
                return {
                    "success": True,
                    "operation": "rref",
                    "result": result.tolist(),
                    "pivot_columns": list(pivots),
                }

            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}

        except Exception as e:
            return {"success": False, "error": str(e)}


class Z3Solver:
    """
    SMT/SAT Solver using Z3.

    Capabilities:
    - Boolean satisfiability (SAT)
    - Linear arithmetic constraints
    - Integer constraints
    - Logical implication checking
    - Model finding
    """

    def __init__(self):
        self._z3 = None

    @property
    def z3(self):
        """Lazy import of Z3."""
        if self._z3 is None:
            try:
                import z3
                self._z3 = z3
            except ImportError:
                raise ImportError("Z3 is required for SMT solving. Install with: pip install z3-solver")
        return self._z3

    def check_satisfiability(
        self,
        constraints: List[str],
        variables: Dict[str, str],  # name -> type ("bool", "int", "real")
    ) -> Dict[str, Any]:
        """
        Check if a set of constraints is satisfiable.

        Args:
            constraints: List of constraint strings (e.g., "x > 5", "x + y == 10")
            variables: Dict mapping variable names to types

        Returns:
            Dict with satisfiability result and model if SAT
        """
        try:
            z3 = self.z3

            # Create variables
            vars_z3 = {}
            for name, vtype in variables.items():
                if vtype == "bool":
                    vars_z3[name] = z3.Bool(name)
                elif vtype == "int":
                    vars_z3[name] = z3.Int(name)
                elif vtype == "real":
                    vars_z3[name] = z3.Real(name)
                else:
                    raise ValueError(f"Unknown variable type: {vtype}")

            # Create solver
            solver = z3.Solver()

            # Parse and add constraints
            for constraint in constraints:
                # Simple parsing for common patterns
                expr = self._parse_constraint(constraint, vars_z3)
                solver.add(expr)

            # Check satisfiability
            result = solver.check()

            if result == z3.sat:
                model = solver.model()
                solution = {}
                for name, var in vars_z3.items():
                    val = model.eval(var)
                    solution[name] = str(val)

                return {
                    "success": True,
                    "satisfiable": True,
                    "model": solution,
                }
            elif result == z3.unsat:
                return {
                    "success": True,
                    "satisfiable": False,
                    "reason": "No solution exists that satisfies all constraints",
                }
            else:
                return {
                    "success": True,
                    "satisfiable": None,
                    "reason": "Unknown (solver timeout or complexity)",
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _parse_constraint(self, constraint: str, vars_z3: Dict) -> Any:
        """Parse a constraint string into Z3 expression."""
        z3 = self.z3

        # Replace variable names with Z3 variables
        expr_str = constraint

        # Handle comparisons
        for op, z3_op in [("==", "=="), ("!=", "!="), (">=", ">="),
                          ("<=", "<="), (">", ">"), ("<", "<")]:
            if op in expr_str:
                parts = expr_str.split(op)
                if len(parts) == 2:
                    lhs = self._parse_expr(parts[0].strip(), vars_z3)
                    rhs = self._parse_expr(parts[1].strip(), vars_z3)

                    if op == "==":
                        return lhs == rhs
                    elif op == "!=":
                        return lhs != rhs
                    elif op == ">=":
                        return lhs >= rhs
                    elif op == "<=":
                        return lhs <= rhs
                    elif op == ">":
                        return lhs > rhs
                    elif op == "<":
                        return lhs < rhs

        # Handle logical operators
        if " and " in expr_str.lower():
            parts = expr_str.lower().split(" and ")
            return z3.And([self._parse_constraint(p.strip(), vars_z3) for p in parts])

        if " or " in expr_str.lower():
            parts = expr_str.lower().split(" or ")
            return z3.Or([self._parse_constraint(p.strip(), vars_z3) for p in parts])

        if expr_str.lower().startswith("not "):
            return z3.Not(self._parse_constraint(expr_str[4:].strip(), vars_z3))

        # Single variable or boolean
        return self._parse_expr(expr_str, vars_z3)

    def _parse_expr(self, expr_str: str, vars_z3: Dict) -> Any:
        """Parse an arithmetic expression."""
        z3 = self.z3

        expr_str = expr_str.strip()

        # Check if it's a variable
        if expr_str in vars_z3:
            return vars_z3[expr_str]

        # Check if it's a number
        try:
            if "." in expr_str:
                return z3.RealVal(expr_str)
            else:
                return z3.IntVal(int(expr_str))
        except ValueError:
            pass

        # Handle arithmetic operations
        for op in ["+", "-", "*", "/"]:
            if op in expr_str:
                # Find the operator (handling precedence would need more complex parsing)
                parts = expr_str.split(op, 1)
                if len(parts) == 2:
                    lhs = self._parse_expr(parts[0].strip(), vars_z3)
                    rhs = self._parse_expr(parts[1].strip(), vars_z3)

                    if op == "+":
                        return lhs + rhs
                    elif op == "-":
                        return lhs - rhs
                    elif op == "*":
                        return lhs * rhs
                    elif op == "/":
                        return lhs / rhs

        raise ValueError(f"Cannot parse expression: {expr_str}")

    def prove(
        self,
        premise: List[str],
        conclusion: str,
        variables: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Attempt to prove a logical statement.

        Proves by showing that premise AND NOT(conclusion) is unsatisfiable.
        """
        try:
            # Add negation of conclusion to premises
            all_constraints = premise + [f"not ({conclusion})"]

            result = self.check_satisfiability(all_constraints, variables)

            if not result["success"]:
                return result

            if result["satisfiable"] == False:
                return {
                    "success": True,
                    "proven": True,
                    "explanation": "The conclusion follows necessarily from the premises",
                }
            elif result["satisfiable"] == True:
                return {
                    "success": True,
                    "proven": False,
                    "counterexample": result.get("model"),
                    "explanation": "Found a counterexample where premises are true but conclusion is false",
                }
            else:
                return {
                    "success": True,
                    "proven": None,
                    "explanation": "Could not determine provability",
                }

        except Exception as e:
            return {"success": False, "error": str(e)}


class MathEngine:
    """Combined mathematical reasoning engine."""

    def __init__(self):
        self.symbolic = SymbolicSolver()
        self.logic = Z3Solver()

    def solve(self, problem_type: str, **kwargs) -> Dict[str, Any]:
        """Route to appropriate solver based on problem type."""
        if problem_type == "simplify":
            return self.symbolic.simplify(kwargs.get("expression", ""))
        elif problem_type == "solve_equation":
            return self.symbolic.solve_equation(
                kwargs.get("equation", ""),
                kwargs.get("variable", "x"),
            )
        elif problem_type == "solve_system":
            return self.symbolic.solve_system(
                kwargs.get("equations", []),
                kwargs.get("variables", []),
            )
        elif problem_type == "differentiate":
            return self.symbolic.differentiate(
                kwargs.get("expression", ""),
                kwargs.get("variable", "x"),
                kwargs.get("order", 1),
            )
        elif problem_type == "integrate":
            return self.symbolic.integrate(
                kwargs.get("expression", ""),
                kwargs.get("variable", "x"),
                kwargs.get("lower"),
                kwargs.get("upper"),
            )
        elif problem_type == "evaluate":
            expr = kwargs.pop("expression", "")
            return self.symbolic.evaluate(expr, **kwargs)
        elif problem_type == "check_sat":
            return self.logic.check_satisfiability(
                kwargs.get("constraints", []),
                kwargs.get("variables", {}),
            )
        elif problem_type == "prove":
            return self.logic.prove(
                kwargs.get("premises", []),
                kwargs.get("conclusion", ""),
                kwargs.get("variables", {}),
            )
        else:
            return {"success": False, "error": f"Unknown problem type: {problem_type}"}


# Tool implementations

def symbolic_math_tool(
    operation: str,
    expression: str = "",
    variable: Optional[str] = None,
    values: Optional[str] = None,
    point: Optional[str] = None,
    order: Optional[int] = None,
    matrix: Optional[str] = None,
    matrix2: Optional[str] = None,
) -> ToolResult:
    """Tool function for symbolic math."""
    solver = SymbolicSolver()

    try:
        if operation == "simplify":
            result = solver.simplify(expression)
        elif operation == "solve":
            result = solver.solve_equation(expression, variable or "x")
        elif operation == "differentiate":
            result = solver.differentiate(expression, variable or "x")
        elif operation == "integrate":
            result = solver.integrate(expression, variable or "x")
        elif operation == "evaluate":
            vals = json.loads(values) if values else {}
            result = solver.evaluate(expression, **vals)
        elif operation == "limit":
            result = solver.limit(
                expression,
                variable or "x",
                point or "oo",
            )
        elif operation == "series":
            result = solver.series(
                expression,
                variable or "x",
                point or "0",
                order or 6,
            )
        elif operation == "factor":
            result = solver.factor(expression)
        elif operation == "expand":
            result = solver.expand(expression)
        elif operation == "matrix":
            mat = json.loads(matrix) if matrix else []
            mat2 = json.loads(matrix2) if matrix2 else None
            mat_op = values or "determinant"  # reuse values param for matrix operation
            result = solver.matrix_operations(mat_op, mat, mat2)
        else:
            return ToolResult(
                status=ToolStatus.INVALID_INPUT,
                output=None,
                error=f"Unknown operation: {operation}",
            )

        if result.get("success"):
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=result,
            )
        else:
            return ToolResult(
                status=ToolStatus.ERROR,
                output=None,
                error=result.get("error", "Unknown error"),
            )

    except Exception as e:
        return ToolResult(
            status=ToolStatus.ERROR,
            output=None,
            error=str(e),
        )


def logic_solver_tool(
    operation: str,
    constraints: str,
    variables: str,
    conclusion: Optional[str] = None,
) -> ToolResult:
    """Tool function for logic solving."""
    solver = Z3Solver()

    try:
        constraints_list = json.loads(constraints)
        variables_dict = json.loads(variables)

        if operation == "check_sat":
            result = solver.check_satisfiability(constraints_list, variables_dict)
        elif operation == "prove":
            if not conclusion:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output=None,
                    error="Conclusion required for prove operation",
                )
            result = solver.prove(constraints_list, conclusion, variables_dict)
        else:
            return ToolResult(
                status=ToolStatus.INVALID_INPUT,
                output=None,
                error=f"Unknown operation: {operation}",
            )

        if result.get("success"):
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=result,
            )
        else:
            return ToolResult(
                status=ToolStatus.ERROR,
                output=None,
                error=result.get("error", "Unknown error"),
            )

    except json.JSONDecodeError as e:
        return ToolResult(
            status=ToolStatus.INVALID_INPUT,
            output=None,
            error=f"Invalid JSON: {e}",
        )
    except Exception as e:
        return ToolResult(
            status=ToolStatus.ERROR,
            output=None,
            error=str(e),
        )


def create_symbolic_math_tool() -> Tool:
    """Create the symbolic math tool."""
    return Tool(
        name="symbolic_math",
        description="Perform symbolic mathematics: simplification, equation solving, calculus, limits, series, factoring, matrix operations. Use this to verify mathematical reasoning or compute exact symbolic results.",
        parameters=[
            ToolParameter(
                name="operation",
                description="Operation to perform",
                type="string",
                required=True,
                enum=["simplify", "solve", "differentiate", "integrate", "evaluate", "limit", "series", "factor", "expand", "matrix"],
            ),
            ToolParameter(
                name="expression",
                description="Mathematical expression (e.g., 'x**2 + 2*x + 1', 'sin(x)')",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="variable",
                description="Variable to solve for / differentiate with respect to (default: x)",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="values",
                description="JSON object of variable values for evaluation, or matrix operation name for matrix op",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="point",
                description="Point for limit or series expansion (e.g., '0', 'oo' for infinity)",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="order",
                description="Order of series expansion (default: 6)",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="matrix",
                description="JSON array for matrix operations (e.g., '[[1,2],[3,4]]')",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="matrix2",
                description="Second matrix for matrix multiplication",
                type="string",
                required=False,
            ),
        ],
        execute_fn=symbolic_math_tool,
        timeout_ms=10000,
    )


def create_logic_solver_tool() -> Tool:
    """Create the logic solver tool."""
    return Tool(
        name="logic_solver",
        description="Check logical satisfiability or prove logical statements. Use this to verify logical reasoning, find counterexamples, or prove theorems.",
        parameters=[
            ToolParameter(
                name="operation",
                description="Operation to perform",
                type="string",
                required=True,
                enum=["check_sat", "prove"],
            ),
            ToolParameter(
                name="constraints",
                description="JSON array of constraint strings (e.g., '[\"x > 0\", \"x < 10\"]')",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="variables",
                description="JSON object mapping variable names to types (e.g., '{\"x\": \"int\", \"y\": \"bool\"}')",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="conclusion",
                description="Conclusion to prove (required for 'prove' operation)",
                type="string",
                required=False,
            ),
        ],
        execute_fn=logic_solver_tool,
        timeout_ms=30000,
    )


def register_math_tools(registry: ToolRegistry) -> None:
    """Register math tools with the registry."""
    registry.register(create_symbolic_math_tool())
    registry.register(create_logic_solver_tool())
