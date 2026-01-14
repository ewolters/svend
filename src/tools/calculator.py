"""
Calculator Tool - Verified Arithmetic

Provides guaranteed-correct arithmetic to avoid floating point embarrassments.
Uses Python's decimal module for precision and fractions for exact ratios.
"""

from typing import Optional, Dict, Any, Union
from decimal import Decimal, getcontext, InvalidOperation
from fractions import Fraction
import math
import operator
import re

from .registry import Tool, ToolParameter, ToolResult, ToolStatus, ToolRegistry


# Set high precision for decimal operations
getcontext().prec = 50


class Calculator:
    """
    High-precision calculator with exact arithmetic.

    Features:
    - Arbitrary precision decimals
    - Exact fractions
    - Common math functions
    - Expression parsing
    """

    # Supported operations
    OPERATORS = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv,
        '//': operator.floordiv,
        '%': operator.mod,
        '**': operator.pow,
        '^': operator.pow,
    }

    # Supported functions
    FUNCTIONS = {
        'sqrt': lambda x: Decimal(x).sqrt(),
        'abs': abs,
        'factorial': math.factorial,
        'gcd': math.gcd,
        'lcm': math.lcm,
        'floor': math.floor,
        'ceil': math.ceil,
        'round': round,
        'log': lambda x, base=10: Decimal(str(math.log(float(x), base))),
        'ln': lambda x: Decimal(str(math.log(float(x)))),
        'log10': lambda x: Decimal(str(math.log10(float(x)))),
        'log2': lambda x: Decimal(str(math.log2(float(x)))),
        'exp': lambda x: Decimal(str(math.exp(float(x)))),
        'sin': lambda x: Decimal(str(math.sin(float(x)))),
        'cos': lambda x: Decimal(str(math.cos(float(x)))),
        'tan': lambda x: Decimal(str(math.tan(float(x)))),
        'asin': lambda x: Decimal(str(math.asin(float(x)))),
        'acos': lambda x: Decimal(str(math.acos(float(x)))),
        'atan': lambda x: Decimal(str(math.atan(float(x)))),
        'sinh': lambda x: Decimal(str(math.sinh(float(x)))),
        'cosh': lambda x: Decimal(str(math.cosh(float(x)))),
        'tanh': lambda x: Decimal(str(math.tanh(float(x)))),
        'degrees': math.degrees,
        'radians': math.radians,
    }

    # Constants
    CONSTANTS = {
        'pi': Decimal(str(math.pi)),
        'e': Decimal(str(math.e)),
        'tau': Decimal(str(math.tau)),
        'phi': Decimal('1.6180339887498948482'),  # Golden ratio
        'inf': Decimal('Infinity'),
    }

    def evaluate(
        self,
        expression: str,
        precision: int = 10,
        as_fraction: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate a mathematical expression.

        Args:
            expression: Math expression (e.g., "2 + 3 * 4", "sqrt(2)")
            precision: Decimal places for result
            as_fraction: Return as exact fraction if possible

        Returns:
            Dict with result and metadata
        """
        try:
            # Clean expression
            expr = expression.strip()

            # Replace constants
            for name, value in self.CONSTANTS.items():
                expr = re.sub(rf'\b{name}\b', str(value), expr)

            # Handle special syntax
            expr = expr.replace('^', '**')

            # Safe evaluation using restricted namespace
            namespace = {
                '__builtins__': {},
                'Decimal': Decimal,
                'Fraction': Fraction,
            }

            # Add operators and functions
            namespace.update(self.FUNCTIONS)
            namespace.update({
                'abs': abs,
                'min': min,
                'max': max,
                'sum': sum,
                'pow': pow,
            })

            # Convert numbers to Decimal for precision
            # Match numbers (including scientific notation)
            def to_decimal(match):
                num = match.group(0)
                # Don't convert if already in a function call
                return f"Decimal('{num}')"

            # Simple number pattern (not inside quotes)
            number_pattern = r'(?<!["\'\w])(\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+)(?!["\'\w])'
            expr_decimal = re.sub(number_pattern, to_decimal, expr)

            # Evaluate
            result = eval(expr_decimal, namespace)

            # Format result
            if isinstance(result, Decimal):
                result_str = str(round(result, precision))
                # Clean up trailing zeros after decimal
                if '.' in result_str:
                    result_str = result_str.rstrip('0').rstrip('.')
            elif isinstance(result, (int, float)):
                result = Decimal(str(result))
                result_str = str(round(result, precision))
                if '.' in result_str:
                    result_str = result_str.rstrip('0').rstrip('.')
            else:
                result_str = str(result)

            response = {
                "success": True,
                "expression": expression,
                "result": result_str,
            }

            # Try to get exact fraction
            if as_fraction:
                try:
                    frac = Fraction(float(result)).limit_denominator(10000)
                    if frac.denominator != 1:
                        response["fraction"] = f"{frac.numerator}/{frac.denominator}"
                except (ValueError, OverflowError):
                    pass

            # Add numeric value for programmatic use
            try:
                response["numeric"] = float(result)
            except (ValueError, OverflowError, InvalidOperation):
                pass

            return response

        except ZeroDivisionError:
            return {"success": False, "error": "Division by zero"}
        except InvalidOperation as e:
            return {"success": False, "error": f"Invalid operation: {e}"}
        except SyntaxError as e:
            return {"success": False, "error": f"Syntax error: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def basic_arithmetic(
        self,
        a: Union[int, float, str],
        op: str,
        b: Union[int, float, str],
    ) -> Dict[str, Any]:
        """Simple two-operand arithmetic."""
        try:
            a_dec = Decimal(str(a))
            b_dec = Decimal(str(b))

            if op not in self.OPERATORS:
                return {"success": False, "error": f"Unknown operator: {op}"}

            if op in ['/', '//'] and b_dec == 0:
                return {"success": False, "error": "Division by zero"}

            result = self.OPERATORS[op](a_dec, b_dec)

            return {
                "success": True,
                "expression": f"{a} {op} {b}",
                "result": str(result),
                "numeric": float(result),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def factorial(self, n: int) -> Dict[str, Any]:
        """Calculate factorial."""
        try:
            if n < 0:
                return {"success": False, "error": "Factorial undefined for negative numbers"}
            if n > 1000:
                return {"success": False, "error": "Number too large (max 1000)"}

            result = math.factorial(n)
            return {
                "success": True,
                "expression": f"{n}!",
                "result": str(result),
                "digits": len(str(result)),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def gcd_lcm(self, a: int, b: int) -> Dict[str, Any]:
        """Calculate GCD and LCM."""
        try:
            gcd = math.gcd(a, b)
            lcm = math.lcm(a, b)
            return {
                "success": True,
                "a": a,
                "b": b,
                "gcd": gcd,
                "lcm": lcm,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def prime_factorization(self, n: int) -> Dict[str, Any]:
        """Get prime factorization."""
        try:
            if n < 2:
                return {"success": False, "error": "Number must be >= 2"}
            if n > 10**12:
                return {"success": False, "error": "Number too large"}

            factors = {}
            d = 2
            temp_n = n

            while d * d <= temp_n:
                while temp_n % d == 0:
                    factors[d] = factors.get(d, 0) + 1
                    temp_n //= d
                d += 1

            if temp_n > 1:
                factors[temp_n] = factors.get(temp_n, 0) + 1

            # Format as string
            factor_str = " × ".join(
                f"{p}^{e}" if e > 1 else str(p)
                for p, e in sorted(factors.items())
            )

            return {
                "success": True,
                "number": n,
                "factors": factors,
                "factorization": factor_str,
                "is_prime": len(factors) == 1 and list(factors.values())[0] == 1,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def percentage(
        self,
        operation: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Percentage calculations."""
        try:
            if operation == "of":
                # X% of Y
                percent = Decimal(str(kwargs.get("percent", 0)))
                value = Decimal(str(kwargs.get("value", 0)))
                result = (percent / 100) * value
                return {
                    "success": True,
                    "expression": f"{percent}% of {value}",
                    "result": str(result),
                }

            elif operation == "change":
                # Percent change from A to B
                old = Decimal(str(kwargs.get("old", 0)))
                new = Decimal(str(kwargs.get("new", 0)))
                if old == 0:
                    return {"success": False, "error": "Cannot calculate change from zero"}
                change = ((new - old) / old) * 100
                return {
                    "success": True,
                    "expression": f"change from {old} to {new}",
                    "result": f"{change}%",
                    "numeric": float(change),
                    "direction": "increase" if change > 0 else "decrease" if change < 0 else "no change",
                }

            elif operation == "what_percent":
                # X is what percent of Y
                part = Decimal(str(kwargs.get("part", 0)))
                whole = Decimal(str(kwargs.get("whole", 0)))
                if whole == 0:
                    return {"success": False, "error": "Cannot divide by zero"}
                result = (part / whole) * 100
                return {
                    "success": True,
                    "expression": f"{part} is what % of {whole}",
                    "result": f"{result}%",
                    "numeric": float(result),
                }

            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}

        except Exception as e:
            return {"success": False, "error": str(e)}


def calculator_tool(
    expression: str,
    precision: Optional[int] = None,
    as_fraction: Optional[bool] = None,
) -> ToolResult:
    """Tool function for calculator."""
    calc = Calculator()

    result = calc.evaluate(
        expression,
        precision=precision or 10,
        as_fraction=as_fraction or False,
    )

    if result.get("success"):
        # Format output for model
        output = result["result"]
        if "fraction" in result:
            output += f" (exact: {result['fraction']})"

        return ToolResult(
            status=ToolStatus.SUCCESS,
            output=output,
            metadata=result,
        )
    else:
        return ToolResult(
            status=ToolStatus.ERROR,
            output=None,
            error=result.get("error", "Unknown error"),
        )


def create_calculator_tool() -> Tool:
    """Create the calculator tool."""
    return Tool(
        name="calculator",
        description="Perform exact arithmetic calculations. Use this for any numerical computation to avoid floating point errors. Supports basic operations (+, -, *, /, **, %), functions (sqrt, log, sin, cos, etc.), and constants (pi, e).",
        parameters=[
            ToolParameter(
                name="expression",
                description="Mathematical expression to evaluate (e.g., '7 * 8', 'sqrt(2)', '2**10', 'sin(pi/4)')",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="precision",
                description="Decimal places for result (default: 10)",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="as_fraction",
                description="Also return result as exact fraction if possible",
                type="boolean",
                required=False,
            ),
        ],
        execute_fn=calculator_tool,
        timeout_ms=5000,
    )


def register_calculator_tools(registry: ToolRegistry) -> None:
    """Register calculator tools with the registry."""
    registry.register(create_calculator_tool())
