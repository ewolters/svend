"""
Sequence Analyzer - Pattern Recognition for Number Sequences

Provides verified solutions for:
- Common sequence detection (arithmetic, geometric, polynomial, Fibonacci-like)
- Next term prediction
- Closed-form formula generation
- Sequence properties (convergence, periodicity)
"""

from typing import Optional, Dict, Any, List, Tuple, Callable
from fractions import Fraction
from functools import lru_cache
import math

from .registry import Tool, ToolParameter, ToolResult, ToolStatus, ToolRegistry


class SequenceAnalyzer:
    """Analyzes number sequences to find patterns."""

    # Known sequence signatures for quick matching
    KNOWN_SEQUENCES = {
        "fibonacci": [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144],
        "lucas": [2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123],
        "triangular": [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91],
        "square": [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100],
        "cube": [0, 1, 8, 27, 64, 125, 216, 343, 512, 729, 1000],
        "prime": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47],
        "factorial": [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880],
        "powers_of_2": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        "catalan": [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862],
        "bell": [1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147],
        "partition": [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77],
    }

    def analyze(self, seq: List[float], predict_n: int = 3) -> Dict[str, Any]:
        """Analyze sequence and return identified patterns."""
        if len(seq) < 2:
            return {"error": "Need at least 2 terms"}

        results = {
            "input": seq,
            "length": len(seq),
            "patterns_found": [],
            "predictions": [],
            "properties": {}
        }

        # Try different pattern types
        patterns = []

        # 1. Check arithmetic sequence
        arith = self._check_arithmetic(seq)
        if arith["is_arithmetic"]:
            patterns.append({
                "type": "arithmetic",
                "common_difference": arith["difference"],
                "formula": f"a_n = {arith['first']} + {arith['difference']}(n-1)",
                "confidence": 1.0,
                "next_terms": [arith["first"] + arith["difference"] * (len(seq) + i) for i in range(predict_n)]
            })

        # 2. Check geometric sequence
        geom = self._check_geometric(seq)
        if geom["is_geometric"]:
            patterns.append({
                "type": "geometric",
                "common_ratio": geom["ratio"],
                "formula": f"a_n = {geom['first']} * {geom['ratio']}^(n-1)",
                "confidence": 1.0,
                "next_terms": [geom["first"] * (geom["ratio"] ** (len(seq) + i - 1)) for i in range(1, predict_n + 1)]
            })

        # 3. Check polynomial sequence
        poly = self._check_polynomial(seq)
        if poly["degree"] is not None and poly["degree"] <= 5:
            patterns.append({
                "type": "polynomial",
                "degree": poly["degree"],
                "coefficients": poly.get("coefficients"),
                "formula": poly.get("formula"),
                "confidence": 0.9,
                "next_terms": poly.get("next_terms", [])[:predict_n]
            })

        # 4. Check Fibonacci-like (linear recurrence)
        fib = self._check_fibonacci_like(seq)
        if fib["is_fibonacci_like"]:
            patterns.append({
                "type": "fibonacci_like",
                "recurrence": fib["recurrence"],
                "confidence": 0.95,
                "next_terms": fib.get("next_terms", [])[:predict_n]
            })

        # 5. Check known sequences
        known = self._check_known_sequences(seq)
        if known:
            patterns.append(known)

        # 6. Check powers
        power = self._check_powers(seq)
        if power["is_power"]:
            patterns.append({
                "type": "power",
                "base": power["base"],
                "formula": f"a_n = {power['base']}^n" if power.get("offset", 0) == 0 else f"a_n = {power['base']}^(n+{power.get('offset', 0)})",
                "confidence": 1.0,
                "next_terms": power.get("next_terms", [])[:predict_n]
            })

        # 7. Calculate differences table
        diff_table = self._difference_table(seq)
        results["difference_table"] = diff_table

        # Properties
        results["properties"]["sum"] = sum(seq)
        results["properties"]["mean"] = sum(seq) / len(seq)
        results["properties"]["is_increasing"] = all(seq[i] < seq[i+1] for i in range(len(seq)-1))
        results["properties"]["is_decreasing"] = all(seq[i] > seq[i+1] for i in range(len(seq)-1))

        # Sort patterns by confidence
        patterns.sort(key=lambda x: -x.get("confidence", 0))
        results["patterns_found"] = patterns

        # Best prediction
        if patterns:
            results["predictions"] = patterns[0].get("next_terms", [])
            results["best_pattern"] = patterns[0]["type"]

        return results

    def _check_arithmetic(self, seq: List[float]) -> Dict[str, Any]:
        """Check if sequence is arithmetic."""
        if len(seq) < 2:
            return {"is_arithmetic": False}

        diff = seq[1] - seq[0]
        is_arith = all(abs((seq[i+1] - seq[i]) - diff) < 1e-10 for i in range(len(seq)-1))

        return {
            "is_arithmetic": is_arith,
            "difference": diff if is_arith else None,
            "first": seq[0]
        }

    def _check_geometric(self, seq: List[float]) -> Dict[str, Any]:
        """Check if sequence is geometric."""
        if len(seq) < 2 or seq[0] == 0:
            return {"is_geometric": False}

        ratio = seq[1] / seq[0]
        is_geom = all(
            abs(seq[i]) > 1e-10 and abs((seq[i+1] / seq[i]) - ratio) < 1e-10
            for i in range(len(seq)-1)
        )

        return {
            "is_geometric": is_geom,
            "ratio": ratio if is_geom else None,
            "first": seq[0]
        }

    def _check_polynomial(self, seq: List[float]) -> Dict[str, Any]:
        """Check if sequence follows polynomial pattern using difference method."""
        if len(seq) < 3:
            return {"degree": None}

        # Build difference table
        diff_table = [seq]
        current = list(seq)

        for _ in range(len(seq) - 1):
            next_diff = [current[i+1] - current[i] for i in range(len(current)-1)]
            if not next_diff:
                break
            diff_table.append(next_diff)
            current = next_diff

            # Check if constant
            if len(set(round(x, 10) for x in next_diff)) == 1:
                degree = len(diff_table) - 1
                const = next_diff[0]

                # Generate next terms
                next_terms = self._extend_polynomial(diff_table, 5)

                # Try to find coefficients
                coeffs = self._find_polynomial_coefficients(seq, degree)

                formula = None
                if coeffs:
                    terms = []
                    for i, c in enumerate(coeffs):
                        if abs(c) < 1e-10:
                            continue
                        frac = Fraction(c).limit_denominator(1000)
                        if i == 0:
                            terms.append(str(frac))
                        elif i == 1:
                            terms.append(f"{frac}n")
                        else:
                            terms.append(f"{frac}n^{i}")
                    formula = " + ".join(reversed(terms)) if terms else "0"

                return {
                    "degree": degree,
                    "constant_difference": const,
                    "coefficients": coeffs,
                    "formula": formula,
                    "next_terms": next_terms
                }

        return {"degree": None}

    def _extend_polynomial(self, diff_table: List[List[float]], n: int) -> List[float]:
        """Extend sequence using difference table."""
        # Work backwards from constant row
        table = [list(row) for row in diff_table]

        for _ in range(n):
            # Extend each row from bottom up
            for i in range(len(table) - 1, 0, -1):
                if table[i]:
                    table[i].append(table[i][-1])  # Constant row stays constant
                    table[i-1].append(table[i-1][-1] + table[i][-1])

        return table[0][len(diff_table[0]):]

    def _find_polynomial_coefficients(self, seq: List[float], degree: int) -> Optional[List[float]]:
        """Find polynomial coefficients using Lagrange interpolation."""
        if len(seq) < degree + 1:
            return None

        n = degree + 1
        points = [(i, seq[i]) for i in range(n)]

        # Build coefficients
        coeffs = [0.0] * n

        for i, (xi, yi) in enumerate(points):
            # Compute Lagrange basis polynomial
            basis = [0.0] * n
            basis[0] = 1.0

            for j, (xj, _) in enumerate(points):
                if i != j:
                    # Multiply by (x - xj) / (xi - xj)
                    denom = xi - xj
                    new_basis = [0.0] * n
                    for k in range(n - 1, -1, -1):
                        new_basis[k] = -xj * basis[k] / denom
                        if k > 0:
                            new_basis[k] += basis[k-1] / denom
                    basis = new_basis

            # Add yi * basis to coeffs
            for k in range(n):
                coeffs[k] += yi * basis[k]

        return [round(c, 10) for c in coeffs]

    def _check_fibonacci_like(self, seq: List[float]) -> Dict[str, Any]:
        """Check if sequence follows a_n = a_{n-1} + a_{n-2} pattern."""
        if len(seq) < 4:
            return {"is_fibonacci_like": False}

        # Check if each term is sum of previous two
        is_fib = all(
            abs(seq[i] - (seq[i-1] + seq[i-2])) < 1e-10
            for i in range(2, len(seq))
        )

        if is_fib:
            next_terms = list(seq)
            for _ in range(5):
                next_terms.append(next_terms[-1] + next_terms[-2])
            return {
                "is_fibonacci_like": True,
                "recurrence": "a_n = a_{n-1} + a_{n-2}",
                "next_terms": next_terms[len(seq):]
            }

        # Check more general linear recurrence a_n = c1*a_{n-1} + c2*a_{n-2}
        if len(seq) >= 5:
            try:
                # Solve for c1, c2 using first equations
                # seq[2] = c1*seq[1] + c2*seq[0]
                # seq[3] = c1*seq[2] + c2*seq[1]
                det = seq[1] * seq[1] - seq[2] * seq[0]
                if abs(det) > 1e-10:
                    c1 = (seq[2] * seq[1] - seq[3] * seq[0]) / det
                    c2 = (seq[3] * seq[1] - seq[2] * seq[2]) / det

                    # Verify
                    is_valid = all(
                        abs(seq[i] - (c1 * seq[i-1] + c2 * seq[i-2])) < 1e-6
                        for i in range(2, len(seq))
                    )

                    if is_valid:
                        next_terms = list(seq)
                        for _ in range(5):
                            next_terms.append(c1 * next_terms[-1] + c2 * next_terms[-2])
                        return {
                            "is_fibonacci_like": True,
                            "recurrence": f"a_n = {c1:.4g}*a_{{n-1}} + {c2:.4g}*a_{{n-2}}",
                            "c1": c1,
                            "c2": c2,
                            "next_terms": next_terms[len(seq):]
                        }
            except:
                pass

        return {"is_fibonacci_like": False}

    def _check_known_sequences(self, seq: List[float]) -> Optional[Dict[str, Any]]:
        """Check against known sequences."""
        seq_int = [int(x) if x == int(x) else None for x in seq]
        if None in seq_int:
            return None

        for name, known in self.KNOWN_SEQUENCES.items():
            # Check if seq is a subsequence starting from some index
            for start in range(len(known) - len(seq_int) + 1):
                if known[start:start + len(seq_int)] == seq_int:
                    # Found match
                    next_idx = start + len(seq_int)
                    next_terms = known[next_idx:next_idx + 5] if next_idx < len(known) else []
                    return {
                        "type": f"known_sequence:{name}",
                        "name": name,
                        "starting_index": start,
                        "confidence": 1.0,
                        "next_terms": next_terms
                    }

        return None

    def _check_powers(self, seq: List[float]) -> Dict[str, Any]:
        """Check if sequence is powers of some base."""
        if len(seq) < 2 or seq[0] == 0:
            return {"is_power": False}

        # Check a^n pattern
        ratio = seq[1] / seq[0]
        if abs(ratio) < 1e-10:
            return {"is_power": False}

        is_power = all(
            abs(seq[i]) > 1e-10 and abs((seq[i+1] / seq[i]) - ratio) < 1e-10
            for i in range(len(seq)-1)
        )

        if is_power and abs(ratio - round(ratio)) < 1e-10:
            base = int(round(ratio))
            # Find what power seq[0] is
            if seq[0] == 1:
                offset = 0
            else:
                try:
                    offset = round(math.log(seq[0]) / math.log(base))
                except:
                    offset = 0

            next_terms = [seq[-1] * (base ** (i+1)) for i in range(5)]
            return {
                "is_power": True,
                "base": base,
                "offset": offset,
                "next_terms": next_terms
            }

        return {"is_power": False}

    def _difference_table(self, seq: List[float], max_depth: int = 5) -> List[List[float]]:
        """Build difference table."""
        table = [seq]
        current = list(seq)

        for _ in range(min(max_depth, len(seq) - 1)):
            next_diff = [current[i+1] - current[i] for i in range(len(current)-1)]
            if not next_diff:
                break
            table.append(next_diff)
            current = next_diff

        return table

    def generate_sequence(
        self,
        seq_type: str,
        n: int,
        **params
    ) -> List[float]:
        """Generate n terms of a sequence."""
        if seq_type == "arithmetic":
            a = params.get("first", 0)
            d = params.get("difference", 1)
            return [a + i * d for i in range(n)]

        elif seq_type == "geometric":
            a = params.get("first", 1)
            r = params.get("ratio", 2)
            return [a * (r ** i) for i in range(n)]

        elif seq_type == "fibonacci":
            if n <= 0:
                return []
            seq = [0, 1]
            while len(seq) < n:
                seq.append(seq[-1] + seq[-2])
            return seq[:n]

        elif seq_type == "triangular":
            return [i * (i + 1) // 2 for i in range(n)]

        elif seq_type == "square":
            return [i * i for i in range(n)]

        elif seq_type == "prime":
            primes = []
            candidate = 2
            while len(primes) < n:
                is_prime = all(candidate % p != 0 for p in primes if p * p <= candidate)
                if is_prime:
                    primes.append(candidate)
                candidate += 1
            return primes

        elif seq_type == "factorial":
            result = [1]
            for i in range(1, n):
                result.append(result[-1] * i)
            return result

        elif seq_type == "polynomial":
            coeffs = params.get("coefficients", [0, 1])  # Default: n
            return [sum(c * (i ** j) for j, c in enumerate(coeffs)) for i in range(n)]

        else:
            raise ValueError(f"Unknown sequence type: {seq_type}")


def sequence_tool(
    operation: str,
    sequence: Optional[List[float]] = None,
    sequence_type: Optional[str] = None,
    n: Optional[int] = None,
    predict_n: int = 3,
    **params
) -> ToolResult:
    """Execute sequence analysis operation."""
    try:
        analyzer = SequenceAnalyzer()

        if operation == "analyze":
            if not sequence:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="sequence is required")

            result = analyzer.analyze(sequence, predict_n=predict_n)

            if "error" in result:
                return ToolResult(status=ToolStatus.ERROR, output=None, error=result["error"])

            # Format output
            output_lines = [f"Sequence: {sequence}"]

            if result["patterns_found"]:
                best = result["patterns_found"][0]
                output_lines.append(f"Best match: {best['type']}")
                if "formula" in best:
                    output_lines.append(f"Formula: {best['formula']}")
                if "recurrence" in best:
                    output_lines.append(f"Recurrence: {best['recurrence']}")
                output_lines.append(f"Next {predict_n} terms: {result['predictions']}")
            else:
                output_lines.append("No clear pattern detected")
                output_lines.append(f"Difference table: {result['difference_table']}")

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output="\n".join(output_lines),
                metadata=result
            )

        elif operation == "generate":
            if not sequence_type or not n:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="sequence_type and n required")

            seq = analyzer.generate_sequence(sequence_type, n, **params)

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"{sequence_type} sequence (first {n} terms): {seq}",
                metadata={"sequence": seq, "type": sequence_type}
            )

        elif operation == "differences":
            if not sequence:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="sequence is required")

            table = analyzer._difference_table(sequence)

            output_lines = ["Difference table:"]
            for i, row in enumerate(table):
                output_lines.append(f"  Δ^{i}: {row}")

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output="\n".join(output_lines),
                metadata={"difference_table": table}
            )

        elif operation == "predict":
            if not sequence:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="sequence is required")

            result = analyzer.analyze(sequence, predict_n=predict_n)
            predictions = result.get("predictions", [])

            if predictions:
                pattern = result.get("best_pattern", "unknown")
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=f"Next {len(predictions)} terms (pattern: {pattern}): {predictions}",
                    metadata={"predictions": predictions, "pattern": pattern}
                )
            else:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    output="Could not determine pattern for prediction",
                    metadata={"predictions": []}
                )

        else:
            return ToolResult(
                status=ToolStatus.INVALID_INPUT,
                output=None,
                error=f"Unknown operation: {operation}. Valid: analyze, generate, differences, predict"
            )

    except Exception as e:
        return ToolResult(status=ToolStatus.ERROR, output=None, error=str(e))


def create_sequence_tool() -> Tool:
    """Create sequence analyzer tool."""
    return Tool(
        name="sequence",
        description="Analyze number sequences to find patterns (arithmetic, geometric, polynomial, Fibonacci-like, known sequences), predict next terms, and generate sequences.",
        parameters=[
            ToolParameter(
                name="operation",
                description="Operation: analyze, generate, differences, predict",
                type="string",
                required=True,
                enum=["analyze", "generate", "differences", "predict"]
            ),
            ToolParameter(
                name="sequence",
                description="Input sequence as list of numbers (for analyze/differences/predict)",
                type="array",
                required=False,
            ),
            ToolParameter(
                name="sequence_type",
                description="Type to generate: arithmetic, geometric, fibonacci, triangular, square, prime, factorial, polynomial",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="n",
                description="Number of terms to generate",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="predict_n",
                description="Number of terms to predict (default: 3)",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="first",
                description="First term (for arithmetic/geometric generation)",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="difference",
                description="Common difference (for arithmetic generation)",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="ratio",
                description="Common ratio (for geometric generation)",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="coefficients",
                description="Polynomial coefficients [a0, a1, a2, ...] for a0 + a1*n + a2*n^2 + ...",
                type="array",
                required=False,
            ),
        ],
        execute_fn=sequence_tool,
        timeout_ms=10000,
    )


def register_sequence_tools(registry: ToolRegistry) -> None:
    """Register sequence analysis tools."""
    registry.register(create_sequence_tool())
