"""
Numerical Computing Tool - NumPy/SciPy Operations

For when symbolic math isn't enough: linear algebra, optimization,
differential equations, interpolation, FFT, numerical integration.
"""

from typing import Optional, Dict, Any, List, Union
import json

from .registry import Tool, ToolParameter, ToolResult, ToolStatus, ToolRegistry


class NumericalEngine:
    """
    Numerical computing engine using NumPy/SciPy.

    Lazy imports to avoid loading heavy libraries until needed.
    """

    def __init__(self):
        self._np = None
        self._scipy = None

    @property
    def np(self):
        """Lazy import numpy."""
        if self._np is None:
            try:
                import numpy as np
                self._np = np
            except ImportError:
                raise ImportError("NumPy required. Install with: pip install numpy")
        return self._np

    @property
    def scipy(self):
        """Lazy import scipy."""
        if self._scipy is None:
            try:
                import scipy
                self._scipy = scipy
            except ImportError:
                raise ImportError("SciPy required. Install with: pip install scipy")
        return self._scipy

    # ==================== LINEAR ALGEBRA ====================

    def solve_linear(
        self,
        A: List[List[float]],
        b: List[float],
    ) -> Dict[str, Any]:
        """
        Solve linear system Ax = b.

        Args:
            A: Coefficient matrix
            b: Right-hand side vector

        Returns:
            Solution vector x
        """
        try:
            np = self.np
            A_arr = np.array(A, dtype=float)
            b_arr = np.array(b, dtype=float)

            # Check dimensions
            if A_arr.shape[0] != A_arr.shape[1]:
                return {"success": False, "error": "Matrix A must be square"}
            if A_arr.shape[0] != len(b_arr):
                return {"success": False, "error": "Dimensions of A and b don't match"}

            # Check if singular
            det = np.linalg.det(A_arr)
            if abs(det) < 1e-10:
                return {"success": False, "error": "Matrix is singular or nearly singular"}

            x = np.linalg.solve(A_arr, b_arr)

            return {
                "success": True,
                "x": x.tolist(),
                "determinant": det,
                "method": "LU decomposition",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def matrix_decomposition(
        self,
        matrix: List[List[float]],
        method: str = "lu",
    ) -> Dict[str, Any]:
        """
        Perform matrix decomposition.

        Methods: lu, qr, svd, cholesky, eigen
        """
        try:
            np = self.np
            M = np.array(matrix, dtype=float)

            if method == "lu":
                from scipy.linalg import lu
                P, L, U = lu(M)
                return {
                    "success": True,
                    "method": "LU",
                    "P": P.tolist(),
                    "L": L.tolist(),
                    "U": U.tolist(),
                }

            elif method == "qr":
                Q, R = np.linalg.qr(M)
                return {
                    "success": True,
                    "method": "QR",
                    "Q": Q.tolist(),
                    "R": R.tolist(),
                }

            elif method == "svd":
                U, S, Vh = np.linalg.svd(M)
                return {
                    "success": True,
                    "method": "SVD",
                    "U": U.tolist(),
                    "S": S.tolist(),
                    "Vh": Vh.tolist(),
                    "rank": int(np.sum(S > 1e-10)),
                }

            elif method == "cholesky":
                L = np.linalg.cholesky(M)
                return {
                    "success": True,
                    "method": "Cholesky",
                    "L": L.tolist(),
                    "note": "M = L @ L.T",
                }

            elif method == "eigen":
                eigenvalues, eigenvectors = np.linalg.eig(M)
                return {
                    "success": True,
                    "method": "Eigendecomposition",
                    "eigenvalues": eigenvalues.tolist(),
                    "eigenvectors": eigenvectors.tolist(),
                }

            else:
                return {"success": False, "error": f"Unknown method: {method}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def matrix_properties(
        self,
        matrix: List[List[float]],
    ) -> Dict[str, Any]:
        """Get various matrix properties."""
        try:
            np = self.np
            M = np.array(matrix, dtype=float)

            props = {
                "success": True,
                "shape": list(M.shape),
            }

            if M.shape[0] == M.shape[1]:  # Square matrix
                props["determinant"] = float(np.linalg.det(M))
                props["trace"] = float(np.trace(M))
                props["rank"] = int(np.linalg.matrix_rank(M))

                try:
                    props["condition_number"] = float(np.linalg.cond(M))
                except:
                    props["condition_number"] = None

                eigenvalues = np.linalg.eigvals(M)
                props["eigenvalues"] = [complex(e) if np.iscomplex(e) else float(e.real) for e in eigenvalues]

                # Check properties
                props["is_symmetric"] = bool(np.allclose(M, M.T))
                props["is_positive_definite"] = bool(np.all(eigenvalues > 0))

            props["norm_frobenius"] = float(np.linalg.norm(M, 'fro'))
            props["norm_2"] = float(np.linalg.norm(M, 2))

            return props

        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==================== OPTIMIZATION ====================

    def minimize(
        self,
        function: str,
        x0: List[float],
        method: str = "BFGS",
        bounds: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        """
        Minimize a function.

        Args:
            function: Function as string (e.g., "x[0]**2 + x[1]**2")
            x0: Initial guess
            method: Optimization method (BFGS, Nelder-Mead, L-BFGS-B, etc.)
            bounds: Optional bounds for each variable [(low, high), ...]
        """
        try:
            np = self.np
            from scipy.optimize import minimize as scipy_minimize

            # Create function from string
            def f(x):
                return eval(function, {"__builtins__": {}, "x": x, "np": np,
                                       "sin": np.sin, "cos": np.cos, "exp": np.exp,
                                       "log": np.log, "sqrt": np.sqrt, "abs": np.abs})

            # Convert bounds format
            scipy_bounds = None
            if bounds:
                scipy_bounds = [(b[0], b[1]) for b in bounds]

            result = scipy_minimize(f, x0, method=method, bounds=scipy_bounds)

            return {
                "success": result.success,
                "x": result.x.tolist(),
                "fun": float(result.fun),
                "message": result.message,
                "iterations": int(result.nit) if hasattr(result, 'nit') else None,
                "method": method,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def root_find(
        self,
        function: str,
        x0: Union[float, List[float]],
        method: str = "hybr",
        bracket: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Find root of a function.

        Args:
            function: Function as string (e.g., "x**3 - x - 2")
            x0: Initial guess
            method: Method (hybr, brentq for bracketed, newton)
            bracket: Bracket [a, b] for brentq method
        """
        try:
            np = self.np
            from scipy.optimize import root, brentq

            # Create function from string
            def f(x):
                return eval(function, {"__builtins__": {}, "x": x, "np": np,
                                       "sin": np.sin, "cos": np.cos, "exp": np.exp,
                                       "log": np.log, "sqrt": np.sqrt, "abs": np.abs})

            if bracket and method == "brentq":
                root_val = brentq(f, bracket[0], bracket[1])
                return {
                    "success": True,
                    "root": float(root_val),
                    "method": "brentq (bracketed)",
                    "converged": True,
                }
            else:
                result = root(f, x0, method=method)
                return {
                    "success": result.success,
                    "root": result.x.tolist() if hasattr(result.x, 'tolist') else float(result.x),
                    "method": method,
                    "converged": result.success,
                    "message": result.message,
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def curve_fit(
        self,
        x_data: List[float],
        y_data: List[float],
        model: str = "polynomial",
        degree: int = 2,
    ) -> Dict[str, Any]:
        """
        Fit a curve to data.

        Args:
            x_data: X values
            y_data: Y values
            model: Model type (polynomial, exponential, linear)
            degree: Polynomial degree (if applicable)
        """
        try:
            np = self.np
            x = np.array(x_data)
            y = np.array(y_data)

            if model == "polynomial" or model == "linear":
                if model == "linear":
                    degree = 1

                coeffs = np.polyfit(x, y, degree)
                poly = np.poly1d(coeffs)
                y_fit = poly(x)

                # R-squared
                ss_res = np.sum((y - y_fit) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                # Build equation string
                terms = []
                for i, c in enumerate(coeffs):
                    power = degree - i
                    if power == 0:
                        terms.append(f"{c:.4g}")
                    elif power == 1:
                        terms.append(f"{c:.4g}x")
                    else:
                        terms.append(f"{c:.4g}x^{power}")
                equation = " + ".join(terms).replace("+ -", "- ")

                return {
                    "success": True,
                    "model": model,
                    "coefficients": coeffs.tolist(),
                    "equation": equation,
                    "r_squared": float(r_squared),
                    "degree": degree,
                }

            elif model == "exponential":
                # y = a * exp(b * x)
                from scipy.optimize import curve_fit as scipy_curve_fit

                def exp_func(x, a, b):
                    return a * np.exp(b * x)

                # Need positive y values for log transform initial guess
                if np.any(y <= 0):
                    return {"success": False, "error": "Exponential fit requires positive y values"}

                popt, _ = scipy_curve_fit(exp_func, x, y, p0=[1, 0.1], maxfev=5000)
                y_fit = exp_func(x, *popt)

                ss_res = np.sum((y - y_fit) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                return {
                    "success": True,
                    "model": "exponential",
                    "parameters": {"a": float(popt[0]), "b": float(popt[1])},
                    "equation": f"{popt[0]:.4g} * exp({popt[1]:.4g} * x)",
                    "r_squared": float(r_squared),
                }

            else:
                return {"success": False, "error": f"Unknown model: {model}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==================== DIFFERENTIAL EQUATIONS ====================

    def solve_ode(
        self,
        equation: str,
        y0: List[float],
        t_span: List[float],
        t_eval: Optional[List[float]] = None,
        method: str = "RK45",
    ) -> Dict[str, Any]:
        """
        Solve ordinary differential equation.

        Args:
            equation: dy/dt as function of (t, y), e.g., "-0.5*y[0]" or "[y[1], -9.8]"
            y0: Initial conditions
            t_span: [t_start, t_end]
            t_eval: Optional specific times to evaluate
            method: Integration method (RK45, RK23, DOP853, Radau, BDF)
        """
        try:
            np = self.np
            from scipy.integrate import solve_ivp

            # Create function from string
            def dydt(t, y):
                result = eval(equation, {"__builtins__": {}, "t": t, "y": y, "np": np,
                                         "sin": np.sin, "cos": np.cos, "exp": np.exp,
                                         "log": np.log, "sqrt": np.sqrt, "abs": np.abs})
                if isinstance(result, (int, float)):
                    return [result]
                return result

            if t_eval is None:
                t_eval = np.linspace(t_span[0], t_span[1], 100)

            result = solve_ivp(dydt, t_span, y0, method=method, t_eval=t_eval)

            return {
                "success": result.success,
                "t": result.t.tolist(),
                "y": result.y.tolist(),
                "method": method,
                "message": result.message,
                "nfev": int(result.nfev),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==================== NUMERICAL INTEGRATION ====================

    def integrate(
        self,
        function: str,
        a: float,
        b: float,
        method: str = "quad",
    ) -> Dict[str, Any]:
        """
        Numerical integration of f(x) from a to b.

        Args:
            function: Function as string (e.g., "x**2", "sin(x)")
            a: Lower bound
            b: Upper bound
            method: Integration method (quad, trapz, simpson)
        """
        try:
            np = self.np
            from scipy import integrate

            # Create function from string
            def f(x):
                return eval(function, {"__builtins__": {}, "x": x, "np": np,
                                       "sin": np.sin, "cos": np.cos, "exp": np.exp,
                                       "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
                                       "pi": np.pi, "e": np.e})

            if method == "quad":
                result, error = integrate.quad(f, a, b)
                return {
                    "success": True,
                    "result": float(result),
                    "error": float(error),
                    "method": "adaptive quadrature",
                }

            elif method in ["trapz", "trapezoid"]:
                x = np.linspace(a, b, 1000)
                y = np.array([f(xi) for xi in x])
                result = np.trapz(y, x)
                return {
                    "success": True,
                    "result": float(result),
                    "method": "trapezoidal rule",
                    "points": 1000,
                }

            elif method == "simpson":
                x = np.linspace(a, b, 1001)  # Odd number for Simpson
                y = np.array([f(xi) for xi in x])
                result = integrate.simpson(y, x=x)
                return {
                    "success": True,
                    "result": float(result),
                    "method": "Simpson's rule",
                    "points": 1001,
                }

            else:
                return {"success": False, "error": f"Unknown method: {method}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==================== INTERPOLATION ====================

    def interpolate(
        self,
        x_data: List[float],
        y_data: List[float],
        x_new: List[float],
        method: str = "linear",
    ) -> Dict[str, Any]:
        """
        Interpolate data points.

        Args:
            x_data: Known x values
            y_data: Known y values
            x_new: X values to interpolate at
            method: Interpolation method (linear, cubic, spline)
        """
        try:
            np = self.np
            from scipy import interpolate

            x = np.array(x_data)
            y = np.array(y_data)
            x_interp = np.array(x_new)

            if method == "linear":
                f = interpolate.interp1d(x, y, kind='linear', fill_value='extrapolate')
            elif method == "cubic":
                f = interpolate.interp1d(x, y, kind='cubic')
            elif method == "spline":
                f = interpolate.UnivariateSpline(x, y)
            else:
                return {"success": False, "error": f"Unknown method: {method}"}

            y_new = f(x_interp)

            return {
                "success": True,
                "x": x_interp.tolist(),
                "y": y_new.tolist(),
                "method": method,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==================== FFT ====================

    def fft(
        self,
        data: List[float],
        sample_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Compute Fast Fourier Transform.

        Args:
            data: Time-domain signal
            sample_rate: Sample rate in Hz (for frequency axis)
        """
        try:
            np = self.np

            signal = np.array(data)
            n = len(signal)

            fft_result = np.fft.fft(signal)
            magnitude = np.abs(fft_result)
            phase = np.angle(fft_result)

            # Frequency axis
            if sample_rate:
                freq = np.fft.fftfreq(n, 1/sample_rate)
            else:
                freq = np.fft.fftfreq(n)

            # Only return positive frequencies
            half_n = n // 2
            return {
                "success": True,
                "frequencies": freq[:half_n].tolist(),
                "magnitude": magnitude[:half_n].tolist(),
                "phase": phase[:half_n].tolist(),
                "dominant_frequency": float(freq[np.argmax(magnitude[:half_n])]),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}


def numerical_tool(
    operation: str,
    data: str,  # JSON string for complex inputs
    options: Optional[str] = None,
) -> ToolResult:
    """Tool function for numerical computing."""
    engine = NumericalEngine()

    try:
        data_dict = json.loads(data)
        opts = json.loads(options) if options else {}
    except json.JSONDecodeError as e:
        return ToolResult(
            status=ToolStatus.INVALID_INPUT,
            output=None,
            error=f"Invalid JSON: {e}",
        )

    try:
        # Route to appropriate method
        if operation == "solve_linear":
            result = engine.solve_linear(data_dict.get("A"), data_dict.get("b"))

        elif operation == "decomposition":
            result = engine.matrix_decomposition(
                data_dict.get("matrix"),
                opts.get("method", "lu"),
            )

        elif operation == "matrix_properties":
            result = engine.matrix_properties(data_dict.get("matrix"))

        elif operation == "minimize":
            result = engine.minimize(
                data_dict.get("function"),
                data_dict.get("x0"),
                opts.get("method", "BFGS"),
                data_dict.get("bounds"),
            )

        elif operation == "root_find":
            result = engine.root_find(
                data_dict.get("function"),
                data_dict.get("x0"),
                opts.get("method", "hybr"),
                data_dict.get("bracket"),
            )

        elif operation == "curve_fit":
            result = engine.curve_fit(
                data_dict.get("x"),
                data_dict.get("y"),
                opts.get("model", "polynomial"),
                opts.get("degree", 2),
            )

        elif operation == "solve_ode":
            result = engine.solve_ode(
                data_dict.get("equation"),
                data_dict.get("y0"),
                data_dict.get("t_span"),
                data_dict.get("t_eval"),
                opts.get("method", "RK45"),
            )

        elif operation == "integrate":
            result = engine.integrate(
                data_dict.get("function"),
                data_dict.get("a"),
                data_dict.get("b"),
                opts.get("method", "quad"),
            )

        elif operation == "interpolate":
            result = engine.interpolate(
                data_dict.get("x"),
                data_dict.get("y"),
                data_dict.get("x_new"),
                opts.get("method", "linear"),
            )

        elif operation == "fft":
            result = engine.fft(
                data_dict.get("signal"),
                data_dict.get("sample_rate"),
            )

        else:
            return ToolResult(
                status=ToolStatus.INVALID_INPUT,
                output=None,
                error=f"Unknown operation: {operation}",
            )

        if result.get("success"):
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=json.dumps(result, indent=2),
                metadata=result,
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


def create_numerical_tool() -> Tool:
    """Create the numerical computing tool."""
    return Tool(
        name="numerical",
        description="Numerical computing with NumPy/SciPy. Operations: solve_linear (Ax=b), decomposition (LU/QR/SVD/Cholesky/eigen), matrix_properties, minimize, root_find, curve_fit, solve_ode, integrate, interpolate, fft.",
        parameters=[
            ToolParameter(
                name="operation",
                description="Operation to perform",
                type="string",
                required=True,
                enum=["solve_linear", "decomposition", "matrix_properties",
                      "minimize", "root_find", "curve_fit",
                      "solve_ode", "integrate", "interpolate", "fft"],
            ),
            ToolParameter(
                name="data",
                description="JSON object with input data. Contents depend on operation.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="options",
                description="JSON object with optional parameters (method, degree, etc.)",
                type="string",
                required=False,
            ),
        ],
        execute_fn=numerical_tool,
        timeout_ms=30000,
    )


def register_numerical_tools(registry: ToolRegistry) -> None:
    """Register numerical tools with the registry."""
    registry.register(create_numerical_tool())
