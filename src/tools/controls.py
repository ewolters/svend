"""
Controls Engineering Specialist Tool

Provides control systems analysis and design:
- Transfer functions (poles, zeros, stability)
- PID controller tuning
- Frequency response (Bode, Nyquist)
- Stability analysis (gain/phase margins)
- State space representation
- Root locus analysis

This is epistemic scaffolding - encodes how controls engineers think.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
import math
import cmath
from enum import Enum

from .registry import Tool, ToolParameter, ToolResult, ToolStatus, ToolRegistry


class StabilityStatus(Enum):
    """Stability classification."""
    STABLE = "stable"
    MARGINALLY_STABLE = "marginally_stable"
    UNSTABLE = "unstable"
    UNKNOWN = "unknown"


@dataclass
class TransferFunction:
    """Transfer function G(s) = num(s) / den(s)."""
    numerator: List[float]  # Coefficients [a_n, a_{n-1}, ..., a_0]
    denominator: List[float]
    name: str = "G(s)"

    @property
    def order(self) -> int:
        """System order (degree of denominator)."""
        return len(self.denominator) - 1

    def evaluate(self, s: complex) -> complex:
        """Evaluate G(s) at a point."""
        num = sum(c * (s ** i) for i, c in enumerate(reversed(self.numerator)))
        den = sum(c * (s ** i) for i, c in enumerate(reversed(self.denominator)))
        if abs(den) < 1e-15:
            return complex(float('inf'), 0)
        return num / den

    def frequency_response(self, omega: float) -> Tuple[float, float]:
        """Get magnitude and phase at frequency omega rad/s."""
        s = complex(0, omega)
        g = self.evaluate(s)
        magnitude = abs(g)
        phase = math.degrees(cmath.phase(g))
        return magnitude, phase

    def to_string(self) -> str:
        """String representation."""
        def poly_str(coeffs, var='s'):
            terms = []
            n = len(coeffs) - 1
            for i, c in enumerate(coeffs):
                power = n - i
                if abs(c) < 1e-10:
                    continue
                if power == 0:
                    terms.append(f"{c:.4g}")
                elif power == 1:
                    terms.append(f"{c:.4g}{var}")
                else:
                    terms.append(f"{c:.4g}{var}^{power}")
            return " + ".join(terms) if terms else "0"

        return f"({poly_str(self.numerator)}) / ({poly_str(self.denominator)})"


@dataclass
class PIDController:
    """PID controller parameters."""
    Kp: float = 1.0  # Proportional gain
    Ki: float = 0.0  # Integral gain
    Kd: float = 0.0  # Derivative gain

    @property
    def transfer_function(self) -> TransferFunction:
        """Get transfer function form: Kp + Ki/s + Kd*s."""
        # C(s) = (Kd*s^2 + Kp*s + Ki) / s
        return TransferFunction(
            numerator=[self.Kd, self.Kp, self.Ki],
            denominator=[1, 0],  # s
            name="C(s)"
        )

    def to_dict(self) -> Dict[str, float]:
        return {"Kp": self.Kp, "Ki": self.Ki, "Kd": self.Kd}


class ControlsAnalyzer:
    """
    Control systems analysis engine.

    Encodes how controls engineers approach problems:
    1. Model the system (transfer function)
    2. Analyze stability (poles, margins)
    3. Design controller (PID, lead/lag)
    4. Verify performance (step response, frequency response)
    """

    @staticmethod
    def find_roots(coefficients: List[float]) -> List[complex]:
        """
        Find roots of polynomial using companion matrix method.
        coefficients: [a_n, a_{n-1}, ..., a_0] for a_n*x^n + ...
        """
        if len(coefficients) <= 1:
            return []

        # Normalize
        coeffs = [c / coefficients[0] for c in coefficients[1:]]
        n = len(coeffs)

        if n == 1:
            return [complex(-coeffs[0], 0)]

        if n == 2:
            # Quadratic formula
            a, b, c = 1, coeffs[0], coeffs[1]
            disc = b*b - 4*a*c
            if disc >= 0:
                sqrt_disc = math.sqrt(disc)
                return [complex((-b + sqrt_disc) / 2, 0),
                        complex((-b - sqrt_disc) / 2, 0)]
            else:
                sqrt_disc = math.sqrt(-disc)
                return [complex(-b/2, sqrt_disc/2),
                        complex(-b/2, -sqrt_disc/2)]

        # For higher order, use numpy if available
        try:
            import numpy as np
            return list(np.roots(coefficients))
        except ImportError:
            # Fallback: numerical iteration (simplified)
            return []

    def get_poles(self, tf: TransferFunction) -> List[complex]:
        """Get poles (roots of denominator)."""
        return self.find_roots(tf.denominator)

    def get_zeros(self, tf: TransferFunction) -> List[complex]:
        """Get zeros (roots of numerator)."""
        return self.find_roots(tf.numerator)

    def analyze_stability(self, tf: TransferFunction) -> Dict[str, Any]:
        """
        Analyze system stability from pole locations.

        Stable: All poles in left half plane (Re < 0)
        Marginally stable: Poles on imaginary axis
        Unstable: Any pole in right half plane (Re > 0)
        """
        poles = self.get_poles(tf)

        if not poles:
            return {
                "status": StabilityStatus.UNKNOWN.value,
                "poles": [],
                "reason": "Could not compute poles"
            }

        # Classify poles
        rhp_poles = [p for p in poles if p.real > 1e-10]
        jw_poles = [p for p in poles if abs(p.real) <= 1e-10]
        lhp_poles = [p for p in poles if p.real < -1e-10]

        # Format poles for output
        def format_pole(p):
            if abs(p.imag) < 1e-10:
                return f"{p.real:.4g}"
            elif p.imag >= 0:
                return f"{p.real:.4g} + {p.imag:.4g}j"
            else:
                return f"{p.real:.4g} - {abs(p.imag):.4g}j"

        if rhp_poles:
            status = StabilityStatus.UNSTABLE
            reason = f"Poles in RHP: {[format_pole(p) for p in rhp_poles]}"
        elif jw_poles:
            status = StabilityStatus.MARGINALLY_STABLE
            reason = f"Poles on imaginary axis: {[format_pole(p) for p in jw_poles]}"
        else:
            status = StabilityStatus.STABLE
            reason = "All poles in LHP"

        return {
            "status": status.value,
            "poles": [{"real": p.real, "imag": p.imag, "formatted": format_pole(p)} for p in poles],
            "rhp_count": len(rhp_poles),
            "lhp_count": len(lhp_poles),
            "jw_count": len(jw_poles),
            "reason": reason,
            "dominant_pole": format_pole(max(poles, key=lambda p: p.real)) if poles else None
        }

    def gain_margin(self, tf: TransferFunction, omega_range: Tuple[float, float] = (0.01, 100)) -> Dict[str, Any]:
        """
        Calculate gain margin.

        Gain margin = 1/|G(jw)| at phase crossover frequency (where phase = -180)
        """
        # Search for phase crossover
        omega_low, omega_high = omega_range
        num_points = 1000

        phase_crossover = None
        gm_linear = None

        for i in range(num_points):
            omega = omega_low * (omega_high / omega_low) ** (i / num_points)
            mag, phase = tf.frequency_response(omega)

            # Normalize phase to -180 to 180
            while phase > 180:
                phase -= 360
            while phase < -180:
                phase += 360

            if abs(phase + 180) < 5:  # Close to -180 degrees
                if phase_crossover is None or abs(phase + 180) < abs(tf.frequency_response(phase_crossover)[1] + 180):
                    phase_crossover = omega
                    gm_linear = 1 / mag if mag > 1e-10 else float('inf')

        if phase_crossover is None:
            return {
                "gain_margin_db": float('inf'),
                "gain_margin_linear": float('inf'),
                "phase_crossover_freq": None,
                "interpretation": "System never reaches -180 phase - infinite gain margin"
            }

        gm_db = 20 * math.log10(gm_linear) if gm_linear > 0 and gm_linear != float('inf') else float('inf')

        return {
            "gain_margin_db": round(gm_db, 2) if gm_db != float('inf') else float('inf'),
            "gain_margin_linear": round(gm_linear, 4) if gm_linear != float('inf') else float('inf'),
            "phase_crossover_freq": round(phase_crossover, 4),
            "interpretation": f"Can increase gain by {gm_db:.1f} dB before instability" if gm_db > 0 else "System is unstable (GM < 0 dB)"
        }

    def phase_margin(self, tf: TransferFunction, omega_range: Tuple[float, float] = (0.01, 100)) -> Dict[str, Any]:
        """
        Calculate phase margin.

        Phase margin = 180 + phase(G(jw)) at gain crossover frequency (where |G| = 1)
        """
        omega_low, omega_high = omega_range
        num_points = 1000

        gain_crossover = None
        phase_at_crossover = None

        for i in range(num_points):
            omega = omega_low * (omega_high / omega_low) ** (i / num_points)
            mag, phase = tf.frequency_response(omega)

            if abs(mag - 1) < 0.1:  # Close to 0 dB
                if gain_crossover is None or abs(mag - 1) < abs(tf.frequency_response(gain_crossover)[0] - 1):
                    gain_crossover = omega
                    phase_at_crossover = phase

        if gain_crossover is None:
            return {
                "phase_margin_deg": float('inf'),
                "gain_crossover_freq": None,
                "interpretation": "System never reaches 0 dB gain"
            }

        pm = 180 + phase_at_crossover

        return {
            "phase_margin_deg": round(pm, 2),
            "gain_crossover_freq": round(gain_crossover, 4),
            "phase_at_crossover": round(phase_at_crossover, 2),
            "interpretation": f"{'Stable' if pm > 0 else 'Unstable'} - {abs(pm):.1f} deg {'margin' if pm > 0 else 'deficiency'}"
        }

    def bode_data(self, tf: TransferFunction, omega_range: Tuple[float, float] = (0.01, 100), num_points: int = 100) -> Dict[str, Any]:
        """Generate Bode plot data."""
        omega_low, omega_high = omega_range

        frequencies = []
        magnitudes_db = []
        phases = []

        for i in range(num_points):
            omega = omega_low * (omega_high / omega_low) ** (i / (num_points - 1))
            mag, phase = tf.frequency_response(omega)

            frequencies.append(omega)
            magnitudes_db.append(20 * math.log10(mag) if mag > 1e-15 else -300)
            phases.append(phase)

        return {
            "frequencies": frequencies,
            "magnitude_db": magnitudes_db,
            "phase_deg": phases,
            "dc_gain_db": magnitudes_db[0] if magnitudes_db else None,
            "bandwidth_hz": self._find_bandwidth(frequencies, magnitudes_db)
        }

    def _find_bandwidth(self, freqs: List[float], mags_db: List[float]) -> Optional[float]:
        """Find -3dB bandwidth."""
        if not mags_db:
            return None
        dc_gain = mags_db[0]
        for i, (f, m) in enumerate(zip(freqs, mags_db)):
            if m < dc_gain - 3:
                return f
        return None

    def tune_pid_ziegler_nichols(self, Ku: float, Tu: float) -> PIDController:
        """
        Ziegler-Nichols tuning method.

        Args:
            Ku: Ultimate gain (gain at which system oscillates)
            Tu: Ultimate period (oscillation period at Ku)

        Returns:
            PID controller with tuned parameters
        """
        # Classic Ziegler-Nichols
        Kp = 0.6 * Ku
        Ki = 2 * Kp / Tu
        Kd = Kp * Tu / 8

        return PIDController(Kp=Kp, Ki=Ki, Kd=Kd)

    def tune_pid_cohen_coon(self, K: float, tau: float, theta: float) -> PIDController:
        """
        Cohen-Coon tuning for FOPDT model.

        Args:
            K: Process gain
            tau: Time constant
            theta: Dead time

        Returns:
            PID controller with tuned parameters
        """
        r = theta / tau

        Kp = (1/K) * (tau/theta) * (4/3 + r/4)
        Ti = theta * (32 + 6*r) / (13 + 8*r)
        Td = theta * 4 / (11 + 2*r)

        Ki = Kp / Ti
        Kd = Kp * Td

        return PIDController(Kp=Kp, Ki=Ki, Kd=Kd)

    def step_response_characteristics(self, tf: TransferFunction) -> Dict[str, Any]:
        """
        Estimate step response characteristics from transfer function.

        For second-order systems: rise time, settling time, overshoot, etc.
        """
        poles = self.get_poles(tf)

        if len(poles) < 2:
            return {"error": "Need at least second-order system"}

        # Find dominant poles (closest to imaginary axis)
        dominant = sorted(poles, key=lambda p: p.real, reverse=True)[:2]

        # For complex conjugate dominant poles
        if abs(dominant[0].imag) > 1e-10:
            sigma = abs(dominant[0].real)
            omega_d = abs(dominant[0].imag)
            omega_n = math.sqrt(sigma**2 + omega_d**2)
            zeta = sigma / omega_n

            # Standard second-order approximations
            if zeta < 1:
                overshoot = 100 * math.exp(-zeta * math.pi / math.sqrt(1 - zeta**2))
                rise_time = (1.8 / omega_n)
                settling_time_2pct = 4 / (zeta * omega_n)
                peak_time = math.pi / omega_d
            else:
                overshoot = 0
                rise_time = 2.2 / omega_n
                settling_time_2pct = 4 / omega_n
                peak_time = None

            return {
                "natural_frequency": round(omega_n, 4),
                "damping_ratio": round(zeta, 4),
                "damped_frequency": round(omega_d, 4),
                "overshoot_percent": round(overshoot, 2),
                "rise_time": round(rise_time, 4),
                "settling_time_2pct": round(settling_time_2pct, 4),
                "peak_time": round(peak_time, 4) if peak_time else None,
                "system_type": "underdamped" if zeta < 1 else "critically_damped" if abs(zeta - 1) < 0.01 else "overdamped"
            }
        else:
            # Real poles
            tau = 1 / abs(dominant[0].real) if dominant[0].real != 0 else float('inf')
            return {
                "dominant_time_constant": round(tau, 4),
                "settling_time_2pct": round(4 * tau, 4),
                "rise_time": round(2.2 * tau, 4),
                "overshoot_percent": 0,
                "system_type": "overdamped_or_first_order"
            }


# Tool implementation for registry

def controls_tool(
    operation: str,
    # Transfer function parameters
    numerator: Optional[List[float]] = None,
    denominator: Optional[List[float]] = None,
    # PID parameters
    Kp: Optional[float] = None,
    Ki: Optional[float] = None,
    Kd: Optional[float] = None,
    # Tuning parameters
    Ku: Optional[float] = None,  # Ultimate gain
    Tu: Optional[float] = None,  # Ultimate period
    K: Optional[float] = None,   # Process gain
    tau: Optional[float] = None, # Time constant
    theta: Optional[float] = None, # Dead time
    # Analysis parameters
    omega_min: float = 0.01,
    omega_max: float = 100,
) -> ToolResult:
    """Execute controls analysis operation."""
    try:
        analyzer = ControlsAnalyzer()

        # Build transfer function if provided
        tf = None
        if numerator is not None and denominator is not None:
            tf = TransferFunction(numerator=numerator, denominator=denominator)

        if operation == "stability":
            if tf is None:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output="Need numerator and denominator for stability analysis"
                )

            result = analyzer.analyze_stability(tf)

            output = f"Transfer Function: {tf.to_string()}\n"
            output += f"System Order: {tf.order}\n"
            output += f"Stability: {result['status'].upper()}\n"
            output += f"Reason: {result['reason']}\n"
            output += f"Poles: {[p['formatted'] for p in result['poles']]}"

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=output,
                metadata=result
            )

        elif operation == "poles_zeros":
            if tf is None:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output="Need numerator and denominator"
                )

            poles = analyzer.get_poles(tf)
            zeros = analyzer.get_zeros(tf)

            def fmt(c):
                if abs(c.imag) < 1e-10:
                    return f"{c.real:.4g}"
                return f"{c.real:.4g} + {c.imag:.4g}j" if c.imag >= 0 else f"{c.real:.4g} - {abs(c.imag):.4g}j"

            output = f"Transfer Function: {tf.to_string()}\n"
            output += f"Poles: {[fmt(p) for p in poles]}\n"
            output += f"Zeros: {[fmt(z) for z in zeros]}"

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=output,
                metadata={
                    "poles": [{"real": p.real, "imag": p.imag} for p in poles],
                    "zeros": [{"real": z.real, "imag": z.imag} for z in zeros]
                }
            )

        elif operation == "margins":
            if tf is None:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output="Need numerator and denominator"
                )

            gm = analyzer.gain_margin(tf, (omega_min, omega_max))
            pm = analyzer.phase_margin(tf, (omega_min, omega_max))

            output = f"Transfer Function: {tf.to_string()}\n\n"
            output += f"GAIN MARGIN:\n"
            output += f"  GM = {gm['gain_margin_db']} dB ({gm['gain_margin_linear']} linear)\n"
            output += f"  Phase crossover: {gm['phase_crossover_freq']} rad/s\n"
            output += f"  {gm['interpretation']}\n\n"
            output += f"PHASE MARGIN:\n"
            output += f"  PM = {pm['phase_margin_deg']} deg\n"
            output += f"  Gain crossover: {pm['gain_crossover_freq']} rad/s\n"
            output += f"  {pm['interpretation']}"

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=output,
                metadata={"gain_margin": gm, "phase_margin": pm}
            )

        elif operation == "bode":
            if tf is None:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output="Need numerator and denominator"
                )

            data = analyzer.bode_data(tf, (omega_min, omega_max))

            output = f"Transfer Function: {tf.to_string()}\n"
            output += f"DC Gain: {data['dc_gain_db']:.2f} dB\n"
            output += f"Bandwidth (-3dB): {data['bandwidth_hz']:.4f} rad/s" if data['bandwidth_hz'] else "Bandwidth: Not found"
            output += f"\n\nBode data: {len(data['frequencies'])} points from {omega_min} to {omega_max} rad/s"

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=output,
                metadata=data
            )

        elif operation == "step_response":
            if tf is None:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output="Need numerator and denominator"
                )

            result = analyzer.step_response_characteristics(tf)

            if "error" in result:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    output=result["error"]
                )

            output = f"Transfer Function: {tf.to_string()}\n\n"
            output += f"Step Response Characteristics:\n"
            output += f"  System type: {result.get('system_type', 'unknown')}\n"
            if 'natural_frequency' in result:
                output += f"  Natural frequency (wn): {result['natural_frequency']} rad/s\n"
                output += f"  Damping ratio (zeta): {result['damping_ratio']}\n"
            output += f"  Rise time: {result['rise_time']} s\n"
            output += f"  Settling time (2%): {result['settling_time_2pct']} s\n"
            output += f"  Overshoot: {result['overshoot_percent']}%"

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=output,
                metadata=result
            )

        elif operation == "tune_pid_zn":
            if Ku is None or Tu is None:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output="Need Ku (ultimate gain) and Tu (ultimate period) for Ziegler-Nichols tuning"
                )

            pid = analyzer.tune_pid_ziegler_nichols(Ku, Tu)

            output = f"Ziegler-Nichols PID Tuning\n"
            output += f"Input: Ku = {Ku}, Tu = {Tu}\n\n"
            output += f"PID Parameters:\n"
            output += f"  Kp = {pid.Kp:.4g}\n"
            output += f"  Ki = {pid.Ki:.4g}\n"
            output += f"  Kd = {pid.Kd:.4g}\n\n"
            output += f"Transfer Function: C(s) = {pid.Kp:.4g} + {pid.Ki:.4g}/s + {pid.Kd:.4g}*s"

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=output,
                metadata=pid.to_dict()
            )

        elif operation == "tune_pid_cc":
            if K is None or tau is None or theta is None:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output="Need K (gain), tau (time constant), theta (dead time) for Cohen-Coon tuning"
                )

            pid = analyzer.tune_pid_cohen_coon(K, tau, theta)

            output = f"Cohen-Coon PID Tuning (FOPDT Model)\n"
            output += f"Input: K = {K}, tau = {tau}, theta = {theta}\n\n"
            output += f"PID Parameters:\n"
            output += f"  Kp = {pid.Kp:.4g}\n"
            output += f"  Ki = {pid.Ki:.4g}\n"
            output += f"  Kd = {pid.Kd:.4g}"

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=output,
                metadata=pid.to_dict()
            )

        elif operation == "pid_tf":
            if Kp is None:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output="Need at least Kp for PID transfer function"
                )

            pid = PIDController(Kp=Kp, Ki=Ki or 0, Kd=Kd or 0)
            tf = pid.transfer_function

            output = f"PID Controller: Kp={pid.Kp}, Ki={pid.Ki}, Kd={pid.Kd}\n"
            output += f"Transfer Function: {tf.to_string()}"

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=output,
                metadata={
                    "pid": pid.to_dict(),
                    "numerator": tf.numerator,
                    "denominator": tf.denominator
                }
            )

        else:
            return ToolResult(
                status=ToolStatus.INVALID_INPUT,
                output=f"Unknown operation: {operation}. Valid: stability, poles_zeros, margins, bode, step_response, tune_pid_zn, tune_pid_cc, pid_tf"
            )

    except Exception as e:
        return ToolResult(
            status=ToolStatus.ERROR,
            output=None,
            error=str(e)
        )


def create_controls_tool() -> Tool:
    """Create controls engineering tool."""
    return Tool(
        name="controls",
        description="Control systems analysis: transfer functions, stability (poles/zeros), gain/phase margins, PID tuning (Ziegler-Nichols, Cohen-Coon), step response characteristics",
        parameters=[
            ToolParameter(
                name="operation",
                description="stability, poles_zeros, margins, bode, step_response, tune_pid_zn, tune_pid_cc, pid_tf",
                type="string",
                required=True,
                enum=["stability", "poles_zeros", "margins", "bode", "step_response", "tune_pid_zn", "tune_pid_cc", "pid_tf"]
            ),
            ToolParameter(
                name="numerator",
                description="Numerator coefficients [a_n, ..., a_0] for transfer function",
                type="array",
                required=False,
            ),
            ToolParameter(
                name="denominator",
                description="Denominator coefficients [b_n, ..., b_0] for transfer function",
                type="array",
                required=False,
            ),
            ToolParameter(
                name="Kp",
                description="Proportional gain for PID",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="Ki",
                description="Integral gain for PID",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="Kd",
                description="Derivative gain for PID",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="Ku",
                description="Ultimate gain for Ziegler-Nichols tuning",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="Tu",
                description="Ultimate period for Ziegler-Nichols tuning",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="K",
                description="Process gain for Cohen-Coon tuning",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="tau",
                description="Time constant for Cohen-Coon tuning",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="theta",
                description="Dead time for Cohen-Coon tuning",
                type="number",
                required=False,
            ),
        ],
        execute_fn=controls_tool,
        timeout_ms=10000,
    )


def register_controls_tools(registry: ToolRegistry) -> None:
    """Register controls engineering tools."""
    registry.register(create_controls_tool())
