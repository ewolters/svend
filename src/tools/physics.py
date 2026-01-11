"""
Physics Specialist Tool

Provides physics calculations:
- Unit conversions
- Kinematics equations
- Thermodynamics
- Electricity basics
- Wave mechanics

Uses Pint for unit handling when available, with fallback to basic implementation.
"""

from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
import math

from .specialists import SpecialistTool, ToolOperation, ToolResult, ToolCategory


# Physical constants
CONSTANTS = {
    "c": {"value": 299792458, "unit": "m/s", "name": "Speed of light"},
    "G": {"value": 6.67430e-11, "unit": "m^3/(kg*s^2)", "name": "Gravitational constant"},
    "h": {"value": 6.62607e-34, "unit": "J*s", "name": "Planck constant"},
    "hbar": {"value": 1.054572e-34, "unit": "J*s", "name": "Reduced Planck constant"},
    "e": {"value": 1.602176e-19, "unit": "C", "name": "Elementary charge"},
    "k_B": {"value": 1.380649e-23, "unit": "J/K", "name": "Boltzmann constant"},
    "N_A": {"value": 6.02214e23, "unit": "1/mol", "name": "Avogadro constant"},
    "R": {"value": 8.314462, "unit": "J/(mol*K)", "name": "Gas constant"},
    "g": {"value": 9.80665, "unit": "m/s^2", "name": "Standard gravity"},
    "epsilon_0": {"value": 8.854188e-12, "unit": "F/m", "name": "Vacuum permittivity"},
    "mu_0": {"value": 1.256637e-6, "unit": "H/m", "name": "Vacuum permeability"},
    "m_e": {"value": 9.109384e-31, "unit": "kg", "name": "Electron mass"},
    "m_p": {"value": 1.672622e-27, "unit": "kg", "name": "Proton mass"},
    "sigma": {"value": 5.670374e-8, "unit": "W/(m^2*K^4)", "name": "Stefan-Boltzmann constant"},
}

# Unit conversion factors (to SI base units)
UNIT_CONVERSIONS = {
    # Length
    "m": 1.0,
    "km": 1000.0,
    "cm": 0.01,
    "mm": 0.001,
    "um": 1e-6,
    "nm": 1e-9,
    "mi": 1609.344,
    "ft": 0.3048,
    "in": 0.0254,
    "yd": 0.9144,
    "au": 1.496e11,
    "ly": 9.461e15,

    # Mass
    "kg": 1.0,
    "g": 0.001,
    "mg": 1e-6,
    "lb": 0.453592,
    "oz": 0.0283495,
    "ton": 1000.0,

    # Time
    "s": 1.0,
    "ms": 0.001,
    "us": 1e-6,
    "ns": 1e-9,
    "min": 60.0,
    "hr": 3600.0,
    "day": 86400.0,
    "yr": 31557600.0,

    # Temperature (special handling needed)
    "K": 1.0,  # Kelvin is base
    "C": "celsius",  # Special case
    "F": "fahrenheit",  # Special case

    # Force
    "N": 1.0,
    "kN": 1000.0,
    "dyn": 1e-5,
    "lbf": 4.44822,

    # Energy
    "J": 1.0,
    "kJ": 1000.0,
    "cal": 4.184,
    "kcal": 4184.0,
    "eV": 1.602176e-19,
    "kWh": 3.6e6,
    "BTU": 1055.06,

    # Power
    "W": 1.0,
    "kW": 1000.0,
    "MW": 1e6,
    "hp": 745.7,

    # Pressure
    "Pa": 1.0,
    "kPa": 1000.0,
    "MPa": 1e6,
    "bar": 1e5,
    "atm": 101325.0,
    "psi": 6894.76,
    "mmHg": 133.322,
    "torr": 133.322,

    # Speed
    "m/s": 1.0,
    "km/h": 1/3.6,
    "mph": 0.44704,
    "kn": 0.514444,

    # Angle
    "rad": 1.0,
    "deg": math.pi/180,
    "rev": 2*math.pi,
}


def convert_units(
    value: float,
    from_unit: str,
    to_unit: str,
) -> Dict[str, Any]:
    """
    Convert a value between units.

    Handles temperature specially (non-linear conversion).
    """
    # Handle temperature specially
    if from_unit in ["C", "F", "K"] and to_unit in ["C", "F", "K"]:
        return _convert_temperature(value, from_unit, to_unit)

    # Check units exist
    if from_unit not in UNIT_CONVERSIONS:
        return {"success": False, "error": f"Unknown unit: {from_unit}"}
    if to_unit not in UNIT_CONVERSIONS:
        return {"success": False, "error": f"Unknown unit: {to_unit}"}

    # Convert through SI base
    si_value = value * UNIT_CONVERSIONS[from_unit]
    result = si_value / UNIT_CONVERSIONS[to_unit]

    return {
        "success": True,
        "original": {"value": value, "unit": from_unit},
        "result": {"value": result, "unit": to_unit},
    }


def _convert_temperature(value: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
    """Convert temperature between C, F, K."""
    # Convert to Kelvin first
    if from_unit == "C":
        kelvin = value + 273.15
    elif from_unit == "F":
        kelvin = (value - 32) * 5/9 + 273.15
    else:  # K
        kelvin = value

    # Convert from Kelvin to target
    if to_unit == "C":
        result = kelvin - 273.15
    elif to_unit == "F":
        result = (kelvin - 273.15) * 9/5 + 32
    else:  # K
        result = kelvin

    return {
        "success": True,
        "original": {"value": value, "unit": from_unit},
        "result": {"value": result, "unit": to_unit},
    }


def kinematics(
    known: Dict[str, float],
    find: str,
) -> Dict[str, Any]:
    """
    Solve kinematics equations.

    Variables:
    - v0: initial velocity (m/s)
    - v: final velocity (m/s)
    - a: acceleration (m/s^2)
    - t: time (s)
    - d or x: displacement (m)

    Provide at least 3 known values, find the others.
    """
    # Normalize variable names
    if "d" in known:
        known["x"] = known.pop("d")

    # Extract known values
    v0 = known.get("v0")
    v = known.get("v")
    a = known.get("a")
    t = known.get("t")
    x = known.get("x")

    # Count known values
    known_count = sum(1 for val in [v0, v, a, t, x] if val is not None)

    if known_count < 3:
        return {
            "success": False,
            "error": "Need at least 3 known values",
            "provided": list(known.keys()),
        }

    result = {"success": True}

    try:
        # Solve based on what's known
        if find == "v" and v is None:
            if v0 is not None and a is not None and t is not None:
                result["v"] = v0 + a * t
                result["equation"] = "v = v0 + at"
            elif v0 is not None and a is not None and x is not None:
                result["v"] = math.sqrt(v0**2 + 2*a*x)
                result["equation"] = "v² = v0² + 2ax"
            else:
                return {"success": False, "error": "Cannot solve for v with given values"}

        elif find == "v0" and v0 is None:
            if v is not None and a is not None and t is not None:
                result["v0"] = v - a * t
                result["equation"] = "v0 = v - at"
            elif v is not None and a is not None and x is not None:
                result["v0"] = math.sqrt(v**2 - 2*a*x)
                result["equation"] = "v0² = v² - 2ax"
            else:
                return {"success": False, "error": "Cannot solve for v0 with given values"}

        elif find == "a" and a is None:
            if v0 is not None and v is not None and t is not None:
                result["a"] = (v - v0) / t
                result["equation"] = "a = (v - v0) / t"
            elif v0 is not None and v is not None and x is not None and x != 0:
                result["a"] = (v**2 - v0**2) / (2 * x)
                result["equation"] = "a = (v² - v0²) / (2x)"
            else:
                return {"success": False, "error": "Cannot solve for a with given values"}

        elif find == "t" and t is None:
            if v0 is not None and v is not None and a is not None and a != 0:
                result["t"] = (v - v0) / a
                result["equation"] = "t = (v - v0) / a"
            elif v0 is not None and x is not None and a is not None:
                # Quadratic: x = v0*t + 0.5*a*t^2
                discriminant = v0**2 + 2*a*x
                if discriminant < 0:
                    return {"success": False, "error": "No real solution for time"}
                t1 = (-v0 + math.sqrt(discriminant)) / a
                t2 = (-v0 - math.sqrt(discriminant)) / a
                result["t"] = max(t1, t2) if t1 > 0 or t2 > 0 else t1
                result["equation"] = "x = v0*t + ½at² (solved for t)"
            else:
                return {"success": False, "error": "Cannot solve for t with given values"}

        elif find in ["x", "d"] and x is None:
            if v0 is not None and v is not None and t is not None:
                result["x"] = (v0 + v) / 2 * t
                result["equation"] = "x = (v0 + v)/2 × t"
            elif v0 is not None and a is not None and t is not None:
                result["x"] = v0 * t + 0.5 * a * t**2
                result["equation"] = "x = v0×t + ½at²"
            elif v0 is not None and v is not None and a is not None and a != 0:
                result["x"] = (v**2 - v0**2) / (2 * a)
                result["equation"] = "x = (v² - v0²) / (2a)"
            else:
                return {"success": False, "error": "Cannot solve for displacement with given values"}

        else:
            return {"success": False, "error": f"Cannot find '{find}' or already known"}

        result["known_values"] = known

    except Exception as e:
        return {"success": False, "error": str(e)}

    return result


def ideal_gas_law(
    known: Dict[str, float],
    find: str,
) -> Dict[str, Any]:
    """
    Solve ideal gas law: PV = nRT

    Variables:
    - P: pressure (Pa)
    - V: volume (m³)
    - n: moles
    - T: temperature (K)

    R = 8.314462 J/(mol·K)
    """
    R = 8.314462

    P = known.get("P")
    V = known.get("V")
    n = known.get("n")
    T = known.get("T")

    result = {"success": True, "equation": "PV = nRT", "R": R}

    try:
        if find == "P" and P is None:
            if V is not None and n is not None and T is not None:
                result["P"] = n * R * T / V
                result["unit"] = "Pa"
            else:
                return {"success": False, "error": "Need V, n, and T to find P"}

        elif find == "V" and V is None:
            if P is not None and n is not None and T is not None:
                result["V"] = n * R * T / P
                result["unit"] = "m³"
            else:
                return {"success": False, "error": "Need P, n, and T to find V"}

        elif find == "n" and n is None:
            if P is not None and V is not None and T is not None:
                result["n"] = P * V / (R * T)
                result["unit"] = "mol"
            else:
                return {"success": False, "error": "Need P, V, and T to find n"}

        elif find == "T" and T is None:
            if P is not None and V is not None and n is not None:
                result["T"] = P * V / (n * R)
                result["unit"] = "K"
            else:
                return {"success": False, "error": "Need P, V, and n to find T"}

        else:
            return {"success": False, "error": f"Cannot find '{find}'"}

    except Exception as e:
        return {"success": False, "error": str(e)}

    return result


def energy_work(
    operation: str,
    **kwargs,
) -> Dict[str, Any]:
    """
    Energy and work calculations.

    Operations:
    - kinetic_energy: KE = ½mv²
    - potential_energy: PE = mgh (gravitational) or ½kx² (spring)
    - work: W = Fd cos(θ)
    - power: P = W/t or P = Fv
    - conservation: Total energy in closed system
    """
    result = {"success": True}

    try:
        if operation == "kinetic_energy":
            m = kwargs.get("m")  # mass (kg)
            v = kwargs.get("v")  # velocity (m/s)
            KE = kwargs.get("KE")  # kinetic energy (J)

            if m is not None and v is not None:
                result["KE"] = 0.5 * m * v**2
                result["equation"] = "KE = ½mv²"
            elif KE is not None and m is not None:
                result["v"] = math.sqrt(2 * KE / m)
                result["equation"] = "v = √(2KE/m)"
            elif KE is not None and v is not None:
                result["m"] = 2 * KE / v**2
                result["equation"] = "m = 2KE/v²"
            else:
                return {"success": False, "error": "Need 2 of: m, v, KE"}

        elif operation == "potential_energy":
            pe_type = kwargs.get("type", "gravitational")

            if pe_type == "gravitational":
                m = kwargs.get("m")
                g = kwargs.get("g", CONSTANTS["g"]["value"])
                h = kwargs.get("h")
                PE = kwargs.get("PE")

                if m is not None and h is not None:
                    result["PE"] = m * g * h
                    result["equation"] = "PE = mgh"
                elif PE is not None and m is not None:
                    result["h"] = PE / (m * g)
                    result["equation"] = "h = PE/(mg)"
                elif PE is not None and h is not None:
                    result["m"] = PE / (g * h)
                    result["equation"] = "m = PE/(gh)"
                else:
                    return {"success": False, "error": "Need 2 of: m, h, PE"}

            elif pe_type == "spring":
                k = kwargs.get("k")  # spring constant (N/m)
                x = kwargs.get("x")  # displacement (m)
                PE = kwargs.get("PE")

                if k is not None and x is not None:
                    result["PE"] = 0.5 * k * x**2
                    result["equation"] = "PE = ½kx²"
                elif PE is not None and k is not None:
                    result["x"] = math.sqrt(2 * PE / k)
                    result["equation"] = "x = √(2PE/k)"
                else:
                    return {"success": False, "error": "Need 2 of: k, x, PE"}

        elif operation == "work":
            F = kwargs.get("F")  # force (N)
            d = kwargs.get("d")  # distance (m)
            theta = kwargs.get("theta", 0)  # angle in degrees
            W = kwargs.get("W")

            theta_rad = theta * math.pi / 180

            if F is not None and d is not None:
                result["W"] = F * d * math.cos(theta_rad)
                result["equation"] = "W = Fd cos(θ)"
            elif W is not None and d is not None:
                result["F"] = W / (d * math.cos(theta_rad))
                result["equation"] = "F = W/(d cos(θ))"
            else:
                return {"success": False, "error": "Need F and d, or W and d"}

        elif operation == "power":
            W = kwargs.get("W")  # work (J)
            t = kwargs.get("t")  # time (s)
            F = kwargs.get("F")  # force (N)
            v = kwargs.get("v")  # velocity (m/s)
            P = kwargs.get("P")  # power (W)

            if W is not None and t is not None:
                result["P"] = W / t
                result["equation"] = "P = W/t"
            elif F is not None and v is not None:
                result["P"] = F * v
                result["equation"] = "P = Fv"
            elif P is not None and t is not None:
                result["W"] = P * t
                result["equation"] = "W = Pt"
            else:
                return {"success": False, "error": "Need W and t, or F and v"}

        else:
            return {"success": False, "error": f"Unknown energy operation: {operation}"}

    except Exception as e:
        return {"success": False, "error": str(e)}

    return result


def electricity(
    operation: str,
    **kwargs,
) -> Dict[str, Any]:
    """
    Electricity and circuit calculations.

    Operations:
    - ohms_law: V = IR
    - power_electric: P = IV = I²R = V²/R
    - series_resistance: R_total = R1 + R2 + ...
    - parallel_resistance: 1/R_total = 1/R1 + 1/R2 + ...
    - capacitance: C = Q/V, series/parallel
    - coulombs_law: F = kq1q2/r²
    """
    result = {"success": True}
    k_e = 8.99e9  # Coulomb's constant (N⋅m²/C²)

    try:
        if operation == "ohms_law":
            V = kwargs.get("V")  # voltage (V)
            I = kwargs.get("I")  # current (A)
            R = kwargs.get("R")  # resistance (Ω)

            if V is not None and I is not None:
                result["R"] = V / I
                result["equation"] = "R = V/I"
            elif V is not None and R is not None:
                result["I"] = V / R
                result["equation"] = "I = V/R"
            elif I is not None and R is not None:
                result["V"] = I * R
                result["equation"] = "V = IR"
            else:
                return {"success": False, "error": "Need 2 of: V, I, R"}

        elif operation == "power_electric":
            P = kwargs.get("P")  # power (W)
            V = kwargs.get("V")  # voltage (V)
            I = kwargs.get("I")  # current (A)
            R = kwargs.get("R")  # resistance (Ω)

            if I is not None and V is not None:
                result["P"] = I * V
                result["equation"] = "P = IV"
            elif I is not None and R is not None:
                result["P"] = I**2 * R
                result["equation"] = "P = I²R"
            elif V is not None and R is not None:
                result["P"] = V**2 / R
                result["equation"] = "P = V²/R"
            elif P is not None and V is not None:
                result["I"] = P / V
                result["equation"] = "I = P/V"
            elif P is not None and I is not None:
                result["V"] = P / I
                result["equation"] = "V = P/I"
            else:
                return {"success": False, "error": "Need 2 of: P, V, I, R"}

        elif operation == "series_resistance":
            resistances = kwargs.get("resistances", [])
            if not resistances:
                return {"success": False, "error": "Need list of resistances"}
            result["R_total"] = sum(resistances)
            result["equation"] = "R_total = R1 + R2 + ..."

        elif operation == "parallel_resistance":
            resistances = kwargs.get("resistances", [])
            if not resistances:
                return {"success": False, "error": "Need list of resistances"}
            result["R_total"] = 1 / sum(1/r for r in resistances)
            result["equation"] = "1/R_total = 1/R1 + 1/R2 + ..."

        elif operation == "capacitance":
            Q = kwargs.get("Q")  # charge (C)
            V = kwargs.get("V")  # voltage (V)
            C = kwargs.get("C")  # capacitance (F)

            if Q is not None and V is not None:
                result["C"] = Q / V
                result["equation"] = "C = Q/V"
            elif C is not None and V is not None:
                result["Q"] = C * V
                result["equation"] = "Q = CV"
            elif C is not None and Q is not None:
                result["V"] = Q / C
                result["equation"] = "V = Q/C"
            else:
                return {"success": False, "error": "Need 2 of: Q, V, C"}

        elif operation == "coulombs_law":
            q1 = kwargs.get("q1")  # charge 1 (C)
            q2 = kwargs.get("q2")  # charge 2 (C)
            r = kwargs.get("r")   # distance (m)
            F = kwargs.get("F")   # force (N)

            if q1 is not None and q2 is not None and r is not None:
                result["F"] = k_e * abs(q1 * q2) / r**2
                result["direction"] = "attractive" if q1 * q2 < 0 else "repulsive"
                result["equation"] = "F = k|q1q2|/r²"
            elif F is not None and q1 is not None and q2 is not None:
                result["r"] = math.sqrt(k_e * abs(q1 * q2) / F)
                result["equation"] = "r = √(k|q1q2|/F)"
            else:
                return {"success": False, "error": "Need q1, q2, and r (or F)"}

        else:
            return {"success": False, "error": f"Unknown electricity operation: {operation}"}

    except Exception as e:
        return {"success": False, "error": str(e)}

    return result


def optics(
    operation: str,
    **kwargs,
) -> Dict[str, Any]:
    """
    Optics calculations.

    Operations:
    - thin_lens: 1/f = 1/do + 1/di (lens equation)
    - magnification: M = -di/do = hi/ho
    - snells_law: n1 sin(θ1) = n2 sin(θ2)
    - critical_angle: θc = arcsin(n2/n1)
    """
    result = {"success": True}

    try:
        if operation == "thin_lens":
            f = kwargs.get("f")    # focal length (m)
            do = kwargs.get("do")  # object distance (m)
            di = kwargs.get("di")  # image distance (m)

            if do is not None and di is not None:
                result["f"] = 1 / (1/do + 1/di)
                result["equation"] = "1/f = 1/do + 1/di"
            elif f is not None and do is not None:
                result["di"] = 1 / (1/f - 1/do)
                result["image_type"] = "real" if result["di"] > 0 else "virtual"
                result["equation"] = "di = 1/(1/f - 1/do)"
            elif f is not None and di is not None:
                result["do"] = 1 / (1/f - 1/di)
                result["equation"] = "do = 1/(1/f - 1/di)"
            else:
                return {"success": False, "error": "Need 2 of: f, do, di"}

            # Calculate magnification if we have distances
            if "di" in result and do is not None:
                result["magnification"] = -result["di"] / do
            elif di is not None and do is not None:
                result["magnification"] = -di / do

        elif operation == "magnification":
            di = kwargs.get("di")
            do = kwargs.get("do")
            hi = kwargs.get("hi")  # image height
            ho = kwargs.get("ho")  # object height
            M = kwargs.get("M")

            if di is not None and do is not None:
                result["M"] = -di / do
                result["equation"] = "M = -di/do"
            elif hi is not None and ho is not None:
                result["M"] = hi / ho
                result["equation"] = "M = hi/ho"
            elif M is not None and ho is not None:
                result["hi"] = M * ho
                result["equation"] = "hi = M × ho"
            else:
                return {"success": False, "error": "Need (di, do) or (hi, ho) or (M, ho)"}

            if "M" in result:
                result["image_orientation"] = "inverted" if result["M"] < 0 else "upright"
                result["image_size"] = "magnified" if abs(result["M"]) > 1 else "reduced"

        elif operation == "snells_law":
            n1 = kwargs.get("n1")  # refractive index 1
            n2 = kwargs.get("n2")  # refractive index 2
            theta1 = kwargs.get("theta1")  # angle 1 (degrees)
            theta2 = kwargs.get("theta2")  # angle 2 (degrees)

            if n1 is not None and theta1 is not None and n2 is not None:
                sin_theta2 = n1 * math.sin(math.radians(theta1)) / n2
                if abs(sin_theta2) > 1:
                    result["total_internal_reflection"] = True
                    result["critical_angle"] = math.degrees(math.asin(n2/n1))
                else:
                    result["theta2"] = math.degrees(math.asin(sin_theta2))
                result["equation"] = "n1 sin(θ1) = n2 sin(θ2)"
            elif n1 is not None and n2 is not None and theta2 is not None:
                result["theta1"] = math.degrees(math.asin(n2 * math.sin(math.radians(theta2)) / n1))
                result["equation"] = "θ1 = arcsin(n2 sin(θ2) / n1)"
            else:
                return {"success": False, "error": "Need n1, n2, and one angle"}

        elif operation == "critical_angle":
            n1 = kwargs.get("n1")  # denser medium
            n2 = kwargs.get("n2")  # less dense medium

            if n1 is not None and n2 is not None:
                if n1 <= n2:
                    return {"success": False, "error": "n1 must be greater than n2 for TIR"}
                result["critical_angle"] = math.degrees(math.asin(n2/n1))
                result["equation"] = "θc = arcsin(n2/n1)"
            else:
                return {"success": False, "error": "Need both n1 and n2"}

        else:
            return {"success": False, "error": f"Unknown optics operation: {operation}"}

    except Exception as e:
        return {"success": False, "error": str(e)}

    return result


def projectile_motion(
    **kwargs,
) -> Dict[str, Any]:
    """
    Projectile motion calculations.

    Given initial velocity and angle, calculates:
    - Maximum height
    - Range (horizontal distance)
    - Time of flight
    - Velocity at any time
    """
    result = {"success": True}

    try:
        v0 = kwargs.get("v0")  # initial velocity (m/s)
        theta = kwargs.get("theta")  # launch angle (degrees)
        g = kwargs.get("g", CONSTANTS["g"]["value"])
        h0 = kwargs.get("h0", 0)  # initial height (m)

        if v0 is None or theta is None:
            return {"success": False, "error": "Need initial velocity (v0) and angle (theta)"}

        theta_rad = math.radians(theta)
        v0x = v0 * math.cos(theta_rad)
        v0y = v0 * math.sin(theta_rad)

        # Time of flight (solving y = h0 + v0y*t - 0.5*g*t² = 0)
        discriminant = v0y**2 + 2*g*h0
        if discriminant < 0:
            return {"success": False, "error": "Invalid trajectory (negative discriminant)"}

        t_flight = (v0y + math.sqrt(discriminant)) / g

        # Maximum height
        t_max_height = v0y / g
        max_height = h0 + v0y * t_max_height - 0.5 * g * t_max_height**2

        # Range
        range_x = v0x * t_flight

        result["initial_velocity"] = {"vx": v0x, "vy": v0y, "total": v0}
        result["time_of_flight"] = t_flight
        result["max_height"] = max_height
        result["range"] = range_x
        result["time_to_max_height"] = t_max_height

        # Final velocity
        vfy = v0y - g * t_flight
        result["final_velocity"] = {
            "vx": v0x,
            "vy": vfy,
            "total": math.sqrt(v0x**2 + vfy**2),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

    return result


def simple_harmonic_motion(
    **kwargs,
) -> Dict[str, Any]:
    """
    Simple harmonic motion (SHM) calculations.

    For mass-spring or pendulum systems.
    """
    result = {"success": True}

    try:
        system_type = kwargs.get("type", "spring")

        if system_type == "spring":
            m = kwargs.get("m")  # mass (kg)
            k = kwargs.get("k")  # spring constant (N/m)
            A = kwargs.get("A")  # amplitude (m)

            if m is not None and k is not None:
                omega = math.sqrt(k/m)  # angular frequency
                T = 2 * math.pi / omega  # period
                f = 1 / T  # frequency

                result["angular_frequency"] = omega
                result["period"] = T
                result["frequency"] = f
                result["equations"] = {
                    "position": "x(t) = A cos(ωt + φ)",
                    "velocity": "v(t) = -Aω sin(ωt + φ)",
                    "acceleration": "a(t) = -Aω² cos(ωt + φ)",
                }

                if A is not None:
                    result["max_velocity"] = A * omega
                    result["max_acceleration"] = A * omega**2
            else:
                return {"success": False, "error": "Need mass (m) and spring constant (k)"}

        elif system_type == "pendulum":
            L = kwargs.get("L")  # length (m)
            g = kwargs.get("g", CONSTANTS["g"]["value"])

            if L is not None:
                T = 2 * math.pi * math.sqrt(L/g)
                f = 1 / T
                omega = 2 * math.pi * f

                result["period"] = T
                result["frequency"] = f
                result["angular_frequency"] = omega
                result["equation"] = "T = 2π√(L/g)"
            else:
                return {"success": False, "error": "Need pendulum length (L)"}

        else:
            return {"success": False, "error": f"Unknown SHM type: {system_type}"}

    except Exception as e:
        return {"success": False, "error": str(e)}

    return result


def wave_mechanics(
    operation: str,
    **kwargs,
) -> Dict[str, Any]:
    """
    Wave mechanics calculations.

    Operations:
    - frequency_wavelength: v = f × λ
    - wave_energy: E = hf (photon energy)
    - doppler: frequency shift
    """
    result = {"success": True}

    try:
        if operation == "frequency_wavelength":
            # v = f × λ, solve for unknown
            v = kwargs.get("v", CONSTANTS["c"]["value"])  # Default to speed of light
            f = kwargs.get("f")
            wavelength = kwargs.get("wavelength") or kwargs.get("lambda")

            if f is not None and wavelength is not None:
                result["v"] = f * wavelength
                result["equation"] = "v = f × λ"
            elif v is not None and wavelength is not None:
                result["f"] = v / wavelength
                result["equation"] = "f = v / λ"
            elif v is not None and f is not None:
                result["wavelength"] = v / f
                result["equation"] = "λ = v / f"
            else:
                return {"success": False, "error": "Need at least 2 of: v, f, wavelength"}

        elif operation == "wave_energy":
            # E = hf or E = hc/λ
            h = CONSTANTS["h"]["value"]
            c = CONSTANTS["c"]["value"]

            f = kwargs.get("f")
            wavelength = kwargs.get("wavelength") or kwargs.get("lambda")
            E = kwargs.get("E")

            if f is not None:
                result["E"] = h * f
                result["E_eV"] = result["E"] / CONSTANTS["e"]["value"]
                result["equation"] = "E = hf"
            elif wavelength is not None:
                result["E"] = h * c / wavelength
                result["E_eV"] = result["E"] / CONSTANTS["e"]["value"]
                result["equation"] = "E = hc/λ"
            elif E is not None:
                result["f"] = E / h
                result["wavelength"] = h * c / E
                result["equation"] = "f = E/h, λ = hc/E"
            else:
                return {"success": False, "error": "Need f, wavelength, or E"}

        elif operation == "doppler":
            # f_observed = f_source × (v ± v_observer) / (v ∓ v_source)
            v = kwargs.get("v", 343)  # Default to speed of sound
            f_source = kwargs.get("f_source")
            v_observer = kwargs.get("v_observer", 0)
            v_source = kwargs.get("v_source", 0)
            approaching = kwargs.get("approaching", True)

            if f_source is None:
                return {"success": False, "error": "Need source frequency (f_source)"}

            if approaching:
                f_observed = f_source * (v + v_observer) / (v - v_source)
            else:
                f_observed = f_source * (v - v_observer) / (v + v_source)

            result["f_observed"] = f_observed
            result["shift"] = f_observed - f_source
            result["approaching"] = approaching

        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    except Exception as e:
        return {"success": False, "error": str(e)}

    return result


class PhysicsTool(SpecialistTool):
    """
    Physics specialist tool.

    Provides physics calculations for mechanics, thermodynamics, waves, etc.
    """

    name = "physics"
    category = ToolCategory.PHYSICS
    description = "Physics calculations: kinematics, unit conversion, thermodynamics, waves"
    version = "1.0.0"

    def _register_operations(self):
        """Register physics operations."""

        self.register_operation(ToolOperation(
            name="unit_convert",
            description="Convert between physical units",
            parameters={
                "value": {
                    "type": "number",
                    "description": "Value to convert",
                    "required": True,
                },
                "from_unit": {
                    "type": "string",
                    "description": "Source unit (e.g., 'm', 'km', 'ft', 'C', 'F')",
                    "required": True,
                },
                "to_unit": {
                    "type": "string",
                    "description": "Target unit",
                    "required": True,
                },
            },
            returns="Converted value",
            examples=[
                {"value": 100, "from_unit": "km/h", "to_unit": "m/s"},
                {"value": 32, "from_unit": "F", "to_unit": "C"},
            ],
        ))

        self.register_operation(ToolOperation(
            name="kinematics",
            description="Solve kinematics equations for motion",
            parameters={
                "known": {
                    "type": "object",
                    "description": "Known values: v0 (initial velocity), v (final velocity), a (acceleration), t (time), x or d (displacement)",
                    "required": True,
                },
                "find": {
                    "type": "string",
                    "description": "Variable to solve for: v0, v, a, t, x",
                    "required": True,
                },
            },
            returns="Calculated value with equation used",
        ))

        self.register_operation(ToolOperation(
            name="ideal_gas",
            description="Solve ideal gas law PV = nRT",
            parameters={
                "known": {
                    "type": "object",
                    "description": "Known values: P (pressure Pa), V (volume m³), n (moles), T (temperature K)",
                    "required": True,
                },
                "find": {
                    "type": "string",
                    "description": "Variable to solve for: P, V, n, or T",
                    "required": True,
                },
            },
            returns="Calculated value",
        ))

        self.register_operation(ToolOperation(
            name="waves",
            description="Wave mechanics calculations",
            parameters={
                "operation": {
                    "type": "string",
                    "description": "Operation: 'frequency_wavelength', 'wave_energy', 'doppler'",
                    "required": True,
                },
                "params": {
                    "type": "object",
                    "description": "Parameters for the operation (v, f, wavelength, E, etc.)",
                    "required": True,
                },
            },
            returns="Calculated wave properties",
        ))

        self.register_operation(ToolOperation(
            name="constant",
            description="Get physical constant value",
            parameters={
                "name": {
                    "type": "string",
                    "description": "Constant name: c, G, h, e, k_B, N_A, R, g, etc.",
                    "required": True,
                },
            },
            returns="Constant value, unit, and description",
        ))

        self.register_operation(ToolOperation(
            name="energy",
            description="Energy and work calculations (kinetic, potential, work, power)",
            parameters={
                "operation": {
                    "type": "string",
                    "description": "Operation: 'kinetic_energy', 'potential_energy', 'work', 'power'",
                    "required": True,
                },
                "params": {
                    "type": "object",
                    "description": "Parameters (m, v, h, k, x, F, d, theta, W, t, P, etc.)",
                    "required": True,
                },
            },
            returns="Calculated energy/work/power values",
        ))

        self.register_operation(ToolOperation(
            name="electricity",
            description="Electric circuit calculations (Ohm's law, power, resistance, capacitance, Coulomb's law)",
            parameters={
                "operation": {
                    "type": "string",
                    "description": "Operation: 'ohms_law', 'power_electric', 'series_resistance', 'parallel_resistance', 'capacitance', 'coulombs_law'",
                    "required": True,
                },
                "params": {
                    "type": "object",
                    "description": "Parameters (V, I, R, P, Q, C, q1, q2, r, resistances, etc.)",
                    "required": True,
                },
            },
            returns="Calculated electrical values",
        ))

        self.register_operation(ToolOperation(
            name="optics",
            description="Optics calculations (lens equation, magnification, Snell's law, critical angle)",
            parameters={
                "operation": {
                    "type": "string",
                    "description": "Operation: 'thin_lens', 'magnification', 'snells_law', 'critical_angle'",
                    "required": True,
                },
                "params": {
                    "type": "object",
                    "description": "Parameters (f, do, di, hi, ho, M, n1, n2, theta1, theta2, etc.)",
                    "required": True,
                },
            },
            returns="Calculated optical values",
        ))

        self.register_operation(ToolOperation(
            name="projectile",
            description="Projectile motion calculations",
            parameters={
                "v0": {
                    "type": "number",
                    "description": "Initial velocity (m/s)",
                    "required": True,
                },
                "theta": {
                    "type": "number",
                    "description": "Launch angle (degrees from horizontal)",
                    "required": True,
                },
                "h0": {
                    "type": "number",
                    "description": "Initial height (m), default 0",
                    "required": False,
                },
            },
            returns="Range, max height, time of flight, velocities",
        ))

        self.register_operation(ToolOperation(
            name="shm",
            description="Simple harmonic motion (spring or pendulum)",
            parameters={
                "type": {
                    "type": "string",
                    "description": "System type: 'spring' or 'pendulum'",
                    "required": True,
                },
                "params": {
                    "type": "object",
                    "description": "For spring: m (mass), k (spring constant), A (amplitude). For pendulum: L (length)",
                    "required": True,
                },
            },
            returns="Period, frequency, angular frequency, max velocity/acceleration",
        ))

    def _execute_operation(self, operation: str, **kwargs) -> Any:
        """Execute a physics operation."""

        if operation == "unit_convert":
            return convert_units(
                kwargs["value"],
                kwargs["from_unit"],
                kwargs["to_unit"],
            )

        elif operation == "kinematics":
            return kinematics(kwargs["known"], kwargs["find"])

        elif operation == "ideal_gas":
            return ideal_gas_law(kwargs["known"], kwargs["find"])

        elif operation == "waves":
            params = kwargs.get("params", {})
            return wave_mechanics(kwargs["operation"], **params)

        elif operation == "constant":
            name = kwargs["name"]
            if name in CONSTANTS:
                return {
                    "success": True,
                    "name": name,
                    **CONSTANTS[name],
                }
            else:
                return {
                    "success": False,
                    "error": f"Unknown constant: {name}",
                    "available": list(CONSTANTS.keys()),
                }

        elif operation == "energy":
            params = kwargs.get("params", {})
            op = kwargs.get("operation", params.get("operation"))
            return energy_work(op, **params)

        elif operation == "electricity":
            params = kwargs.get("params", {})
            op = kwargs.get("operation", params.get("operation"))
            return electricity(op, **params)

        elif operation == "optics":
            params = kwargs.get("params", {})
            op = kwargs.get("operation", params.get("operation"))
            return optics(op, **params)

        elif operation == "projectile":
            return projectile_motion(**kwargs)

        elif operation == "shm":
            params = kwargs.get("params", {})
            params["type"] = kwargs.get("type", params.get("type", "spring"))
            return simple_harmonic_motion(**params)

        else:
            return ToolResult(
                success=False,
                error=f"Unknown operation: {operation}",
            )


def create_physics_tool() -> PhysicsTool:
    """Factory function to create physics tool."""
    return PhysicsTool()
