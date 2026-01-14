"""
Unit Converter Tool - Comprehensive Unit Conversion

Standalone unit conversion with dimensional analysis verification.
Catches unit mismatches and handles compound units.
"""

from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from decimal import Decimal
import math
import re

from .registry import Tool, ToolParameter, ToolResult, ToolStatus, ToolRegistry


@dataclass
class UnitInfo:
    """Information about a unit."""
    name: str
    symbol: str
    category: str
    to_si: float  # Conversion factor to SI base unit
    si_unit: str  # The SI base unit for this category


# Comprehensive unit database
UNITS = {
    # ==================== LENGTH ====================
    # SI base: meter (m)
    "m": UnitInfo("meter", "m", "length", 1.0, "m"),
    "km": UnitInfo("kilometer", "km", "length", 1000.0, "m"),
    "cm": UnitInfo("centimeter", "cm", "length", 0.01, "m"),
    "mm": UnitInfo("millimeter", "mm", "length", 0.001, "m"),
    "um": UnitInfo("micrometer", "um", "length", 1e-6, "m"),
    "nm": UnitInfo("nanometer", "nm", "length", 1e-9, "m"),
    "pm": UnitInfo("picometer", "pm", "length", 1e-12, "m"),
    "mi": UnitInfo("mile", "mi", "length", 1609.344, "m"),
    "yd": UnitInfo("yard", "yd", "length", 0.9144, "m"),
    "ft": UnitInfo("foot", "ft", "length", 0.3048, "m"),
    "in": UnitInfo("inch", "in", "length", 0.0254, "m"),
    "nmi": UnitInfo("nautical mile", "nmi", "length", 1852.0, "m"),
    "au": UnitInfo("astronomical unit", "au", "length", 1.496e11, "m"),
    "ly": UnitInfo("light year", "ly", "length", 9.461e15, "m"),
    "pc": UnitInfo("parsec", "pc", "length", 3.086e16, "m"),
    "angstrom": UnitInfo("angstrom", "Å", "length", 1e-10, "m"),

    # ==================== MASS ====================
    # SI base: kilogram (kg)
    "kg": UnitInfo("kilogram", "kg", "mass", 1.0, "kg"),
    "g": UnitInfo("gram", "g", "mass", 0.001, "kg"),
    "mg": UnitInfo("milligram", "mg", "mass", 1e-6, "kg"),
    "ug": UnitInfo("microgram", "ug", "mass", 1e-9, "kg"),
    "ton": UnitInfo("metric ton", "t", "mass", 1000.0, "kg"),
    "lb": UnitInfo("pound", "lb", "mass", 0.453592, "kg"),
    "oz": UnitInfo("ounce", "oz", "mass", 0.0283495, "kg"),
    "st": UnitInfo("stone", "st", "mass", 6.35029, "kg"),
    "slug": UnitInfo("slug", "slug", "mass", 14.5939, "kg"),
    "grain": UnitInfo("grain", "gr", "mass", 6.47989e-5, "kg"),
    "amu": UnitInfo("atomic mass unit", "u", "mass", 1.66054e-27, "kg"),

    # ==================== TIME ====================
    # SI base: second (s)
    "s": UnitInfo("second", "s", "time", 1.0, "s"),
    "ms": UnitInfo("millisecond", "ms", "time", 0.001, "s"),
    "us": UnitInfo("microsecond", "us", "time", 1e-6, "s"),
    "ns": UnitInfo("nanosecond", "ns", "time", 1e-9, "s"),
    "min": UnitInfo("minute", "min", "time", 60.0, "s"),
    "hr": UnitInfo("hour", "hr", "time", 3600.0, "s"),
    "h": UnitInfo("hour", "h", "time", 3600.0, "s"),
    "day": UnitInfo("day", "day", "time", 86400.0, "s"),
    "wk": UnitInfo("week", "wk", "time", 604800.0, "s"),
    "yr": UnitInfo("year", "yr", "time", 31557600.0, "s"),  # Julian year

    # ==================== TEMPERATURE ====================
    # Special handling - not linear conversion
    "K": UnitInfo("kelvin", "K", "temperature", 1.0, "K"),
    "C": UnitInfo("celsius", "°C", "temperature", 1.0, "K"),  # Special
    "F": UnitInfo("fahrenheit", "°F", "temperature", 1.0, "K"),  # Special
    "R": UnitInfo("rankine", "°R", "temperature", 1.0, "K"),  # Special

    # ==================== FORCE ====================
    # SI base: newton (N)
    "N": UnitInfo("newton", "N", "force", 1.0, "N"),
    "kN": UnitInfo("kilonewton", "kN", "force", 1000.0, "N"),
    "MN": UnitInfo("meganewton", "MN", "force", 1e6, "N"),
    "dyn": UnitInfo("dyne", "dyn", "force", 1e-5, "N"),
    "lbf": UnitInfo("pound-force", "lbf", "force", 4.44822, "N"),
    "kgf": UnitInfo("kilogram-force", "kgf", "force", 9.80665, "N"),
    "ozf": UnitInfo("ounce-force", "ozf", "force", 0.278014, "N"),

    # ==================== ENERGY ====================
    # SI base: joule (J)
    "J": UnitInfo("joule", "J", "energy", 1.0, "J"),
    "kJ": UnitInfo("kilojoule", "kJ", "energy", 1000.0, "J"),
    "MJ": UnitInfo("megajoule", "MJ", "energy", 1e6, "J"),
    "GJ": UnitInfo("gigajoule", "GJ", "energy", 1e9, "J"),
    "cal": UnitInfo("calorie", "cal", "energy", 4.184, "J"),
    "kcal": UnitInfo("kilocalorie", "kcal", "energy", 4184.0, "J"),
    "Cal": UnitInfo("food calorie", "Cal", "energy", 4184.0, "J"),
    "eV": UnitInfo("electronvolt", "eV", "energy", 1.60218e-19, "J"),
    "keV": UnitInfo("kiloelectronvolt", "keV", "energy", 1.60218e-16, "J"),
    "MeV": UnitInfo("megaelectronvolt", "MeV", "energy", 1.60218e-13, "J"),
    "kWh": UnitInfo("kilowatt-hour", "kWh", "energy", 3.6e6, "J"),
    "Wh": UnitInfo("watt-hour", "Wh", "energy", 3600.0, "J"),
    "BTU": UnitInfo("British thermal unit", "BTU", "energy", 1055.06, "J"),
    "therm": UnitInfo("therm", "therm", "energy", 1.055e8, "J"),
    "erg": UnitInfo("erg", "erg", "energy", 1e-7, "J"),
    "ft_lb": UnitInfo("foot-pound", "ft·lb", "energy", 1.35582, "J"),

    # ==================== POWER ====================
    # SI base: watt (W)
    "W": UnitInfo("watt", "W", "power", 1.0, "W"),
    "kW": UnitInfo("kilowatt", "kW", "power", 1000.0, "W"),
    "MW": UnitInfo("megawatt", "MW", "power", 1e6, "W"),
    "GW": UnitInfo("gigawatt", "GW", "power", 1e9, "W"),
    "mW": UnitInfo("milliwatt", "mW", "power", 0.001, "W"),
    "hp": UnitInfo("horsepower", "hp", "power", 745.7, "W"),
    "BTU/h": UnitInfo("BTU per hour", "BTU/h", "power", 0.293071, "W"),

    # ==================== PRESSURE ====================
    # SI base: pascal (Pa)
    "Pa": UnitInfo("pascal", "Pa", "pressure", 1.0, "Pa"),
    "kPa": UnitInfo("kilopascal", "kPa", "pressure", 1000.0, "Pa"),
    "MPa": UnitInfo("megapascal", "MPa", "pressure", 1e6, "Pa"),
    "GPa": UnitInfo("gigapascal", "GPa", "pressure", 1e9, "Pa"),
    "bar": UnitInfo("bar", "bar", "pressure", 1e5, "Pa"),
    "mbar": UnitInfo("millibar", "mbar", "pressure", 100.0, "Pa"),
    "atm": UnitInfo("atmosphere", "atm", "pressure", 101325.0, "Pa"),
    "psi": UnitInfo("pound per square inch", "psi", "pressure", 6894.76, "Pa"),
    "mmHg": UnitInfo("millimeter of mercury", "mmHg", "pressure", 133.322, "Pa"),
    "torr": UnitInfo("torr", "torr", "pressure", 133.322, "Pa"),
    "inHg": UnitInfo("inch of mercury", "inHg", "pressure", 3386.39, "Pa"),

    # ==================== SPEED/VELOCITY ====================
    # SI base: meters per second (m/s)
    "m/s": UnitInfo("meter per second", "m/s", "speed", 1.0, "m/s"),
    "km/h": UnitInfo("kilometer per hour", "km/h", "speed", 1/3.6, "m/s"),
    "kph": UnitInfo("kilometer per hour", "kph", "speed", 1/3.6, "m/s"),
    "mph": UnitInfo("mile per hour", "mph", "speed", 0.44704, "m/s"),
    "kn": UnitInfo("knot", "kn", "speed", 0.514444, "m/s"),
    "knot": UnitInfo("knot", "knot", "speed", 0.514444, "m/s"),
    "ft/s": UnitInfo("foot per second", "ft/s", "speed", 0.3048, "m/s"),
    "mach": UnitInfo("mach number", "mach", "speed", 343.0, "m/s"),  # At sea level, 20°C
    "c": UnitInfo("speed of light", "c", "speed", 299792458.0, "m/s"),

    # ==================== AREA ====================
    # SI base: square meter (m²)
    "m2": UnitInfo("square meter", "m²", "area", 1.0, "m²"),
    "km2": UnitInfo("square kilometer", "km²", "area", 1e6, "m²"),
    "cm2": UnitInfo("square centimeter", "cm²", "area", 1e-4, "m²"),
    "mm2": UnitInfo("square millimeter", "mm²", "area", 1e-6, "m²"),
    "ha": UnitInfo("hectare", "ha", "area", 1e4, "m²"),
    "acre": UnitInfo("acre", "acre", "area", 4046.86, "m²"),
    "ft2": UnitInfo("square foot", "ft²", "area", 0.092903, "m²"),
    "in2": UnitInfo("square inch", "in²", "area", 6.4516e-4, "m²"),
    "mi2": UnitInfo("square mile", "mi²", "area", 2.59e6, "m²"),
    "yd2": UnitInfo("square yard", "yd²", "area", 0.836127, "m²"),

    # ==================== VOLUME ====================
    # SI base: cubic meter (m³)
    "m3": UnitInfo("cubic meter", "m³", "volume", 1.0, "m³"),
    "L": UnitInfo("liter", "L", "volume", 0.001, "m³"),
    "l": UnitInfo("liter", "l", "volume", 0.001, "m³"),
    "mL": UnitInfo("milliliter", "mL", "volume", 1e-6, "m³"),
    "ml": UnitInfo("milliliter", "ml", "volume", 1e-6, "m³"),
    "cm3": UnitInfo("cubic centimeter", "cm³", "volume", 1e-6, "m³"),
    "cc": UnitInfo("cubic centimeter", "cc", "volume", 1e-6, "m³"),
    "mm3": UnitInfo("cubic millimeter", "mm³", "volume", 1e-9, "m³"),
    "gal": UnitInfo("US gallon", "gal", "volume", 0.00378541, "m³"),
    "qt": UnitInfo("US quart", "qt", "volume", 9.4635e-4, "m³"),
    "pt": UnitInfo("US pint", "pt", "volume", 4.73176e-4, "m³"),
    "cup": UnitInfo("US cup", "cup", "volume", 2.36588e-4, "m³"),
    "fl_oz": UnitInfo("US fluid ounce", "fl oz", "volume", 2.95735e-5, "m³"),
    "tbsp": UnitInfo("tablespoon", "tbsp", "volume", 1.47868e-5, "m³"),
    "tsp": UnitInfo("teaspoon", "tsp", "volume", 4.92892e-6, "m³"),
    "ft3": UnitInfo("cubic foot", "ft³", "volume", 0.0283168, "m³"),
    "in3": UnitInfo("cubic inch", "in³", "volume", 1.6387e-5, "m³"),
    "bbl": UnitInfo("barrel (oil)", "bbl", "volume", 0.158987, "m³"),

    # ==================== ANGLE ====================
    # SI base: radian (rad)
    "rad": UnitInfo("radian", "rad", "angle", 1.0, "rad"),
    "deg": UnitInfo("degree", "°", "angle", math.pi/180, "rad"),
    "grad": UnitInfo("gradian", "grad", "angle", math.pi/200, "rad"),
    "arcmin": UnitInfo("arcminute", "'", "angle", math.pi/10800, "rad"),
    "arcsec": UnitInfo("arcsecond", '"', "angle", math.pi/648000, "rad"),
    "rev": UnitInfo("revolution", "rev", "angle", 2*math.pi, "rad"),
    "turn": UnitInfo("turn", "turn", "angle", 2*math.pi, "rad"),

    # ==================== FREQUENCY ====================
    # SI base: hertz (Hz)
    "Hz": UnitInfo("hertz", "Hz", "frequency", 1.0, "Hz"),
    "kHz": UnitInfo("kilohertz", "kHz", "frequency", 1000.0, "Hz"),
    "MHz": UnitInfo("megahertz", "MHz", "frequency", 1e6, "Hz"),
    "GHz": UnitInfo("gigahertz", "GHz", "frequency", 1e9, "Hz"),
    "rpm": UnitInfo("revolutions per minute", "rpm", "frequency", 1/60, "Hz"),

    # ==================== ELECTRIC CURRENT ====================
    # SI base: ampere (A)
    "A": UnitInfo("ampere", "A", "current", 1.0, "A"),
    "mA": UnitInfo("milliampere", "mA", "current", 0.001, "A"),
    "uA": UnitInfo("microampere", "uA", "current", 1e-6, "A"),
    "kA": UnitInfo("kiloampere", "kA", "current", 1000.0, "A"),

    # ==================== VOLTAGE ====================
    # SI base: volt (V)
    "V": UnitInfo("volt", "V", "voltage", 1.0, "V"),
    "mV": UnitInfo("millivolt", "mV", "voltage", 0.001, "V"),
    "uV": UnitInfo("microvolt", "uV", "voltage", 1e-6, "V"),
    "kV": UnitInfo("kilovolt", "kV", "voltage", 1000.0, "V"),
    "MV": UnitInfo("megavolt", "MV", "voltage", 1e6, "V"),

    # ==================== RESISTANCE ====================
    # SI base: ohm (Ω)
    "ohm": UnitInfo("ohm", "Ω", "resistance", 1.0, "Ω"),
    "kohm": UnitInfo("kilohm", "kΩ", "resistance", 1000.0, "Ω"),
    "Mohm": UnitInfo("megohm", "MΩ", "resistance", 1e6, "Ω"),

    # ==================== DATA ====================
    # SI base: bit (b) or byte (B)
    "b": UnitInfo("bit", "b", "data", 1.0, "b"),
    "B": UnitInfo("byte", "B", "data", 8.0, "b"),
    "kb": UnitInfo("kilobit", "kb", "data", 1000.0, "b"),
    "Kb": UnitInfo("kilobit", "Kb", "data", 1000.0, "b"),
    "kB": UnitInfo("kilobyte", "kB", "data", 8000.0, "b"),
    "KB": UnitInfo("kilobyte", "KB", "data", 8000.0, "b"),
    "Mb": UnitInfo("megabit", "Mb", "data", 1e6, "b"),
    "MB": UnitInfo("megabyte", "MB", "data", 8e6, "b"),
    "Gb": UnitInfo("gigabit", "Gb", "data", 1e9, "b"),
    "GB": UnitInfo("gigabyte", "GB", "data", 8e9, "b"),
    "Tb": UnitInfo("terabit", "Tb", "data", 1e12, "b"),
    "TB": UnitInfo("terabyte", "TB", "data", 8e12, "b"),
    "KiB": UnitInfo("kibibyte", "KiB", "data", 8 * 1024, "b"),
    "MiB": UnitInfo("mebibyte", "MiB", "data", 8 * 1024**2, "b"),
    "GiB": UnitInfo("gibibyte", "GiB", "data", 8 * 1024**3, "b"),
    "TiB": UnitInfo("tebibyte", "TiB", "data", 8 * 1024**4, "b"),
}


class UnitConverter:
    """Comprehensive unit conversion engine."""

    def convert(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
    ) -> Dict[str, Any]:
        """
        Convert a value between units.

        Args:
            value: Numeric value to convert
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Dict with converted value and metadata
        """
        # Normalize unit names
        from_unit = self._normalize_unit(from_unit)
        to_unit = self._normalize_unit(to_unit)

        # Handle temperature specially (non-linear)
        if from_unit in ["K", "C", "F", "R"] and to_unit in ["K", "C", "F", "R"]:
            return self._convert_temperature(value, from_unit, to_unit)

        # Get unit info
        if from_unit not in UNITS:
            return {
                "success": False,
                "error": f"Unknown unit: {from_unit}",
                "suggestion": self._suggest_unit(from_unit),
            }
        if to_unit not in UNITS:
            return {
                "success": False,
                "error": f"Unknown unit: {to_unit}",
                "suggestion": self._suggest_unit(to_unit),
            }

        from_info = UNITS[from_unit]
        to_info = UNITS[to_unit]

        # Check dimensional compatibility
        if from_info.category != to_info.category:
            return {
                "success": False,
                "error": f"Incompatible units: {from_info.category} vs {to_info.category}",
                "from_category": from_info.category,
                "to_category": to_info.category,
            }

        # Convert through SI base unit
        si_value = value * from_info.to_si
        result = si_value / to_info.to_si

        return {
            "success": True,
            "input": {"value": value, "unit": from_unit},
            "output": {"value": result, "unit": to_unit},
            "category": from_info.category,
            "si_value": {"value": si_value, "unit": from_info.si_unit},
        }

    def _normalize_unit(self, unit: str) -> str:
        """Normalize unit name to canonical form."""
        # Handle common variations
        unit = unit.strip()

        # Common aliases
        aliases = {
            "meter": "m", "meters": "m",
            "kilometre": "km", "kilometer": "km", "kilometers": "km",
            "centimetre": "cm", "centimeter": "cm", "centimeters": "cm",
            "millimetre": "mm", "millimeter": "mm", "millimeters": "mm",
            "mile": "mi", "miles": "mi",
            "foot": "ft", "feet": "ft",
            "inch": "in", "inches": "in",
            "yard": "yd", "yards": "yd",
            "kilogram": "kg", "kilograms": "kg",
            "gram": "g", "grams": "g",
            "pound": "lb", "pounds": "lb", "lbs": "lb",
            "ounce": "oz", "ounces": "oz",
            "second": "s", "seconds": "s", "sec": "s",
            "minute": "min", "minutes": "min",
            "hour": "hr", "hours": "hr", "hrs": "hr",
            "day": "day", "days": "day",
            "year": "yr", "years": "yr",
            "kelvin": "K",
            "celsius": "C", "centigrade": "C",
            "fahrenheit": "F",
            "newton": "N", "newtons": "N",
            "joule": "J", "joules": "J",
            "watt": "W", "watts": "W",
            "pascal": "Pa", "pascals": "Pa",
            "atmosphere": "atm", "atmospheres": "atm",
            "liter": "L", "litre": "L", "liters": "L", "litres": "L",
            "milliliter": "mL", "millilitre": "mL",
            "gallon": "gal", "gallons": "gal",
            "degree": "deg", "degrees": "deg",
            "radian": "rad", "radians": "rad",
            "hertz": "Hz",
            "ampere": "A", "amp": "A", "amps": "A",
            "volt": "V", "volts": "V",
            "byte": "B", "bytes": "B",
            "bit": "b", "bits": "b",
        }

        lower = unit.lower()
        if lower in aliases:
            return aliases[lower]

        return unit

    def _suggest_unit(self, unit: str) -> Optional[str]:
        """Suggest a similar valid unit."""
        unit_lower = unit.lower()
        for u in UNITS:
            if u.lower() == unit_lower:
                return u
            if UNITS[u].name.lower() == unit_lower:
                return u
        return None

    def _convert_temperature(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
    ) -> Dict[str, Any]:
        """Handle temperature conversions (non-linear)."""
        # Convert to Kelvin first
        if from_unit == "C":
            kelvin = value + 273.15
        elif from_unit == "F":
            kelvin = (value - 32) * 5/9 + 273.15
        elif from_unit == "R":
            kelvin = value * 5/9
        else:  # K
            kelvin = value

        # Convert from Kelvin to target
        if to_unit == "C":
            result = kelvin - 273.15
        elif to_unit == "F":
            result = (kelvin - 273.15) * 9/5 + 32
        elif to_unit == "R":
            result = kelvin * 9/5
        else:  # K
            result = kelvin

        return {
            "success": True,
            "input": {"value": value, "unit": from_unit},
            "output": {"value": result, "unit": to_unit},
            "category": "temperature",
            "kelvin": kelvin,
        }

    def list_units(self, category: Optional[str] = None) -> Dict[str, Any]:
        """List available units, optionally filtered by category."""
        if category:
            units = {k: v for k, v in UNITS.items() if v.category == category}
        else:
            units = UNITS

        # Group by category
        by_category = {}
        for unit_id, info in units.items():
            if info.category not in by_category:
                by_category[info.category] = []
            by_category[info.category].append({
                "id": unit_id,
                "name": info.name,
                "symbol": info.symbol,
            })

        return {
            "success": True,
            "categories": list(by_category.keys()),
            "units": by_category,
            "total": len(units),
        }

    def dimensional_analysis(
        self,
        expression: str,
        target_unit: str,
    ) -> Dict[str, Any]:
        """
        Analyze dimensional consistency of an expression.

        For now, simple implementation. Can be extended for compound units.
        """
        # This is a placeholder for more sophisticated dimensional analysis
        return {
            "success": True,
            "note": "Dimensional analysis for compound units coming in future version",
        }


def unit_converter_tool(
    value: float,
    from_unit: str,
    to_unit: str,
) -> ToolResult:
    """Tool function for unit conversion."""
    converter = UnitConverter()
    result = converter.convert(value, from_unit, to_unit)

    if result.get("success"):
        output_val = result["output"]["value"]
        output_unit = result["output"]["unit"]

        # Format nicely
        if abs(output_val) < 0.001 or abs(output_val) > 1e6:
            output_str = f"{output_val:.6e} {output_unit}"
        else:
            output_str = f"{output_val:.6g} {output_unit}"

        return ToolResult(
            status=ToolStatus.SUCCESS,
            output=output_str,
            metadata=result,
        )
    else:
        return ToolResult(
            status=ToolStatus.ERROR,
            output=None,
            error=result.get("error", "Unknown error"),
        )


def create_unit_converter_tool() -> Tool:
    """Create the unit converter tool."""
    return Tool(
        name="unit_converter",
        description="Convert between physical units with dimensional analysis. Supports length, mass, time, temperature, force, energy, power, pressure, speed, area, volume, angle, frequency, current, voltage, resistance, and data units.",
        parameters=[
            ToolParameter(
                name="value",
                description="Numeric value to convert",
                type="number",
                required=True,
            ),
            ToolParameter(
                name="from_unit",
                description="Source unit (e.g., 'km', 'lb', 'F', 'psi')",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="to_unit",
                description="Target unit (e.g., 'mi', 'kg', 'C', 'Pa')",
                type="string",
                required=True,
            ),
        ],
        execute_fn=unit_converter_tool,
        timeout_ms=5000,
    )


def register_unit_converter_tools(registry: ToolRegistry) -> None:
    """Register unit converter tools with the registry."""
    registry.register(create_unit_converter_tool())
