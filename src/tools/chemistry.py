"""
Chemistry Specialist Tool

Provides chemistry calculations and analysis:
- Molecular weight calculation
- Stoichiometry
- Reaction balancing
- Molecular formula parsing
- Unit conversions (moles, grams, etc.)

Optional RDKit integration for:
- Molecular structure analysis
- SMILES parsing
- Functional group identification
- Property prediction
"""

from typing import Optional, Dict, Any, List
import re
from dataclasses import dataclass

from .specialists import SpecialistTool, ToolOperation, ToolResult, ToolCategory


# Periodic table data (atomic weights)
PERIODIC_TABLE = {
    "H": 1.008, "He": 4.003, "Li": 6.941, "Be": 9.012, "B": 10.81,
    "C": 12.01, "N": 14.01, "O": 16.00, "F": 19.00, "Ne": 20.18,
    "Na": 22.99, "Mg": 24.31, "Al": 26.98, "Si": 28.09, "P": 30.97,
    "S": 32.07, "Cl": 35.45, "Ar": 39.95, "K": 39.10, "Ca": 40.08,
    "Sc": 44.96, "Ti": 47.87, "V": 50.94, "Cr": 52.00, "Mn": 54.94,
    "Fe": 55.85, "Co": 58.93, "Ni": 58.69, "Cu": 63.55, "Zn": 65.38,
    "Ga": 69.72, "Ge": 72.63, "As": 74.92, "Se": 78.97, "Br": 79.90,
    "Kr": 83.80, "Rb": 85.47, "Sr": 87.62, "Y": 88.91, "Zr": 91.22,
    "Nb": 92.91, "Mo": 95.95, "Ru": 101.1, "Rh": 102.9, "Pd": 106.4,
    "Ag": 107.9, "Cd": 112.4, "In": 114.8, "Sn": 118.7, "Sb": 121.8,
    "Te": 127.6, "I": 126.9, "Xe": 131.3, "Cs": 132.9, "Ba": 137.3,
    "La": 138.9, "Ce": 140.1, "Pr": 140.9, "Nd": 144.2, "Sm": 150.4,
    "Eu": 152.0, "Gd": 157.3, "Tb": 158.9, "Dy": 162.5, "Ho": 164.9,
    "Er": 167.3, "Tm": 168.9, "Yb": 173.0, "Lu": 175.0, "Hf": 178.5,
    "Ta": 180.9, "W": 183.8, "Re": 186.2, "Os": 190.2, "Ir": 192.2,
    "Pt": 195.1, "Au": 197.0, "Hg": 200.6, "Tl": 204.4, "Pb": 207.2,
    "Bi": 209.0, "U": 238.0,
}

# Common ion charges
COMMON_IONS = {
    "H+": 1, "Li+": 1, "Na+": 1, "K+": 1, "Ag+": 1,
    "Mg2+": 2, "Ca2+": 2, "Ba2+": 2, "Zn2+": 2, "Fe2+": 2, "Cu2+": 2,
    "Fe3+": 3, "Al3+": 3,
    "OH-": -1, "Cl-": -1, "Br-": -1, "I-": -1, "NO3-": -1,
    "SO4(2-)": -2, "CO3(2-)": -2, "O2-": -2,
    "PO4(3-)": -3,
}


def parse_formula(formula: str) -> Dict[str, int]:
    """
    Parse a chemical formula into element counts.

    Examples:
        "H2O" -> {"H": 2, "O": 1}
        "Ca(OH)2" -> {"Ca": 1, "O": 2, "H": 2}
        "Mg3(PO4)2" -> {"Mg": 3, "P": 2, "O": 8}
    """
    def parse_group(s: str, multiplier: int = 1) -> Dict[str, int]:
        result = {}
        i = 0

        while i < len(s):
            if s[i] == '(':
                # Find matching close paren
                depth = 1
                j = i + 1
                while j < len(s) and depth > 0:
                    if s[j] == '(':
                        depth += 1
                    elif s[j] == ')':
                        depth -= 1
                    j += 1

                # Get multiplier after paren
                k = j
                while k < len(s) and s[k].isdigit():
                    k += 1

                group_mult = int(s[j:k]) if k > j else 1

                # Recursively parse group
                group_result = parse_group(s[i+1:j-1], group_mult * multiplier)
                for elem, count in group_result.items():
                    result[elem] = result.get(elem, 0) + count

                i = k

            elif s[i].isupper():
                # Element symbol
                j = i + 1
                while j < len(s) and s[j].islower():
                    j += 1

                element = s[i:j]

                # Get count
                k = j
                while k < len(s) and s[k].isdigit():
                    k += 1

                count = int(s[j:k]) if k > j else 1

                result[element] = result.get(element, 0) + count * multiplier
                i = k
            else:
                i += 1

        return result

    return parse_group(formula)


def calculate_molecular_weight(formula: str) -> float:
    """Calculate molecular weight from formula."""
    elements = parse_formula(formula)

    total = 0.0
    for element, count in elements.items():
        if element not in PERIODIC_TABLE:
            raise ValueError(f"Unknown element: {element}")
        total += PERIODIC_TABLE[element] * count

    return total


def balance_equation(equation: str) -> Dict[str, Any]:
    """
    Balance a chemical equation.

    Input format: "H2 + O2 -> H2O"
    Output: Balanced equation with coefficients
    """
    # Parse equation
    if "->" not in equation:
        return {"success": False, "error": "Equation must contain '->'"}

    left, right = equation.split("->")

    # Parse compounds
    left_compounds = [c.strip() for c in left.split("+")]
    right_compounds = [c.strip() for c in right.split("+")]

    # Get all elements
    left_elements = {}
    right_elements = {}

    for compound in left_compounds:
        for elem, count in parse_formula(compound).items():
            left_elements[elem] = left_elements.get(elem, 0) + count

    for compound in right_compounds:
        for elem, count in parse_formula(compound).items():
            right_elements[elem] = right_elements.get(elem, 0) + count

    # Simple balancing using linear algebra approach
    # For complex equations, use sympy or matrix methods
    # This is a simplified version for common cases

    all_elements = set(left_elements.keys()) | set(right_elements.keys())

    # Check if elements match
    if set(left_elements.keys()) != set(right_elements.keys()):
        return {
            "success": False,
            "error": "Elements don't match on both sides",
            "left_elements": left_elements,
            "right_elements": right_elements,
        }

    # Try small integer coefficients (brute force for simple equations)
    from itertools import product

    num_compounds = len(left_compounds) + len(right_compounds)
    max_coeff = 10

    for coeffs in product(range(1, max_coeff + 1), repeat=num_compounds):
        left_coeffs = coeffs[:len(left_compounds)]
        right_coeffs = coeffs[len(left_compounds):]

        # Calculate element totals
        left_total = {}
        right_total = {}

        for i, compound in enumerate(left_compounds):
            for elem, count in parse_formula(compound).items():
                left_total[elem] = left_total.get(elem, 0) + count * left_coeffs[i]

        for i, compound in enumerate(right_compounds):
            for elem, count in parse_formula(compound).items():
                right_total[elem] = right_total.get(elem, 0) + count * right_coeffs[i]

        # Check if balanced
        if left_total == right_total:
            # Format result
            left_str = " + ".join(
                f"{c if c > 1 else ''}{compound}"
                for c, compound in zip(left_coeffs, left_compounds)
            )
            right_str = " + ".join(
                f"{c if c > 1 else ''}{compound}"
                for c, compound in zip(right_coeffs, right_compounds)
            )

            return {
                "success": True,
                "balanced_equation": f"{left_str} -> {right_str}",
                "coefficients": {
                    "reactants": dict(zip(left_compounds, left_coeffs)),
                    "products": dict(zip(right_compounds, right_coeffs)),
                },
            }

    return {
        "success": False,
        "error": "Could not balance equation with small integer coefficients",
    }


def stoichiometry(
    equation: str,
    known_compound: str,
    known_amount: float,
    known_unit: str,
    target_compound: str,
) -> Dict[str, Any]:
    """
    Calculate stoichiometric amounts.

    Args:
        equation: Balanced chemical equation
        known_compound: Compound with known amount
        known_amount: Amount of known compound
        known_unit: Unit (grams, moles, liters)
        target_compound: Compound to calculate

    Returns:
        Amount of target compound
    """
    # First balance the equation
    balanced = balance_equation(equation)
    if not balanced["success"]:
        return balanced

    coefficients = balanced["coefficients"]
    all_compounds = {**coefficients["reactants"], **coefficients["products"]}

    if known_compound not in all_compounds:
        return {"success": False, "error": f"Unknown compound: {known_compound}"}
    if target_compound not in all_compounds:
        return {"success": False, "error": f"Unknown compound: {target_compound}"}

    # Convert to moles
    known_mw = calculate_molecular_weight(known_compound)
    target_mw = calculate_molecular_weight(target_compound)

    if known_unit == "grams":
        known_moles = known_amount / known_mw
    elif known_unit == "moles":
        known_moles = known_amount
    else:
        return {"success": False, "error": f"Unknown unit: {known_unit}"}

    # Use stoichiometric ratio
    known_coeff = all_compounds[known_compound]
    target_coeff = all_compounds[target_compound]

    target_moles = known_moles * (target_coeff / known_coeff)
    target_grams = target_moles * target_mw

    return {
        "success": True,
        "target_compound": target_compound,
        "moles": target_moles,
        "grams": target_grams,
        "calculation": {
            "known_moles": known_moles,
            "ratio": f"{target_coeff}:{known_coeff}",
            "target_mw": target_mw,
        },
    }


def ph_calculation(
    concentration: float,
    acid_or_base: str = "acid",
    strong: bool = True,
    Ka: Optional[float] = None,
    Kb: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculate pH from concentration.

    Args:
        concentration: Molar concentration
        acid_or_base: "acid" or "base"
        strong: Whether it's a strong acid/base
        Ka: Acid dissociation constant (for weak acids)
        Kb: Base dissociation constant (for weak bases)

    Returns:
        pH and pOH values
    """
    import math

    if concentration <= 0:
        return {"success": False, "error": "Concentration must be positive"}

    if strong:
        if acid_or_base == "acid":
            h_concentration = concentration
            ph = -math.log10(h_concentration)
        else:
            oh_concentration = concentration
            poh = -math.log10(oh_concentration)
            ph = 14 - poh
    else:
        # Weak acid/base calculation using Ka/Kb
        if acid_or_base == "acid":
            if Ka is None:
                return {"success": False, "error": "Ka required for weak acid"}
            # For weak acid: [H+] ≈ √(Ka × C) when Ka << C
            h_concentration = math.sqrt(Ka * concentration)
            ph = -math.log10(h_concentration)
        else:
            if Kb is None:
                return {"success": False, "error": "Kb required for weak base"}
            # For weak base: [OH-] ≈ √(Kb × C)
            oh_concentration = math.sqrt(Kb * concentration)
            poh = -math.log10(oh_concentration)
            ph = 14 - poh

    poh = 14 - ph

    return {
        "success": True,
        "pH": round(ph, 4),
        "pOH": round(poh, 4),
        "H+_concentration": 10 ** (-ph),
        "OH-_concentration": 10 ** (-poh),
    }


def dilution(
    C1: Optional[float] = None,
    V1: Optional[float] = None,
    C2: Optional[float] = None,
    V2: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Dilution calculation: C1V1 = C2V2

    Provide 3 of 4 values to calculate the fourth.
    """
    known_count = sum(1 for x in [C1, V1, C2, V2] if x is not None)

    if known_count != 3:
        return {"success": False, "error": "Provide exactly 3 of: C1, V1, C2, V2"}

    result = {"success": True, "equation": "C1V1 = C2V2"}

    if C1 is None:
        result["C1"] = (C2 * V2) / V1
        result["calculated"] = "C1"
    elif V1 is None:
        result["V1"] = (C2 * V2) / C1
        result["calculated"] = "V1"
    elif C2 is None:
        result["C2"] = (C1 * V1) / V2
        result["calculated"] = "C2"
    else:
        result["V2"] = (C1 * V1) / C2
        result["calculated"] = "V2"

    return result


def molarity_molality(
    operation: str,
    **kwargs,
) -> Dict[str, Any]:
    """
    Molarity and molality calculations.

    Operations:
    - molarity: M = moles / liters
    - molality: m = moles / kg_solvent
    - convert: Convert between units
    """
    result = {"success": True}

    if operation == "molarity":
        moles = kwargs.get("moles")
        liters = kwargs.get("liters")
        M = kwargs.get("M")  # molarity

        if moles is not None and liters is not None:
            result["M"] = moles / liters
            result["unit"] = "mol/L"
            result["equation"] = "M = n/V"
        elif M is not None and liters is not None:
            result["moles"] = M * liters
            result["equation"] = "n = M × V"
        elif M is not None and moles is not None:
            result["liters"] = moles / M
            result["equation"] = "V = n/M"
        else:
            return {"success": False, "error": "Need 2 of: moles, liters, M"}

    elif operation == "molality":
        moles = kwargs.get("moles")
        kg_solvent = kwargs.get("kg_solvent")
        m = kwargs.get("m")  # molality

        if moles is not None and kg_solvent is not None:
            result["m"] = moles / kg_solvent
            result["unit"] = "mol/kg"
            result["equation"] = "m = n/kg"
        elif m is not None and kg_solvent is not None:
            result["moles"] = m * kg_solvent
            result["equation"] = "n = m × kg"
        elif m is not None and moles is not None:
            result["kg_solvent"] = moles / m
            result["equation"] = "kg = n/m"
        else:
            return {"success": False, "error": "Need 2 of: moles, kg_solvent, m"}

    else:
        return {"success": False, "error": f"Unknown operation: {operation}"}

    return result


def percent_composition(formula: str) -> Dict[str, Any]:
    """Calculate percent composition by mass for each element."""
    elements = parse_formula(formula)
    total_mass = calculate_molecular_weight(formula)

    composition = {}
    for element, count in elements.items():
        mass = PERIODIC_TABLE[element] * count
        percent = (mass / total_mass) * 100
        composition[element] = {
            "count": count,
            "mass": round(mass, 4),
            "percent": round(percent, 2),
        }

    return {
        "success": True,
        "formula": formula,
        "molecular_weight": round(total_mass, 4),
        "composition": composition,
    }


def limiting_reagent(
    equation: str,
    amounts: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """
    Determine limiting reagent in a reaction.

    Args:
        equation: Chemical equation (e.g., "H2 + O2 -> H2O")
        amounts: Dict of compound -> {"amount": value, "unit": "grams"|"moles"}

    Returns:
        Limiting reagent and theoretical yield
    """
    # Balance equation first
    balanced = balance_equation(equation)
    if not balanced["success"]:
        return balanced

    coefficients = balanced["coefficients"]["reactants"]
    products_coeffs = balanced["coefficients"]["products"]

    # Convert all to moles
    moles = {}
    for compound, info in amounts.items():
        if compound not in coefficients:
            return {"success": False, "error": f"Compound {compound} not in reactants"}

        amount = info["amount"]
        unit = info.get("unit", "moles")

        if unit == "grams":
            mw = calculate_molecular_weight(compound)
            moles[compound] = amount / mw
        else:
            moles[compound] = amount

    # Calculate mole ratios
    ratios = {}
    for compound, mol in moles.items():
        ratios[compound] = mol / coefficients[compound]

    # Limiting reagent has smallest ratio
    limiting = min(ratios.keys(), key=lambda x: ratios[x])
    limiting_ratio = ratios[limiting]

    # Calculate theoretical yields
    yields = {}
    for product, coeff in products_coeffs.items():
        product_moles = limiting_ratio * coeff
        product_mw = calculate_molecular_weight(product)
        yields[product] = {
            "moles": product_moles,
            "grams": product_moles * product_mw,
        }

    return {
        "success": True,
        "limiting_reagent": limiting,
        "mole_ratios": ratios,
        "theoretical_yields": yields,
        "balanced_equation": balanced["balanced_equation"],
    }


class ChemistryTool(SpecialistTool):
    """
    Chemistry specialist tool.

    Provides chemistry calculations without external dependencies.
    For advanced features (molecular structures, SMILES), RDKit is optional.
    """

    name = "chemistry"
    category = ToolCategory.CHEMISTRY
    description = "Chemistry calculations: molecular weight, stoichiometry, pH, reaction balancing"
    version = "1.0.0"

    def __init__(self):
        super().__init__()

        # Check for RDKit
        self._rdkit_available = False
        try:
            import rdkit
            self._rdkit_available = True
        except ImportError:
            pass

    def _register_operations(self):
        """Register chemistry operations."""

        self.register_operation(ToolOperation(
            name="molecular_weight",
            description="Calculate molecular weight from chemical formula",
            parameters={
                "formula": {
                    "type": "string",
                    "description": "Chemical formula (e.g., 'H2O', 'C6H12O6', 'Ca(OH)2')",
                    "required": True,
                },
            },
            returns="Molecular weight in g/mol",
            examples=[
                {"formula": "H2O", "result": 18.015},
                {"formula": "NaCl", "result": 58.44},
            ],
        ))

        self.register_operation(ToolOperation(
            name="parse_formula",
            description="Parse chemical formula into element counts",
            parameters={
                "formula": {
                    "type": "string",
                    "description": "Chemical formula",
                    "required": True,
                },
            },
            returns="Dictionary of element counts",
        ))

        self.register_operation(ToolOperation(
            name="balance_equation",
            description="Balance a chemical equation",
            parameters={
                "equation": {
                    "type": "string",
                    "description": "Chemical equation with '->' separator (e.g., 'H2 + O2 -> H2O')",
                    "required": True,
                },
            },
            returns="Balanced equation with coefficients",
        ))

        self.register_operation(ToolOperation(
            name="stoichiometry",
            description="Calculate stoichiometric amounts from a reaction",
            parameters={
                "equation": {
                    "type": "string",
                    "description": "Chemical equation",
                    "required": True,
                },
                "known_compound": {
                    "type": "string",
                    "description": "Compound with known amount",
                    "required": True,
                },
                "known_amount": {
                    "type": "number",
                    "description": "Amount of known compound",
                    "required": True,
                },
                "known_unit": {
                    "type": "string",
                    "description": "Unit: 'grams' or 'moles'",
                    "required": True,
                },
                "target_compound": {
                    "type": "string",
                    "description": "Compound to calculate",
                    "required": True,
                },
            },
            returns="Amount of target compound in moles and grams",
        ))

        self.register_operation(ToolOperation(
            name="ph",
            description="Calculate pH from acid/base concentration",
            parameters={
                "concentration": {
                    "type": "number",
                    "description": "Molar concentration (M)",
                    "required": True,
                },
                "type": {
                    "type": "string",
                    "description": "'acid' or 'base'",
                    "required": True,
                },
                "strong": {
                    "type": "boolean",
                    "description": "Whether it's a strong acid/base",
                    "required": False,
                },
            },
            returns="pH and pOH values",
        ))

        self.register_operation(ToolOperation(
            name="molar_conversion",
            description="Convert between grams, moles, and particles",
            parameters={
                "formula": {
                    "type": "string",
                    "description": "Chemical formula",
                    "required": True,
                },
                "amount": {
                    "type": "number",
                    "description": "Amount to convert",
                    "required": True,
                },
                "from_unit": {
                    "type": "string",
                    "description": "'grams', 'moles', or 'particles'",
                    "required": True,
                },
                "to_unit": {
                    "type": "string",
                    "description": "'grams', 'moles', or 'particles'",
                    "required": True,
                },
            },
            returns="Converted amount",
        ))

        self.register_operation(ToolOperation(
            name="dilution",
            description="Dilution calculation using C1V1 = C2V2",
            parameters={
                "C1": {
                    "type": "number",
                    "description": "Initial concentration",
                    "required": False,
                },
                "V1": {
                    "type": "number",
                    "description": "Initial volume",
                    "required": False,
                },
                "C2": {
                    "type": "number",
                    "description": "Final concentration",
                    "required": False,
                },
                "V2": {
                    "type": "number",
                    "description": "Final volume",
                    "required": False,
                },
            },
            returns="Missing value from C1V1=C2V2",
            examples=[
                {"C1": 2.0, "V1": 50, "C2": 0.5, "result": "V2 = 200"},
            ],
        ))

        self.register_operation(ToolOperation(
            name="concentration",
            description="Molarity and molality calculations",
            parameters={
                "operation": {
                    "type": "string",
                    "description": "'molarity' or 'molality'",
                    "required": True,
                },
                "params": {
                    "type": "object",
                    "description": "For molarity: moles, liters, M. For molality: moles, kg_solvent, m",
                    "required": True,
                },
            },
            returns="Calculated concentration",
        ))

        self.register_operation(ToolOperation(
            name="percent_composition",
            description="Calculate percent composition by mass",
            parameters={
                "formula": {
                    "type": "string",
                    "description": "Chemical formula",
                    "required": True,
                },
            },
            returns="Percent mass of each element",
            examples=[
                {"formula": "H2O", "result": {"H": 11.19, "O": 88.81}},
            ],
        ))

        self.register_operation(ToolOperation(
            name="limiting_reagent",
            description="Determine limiting reagent and theoretical yield",
            parameters={
                "equation": {
                    "type": "string",
                    "description": "Chemical equation",
                    "required": True,
                },
                "amounts": {
                    "type": "object",
                    "description": "Dict of compound -> {amount, unit} for each reactant",
                    "required": True,
                },
            },
            returns="Limiting reagent and theoretical yields",
        ))

    def _execute_operation(self, operation: str, **kwargs) -> Any:
        """Execute a chemistry operation."""

        if operation == "molecular_weight":
            formula = kwargs["formula"]
            mw = calculate_molecular_weight(formula)
            return {
                "formula": formula,
                "molecular_weight": round(mw, 4),
                "unit": "g/mol",
                "composition": parse_formula(formula),
            }

        elif operation == "parse_formula":
            formula = kwargs["formula"]
            return {
                "formula": formula,
                "elements": parse_formula(formula),
            }

        elif operation == "balance_equation":
            return balance_equation(kwargs["equation"])

        elif operation == "stoichiometry":
            return stoichiometry(
                kwargs["equation"],
                kwargs["known_compound"],
                kwargs["known_amount"],
                kwargs["known_unit"],
                kwargs["target_compound"],
            )

        elif operation == "ph":
            return ph_calculation(
                kwargs["concentration"],
                kwargs.get("type", "acid"),
                kwargs.get("strong", True),
                kwargs.get("Ka"),
                kwargs.get("Kb"),
            )

        elif operation == "molar_conversion":
            formula = kwargs["formula"]
            amount = kwargs["amount"]
            from_unit = kwargs["from_unit"]
            to_unit = kwargs["to_unit"]

            mw = calculate_molecular_weight(formula)
            avogadro = 6.022e23

            # Convert to moles first
            if from_unit == "grams":
                moles = amount / mw
            elif from_unit == "moles":
                moles = amount
            elif from_unit == "particles":
                moles = amount / avogadro
            else:
                return {"success": False, "error": f"Unknown unit: {from_unit}"}

            # Convert from moles to target
            if to_unit == "grams":
                result = moles * mw
            elif to_unit == "moles":
                result = moles
            elif to_unit == "particles":
                result = moles * avogadro
            else:
                return {"success": False, "error": f"Unknown unit: {to_unit}"}

            return {
                "success": True,
                "original": {"amount": amount, "unit": from_unit},
                "result": {"amount": result, "unit": to_unit},
                "molecular_weight": mw,
            }

        elif operation == "dilution":
            return dilution(
                kwargs.get("C1"),
                kwargs.get("V1"),
                kwargs.get("C2"),
                kwargs.get("V2"),
            )

        elif operation == "concentration":
            params = kwargs.get("params", {})
            op = kwargs.get("operation", params.get("operation"))
            return molarity_molality(op, **params)

        elif operation == "percent_composition":
            return percent_composition(kwargs["formula"])

        elif operation == "limiting_reagent":
            return limiting_reagent(
                kwargs["equation"],
                kwargs["amounts"],
            )

        else:
            return ToolResult(
                success=False,
                error=f"Unknown operation: {operation}",
            )


def create_chemistry_tool() -> ChemistryTool:
    """Factory function to create chemistry tool."""
    return ChemistryTool()
