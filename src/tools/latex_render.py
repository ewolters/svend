"""
LaTeX Render Tool - Mathematical Typesetting

Convert LaTeX expressions to rendered images or formatted text.
Uses matplotlib for rendering, no external LaTeX installation needed.
"""

from typing import Optional, Dict, Any
import base64
import io

from .registry import Tool, ToolParameter, ToolResult, ToolStatus, ToolRegistry


class LaTeXRenderer:
    """
    LaTeX rendering engine using matplotlib.

    Can render LaTeX to:
    - PNG images (base64 encoded)
    - SVG (for web display)
    - Unicode approximation (for terminals)
    """

    def __init__(self):
        self._plt = None
        self._np = None

    @property
    def plt(self):
        if self._plt is None:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            # Enable LaTeX-like rendering
            plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern
            plt.rcParams['font.family'] = 'serif'
            self._plt = plt
        return self._plt

    def render_to_image(
        self,
        latex: str,
        fontsize: int = 20,
        dpi: int = 150,
        format: str = "png",
        transparent: bool = True,
    ) -> Dict[str, Any]:
        """
        Render LaTeX to image.

        Args:
            latex: LaTeX expression (with or without $ delimiters)
            fontsize: Font size for rendering
            dpi: Resolution
            format: Image format (png, svg)
            transparent: Transparent background
        """
        try:
            plt = self.plt

            # Ensure LaTeX is wrapped in $ for matplotlib
            if not latex.startswith('$'):
                latex = f'${latex}$'

            # Create figure with just the equation
            fig = plt.figure(figsize=(0.01, 0.01))
            fig.text(0, 0, latex, fontsize=fontsize, ha='left', va='bottom')

            # Adjust figure size to fit text
            fig.canvas.draw()
            bbox = fig.get_tightbbox(fig.canvas.get_renderer())
            fig.set_size_inches(bbox.width, bbox.height)

            # Render to buffer
            buf = io.BytesIO()
            fig.savefig(buf, format=format, dpi=dpi,
                       transparent=transparent, bbox_inches='tight', pad_inches=0.1)
            buf.seek(0)

            if format == "png":
                img_str = base64.b64encode(buf.read()).decode('utf-8')
            else:
                img_str = buf.read().decode('utf-8')

            buf.close()
            plt.close(fig)

            return {
                "success": True,
                "latex": latex,
                "image": img_str,
                "format": f"base64_{format}" if format == "png" else format,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def render_equation_set(
        self,
        equations: list,
        labels: Optional[list] = None,
        fontsize: int = 16,
    ) -> Dict[str, Any]:
        """
        Render a set of equations with optional labels.

        Args:
            equations: List of LaTeX equations
            labels: Optional equation labels/numbers
            fontsize: Font size
        """
        try:
            plt = self.plt

            n = len(equations)
            fig, ax = plt.subplots(figsize=(10, n * 0.8))
            ax.axis('off')

            y_positions = [1 - (i + 0.5) / n for i in range(n)]

            for i, eq in enumerate(equations):
                if not eq.startswith('$'):
                    eq = f'${eq}$'

                label = f"({labels[i]})" if labels and i < len(labels) else f"({i+1})"

                ax.text(0.5, y_positions[i], eq, fontsize=fontsize,
                       ha='center', va='center', transform=ax.transAxes)
                ax.text(0.95, y_positions[i], label, fontsize=fontsize-2,
                       ha='right', va='center', transform=ax.transAxes)

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            plt.close(fig)

            return {
                "success": True,
                "equations": equations,
                "image": img_str,
                "format": "base64_png",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def to_unicode(
        self,
        latex: str,
    ) -> Dict[str, Any]:
        """
        Convert simple LaTeX to Unicode approximation.

        Good for terminal/text display. Limited to common symbols.
        """
        try:
            # Unicode replacements
            replacements = {
                # Greek letters
                r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
                r'\epsilon': 'ε', r'\zeta': 'ζ', r'\eta': 'η', r'\theta': 'θ',
                r'\iota': 'ι', r'\kappa': 'κ', r'\lambda': 'λ', r'\mu': 'μ',
                r'\nu': 'ν', r'\xi': 'ξ', r'\pi': 'π', r'\rho': 'ρ',
                r'\sigma': 'σ', r'\tau': 'τ', r'\upsilon': 'υ', r'\phi': 'φ',
                r'\chi': 'χ', r'\psi': 'ψ', r'\omega': 'ω',
                r'\Gamma': 'Γ', r'\Delta': 'Δ', r'\Theta': 'Θ', r'\Lambda': 'Λ',
                r'\Xi': 'Ξ', r'\Pi': 'Π', r'\Sigma': 'Σ', r'\Phi': 'Φ',
                r'\Psi': 'Ψ', r'\Omega': 'Ω',

                # Operators and symbols
                r'\times': '×', r'\div': '÷', r'\pm': '±', r'\mp': '∓',
                r'\cdot': '·', r'\star': '⋆', r'\circ': '∘',
                r'\leq': '≤', r'\geq': '≥', r'\neq': '≠', r'\approx': '≈',
                r'\equiv': '≡', r'\sim': '∼', r'\propto': '∝',
                r'\infty': '∞', r'\partial': '∂', r'\nabla': '∇',
                r'\forall': '∀', r'\exists': '∃', r'\neg': '¬',
                r'\wedge': '∧', r'\vee': '∨', r'\cap': '∩', r'\cup': '∪',
                r'\in': '∈', r'\notin': '∉', r'\subset': '⊂', r'\supset': '⊃',
                r'\subseteq': '⊆', r'\supseteq': '⊇',
                r'\emptyset': '∅', r'\varnothing': '∅',
                r'\rightarrow': '→', r'\leftarrow': '←', r'\Rightarrow': '⇒',
                r'\Leftarrow': '⇐', r'\leftrightarrow': '↔', r'\Leftrightarrow': '⇔',
                r'\to': '→', r'\gets': '←',
                r'\sum': '∑', r'\prod': '∏', r'\int': '∫',
                r'\sqrt': '√', r'\surd': '√',

                # Fractions and special
                r'\frac{1}{2}': '½', r'\frac{1}{3}': '⅓', r'\frac{1}{4}': '¼',
                r'\frac{2}{3}': '⅔', r'\frac{3}{4}': '¾',

                # Superscripts
                r'^0': '⁰', r'^1': '¹', r'^2': '²', r'^3': '³',
                r'^4': '⁴', r'^5': '⁵', r'^6': '⁶', r'^7': '⁷',
                r'^8': '⁸', r'^9': '⁹', r'^+': '⁺', r'^-': '⁻',
                r'^n': 'ⁿ', r'^i': 'ⁱ',

                # Subscripts
                r'_0': '₀', r'_1': '₁', r'_2': '₂', r'_3': '₃',
                r'_4': '₄', r'_5': '₅', r'_6': '₆', r'_7': '₇',
                r'_8': '₈', r'_9': '₉', r'_+': '₊', r'_-': '₋',
                r'_n': 'ₙ', r'_i': 'ᵢ', r'_j': 'ⱼ', r'_k': 'ₖ',
            }

            result = latex
            for tex, uni in replacements.items():
                result = result.replace(tex, uni)

            # Clean up remaining LaTeX artifacts
            import re
            result = re.sub(r'\\[a-zA-Z]+', '', result)  # Remove unknown commands
            result = result.replace('{', '').replace('}', '')
            result = result.replace('$', '')

            return {
                "success": True,
                "latex": latex,
                "unicode": result.strip(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def validate(
        self,
        latex: str,
    ) -> Dict[str, Any]:
        """
        Validate LaTeX syntax by attempting to render.

        Returns whether the LaTeX is valid and any error messages.
        """
        try:
            plt = self.plt

            if not latex.startswith('$'):
                latex = f'${latex}$'

            fig = plt.figure(figsize=(1, 1))
            try:
                fig.text(0.5, 0.5, latex, fontsize=12)
                fig.canvas.draw()
                plt.close(fig)
                return {
                    "success": True,
                    "valid": True,
                    "latex": latex,
                }
            except Exception as e:
                plt.close(fig)
                return {
                    "success": True,
                    "valid": False,
                    "latex": latex,
                    "error": str(e),
                }

        except Exception as e:
            return {"success": False, "error": str(e)}


def latex_tool(
    operation: str,
    latex: str,
    options: Optional[str] = None,
) -> ToolResult:
    """Tool function for LaTeX rendering."""
    import json

    renderer = LaTeXRenderer()

    try:
        opts = json.loads(options) if options else {}
    except:
        opts = {}

    try:
        if operation == "render":
            result = renderer.render_to_image(
                latex,
                fontsize=opts.get("fontsize", 20),
                dpi=opts.get("dpi", 150),
                format=opts.get("format", "png"),
            )

        elif operation == "unicode":
            result = renderer.to_unicode(latex)

        elif operation == "validate":
            result = renderer.validate(latex)

        elif operation == "equations":
            equations = json.loads(latex) if latex.startswith('[') else [latex]
            result = renderer.render_equation_set(
                equations,
                labels=opts.get("labels"),
                fontsize=opts.get("fontsize", 16),
            )

        else:
            return ToolResult(
                status=ToolStatus.INVALID_INPUT,
                output=None,
                error=f"Unknown operation: {operation}",
            )

        if result.get("success"):
            if operation == "unicode":
                output = result["unicode"]
            elif operation == "validate":
                output = f"Valid: {result['valid']}"
                if not result["valid"]:
                    output += f"\nError: {result.get('error', 'Unknown')}"
            else:
                output = f"LaTeX rendered ({result.get('format', 'png')})"

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=output,
                metadata=result,
            )
        else:
            return ToolResult(
                status=ToolStatus.ERROR,
                output=None,
                error=result.get("error"),
            )

    except Exception as e:
        return ToolResult(
            status=ToolStatus.ERROR,
            output=None,
            error=str(e),
        )


def create_latex_tool() -> Tool:
    """Create the LaTeX rendering tool."""
    return Tool(
        name="latex_render",
        description="Render LaTeX mathematical expressions. Operations: 'render' (to image), 'unicode' (to Unicode text), 'validate' (check syntax), 'equations' (render multiple equations).",
        parameters=[
            ToolParameter(
                name="operation",
                description="Operation: 'render', 'unicode', 'validate', 'equations'",
                type="string",
                required=True,
                enum=["render", "unicode", "validate", "equations"],
            ),
            ToolParameter(
                name="latex",
                description="LaTeX expression(s). For 'equations', can be JSON array.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="options",
                description="JSON object with options (fontsize, dpi, format, labels)",
                type="string",
                required=False,
            ),
        ],
        execute_fn=latex_tool,
        timeout_ms=10000,
    )


def register_latex_tools(registry: ToolRegistry) -> None:
    """Register LaTeX tools with the registry."""
    registry.register(create_latex_tool())
