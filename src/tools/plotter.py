"""
Plotter Tool - Data Visualization with Matplotlib

Generate plots for functions, data, and statistical visualizations.
Returns base64-encoded images or plot descriptions.
"""

from typing import Optional, Dict, Any, List, Union
import json
import base64
import io

from .registry import Tool, ToolParameter, ToolResult, ToolStatus, ToolRegistry


class Plotter:
    """
    Visualization engine using Matplotlib.

    Generates plots and returns as base64 images.
    """

    def __init__(self):
        self._plt = None
        self._np = None

    @property
    def plt(self):
        if self._plt is None:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            self._plt = plt
        return self._plt

    @property
    def np(self):
        if self._np is None:
            import numpy as np
            self._np = np
        return self._np

    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return img_str

    def _apply_style(self, ax, options: Dict[str, Any]):
        """Apply common styling options."""
        if options.get("title"):
            ax.set_title(options["title"])
        if options.get("xlabel"):
            ax.set_xlabel(options["xlabel"])
        if options.get("ylabel"):
            ax.set_ylabel(options["ylabel"])
        if options.get("grid", True):
            ax.grid(True, alpha=0.3)
        if options.get("legend"):
            ax.legend()

    def plot_function(
        self,
        functions: List[str],
        x_range: List[float],
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Plot mathematical functions.

        Args:
            functions: List of function strings (e.g., ["sin(x)", "cos(x)"])
            x_range: [x_min, x_max]
            options: Styling options (title, labels, colors, etc.)
        """
        try:
            plt = self.plt
            np = self.np
            options = options or {}

            # Create x values
            n_points = options.get("n_points", 500)
            x = np.linspace(x_range[0], x_range[1], n_points)

            # Create figure
            fig, ax = plt.subplots(figsize=options.get("figsize", (10, 6)))

            # Plot each function
            colors = options.get("colors", plt.cm.tab10.colors)
            labels = options.get("labels", functions)

            for i, func_str in enumerate(functions):
                # Create safe evaluation namespace
                namespace = {
                    "__builtins__": {},
                    "x": x,
                    "np": np,
                    "sin": np.sin, "cos": np.cos, "tan": np.tan,
                    "exp": np.exp, "log": np.log, "log10": np.log10,
                    "sqrt": np.sqrt, "abs": np.abs,
                    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
                    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
                    "pi": np.pi, "e": np.e,
                }

                y = eval(func_str, namespace)
                ax.plot(x, y, color=colors[i % len(colors)], label=labels[i])

            self._apply_style(ax, options)
            if len(functions) > 1 or options.get("legend"):
                ax.legend()

            # Set axis limits
            if options.get("y_range"):
                ax.set_ylim(options["y_range"])

            # Convert to base64
            img_str = self._fig_to_base64(fig)
            plt.close(fig)

            return {
                "success": True,
                "image": img_str,
                "format": "base64_png",
                "functions": functions,
                "x_range": x_range,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def plot_scatter(
        self,
        x: List[float],
        y: List[float],
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create scatter plot.

        Args:
            x: X values
            y: Y values
            options: Styling options
        """
        try:
            plt = self.plt
            np = self.np
            options = options or {}

            fig, ax = plt.subplots(figsize=options.get("figsize", (10, 6)))

            # Optional: color by third variable
            c = options.get("c")
            s = options.get("s", 50)  # Point size
            alpha = options.get("alpha", 0.7)
            cmap = options.get("cmap", "viridis")

            scatter = ax.scatter(x, y, c=c, s=s, alpha=alpha, cmap=cmap)

            if c is not None:
                plt.colorbar(scatter, ax=ax, label=options.get("colorbar_label", ""))

            # Optional: add regression line
            if options.get("regression_line"):
                from scipy import stats
                slope, intercept, _, _, _ = stats.linregress(x, y)
                x_line = np.array([min(x), max(x)])
                ax.plot(x_line, slope * x_line + intercept, 'r--', alpha=0.8, label=f'y = {slope:.3f}x + {intercept:.3f}')
                ax.legend()

            self._apply_style(ax, options)

            img_str = self._fig_to_base64(fig)
            plt.close(fig)

            return {
                "success": True,
                "image": img_str,
                "format": "base64_png",
                "n_points": len(x),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def plot_line(
        self,
        data: Dict[str, List[float]],
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create line plot.

        Args:
            data: Dict with 'x' and one or more y series
            options: Styling options
        """
        try:
            plt = self.plt
            options = options or {}

            fig, ax = plt.subplots(figsize=options.get("figsize", (10, 6)))

            x = data.get("x", list(range(len(list(data.values())[0]))))
            colors = options.get("colors", plt.cm.tab10.colors)

            i = 0
            for key, values in data.items():
                if key == "x":
                    continue
                ax.plot(x, values, color=colors[i % len(colors)], label=key,
                       linewidth=options.get("linewidth", 2))
                i += 1

            self._apply_style(ax, options)
            if len(data) > 2:  # More than just x and one y
                ax.legend()

            img_str = self._fig_to_base64(fig)
            plt.close(fig)

            return {
                "success": True,
                "image": img_str,
                "format": "base64_png",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def plot_bar(
        self,
        categories: List[str],
        values: List[float],
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create bar chart.

        Args:
            categories: Category labels
            values: Bar heights
            options: Styling options
        """
        try:
            plt = self.plt
            np = self.np
            options = options or {}

            fig, ax = plt.subplots(figsize=options.get("figsize", (10, 6)))

            colors = options.get("colors", plt.cm.tab10.colors)
            bars = ax.bar(categories, values, color=[colors[i % len(colors)] for i in range(len(values))])

            # Add value labels on bars
            if options.get("show_values", True):
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{val:.2f}', ha='center', va='bottom')

            self._apply_style(ax, options)

            # Rotate x labels if needed
            if options.get("rotate_labels"):
                plt.xticks(rotation=options["rotate_labels"])

            img_str = self._fig_to_base64(fig)
            plt.close(fig)

            return {
                "success": True,
                "image": img_str,
                "format": "base64_png",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def plot_histogram(
        self,
        data: List[float],
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create histogram.

        Args:
            data: Data values
            options: bins, density, cumulative, etc.
        """
        try:
            plt = self.plt
            np = self.np
            options = options or {}

            fig, ax = plt.subplots(figsize=options.get("figsize", (10, 6)))

            bins = options.get("bins", "auto")
            density = options.get("density", False)
            alpha = options.get("alpha", 0.7)
            color = options.get("color", "steelblue")

            n, bins_out, patches = ax.hist(data, bins=bins, density=density,
                                           alpha=alpha, color=color, edgecolor='black')

            # Optional: overlay normal distribution
            if options.get("fit_normal") and density:
                from scipy import stats
                mu, std = np.mean(data), np.std(data)
                x = np.linspace(min(data), max(data), 100)
                ax.plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2,
                       label=f'Normal fit (μ={mu:.2f}, σ={std:.2f})')
                ax.legend()

            self._apply_style(ax, options)

            img_str = self._fig_to_base64(fig)
            plt.close(fig)

            # Calculate basic stats
            return {
                "success": True,
                "image": img_str,
                "format": "base64_png",
                "stats": {
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data)),
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                    "n": len(data),
                },
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def plot_heatmap(
        self,
        matrix: List[List[float]],
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create heatmap.

        Args:
            matrix: 2D array of values
            options: cmap, labels, etc.
        """
        try:
            plt = self.plt
            np = self.np
            options = options or {}

            fig, ax = plt.subplots(figsize=options.get("figsize", (10, 8)))

            data = np.array(matrix)
            cmap = options.get("cmap", "viridis")

            im = ax.imshow(data, cmap=cmap, aspect='auto')
            plt.colorbar(im, ax=ax)

            # Add labels
            if options.get("x_labels"):
                ax.set_xticks(range(len(options["x_labels"])))
                ax.set_xticklabels(options["x_labels"], rotation=45, ha='right')
            if options.get("y_labels"):
                ax.set_yticks(range(len(options["y_labels"])))
                ax.set_yticklabels(options["y_labels"])

            # Annotate cells
            if options.get("annotate"):
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center',
                               color='white' if data[i, j] < (data.max() + data.min())/2 else 'black')

            self._apply_style(ax, options)

            img_str = self._fig_to_base64(fig)
            plt.close(fig)

            return {
                "success": True,
                "image": img_str,
                "format": "base64_png",
                "shape": list(data.shape),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def plot_pie(
        self,
        values: List[float],
        labels: List[str],
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create pie chart.

        Args:
            values: Slice sizes
            labels: Slice labels
            options: colors, explode, etc.
        """
        try:
            plt = self.plt
            options = options or {}

            fig, ax = plt.subplots(figsize=options.get("figsize", (10, 8)))

            explode = options.get("explode")
            colors = options.get("colors")
            autopct = options.get("autopct", '%1.1f%%')

            ax.pie(values, labels=labels, explode=explode, colors=colors,
                  autopct=autopct, shadow=options.get("shadow", False),
                  startangle=options.get("startangle", 90))

            ax.axis('equal')

            if options.get("title"):
                ax.set_title(options["title"])

            img_str = self._fig_to_base64(fig)
            plt.close(fig)

            return {
                "success": True,
                "image": img_str,
                "format": "base64_png",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def plot_boxplot(
        self,
        data: Union[List[float], List[List[float]]],
        labels: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create box plot.

        Args:
            data: Single list or list of lists for multiple boxes
            labels: Labels for each box
            options: Styling options
        """
        try:
            plt = self.plt
            np = self.np
            options = options or {}

            fig, ax = plt.subplots(figsize=options.get("figsize", (10, 6)))

            # Handle single list vs multiple
            if isinstance(data[0], (int, float)):
                data = [data]

            bp = ax.boxplot(data, labels=labels, patch_artist=True)

            # Color boxes
            colors = options.get("colors", plt.cm.Pastel1.colors)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)

            self._apply_style(ax, options)

            img_str = self._fig_to_base64(fig)
            plt.close(fig)

            # Calculate stats
            stats_list = []
            for d in data:
                arr = np.array(d)
                stats_list.append({
                    "median": float(np.median(arr)),
                    "q1": float(np.percentile(arr, 25)),
                    "q3": float(np.percentile(arr, 75)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                })

            return {
                "success": True,
                "image": img_str,
                "format": "base64_png",
                "stats": stats_list,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def plot_contour(
        self,
        function: str,
        x_range: List[float],
        y_range: List[float],
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create contour plot.

        Args:
            function: f(x, y) as string
            x_range: [x_min, x_max]
            y_range: [y_min, y_max]
            options: levels, cmap, etc.
        """
        try:
            plt = self.plt
            np = self.np
            options = options or {}

            n_points = options.get("n_points", 100)
            x = np.linspace(x_range[0], x_range[1], n_points)
            y = np.linspace(y_range[0], y_range[1], n_points)
            X, Y = np.meshgrid(x, y)

            # Evaluate function
            namespace = {
                "__builtins__": {},
                "x": X, "y": Y,
                "np": np,
                "sin": np.sin, "cos": np.cos, "exp": np.exp,
                "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
                "pi": np.pi, "e": np.e,
            }
            Z = eval(function, namespace)

            fig, ax = plt.subplots(figsize=options.get("figsize", (10, 8)))

            levels = options.get("levels", 20)
            cmap = options.get("cmap", "viridis")

            if options.get("filled", True):
                cs = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
            else:
                cs = ax.contour(X, Y, Z, levels=levels, cmap=cmap)

            plt.colorbar(cs, ax=ax)

            self._apply_style(ax, options)

            img_str = self._fig_to_base64(fig)
            plt.close(fig)

            return {
                "success": True,
                "image": img_str,
                "format": "base64_png",
                "function": function,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}


def plotter_tool(
    plot_type: str,
    data: str,
    options: Optional[str] = None,
) -> ToolResult:
    """Tool function for plotting."""
    plotter = Plotter()

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
        if plot_type == "function":
            result = plotter.plot_function(
                data_dict.get("functions"),
                data_dict.get("x_range"),
                opts,
            )

        elif plot_type == "scatter":
            result = plotter.plot_scatter(
                data_dict.get("x"),
                data_dict.get("y"),
                opts,
            )

        elif plot_type == "line":
            result = plotter.plot_line(data_dict, opts)

        elif plot_type == "bar":
            result = plotter.plot_bar(
                data_dict.get("categories"),
                data_dict.get("values"),
                opts,
            )

        elif plot_type == "histogram":
            result = plotter.plot_histogram(
                data_dict.get("data"),
                opts,
            )

        elif plot_type == "heatmap":
            result = plotter.plot_heatmap(
                data_dict.get("matrix"),
                opts,
            )

        elif plot_type == "pie":
            result = plotter.plot_pie(
                data_dict.get("values"),
                data_dict.get("labels"),
                opts,
            )

        elif plot_type == "boxplot":
            result = plotter.plot_boxplot(
                data_dict.get("data"),
                data_dict.get("labels"),
                opts,
            )

        elif plot_type == "contour":
            result = plotter.plot_contour(
                data_dict.get("function"),
                data_dict.get("x_range"),
                data_dict.get("y_range"),
                opts,
            )

        else:
            return ToolResult(
                status=ToolStatus.INVALID_INPUT,
                output=None,
                error=f"Unknown plot type: {plot_type}",
            )

        if result.get("success"):
            # Return just confirmation and image reference
            output = f"Plot generated ({plot_type})"
            if "stats" in result:
                output += f"\nStats: {json.dumps(result['stats'], indent=2)}"

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


def create_plotter_tool() -> Tool:
    """Create the plotter tool."""
    return Tool(
        name="plotter",
        description="Create visualizations: function plots, scatter plots, line charts, bar charts, histograms, heatmaps, pie charts, box plots, contour plots. Returns base64-encoded PNG images.",
        parameters=[
            ToolParameter(
                name="plot_type",
                description="Type of plot to create",
                type="string",
                required=True,
                enum=["function", "scatter", "line", "bar", "histogram",
                      "heatmap", "pie", "boxplot", "contour"],
            ),
            ToolParameter(
                name="data",
                description="JSON object with plot data. Contents depend on plot_type.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="options",
                description="JSON object with styling options (title, labels, colors, etc.)",
                type="string",
                required=False,
            ),
        ],
        execute_fn=plotter_tool,
        timeout_ms=30000,
    )


def register_plotter_tools(registry: ToolRegistry) -> None:
    """Register plotter tools with the registry."""
    registry.register(create_plotter_tool())
