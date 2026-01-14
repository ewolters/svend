"""
Specialist Visualization Tools

Domain-specific visualizations for engineering specialists.

Plot types:
- bode: Bode plot (magnitude & phase) for controls
- nyquist: Nyquist stability plot
- step_response: System step response
- gantt: Gantt chart for project management
- earned_value: S-curve and EVM chart
- weibull: Weibull probability plot
- fault_tree: Fault tree diagram
- capability: Process capability histogram with spec limits
- control_chart: SPC control chart
- fmea_pareto: FMEA Pareto chart
- n_squared: N-squared interface diagram
"""

from typing import Any, Dict, List, Optional, Tuple
import math
import json
import base64
import io

from .registry import Tool, ToolParameter, ToolResult, ToolStatus, ToolRegistry


class SpecialistPlotter:
    """
    Domain-specific visualization engine.

    Generates engineering plots using Matplotlib.
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

    # ============ CONTROLS PLOTS ============

    def plot_bode(
        self,
        num: List[float],
        den: List[float],
        freq_range: Optional[Tuple[float, float]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create Bode plot (magnitude and phase).

        Args:
            num: Numerator coefficients [highest power first]
            den: Denominator coefficients
            freq_range: (omega_min, omega_max) in rad/s
            options: title, margins, etc.
        """
        try:
            plt = self.plt
            np = self.np
            options = options or {}

            # Frequency range
            if freq_range:
                omega = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), 500)
            else:
                omega = np.logspace(-2, 3, 500)

            # Calculate frequency response
            s = 1j * omega
            num_poly = np.polyval(num, s)
            den_poly = np.polyval(den, s)
            H = num_poly / den_poly

            magnitude = 20 * np.log10(np.abs(H))
            phase = np.angle(H, deg=True)

            # Unwrap phase
            phase = np.unwrap(phase * np.pi / 180) * 180 / np.pi

            # Create dual plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=options.get("figsize", (10, 8)), sharex=True)

            # Magnitude plot
            ax1.semilogx(omega, magnitude, 'b-', linewidth=2)
            ax1.set_ylabel('Magnitude (dB)')
            ax1.grid(True, which='both', alpha=0.3)
            ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)

            # Add -3dB line for bandwidth
            ax1.axhline(y=-3, color='r', linestyle=':', alpha=0.5, label='-3 dB')

            # Phase plot
            ax2.semilogx(omega, phase, 'r-', linewidth=2)
            ax2.set_ylabel('Phase (degrees)')
            ax2.set_xlabel('Frequency (rad/s)')
            ax2.grid(True, which='both', alpha=0.3)
            ax2.axhline(y=-180, color='k', linestyle='--', alpha=0.5)

            # Add margins if requested
            if options.get("show_margins"):
                # Find gain crossover (magnitude = 0 dB)
                try:
                    gc_idx = np.where(np.diff(np.sign(magnitude)))[0]
                    if len(gc_idx) > 0:
                        gc_omega = omega[gc_idx[0]]
                        gc_phase = phase[gc_idx[0]]
                        phase_margin = 180 + gc_phase
                        ax2.annotate(f'PM = {phase_margin:.1f}°',
                                    xy=(gc_omega, gc_phase),
                                    xytext=(gc_omega * 2, gc_phase + 20),
                                    arrowprops=dict(arrowstyle='->', color='green'))
                except Exception:
                    pass

            if options.get("title"):
                ax1.set_title(options["title"])
            else:
                ax1.set_title("Bode Plot")

            plt.tight_layout()

            img_str = self._fig_to_base64(fig)
            plt.close(fig)

            return {
                "success": True,
                "image": img_str,
                "format": "base64_png",
                "frequency_range": [float(omega[0]), float(omega[-1])]
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def plot_nyquist(
        self,
        num: List[float],
        den: List[float],
        freq_range: Optional[Tuple[float, float]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create Nyquist plot.

        Args:
            num: Numerator coefficients
            den: Denominator coefficients
            freq_range: (omega_min, omega_max)
            options: Styling options
        """
        try:
            plt = self.plt
            np = self.np
            options = options or {}

            if freq_range:
                omega = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), 1000)
            else:
                omega = np.logspace(-3, 4, 1000)

            s = 1j * omega
            num_poly = np.polyval(num, s)
            den_poly = np.polyval(den, s)
            H = num_poly / den_poly

            fig, ax = plt.subplots(figsize=options.get("figsize", (10, 10)))

            # Plot positive frequencies
            ax.plot(H.real, H.imag, 'b-', linewidth=2, label='ω > 0')
            # Plot negative frequencies (mirror)
            ax.plot(H.real, -H.imag, 'b--', linewidth=1, alpha=0.5, label='ω < 0')

            # Mark critical point (-1, 0)
            ax.plot(-1, 0, 'rx', markersize=15, markeredgewidth=3, label='Critical point (-1,0)')

            # Add unit circle for reference
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta), 'k:', alpha=0.3)

            # Add arrows for direction
            for i in range(0, len(H) - 1, len(H) // 10):
                ax.annotate('', xy=(H[i+1].real, H[i+1].imag),
                           xytext=(H[i].real, H[i].imag),
                           arrowprops=dict(arrowstyle='->', color='blue', alpha=0.5))

            ax.set_xlabel('Real')
            ax.set_ylabel('Imaginary')
            ax.set_title(options.get("title", "Nyquist Plot"))
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.axis('equal')

            # Set limits
            margin = 0.5
            max_extent = max(abs(H.real).max(), abs(H.imag).max(), 2)
            ax.set_xlim(-max_extent - margin, max_extent + margin)
            ax.set_ylim(-max_extent - margin, max_extent + margin)

            img_str = self._fig_to_base64(fig)
            plt.close(fig)

            return {
                "success": True,
                "image": img_str,
                "format": "base64_png"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def plot_step_response(
        self,
        time: List[float],
        response: List[float],
        reference: float = 1.0,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Plot step response with key characteristics.

        Args:
            time: Time vector
            response: Response values
            reference: Target value (default 1.0)
            options: Styling options
        """
        try:
            plt = self.plt
            np = self.np
            options = options or {}

            fig, ax = plt.subplots(figsize=options.get("figsize", (10, 6)))

            ax.plot(time, response, 'b-', linewidth=2, label='Response')
            ax.axhline(y=reference, color='r', linestyle='--', alpha=0.7, label='Reference')

            # Add settling bounds (2% and 5%)
            ax.axhline(y=reference * 1.02, color='g', linestyle=':', alpha=0.5)
            ax.axhline(y=reference * 0.98, color='g', linestyle=':', alpha=0.5, label='±2% bounds')

            # Find key metrics
            response = np.array(response)
            time = np.array(time)

            # Overshoot
            peak = response.max()
            if peak > reference:
                overshoot = (peak - reference) / reference * 100
                peak_idx = response.argmax()
                ax.annotate(f'OS = {overshoot:.1f}%',
                           xy=(time[peak_idx], peak),
                           xytext=(time[peak_idx] + (time[-1] - time[0]) * 0.1, peak),
                           arrowprops=dict(arrowstyle='->', color='red'))

            # Rise time (10% to 90%)
            try:
                idx_10 = np.where(response >= 0.1 * reference)[0][0]
                idx_90 = np.where(response >= 0.9 * reference)[0][0]
                rise_time = time[idx_90] - time[idx_10]
                ax.axvline(x=time[idx_10], color='m', linestyle=':', alpha=0.3)
                ax.axvline(x=time[idx_90], color='m', linestyle=':', alpha=0.3)
            except IndexError:
                rise_time = None

            # Settling time (2% criterion)
            try:
                settled = np.abs(response - reference) <= 0.02 * reference
                # Find last time not settled
                not_settled = np.where(~settled)[0]
                if len(not_settled) > 0:
                    settling_time = time[not_settled[-1]]
                    ax.axvline(x=settling_time, color='g', linestyle='--', alpha=0.5, label=f'Ts = {settling_time:.2f}')
                else:
                    settling_time = time[0]
            except Exception:
                settling_time = None

            ax.set_xlabel('Time')
            ax.set_ylabel('Response')
            ax.set_title(options.get("title", "Step Response"))
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')

            img_str = self._fig_to_base64(fig)
            plt.close(fig)

            return {
                "success": True,
                "image": img_str,
                "format": "base64_png",
                "metrics": {
                    "overshoot_percent": float(overshoot) if 'overshoot' in dir() else 0,
                    "rise_time": float(rise_time) if rise_time else None,
                    "settling_time": float(settling_time) if settling_time else None
                }
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    # ============ PROJECT MANAGEMENT PLOTS ============

    def plot_gantt(
        self,
        tasks: List[Dict[str, Any]],
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create Gantt chart.

        Args:
            tasks: List of {name, start, duration, critical (optional)}
            options: Styling options
        """
        try:
            plt = self.plt
            np = self.np
            options = options or {}

            fig, ax = plt.subplots(figsize=options.get("figsize", (12, max(6, len(tasks) * 0.4))))

            y_positions = range(len(tasks))

            for i, task in enumerate(tasks):
                start = task["start"]
                duration = task["duration"]
                is_critical = task.get("critical", False)

                color = 'red' if is_critical else 'steelblue'
                ax.barh(i, duration, left=start, height=0.6,
                       color=color, alpha=0.8, edgecolor='black')

                # Add task name inside bar
                ax.text(start + duration/2, i, task["name"],
                       va='center', ha='center', fontsize=9, color='white', fontweight='bold')

            ax.set_yticks(y_positions)
            ax.set_yticklabels([t["name"] for t in tasks])
            ax.set_xlabel('Time')
            ax.set_title(options.get("title", "Project Gantt Chart"))
            ax.grid(True, axis='x', alpha=0.3)

            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', alpha=0.8, label='Critical Path'),
                Patch(facecolor='steelblue', alpha=0.8, label='Non-Critical')
            ]
            ax.legend(handles=legend_elements, loc='upper right')

            ax.invert_yaxis()  # Tasks from top to bottom

            plt.tight_layout()
            img_str = self._fig_to_base64(fig)
            plt.close(fig)

            return {
                "success": True,
                "image": img_str,
                "format": "base64_png"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def plot_earned_value(
        self,
        periods: List[int],
        pv: List[float],
        ev: List[float],
        ac: List[float],
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create Earned Value Management chart.

        Args:
            periods: Time periods
            pv: Planned Value
            ev: Earned Value
            ac: Actual Cost
            options: Styling options
        """
        try:
            plt = self.plt
            options = options or {}

            fig, ax = plt.subplots(figsize=options.get("figsize", (12, 6)))

            ax.plot(periods, pv, 'b-', linewidth=2, marker='o', label='PV (Planned Value)')
            ax.plot(periods, ev, 'g-', linewidth=2, marker='s', label='EV (Earned Value)')
            ax.plot(periods, ac, 'r-', linewidth=2, marker='^', label='AC (Actual Cost)')

            # Fill areas for variance visualization
            ax.fill_between(periods, ev, pv, alpha=0.2, color='blue',
                           label='Schedule Variance')
            ax.fill_between(periods, ev, ac, alpha=0.2, color='green',
                           label='Cost Variance')

            ax.set_xlabel('Period')
            ax.set_ylabel('Value ($)')
            ax.set_title(options.get("title", "Earned Value Management"))
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')

            # Add current period metrics
            if len(periods) > 0:
                current = len(periods) - 1
                sv = ev[current] - pv[current]
                cv = ev[current] - ac[current]
                spi = ev[current] / pv[current] if pv[current] > 0 else 0
                cpi = ev[current] / ac[current] if ac[current] > 0 else 0

                textstr = f'SV: ${sv:,.0f}\nCV: ${cv:,.0f}\nSPI: {spi:.2f}\nCPI: {cpi:.2f}'
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                ax.text(0.98, 0.02, textstr, transform=ax.transAxes,
                       fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                       bbox=props)

            plt.tight_layout()
            img_str = self._fig_to_base64(fig)
            plt.close(fig)

            return {
                "success": True,
                "image": img_str,
                "format": "base64_png"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    # ============ RELIABILITY PLOTS ============

    def plot_weibull(
        self,
        beta: float,
        eta: float,
        gamma: float = 0,
        time_range: Optional[Tuple[float, float]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create Weibull probability plot.

        Args:
            beta: Shape parameter
            eta: Scale parameter (characteristic life)
            gamma: Location parameter (failure-free period)
            time_range: (t_min, t_max)
            options: Styling options
        """
        try:
            plt = self.plt
            np = self.np
            options = options or {}

            if time_range:
                t = np.linspace(time_range[0], time_range[1], 200)
            else:
                t = np.linspace(gamma, gamma + eta * 2.5, 200)

            # Reliability function
            t_adj = np.maximum(t - gamma, 1e-10)
            R = np.exp(-(t_adj / eta) ** beta)
            F = 1 - R  # Unreliability

            # Failure rate
            h = (beta / eta) * (t_adj / eta) ** (beta - 1)

            fig, axes = plt.subplots(2, 2, figsize=options.get("figsize", (12, 10)))

            # R(t) plot
            ax1 = axes[0, 0]
            ax1.plot(t, R, 'b-', linewidth=2)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Reliability R(t)')
            ax1.set_title('Reliability Function')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)

            # F(t) plot
            ax2 = axes[0, 1]
            ax2.plot(t, F, 'r-', linewidth=2)
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Unreliability F(t)')
            ax2.set_title('Cumulative Distribution')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)

            # Failure rate h(t)
            ax3 = axes[1, 0]
            ax3.plot(t, h, 'g-', linewidth=2)
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Failure Rate h(t)')
            ax3.set_title(f'Hazard Function (β={beta:.2f})')
            ax3.grid(True, alpha=0.3)

            # Weibull probability plot
            ax4 = axes[1, 1]
            # Plot ln(ln(1/(1-F))) vs ln(t)
            F_plot = np.linspace(0.01, 0.99, 50)
            y_plot = np.log(np.log(1 / (1 - F_plot)))
            x_plot = np.log(eta * (-np.log(1 - F_plot)) ** (1/beta))
            ax4.plot(x_plot, y_plot, 'ko-', markersize=3)

            ax4.set_xlabel('ln(t)')
            ax4.set_ylabel('ln(ln(1/(1-F)))')
            ax4.set_title('Weibull Probability Plot')
            ax4.grid(True, alpha=0.3)

            # Add parameter annotation
            textstr = f'β = {beta:.2f}\nη = {eta:.1f}\nγ = {gamma:.1f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax1.text(0.95, 0.95, textstr, transform=ax1.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=props)

            plt.tight_layout()
            img_str = self._fig_to_base64(fig)
            plt.close(fig)

            return {
                "success": True,
                "image": img_str,
                "format": "base64_png",
                "parameters": {"beta": beta, "eta": eta, "gamma": gamma}
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def plot_fault_tree(
        self,
        tree_structure: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create simplified fault tree diagram.

        Args:
            tree_structure: Tree with name, gate, children, probability
            options: Styling options
        """
        try:
            plt = self.plt
            options = options or {}

            fig, ax = plt.subplots(figsize=options.get("figsize", (14, 10)))
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')

            def draw_node(node: Dict, x: float, y: float, level: int = 0):
                """Recursively draw tree nodes."""
                # Draw box
                box_width = 1.5
                box_height = 0.8

                if "probability" in node:
                    # Basic event (circle)
                    circle = plt.Circle((x, y), 0.3, fill=False, color='blue')
                    ax.add_patch(circle)
                    ax.text(x, y, f'{node["probability"]:.3f}',
                           ha='center', va='center', fontsize=8)
                    ax.text(x, y - 0.5, node["name"][:15],
                           ha='center', va='top', fontsize=7)
                else:
                    # Gate (rectangle with gate symbol)
                    rect = plt.Rectangle((x - box_width/2, y - box_height/2),
                                        box_width, box_height, fill=False, color='black')
                    ax.add_patch(rect)

                    gate = node.get("gate", "or").upper()
                    ax.text(x, y + 0.1, gate, ha='center', va='center',
                           fontsize=10, fontweight='bold')
                    ax.text(x, y - box_height/2 - 0.2, node["name"][:20],
                           ha='center', va='top', fontsize=8)

                    # Draw children
                    children = node.get("children", [])
                    if children:
                        n_children = len(children)
                        # Space children evenly
                        spacing = min(2.5, 8 / max(n_children, 1))
                        start_x = x - (n_children - 1) * spacing / 2

                        for i, child in enumerate(children):
                            child_x = start_x + i * spacing
                            child_y = y - 2

                            # Draw connector line
                            ax.plot([x, child_x], [y - box_height/2, child_y + 0.5],
                                   'k-', linewidth=1)

                            draw_node(child, child_x, child_y, level + 1)

            # Draw from top
            draw_node(tree_structure, 5, 9)

            ax.set_title(options.get("title", "Fault Tree Analysis"))

            img_str = self._fig_to_base64(fig)
            plt.close(fig)

            return {
                "success": True,
                "image": img_str,
                "format": "base64_png"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    # ============ MANUFACTURING PLOTS ============

    def plot_capability(
        self,
        data: List[float],
        lsl: float,
        usl: float,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create process capability histogram with spec limits.

        Args:
            data: Measurement data
            lsl: Lower Specification Limit
            usl: Upper Specification Limit
            options: Styling options
        """
        try:
            plt = self.plt
            np = self.np
            options = options or {}

            data = np.array(data)
            mean = np.mean(data)
            std = np.std(data, ddof=1)

            fig, ax = plt.subplots(figsize=options.get("figsize", (10, 6)))

            # Histogram
            n, bins, patches = ax.hist(data, bins='auto', density=True,
                                       alpha=0.7, color='steelblue', edgecolor='black')

            # Normal curve
            from scipy import stats
            x = np.linspace(mean - 4*std, mean + 4*std, 100)
            ax.plot(x, stats.norm.pdf(x, mean, std), 'r-', linewidth=2, label='Normal fit')

            # Spec limits
            ax.axvline(x=lsl, color='red', linestyle='--', linewidth=2, label=f'LSL = {lsl}')
            ax.axvline(x=usl, color='red', linestyle='--', linewidth=2, label=f'USL = {usl}')
            ax.axvline(x=mean, color='green', linestyle='-', linewidth=2, label=f'Mean = {mean:.3f}')

            # Calculate Cp, Cpk
            cp = (usl - lsl) / (6 * std)
            cpu = (usl - mean) / (3 * std)
            cpl = (mean - lsl) / (3 * std)
            cpk = min(cpu, cpl)

            # Add annotation
            textstr = f'Cp = {cp:.3f}\nCpk = {cpk:.3f}\nσ = {std:.4f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.95, 0.95, textstr, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=props)

            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.set_title(options.get("title", "Process Capability Analysis"))
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)

            img_str = self._fig_to_base64(fig)
            plt.close(fig)

            return {
                "success": True,
                "image": img_str,
                "format": "base64_png",
                "metrics": {
                    "cp": float(cp),
                    "cpk": float(cpk),
                    "mean": float(mean),
                    "std": float(std)
                }
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def plot_control_chart(
        self,
        data: List[float],
        subgroup_size: int = 1,
        chart_type: str = "xbar",
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create SPC control chart.

        Args:
            data: Measurement data
            subgroup_size: Samples per subgroup
            chart_type: "xbar", "individuals", "range"
            options: Styling options
        """
        try:
            plt = self.plt
            np = self.np
            options = options or {}

            data = np.array(data)

            fig, ax = plt.subplots(figsize=options.get("figsize", (12, 6)))

            if chart_type == "individuals" or subgroup_size == 1:
                # Individuals chart
                mean = np.mean(data)
                # Moving range for sigma estimate
                mr = np.abs(np.diff(data))
                mr_bar = np.mean(mr)
                sigma = mr_bar / 1.128  # d2 for n=2

                ucl = mean + 3 * sigma
                lcl = mean - 3 * sigma

                ax.plot(range(len(data)), data, 'b-o', markersize=4)
                chart_name = "Individuals (I)"
            else:
                # X-bar chart
                n_subgroups = len(data) // subgroup_size
                subgroups = data[:n_subgroups * subgroup_size].reshape(n_subgroups, subgroup_size)
                means = subgroups.mean(axis=1)
                ranges = subgroups.ptp(axis=1)

                x_bar = np.mean(means)
                r_bar = np.mean(ranges)

                # A2 constant
                a2_table = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483}
                a2 = a2_table.get(subgroup_size, 0.577)

                ucl = x_bar + a2 * r_bar
                lcl = x_bar - a2 * r_bar
                mean = x_bar
                data = means

                ax.plot(range(len(means)), means, 'b-o', markersize=4)
                chart_name = "X-bar"

            # Control limits
            ax.axhline(y=mean, color='green', linestyle='-', linewidth=2, label=f'CL = {mean:.3f}')
            ax.axhline(y=ucl, color='red', linestyle='--', linewidth=2, label=f'UCL = {ucl:.3f}')
            ax.axhline(y=lcl, color='red', linestyle='--', linewidth=2, label=f'LCL = {lcl:.3f}')

            # Highlight out-of-control points
            ooc = (data > ucl) | (data < lcl)
            if np.any(ooc):
                ax.scatter(np.where(ooc)[0], data[ooc], color='red', s=100, zorder=5,
                          label=f'Out of Control ({np.sum(ooc)})')

            ax.set_xlabel('Subgroup / Sample')
            ax.set_ylabel('Value')
            ax.set_title(options.get("title", f"{chart_name} Control Chart"))
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

            img_str = self._fig_to_base64(fig)
            plt.close(fig)

            return {
                "success": True,
                "image": img_str,
                "format": "base64_png",
                "control_limits": {
                    "ucl": float(ucl),
                    "cl": float(mean),
                    "lcl": float(lcl)
                },
                "out_of_control": int(np.sum(ooc))
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def plot_fmea_pareto(
        self,
        items: List[Dict[str, Any]],
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create FMEA Pareto chart by RPN.

        Args:
            items: List of {name, rpn} or {name, severity, occurrence, detection}
            options: Styling options
        """
        try:
            plt = self.plt
            np = self.np
            options = options or {}

            # Calculate RPN if needed
            for item in items:
                if "rpn" not in item:
                    item["rpn"] = item.get("severity", 1) * item.get("occurrence", 1) * item.get("detection", 1)

            # Sort by RPN descending
            items = sorted(items, key=lambda x: x["rpn"], reverse=True)

            names = [i["name"][:20] for i in items]
            rpns = [i["rpn"] for i in items]
            total_rpn = sum(rpns)

            # Cumulative percentage
            cumulative = np.cumsum(rpns) / total_rpn * 100

            fig, ax1 = plt.subplots(figsize=options.get("figsize", (12, 6)))

            # Bar chart
            x = range(len(names))
            bars = ax1.bar(x, rpns, color='steelblue', alpha=0.8)
            ax1.set_xlabel('Failure Mode')
            ax1.set_ylabel('Risk Priority Number (RPN)', color='steelblue')
            ax1.tick_params(axis='y', labelcolor='steelblue')

            # Add 80% line
            ax1.axhline(y=total_rpn * 0.8 / len(items) if len(items) > 0 else 0,
                       color='gray', linestyle=':', alpha=0.5)

            # Cumulative line
            ax2 = ax1.twinx()
            ax2.plot(x, cumulative, 'r-o', linewidth=2, markersize=6)
            ax2.set_ylabel('Cumulative %', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80%')
            ax2.set_ylim(0, 105)

            # X-axis labels
            ax1.set_xticks(x)
            ax1.set_xticklabels(names, rotation=45, ha='right')

            ax1.set_title(options.get("title", "FMEA Pareto Analysis"))

            plt.tight_layout()
            img_str = self._fig_to_base64(fig)
            plt.close(fig)

            # Find items contributing to 80%
            critical_idx = np.searchsorted(cumulative, 80) + 1
            critical_items = [items[i]["name"] for i in range(min(critical_idx, len(items)))]

            return {
                "success": True,
                "image": img_str,
                "format": "base64_png",
                "vital_few": critical_items,
                "total_rpn": float(total_rpn)
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def plot_n_squared(
        self,
        interfaces: List[Dict[str, Any]],
        systems: List[str],
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create N² (N-squared) interface diagram.

        Args:
            interfaces: List of {from, to, type} interface definitions
            systems: List of system/subsystem names
            options: Styling options
        """
        try:
            plt = self.plt
            np = self.np
            options = options or {}

            n = len(systems)
            fig, ax = plt.subplots(figsize=options.get("figsize", (10, 10)))

            # Create interface matrix
            matrix = np.zeros((n, n), dtype=int)
            for iface in interfaces:
                try:
                    i = systems.index(iface["from"])
                    j = systems.index(iface["to"])
                    matrix[i, j] = 1
                except ValueError:
                    continue

            # Draw grid
            for i in range(n + 1):
                ax.axhline(y=i, color='black', linewidth=1)
                ax.axvline(x=i, color='black', linewidth=1)

            # Fill diagonal with system names
            for i, name in enumerate(systems):
                ax.add_patch(plt.Rectangle((i, n - 1 - i), 1, 1,
                            facecolor='lightblue', edgecolor='black'))
                ax.text(i + 0.5, n - 0.5 - i, name[:10],
                       ha='center', va='center', fontsize=8, fontweight='bold')

            # Fill interfaces
            for i in range(n):
                for j in range(n):
                    if i != j and matrix[i, j]:
                        color = 'lightgreen' if j > i else 'lightyellow'
                        ax.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1,
                                    facecolor=color, edgecolor='black'))
                        ax.text(j + 0.5, n - 0.5 - i, '→',
                               ha='center', va='center', fontsize=12)

            ax.set_xlim(0, n)
            ax.set_ylim(0, n)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(options.get("title", "N² Interface Diagram"))

            # Legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='lightblue', label='System'),
                Patch(facecolor='lightgreen', label='Output (below diagonal)'),
                Patch(facecolor='lightyellow', label='Input (above diagonal)')
            ]
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

            plt.tight_layout()
            img_str = self._fig_to_base64(fig)
            plt.close(fig)

            return {
                "success": True,
                "image": img_str,
                "format": "base64_png",
                "interface_count": int(matrix.sum())
            }

        except Exception as e:
            return {"success": False, "error": str(e)}


def specialist_plot_tool(
    plot_type: str,
    data: str,
    options: Optional[str] = None,
) -> ToolResult:
    """Tool function for specialist plotting."""
    plotter = SpecialistPlotter()

    try:
        data_dict = json.loads(data)
        opts = json.loads(options) if options else {}
    except json.JSONDecodeError as e:
        return ToolResult(
            status=ToolStatus.INVALID_INPUT,
            result=None,
            error=f"Invalid JSON: {e}",
        )

    try:
        if plot_type == "bode":
            result = plotter.plot_bode(
                data_dict.get("num"),
                data_dict.get("den"),
                data_dict.get("freq_range"),
                opts
            )
        elif plot_type == "nyquist":
            result = plotter.plot_nyquist(
                data_dict.get("num"),
                data_dict.get("den"),
                data_dict.get("freq_range"),
                opts
            )
        elif plot_type == "step_response":
            result = plotter.plot_step_response(
                data_dict.get("time"),
                data_dict.get("response"),
                data_dict.get("reference", 1.0),
                opts
            )
        elif plot_type == "gantt":
            result = plotter.plot_gantt(
                data_dict.get("tasks"),
                opts
            )
        elif plot_type == "earned_value":
            result = plotter.plot_earned_value(
                data_dict.get("periods"),
                data_dict.get("pv"),
                data_dict.get("ev"),
                data_dict.get("ac"),
                opts
            )
        elif plot_type == "weibull":
            result = plotter.plot_weibull(
                data_dict.get("beta"),
                data_dict.get("eta"),
                data_dict.get("gamma", 0),
                data_dict.get("time_range"),
                opts
            )
        elif plot_type == "fault_tree":
            result = plotter.plot_fault_tree(
                data_dict.get("tree_structure"),
                opts
            )
        elif plot_type == "capability":
            result = plotter.plot_capability(
                data_dict.get("data"),
                data_dict.get("lsl"),
                data_dict.get("usl"),
                opts
            )
        elif plot_type == "control_chart":
            result = plotter.plot_control_chart(
                data_dict.get("data"),
                data_dict.get("subgroup_size", 1),
                data_dict.get("chart_type", "individuals"),
                opts
            )
        elif plot_type == "fmea_pareto":
            result = plotter.plot_fmea_pareto(
                data_dict.get("items"),
                opts
            )
        elif plot_type == "n_squared":
            result = plotter.plot_n_squared(
                data_dict.get("interfaces"),
                data_dict.get("systems"),
                opts
            )
        else:
            return ToolResult(
                status=ToolStatus.INVALID_INPUT,
                result=None,
                error=f"Unknown plot type: {plot_type}. Available: bode, nyquist, step_response, gantt, earned_value, weibull, fault_tree, capability, control_chart, fmea_pareto, n_squared",
            )

        if result.get("success"):
            output = f"Specialist plot generated ({plot_type})"
            if "metrics" in result:
                output += f"\nMetrics: {json.dumps(result['metrics'], indent=2)}"

            return ToolResult(
                status=ToolStatus.SUCCESS,
                result=output,
                metadata=result,
            )
        else:
            return ToolResult(
                status=ToolStatus.ERROR,
                result=None,
                error=result.get("error"),
            )

    except Exception as e:
        return ToolResult(
            status=ToolStatus.ERROR,
            result=None,
            error=str(e),
        )


def create_specialist_plotter_tool() -> Tool:
    """Create the specialist plotter tool."""
    return Tool(
        name="specialist_plot",
        description="""Create domain-specific engineering visualizations.

Plot types:
- bode: Bode plot (magnitude/phase) for transfer functions
  Data: {num: [...], den: [...], freq_range: [min, max]}

- nyquist: Nyquist stability plot
  Data: {num: [...], den: [...]}

- step_response: System step response with metrics
  Data: {time: [...], response: [...], reference: 1.0}

- gantt: Gantt chart for project management
  Data: {tasks: [{name, start, duration, critical}, ...]}

- earned_value: EVM S-curve chart
  Data: {periods: [...], pv: [...], ev: [...], ac: [...]}

- weibull: Weibull probability plot (reliability)
  Data: {beta: 2.0, eta: 1000, gamma: 0}

- fault_tree: Simplified fault tree diagram
  Data: {tree_structure: {name, gate, children, probability}}

- capability: Process capability histogram
  Data: {data: [...], lsl: value, usl: value}

- control_chart: SPC control chart
  Data: {data: [...], subgroup_size: 1, chart_type: "individuals"}

- fmea_pareto: FMEA Pareto chart by RPN
  Data: {items: [{name, rpn} or {name, severity, occurrence, detection}]}

- n_squared: N² interface diagram
  Data: {systems: [...], interfaces: [{from, to}]}

Returns base64-encoded PNG images.""",
        parameters=[
            ToolParameter(
                name="plot_type",
                type="string",
                description="Type of specialist plot to create",
                required=True,
                enum=["bode", "nyquist", "step_response", "gantt", "earned_value",
                      "weibull", "fault_tree", "capability", "control_chart",
                      "fmea_pareto", "n_squared"]
            ),
            ToolParameter(
                name="data",
                type="string",
                description="JSON object with plot data. Contents depend on plot_type.",
                required=True
            ),
            ToolParameter(
                name="options",
                type="string",
                description="JSON object with styling options (title, figsize, etc.)",
                required=False
            )
        ],
        execute_fn=specialist_plot_tool
    )


def register_specialist_plot_tools(registry: ToolRegistry) -> None:
    """Register specialist plot tools with the registry."""
    registry.register(create_specialist_plotter_tool())
