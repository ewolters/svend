"""
Economics tools for Svend reasoning system.

Covers micro/macro economics calculations:
- Supply/demand equilibrium
- Elasticity calculations
- Present/future value
- GDP components
- Marginal analysis
- Game theory (simple)
"""

import math
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
from enum import Enum


class ElasticityType(Enum):
    ELASTIC = "elastic"           # |E| > 1
    INELASTIC = "inelastic"       # |E| < 1
    UNIT_ELASTIC = "unit_elastic" # |E| = 1
    PERFECTLY_ELASTIC = "perfectly_elastic"     # |E| = infinity
    PERFECTLY_INELASTIC = "perfectly_inelastic" # |E| = 0


@dataclass
class EquilibriumResult:
    """Result of supply/demand equilibrium calculation."""
    price: float
    quantity: float
    consumer_surplus: float
    producer_surplus: float
    total_surplus: float
    steps: List[str]


@dataclass
class ElasticityResult:
    """Result of elasticity calculation."""
    elasticity: float
    elasticity_type: ElasticityType
    interpretation: str
    steps: List[str]


@dataclass
class TimeValueResult:
    """Result of time value of money calculation."""
    present_value: float
    future_value: float
    periods: int
    rate: float
    steps: List[str]


class EconomicsEngine:
    """
    Economics calculation engine.

    Handles common economics problems with step-by-step solutions.
    """

    def __init__(self):
        self.precision = 4

    def find_equilibrium(
        self,
        demand_intercept: float,
        demand_slope: float,
        supply_intercept: float,
        supply_slope: float
    ) -> EquilibriumResult:
        """
        Find market equilibrium given linear supply and demand curves.

        Demand: Qd = demand_intercept + demand_slope * P  (slope usually negative)
        Supply: Qs = supply_intercept + supply_slope * P  (slope usually positive)

        Args:
            demand_intercept: Quantity demanded when price is 0
            demand_slope: Change in Qd per unit change in P (negative)
            supply_intercept: Quantity supplied when price is 0
            supply_slope: Change in Qs per unit change in P (positive)

        Returns:
            EquilibriumResult with price, quantity, and surplus calculations
        """
        steps = []

        # Step 1: Set up equations
        steps.append(f"Demand curve: Qd = {demand_intercept} + ({demand_slope})P")
        steps.append(f"Supply curve: Qs = {supply_intercept} + ({supply_slope})P")

        # Step 2: Set Qd = Qs and solve for P
        steps.append("At equilibrium: Qd = Qs")
        steps.append(f"{demand_intercept} + ({demand_slope})P = {supply_intercept} + ({supply_slope})P")

        # Solve: demand_intercept - supply_intercept = supply_slope*P - demand_slope*P
        price_diff = demand_intercept - supply_intercept
        slope_diff = supply_slope - demand_slope

        if abs(slope_diff) < 1e-10:
            raise ValueError("Parallel curves - no unique equilibrium")

        eq_price = price_diff / slope_diff
        steps.append(f"{price_diff} = ({slope_diff})P")
        steps.append(f"P* = {price_diff}/{slope_diff} = {round(eq_price, self.precision)}")

        # Step 3: Find equilibrium quantity
        eq_quantity = demand_intercept + demand_slope * eq_price
        steps.append(f"Q* = {demand_intercept} + ({demand_slope})({round(eq_price, self.precision)}) = {round(eq_quantity, self.precision)}")

        if eq_price < 0 or eq_quantity < 0:
            steps.append("Warning: Negative equilibrium values - check curve parameters")

        # Step 4: Calculate surpluses (assuming linear curves)
        # Consumer surplus: area of triangle above P*, below demand curve
        # Need price where Qd = 0 (choke price)
        if abs(demand_slope) > 1e-10:
            choke_price = -demand_intercept / demand_slope
            consumer_surplus = 0.5 * eq_quantity * (choke_price - eq_price)
            steps.append(f"Consumer surplus = 0.5 × {round(eq_quantity, self.precision)} × ({round(choke_price, self.precision)} - {round(eq_price, self.precision)}) = {round(consumer_surplus, self.precision)}")
        else:
            consumer_surplus = float('inf')
            steps.append("Consumer surplus: infinite (perfectly elastic demand)")

        # Producer surplus: area of triangle below P*, above supply curve
        if abs(supply_slope) > 1e-10:
            min_price = -supply_intercept / supply_slope  # Price where Qs = 0
            if min_price < eq_price:
                producer_surplus = 0.5 * eq_quantity * (eq_price - max(0, min_price))
            else:
                producer_surplus = 0.5 * eq_quantity * eq_price
            steps.append(f"Producer surplus = {round(producer_surplus, self.precision)}")
        else:
            producer_surplus = float('inf')
            steps.append("Producer surplus: infinite (perfectly elastic supply)")

        total_surplus = consumer_surplus + producer_surplus

        return EquilibriumResult(
            price=round(eq_price, self.precision),
            quantity=round(eq_quantity, self.precision),
            consumer_surplus=round(consumer_surplus, self.precision),
            producer_surplus=round(producer_surplus, self.precision),
            total_surplus=round(total_surplus, self.precision),
            steps=steps
        )

    def calculate_elasticity(
        self,
        p1: float,
        q1: float,
        p2: float,
        q2: float,
        method: str = "midpoint"
    ) -> ElasticityResult:
        """
        Calculate price elasticity of demand.

        Args:
            p1, q1: Initial price and quantity
            p2, q2: New price and quantity
            method: "midpoint" (arc elasticity) or "point"

        Returns:
            ElasticityResult with elasticity value and interpretation
        """
        steps = []

        steps.append(f"Initial point: (P₁={p1}, Q₁={q1})")
        steps.append(f"New point: (P₂={p2}, Q₂={q2})")

        delta_q = q2 - q1
        delta_p = p2 - p1

        if abs(delta_p) < 1e-10:
            raise ValueError("Price change is zero - cannot calculate elasticity")

        if method == "midpoint":
            # Arc elasticity using midpoint method
            avg_q = (q1 + q2) / 2
            avg_p = (p1 + p2) / 2

            pct_change_q = delta_q / avg_q
            pct_change_p = delta_p / avg_p

            steps.append(f"Midpoint method:")
            steps.append(f"  Average Q = ({q1} + {q2})/2 = {avg_q}")
            steps.append(f"  Average P = ({p1} + {p2})/2 = {avg_p}")
            steps.append(f"  %ΔQ = {delta_q}/{avg_q} = {round(pct_change_q, self.precision)}")
            steps.append(f"  %ΔP = {delta_p}/{avg_p} = {round(pct_change_p, self.precision)}")
        else:
            # Point elasticity
            pct_change_q = delta_q / q1
            pct_change_p = delta_p / p1

            steps.append(f"Point elasticity method:")
            steps.append(f"  %ΔQ = {delta_q}/{q1} = {round(pct_change_q, self.precision)}")
            steps.append(f"  %ΔP = {delta_p}/{p1} = {round(pct_change_p, self.precision)}")

        if abs(pct_change_p) < 1e-10:
            elasticity = float('inf') if pct_change_q != 0 else 0
        else:
            elasticity = pct_change_q / pct_change_p

        steps.append(f"Elasticity = %ΔQ / %ΔP = {round(elasticity, self.precision)}")

        # Classify elasticity
        abs_e = abs(elasticity)
        if abs_e == float('inf'):
            e_type = ElasticityType.PERFECTLY_ELASTIC
            interpretation = "Perfectly elastic: any price increase causes quantity demanded to fall to zero"
        elif abs_e == 0:
            e_type = ElasticityType.PERFECTLY_INELASTIC
            interpretation = "Perfectly inelastic: quantity demanded doesn't respond to price changes"
        elif abs_e > 1:
            e_type = ElasticityType.ELASTIC
            interpretation = f"Elastic (|E|={round(abs_e, 2)} > 1): quantity responds more than proportionally to price changes"
        elif abs_e < 1:
            e_type = ElasticityType.INELASTIC
            interpretation = f"Inelastic (|E|={round(abs_e, 2)} < 1): quantity responds less than proportionally to price changes"
        else:
            e_type = ElasticityType.UNIT_ELASTIC
            interpretation = "Unit elastic: percentage change in quantity equals percentage change in price"

        steps.append(f"Classification: {e_type.value}")

        return ElasticityResult(
            elasticity=round(elasticity, self.precision),
            elasticity_type=e_type,
            interpretation=interpretation,
            steps=steps
        )

    def present_value(
        self,
        future_value: float,
        rate: float,
        periods: int,
        compounding: str = "annual"
    ) -> TimeValueResult:
        """
        Calculate present value of a future sum.

        Args:
            future_value: Future amount
            rate: Interest rate (as decimal, e.g., 0.05 for 5%)
            periods: Number of periods
            compounding: "annual", "semi-annual", "quarterly", "monthly", "continuous"

        Returns:
            TimeValueResult with present value
        """
        steps = []

        steps.append(f"Future Value: ${future_value:,.2f}")
        steps.append(f"Rate: {rate*100}% per period")
        steps.append(f"Periods: {periods}")
        steps.append(f"Compounding: {compounding}")

        if compounding == "continuous":
            pv = future_value * math.exp(-rate * periods)
            steps.append(f"PV = FV × e^(-rt)")
            steps.append(f"PV = {future_value} × e^(-{rate} × {periods})")
        else:
            # Adjust rate and periods for compounding frequency
            freq_map = {"annual": 1, "semi-annual": 2, "quarterly": 4, "monthly": 12}
            freq = freq_map.get(compounding, 1)

            adj_rate = rate / freq
            adj_periods = periods * freq

            pv = future_value / ((1 + adj_rate) ** adj_periods)
            steps.append(f"PV = FV / (1 + r)^n")
            steps.append(f"PV = {future_value} / (1 + {adj_rate})^{adj_periods}")

        steps.append(f"PV = ${pv:,.2f}")

        return TimeValueResult(
            present_value=round(pv, 2),
            future_value=future_value,
            periods=periods,
            rate=rate,
            steps=steps
        )

    def future_value(
        self,
        present_value: float,
        rate: float,
        periods: int,
        compounding: str = "annual"
    ) -> TimeValueResult:
        """
        Calculate future value of a present sum.

        Args:
            present_value: Current amount
            rate: Interest rate (as decimal)
            periods: Number of periods
            compounding: "annual", "semi-annual", "quarterly", "monthly", "continuous"

        Returns:
            TimeValueResult with future value
        """
        steps = []

        steps.append(f"Present Value: ${present_value:,.2f}")
        steps.append(f"Rate: {rate*100}% per period")
        steps.append(f"Periods: {periods}")
        steps.append(f"Compounding: {compounding}")

        if compounding == "continuous":
            fv = present_value * math.exp(rate * periods)
            steps.append(f"FV = PV × e^(rt)")
            steps.append(f"FV = {present_value} × e^({rate} × {periods})")
        else:
            freq_map = {"annual": 1, "semi-annual": 2, "quarterly": 4, "monthly": 12}
            freq = freq_map.get(compounding, 1)

            adj_rate = rate / freq
            adj_periods = periods * freq

            fv = present_value * ((1 + adj_rate) ** adj_periods)
            steps.append(f"FV = PV × (1 + r)^n")
            steps.append(f"FV = {present_value} × (1 + {adj_rate})^{adj_periods}")

        steps.append(f"FV = ${fv:,.2f}")

        return TimeValueResult(
            present_value=present_value,
            future_value=round(fv, 2),
            periods=periods,
            rate=rate,
            steps=steps
        )

    def marginal_analysis(
        self,
        total_cost_func: str,
        total_revenue_func: str,
        quantity: float
    ) -> Dict[str, Any]:
        """
        Perform marginal analysis at a given quantity.

        Uses numerical differentiation for simplicity.
        For symbolic analysis, use math_engine.

        Args:
            total_cost_func: Cost function as string (e.g., "100 + 5*q + 0.1*q**2")
            total_revenue_func: Revenue function as string (e.g., "20*q - 0.05*q**2")
            quantity: Quantity to analyze

        Returns:
            Dictionary with MC, MR, profit, and recommendation
        """
        steps = []
        q = quantity
        h = 0.0001  # Small delta for numerical derivative

        # Evaluate functions
        tc = eval(total_cost_func.replace('q', str(q)))
        tr = eval(total_revenue_func.replace('q', str(q)))
        profit = tr - tc

        steps.append(f"At Q = {q}:")
        steps.append(f"  Total Cost = {round(tc, 2)}")
        steps.append(f"  Total Revenue = {round(tr, 2)}")
        steps.append(f"  Profit = {round(profit, 2)}")

        # Marginal cost (derivative of TC)
        tc_plus = eval(total_cost_func.replace('q', str(q + h)))
        mc = (tc_plus - tc) / h

        # Marginal revenue (derivative of TR)
        tr_plus = eval(total_revenue_func.replace('q', str(q + h)))
        mr = (tr_plus - tr) / h

        steps.append(f"  Marginal Cost (MC) = {round(mc, 2)}")
        steps.append(f"  Marginal Revenue (MR) = {round(mr, 2)}")

        # Recommendation
        if abs(mr - mc) < 0.01:
            recommendation = "At profit-maximizing quantity (MR ≈ MC)"
        elif mr > mc:
            recommendation = "Increase production (MR > MC)"
        else:
            recommendation = "Decrease production (MR < MC)"

        steps.append(f"  Recommendation: {recommendation}")

        return {
            "quantity": q,
            "total_cost": round(tc, 2),
            "total_revenue": round(tr, 2),
            "profit": round(profit, 2),
            "marginal_cost": round(mc, 2),
            "marginal_revenue": round(mr, 2),
            "recommendation": recommendation,
            "steps": steps
        }

    def nash_equilibrium_2x2(
        self,
        payoff_matrix: List[List[Tuple[float, float]]]
    ) -> Dict[str, Any]:
        """
        Find Nash equilibrium in a 2x2 game.

        Args:
            payoff_matrix: 2x2 matrix where each cell is (player1_payoff, player2_payoff)
                          Row player chooses row, column player chooses column

        Returns:
            Dictionary with equilibrium strategies and payoffs
        """
        steps = []

        # Extract payoffs
        # payoff_matrix[row][col] = (row_player_payoff, col_player_payoff)
        a11, b11 = payoff_matrix[0][0]
        a12, b12 = payoff_matrix[0][1]
        a21, b21 = payoff_matrix[1][0]
        a22, b22 = payoff_matrix[1][1]

        steps.append("Payoff matrix:")
        steps.append(f"                Col1        Col2")
        steps.append(f"  Row1    ({a11},{b11})    ({a12},{b12})")
        steps.append(f"  Row2    ({a21},{b21})    ({a22},{b22})")

        # Find pure strategy Nash equilibria
        pure_ne = []

        # Check each cell
        cells = [
            ((0, 0), a11, b11, "Row1, Col1"),
            ((0, 1), a12, b12, "Row1, Col2"),
            ((1, 0), a21, b21, "Row2, Col1"),
            ((1, 1), a22, b22, "Row2, Col2"),
        ]

        steps.append("\nChecking for pure strategy Nash equilibria:")

        for (r, c), pay_r, pay_c, name in cells:
            # Is this a best response for row player given column's choice?
            if c == 0:
                row_br = pay_r >= a21 if r == 0 else pay_r >= a11
            else:
                row_br = pay_r >= a22 if r == 0 else pay_r >= a12

            # Is this a best response for column player given row's choice?
            if r == 0:
                col_br = pay_c >= b12 if c == 0 else pay_c >= b11
            else:
                col_br = pay_c >= b22 if c == 0 else pay_c >= b21

            steps.append(f"  {name}: Row BR={row_br}, Col BR={col_br}")

            if row_br and col_br:
                pure_ne.append((name, pay_r, pay_c))

        # Find mixed strategy equilibrium
        # Row player mixes to make column indifferent
        # Column player mixes to make row indifferent

        steps.append("\nMixed strategy equilibrium:")

        # p = probability row plays Row1
        # Column indifferent when: p*b11 + (1-p)*b21 = p*b12 + (1-p)*b22
        denom_p = (b11 - b21) - (b12 - b22)
        if abs(denom_p) > 1e-10:
            p = (b22 - b21) / denom_p
            if 0 <= p <= 1:
                steps.append(f"  Row player: play Row1 with p = {round(p, 4)}")
            else:
                p = None
                steps.append("  No interior mixed strategy for row player")
        else:
            p = None
            steps.append("  Row player indifferent (dominant/dominated strategy)")

        # q = probability column plays Col1
        # Row indifferent when: q*a11 + (1-q)*a12 = q*a21 + (1-q)*a22
        denom_q = (a11 - a12) - (a21 - a22)
        if abs(denom_q) > 1e-10:
            q = (a22 - a12) / denom_q
            if 0 <= q <= 1:
                steps.append(f"  Col player: play Col1 with q = {round(q, 4)}")
            else:
                q = None
                steps.append("  No interior mixed strategy for column player")
        else:
            q = None
            steps.append("  Column player indifferent (dominant/dominated strategy)")

        mixed_ne = None
        if p is not None and q is not None:
            # Calculate expected payoffs
            exp_row = q * (p * a11 + (1-p) * a21) + (1-q) * (p * a12 + (1-p) * a22)
            exp_col = p * (q * b11 + (1-q) * b12) + (1-p) * (q * b21 + (1-q) * b22)
            mixed_ne = {
                "row_strategy": (round(p, 4), round(1-p, 4)),
                "col_strategy": (round(q, 4), round(1-q, 4)),
                "expected_payoffs": (round(exp_row, 4), round(exp_col, 4))
            }
            steps.append(f"  Expected payoffs: Row={round(exp_row, 4)}, Col={round(exp_col, 4)}")

        return {
            "pure_equilibria": pure_ne,
            "mixed_equilibrium": mixed_ne,
            "steps": steps
        }

    def gdp_components(
        self,
        consumption: float,
        investment: float,
        government: float,
        exports: float,
        imports: float
    ) -> Dict[str, Any]:
        """
        Calculate GDP using expenditure approach.

        GDP = C + I + G + (X - M)

        Args:
            consumption: Household consumption spending
            investment: Business investment
            government: Government spending
            exports: Exports
            imports: Imports

        Returns:
            Dictionary with GDP and component shares
        """
        steps = []

        net_exports = exports - imports
        gdp = consumption + investment + government + net_exports

        steps.append("GDP = C + I + G + (X - M)")
        steps.append(f"GDP = {consumption:,.0f} + {investment:,.0f} + {government:,.0f} + ({exports:,.0f} - {imports:,.0f})")
        steps.append(f"GDP = {consumption:,.0f} + {investment:,.0f} + {government:,.0f} + {net_exports:,.0f}")
        steps.append(f"GDP = {gdp:,.0f}")

        # Calculate shares
        c_share = consumption / gdp * 100
        i_share = investment / gdp * 100
        g_share = government / gdp * 100
        nx_share = net_exports / gdp * 100

        steps.append(f"\nComponent shares:")
        steps.append(f"  Consumption: {c_share:.1f}%")
        steps.append(f"  Investment: {i_share:.1f}%")
        steps.append(f"  Government: {g_share:.1f}%")
        steps.append(f"  Net Exports: {nx_share:.1f}%")

        return {
            "gdp": gdp,
            "consumption": consumption,
            "investment": investment,
            "government": government,
            "net_exports": net_exports,
            "shares": {
                "consumption": round(c_share, 1),
                "investment": round(i_share, 1),
                "government": round(g_share, 1),
                "net_exports": round(nx_share, 1)
            },
            "steps": steps
        }


# Tool interface for Svend
def economics_tool(operation: str, **kwargs) -> Dict[str, Any]:
    """
    Main entry point for economics tool.

    Operations:
        - equilibrium: Find market equilibrium
        - elasticity: Calculate price elasticity
        - present_value: Calculate PV
        - future_value: Calculate FV
        - marginal: Marginal analysis
        - nash: Find Nash equilibrium
        - gdp: Calculate GDP components
    """
    engine = EconomicsEngine()

    if operation == "equilibrium":
        result = engine.find_equilibrium(
            kwargs["demand_intercept"],
            kwargs["demand_slope"],
            kwargs["supply_intercept"],
            kwargs["supply_slope"]
        )
        return {
            "price": result.price,
            "quantity": result.quantity,
            "consumer_surplus": result.consumer_surplus,
            "producer_surplus": result.producer_surplus,
            "total_surplus": result.total_surplus,
            "steps": result.steps
        }

    elif operation == "elasticity":
        result = engine.calculate_elasticity(
            kwargs["p1"], kwargs["q1"],
            kwargs["p2"], kwargs["q2"],
            kwargs.get("method", "midpoint")
        )
        return {
            "elasticity": result.elasticity,
            "type": result.elasticity_type.value,
            "interpretation": result.interpretation,
            "steps": result.steps
        }

    elif operation == "present_value":
        result = engine.present_value(
            kwargs["future_value"],
            kwargs["rate"],
            kwargs["periods"],
            kwargs.get("compounding", "annual")
        )
        return {
            "present_value": result.present_value,
            "steps": result.steps
        }

    elif operation == "future_value":
        result = engine.future_value(
            kwargs["present_value"],
            kwargs["rate"],
            kwargs["periods"],
            kwargs.get("compounding", "annual")
        )
        return {
            "future_value": result.future_value,
            "steps": result.steps
        }

    elif operation == "marginal":
        return engine.marginal_analysis(
            kwargs["cost_function"],
            kwargs["revenue_function"],
            kwargs["quantity"]
        )

    elif operation == "nash":
        return engine.nash_equilibrium_2x2(kwargs["payoff_matrix"])

    elif operation == "gdp":
        return engine.gdp_components(
            kwargs["consumption"],
            kwargs["investment"],
            kwargs["government"],
            kwargs["exports"],
            kwargs["imports"]
        )

    else:
        raise ValueError(f"Unknown operation: {operation}")


def register_economics_tools(registry) -> None:
    """Register economics tools with the registry."""
    from .registry import Tool, ToolParameter, ToolResult, ToolStatus

    def _economics_execute(**kwargs) -> ToolResult:
        try:
            result = economics_tool(**kwargs)
            return ToolResult(status=ToolStatus.SUCCESS, output=result)
        except Exception as e:
            return ToolResult(status=ToolStatus.ERROR, output=None, error=str(e))

    registry.register(Tool(
        name="economics",
        description="Economics calculations: equilibrium, elasticity, time value of money, marginal analysis, GDP",
        parameters=[
            ToolParameter(name="operation", type="string", description="Operation: equilibrium, elasticity, present_value, future_value, marginal, nash, gdp", required=True),
            ToolParameter(name="demand_intercept", type="number", description="Demand curve intercept (for equilibrium)", required=False),
            ToolParameter(name="demand_slope", type="number", description="Demand curve slope (for equilibrium)", required=False),
            ToolParameter(name="supply_intercept", type="number", description="Supply curve intercept (for equilibrium)", required=False),
            ToolParameter(name="supply_slope", type="number", description="Supply curve slope (for equilibrium)", required=False),
            ToolParameter(name="p1", type="number", description="Initial price (for elasticity)", required=False),
            ToolParameter(name="q1", type="number", description="Initial quantity (for elasticity)", required=False),
            ToolParameter(name="p2", type="number", description="New price (for elasticity)", required=False),
            ToolParameter(name="q2", type="number", description="New quantity (for elasticity)", required=False),
            ToolParameter(name="future_value", type="number", description="Future value amount (for PV)", required=False),
            ToolParameter(name="present_value", type="number", description="Present value amount (for FV)", required=False),
            ToolParameter(name="rate", type="number", description="Interest rate as decimal (for TVM)", required=False),
            ToolParameter(name="periods", type="integer", description="Number of periods (for TVM)", required=False),
            ToolParameter(name="compounding", type="string", description="Compounding frequency: annual, semi-annual, quarterly, monthly, continuous", required=False),
            ToolParameter(name="cost_function", type="string", description="Cost function as string with 'q' (for marginal)", required=False),
            ToolParameter(name="revenue_function", type="string", description="Revenue function as string with 'q' (for marginal)", required=False),
            ToolParameter(name="quantity", type="number", description="Quantity to analyze (for marginal)", required=False),
            ToolParameter(name="payoff_matrix", type="array", description="2x2 payoff matrix for Nash equilibrium", required=False),
            ToolParameter(name="consumption", type="number", description="Consumption spending (for GDP)", required=False),
            ToolParameter(name="investment", type="number", description="Investment spending (for GDP)", required=False),
            ToolParameter(name="government", type="number", description="Government spending (for GDP)", required=False),
            ToolParameter(name="exports", type="number", description="Exports (for GDP)", required=False),
            ToolParameter(name="imports", type="number", description="Imports (for GDP)", required=False),
        ],
        execute_fn=_economics_execute
    ))
