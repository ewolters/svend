"""
Finance Tool - Financial and Economic Calculations

Provides verified solutions for:
- Time value of money (PV, FV, annuities)
- Loan amortization
- Investment analysis (NPV, IRR, ROI)
- Bond pricing
- Depreciation methods
"""

from typing import Optional, Dict, Any, List, Tuple
from decimal import Decimal, getcontext, ROUND_HALF_UP
import math

from .registry import Tool, ToolParameter, ToolResult, ToolStatus, ToolRegistry


# Financial precision
getcontext().prec = 20


class FinanceCalculator:
    """Financial calculations with high precision."""

    @staticmethod
    def round_currency(value: float, decimals: int = 2) -> float:
        """Round to currency precision."""
        return float(Decimal(str(value)).quantize(Decimal(10) ** -decimals, rounding=ROUND_HALF_UP))

    # === Time Value of Money ===

    @staticmethod
    def future_value(
        present_value: float,
        rate: float,
        periods: int,
        compounding: str = "annual"
    ) -> Dict[str, Any]:
        """
        Calculate future value of a lump sum.
        FV = PV * (1 + r/n)^(n*t)
        """
        # Compounding frequency
        n = {"annual": 1, "semiannual": 2, "quarterly": 4, "monthly": 12, "daily": 365, "continuous": float('inf')}.get(compounding, 1)

        if compounding == "continuous":
            fv = present_value * math.exp(rate * periods)
        else:
            fv = present_value * ((1 + rate / n) ** (n * periods))

        return {
            "present_value": present_value,
            "future_value": FinanceCalculator.round_currency(fv),
            "rate": rate,
            "periods": periods,
            "compounding": compounding,
            "total_interest": FinanceCalculator.round_currency(fv - present_value)
        }

    @staticmethod
    def present_value(
        future_value: float,
        rate: float,
        periods: int,
        compounding: str = "annual"
    ) -> Dict[str, Any]:
        """
        Calculate present value of a future sum.
        PV = FV / (1 + r/n)^(n*t)
        """
        n = {"annual": 1, "semiannual": 2, "quarterly": 4, "monthly": 12, "daily": 365, "continuous": float('inf')}.get(compounding, 1)

        if compounding == "continuous":
            pv = future_value / math.exp(rate * periods)
        else:
            pv = future_value / ((1 + rate / n) ** (n * periods))

        return {
            "future_value": future_value,
            "present_value": FinanceCalculator.round_currency(pv),
            "rate": rate,
            "periods": periods,
            "compounding": compounding,
            "discount": FinanceCalculator.round_currency(future_value - pv)
        }

    @staticmethod
    def annuity_fv(
        payment: float,
        rate: float,
        periods: int,
        annuity_type: str = "ordinary"  # ordinary or due
    ) -> Dict[str, Any]:
        """
        Future value of annuity.
        Ordinary: FV = PMT * [(1+r)^n - 1] / r
        Due: FV = PMT * [(1+r)^n - 1] / r * (1+r)
        """
        if rate == 0:
            fv = payment * periods
        else:
            fv = payment * (((1 + rate) ** periods - 1) / rate)
            if annuity_type == "due":
                fv *= (1 + rate)

        return {
            "payment": payment,
            "future_value": FinanceCalculator.round_currency(fv),
            "rate": rate,
            "periods": periods,
            "type": annuity_type,
            "total_contributions": FinanceCalculator.round_currency(payment * periods),
            "total_interest": FinanceCalculator.round_currency(fv - payment * periods)
        }

    @staticmethod
    def annuity_pv(
        payment: float,
        rate: float,
        periods: int,
        annuity_type: str = "ordinary"
    ) -> Dict[str, Any]:
        """
        Present value of annuity.
        Ordinary: PV = PMT * [1 - (1+r)^-n] / r
        Due: PV = PMT * [1 - (1+r)^-n] / r * (1+r)
        """
        if rate == 0:
            pv = payment * periods
        else:
            pv = payment * ((1 - (1 + rate) ** -periods) / rate)
            if annuity_type == "due":
                pv *= (1 + rate)

        return {
            "payment": payment,
            "present_value": FinanceCalculator.round_currency(pv),
            "rate": rate,
            "periods": periods,
            "type": annuity_type
        }

    @staticmethod
    def payment_for_loan(
        principal: float,
        rate: float,
        periods: int
    ) -> Dict[str, Any]:
        """
        Calculate payment for a loan.
        PMT = PV * r / [1 - (1+r)^-n]
        """
        if rate == 0:
            payment = principal / periods
        else:
            payment = principal * (rate / (1 - (1 + rate) ** -periods))

        total_paid = payment * periods
        total_interest = total_paid - principal

        return {
            "principal": principal,
            "payment": FinanceCalculator.round_currency(payment),
            "rate": rate,
            "periods": periods,
            "total_paid": FinanceCalculator.round_currency(total_paid),
            "total_interest": FinanceCalculator.round_currency(total_interest)
        }

    # === Loan Amortization ===

    @staticmethod
    def amortization_schedule(
        principal: float,
        annual_rate: float,
        years: int,
        payments_per_year: int = 12
    ) -> Dict[str, Any]:
        """Generate full amortization schedule."""
        rate = annual_rate / payments_per_year
        periods = years * payments_per_year

        if rate == 0:
            payment = principal / periods
        else:
            payment = principal * (rate / (1 - (1 + rate) ** -periods))

        schedule = []
        balance = principal
        total_interest = 0

        for i in range(1, periods + 1):
            interest_payment = balance * rate
            principal_payment = payment - interest_payment
            balance -= principal_payment
            total_interest += interest_payment

            schedule.append({
                "period": i,
                "payment": FinanceCalculator.round_currency(payment),
                "principal": FinanceCalculator.round_currency(principal_payment),
                "interest": FinanceCalculator.round_currency(interest_payment),
                "balance": FinanceCalculator.round_currency(max(0, balance))
            })

        return {
            "principal": principal,
            "annual_rate": annual_rate,
            "years": years,
            "payments_per_year": payments_per_year,
            "monthly_payment": FinanceCalculator.round_currency(payment),
            "total_paid": FinanceCalculator.round_currency(payment * periods),
            "total_interest": FinanceCalculator.round_currency(total_interest),
            "schedule": schedule if periods <= 60 else schedule[:12] + [{"...": f"{periods - 24} more periods"}] + schedule[-12:]
        }

    # === Investment Analysis ===

    @staticmethod
    def npv(
        rate: float,
        cash_flows: List[float]
    ) -> Dict[str, Any]:
        """
        Net Present Value.
        NPV = Σ CF_t / (1+r)^t
        """
        npv_value = sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cash_flows))

        return {
            "rate": rate,
            "cash_flows": cash_flows,
            "npv": FinanceCalculator.round_currency(npv_value),
            "decision": "Accept" if npv_value > 0 else "Reject"
        }

    @staticmethod
    def irr(
        cash_flows: List[float],
        guess: float = 0.1,
        max_iterations: int = 1000,
        tolerance: float = 1e-10
    ) -> Dict[str, Any]:
        """
        Internal Rate of Return using Newton-Raphson.
        Find r where NPV = 0.
        """
        rate = guess

        for _ in range(max_iterations):
            npv = sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cash_flows))
            npv_prime = sum(-t * cf / ((1 + rate) ** (t + 1)) for t, cf in enumerate(cash_flows))

            if abs(npv_prime) < 1e-15:
                break

            new_rate = rate - npv / npv_prime

            if abs(new_rate - rate) < tolerance:
                return {
                    "cash_flows": cash_flows,
                    "irr": round(new_rate, 6),
                    "irr_percent": f"{new_rate * 100:.4f}%"
                }

            rate = new_rate

        return {
            "cash_flows": cash_flows,
            "irr": None,
            "error": "IRR did not converge"
        }

    @staticmethod
    def roi(
        initial_investment: float,
        final_value: float,
        periods: Optional[int] = None
    ) -> Dict[str, Any]:
        """Return on Investment."""
        gain = final_value - initial_investment
        roi_simple = gain / initial_investment

        result = {
            "initial_investment": initial_investment,
            "final_value": final_value,
            "gain": FinanceCalculator.round_currency(gain),
            "roi": round(roi_simple, 4),
            "roi_percent": f"{roi_simple * 100:.2f}%"
        }

        if periods:
            # Annualized ROI
            annualized = ((1 + roi_simple) ** (1 / periods)) - 1
            result["periods"] = periods
            result["annualized_roi"] = round(annualized, 4)
            result["annualized_percent"] = f"{annualized * 100:.2f}%"

        return result

    @staticmethod
    def payback_period(
        initial_investment: float,
        cash_flows: List[float]
    ) -> Dict[str, Any]:
        """Calculate payback period."""
        cumulative = 0
        for i, cf in enumerate(cash_flows):
            cumulative += cf
            if cumulative >= initial_investment:
                # Interpolate for partial period
                if i == 0:
                    partial = initial_investment / cf
                else:
                    prev_cumulative = cumulative - cf
                    remaining = initial_investment - prev_cumulative
                    partial = i + remaining / cf

                return {
                    "initial_investment": initial_investment,
                    "payback_period": round(partial, 2),
                    "unit": "periods"
                }

        return {
            "initial_investment": initial_investment,
            "payback_period": None,
            "error": "Investment not recovered within given cash flows"
        }

    # === Bond Pricing ===

    @staticmethod
    def bond_price(
        face_value: float,
        coupon_rate: float,
        years_to_maturity: float,
        yield_to_maturity: float,
        payments_per_year: int = 2
    ) -> Dict[str, Any]:
        """
        Calculate bond price.
        Price = C * [1 - (1+r)^-n] / r + F / (1+r)^n
        """
        coupon = (face_value * coupon_rate) / payments_per_year
        rate = yield_to_maturity / payments_per_year
        periods = int(years_to_maturity * payments_per_year)

        if rate == 0:
            pv_coupons = coupon * periods
        else:
            pv_coupons = coupon * ((1 - (1 + rate) ** -periods) / rate)

        pv_face = face_value / ((1 + rate) ** periods)
        price = pv_coupons + pv_face

        return {
            "face_value": face_value,
            "coupon_rate": coupon_rate,
            "years_to_maturity": years_to_maturity,
            "yield_to_maturity": yield_to_maturity,
            "price": FinanceCalculator.round_currency(price),
            "pv_of_coupons": FinanceCalculator.round_currency(pv_coupons),
            "pv_of_face_value": FinanceCalculator.round_currency(pv_face),
            "premium_discount": "Premium" if price > face_value else "Discount" if price < face_value else "Par"
        }

    @staticmethod
    def bond_ytm(
        price: float,
        face_value: float,
        coupon_rate: float,
        years_to_maturity: float,
        payments_per_year: int = 2
    ) -> Dict[str, Any]:
        """Calculate yield to maturity using Newton-Raphson."""
        coupon = (face_value * coupon_rate) / payments_per_year
        periods = int(years_to_maturity * payments_per_year)

        # Initial guess
        ytm = coupon_rate

        for _ in range(1000):
            rate = ytm / payments_per_year
            if abs(rate) < 1e-10:
                calc_price = coupon * periods + face_value
            else:
                calc_price = coupon * ((1 - (1 + rate) ** -periods) / rate) + face_value / ((1 + rate) ** periods)

            # Derivative
            if abs(rate) < 1e-10:
                derivative = 0
            else:
                derivative = (-coupon * periods / (payments_per_year * (1 + rate) ** (periods + 1))) + \
                            coupon / (payments_per_year * rate ** 2) * (1 - (1 + rate) ** -periods) - \
                            face_value * periods / (payments_per_year * (1 + rate) ** (periods + 1))

            if abs(derivative) < 1e-15:
                break

            new_ytm = ytm - (calc_price - price) / derivative

            if abs(new_ytm - ytm) < 1e-10:
                return {
                    "price": price,
                    "face_value": face_value,
                    "coupon_rate": coupon_rate,
                    "years_to_maturity": years_to_maturity,
                    "ytm": round(new_ytm, 6),
                    "ytm_percent": f"{new_ytm * 100:.4f}%"
                }

            ytm = new_ytm

        return {"error": "YTM calculation did not converge"}

    # === Depreciation ===

    @staticmethod
    def depreciation(
        cost: float,
        salvage: float,
        life: int,
        method: str = "straight_line"
    ) -> Dict[str, Any]:
        """Calculate depreciation schedule."""
        depreciable = cost - salvage

        schedule = []

        if method == "straight_line":
            annual = depreciable / life
            book_value = cost
            for year in range(1, life + 1):
                book_value -= annual
                schedule.append({
                    "year": year,
                    "depreciation": FinanceCalculator.round_currency(annual),
                    "accumulated": FinanceCalculator.round_currency(annual * year),
                    "book_value": FinanceCalculator.round_currency(max(salvage, book_value))
                })

        elif method == "double_declining":
            rate = 2 / life
            book_value = cost
            accumulated = 0
            for year in range(1, life + 1):
                dep = min(book_value * rate, book_value - salvage)
                accumulated += dep
                book_value -= dep
                schedule.append({
                    "year": year,
                    "depreciation": FinanceCalculator.round_currency(dep),
                    "accumulated": FinanceCalculator.round_currency(accumulated),
                    "book_value": FinanceCalculator.round_currency(book_value)
                })

        elif method == "sum_of_years":
            syd = life * (life + 1) / 2
            book_value = cost
            accumulated = 0
            for year in range(1, life + 1):
                dep = depreciable * (life - year + 1) / syd
                accumulated += dep
                book_value -= dep
                schedule.append({
                    "year": year,
                    "depreciation": FinanceCalculator.round_currency(dep),
                    "accumulated": FinanceCalculator.round_currency(accumulated),
                    "book_value": FinanceCalculator.round_currency(book_value)
                })

        else:
            return {"error": f"Unknown method: {method}"}

        return {
            "cost": cost,
            "salvage": salvage,
            "life": life,
            "method": method,
            "total_depreciation": FinanceCalculator.round_currency(depreciable),
            "schedule": schedule
        }

    # === Compound Interest ===

    @staticmethod
    def compound_interest(
        principal: float,
        rate: float,
        time: float,
        compounding: str = "annual"
    ) -> Dict[str, Any]:
        """Calculate compound interest with different compounding frequencies."""
        result = FinanceCalculator.future_value(principal, rate, time, compounding)
        result["interest_earned"] = result["total_interest"]
        return result

    @staticmethod
    def rule_of_72(rate: float) -> Dict[str, Any]:
        """Estimate doubling time using Rule of 72."""
        if rate <= 0:
            return {"error": "Rate must be positive"}

        rate_percent = rate * 100 if rate < 1 else rate
        approx_years = 72 / rate_percent
        exact_years = math.log(2) / math.log(1 + rate_percent / 100)

        return {
            "rate_percent": rate_percent,
            "rule_of_72_years": round(approx_years, 2),
            "exact_years": round(exact_years, 4)
        }


def finance_tool(
    operation: str,
    principal: Optional[float] = None,
    present_value: Optional[float] = None,
    future_value: Optional[float] = None,
    payment: Optional[float] = None,
    rate: Optional[float] = None,
    periods: Optional[int] = None,
    years: Optional[float] = None,
    cash_flows: Optional[List[float]] = None,
    compounding: str = "annual",
    annuity_type: str = "ordinary",
    face_value: Optional[float] = None,
    coupon_rate: Optional[float] = None,
    yield_to_maturity: Optional[float] = None,
    price: Optional[float] = None,
    cost: Optional[float] = None,
    salvage: Optional[float] = None,
    life: Optional[int] = None,
    method: str = "straight_line",
    initial_investment: Optional[float] = None,
    final_value: Optional[float] = None,
) -> ToolResult:
    """Execute financial calculation."""
    try:
        calc = FinanceCalculator()

        if operation == "future_value":
            if present_value is None or rate is None or periods is None:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need present_value, rate, periods")
            result = calc.future_value(present_value, rate, periods, compounding)

        elif operation == "present_value":
            if future_value is None or rate is None or periods is None:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need future_value, rate, periods")
            result = calc.present_value(future_value, rate, periods, compounding)

        elif operation == "annuity_fv":
            if payment is None or rate is None or periods is None:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need payment, rate, periods")
            result = calc.annuity_fv(payment, rate, periods, annuity_type)

        elif operation == "annuity_pv":
            if payment is None or rate is None or periods is None:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need payment, rate, periods")
            result = calc.annuity_pv(payment, rate, periods, annuity_type)

        elif operation == "loan_payment":
            if principal is None or rate is None or periods is None:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need principal, rate, periods")
            result = calc.payment_for_loan(principal, rate, periods)

        elif operation == "amortization":
            if principal is None or rate is None or years is None:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need principal, rate, years")
            result = calc.amortization_schedule(principal, rate, int(years))

        elif operation == "npv":
            if rate is None or cash_flows is None:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need rate and cash_flows")
            result = calc.npv(rate, cash_flows)

        elif operation == "irr":
            if cash_flows is None:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need cash_flows")
            result = calc.irr(cash_flows)

        elif operation == "roi":
            if initial_investment is None or final_value is None:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need initial_investment and final_value")
            result = calc.roi(initial_investment, final_value, periods)

        elif operation == "payback":
            if initial_investment is None or cash_flows is None:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need initial_investment and cash_flows")
            result = calc.payback_period(initial_investment, cash_flows)

        elif operation == "bond_price":
            if face_value is None or coupon_rate is None or years is None or yield_to_maturity is None:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need face_value, coupon_rate, years, yield_to_maturity")
            result = calc.bond_price(face_value, coupon_rate, years, yield_to_maturity)

        elif operation == "bond_ytm":
            if price is None or face_value is None or coupon_rate is None or years is None:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need price, face_value, coupon_rate, years")
            result = calc.bond_ytm(price, face_value, coupon_rate, years)

        elif operation == "depreciation":
            if cost is None or salvage is None or life is None:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need cost, salvage, life")
            result = calc.depreciation(cost, salvage, life, method)

        elif operation == "compound":
            if principal is None or rate is None:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need principal and rate")
            time = years or periods or 1
            result = calc.compound_interest(principal, rate, time, compounding)

        elif operation == "rule_of_72":
            if rate is None:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need rate")
            result = calc.rule_of_72(rate)

        else:
            return ToolResult(
                status=ToolStatus.INVALID_INPUT,
                output=None,
                error=f"Unknown operation: {operation}. Valid: future_value, present_value, annuity_fv, annuity_pv, loan_payment, amortization, npv, irr, roi, payback, bond_price, bond_ytm, depreciation, compound, rule_of_72"
            )

        if "error" in result:
            return ToolResult(status=ToolStatus.ERROR, output=None, error=result["error"])

        # Format output
        output_lines = [f"{k}: {v}" for k, v in result.items() if k != "schedule"]
        if "schedule" in result:
            output_lines.append(f"Schedule: {len(result.get('schedule', []))} periods")

        return ToolResult(
            status=ToolStatus.SUCCESS,
            output="\n".join(output_lines),
            metadata=result
        )

    except Exception as e:
        return ToolResult(status=ToolStatus.ERROR, output=None, error=str(e))


def create_finance_tool() -> Tool:
    """Create finance tool."""
    return Tool(
        name="finance",
        description="Financial calculations: time value of money (PV, FV), annuities, loan amortization, NPV, IRR, ROI, bond pricing, depreciation.",
        parameters=[
            ToolParameter(
                name="operation",
                description="Operation: future_value, present_value, annuity_fv, annuity_pv, loan_payment, amortization, npv, irr, roi, payback, bond_price, bond_ytm, depreciation, compound, rule_of_72",
                type="string",
                required=True,
            ),
            ToolParameter(name="principal", description="Loan/investment principal", type="number", required=False),
            ToolParameter(name="present_value", description="Present value", type="number", required=False),
            ToolParameter(name="future_value", description="Future value", type="number", required=False),
            ToolParameter(name="payment", description="Periodic payment", type="number", required=False),
            ToolParameter(name="rate", description="Interest rate (as decimal, e.g., 0.05 for 5%)", type="number", required=False),
            ToolParameter(name="periods", description="Number of periods", type="number", required=False),
            ToolParameter(name="years", description="Time in years", type="number", required=False),
            ToolParameter(name="cash_flows", description="List of cash flows (CF0, CF1, ...)", type="array", required=False),
            ToolParameter(name="compounding", description="Compounding frequency: annual, semiannual, quarterly, monthly, daily, continuous", type="string", required=False),
            ToolParameter(name="annuity_type", description="ordinary or due", type="string", required=False),
            ToolParameter(name="face_value", description="Bond face value", type="number", required=False),
            ToolParameter(name="coupon_rate", description="Bond coupon rate", type="number", required=False),
            ToolParameter(name="yield_to_maturity", description="Bond YTM", type="number", required=False),
            ToolParameter(name="price", description="Bond price", type="number", required=False),
            ToolParameter(name="cost", description="Asset cost (for depreciation)", type="number", required=False),
            ToolParameter(name="salvage", description="Salvage value", type="number", required=False),
            ToolParameter(name="life", description="Asset life in years", type="number", required=False),
            ToolParameter(name="method", description="Depreciation method: straight_line, double_declining, sum_of_years", type="string", required=False),
            ToolParameter(name="initial_investment", description="Initial investment (for ROI/payback)", type="number", required=False),
            ToolParameter(name="final_value", description="Final value (for ROI)", type="number", required=False),
        ],
        execute_fn=finance_tool,
        timeout_ms=10000,
    )


def register_finance_tools(registry: ToolRegistry) -> None:
    """Register finance tools."""
    registry.register(create_finance_tool())
