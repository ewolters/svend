"""
Statistics Tool - Statistical Analysis and Hypothesis Testing

Provides statistical calculations, distributions, hypothesis tests,
and regression analysis.
"""

from typing import Optional, Dict, Any, List, Union
import json
import math

from .registry import Tool, ToolParameter, ToolResult, ToolStatus, ToolRegistry


class StatisticsEngine:
    """
    Statistical analysis engine.

    Uses scipy.stats for distributions and tests.
    """

    def __init__(self):
        self._np = None
        self._stats = None

    @property
    def np(self):
        if self._np is None:
            import numpy as np
            self._np = np
        return self._np

    @property
    def stats(self):
        if self._stats is None:
            from scipy import stats
            self._stats = stats
        return self._stats

    # ==================== DESCRIPTIVE STATISTICS ====================

    def descriptive(
        self,
        data: List[float],
    ) -> Dict[str, Any]:
        """Calculate descriptive statistics for a dataset."""
        try:
            np = self.np
            arr = np.array(data)

            n = len(arr)
            mean = float(np.mean(arr))
            median = float(np.median(arr))
            std = float(np.std(arr, ddof=1))  # Sample std
            var = float(np.var(arr, ddof=1))  # Sample variance
            sem = std / math.sqrt(n)  # Standard error of mean

            q1, q2, q3 = np.percentile(arr, [25, 50, 75])
            iqr = q3 - q1

            # Skewness and kurtosis
            from scipy.stats import skew, kurtosis
            skewness = float(skew(arr))
            kurt = float(kurtosis(arr))

            return {
                "success": True,
                "n": n,
                "mean": mean,
                "median": median,
                "mode": float(self.stats.mode(arr, keepdims=False).mode),
                "std": std,
                "variance": var,
                "sem": sem,
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "range": float(np.max(arr) - np.min(arr)),
                "q1": float(q1),
                "q2": float(q2),
                "q3": float(q3),
                "iqr": float(iqr),
                "skewness": skewness,
                "kurtosis": kurt,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==================== DISTRIBUTIONS ====================

    def distribution(
        self,
        dist_type: str,
        operation: str,
        x: Optional[Union[float, List[float]]] = None,
        params: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Work with probability distributions.

        Distributions: normal, t, chi2, f, binomial, poisson, exponential, uniform
        Operations: pdf, cdf, ppf (inverse cdf), random, stats
        """
        try:
            stats = self.stats
            np = self.np
            params = params or {}

            # Get distribution
            if dist_type == "normal":
                dist = stats.norm(loc=params.get("mean", 0), scale=params.get("std", 1))
            elif dist_type == "t":
                dist = stats.t(df=params.get("df", 10))
            elif dist_type == "chi2":
                dist = stats.chi2(df=params.get("df", 1))
            elif dist_type == "f":
                dist = stats.f(dfn=params.get("dfn", 1), dfd=params.get("dfd", 1))
            elif dist_type == "binomial":
                dist = stats.binom(n=params.get("n", 10), p=params.get("p", 0.5))
            elif dist_type == "poisson":
                dist = stats.poisson(mu=params.get("mu", 1))
            elif dist_type == "exponential":
                dist = stats.expon(scale=params.get("scale", 1))
            elif dist_type == "uniform":
                dist = stats.uniform(loc=params.get("a", 0), scale=params.get("b", 1) - params.get("a", 0))
            elif dist_type == "beta":
                dist = stats.beta(a=params.get("a", 1), b=params.get("b", 1))
            elif dist_type == "gamma":
                dist = stats.gamma(a=params.get("a", 1), scale=params.get("scale", 1))
            else:
                return {"success": False, "error": f"Unknown distribution: {dist_type}"}

            result = {"success": True, "distribution": dist_type, "params": params}

            if operation == "pdf":
                result["pdf"] = float(dist.pdf(x)) if isinstance(x, (int, float)) else [float(v) for v in dist.pdf(x)]
                result["x"] = x

            elif operation == "cdf":
                result["cdf"] = float(dist.cdf(x)) if isinstance(x, (int, float)) else [float(v) for v in dist.cdf(x)]
                result["x"] = x
                result["probability"] = f"P(X <= {x})"

            elif operation == "ppf":  # Inverse CDF / quantile function
                result["ppf"] = float(dist.ppf(x)) if isinstance(x, (int, float)) else [float(v) for v in dist.ppf(x)]
                result["probability"] = x
                result["note"] = f"Value x where P(X <= x) = {x}"

            elif operation == "random":
                n = int(x) if x else 1
                samples = dist.rvs(size=n)
                result["samples"] = samples.tolist() if n > 1 else float(samples)
                result["n"] = n

            elif operation == "stats":
                mean, var, skew, kurt = dist.stats(moments='mvsk')
                result["mean"] = float(mean)
                result["variance"] = float(var)
                result["skewness"] = float(skew)
                result["kurtosis"] = float(kurt)
                result["std"] = float(np.sqrt(var))

            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}

            return result

        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==================== HYPOTHESIS TESTS ====================

    def t_test(
        self,
        sample1: List[float],
        sample2: Optional[List[float]] = None,
        mu: Optional[float] = None,
        alternative: str = "two-sided",
        paired: bool = False,
    ) -> Dict[str, Any]:
        """
        Perform t-test.

        - One-sample: Compare sample to hypothesized mean (mu)
        - Two-sample: Compare two independent samples
        - Paired: Compare paired samples
        """
        try:
            stats = self.stats
            np = self.np

            arr1 = np.array(sample1)

            if sample2 is None and mu is not None:
                # One-sample t-test
                t_stat, p_value = stats.ttest_1samp(arr1, mu, alternative=alternative)
                test_type = "one-sample"
                df = len(arr1) - 1

            elif sample2 is not None:
                arr2 = np.array(sample2)
                if paired:
                    # Paired t-test
                    t_stat, p_value = stats.ttest_rel(arr1, arr2, alternative=alternative)
                    test_type = "paired"
                    df = len(arr1) - 1
                else:
                    # Independent two-sample t-test
                    t_stat, p_value = stats.ttest_ind(arr1, arr2, alternative=alternative)
                    test_type = "independent two-sample"
                    df = len(arr1) + len(arr2) - 2

            else:
                return {"success": False, "error": "Provide either sample2 or mu"}

            # Effect size (Cohen's d)
            if sample2 is not None:
                pooled_std = np.sqrt(((len(arr1)-1)*np.var(arr1, ddof=1) + (len(arr2)-1)*np.var(arr2, ddof=1)) / (len(arr1)+len(arr2)-2))
                cohens_d = (np.mean(arr1) - np.mean(arr2)) / pooled_std if pooled_std > 0 else 0
            else:
                cohens_d = (np.mean(arr1) - mu) / np.std(arr1, ddof=1) if np.std(arr1, ddof=1) > 0 else 0

            return {
                "success": True,
                "test_type": test_type,
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "df": int(df),
                "alternative": alternative,
                "effect_size_cohens_d": float(cohens_d),
                "significant_at_0.05": p_value < 0.05,
                "significant_at_0.01": p_value < 0.01,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def chi_square(
        self,
        observed: List[float],
        expected: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Chi-square goodness of fit test.

        If expected is None, assumes uniform distribution.
        """
        try:
            stats = self.stats
            np = self.np

            obs = np.array(observed)
            if expected is None:
                exp = np.full_like(obs, np.sum(obs) / len(obs))
            else:
                exp = np.array(expected)

            chi2, p_value = stats.chisquare(obs, exp)
            df = len(obs) - 1

            return {
                "success": True,
                "test": "chi-square goodness of fit",
                "chi2_statistic": float(chi2),
                "p_value": float(p_value),
                "df": df,
                "significant_at_0.05": p_value < 0.05,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def anova(
        self,
        groups: List[List[float]],
    ) -> Dict[str, Any]:
        """
        One-way ANOVA test.

        Tests if there are significant differences between group means.
        """
        try:
            stats = self.stats
            np = self.np

            f_stat, p_value = stats.f_oneway(*[np.array(g) for g in groups])

            # Calculate eta-squared (effect size)
            all_data = np.concatenate([np.array(g) for g in groups])
            grand_mean = np.mean(all_data)
            ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
            ss_total = np.sum((all_data - grand_mean)**2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0

            return {
                "success": True,
                "test": "one-way ANOVA",
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "n_groups": len(groups),
                "effect_size_eta_squared": float(eta_squared),
                "significant_at_0.05": p_value < 0.05,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def mann_whitney(
        self,
        sample1: List[float],
        sample2: List[float],
        alternative: str = "two-sided",
    ) -> Dict[str, Any]:
        """
        Mann-Whitney U test (non-parametric alternative to t-test).
        """
        try:
            stats = self.stats
            np = self.np

            stat, p_value = stats.mannwhitneyu(
                np.array(sample1),
                np.array(sample2),
                alternative=alternative,
            )

            return {
                "success": True,
                "test": "Mann-Whitney U",
                "u_statistic": float(stat),
                "p_value": float(p_value),
                "alternative": alternative,
                "significant_at_0.05": p_value < 0.05,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def shapiro_wilk(
        self,
        data: List[float],
    ) -> Dict[str, Any]:
        """
        Shapiro-Wilk test for normality.
        """
        try:
            stats = self.stats
            stat, p_value = stats.shapiro(data)

            return {
                "success": True,
                "test": "Shapiro-Wilk normality test",
                "w_statistic": float(stat),
                "p_value": float(p_value),
                "is_normal_at_0.05": p_value > 0.05,
                "interpretation": "Data appears normally distributed" if p_value > 0.05 else "Data may not be normally distributed",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==================== CORRELATION ====================

    def correlation(
        self,
        x: List[float],
        y: List[float],
        method: str = "pearson",
    ) -> Dict[str, Any]:
        """
        Calculate correlation coefficient.

        Methods: pearson, spearman, kendall
        """
        try:
            stats = self.stats
            np = self.np

            x_arr = np.array(x)
            y_arr = np.array(y)

            if method == "pearson":
                r, p_value = stats.pearsonr(x_arr, y_arr)
            elif method == "spearman":
                r, p_value = stats.spearmanr(x_arr, y_arr)
            elif method == "kendall":
                r, p_value = stats.kendalltau(x_arr, y_arr)
            else:
                return {"success": False, "error": f"Unknown method: {method}"}

            # Interpret strength
            abs_r = abs(r)
            if abs_r < 0.3:
                strength = "weak"
            elif abs_r < 0.7:
                strength = "moderate"
            else:
                strength = "strong"

            return {
                "success": True,
                "method": method,
                "correlation": float(r),
                "p_value": float(p_value),
                "r_squared": float(r**2),
                "strength": strength,
                "direction": "positive" if r > 0 else "negative" if r < 0 else "none",
                "significant_at_0.05": p_value < 0.05,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==================== REGRESSION ====================

    def regression(
        self,
        x: List[float],
        y: List[float],
        model: str = "linear",
    ) -> Dict[str, Any]:
        """
        Perform regression analysis.

        Models: linear, logistic
        """
        try:
            stats = self.stats
            np = self.np

            x_arr = np.array(x)
            y_arr = np.array(y)

            if model == "linear":
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_arr, y_arr)

                # Predictions
                y_pred = slope * x_arr + intercept
                residuals = y_arr - y_pred

                # Additional stats
                n = len(x_arr)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y_arr - np.mean(y_arr))**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                mse = ss_res / (n - 2)
                rmse = np.sqrt(mse)

                return {
                    "success": True,
                    "model": "linear",
                    "equation": f"y = {slope:.4g}x + {intercept:.4g}",
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "r_squared": float(r_squared),
                    "r_value": float(r_value),
                    "p_value": float(p_value),
                    "std_error": float(std_err),
                    "rmse": float(rmse),
                    "significant_at_0.05": p_value < 0.05,
                }

            elif model == "logistic":
                # Simple logistic regression using maximum likelihood
                from scipy.optimize import minimize

                def neg_log_likelihood(params):
                    b0, b1 = params
                    z = b0 + b1 * x_arr
                    p = 1 / (1 + np.exp(-z))
                    p = np.clip(p, 1e-10, 1-1e-10)
                    return -np.sum(y_arr * np.log(p) + (1 - y_arr) * np.log(1 - p))

                result = minimize(neg_log_likelihood, [0, 0], method='BFGS')
                b0, b1 = result.x

                # Predictions
                z = b0 + b1 * x_arr
                y_pred = 1 / (1 + np.exp(-z))

                return {
                    "success": True,
                    "model": "logistic",
                    "equation": f"p = 1 / (1 + exp(-({b0:.4g} + {b1:.4g}x)))",
                    "intercept": float(b0),
                    "coefficient": float(b1),
                    "odds_ratio": float(np.exp(b1)),
                }

            else:
                return {"success": False, "error": f"Unknown model: {model}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==================== CONFIDENCE INTERVALS ====================

    def confidence_interval(
        self,
        data: List[float],
        confidence: float = 0.95,
        ci_type: str = "mean",
    ) -> Dict[str, Any]:
        """
        Calculate confidence interval.

        Types: mean, proportion
        """
        try:
            stats = self.stats
            np = self.np

            arr = np.array(data)
            n = len(arr)
            alpha = 1 - confidence

            if ci_type == "mean":
                mean = np.mean(arr)
                sem = stats.sem(arr)
                t_crit = stats.t.ppf(1 - alpha/2, n - 1)

                margin = t_crit * sem
                ci_low = mean - margin
                ci_high = mean + margin

                return {
                    "success": True,
                    "type": "mean",
                    "point_estimate": float(mean),
                    "ci_lower": float(ci_low),
                    "ci_upper": float(ci_high),
                    "margin_of_error": float(margin),
                    "confidence_level": confidence,
                    "sample_size": n,
                }

            elif ci_type == "proportion":
                # Assuming data is 0/1
                p = np.mean(arr)
                z_crit = stats.norm.ppf(1 - alpha/2)

                # Wilson score interval
                denominator = 1 + z_crit**2/n
                center = (p + z_crit**2/(2*n)) / denominator
                margin = z_crit * np.sqrt((p*(1-p) + z_crit**2/(4*n))/n) / denominator

                return {
                    "success": True,
                    "type": "proportion",
                    "point_estimate": float(p),
                    "ci_lower": float(center - margin),
                    "ci_upper": float(center + margin),
                    "confidence_level": confidence,
                    "sample_size": n,
                    "method": "Wilson score",
                }

            else:
                return {"success": False, "error": f"Unknown CI type: {ci_type}"}

        except Exception as e:
            return {"success": False, "error": str(e)}


def statistics_tool(
    operation: str,
    data: str,
    options: Optional[str] = None,
) -> ToolResult:
    """Tool function for statistics."""
    engine = StatisticsEngine()

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
        if operation == "descriptive":
            result = engine.descriptive(data_dict.get("data", data_dict))

        elif operation == "distribution":
            result = engine.distribution(
                data_dict.get("type"),
                data_dict.get("operation"),
                data_dict.get("x"),
                data_dict.get("params"),
            )

        elif operation == "t_test":
            result = engine.t_test(
                data_dict.get("sample1"),
                data_dict.get("sample2"),
                data_dict.get("mu"),
                opts.get("alternative", "two-sided"),
                opts.get("paired", False),
            )

        elif operation == "chi_square":
            result = engine.chi_square(
                data_dict.get("observed"),
                data_dict.get("expected"),
            )

        elif operation == "anova":
            result = engine.anova(data_dict.get("groups"))

        elif operation == "mann_whitney":
            result = engine.mann_whitney(
                data_dict.get("sample1"),
                data_dict.get("sample2"),
                opts.get("alternative", "two-sided"),
            )

        elif operation == "shapiro_wilk":
            result = engine.shapiro_wilk(data_dict.get("data", data_dict))

        elif operation == "correlation":
            result = engine.correlation(
                data_dict.get("x"),
                data_dict.get("y"),
                opts.get("method", "pearson"),
            )

        elif operation == "regression":
            result = engine.regression(
                data_dict.get("x"),
                data_dict.get("y"),
                opts.get("model", "linear"),
            )

        elif operation == "confidence_interval":
            result = engine.confidence_interval(
                data_dict.get("data", data_dict),
                opts.get("confidence", 0.95),
                opts.get("type", "mean"),
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
                error=result.get("error"),
            )

    except Exception as e:
        return ToolResult(
            status=ToolStatus.ERROR,
            output=None,
            error=str(e),
        )


def create_statistics_tool() -> Tool:
    """Create the statistics tool."""
    return Tool(
        name="statistics",
        description="Statistical analysis: descriptive stats, distributions (pdf/cdf/ppf), hypothesis tests (t-test, chi-square, ANOVA, Mann-Whitney), correlation (Pearson/Spearman/Kendall), regression (linear/logistic), confidence intervals.",
        parameters=[
            ToolParameter(
                name="operation",
                description="Statistical operation to perform",
                type="string",
                required=True,
                enum=["descriptive", "distribution", "t_test", "chi_square",
                      "anova", "mann_whitney", "shapiro_wilk", "correlation",
                      "regression", "confidence_interval"],
            ),
            ToolParameter(
                name="data",
                description="JSON object with input data (samples, x/y arrays, etc.)",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="options",
                description="JSON object with options (alternative, confidence, method, etc.)",
                type="string",
                required=False,
            ),
        ],
        execute_fn=statistics_tool,
        timeout_ms=30000,
    )


def register_statistics_tools(registry: ToolRegistry) -> None:
    """Register statistics tools with the registry."""
    registry.register(create_statistics_tool())
