"""
Game theory tools for Svend reasoning system.

Covers:
- Normal form games (arbitrary size)
- Nash equilibrium (pure and mixed)
- Dominant/dominated strategy elimination
- Minimax for zero-sum games
- Extensive form games (game trees)
- Repeated games (tit-for-tat, etc.)
- Auction theory basics
- Mechanism design primitives
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum
from itertools import product
import numpy as np


class GameType(Enum):
    ZERO_SUM = "zero_sum"
    GENERAL_SUM = "general_sum"
    COORDINATION = "coordination"
    ANTI_COORDINATION = "anti_coordination"
    PRISONERS_DILEMMA = "prisoners_dilemma"


@dataclass
class Strategy:
    """A strategy (pure or mixed)."""
    name: str
    probabilities: Optional[List[float]] = None  # None = pure strategy

    def is_pure(self) -> bool:
        return self.probabilities is None


@dataclass
class GameResult:
    """Result of game theory analysis."""
    equilibria: List[Dict[str, Any]]
    dominant_strategies: Dict[str, Optional[str]]
    game_type: GameType
    analysis: Dict[str, Any]
    steps: List[str]


class GameTheoryEngine:
    """
    Game theory analysis engine.

    Handles strategic game analysis with step-by-step solutions.
    """

    def __init__(self):
        self.precision = 4

    def analyze_normal_form(
        self,
        players: List[str],
        strategies: Dict[str, List[str]],
        payoffs: Dict[Tuple, Tuple[float, ...]]
    ) -> GameResult:
        """
        Analyze a normal form game.

        Args:
            players: List of player names
            strategies: Dict mapping player -> list of strategy names
            payoffs: Dict mapping strategy profile tuple -> payoff tuple

        Returns:
            GameResult with equilibria and analysis
        """
        steps = []
        n_players = len(players)

        steps.append(f"Players: {players}")
        for p in players:
            steps.append(f"  {p}'s strategies: {strategies[p]}")
        steps.append("")

        # Display payoff matrix (for 2 players)
        if n_players == 2:
            steps.append("Payoff Matrix:")
            p1, p2 = players
            header = "".join(f"{s:>12}" for s in strategies[p2])
            steps.append(f"{'':>12}{header}")

            for s1 in strategies[p1]:
                row = f"{s1:>12}"
                for s2 in strategies[p2]:
                    payoff = payoffs.get((s1, s2), payoffs.get((strategies[p1].index(s1), strategies[p2].index(s2))))
                    row += f"{str(payoff):>12}"
                steps.append(row)
            steps.append("")

        # Check for dominant strategies
        steps.append("Checking for dominant strategies...")
        dominant = {}

        for i, player in enumerate(players):
            player_strats = strategies[player]
            dominant[player] = None

            for s in player_strats:
                is_dominant = True
                is_strictly_dominant = True

                for s_other in player_strats:
                    if s == s_other:
                        continue

                    # Check if s dominates s_other
                    dominates = True
                    strictly_dominates = True

                    # Compare across all opponent strategy profiles
                    other_players = [p for p in players if p != player]
                    other_strats = [strategies[p] for p in other_players]

                    for opp_profile in product(*other_strats):
                        # Build full strategy profile
                        profile_s = list(opp_profile)
                        profile_s.insert(i, s)
                        profile_other = list(opp_profile)
                        profile_other.insert(i, s_other)

                        payoff_s = payoffs[tuple(profile_s)][i]
                        payoff_other = payoffs[tuple(profile_other)][i]

                        if payoff_s < payoff_other:
                            dominates = False
                            strictly_dominates = False
                            break
                        if payoff_s == payoff_other:
                            strictly_dominates = False

                    if not dominates:
                        is_dominant = False
                        is_strictly_dominant = False
                        break

                if is_strictly_dominant:
                    dominant[player] = s
                    steps.append(f"  {player}: '{s}' is strictly dominant")
                    break
                elif is_dominant and dominant[player] is None:
                    dominant[player] = s
                    steps.append(f"  {player}: '{s}' is weakly dominant")

            if dominant[player] is None:
                steps.append(f"  {player}: no dominant strategy")

        steps.append("")

        # Find pure strategy Nash equilibria
        steps.append("Finding pure strategy Nash equilibria...")
        pure_equilibria = []

        all_profiles = list(product(*[strategies[p] for p in players]))

        for profile in all_profiles:
            is_nash = True

            for i, player in enumerate(players):
                current_payoff = payoffs[profile][i]

                # Check if player can improve by deviating
                for alt_strat in strategies[player]:
                    if alt_strat == profile[i]:
                        continue

                    alt_profile = list(profile)
                    alt_profile[i] = alt_strat
                    alt_payoff = payoffs[tuple(alt_profile)][i]

                    if alt_payoff > current_payoff:
                        is_nash = False
                        break

                if not is_nash:
                    break

            if is_nash:
                pure_equilibria.append({
                    "type": "pure",
                    "strategies": dict(zip(players, profile)),
                    "payoffs": dict(zip(players, payoffs[profile]))
                })
                steps.append(f"  Found: {profile} -> {payoffs[profile]}")

        if not pure_equilibria:
            steps.append("  No pure strategy Nash equilibrium")
        steps.append("")

        # For 2-player games, find mixed equilibria
        mixed_equilibria = []
        if n_players == 2 and len(strategies[players[0]]) == 2 and len(strategies[players[1]]) == 2:
            steps.append("Finding mixed strategy Nash equilibrium...")
            mixed_eq = self._find_2x2_mixed_equilibrium(players, strategies, payoffs, steps)
            if mixed_eq:
                mixed_equilibria.append(mixed_eq)

        # Classify game type
        game_type = self._classify_game(players, strategies, payoffs)
        steps.append(f"Game classification: {game_type.value}")

        # Additional analysis
        analysis = {
            "is_zero_sum": game_type == GameType.ZERO_SUM,
            "has_dominant_strategy_equilibrium": all(d is not None for d in dominant.values()),
            "num_pure_equilibria": len(pure_equilibria),
            "num_mixed_equilibria": len(mixed_equilibria)
        }

        return GameResult(
            equilibria=pure_equilibria + mixed_equilibria,
            dominant_strategies=dominant,
            game_type=game_type,
            analysis=analysis,
            steps=steps
        )

    def _find_2x2_mixed_equilibrium(
        self,
        players: List[str],
        strategies: Dict[str, List[str]],
        payoffs: Dict[Tuple, Tuple[float, float]],
        steps: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Find mixed strategy equilibrium for 2x2 game."""
        p1, p2 = players
        s1_list = strategies[p1]
        s2_list = strategies[p2]

        # Extract payoffs
        a11 = payoffs[(s1_list[0], s2_list[0])]
        a12 = payoffs[(s1_list[0], s2_list[1])]
        a21 = payoffs[(s1_list[1], s2_list[0])]
        a22 = payoffs[(s1_list[1], s2_list[1])]

        # Player 1's payoffs
        u1_11, u1_12, u1_21, u1_22 = a11[0], a12[0], a21[0], a22[0]
        # Player 2's payoffs
        u2_11, u2_12, u2_21, u2_22 = a11[1], a12[1], a21[1], a22[1]

        # Find p (P1 plays first strategy) to make P2 indifferent
        # P2's expected payoff from s2_list[0]: p*u2_11 + (1-p)*u2_21
        # P2's expected payoff from s2_list[1]: p*u2_12 + (1-p)*u2_22
        # Set equal and solve for p

        denom_p = (u2_11 - u2_21) - (u2_12 - u2_22)
        if abs(denom_p) < 1e-10:
            steps.append("  No interior mixed equilibrium (P2 indifference condition)")
            return None

        p = (u2_22 - u2_21) / denom_p

        # Find q (P2 plays first strategy) to make P1 indifferent
        denom_q = (u1_11 - u1_12) - (u1_21 - u1_22)
        if abs(denom_q) < 1e-10:
            steps.append("  No interior mixed equilibrium (P1 indifference condition)")
            return None

        q = (u1_22 - u1_12) / denom_q

        # Check if probabilities are valid
        if not (0 < p < 1 and 0 < q < 1):
            steps.append(f"  Mixed equilibrium outside [0,1]: p={p:.4f}, q={q:.4f}")
            return None

        # Calculate expected payoffs
        exp_u1 = q * (p * u1_11 + (1-p) * u1_21) + (1-q) * (p * u1_12 + (1-p) * u1_22)
        exp_u2 = p * (q * u2_11 + (1-q) * u2_12) + (1-p) * (q * u2_21 + (1-q) * u2_22)

        steps.append(f"  {p1} mixes: {s1_list[0]} w.p. {p:.4f}, {s1_list[1]} w.p. {1-p:.4f}")
        steps.append(f"  {p2} mixes: {s2_list[0]} w.p. {q:.4f}, {s2_list[1]} w.p. {1-q:.4f}")
        steps.append(f"  Expected payoffs: ({exp_u1:.4f}, {exp_u2:.4f})")

        return {
            "type": "mixed",
            "strategies": {
                p1: {s1_list[0]: round(p, 4), s1_list[1]: round(1-p, 4)},
                p2: {s2_list[0]: round(q, 4), s2_list[1]: round(1-q, 4)}
            },
            "expected_payoffs": {p1: round(exp_u1, 4), p2: round(exp_u2, 4)}
        }

    def _classify_game(
        self,
        players: List[str],
        strategies: Dict[str, List[str]],
        payoffs: Dict[Tuple, Tuple]
    ) -> GameType:
        """Classify the type of game."""
        if len(players) != 2:
            return GameType.GENERAL_SUM

        # Check if zero-sum
        is_zero_sum = True
        for profile, pays in payoffs.items():
            if abs(sum(pays)) > 1e-10:
                is_zero_sum = False
                break

        if is_zero_sum:
            return GameType.ZERO_SUM

        # Check for common game patterns (2x2 only)
        if len(strategies[players[0]]) == 2 and len(strategies[players[1]]) == 2:
            p1, p2 = players
            s1, s2 = strategies[p1]
            t1, t2 = strategies[p2]

            # Get payoffs
            cc = payoffs[(s1, t1)]
            cd = payoffs[(s1, t2)]
            dc = payoffs[(s2, t1)]
            dd = payoffs[(s2, t2)]

            # Prisoner's dilemma check: DC > CC > DD > CD for both (symmetric)
            # Temptation > Reward > Punishment > Sucker
            if (dc[0] > cc[0] > dd[0] > cd[0] and
                cd[1] > cc[1] > dd[1] > dc[1]):
                return GameType.PRISONERS_DILEMMA

            # Coordination: both prefer same action
            if cc[0] > cd[0] and cc[0] > dc[0] and dd[0] > cd[0] and dd[0] > dc[0]:
                if cc[1] > cd[1] and cc[1] > dc[1] and dd[1] > cd[1] and dd[1] > dc[1]:
                    return GameType.COORDINATION

            # Anti-coordination: both prefer different actions
            if cd[0] > cc[0] and cd[0] > dd[0] and dc[0] > cc[0] and dc[0] > dd[0]:
                return GameType.ANTI_COORDINATION

        return GameType.GENERAL_SUM

    def minimax(
        self,
        payoff_matrix: List[List[float]],
        row_strategies: List[str],
        col_strategies: List[str]
    ) -> Dict[str, Any]:
        """
        Find minimax solution for zero-sum game.

        Args:
            payoff_matrix: Row player's payoffs (col player gets negative)
            row_strategies: Names of row player's strategies
            col_strategies: Names of column player's strategies

        Returns:
            Dictionary with minimax strategies and value
        """
        steps = []
        matrix = np.array(payoff_matrix)

        steps.append("Payoff matrix (row player's perspective):")
        for i, row in enumerate(payoff_matrix):
            steps.append(f"  {row_strategies[i]}: {row}")
        steps.append("")

        # Row player's maximin: maximize minimum payoff
        row_mins = matrix.min(axis=1)
        maximin_value = row_mins.max()
        maximin_strategy = row_strategies[row_mins.argmax()]

        steps.append("Row player (maximin):")
        steps.append(f"  Row minimums: {row_mins.tolist()}")
        steps.append(f"  Maximin value: {maximin_value}")
        steps.append(f"  Maximin strategy: {maximin_strategy}")
        steps.append("")

        # Column player's minimax: minimize maximum loss
        col_maxs = matrix.max(axis=0)
        minimax_value = col_maxs.min()
        minimax_strategy = col_strategies[col_maxs.argmin()]

        steps.append("Column player (minimax):")
        steps.append(f"  Column maximums: {col_maxs.tolist()}")
        steps.append(f"  Minimax value: {minimax_value}")
        steps.append(f"  Minimax strategy: {minimax_strategy}")
        steps.append("")

        # Check for saddle point
        has_saddle = abs(maximin_value - minimax_value) < 1e-10

        if has_saddle:
            steps.append(f"Saddle point exists! Game value = {maximin_value}")
            steps.append(f"Pure strategy equilibrium: ({maximin_strategy}, {minimax_strategy})")
        else:
            steps.append("No saddle point - mixed strategy required")
            steps.append(f"Maximin ({maximin_value}) < Minimax ({minimax_value})")

            # Find mixed strategy for 2x2
            if matrix.shape == (2, 2):
                steps.append("")
                steps.append("Finding mixed strategy equilibrium...")

                a, b = matrix[0]
                c, d = matrix[1]

                # Row player's optimal mix
                denom = (a - b - c + d)
                if abs(denom) > 1e-10:
                    p = (d - b) / denom
                    if 0 <= p <= 1:
                        steps.append(f"Row player: {row_strategies[0]} w.p. {p:.4f}")

                        # Game value
                        value = p * a + (1-p) * c  # When col plays first
                        # Or equivalently with optimal col response
                        q = (d - c) / denom
                        value = p * (q * a + (1-q) * b) + (1-p) * (q * c + (1-q) * d)
                        steps.append(f"Game value: {value:.4f}")

        return {
            "maximin_strategy": maximin_strategy,
            "maximin_value": float(maximin_value),
            "minimax_strategy": minimax_strategy,
            "minimax_value": float(minimax_value),
            "has_saddle_point": has_saddle,
            "game_value": float(maximin_value) if has_saddle else None,
            "steps": steps
        }

    def iterated_elimination(
        self,
        players: List[str],
        strategies: Dict[str, List[str]],
        payoffs: Dict[Tuple, Tuple],
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Perform iterated elimination of dominated strategies.

        Args:
            players: List of player names
            strategies: Dict mapping player -> list of strategy names
            payoffs: Dict mapping strategy profile -> payoff tuple
            strict: If True, only eliminate strictly dominated strategies

        Returns:
            Dictionary with remaining strategies and elimination history
        """
        steps = []
        remaining = {p: set(strategies[p]) for p in players}
        history = []

        steps.append("Iterated elimination of dominated strategies")
        steps.append(f"Mode: {'strict' if strict else 'weak'} dominance")
        steps.append("")

        changed = True
        round_num = 0

        while changed:
            changed = False
            round_num += 1
            steps.append(f"Round {round_num}:")

            for i, player in enumerate(players):
                strats = list(remaining[player])
                if len(strats) <= 1:
                    continue

                dominated = set()

                for s in strats:
                    for s_other in strats:
                        if s == s_other:
                            continue

                        # Check if s is dominated by s_other
                        is_dominated = True
                        is_strictly_dominated = True

                        other_players = [p for p in players if p != player]
                        other_strats = [list(remaining[p]) for p in other_players]

                        for opp_profile in product(*other_strats):
                            profile_s = list(opp_profile)
                            profile_s.insert(i, s)
                            profile_other = list(opp_profile)
                            profile_other.insert(i, s_other)

                            payoff_s = payoffs[tuple(profile_s)][i]
                            payoff_other = payoffs[tuple(profile_other)][i]

                            if payoff_s > payoff_other:
                                is_dominated = False
                                is_strictly_dominated = False
                                break
                            if payoff_s == payoff_other:
                                is_strictly_dominated = False

                        if strict and is_strictly_dominated:
                            dominated.add(s)
                            steps.append(f"  {player}: '{s}' strictly dominated by '{s_other}'")
                            break
                        elif not strict and is_dominated:
                            dominated.add(s)
                            steps.append(f"  {player}: '{s}' weakly dominated by '{s_other}'")
                            break

                if dominated:
                    remaining[player] -= dominated
                    history.append({
                        "round": round_num,
                        "player": player,
                        "eliminated": list(dominated)
                    })
                    changed = True

            if not changed:
                steps.append("  No more dominated strategies found")

        steps.append("")
        steps.append("Remaining strategies:")
        for p in players:
            steps.append(f"  {p}: {list(remaining[p])}")

        # Check if unique prediction
        unique = all(len(remaining[p]) == 1 for p in players)
        if unique:
            result = {p: list(remaining[p])[0] for p in players}
            steps.append(f"\nUnique prediction: {result}")

        return {
            "remaining_strategies": {p: list(remaining[p]) for p in players},
            "elimination_history": history,
            "is_unique": unique,
            "steps": steps
        }

    def repeated_game_strategies(
        self,
        base_game: Dict[str, Any],
        num_rounds: int,
        discount_factor: float = 1.0
    ) -> Dict[str, Any]:
        """
        Analyze repeated game strategies.

        Args:
            base_game: Stage game specification
            num_rounds: Number of repetitions (use large number for "infinite")
            discount_factor: Discount factor delta in (0, 1]

        Returns:
            Analysis of sustainable cooperation
        """
        steps = []

        # Extract Prisoner's Dilemma payoffs (T > R > P > S)
        # Assume standard form: Cooperate/Defect
        steps.append("Analyzing repeated game...")
        steps.append(f"Rounds: {num_rounds}")
        steps.append(f"Discount factor: {discount_factor}")
        steps.append("")

        # Standard PD payoffs for analysis
        T = base_game.get("temptation", 5)   # Defect vs Cooperate
        R = base_game.get("reward", 3)       # Both Cooperate
        P = base_game.get("punishment", 1)   # Both Defect
        S = base_game.get("sucker", 0)       # Cooperate vs Defect

        steps.append(f"Stage game payoffs:")
        steps.append(f"  T (temptation) = {T}")
        steps.append(f"  R (reward) = {R}")
        steps.append(f"  P (punishment) = {P}")
        steps.append(f"  S (sucker) = {S}")
        steps.append("")

        # Check if cooperation can be sustained with grim trigger
        # Cooperate if: R/(1-δ) >= T + δP/(1-δ)
        # Simplifies to: δ >= (T-R)/(T-P)

        if T > P:
            min_delta_grim = (T - R) / (T - P)
        else:
            min_delta_grim = 0

        steps.append("Grim Trigger Strategy:")
        steps.append(f"  Cooperation sustainable if δ >= {min_delta_grim:.4f}")
        steps.append(f"  Current δ = {discount_factor}")

        grim_works = discount_factor >= min_delta_grim
        steps.append(f"  Cooperation sustainable: {grim_works}")
        steps.append("")

        # Tit-for-Tat analysis
        # TFT payoff against TFT: R forever
        # TFT payoff against AllD: S then P forever
        # AllD payoff against TFT: T then P forever

        steps.append("Tit-for-Tat Analysis:")
        if discount_factor < 1:
            tft_vs_tft = R / (1 - discount_factor)
            alld_vs_tft = T + (discount_factor * P) / (1 - discount_factor)
            steps.append(f"  TFT vs TFT payoff: {tft_vs_tft:.4f}")
            steps.append(f"  AllD vs TFT payoff: {alld_vs_tft:.4f}")
            steps.append(f"  TFT is best response to itself: {tft_vs_tft >= alld_vs_tft}")
        else:
            steps.append("  With δ=1, compare average per-round payoffs")
            steps.append(f"  TFT vs TFT: {R} per round")
            steps.append(f"  AllD vs TFT: ~{P} per round (after first)")

        # Folk theorem check
        steps.append("")
        steps.append("Folk Theorem:")
        steps.append(f"  Minimax payoff (defection): {P}")
        steps.append(f"  Pareto optimal (cooperation): {R}")

        if discount_factor >= min_delta_grim:
            steps.append(f"  Any payoff in [{P}, {R}] is sustainable as SPNE")
        else:
            steps.append(f"  Only stage game NE ({P}) is sustainable")

        return {
            "stage_game": {"T": T, "R": R, "P": P, "S": S},
            "min_delta_cooperation": round(min_delta_grim, 4),
            "cooperation_sustainable": grim_works,
            "strategies_analyzed": ["Grim Trigger", "Tit-for-Tat", "Always Defect"],
            "steps": steps
        }

    def auction_analysis(
        self,
        auction_type: str,
        num_bidders: int,
        value_distribution: Dict[str, float],
        reserve_price: float = 0
    ) -> Dict[str, Any]:
        """
        Analyze auction formats.

        Args:
            auction_type: "first_price", "second_price", "english", "dutch"
            num_bidders: Number of bidders
            value_distribution: {"type": "uniform", "low": 0, "high": 100}
            reserve_price: Minimum acceptable bid

        Returns:
            Analysis with bidding strategies and expected revenue
        """
        steps = []

        steps.append(f"Auction type: {auction_type}")
        steps.append(f"Number of bidders: {num_bidders}")
        steps.append(f"Value distribution: {value_distribution}")
        steps.append(f"Reserve price: {reserve_price}")
        steps.append("")

        n = num_bidders
        dist_type = value_distribution.get("type", "uniform")
        low = value_distribution.get("low", 0)
        high = value_distribution.get("high", 100)

        if dist_type == "uniform":
            # Uniform distribution on [low, high]
            expected_value = (low + high) / 2

            if auction_type == "second_price":
                steps.append("Second-Price (Vickrey) Auction:")
                steps.append("  Dominant strategy: bid true value")
                steps.append(f"  Optimal bid b(v) = v")
                steps.append("")

                # Expected revenue: E[second highest value]
                # For uniform, this is (n-1)/(n+1) * high
                exp_second_highest = low + (n - 1) / (n + 1) * (high - low)
                steps.append(f"  E[second highest value] = {exp_second_highest:.2f}")
                steps.append(f"  Expected revenue = {exp_second_highest:.2f}")

                return {
                    "auction_type": auction_type,
                    "optimal_strategy": "bid true value",
                    "expected_revenue": round(exp_second_highest, 2),
                    "steps": steps
                }

            elif auction_type == "first_price":
                steps.append("First-Price Sealed-Bid Auction:")
                steps.append("  Bidders shade their bids below value")
                steps.append(f"  Optimal bid b(v) = v - (v - {low})/{n}")
                steps.append(f"           = ({n-1}/{n}) * v + {low}/{n}")
                steps.append("")

                # With n bidders, shade by (v-low)/n
                # Expected revenue same as second-price (Revenue Equivalence)
                exp_revenue = low + (n - 1) / (n + 1) * (high - low)
                steps.append("  By Revenue Equivalence Theorem:")
                steps.append(f"  Expected revenue = {exp_revenue:.2f}")
                steps.append("  (Same as second-price auction)")

                return {
                    "auction_type": auction_type,
                    "optimal_strategy": f"bid (n-1)/n * v = {(n-1)/n:.3f} * v",
                    "expected_revenue": round(exp_revenue, 2),
                    "revenue_equivalence": True,
                    "steps": steps
                }

            elif auction_type == "english":
                steps.append("English (Ascending) Auction:")
                steps.append("  Strategy: stay in until price reaches value")
                steps.append("  Strategically equivalent to second-price")
                steps.append("")

                exp_revenue = low + (n - 1) / (n + 1) * (high - low)
                steps.append(f"  Expected revenue = {exp_revenue:.2f}")

                return {
                    "auction_type": auction_type,
                    "optimal_strategy": "drop out when price exceeds value",
                    "expected_revenue": round(exp_revenue, 2),
                    "steps": steps
                }

            elif auction_type == "dutch":
                steps.append("Dutch (Descending) Auction:")
                steps.append("  Strategically equivalent to first-price")
                steps.append("  Optimal: stop clock at same bid as first-price")
                steps.append("")

                exp_revenue = low + (n - 1) / (n + 1) * (high - low)
                steps.append(f"  Expected revenue = {exp_revenue:.2f}")

                return {
                    "auction_type": auction_type,
                    "optimal_strategy": f"stop at (n-1)/n * v",
                    "expected_revenue": round(exp_revenue, 2),
                    "steps": steps
                }

        return {
            "auction_type": auction_type,
            "error": "Unsupported distribution type",
            "steps": steps
        }


# Tool interface for Svend
def game_theory_tool(operation: str, **kwargs) -> Dict[str, Any]:
    """
    Main entry point for game theory tool.

    Operations:
        - analyze: Full analysis of normal form game
        - minimax: Minimax solution for zero-sum game
        - eliminate: Iterated elimination of dominated strategies
        - repeated: Analyze repeated game strategies
        - auction: Auction theory analysis
    """
    engine = GameTheoryEngine()

    if operation == "analyze":
        result = engine.analyze_normal_form(
            kwargs["players"],
            kwargs["strategies"],
            kwargs["payoffs"]
        )
        return {
            "equilibria": result.equilibria,
            "dominant_strategies": result.dominant_strategies,
            "game_type": result.game_type.value,
            "analysis": result.analysis,
            "steps": result.steps
        }

    elif operation == "minimax":
        return engine.minimax(
            kwargs["payoff_matrix"],
            kwargs["row_strategies"],
            kwargs["col_strategies"]
        )

    elif operation == "eliminate":
        return engine.iterated_elimination(
            kwargs["players"],
            kwargs["strategies"],
            kwargs["payoffs"],
            kwargs.get("strict", True)
        )

    elif operation == "repeated":
        return engine.repeated_game_strategies(
            kwargs["base_game"],
            kwargs["num_rounds"],
            kwargs.get("discount_factor", 1.0)
        )

    elif operation == "auction":
        return engine.auction_analysis(
            kwargs["auction_type"],
            kwargs["num_bidders"],
            kwargs["value_distribution"],
            kwargs.get("reserve_price", 0)
        )

    else:
        raise ValueError(f"Unknown operation: {operation}")


def register_game_theory_tools(registry) -> None:
    """Register game theory tools with the registry."""
    from .registry import Tool, ToolParameter, ToolResult, ToolStatus

    def _game_theory_execute(**kwargs) -> ToolResult:
        try:
            result = game_theory_tool(**kwargs)
            return ToolResult(status=ToolStatus.SUCCESS, output=result)
        except Exception as e:
            return ToolResult(status=ToolStatus.ERROR, output=None, error=str(e))

    registry.register(Tool(
        name="game_theory",
        description="Game theory analysis: Nash equilibrium, minimax, dominated strategies, repeated games, auctions",
        parameters=[
            ToolParameter(name="operation", type="string", description="Operation: analyze, minimax, eliminate, repeated, auction", required=True),
            ToolParameter(name="players", type="array", description="List of player names (for analyze, eliminate)", required=False),
            ToolParameter(name="strategies", type="object", description="Dict mapping player -> list of strategy names (for analyze, eliminate)", required=False),
            ToolParameter(name="payoffs", type="object", description="Dict mapping strategy tuple -> payoff tuple (for analyze, eliminate)", required=False),
            ToolParameter(name="payoff_matrix", type="array", description="2D array of row player payoffs (for minimax)", required=False),
            ToolParameter(name="row_strategies", type="array", description="Row strategy names (for minimax)", required=False),
            ToolParameter(name="col_strategies", type="array", description="Column strategy names (for minimax)", required=False),
            ToolParameter(name="strict", type="boolean", description="Use strict dominance only (for eliminate)", required=False),
            ToolParameter(name="base_game", type="object", description="Stage game payoffs: temptation, reward, punishment, sucker (for repeated)", required=False),
            ToolParameter(name="num_rounds", type="integer", description="Number of rounds (for repeated)", required=False),
            ToolParameter(name="discount_factor", type="number", description="Discount factor delta in (0,1] (for repeated)", required=False),
            ToolParameter(name="auction_type", type="string", description="Auction type: first_price, second_price, english, dutch (for auction)", required=False),
            ToolParameter(name="num_bidders", type="integer", description="Number of bidders (for auction)", required=False),
            ToolParameter(name="value_distribution", type="object", description="Value distribution: type, low, high (for auction)", required=False),
            ToolParameter(name="reserve_price", type="number", description="Reserve price (for auction)", required=False),
        ],
        execute_fn=_game_theory_execute
    ))
