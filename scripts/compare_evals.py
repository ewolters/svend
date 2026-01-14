#!/usr/bin/env python3
"""
Compare two evaluation runs and highlight regressions/improvements.

Usage:
    py -3 scripts/compare_evals.py evaluations/run1/report.json evaluations/run2/report.json
    py -3 scripts/compare_evals.py --latest 2  # Compare two most recent runs
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple


def load_report(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def compare_metric(name: str, old: float, new: float, higher_is_better: bool = True) -> Tuple[str, str]:
    """Compare a metric and return (change_str, status)."""
    if old == 0:
        pct = "N/A"
    else:
        pct = ((new - old) / old) * 100

    diff = new - old

    if higher_is_better:
        if diff > 0.01:
            status = "\033[92mIMPROVED\033[0m"
        elif diff < -0.01:
            status = "\033[91mREGRESSED\033[0m"
        else:
            status = "unchanged"
    else:
        if diff < -0.01:
            status = "\033[92mIMPROVED\033[0m"
        elif diff > 0.01:
            status = "\033[91mREGRESSED\033[0m"
        else:
            status = "unchanged"

    if isinstance(pct, float):
        change = f"{diff:+.3f} ({pct:+.1f}%)"
    else:
        change = f"{diff:+.3f}"

    return change, status


def compare_reports(old: Dict, new: Dict) -> None:
    """Compare two evaluation reports."""
    old_s = old["summary"]
    new_s = new["summary"]

    print("=" * 70)
    print("EVALUATION COMPARISON")
    print("=" * 70)
    print(f"Old: {old['meta']['run_id']} ({old_s['total_tests']} tests)")
    print(f"New: {new['meta']['run_id']} ({new_s['total_tests']} tests)")
    print()

    # Key metrics
    print("KEY METRICS:")
    print("-" * 70)
    print(f"{'Metric':<30} {'Old':>10} {'New':>10} {'Change':>15} {'Status'}")
    print("-" * 70)

    metrics = [
        ("Overall Accuracy", "overall_accuracy", True),
        ("Refusal Accuracy", "refusal_accuracy", True),
        ("False Positive Rate", "false_positive_rate", False),
        ("False Negative Rate", "false_negative_rate", False),
        ("Norwegian Score", "avg_norwegian_score", True),
    ]

    for name, key, higher_better in metrics:
        old_val = old_s.get(key, 0)
        new_val = new_s.get(key, 0)
        change, status = compare_metric(name, old_val, new_val, higher_better)
        print(f"{name:<30} {old_val:>10.3f} {new_val:>10.3f} {change:>15} {status}")

    print()

    # Critical failures
    old_critical = set(old_s.get("critical_failures", []))
    new_critical = set(new_s.get("critical_failures", []))

    new_failures = new_critical - old_critical
    fixed_failures = old_critical - new_critical

    if new_failures:
        print("\033[91mNEW CRITICAL FAILURES:\033[0m")
        for f in sorted(new_failures):
            print(f"  - {f}")
        print()

    if fixed_failures:
        print("\033[92mFIXED CRITICAL FAILURES:\033[0m")
        for f in sorted(fixed_failures):
            print(f"  + {f}")
        print()

    # Norwegian score distribution
    print("NORWEGIAN SCORE DISTRIBUTION:")
    print("-" * 70)
    old_norw = old_s.get("norwegian_score_distribution", {})
    new_norw = new_s.get("norwegian_score_distribution", {})

    for bucket in ["excellent", "good", "fair", "poor"]:
        old_v = old_norw.get(bucket, 0)
        new_v = new_norw.get(bucket, 0)
        diff = new_v - old_v
        if diff > 0:
            symbol = "\033[92m+" + str(diff) + "\033[0m"
        elif diff < 0:
            symbol = "\033[91m" + str(diff) + "\033[0m"
        else:
            symbol = "="
        print(f"  {bucket:<12} {old_v:>5} -> {new_v:>5}  {symbol}")

    print()

    # Accuracy by attack category
    print("ACCURACY BY ATTACK TYPE:")
    print("-" * 70)

    old_attack = old_s.get("accuracy_by_attack", {})
    new_attack = new_s.get("accuracy_by_attack", {})
    all_attacks = set(old_attack.keys()) | set(new_attack.keys())

    for attack in sorted(all_attacks):
        old_v = old_attack.get(attack, 0)
        new_v = new_attack.get(attack, 0)
        change, status = compare_metric(attack, old_v, new_v, True)
        print(f"  {attack:<20} {old_v:.1%} -> {new_v:.1%}  {status}")

    print()

    # Red flags comparison
    print("RED FLAGS:")
    print("-" * 70)
    old_flags = old_s.get("red_flags_distribution", {})
    new_flags = new_s.get("red_flags_distribution", {})

    all_flags = set(old_flags.keys()) | set(new_flags.keys())
    for flag in sorted(all_flags):
        old_v = old_flags.get(flag, 0)
        new_v = new_flags.get(flag, 0)
        if old_v != new_v:
            diff = new_v - old_v
            color = "\033[91m" if diff > 0 else "\033[92m"
            print(f"  {flag}: {old_v} -> {new_v} ({color}{diff:+d}\033[0m)")

    print()
    print("=" * 70)

    # Summary verdict
    safety_improved = new_s.get("false_negative_rate", 1) < old_s.get("false_negative_rate", 1)
    tone_improved = new_s.get("avg_norwegian_score", 0) > old_s.get("avg_norwegian_score", 0)
    new_criticals = len(new_failures) > 0

    print("VERDICT:")
    if new_criticals:
        print("  \033[91mBLOCKED: New critical safety failures\033[0m")
    elif safety_improved and tone_improved:
        print("  \033[92mGOOD: Safety and tone both improved\033[0m")
    elif safety_improved:
        print("  \033[93mOK: Safety improved, tone unchanged/regressed\033[0m")
    elif tone_improved:
        print("  \033[93mOK: Tone improved, safety unchanged/regressed\033[0m")
    else:
        print("  \033[93mNEUTRAL: No significant changes\033[0m")

    print("=" * 70)


def find_latest_reports(n: int = 2) -> List[Path]:
    """Find the n most recent evaluation reports."""
    eval_dir = Path("evaluations")
    reports = sorted(eval_dir.glob("adversarial_*/report_*.json"), reverse=True)
    return reports[:n]


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation runs")
    parser.add_argument("reports", nargs="*", help="Two report JSON files to compare")
    parser.add_argument("--latest", type=int, help="Compare N most recent runs")

    args = parser.parse_args()

    if args.latest:
        reports = find_latest_reports(args.latest)
        if len(reports) < 2:
            print("Not enough reports to compare")
            sys.exit(1)
        old_path, new_path = reports[1], reports[0]
    elif len(args.reports) == 2:
        old_path, new_path = Path(args.reports[0]), Path(args.reports[1])
    else:
        parser.print_help()
        sys.exit(1)

    old = load_report(old_path)
    new = load_report(new_path)

    compare_reports(old, new)


if __name__ == "__main__":
    main()
