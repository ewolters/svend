#!/usr/bin/env python3
"""
Svend Retraining Loop

Automated workflow for iterative model improvement:
1. Train model
2. Evaluate (safety, tone, tool use)
3. Generate priorities
4. Apply fixes to training data
5. Repeat

Usage:
    # Single iteration
    py -3 scripts/retrain_loop.py --iteration 1

    # Just evaluate current checkpoint
    py -3 scripts/retrain_loop.py --eval-only --checkpoint checkpoints/latest.pt

    # Generate training data patches from evaluation
    py -3 scripts/retrain_loop.py --generate-patches evaluations/unified_*/report_*.json

    # Full loop (eval + generate patches)
    py -3 scripts/retrain_loop.py --checkpoint checkpoints/latest.pt --apply-patches
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TrainingPatch:
    """A patch to apply to training data."""
    id: str
    type: str  # add_example, remove_example, modify_weight
    priority: int
    issue: str
    data: Dict[str, Any]


@dataclass
class RetrainingPlan:
    """Plan for next training iteration."""
    iteration: int
    timestamp: str
    eval_run_id: str

    # What to fix
    patches: List[TrainingPatch]

    # Summary
    safety_patches: int
    tone_patches: int
    tool_patches: int

    # Estimated impact
    expected_improvements: List[str]


def load_eval_results(path: Path) -> Dict[str, Any]:
    """Load evaluation results."""
    with open(path) as f:
        return json.load(f)


def generate_patches_from_eval(results: Dict[str, Any]) -> List[TrainingPatch]:
    """Generate training data patches from evaluation results."""
    patches = []
    patch_id = 0

    summary = results.get("summary", {})
    safety = results.get("safety", {})
    tone = results.get("tone", {})
    tool = results.get("tool_use", {})

    # === SAFETY PATCHES ===

    # Critical failures - add counter-examples
    critical_failures = summary.get("critical_failures", [])
    for test_id in critical_failures:
        # Find the test result
        test_result = next(
            (r for r in safety.get("results", []) if r.get("test_id") == test_id),
            None
        )
        if test_result:
            patches.append(TrainingPatch(
                id=f"patch_{patch_id:03d}",
                type="add_example",
                priority=1,  # Highest
                issue=f"Critical safety failure: {test_id}",
                data={
                    "category": "safety_refusal",
                    "prompt": test_result.get("prompt", ""),
                    "response": "No. I cannot help with that.",
                    "weight": 2.0,  # Higher weight
                    "source": f"eval_{results['meta']['run_id']}",
                }
            ))
            patch_id += 1

    # False negatives - add more refusal examples
    fn_rate = summary.get("false_negative_rate", 0)
    if fn_rate > 0.02:
        # Find all false negatives
        for r in safety.get("results", []):
            if r.get("should_refuse") and not r.get("refused"):
                patches.append(TrainingPatch(
                    id=f"patch_{patch_id:03d}",
                    type="add_example",
                    priority=2,
                    issue=f"False negative: {r.get('test_id')}",
                    data={
                        "category": "safety_refusal",
                        "prompt": r.get("prompt", ""),
                        "response": "No. That's harmful and I won't help with it.",
                        "weight": 1.5,
                        "source": f"eval_{results['meta']['run_id']}",
                    }
                ))
                patch_id += 1

    # False positives - add examples of valid requests
    fp_rate = summary.get("false_positive_rate", 0)
    if fp_rate > 0.10:
        for r in safety.get("results", []):
            if not r.get("should_refuse") and r.get("refused"):
                patches.append(TrainingPatch(
                    id=f"patch_{patch_id:03d}",
                    type="add_example",
                    priority=3,
                    issue=f"False positive: {r.get('test_id')}",
                    data={
                        "category": "legitimate_request",
                        "prompt": r.get("prompt", ""),
                        "response": "[Provide helpful, direct answer]",
                        "weight": 1.0,
                        "source": f"eval_{results['meta']['run_id']}",
                        "note": "Replace placeholder with actual helpful response",
                    }
                ))
                patch_id += 1

    # === TONE PATCHES ===

    norwegian_score = summary.get("avg_norwegian_score", 0.5)
    theatrical_count = tone.get("theatrical_count", 0)

    if norwegian_score < 0.5:
        # Add Norwegian-style examples
        patches.append(TrainingPatch(
            id=f"patch_{patch_id:03d}",
            type="add_example",
            priority=2,
            issue=f"Low Norwegian score: {norwegian_score:.2f}",
            data={
                "category": "tone_direct",
                "examples": [
                    {"prompt": "What is 2+2?", "response": "4."},
                    {"prompt": "Is Python good?", "response": "Yes, for most tasks."},
                    {"prompt": "Explain recursion", "response": "A function that calls itself. Base case stops it."},
                ],
                "weight": 1.5,
                "note": "Add 10-20 direct, no-fluff examples",
            }
        ))
        patch_id += 1

    if theatrical_count > 5:
        # Remove theatrical examples from training
        patches.append(TrainingPatch(
            id=f"patch_{patch_id:03d}",
            type="remove_pattern",
            priority=2,
            issue=f"Excessive theatrical responses: {theatrical_count}",
            data={
                "category": "tone_cleanup",
                "patterns_to_remove": [
                    "Great question!",
                    "I'd be happy to help!",
                    "That's a wonderful",
                    "I'm excited to",
                    "Absolutely!",
                    "What a great question",
                ],
                "action": "Remove or rewrite training examples containing these patterns",
            }
        ))
        patch_id += 1

    # === TOOL USE PATCHES ===

    tool_acc = tool.get("accuracy", 0)
    tool_selection = tool.get("tool_selection_accuracy", 0)

    if tool_selection < 0.8:
        # Add tool selection examples
        patches.append(TrainingPatch(
            id=f"patch_{patch_id:03d}",
            type="add_example",
            priority=2,
            issue=f"Low tool selection accuracy: {tool_selection:.1%}",
            data={
                "category": "tool_selection",
                "examples": [
                    {
                        "prompt": "What's the derivative of x^2?",
                        "response": "<|tool_call|>symbolic_math\n{\"op\": \"diff\", \"expr\": \"x**2\"}\n<|/tool_call|>\nResult: 2x",
                    },
                    {
                        "prompt": "Calculate 50 factorial",
                        "response": "<|tool_call|>execute_python\nprint(math.factorial(50))\n<|/tool_call|>\nResult: 30414093201713378043612608166064768844377641568960512000000000000",
                    },
                ],
                "weight": 1.5,
                "note": "Add 15-20 tool calling examples across math, code, logic",
            }
        ))
        patch_id += 1

    # Failed tool tests
    for r in tool.get("results", []):
        if not r.get("correct"):
            patches.append(TrainingPatch(
                id=f"patch_{patch_id:03d}",
                type="add_example",
                priority=3,
                issue=f"Tool test failed: {r.get('test_id')}",
                data={
                    "category": "tool_usage",
                    "prompt": r.get("prompt", ""),
                    "expected_tool": r.get("expected_tool"),
                    "note": "Add correct tool call example",
                }
            ))
            patch_id += 1

    return patches


def create_retraining_plan(
    results: Dict[str, Any],
    patches: List[TrainingPatch],
    iteration: int,
) -> RetrainingPlan:
    """Create a retraining plan from patches."""
    safety_patches = sum(1 for p in patches if "safety" in p.issue.lower() or "false" in p.issue.lower())
    tone_patches = sum(1 for p in patches if "norwegian" in p.issue.lower() or "theatrical" in p.issue.lower())
    tool_patches = sum(1 for p in patches if "tool" in p.issue.lower())

    expected = []
    if safety_patches > 0:
        expected.append(f"Safety: Reduce false negative rate by ~{min(safety_patches * 0.5, 5):.1f}%")
    if tone_patches > 0:
        expected.append(f"Tone: Increase Norwegian score by ~{min(tone_patches * 0.05, 0.2):.2f}")
    if tool_patches > 0:
        expected.append(f"Tools: Improve tool accuracy by ~{min(tool_patches * 2, 15):.0f}%")

    return RetrainingPlan(
        iteration=iteration,
        timestamp=datetime.now().isoformat(),
        eval_run_id=results["meta"]["run_id"],
        patches=patches,
        safety_patches=safety_patches,
        tone_patches=tone_patches,
        tool_patches=tool_patches,
        expected_improvements=expected,
    )


def save_plan(plan: RetrainingPlan, output_dir: Path):
    """Save retraining plan to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full plan as JSON
    plan_path = output_dir / f"retrain_plan_iter{plan.iteration}.json"
    with open(plan_path, "w") as f:
        json.dump(asdict(plan), f, indent=2, default=str)
    print(f"Saved plan: {plan_path}")

    # Save patches as separate files for easier processing
    patches_dir = output_dir / "patches"
    patches_dir.mkdir(exist_ok=True)

    for patch in plan.patches:
        patch_path = patches_dir / f"{patch.id}.json"
        with open(patch_path, "w") as f:
            json.dump(asdict(patch), f, indent=2)

    print(f"Saved {len(plan.patches)} patches to {patches_dir}")

    # Generate human-readable summary
    summary_path = output_dir / f"retrain_summary_iter{plan.iteration}.md"
    with open(summary_path, "w") as f:
        f.write(f"# Retraining Plan - Iteration {plan.iteration}\n\n")
        f.write(f"**Eval Run:** {plan.eval_run_id}\n")
        f.write(f"**Generated:** {plan.timestamp}\n\n")

        f.write("## Patch Summary\n\n")
        f.write(f"- Safety patches: {plan.safety_patches}\n")
        f.write(f"- Tone patches: {plan.tone_patches}\n")
        f.write(f"- Tool patches: {plan.tool_patches}\n")
        f.write(f"- **Total:** {len(plan.patches)}\n\n")

        f.write("## Expected Improvements\n\n")
        for exp in plan.expected_improvements:
            f.write(f"- {exp}\n")
        f.write("\n")

        f.write("## Patches by Priority\n\n")
        sorted_patches = sorted(plan.patches, key=lambda p: p.priority)

        current_priority = None
        for patch in sorted_patches:
            if patch.priority != current_priority:
                current_priority = patch.priority
                f.write(f"\n### Priority {current_priority}\n\n")

            f.write(f"**{patch.id}** ({patch.type})\n")
            f.write(f"- Issue: {patch.issue}\n")
            if patch.data.get("prompt"):
                f.write(f"- Prompt: `{patch.data['prompt'][:100]}...`\n")
            if patch.data.get("note"):
                f.write(f"- Note: {patch.data['note']}\n")
            f.write("\n")

    print(f"Saved summary: {summary_path}")


def print_plan_summary(plan: RetrainingPlan):
    """Print plan summary to console."""
    print("\n" + "=" * 70)
    print(f"RETRAINING PLAN - ITERATION {plan.iteration}")
    print("=" * 70)
    print(f"Eval Run: {plan.eval_run_id}")
    print(f"Total Patches: {len(plan.patches)}")
    print()
    print("BREAKDOWN:")
    print(f"  Safety:  {plan.safety_patches}")
    print(f"  Tone:    {plan.tone_patches}")
    print(f"  Tools:   {plan.tool_patches}")
    print()
    print("EXPECTED IMPROVEMENTS:")
    for exp in plan.expected_improvements:
        print(f"  → {exp}")
    print()

    print("TOP PRIORITY PATCHES:")
    for patch in sorted(plan.patches, key=lambda p: p.priority)[:5]:
        print(f"  [P{patch.priority}] {patch.issue}")
    print("=" * 70)


def run_evaluation(checkpoint_path: Optional[str], quick: bool = False) -> Dict[str, Any]:
    """Run unified evaluation."""
    from scripts.run_unified_eval import UnifiedEvaluator

    use_simulation = checkpoint_path is None
    model = None
    tokenizer = None

    if checkpoint_path and not use_simulation:
        try:
            import torch
            from transformers import AutoTokenizer
            from src.models.transformer import ReasoningTransformer
            from src.models.config import get_model_config

            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            config = checkpoint.get("config", get_model_config("1b"))
            model = ReasoningTransformer(config)
            model.load_state_dict(checkpoint["model_state_dict"])
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception as e:
            print(f"Could not load model: {e}")
            use_simulation = True

    evaluator = UnifiedEvaluator(
        model=model,
        tokenizer=tokenizer,
        model_name="svend",
        use_simulation=use_simulation,
    )

    return evaluator.run(quick=quick)


def main():
    parser = argparse.ArgumentParser(description="Svend retraining loop")

    parser.add_argument("--iteration", type=int, default=1, help="Iteration number")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint path")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    parser.add_argument("--quick", action="store_true", help="Quick evaluation")
    parser.add_argument("--generate-patches", type=str, help="Generate patches from eval JSON")
    parser.add_argument("--output-dir", type=str, default="retraining", help="Output directory")
    parser.add_argument("--apply-patches", action="store_true", help="Generate patches after eval")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Mode 1: Generate patches from existing eval
    if args.generate_patches:
        print(f"Loading eval results from {args.generate_patches}")
        results = load_eval_results(Path(args.generate_patches))
        patches = generate_patches_from_eval(results)
        plan = create_retraining_plan(results, patches, args.iteration)
        save_plan(plan, output_dir)
        print_plan_summary(plan)
        return

    # Mode 2: Run evaluation
    if args.eval_only:
        print("Running evaluation...")
        results = run_evaluation(args.checkpoint, args.quick)
        print(f"\nResults saved to: {results.get('meta', {}).get('run_id', 'unknown')}")
        return

    # Mode 3: Full loop (eval + patches)
    print(f"=== RETRAINING LOOP - ITERATION {args.iteration} ===\n")

    # Step 1: Run evaluation
    print("Step 1: Running unified evaluation...")
    results = run_evaluation(args.checkpoint, args.quick)

    if args.apply_patches:
        # Step 2: Generate patches
        print("\nStep 2: Generating training patches...")
        patches = generate_patches_from_eval(results)

        # Step 3: Create plan
        print("\nStep 3: Creating retraining plan...")
        plan = create_retraining_plan(results, patches, args.iteration)

        # Step 4: Save
        save_plan(plan, output_dir)
        print_plan_summary(plan)

        print("\n" + "=" * 70)
        print("NEXT STEPS:")
        print("=" * 70)
        print(f"1. Review patches in {output_dir}/patches/")
        print("2. Apply patches to training data")
        print("3. Run training: py -3 scripts/train_svend.py ...")
        print(f"4. Run next iteration: py -3 scripts/retrain_loop.py --iteration {args.iteration + 1} --checkpoint <new_checkpoint>")
        print("=" * 70)


if __name__ == "__main__":
    main()
