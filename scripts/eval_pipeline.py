"""
Svend Model Evaluation Pipeline with Approval Gates

Two-stage gated evaluation for the training pipeline:
  Stage 1: Language Model → eval_language_model() → approve/reject
  Stage 2: Reasoning Model → eval_reasoning_model() → approve/reject

Integrates with existing evaluation infrastructure:
  - ResponseAnalyzer for Norwegian scoring
  - AdversarialTestSuite for safety
  - Tool use benchmarks

Usage:
  # Evaluate language model checkpoint (Stage 1)
  python scripts/eval_pipeline.py --checkpoint path/to/language-model.pt --stage language

  # Evaluate reasoning model checkpoint (Stage 2)
  python scripts/eval_pipeline.py --checkpoint path/to/reasoning-specialist.pt --stage reasoning

  # Auto-approve if criteria met (for CI/scripts)
  python scripts/eval_pipeline.py --checkpoint path/to/model.pt --stage language --auto-approve

  # Use CPU (no CUDA)
  python scripts/eval_pipeline.py --checkpoint path/to/model.pt --stage language --device cpu
"""

import argparse
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict, field
import math
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.config import TransformerConfig
from src.models.transformer import ReasoningTransformer
from src.evaluation.response_analyzer import ResponseAnalyzer, ToneAnalysis
from transformers import AutoTokenizer


# =============================================================================
# CRITERIA
# =============================================================================

@dataclass
class LanguageModelCriteria:
    """Pass/fail criteria for language model (Stage 1)."""
    max_perplexity: float = 50.0
    min_coherence_score: float = 0.5
    min_completion_rate: float = 0.8
    max_repetition_rate: float = 0.3
    min_norwegian_score: float = 0.4  # Communication directness


@dataclass
class ReasoningModelCriteria:
    """Pass/fail criteria for reasoning model (Stage 2)."""
    min_gsm8k_accuracy: float = 0.25
    min_format_compliance: float = 0.7
    min_norwegian_score: float = 0.5
    max_theatrical_rate: float = 0.3
    max_preachy_rate: float = 0.1


# =============================================================================
# RESULT TYPES
# =============================================================================

@dataclass
class EvalResult:
    """Single evaluation metric result."""
    name: str
    passed: bool
    score: float
    threshold: float
    details: str = ""


@dataclass
class StageReport:
    """Complete evaluation report for a training stage."""
    stage: str
    checkpoint_path: str
    timestamp: str
    results: List[EvalResult]
    overall_passed: bool
    recommendation: str
    sample_outputs: List[Dict[str, str]]
    norwegian_analysis: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "checkpoint_path": self.checkpoint_path,
            "timestamp": self.timestamp,
            "results": [asdict(r) for r in self.results],
            "overall_passed": self.overall_passed,
            "recommendation": self.recommendation,
            "sample_outputs": self.sample_outputs,
            "norwegian_analysis": self.norwegian_analysis,
        }

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_summary(self):
        print("\n" + "=" * 70)
        print(f"EVALUATION REPORT: {self.stage.upper()}")
        print("=" * 70)
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"Timestamp: {self.timestamp}")
        print("-" * 70)

        for result in self.results:
            status = "[PASS]" if result.passed else "[FAIL]"
            print(f"{status}  {result.name}: {result.score:.3f} (threshold: {result.threshold:.3f})")
            if result.details:
                print(f"        {result.details}")

        if self.norwegian_analysis:
            print("-" * 70)
            print("NORWEGIAN SCORE BREAKDOWN:")
            na = self.norwegian_analysis
            print(f"  Average Score: {na.get('avg_score', 0):.2f}")
            print(f"  Theatrical: {na.get('theatrical_count', 0)}")
            print(f"  Preachy: {na.get('preachy_count', 0)}")
            print(f"  Direct: {na.get('direct_count', 0)}")

        print("-" * 70)
        status_color = "\033[92m" if self.overall_passed else "\033[91m"
        print(f"OVERALL: {status_color}{'PASSED' if self.overall_passed else 'FAILED'}\033[0m")
        print(f"RECOMMENDATION: {self.recommendation}")
        print("=" * 70)


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(checkpoint_path: str, device: str = "cuda") -> Tuple[ReasoningTransformer, AutoTokenizer, TransformerConfig]:
    """Load model and tokenizer from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Remove computed fields that shouldn't be passed to __init__
    config_dict = {k: v for k, v in checkpoint["config"].items() if k != 'head_dim'}
    config = TransformerConfig(**config_dict)
    print(f"  Model: {config.name}")
    print(f"  Parameters: {config.num_parameters() / 1e6:.0f}M")

    model = ReasoningTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    tokenizer_name = checkpoint.get("tokenizer_name", "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    if "special_tokens" in checkpoint:
        tokenizer.add_special_tokens(checkpoint["special_tokens"])

    print(f"  Tokenizer: {tokenizer_name} ({len(tokenizer)} tokens)")

    return model, tokenizer, config


# =============================================================================
# GENERATION UTILITIES
# =============================================================================

@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
) -> str:
    """Generate text completion."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    try:
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        return f"[Generation Error: {e}]"


@torch.no_grad()
def compute_perplexity(model, tokenizer, texts: List[str], device: str, max_length: int = 512) -> float:
    """Compute perplexity on texts."""
    total_loss = 0
    total_tokens = 0

    for text in texts:
        encoding = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
        input_ids = encoding["input_ids"].to(device)

        if input_ids.shape[1] < 2:
            continue

        outputs = model(input_ids, labels=input_ids)
        loss = outputs["loss"]

        num_tokens = input_ids.shape[1] - 1
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    if total_tokens == 0:
        return float("inf")

    return math.exp(min(total_loss / total_tokens, 100))


def check_repetition(text: str, n: int = 3) -> bool:
    """Check for n-gram repetition."""
    words = text.lower().split()
    if len(words) < n * 2:
        return False

    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return False

    unique_ratio = len(set(ngrams)) / len(ngrams)
    return unique_ratio < 0.5


# =============================================================================
# STAGE 1: LANGUAGE MODEL EVALUATION
# =============================================================================

def eval_language_model(
    checkpoint_path: str,
    device: str = "cuda",
    criteria: Optional[LanguageModelCriteria] = None,
) -> StageReport:
    """
    Evaluate language model checkpoint (Stage 1).

    Tests:
    1. Perplexity on held-out text
    2. Completion coherence
    3. Repetition rate
    4. Norwegian score (communication directness)
    """
    if criteria is None:
        criteria = LanguageModelCriteria()

    model, tokenizer, config = load_model(checkpoint_path, device)
    analyzer = ResponseAnalyzer()
    results = []

    # --- Test 1: Perplexity ---
    print("\n[1/4] Computing perplexity...")
    eval_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In mathematics, a prime number is a natural number greater than 1.",
        "The capital of France is Paris, known for the Eiffel Tower.",
        "Water freezes at 0 degrees Celsius under standard pressure.",
        "Machine learning enables systems to learn patterns from data.",
    ]

    ppl = compute_perplexity(model, tokenizer, eval_texts, device)
    results.append(EvalResult(
        name="Perplexity",
        passed=ppl < criteria.max_perplexity,
        score=ppl,
        threshold=criteria.max_perplexity,
        details=f"Lower is better",
    ))
    print(f"    Perplexity: {ppl:.2f} (target < {criteria.max_perplexity})")

    # --- Test 2-4: Generation tests ---
    print("\n[2/4] Generating completions...")
    prompts = [
        "The capital of France is",
        "In 1969, humans first",
        "The theory of relativity states that",
        "Water boils at",
        "The largest planet in our solar system is",
        "Photosynthesis is the process by which",
        "The Great Wall of China was built to",
        "Programming languages are used to",
    ]

    completions = []
    norwegian_scores = []
    theatrical_count = 0
    preachy_count = 0
    direct_count = 0

    for prompt in prompts:
        output = generate_text(model, tokenizer, prompt, device, max_new_tokens=50)
        completion = output[len(prompt):].strip() if output.startswith(prompt) else output

        # Analyze tone
        analysis = analyzer.analyze(output)
        norwegian_scores.append(analysis.tone.norwegian_score)

        if analysis.tone.is_theatrical:
            theatrical_count += 1
        if analysis.tone.is_preachy:
            preachy_count += 1
        if analysis.tone.is_direct:
            direct_count += 1

        completions.append({
            "prompt": prompt,
            "completion": completion,
            "norwegian_score": analysis.tone.norwegian_score,
            "success": len(completion) > 0 and "[Generation Error" not in completion,
        })

    # Completion rate
    successful = sum(1 for c in completions if c["success"])
    completion_rate = successful / len(completions)
    results.append(EvalResult(
        name="Completion Rate",
        passed=completion_rate >= criteria.min_completion_rate,
        score=completion_rate,
        threshold=criteria.min_completion_rate,
        details=f"{successful}/{len(completions)} successful",
    ))
    print(f"    Completion rate: {completion_rate:.1%}")

    # Repetition
    print("\n[3/4] Checking repetition...")
    repetitive = sum(1 for c in completions if c["success"] and check_repetition(c["completion"]))
    repetition_rate = repetitive / max(successful, 1)
    results.append(EvalResult(
        name="Repetition Rate",
        passed=repetition_rate <= criteria.max_repetition_rate,
        score=repetition_rate,
        threshold=criteria.max_repetition_rate,
        details=f"{repetitive}/{successful} repetitive (lower is better)",
    ))
    print(f"    Repetition rate: {repetition_rate:.1%}")

    # Norwegian score
    print("\n[4/4] Computing Norwegian score...")
    avg_norwegian = sum(norwegian_scores) / len(norwegian_scores) if norwegian_scores else 0
    results.append(EvalResult(
        name="Norwegian Score",
        passed=avg_norwegian >= criteria.min_norwegian_score,
        score=avg_norwegian,
        threshold=criteria.min_norwegian_score,
        details="Communication directness (higher is better)",
    ))
    print(f"    Norwegian score: {avg_norwegian:.2f}")

    # --- Generate Report ---
    overall_passed = all(r.passed for r in results)

    if overall_passed:
        recommendation = "APPROVE - Proceed to Stage 2 (reasoning fine-tuning)"
    else:
        failed = [r.name for r in results if not r.passed]
        recommendation = f"REJECT - Failed: {', '.join(failed)}. Retrain with more data/steps."

    return StageReport(
        stage="language_model",
        checkpoint_path=checkpoint_path,
        timestamp=datetime.now().isoformat(),
        results=results,
        overall_passed=overall_passed,
        recommendation=recommendation,
        sample_outputs=completions[:5],
        norwegian_analysis={
            "avg_score": avg_norwegian,
            "theatrical_count": theatrical_count,
            "preachy_count": preachy_count,
            "direct_count": direct_count,
        },
    )


# =============================================================================
# STAGE 2: REASONING MODEL EVALUATION
# =============================================================================

GSM8K_TEST = [
    {"q": "Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. She sells the rest for $2 each. How much does she make daily?", "a": "18"},
    {"q": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts total?", "a": "3"},
    {"q": "Josh buys a house for $80,000, puts in $50,000 repairs, increasing value by 150%. What's his profit?", "a": "70000"},
    {"q": "James runs 3 sprints 3 times a week, 60 meters each. Total meters per week?", "a": "540"},
    {"q": "A train travels 60 mph for 2 hours, then 80 mph for 1.5 hours. Total distance?", "a": "240"},
    {"q": "If x + 3 = 7, what is x?", "a": "4"},
    {"q": "A rectangle has perimeter 24 cm. Length is twice width. What's the width?", "a": "4"},
    {"q": "5 red balls and 3 blue balls in a bag. Probability of drawing red?", "a": "5/8"},
    {"q": "What is 15% of 80?", "a": "12"},
    {"q": "A car travels 240 miles in 4 hours. What's the average speed?", "a": "60"},
]


def extract_answer(text: str) -> Optional[str]:
    """Extract numerical answer from model output."""
    import re

    # Look for <|answer|>...<|/answer|>
    match = re.search(r'<\|answer\|>(.*?)<\|/answer\|>', text, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        num = re.search(r'[\d/]+\.?\d*', answer.replace(',', ''))
        if num:
            return num.group()

    # Look for ####
    match = re.search(r'####\s*([\d/,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '')

    # Last number
    numbers = re.findall(r'[\d/]+\.?\d*', text)
    return numbers[-1] if numbers else None


def check_format(text: str) -> bool:
    """Check for proper reasoning format."""
    return "<|think|>" in text or "<|answer|>" in text or "####" in text


def eval_reasoning_model(
    checkpoint_path: str,
    device: str = "cuda",
    criteria: Optional[ReasoningModelCriteria] = None,
) -> StageReport:
    """
    Evaluate reasoning model checkpoint (Stage 2).

    Tests:
    1. GSM8K accuracy
    2. Format compliance
    3. Norwegian score
    """
    if criteria is None:
        criteria = ReasoningModelCriteria()

    model, tokenizer, config = load_model(checkpoint_path, device)
    analyzer = ResponseAnalyzer()
    results = []

    # --- Test 1: GSM8K Accuracy ---
    print("\n[1/3] Evaluating GSM8K accuracy...")
    correct = 0
    outputs = []
    norwegian_scores = []
    theatrical_count = 0
    preachy_count = 0

    for i, sample in enumerate(GSM8K_TEST):
        prompt = f"Question: {sample['q']}\n\n<|think|>\n"

        output = generate_text(model, tokenizer, prompt, device, max_new_tokens=256, temperature=0.3)
        predicted = extract_answer(output)
        expected = sample["a"]

        # Normalize comparison
        is_correct = predicted is not None and predicted.strip() == expected.strip()
        if is_correct:
            correct += 1

        # Analyze tone
        analysis = analyzer.analyze(output)
        norwegian_scores.append(analysis.tone.norwegian_score)
        if analysis.tone.is_theatrical:
            theatrical_count += 1
        if analysis.tone.is_preachy:
            preachy_count += 1

        outputs.append({
            "question": sample["q"][:80] + "...",
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "output": output[:300],
        })

        status = "[OK]" if is_correct else "[X]"
        print(f"    [{i+1}/{len(GSM8K_TEST)}] {status} expected={expected}, got={predicted}")

    accuracy = correct / len(GSM8K_TEST)
    results.append(EvalResult(
        name="GSM8K Accuracy",
        passed=accuracy >= criteria.min_gsm8k_accuracy,
        score=accuracy,
        threshold=criteria.min_gsm8k_accuracy,
        details=f"{correct}/{len(GSM8K_TEST)} correct",
    ))

    # --- Test 2: Format Compliance ---
    print("\n[2/3] Checking format compliance...")
    compliant = sum(1 for o in outputs if check_format(o["output"]))
    compliance_rate = compliant / len(outputs)
    results.append(EvalResult(
        name="Format Compliance",
        passed=compliance_rate >= criteria.min_format_compliance,
        score=compliance_rate,
        threshold=criteria.min_format_compliance,
        details=f"{compliant}/{len(outputs)} used reasoning format",
    ))
    print(f"    Format compliance: {compliance_rate:.1%}")

    # --- Test 3: Norwegian Score ---
    print("\n[3/3] Computing Norwegian score...")
    avg_norwegian = sum(norwegian_scores) / len(norwegian_scores) if norwegian_scores else 0
    results.append(EvalResult(
        name="Norwegian Score",
        passed=avg_norwegian >= criteria.min_norwegian_score,
        score=avg_norwegian,
        threshold=criteria.min_norwegian_score,
        details="Communication directness",
    ))
    print(f"    Norwegian score: {avg_norwegian:.2f}")

    # Theatrical/preachy check
    theatrical_rate = theatrical_count / len(outputs)
    preachy_rate = preachy_count / len(outputs)

    results.append(EvalResult(
        name="Theatrical Rate",
        passed=theatrical_rate <= criteria.max_theatrical_rate,
        score=theatrical_rate,
        threshold=criteria.max_theatrical_rate,
        details="Lower is better",
    ))

    results.append(EvalResult(
        name="Preachy Rate",
        passed=preachy_rate <= criteria.max_preachy_rate,
        score=preachy_rate,
        threshold=criteria.max_preachy_rate,
        details="Lower is better",
    ))

    # --- Generate Report ---
    overall_passed = all(r.passed for r in results)

    if overall_passed:
        recommendation = "APPROVE - Model ready for deployment or further fine-tuning"
    else:
        failed = [r.name for r in results if not r.passed]
        recommendation = f"REJECT - Failed: {', '.join(failed)}. More reasoning data or longer training needed."

    return StageReport(
        stage="reasoning_model",
        checkpoint_path=checkpoint_path,
        timestamp=datetime.now().isoformat(),
        results=results,
        overall_passed=overall_passed,
        recommendation=recommendation,
        sample_outputs=outputs[:5],
        norwegian_analysis={
            "avg_score": avg_norwegian,
            "theatrical_count": theatrical_count,
            "preachy_count": preachy_count,
        },
    )


# =============================================================================
# APPROVAL GATE
# =============================================================================

def approval_gate(report: StageReport, auto_approve: bool = False) -> bool:
    """Interactive approval gate."""
    report.print_summary()

    # Show samples
    print("\n" + "-" * 70)
    print("SAMPLE OUTPUTS:")
    print("-" * 70)

    for i, sample in enumerate(report.sample_outputs[:3], 1):
        print(f"\n[Sample {i}]")
        if "prompt" in sample:
            print(f"Prompt: {sample['prompt']}")
            print(f"Output: {sample.get('completion', '')[:150]}...")
            print(f"Norwegian: {sample.get('norwegian_score', 0):.2f}")
        elif "question" in sample:
            print(f"Q: {sample['question']}")
            print(f"Expected: {sample['expected']} | Got: {sample['predicted']} | {'[OK]' if sample.get('correct') else '[X]'}")

    print("-" * 70)

    if auto_approve:
        print("\n[Auto-approve mode]")
        return report.overall_passed

    print()
    if report.overall_passed:
        prompt = "Model PASSED. Approve and continue? [Y/n]: "
        default = True
    else:
        prompt = "Model FAILED. Override and approve anyway? [y/N]: "
        default = False

    try:
        response = input(prompt).strip().lower()
        if not response:
            return default
        return response in ("y", "yes")
    except EOFError:
        return report.overall_passed


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Svend Model Evaluation Pipeline")
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to checkpoint")
    parser.add_argument("--stage", "-s", required=True, choices=["language", "reasoning"])
    parser.add_argument("--output-dir", "-o", default="evaluations", help="Output directory")
    parser.add_argument("--auto-approve", "-y", action="store_true", help="Auto-approve if passed")
    parser.add_argument("--device", "-d", default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print(f"SVEND EVALUATION PIPELINE - STAGE: {args.stage.upper()}")
    print("=" * 70)

    # Run evaluation
    if args.stage == "language":
        report = eval_language_model(args.checkpoint, args.device)
    else:
        report = eval_reasoning_model(args.checkpoint, args.device)

    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"{args.stage}_eval_{timestamp}.json"
    report.save(str(report_path))
    print(f"\nReport saved: {report_path}")

    # Approval gate
    approved = approval_gate(report, args.auto_approve)

    if approved:
        print("\n[APPROVED]")
        approval_file = output_dir / f"{args.stage}_approved.json"
        with open(approval_file, "w") as f:
            json.dump({
                "checkpoint": args.checkpoint,
                "approved_at": datetime.now().isoformat(),
                "report": str(report_path),
            }, f, indent=2)
        sys.exit(0)
    else:
        print("\n[REJECTED]")
        sys.exit(1)


if __name__ == "__main__":
    main()
