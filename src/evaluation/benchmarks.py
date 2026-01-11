"""
Evaluation and benchmarking system for reasoning models.

Includes:
- Standard benchmarks (GSM8K, MATH, HumanEval, etc.)
- Custom reasoning evaluation
- Comparison between teacher and student
- Inference speed benchmarks
"""

import time
import json
import re
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass
from pathlib import Path
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset


@dataclass
class BenchmarkConfig:
    """Configuration for evaluation benchmarks."""

    # Which benchmarks to run
    benchmarks: List[str] = None  # ["gsm8k", "math", "humaneval", "logic"]

    # Evaluation settings
    batch_size: int = 1  # For generation
    max_new_tokens: int = 512
    temperature: float = 0.0  # Greedy for evaluation
    top_p: float = 1.0

    # Number of samples (None = full dataset)
    num_samples: Optional[int] = None

    # Output
    output_dir: str = "evaluation_results"
    save_generations: bool = True

    def __post_init__(self):
        if self.benchmarks is None:
            self.benchmarks = ["gsm8k", "reasoning_custom"]


class BenchmarkBase:
    """Base class for evaluation benchmarks."""

    name: str = "base"

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def load_data(self) -> List[Dict[str, Any]]:
        """Load evaluation data."""
        raise NotImplementedError

    def format_prompt(self, example: Dict[str, Any]) -> str:
        """Format example into model prompt."""
        raise NotImplementedError

    def extract_answer(self, generation: str) -> Any:
        """Extract answer from model generation."""
        raise NotImplementedError

    def check_answer(self, predicted: Any, expected: Any) -> bool:
        """Check if predicted answer is correct."""
        raise NotImplementedError

    def compute_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Compute aggregate metrics from results."""
        correct = sum(1 for r in results if r["correct"])
        total = len(results)
        return {
            f"{self.name}/accuracy": correct / total if total > 0 else 0,
            f"{self.name}/correct": correct,
            f"{self.name}/total": total,
        }


class GSM8KBenchmark(BenchmarkBase):
    """Grade School Math benchmark (GSM8K)."""

    name = "gsm8k"

    def load_data(self) -> List[Dict[str, Any]]:
        dataset = load_dataset("gsm8k", "main", split="test")
        if self.config.num_samples:
            dataset = dataset.select(range(min(self.config.num_samples, len(dataset))))
        return list(dataset)

    def format_prompt(self, example: Dict[str, Any]) -> str:
        question = example["question"]
        return f"""Solve this math problem step by step. Show your reasoning clearly.

Problem: {question}

Solution:"""

    def extract_answer(self, generation: str) -> Optional[float]:
        """Extract numerical answer from generation."""
        # Look for pattern like "#### 42" or "The answer is 42"
        patterns = [
            r"####\s*(-?[\d,]+(?:\.\d+)?)",
            r"answer\s*(?:is|=)\s*(-?[\d,]+(?:\.\d+)?)",
            r"=\s*(-?[\d,]+(?:\.\d+)?)\s*$",
        ]

        for pattern in patterns:
            match = re.search(pattern, generation, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1).replace(",", ""))
                except ValueError:
                    continue

        # Try to find the last number in the text
        numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", generation)
        if numbers:
            try:
                return float(numbers[-1].replace(",", ""))
            except ValueError:
                pass

        return None

    def check_answer(self, predicted: Optional[float], expected: str) -> bool:
        if predicted is None:
            return False

        # Extract expected answer
        match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", expected)
        if not match:
            return False

        try:
            expected_num = float(match.group(1).replace(",", ""))
            return abs(predicted - expected_num) < 1e-6
        except ValueError:
            return False


class MATHBenchmark(BenchmarkBase):
    """MATH benchmark for competition mathematics."""

    name = "math"

    def load_data(self) -> List[Dict[str, Any]]:
        dataset = load_dataset("hendrycks/competition_math", split="test")
        if self.config.num_samples:
            dataset = dataset.select(range(min(self.config.num_samples, len(dataset))))
        return list(dataset)

    def format_prompt(self, example: Dict[str, Any]) -> str:
        problem = example["problem"]
        return f"""Solve this mathematics problem. Show your work and box your final answer.

Problem: {problem}

Solution:"""

    def extract_answer(self, generation: str) -> Optional[str]:
        # Look for boxed answer
        patterns = [
            r"\\boxed\{([^}]+)\}",
            r"\\fbox\{([^}]+)\}",
            r"final\s+answer\s*(?:is)?\s*:?\s*(.+?)(?:\.|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, generation, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def check_answer(self, predicted: Optional[str], expected: str) -> bool:
        if predicted is None:
            return False

        # Extract expected from solution
        match = re.search(r"\\boxed\{([^}]+)\}", expected)
        if not match:
            return False

        expected_answer = match.group(1).strip()

        # Normalize for comparison
        def normalize(s):
            s = s.lower().strip()
            s = re.sub(r"\s+", "", s)
            return s

        return normalize(predicted) == normalize(expected_answer)


class ReasoningBenchmark(BenchmarkBase):
    """Custom reasoning evaluation with diverse problem types."""

    name = "reasoning_custom"

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.problems = self._create_problems()

    def _create_problems(self) -> List[Dict[str, Any]]:
        """Create custom reasoning problems."""
        return [
            # Logic
            {
                "type": "logic",
                "question": "If all cats are mammals, and some mammals are pets, can we conclude that some cats are pets?",
                "answer": "no",
                "explanation": "This is a syllogistic fallacy. While some mammals are pets, we cannot conclude that any of those pet mammals are specifically cats.",
            },
            {
                "type": "logic",
                "question": "In a room, there are 3 people: Alice, Bob, and Charlie. Exactly one of them is lying. Alice says 'Bob is lying.' Bob says 'Charlie is lying.' Charlie says 'Alice and Bob are both lying.' Who is lying?",
                "answer": "charlie",
                "explanation": "If Charlie is telling the truth, both Alice and Bob are lying (2 liars). If Bob is telling the truth, Charlie is lying and Alice must also be telling truth (1 liar = Charlie). If Alice is telling truth, Bob lies, meaning Charlie tells truth, contradiction.",
            },
            # Math reasoning
            {
                "type": "math",
                "question": "A snail climbs 3 meters up a wall during the day but slides down 2 meters at night. If the wall is 10 meters tall, how many days does it take to reach the top?",
                "answer": "8",
                "explanation": "Each day the snail makes net progress of 1 meter. After 7 days, it's at 7 meters. On day 8, it climbs 3 more meters to reach 10 meters before sliding back.",
            },
            {
                "type": "math",
                "question": "If 5 machines can produce 5 widgets in 5 minutes, how many minutes would it take 100 machines to produce 100 widgets?",
                "answer": "5",
                "explanation": "Each machine produces 1 widget in 5 minutes. So 100 machines produce 100 widgets in 5 minutes.",
            },
            # Code reasoning
            {
                "type": "code",
                "question": "What does this Python code print? x = [1, 2, 3]; y = x; y.append(4); print(len(x))",
                "answer": "4",
                "explanation": "y = x creates a reference to the same list, not a copy. When we append to y, we're also modifying x.",
            },
            {
                "type": "code",
                "question": "In Python, what is the result of: bool([]) and bool([0])",
                "answer": "false",
                "explanation": "bool([]) is False because empty list is falsy. Due to short-circuit evaluation, the second part isn't evaluated, result is False.",
            },
            # Commonsense
            {
                "type": "commonsense",
                "question": "Why do we see lightning before we hear thunder?",
                "answer": "light travels faster than sound",
                "explanation": "Light travels at approximately 300,000 km/s while sound travels at about 343 m/s. This massive difference means light from lightning reaches us almost instantly while sound takes longer.",
            },
            # Pattern recognition
            {
                "type": "pattern",
                "question": "What comes next in the sequence: 2, 6, 12, 20, 30, ?",
                "answer": "42",
                "explanation": "The differences are 4, 6, 8, 10, 12... (increasing by 2). So next difference is 12, and 30 + 12 = 42. Or: n(n+1) for n=1,2,3,4,5,6 gives 2,6,12,20,30,42.",
            },
        ]

    def load_data(self) -> List[Dict[str, Any]]:
        return self.problems

    def format_prompt(self, example: Dict[str, Any]) -> str:
        question = example["question"]
        return f"""Answer this reasoning question. Think step by step.

Question: {question}

Answer:"""

    def extract_answer(self, generation: str) -> str:
        # Normalize the generation
        generation = generation.lower().strip()

        # Try to extract structured answer
        patterns = [
            r"(?:the\s+)?answer\s*(?:is)?\s*:?\s*(.+?)(?:\.|,|$)",
            r"therefore\s*,?\s*(.+?)(?:\.|$)",
            r"so\s*,?\s*(.+?)(?:\.|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, generation, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Return first line as fallback
        return generation.split("\n")[0].strip()

    def check_answer(self, predicted: str, expected: str) -> bool:
        predicted = predicted.lower().strip()
        expected = expected.lower().strip()

        # Direct match
        if expected in predicted:
            return True

        # Numeric comparison for numeric answers
        try:
            pred_num = float(re.search(r"-?\d+(?:\.\d+)?", predicted).group())
            exp_num = float(expected)
            return abs(pred_num - exp_num) < 1e-6
        except (ValueError, AttributeError):
            pass

        return False


class InferenceBenchmark:
    """Benchmark inference speed and memory usage."""

    def __init__(self, model: nn.Module, tokenizer, device: torch.device = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def measure_latency(
        self,
        prompt: str = "What is 2 + 2?",
        num_tokens: int = 50,
        num_runs: int = 10,
        warmup_runs: int = 3,
    ) -> Dict[str, float]:
        """Measure generation latency."""
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                self.model.generate(input_ids, max_new_tokens=num_tokens)

        # Measure
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        latencies = []

        for _ in range(num_runs):
            start = time.perf_counter()

            with torch.no_grad():
                output = self.model.generate(input_ids, max_new_tokens=num_tokens)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.perf_counter()

            latencies.append(end - start)

        tokens_generated = output.shape[1] - input_ids.shape[1]

        return {
            "latency_mean_ms": sum(latencies) / len(latencies) * 1000,
            "latency_std_ms": (sum((l - sum(latencies)/len(latencies))**2 for l in latencies) / len(latencies)) ** 0.5 * 1000,
            "tokens_per_second": tokens_generated / (sum(latencies) / len(latencies)),
            "time_per_token_ms": (sum(latencies) / len(latencies)) / tokens_generated * 1000,
        }

    def measure_memory(self) -> Dict[str, float]:
        """Measure GPU memory usage."""
        if not torch.cuda.is_available():
            return {"gpu_memory_gb": 0, "note": "No GPU available"}

        torch.cuda.reset_peak_memory_stats()

        # Run a forward pass
        dummy_input = torch.randint(0, 1000, (1, 512)).to(self.device)
        with torch.no_grad():
            self.model(dummy_input)

        peak_memory = torch.cuda.max_memory_allocated() / 1e9

        return {
            "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "gpu_memory_peak_gb": peak_memory,
        }

    def measure_throughput(
        self,
        batch_sizes: List[int] = [1, 2, 4, 8],
        seq_length: int = 512,
    ) -> Dict[str, Any]:
        """Measure throughput at different batch sizes."""
        results = {}

        for batch_size in batch_sizes:
            try:
                dummy_input = torch.randint(0, 1000, (batch_size, seq_length)).to(self.device)

                # Warmup
                with torch.no_grad():
                    self.model(dummy_input)

                # Measure
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start = time.perf_counter()

                num_iterations = 10
                for _ in range(num_iterations):
                    with torch.no_grad():
                        self.model(dummy_input)

                torch.cuda.synchronize() if torch.cuda.is_available() else None
                elapsed = time.perf_counter() - start

                samples_per_second = (batch_size * num_iterations) / elapsed
                results[f"batch_{batch_size}"] = {
                    "samples_per_second": samples_per_second,
                    "tokens_per_second": samples_per_second * seq_length,
                }

            except RuntimeError as e:
                if "out of memory" in str(e):
                    results[f"batch_{batch_size}"] = {"error": "OOM"}
                    torch.cuda.empty_cache()
                else:
                    raise

        return results


class ModelEvaluator:
    """
    Complete evaluation pipeline for reasoning models.

    Runs all benchmarks and compiles results.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: BenchmarkConfig,
        device: torch.device = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.model.eval()

        # Initialize benchmarks
        self.benchmarks = {
            "gsm8k": GSM8KBenchmark(config),
            "math": MATHBenchmark(config),
            "reasoning_custom": ReasoningBenchmark(config),
        }

        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        """Generate completion for a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=self.config.temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        generated = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        return generated

    def run_benchmark(self, benchmark_name: str) -> Dict[str, Any]:
        """Run a single benchmark."""
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

        benchmark = self.benchmarks[benchmark_name]
        data = benchmark.load_data()

        print(f"\nRunning {benchmark_name} ({len(data)} examples)...")

        results = []
        for i, example in enumerate(data):
            prompt = benchmark.format_prompt(example)
            generation = self.generate(prompt)
            predicted = benchmark.extract_answer(generation)
            correct = benchmark.check_answer(predicted, example.get("answer", ""))

            result = {
                "example_id": i,
                "prompt": prompt,
                "generation": generation,
                "predicted": predicted,
                "expected": example.get("answer"),
                "correct": correct,
            }
            results.append(result)

            if (i + 1) % 10 == 0:
                acc = sum(r["correct"] for r in results) / len(results)
                print(f"  Progress: {i+1}/{len(data)} | Accuracy: {acc:.2%}")

        metrics = benchmark.compute_metrics(results)

        # Save detailed results
        if self.config.save_generations:
            output_path = self.output_dir / f"{benchmark_name}_results.json"
            with open(output_path, "w") as f:
                json.dump({"metrics": metrics, "results": results}, f, indent=2)

        return {"metrics": metrics, "results": results}

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all configured benchmarks."""
        all_results = {}
        all_metrics = {}

        for benchmark_name in self.config.benchmarks:
            if benchmark_name in self.benchmarks:
                result = self.run_benchmark(benchmark_name)
                all_results[benchmark_name] = result
                all_metrics.update(result["metrics"])

        # Run inference benchmarks
        print("\nRunning inference benchmarks...")
        inference_bench = InferenceBenchmark(self.model, self.tokenizer, self.device)
        all_metrics["inference/latency"] = inference_bench.measure_latency()
        all_metrics["inference/memory"] = inference_bench.measure_memory()

        # Summary
        print("\n" + "="*50)
        print("Evaluation Summary")
        print("="*50)
        for key, value in all_metrics.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
            else:
                print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

        # Save summary
        summary_path = self.output_dir / "evaluation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_metrics, f, indent=2)

        return {"metrics": all_metrics, "results": all_results}


def compare_models(
    teacher: nn.Module,
    student: nn.Module,
    tokenizer,
    config: BenchmarkConfig,
) -> Dict[str, Any]:
    """
    Compare teacher and student model performance.

    Useful for measuring distillation quality.
    """
    print("Evaluating Teacher Model")
    print("="*50)
    teacher_eval = ModelEvaluator(teacher, tokenizer, config)
    teacher_results = teacher_eval.run_all_benchmarks()

    print("\nEvaluating Student Model")
    print("="*50)
    student_eval = ModelEvaluator(student, tokenizer, config)
    student_results = student_eval.run_all_benchmarks()

    # Compute comparison metrics
    comparison = {}
    for key in teacher_results["metrics"]:
        if isinstance(teacher_results["metrics"][key], (int, float)):
            t_val = teacher_results["metrics"][key]
            s_val = student_results["metrics"].get(key, 0)
            comparison[key] = {
                "teacher": t_val,
                "student": s_val,
                "retention": s_val / t_val if t_val > 0 else 0,
            }

    print("\n" + "="*50)
    print("Teacher vs Student Comparison")
    print("="*50)
    for key, vals in comparison.items():
        if isinstance(vals, dict) and "retention" in vals:
            print(f"{key}: Teacher={vals['teacher']:.4f}, Student={vals['student']:.4f}, Retention={vals['retention']:.1%}")

    return {
        "teacher": teacher_results,
        "student": student_results,
        "comparison": comparison,
    }
