"""
Synthetic data generation for reasoning tasks.

Uses Claude/GPT-4 API to generate high-quality reasoning traces
from seed problems. This augments open datasets with custom examples.
"""

import json
import asyncio
import random
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import hashlib


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""

    # API settings
    api_provider: str = "anthropic"  # "anthropic" or "openai"
    model: str = "claude-sonnet-4-20250514"  # or "gpt-4-turbo"
    max_tokens: int = 2048
    temperature: float = 0.7

    # Generation settings
    batch_size: int = 10
    max_retries: int = 3

    # Output settings
    output_dir: str = "data/synthetic"
    save_every: int = 100


# Seed problem templates for different categories
SEED_TEMPLATES = {
    "math_word_problem": [
        "A store sells {item1} for ${price1} each and {item2} for ${price2} each. If someone buys {qty1} {item1} and {qty2} {item2}, what is the total cost?",
        "A train travels at {speed1} mph for {hours1} hours, then at {speed2} mph for {hours2} hours. What is the total distance traveled?",
        "If {percent}% of a number is {result}, what is the number?",
        "{person1} is {age1} years old. {person2} is {relation} {person1}'s age. How old is {person2}?",
        "A rectangle has a perimeter of {perimeter} units. If the length is {ratio} times the width, find the dimensions.",
    ],
    "logic_puzzle": [
        "{n} friends are sitting in a row. {constraint1}. {constraint2}. Who is sitting where?",
        "In a family of {n} members, {fact1}. {fact2}. {fact3}. What is {question}?",
        "If all {A} are {B}, and some {B} are {C}, what can we conclude about {A} and {C}?",
        "{person} makes two statements: '{stmt1}' and '{stmt2}'. If exactly one is true, what do we know?",
    ],
    "code_debugging": [
        "The following {language} code is supposed to {intended_behavior}, but it has a bug:\n```{language}\n{buggy_code}\n```\nFind and fix the bug.",
        "This function should {description}, but it fails for certain inputs:\n```{language}\n{code}\n```\nIdentify the edge case and fix it.",
        "Optimize this {language} code that {behavior}:\n```{language}\n{code}\n```",
    ],
    "scientific_reasoning": [
        "Explain why {phenomenon} occurs in terms of {field}.",
        "Design an experiment to test whether {hypothesis}.",
        "Given that {observation1} and {observation2}, what can we infer about {subject}?",
        "Compare and contrast {concept1} and {concept2} in the context of {field}.",
    ],
    "common_sense": [
        "{person} left their {object} at {location1} and went to {location2}. When they returned, where would they look for their {object}?",
        "If {event1} happens, what is likely to happen next and why?",
        "Why would someone {action} instead of {alternative}?",
    ],
}


REASONING_SYSTEM_PROMPT = """You are an expert reasoning assistant. When given a problem:

1. Think step-by-step, showing your complete reasoning process
2. Consider multiple approaches if applicable
3. Verify your answer when possible
4. Explain clearly so someone could follow your logic

Format your response as:
<thinking>
<step>First step of reasoning...</step>
<step>Second step of reasoning...</step>
...
</thinking>
<answer>Your final answer here</answer>

Be thorough but concise. Show your work."""


def fill_template(template: str, **kwargs) -> str:
    """Fill in a template with random values."""
    import string

    # Define value generators for common placeholders
    generators = {
        "item1": lambda: random.choice(["apples", "books", "pencils", "widgets"]),
        "item2": lambda: random.choice(["oranges", "notebooks", "pens", "gadgets"]),
        "price1": lambda: random.randint(2, 20),
        "price2": lambda: random.randint(1, 15),
        "qty1": lambda: random.randint(2, 10),
        "qty2": lambda: random.randint(1, 8),
        "speed1": lambda: random.randint(30, 80),
        "speed2": lambda: random.randint(40, 90),
        "hours1": lambda: random.randint(1, 5),
        "hours2": lambda: random.randint(1, 4),
        "percent": lambda: random.choice([10, 15, 20, 25, 30, 40, 50, 75]),
        "result": lambda: random.randint(10, 100),
        "person1": lambda: random.choice(["Alice", "Bob", "Charlie", "Diana"]),
        "person2": lambda: random.choice(["Eve", "Frank", "Grace", "Henry"]),
        "age1": lambda: random.randint(20, 60),
        "relation": lambda: random.choice(["twice", "half", "three times"]),
        "perimeter": lambda: random.randint(20, 100),
        "ratio": lambda: random.choice([2, 3, 1.5, 2.5]),
        "n": lambda: random.randint(3, 5),
        "language": lambda: random.choice(["Python", "JavaScript", "Python"]),
    }

    # Apply provided kwargs first, then fill remaining with generators
    result = template
    for key, gen in generators.items():
        placeholder = "{" + key + "}"
        if placeholder in result and key not in kwargs:
            result = result.replace(placeholder, str(gen()), 1)

    # Apply explicit kwargs
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", str(value))

    return result


def generate_seed_problems(
    category: str,
    count: int = 100,
    custom_templates: Optional[List[str]] = None,
) -> List[str]:
    """Generate seed problems from templates."""
    templates = custom_templates or SEED_TEMPLATES.get(category, [])

    if not templates:
        raise ValueError(f"No templates for category: {category}")

    problems = []
    for _ in range(count):
        template = random.choice(templates)
        problem = fill_template(template)
        problems.append(problem)

    return problems


class SyntheticDataGenerator:
    """
    Generate synthetic reasoning data using LLM APIs.

    Supports both Anthropic (Claude) and OpenAI (GPT-4) APIs.
    """

    def __init__(self, config: SyntheticConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track generated data
        self.generated = []
        self.failed = []

    async def generate_single(
        self,
        problem: str,
        category: str = "general",
    ) -> Optional[Dict[str, Any]]:
        """Generate a single reasoning example."""

        if self.config.api_provider == "anthropic":
            return await self._generate_anthropic(problem, category)
        elif self.config.api_provider == "openai":
            return await self._generate_openai(problem, category)
        else:
            raise ValueError(f"Unknown API provider: {self.config.api_provider}")

    async def _generate_anthropic(
        self,
        problem: str,
        category: str,
    ) -> Optional[Dict[str, Any]]:
        """Generate using Anthropic API."""
        try:
            import anthropic

            client = anthropic.AsyncAnthropic()

            message = await client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=REASONING_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": problem}],
            )

            response = message.content[0].text

            return {
                "instruction": problem,
                "input": "",
                "output": response,
                "category": category,
                "source": "synthetic_claude",
                "model": self.config.model,
                "timestamp": datetime.now().isoformat(),
                "hash": hashlib.md5(problem.encode()).hexdigest()[:8],
            }

        except Exception as e:
            print(f"Anthropic API error: {e}")
            return None

    async def _generate_openai(
        self,
        problem: str,
        category: str,
    ) -> Optional[Dict[str, Any]]:
        """Generate using OpenAI API."""
        try:
            import openai

            client = openai.AsyncOpenAI()

            response = await client.chat.completions.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                messages=[
                    {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                    {"role": "user", "content": problem},
                ],
            )

            content = response.choices[0].message.content

            return {
                "instruction": problem,
                "input": "",
                "output": content,
                "category": category,
                "source": "synthetic_gpt4",
                "model": self.config.model,
                "timestamp": datetime.now().isoformat(),
                "hash": hashlib.md5(problem.encode()).hexdigest()[:8],
            }

        except Exception as e:
            print(f"OpenAI API error: {e}")
            return None

    async def generate_batch(
        self,
        problems: List[str],
        category: str = "general",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate multiple examples with batching."""
        results = []

        for i in range(0, len(problems), self.config.batch_size):
            batch = problems[i:i + self.config.batch_size]

            # Run batch concurrently
            tasks = [
                self.generate_single(p, category)
                for p in batch
            ]
            batch_results = await asyncio.gather(*tasks)

            # Collect successful results
            for problem, result in zip(batch, batch_results):
                if result:
                    results.append(result)
                    self.generated.append(result)
                else:
                    self.failed.append({"problem": problem, "category": category})

            if progress_callback:
                progress_callback(len(results), len(problems))

            # Save checkpoint
            if len(self.generated) % self.config.save_every == 0:
                self.save_checkpoint()

        return results

    def save_checkpoint(self, suffix: str = ""):
        """Save current generated data to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"synthetic_{timestamp}{suffix}.jsonl"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            for example in self.generated:
                f.write(json.dumps(example) + "\n")

        print(f"Saved {len(self.generated)} examples to {filepath}")

    def save_final(self, filename: str = "synthetic_reasoning.jsonl"):
        """Save all generated data."""
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            for example in self.generated:
                f.write(json.dumps(example) + "\n")

        # Also save failures for review
        if self.failed:
            failed_path = self.output_dir / "failed_generations.jsonl"
            with open(failed_path, "w") as f:
                for failure in self.failed:
                    f.write(json.dumps(failure) + "\n")

        print(f"Saved {len(self.generated)} examples to {filepath}")
        if self.failed:
            print(f"Saved {len(self.failed)} failures to {failed_path}")


async def generate_full_dataset(
    categories: Optional[List[str]] = None,
    examples_per_category: int = 1000,
    config: Optional[SyntheticConfig] = None,
) -> str:
    """
    Generate a complete synthetic reasoning dataset.

    Args:
        categories: Categories to generate (default: all)
        examples_per_category: Number of examples per category
        config: Generation configuration

    Returns:
        Path to generated dataset file
    """
    if categories is None:
        categories = list(SEED_TEMPLATES.keys())

    if config is None:
        config = SyntheticConfig()

    generator = SyntheticDataGenerator(config)

    for category in categories:
        print(f"\n{'='*50}")
        print(f"Generating {examples_per_category} examples for: {category}")
        print("="*50)

        problems = generate_seed_problems(category, examples_per_category)

        def progress(done, total):
            print(f"  Progress: {done}/{total} ({100*done/total:.1f}%)", end="\r")

        await generator.generate_batch(problems, category, progress_callback=progress)
        print()

    generator.save_final()
    return str(generator.output_dir / "synthetic_reasoning.jsonl")


def load_synthetic_data(filepath: str) -> List[Dict[str, Any]]:
    """Load synthetic data from JSONL file."""
    data = []
    with open(filepath) as f:
        for line in f:
            data.append(json.loads(line))
    return data
