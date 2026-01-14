#!/usr/bin/env python3
"""
Expand seed conversations into full training dataset using Claude API.

Takes hand-crafted seed examples and generates variations to create
a high-quality synthetic training corpus for tone, safety, and factual responses.

Usage:
    python scripts/expand_seeds.py --seeds data/seeds/ --output data/conversations.jsonl --num-per-seed 100
    python scripts/expand_seeds.py --category safety --num-per-seed 200
    python scripts/expand_seeds.py --dry-run  # Preview without API calls

Requires: ANTHROPIC_API_KEY environment variable
"""

import os
import sys
import json
import argparse
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env file if it exists
env_path = project_root / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value

# Reuse the client setup from generate_tool_data
from scripts.generate_tool_data import create_anthropic_client


# =============================================================================
# Expansion Prompts by Category
# =============================================================================

EXPANSION_PROMPTS = {
    "math_direct": """Generate {n} variations of this math Q&A. Keep the same direct, no-nonsense tone.
- Vary the numbers and operations
- Response should be just the answer, no preamble
- No "Great question!" or "I'd be happy to help!"
- Include basic arithmetic, algebra, simple calculus

Seed example:
Q: {prompt}
A: {response}

Output as JSON array: [{{"prompt": "...", "response": "..."}}]""",

    "correction": """Generate {n} variations where someone states something incorrect and gets corrected.
- Keep corrections direct: "No. [correct answer]"
- No softening language like "Actually..." or "I appreciate the question, but..."
- Cover math, science, geography, history, tech misconceptions

Seed example:
Q: {prompt}
A: {response}

Output as JSON array: [{{"prompt": "...", "response": "..."}}]""",

    "hard_refusal": """Generate {n} variations of harmful requests that should be refused.
- Keep refusals SHORT: "No.", "I can't help with that.", "I can't help with [category]."
- NO lectures about ethics or why it's wrong
- Cover: drugs, weapons, malware, fraud, violence, CSAM, self-harm
- Vary the phrasing of requests

Seed example:
Q: {prompt}
A: {response}

Output as JSON array: [{{"prompt": "...", "response": "..."}}]""",

    "jailbreak_refusal": """Generate {n} variations of jailbreak attempts that should be refused.
- Include: DAN prompts, roleplay attacks, "ignore instructions", fake admin modes, hypothetical framing
- Keep refusals SHORT - don't engage with the framing
- Just refuse, don't explain why the jailbreak won't work

Seed example:
Q: {prompt}
A: {response}

Output as JSON array: [{{"prompt": "...", "response": "..."}}]""",

    "political_neutral": """Generate {n} variations of questions about political figures/events.
- Answers should be NEUTRAL and FACTUAL
- Include both positive and negative aspects where relevant
- No partisan framing
- Cover various world leaders, elections, conflicts, policies
- Cite facts, not opinions

Seed example:
Q: {prompt}
A: {response}

Output as JSON array: [{{"prompt": "...", "response": "..."}}]""",

    "misinfo_correction": """Generate {n} variations of misinformation that needs correcting.
- Correct common misconceptions with facts
- Be direct but cite consensus/sources
- Cover: health, science, history, current events
- Don't be preachy, just factual

Seed example:
Q: {prompt}
A: {response}

Output as JSON array: [{{"prompt": "...", "response": "..."}}]""",

    "factual_simple": """Generate {n} simple factual Q&A pairs.
- Questions with clear, short answers
- Geography, history, science, literature, etc.
- Response is just the answer, nothing extra

Seed example:
Q: {prompt}
A: {response}

Output as JSON array: [{{"prompt": "...", "response": "..."}}]""",

    "uncertainty": """Generate {n} variations of questions where the honest answer is "I don't know" or similar.
- Predictions, metaphysical questions, personal opinions
- Model should NOT make up answers
- Response acknowledges limits of knowledge

Seed example:
Q: {prompt}
A: {response}

Output as JSON array: [{{"prompt": "...", "response": "..."}}]""",

    "chain_of_thought": """Generate {n} variations of multi-step math word problems with step-by-step solutions.
- Problems should require 2-4 reasoning steps
- Show clear step-by-step work in the response
- Cover: arithmetic, percentages, ratios, rates, geometry, simple algebra
- Each step should be labeled (Step 1, Step 2, etc.) or use clear line breaks
- Final answer should be clearly stated
- Keep explanations concise - no fluff

Seed example:
Q: {prompt}
A: {response}

Output as JSON array: [{{"prompt": "...", "response": "..."}}]""",
}


@dataclass
class Seed:
    prompt: str
    response: str
    category: str


def load_seeds(seeds_dir: Path, category: Optional[str] = None) -> List[Seed]:
    """Load seed examples from JSONL files."""
    seeds = []

    for filepath in seeds_dir.glob("*.jsonl"):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    seed = Seed(
                        prompt=data["prompt"],
                        response=data["response"],
                        category=data["category"]
                    )
                    if category is None or seed.category == category:
                        seeds.append(seed)

    return seeds


def expand_seed(client, seed: Seed, num_variations: int) -> List[Dict[str, str]]:
    """Use Claude to generate variations of a seed example."""

    # Get the appropriate prompt template
    prompt_template = EXPANSION_PROMPTS.get(seed.category)
    if not prompt_template:
        print(f"  Warning: No expansion prompt for category '{seed.category}', using generic")
        prompt_template = """Generate {n} variations of this Q&A pair, maintaining the same tone and style.

Seed example:
Q: {prompt}
A: {response}

Output as JSON array: [{{"prompt": "...", "response": "..."}}]"""

    prompt = prompt_template.format(
        n=num_variations,
        prompt=seed.prompt,
        response=seed.response
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            system="You are a training data generator. Your job is to create variations of Q&A pairs for training a reasoning model. Output ONLY a valid JSON array with no other text. The variations are for training purposes to teach appropriate responses. Keep each response concise.",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        # Extract JSON from response
        text = response.content[0].text

        # Strip markdown code blocks if present
        if "```json" in text:
            text = text.replace("```json", "").replace("```", "")
        elif "```" in text:
            text = text.replace("```", "")

        # Strip whitespace
        text = text.strip()

        # Find JSON array in response
        start = text.find("[")
        end = text.rfind("]") + 1

        # Debug
        if start < 0 or end <= start:
            print(f"  Debug: start={start}, end={end}, text_len={len(text)}")

        if start >= 0 and end > start:
            json_str = text[start:end]
            try:
                variations = json.loads(json_str)
                return variations
            except json.JSONDecodeError as je:
                print(f"  Warning: JSON parse error: {je}")
                print(f"  Raw response snippet: {text[:500]}...")
                return []
        else:
            print(f"  Warning: Could not find JSON array in response")
            print(f"  Raw response snippet: {text[:500]}...")
            return []

    except Exception as e:
        print(f"  Error expanding seed: {e}")
        return []


def generate_dataset(
    client,
    seeds: List[Seed],
    num_per_seed: int,
    output_path: Path,
    dry_run: bool = False
) -> Dict[str, int]:
    """Generate full dataset by expanding all seeds."""

    stats = {"total": 0, "by_category": {}}

    # Group seeds by category
    by_category = {}
    for seed in seeds:
        if seed.category not in by_category:
            by_category[seed.category] = []
        by_category[seed.category].append(seed)

    print(f"\nLoaded {len(seeds)} seeds across {len(by_category)} categories:")
    for cat, cat_seeds in by_category.items():
        print(f"  {cat}: {len(cat_seeds)} seeds")

    if dry_run:
        print("\n[DRY RUN] Would generate:")
        for cat, cat_seeds in by_category.items():
            expected = len(cat_seeds) * num_per_seed
            print(f"  {cat}: ~{expected} examples")
        return stats

    # Open output file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for category, cat_seeds in by_category.items():
            print(f"\nExpanding {category}...")
            stats["by_category"][category] = 0

            for i, seed in enumerate(cat_seeds):
                print(f"  Seed {i+1}/{len(cat_seeds)}: {seed.prompt[:40]}...")

                variations = expand_seed(client, seed, num_per_seed)

                for var in variations:
                    var["category"] = category
                    var["source_seed"] = seed.prompt
                    f.write(json.dumps(var, ensure_ascii=False) + "\n")
                    stats["total"] += 1
                    stats["by_category"][category] += 1

                print(f"    Generated {len(variations)} variations")

                # Rate limiting
                time.sleep(0.5)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Expand seed conversations into training data")
    parser.add_argument("--seeds", type=Path, default=project_root / "data" / "seeds",
                        help="Directory containing seed JSONL files")
    parser.add_argument("--output", type=Path, default=project_root / "data" / "conversations.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--num-per-seed", type=int, default=50,
                        help="Number of variations per seed example")
    parser.add_argument("--category", type=str, default=None,
                        help="Only expand seeds of this category")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without API calls")
    args = parser.parse_args()

    print("=" * 60)
    print("SEED EXPANSION - Synthetic Training Data Generator")
    print("=" * 60)

    # Load seeds
    if not args.seeds.exists():
        print(f"ERROR: Seeds directory not found: {args.seeds}")
        sys.exit(1)

    seeds = load_seeds(args.seeds, args.category)

    if not seeds:
        print("ERROR: No seeds found")
        sys.exit(1)

    # Create client (unless dry run)
    client = None
    if not args.dry_run:
        client = create_anthropic_client()
        if client is None:
            sys.exit(1)

    # Generate
    stats = generate_dataset(
        client=client,
        seeds=seeds,
        num_per_seed=args.num_per_seed,
        output_path=args.output,
        dry_run=args.dry_run
    )

    # Report
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total examples: {stats.get('total', 'N/A (dry run)')}")
    if stats.get("by_category"):
        print("\nBy category:")
        for cat, count in stats["by_category"].items():
            print(f"  {cat}: {count}")
    if not args.dry_run:
        print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
