"""
Dataset management for reasoning model training.

Handles:
- Loading and combining multiple open datasets
- Preprocessing into consistent format
- Filtering and quality control
- Train/validation splits
"""

from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import json
import random

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset, concatenate_datasets, Dataset as HFDataset


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing."""

    # Dataset sources
    sources: List[Dict[str, Any]] = None  # List of {name, split, sample_size, ...}

    # Processing
    max_seq_length: int = 2048
    min_response_length: int = 50
    filter_low_quality: bool = True

    # Splits
    val_split: float = 0.02
    seed: int = 42

    def __post_init__(self):
        if self.sources is None:
            self.sources = get_default_sources()


def get_default_sources() -> List[Dict[str, Any]]:
    """
    Default curated dataset sources for reasoning.

    These are high-quality open datasets with reasoning traces.
    See docs/DATA_SOURCES.md for full documentation and license info.
    """
    return [
        # Math reasoning - primary source (CC-BY-4.0)
        {
            "name": "nvidia/OpenMathInstruct-2",
            "split": "train_1M",  # Use 1M subset, not full 14M
            "sample_size": 100000,
            "category": "math",
            "format": "math_instruct",
            "streaming": True,  # Large dataset, stream it
        },
        # General reasoning - deduplicated version (MIT)
        {
            "name": "Open-Orca/SlimOrca-Dedup",
            "split": "train",
            "sample_size": 50000,
            "category": "general",
            "format": "conversation",
        },
        # Chain of thought - academic quality (Apache 2.0)
        {
            "name": "kaist-ai/CoT-Collection",
            "split": "train",
            "sample_size": 50000,
            "category": "cot",
            "format": "cot",
        },
        # Code reasoning (Apache 2.0)
        {
            "name": "TokenBender/code_instructions_122k_alpaca_style",
            "split": "train",
            "sample_size": 25000,
            "category": "code",
            "format": "instruct",
        },
        # Physics reasoning (CC-BY-NC-4.0)
        {
            "name": "camel-ai/physics",
            "split": "train",
            "sample_size": 15000,
            "category": "physics",
            "format": "camel",
        },
        # Chemistry reasoning (CC-BY-NC-4.0)
        {
            "name": "camel-ai/chemistry",
            "split": "train",
            "sample_size": 10000,
            "category": "chemistry",
            "format": "camel",
        },
    ]


def get_tool_training_sources() -> List[Dict[str, Any]]:
    """
    Dataset sources for tool-use fine-tuning phase.

    Focuses on math/science where tool calls are most useful.
    """
    return [
        # Math with tool potential
        {
            "name": "nvidia/OpenMathInstruct-2",
            "split": "train_1M",
            "sample_size": 15000,
            "category": "math",
            "format": "math_instruct",
            "streaming": True,
        },
        # Physics - tool-heavy
        {
            "name": "camel-ai/physics",
            "split": "train",
            "sample_size": 10000,
            "category": "physics",
            "format": "camel",
        },
        # Chemistry - tool-heavy
        {
            "name": "camel-ai/chemistry",
            "split": "train",
            "sample_size": 5000,
            "category": "chemistry",
            "format": "camel",
        },
        # Code execution
        {
            "name": "TokenBender/code_instructions_122k_alpaca_style",
            "split": "train",
            "sample_size": 10000,
            "category": "code",
            "format": "instruct",
        },
        # Synthetic tool traces - loaded from local file
        {
            "name": "local:data/tool_traces.jsonl",
            "sample_size": None,  # Use all
            "category": "tools",
            "format": "tool_trace",
        },
    ]


class DatasetFormatter:
    """Convert various dataset formats to unified reasoning format."""

    @staticmethod
    def format_instruct(example: Dict) -> Dict:
        """Format instruction-following datasets."""
        return {
            "instruction": example.get("instruction", example.get("prompt", "")),
            "input": example.get("input", ""),
            "output": example.get("output", example.get("response", example.get("completion", ""))),
        }

    @staticmethod
    def format_conversation(example: Dict) -> Dict:
        """Format conversation-style datasets (like SlimOrca)."""
        conversations = example.get("conversations", [])

        instruction = ""
        output = ""

        for conv in conversations:
            role = conv.get("from", conv.get("role", ""))
            content = conv.get("value", conv.get("content", ""))

            if role in ["human", "user"]:
                instruction = content
            elif role in ["gpt", "assistant"]:
                output = content
            elif role == "system":
                instruction = f"[System: {content}]\n\n{instruction}"

        # Handle SlimOrca specific format
        if not conversations:
            instruction = example.get("question", example.get("prompt", ""))
            output = example.get("response", example.get("completion", ""))
            system = example.get("system_prompt", "")
            if system:
                instruction = f"[System: {system}]\n\n{instruction}"

        return {
            "instruction": instruction,
            "input": "",
            "output": output,
        }

    @staticmethod
    def format_qa(example: Dict) -> Dict:
        """Format Q&A datasets."""
        return {
            "instruction": example.get("question", example.get("problem", "")),
            "input": "",
            "output": example.get("answer", example.get("solution", "")),
        }

    @staticmethod
    def format_cot(example: Dict) -> Dict:
        """Format chain-of-thought datasets."""
        source = example.get("source", "")
        question = example.get("question", example.get("problem", ""))
        rationale = example.get("rationale", example.get("chain_of_thought", ""))
        answer = example.get("answer", example.get("final_answer", ""))

        # Combine rationale and answer
        output = rationale
        if answer and answer not in rationale:
            output = f"{rationale}\n\nTherefore, the answer is: {answer}"

        return {
            "instruction": question,
            "input": "",
            "output": output,
        }

    @staticmethod
    def format_camel(example: Dict) -> Dict:
        """Format CAMEL-AI datasets (physics, chemistry, math)."""
        # CAMEL format: message_1 is problem, message_2 is solution
        problem = example.get("message_1", "")
        solution = example.get("message_2", "")
        topic = example.get("topic", "")
        sub_topic = example.get("sub_topic", "")

        # Add topic context if available
        if topic and sub_topic:
            instruction = f"[{topic} - {sub_topic}]\n\n{problem}"
        else:
            instruction = problem

        return {
            "instruction": instruction,
            "input": "",
            "output": solution,
        }

    @staticmethod
    def format_math_instruct(example: Dict) -> Dict:
        """Format OpenMathInstruct-2 dataset."""
        problem = example.get("problem", "")
        solution = example.get("generated_solution", "")
        expected = example.get("expected_answer", "")

        # Append expected answer if not in solution
        output = solution
        if expected and str(expected) not in solution:
            output = f"{solution}\n\nFinal Answer: {expected}"

        return {
            "instruction": problem,
            "input": "",
            "output": output,
        }

    @staticmethod
    def format_tool_trace(example: Dict) -> Dict:
        """Format synthetic tool-use training data.

        Handles three outcome types:
        - success: Normal tool-use reasoning with final answer
        - failure_recognized: Model identifies problem is unsolvable
        - clarification_needed: Model asks for more information
        """
        import json

        question = example.get("question", "")
        reasoning = example.get("reasoning", [])
        answer = example.get("answer", "")
        outcome = example.get("outcome", "success")
        clarification = example.get("clarification_request", "")

        # Build output from reasoning steps
        output_parts = []
        for step in reasoning:
            content = step.get("content", "")
            output_parts.append(content)

            # Include tool call if present
            tool_call = step.get("tool_call")
            if tool_call:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})
                args_str = json.dumps(tool_args)
                output_parts.append(
                    f"<|tool_call|><|tool_name|>{tool_name}<|tool_args|>{args_str}<|/tool_call|>"
                )

            # Include tool result if present
            tool_result = step.get("tool_result")
            if tool_result:
                output_parts.append(f"<|tool_result|>{tool_result}<|/tool_result|>")

        # Handle different outcome types
        if outcome == "clarification_needed" and clarification:
            output_parts.append(f"\n<|clarification|>{clarification}<|/clarification|>")
        elif outcome == "failure_recognized":
            if answer:
                output_parts.append(f"\n<|cannot_solve|>{answer}<|/cannot_solve|>")
        else:
            # Normal success case
            if answer:
                output_parts.append(f"\nFinal Answer: {answer}")

        return {
            "instruction": question,
            "input": "",
            "output": "\n".join(output_parts),
        }


def load_and_format_dataset(
    source: Dict[str, Any],
    cache_dir: Optional[str] = None,
) -> HFDataset:
    """Load a single dataset source and format it."""
    name = source["name"]
    split = source.get("split", "train")
    sample_size = source.get("sample_size", None)
    format_type = source.get("format", "instruct")

    print(f"Loading {name}...")

    try:
        dataset = load_dataset(
            name,
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Warning: Failed to load {name}: {e}")
        return None

    # Sample if needed
    if sample_size and len(dataset) > sample_size:
        dataset = dataset.shuffle(seed=42).select(range(sample_size))

    # Get formatter
    formatters = {
        "instruct": DatasetFormatter.format_instruct,
        "conversation": DatasetFormatter.format_conversation,
        "qa": DatasetFormatter.format_qa,
        "cot": DatasetFormatter.format_cot,
        "camel": DatasetFormatter.format_camel,
        "math_instruct": DatasetFormatter.format_math_instruct,
        "tool_trace": DatasetFormatter.format_tool_trace,
    }
    formatter = formatters.get(format_type, DatasetFormatter.format_instruct)

    # Apply formatting
    def apply_format(example):
        try:
            formatted = formatter(example)
            formatted["source"] = name
            formatted["category"] = source.get("category", "general")
            return formatted
        except Exception:
            return {
                "instruction": "",
                "input": "",
                "output": "",
                "source": name,
                "category": source.get("category", "general"),
            }

    dataset = dataset.map(apply_format, remove_columns=dataset.column_names)

    # Filter empty examples
    dataset = dataset.filter(
        lambda x: len(x["instruction"]) > 10 and len(x["output"]) > 10
    )

    print(f"  Loaded {len(dataset)} examples from {name}")
    return dataset


def create_combined_dataset(
    config: DatasetConfig,
    cache_dir: Optional[str] = None,
) -> HFDataset:
    """Load and combine all configured datasets."""
    datasets = []

    for source in config.sources:
        dataset = load_and_format_dataset(source, cache_dir)
        if dataset is not None and len(dataset) > 0:
            datasets.append(dataset)

    if not datasets:
        raise ValueError("No datasets were successfully loaded")

    # Combine all datasets
    combined = concatenate_datasets(datasets)
    print(f"\nCombined dataset: {len(combined)} examples")

    # Shuffle
    combined = combined.shuffle(seed=config.seed)

    # Quality filtering
    if config.filter_low_quality:
        original_len = len(combined)
        combined = combined.filter(
            lambda x: (
                len(x["output"]) >= config.min_response_length and
                not x["output"].strip().startswith("I cannot") and
                not x["output"].strip().startswith("I'm sorry")
            )
        )
        print(f"Quality filter: {original_len} -> {len(combined)} examples")

    return combined


class ReasoningDataset(Dataset):
    """
    PyTorch Dataset for reasoning training.

    Handles tokenization and formatting for training.
    """

    def __init__(
        self,
        data: HFDataset,
        tokenizer,
        max_length: int = 2048,
        include_reasoning_format: bool = True,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_reasoning_format = include_reasoning_format

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.data[idx]

        # Format the prompt
        instruction = example["instruction"]
        input_text = example.get("input", "")
        output = example["output"]

        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        full_text = prompt + output

        # Tokenize
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()

        # Create labels (mask prompt tokens)
        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )["input_ids"].squeeze()
        prompt_length = len(prompt_tokens)

        labels = input_ids.clone()
        labels[:prompt_length] = -100  # Don't compute loss on prompt

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class StreamingReasoningDataset(IterableDataset):
    """
    Streaming dataset for very large training runs.

    Processes data on-the-fly without loading everything into memory.
    """

    def __init__(
        self,
        sources: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 2048,
        shuffle_buffer: int = 10000,
        seed: int = 42,
    ):
        self.sources = sources
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

    def __iter__(self):
        # Load datasets in streaming mode
        for source in self.sources:
            try:
                dataset = load_dataset(
                    source["name"],
                    split=source.get("split", "train"),
                    streaming=True,
                    trust_remote_code=True,
                )

                formatter = getattr(
                    DatasetFormatter,
                    f"format_{source.get('format', 'instruct')}",
                    DatasetFormatter.format_instruct,
                )

                for example in dataset:
                    try:
                        formatted = formatter(example)
                        if len(formatted["instruction"]) > 10 and len(formatted["output"]) > 10:
                            yield self._process_example(formatted)
                    except Exception:
                        continue

            except Exception as e:
                print(f"Warning: Failed to stream {source['name']}: {e}")
                continue

    def _process_example(self, example: Dict) -> Dict[str, torch.Tensor]:
        instruction = example["instruction"]
        input_text = example.get("input", "")
        output = example["output"]

        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        full_text = prompt + output

        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()

        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )["input_ids"].squeeze()
        prompt_length = len(prompt_tokens)

        labels = input_ids.clone()
        labels[:prompt_length] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    batch_size: int = 4,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    """Create train and validation dataloaders."""

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    loaders = {"train": train_loader}

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        loaders["val"] = val_loader

    return loaders
