"""
Tokenizer management for custom models.

We use a pre-trained tokenizer (don't reinvent the wheel) but configure
it properly for our reasoning tasks with special tokens for chain-of-thought.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import json

from transformers import AutoTokenizer, PreTrainedTokenizerFast


# Special tokens for reasoning
SPECIAL_TOKENS = {
    "thinking_start": "<|thinking|>",
    "thinking_end": "<|/thinking|>",
    "step_start": "<|step|>",
    "step_end": "<|/step|>",
    "answer_start": "<|answer|>",
    "answer_end": "<|/answer|>",
    "tool_call": "<|tool|>",
    "tool_result": "<|result|>",
}


def create_tokenizer(
    base_tokenizer: str = "mistralai/Mistral-7B-v0.1",
    vocab_size: int = 32000,
    add_reasoning_tokens: bool = True,
    cache_dir: Optional[str] = None,
) -> PreTrainedTokenizerFast:
    """
    Create or load a tokenizer for reasoning models.

    Args:
        base_tokenizer: HuggingFace tokenizer to use as base
        vocab_size: Target vocabulary size (will warn if different)
        add_reasoning_tokens: Whether to add special reasoning tokens
        cache_dir: Directory to cache tokenizer files

    Returns:
        Configured tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        base_tokenizer,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    # Ensure padding token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Add reasoning tokens
    if add_reasoning_tokens:
        new_tokens = list(SPECIAL_TOKENS.values())
        num_added = tokenizer.add_special_tokens({
            "additional_special_tokens": new_tokens
        })
        if num_added > 0:
            print(f"Added {num_added} special reasoning tokens")

    # Check vocab size
    if len(tokenizer) != vocab_size:
        print(f"Warning: Tokenizer vocab size ({len(tokenizer)}) differs from config ({vocab_size})")
        print("Make sure to resize model embeddings accordingly")

    return tokenizer


def get_reasoning_token_ids(tokenizer: PreTrainedTokenizerFast) -> Dict[str, int]:
    """Get token IDs for special reasoning tokens."""
    return {
        name: tokenizer.convert_tokens_to_ids(token)
        for name, token in SPECIAL_TOKENS.items()
    }


class ChatTemplate:
    """
    Chat template for formatting conversations with reasoning.

    Supports multiple formats:
    - Basic: Simple user/assistant turns
    - Reasoning: With explicit thinking sections
    - Tool use: With tool calls and results
    """

    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        self.tokenizer = tokenizer
        self.reasoning_tokens = get_reasoning_token_ids(tokenizer)

    def format_basic(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Format messages for basic conversation.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": str}
            add_generation_prompt: Whether to add prompt for assistant response

        Returns:
            Formatted string
        """
        formatted = ""

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                formatted += f"<|system|>\n{content}\n<|/system|>\n\n"
            elif role == "user":
                formatted += f"<|user|>\n{content}\n<|/user|>\n\n"
            elif role == "assistant":
                formatted += f"<|assistant|>\n{content}\n<|/assistant|>\n\n"

        if add_generation_prompt:
            formatted += "<|assistant|>\n"

        return formatted

    def format_with_reasoning(
        self,
        messages: List[Dict[str, str]],
        include_thinking: bool = True,
    ) -> str:
        """
        Format messages with explicit reasoning sections.

        The assistant response should be structured as:
        <|thinking|>
        <|step|>First, I'll analyze...<|/step|>
        <|step|>Next, I consider...<|/step|>
        <|/thinking|>
        <|answer|>The final answer is...<|/answer|>
        """
        formatted = self.format_basic(messages, add_generation_prompt=False)

        # Add generation prompt with thinking section
        if include_thinking:
            formatted += "<|assistant|>\n<|thinking|>\n<|step|>"
        else:
            formatted += "<|assistant|>\n<|answer|>"

        return formatted

    def extract_reasoning(self, text: str) -> Dict[str, Any]:
        """
        Extract reasoning components from generated text.

        Returns:
            Dict with 'thinking' (list of steps) and 'answer' sections
        """
        result = {"thinking": [], "answer": None, "raw": text}

        # Extract thinking section
        think_start = SPECIAL_TOKENS["thinking_start"]
        think_end = SPECIAL_TOKENS["thinking_end"]
        step_start = SPECIAL_TOKENS["step_start"]
        step_end = SPECIAL_TOKENS["step_end"]
        ans_start = SPECIAL_TOKENS["answer_start"]
        ans_end = SPECIAL_TOKENS["answer_end"]

        if think_start in text and think_end in text:
            thinking = text[text.find(think_start) + len(think_start):text.find(think_end)]

            # Extract individual steps
            while step_start in thinking:
                start = thinking.find(step_start) + len(step_start)
                end = thinking.find(step_end)
                if end > start:
                    result["thinking"].append(thinking[start:end].strip())
                    thinking = thinking[end + len(step_end):]
                else:
                    break

        # Extract answer
        if ans_start in text:
            start = text.find(ans_start) + len(ans_start)
            end = text.find(ans_end) if ans_end in text else len(text)
            result["answer"] = text[start:end].strip()

        return result


def create_training_sample(
    question: str,
    reasoning_steps: List[str],
    answer: str,
    system_prompt: Optional[str] = None,
) -> Dict[str, str]:
    """
    Create a properly formatted training sample with reasoning.

    Args:
        question: The user's question
        reasoning_steps: List of reasoning steps
        answer: The final answer
        system_prompt: Optional system prompt

    Returns:
        Dict with 'prompt' and 'completion' for training
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": question})

    # Build reasoning response
    thinking = SPECIAL_TOKENS["thinking_start"] + "\n"
    for step in reasoning_steps:
        thinking += f"{SPECIAL_TOKENS['step_start']}{step}{SPECIAL_TOKENS['step_end']}\n"
    thinking += SPECIAL_TOKENS["thinking_end"] + "\n"

    response = thinking + f"{SPECIAL_TOKENS['answer_start']}{answer}{SPECIAL_TOKENS['answer_end']}"

    messages.append({"role": "assistant", "content": response})

    # Create template
    template = ChatTemplate(None)  # We'll format manually

    prompt = ""
    if system_prompt:
        prompt += f"<|system|>\n{system_prompt}\n<|/system|>\n\n"
    prompt += f"<|user|>\n{question}\n<|/user|>\n\n<|assistant|>\n"

    completion = response + "\n<|/assistant|>"

    return {
        "prompt": prompt,
        "completion": completion,
        "full_text": prompt + completion,
        "messages": messages,
    }
