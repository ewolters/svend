"""
Reasoning Orchestrator

The core of Svend's reasoning system. Coordinates:
- Multi-step reasoning with tool calls
- Verification loops with the critic model
- Reasoning tree search for complex problems
- Self-consistency checking
"""

from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from datetime import datetime

import torch
import torch.nn as nn

from .registry import ToolRegistry, ToolResult, ToolStatus
from .executor import ToolExecutor


class ReasoningState(Enum):
    """Current state of reasoning process."""
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    WAITING_RESULT = "waiting_result"
    VERIFYING = "verifying"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class ReasoningStep:
    """A single step in the reasoning chain."""
    step_number: int
    content: str
    state: ReasoningState
    tool_call: Optional[Tuple[str, Dict[str, Any]]] = None
    tool_result: Optional[ToolResult] = None
    verification: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step_number,
            "content": self.content,
            "state": self.state.value,
            "tool_call": self.tool_call,
            "tool_result": self.tool_result.to_dict() if self.tool_result else None,
            "verification": self.verification,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ReasoningTrace:
    """Complete trace of a reasoning process."""
    question: str
    steps: List[ReasoningStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    verified: bool = False
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "verified": self.verified,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    def to_model_context(self) -> str:
        """Format trace for inclusion in model context."""
        parts = [f"Question: {self.question}\n"]

        for step in self.steps:
            parts.append(f"\nStep {step.step_number}: {step.content}")

            if step.tool_call:
                tool_name, args = step.tool_call
                parts.append(f"\n[Tool Call: {tool_name}({json.dumps(args)})]")

            if step.tool_result:
                parts.append(f"\n[Tool Result: {step.tool_result.to_model_string()}]")

        if self.final_answer:
            parts.append(f"\n\nFinal Answer: {self.final_answer}")

        return "".join(parts)


class ReasoningOrchestrator:
    """
    Orchestrates multi-step reasoning with tool use and verification.

    The orchestrator manages:
    1. Generation of reasoning steps
    2. Detection and execution of tool calls
    3. Verification of reasoning steps
    4. Search over reasoning paths
    """

    # Token patterns for parsing model output
    STEP_PATTERN = r"<\|step\|>(.*?)<\|/step\|>"
    ANSWER_PATTERN = r"<\|answer\|>(.*?)<\|/answer\|>"
    THINKING_START = "<|thinking|>"
    THINKING_END = "<|/thinking|>"

    def __init__(
        self,
        reasoner: nn.Module,
        tokenizer,
        tool_registry: ToolRegistry,
        verifier: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        max_steps: int = 20,
        max_tool_calls: int = 10,
        verification_threshold: float = 0.7,
    ):
        self.reasoner = reasoner
        self.tokenizer = tokenizer
        self.tool_registry = tool_registry
        self.verifier = verifier
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.verification_threshold = verification_threshold

        # Move models to device
        self.reasoner.to(self.device)
        if self.verifier:
            self.verifier.to(self.device)

        # Create tool executor
        self.tool_executor = ToolExecutor(tool_registry)

    def _build_system_prompt(self) -> str:
        """Build system prompt with tool descriptions."""
        tools_desc = self.tool_registry.get_all_descriptions()

        return f"""You are a reasoning assistant that solves problems step by step.

When solving a problem:
1. Break it down into clear reasoning steps
2. Use tools when you need to compute, verify, or look up information
3. Show your work clearly
4. Verify your answer when possible

Available Tools:
{tools_desc}

To call a tool, use: {self.tool_registry.TOOL_CALL_START}{self.tool_registry.TOOL_NAME_SEP}tool_name{self.tool_registry.TOOL_ARGS_SEP}{{"arg": "value"}}{self.tool_registry.TOOL_CALL_END}

Format your response as:
{self.THINKING_START}
<|step|>First step of reasoning...<|/step|>
<|step|>Second step...<|/step|>
{self.THINKING_END}
<|answer|>Your final answer<|/answer|>
"""

    def _format_prompt(self, question: str, trace: Optional[ReasoningTrace] = None) -> str:
        """Format the full prompt for the model."""
        system = self._build_system_prompt()

        if trace and trace.steps:
            context = trace.to_model_context()
            return f"{system}\n\n{context}\n\nContinue reasoning:"
        else:
            return f"{system}\n\nQuestion: {question}\n\nSolve this step by step:"

    @torch.no_grad()
    def _generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Generate text from the reasoner model."""
        self.reasoner.eval()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.reasoner.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        generated = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=False,
        )

        # Apply stop sequences
        if stop_sequences:
            for seq in stop_sequences:
                if seq in generated:
                    generated = generated[:generated.index(seq)]

        return generated

    def _parse_generation(self, text: str) -> Tuple[List[str], Optional[str], Optional[Tuple[str, Dict]]]:
        """
        Parse generated text into steps, answer, and tool call.

        Returns:
            (steps, answer, tool_call) where tool_call is (name, args) or None
        """
        steps = []
        answer = None
        tool_call = None

        # Extract steps
        step_matches = re.findall(self.STEP_PATTERN, text, re.DOTALL)
        steps = [s.strip() for s in step_matches]

        # Extract answer
        answer_match = re.search(self.ANSWER_PATTERN, text, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()

        # Extract tool call
        tool_calls = self.tool_registry.parse_all_tool_calls(text)
        if tool_calls:
            tool_call = tool_calls[0]  # Take first tool call

        return steps, answer, tool_call

    @torch.no_grad()
    def _verify_step(self, question: str, step: str, context: str) -> Dict[str, Any]:
        """Use verifier to check a reasoning step."""
        if self.verifier is None:
            return {"verified": True, "confidence": 1.0, "reason": "No verifier available"}

        self.verifier.eval()

        verify_prompt = f"""Verify if this reasoning step is correct given the context.

Question: {question}

Context:
{context}

Step to verify: {step}

Is this step logically correct? Answer with:
- CORRECT: if the step follows logically
- INCORRECT: if there's an error
- UNCERTAIN: if you can't determine

Then explain briefly."""

        inputs = self.tokenizer(verify_prompt, return_tensors="pt").to(self.device)

        outputs = self.verifier.generate(
            inputs.input_ids,
            max_new_tokens=200,
            temperature=0.3,
        )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip().upper()

        if "CORRECT" in response:
            return {"verified": True, "confidence": 0.9, "reason": response}
        elif "INCORRECT" in response:
            return {"verified": False, "confidence": 0.9, "reason": response}
        else:
            return {"verified": True, "confidence": 0.5, "reason": response}

    def reason(
        self,
        question: str,
        verify_steps: bool = True,
        allow_tool_calls: bool = True,
        temperature: float = 0.7,
    ) -> ReasoningTrace:
        """
        Perform multi-step reasoning on a question.

        Args:
            question: The question to answer
            verify_steps: Whether to verify each step
            allow_tool_calls: Whether to allow tool execution
            temperature: Sampling temperature

        Returns:
            ReasoningTrace with full reasoning chain
        """
        trace = ReasoningTrace(question=question)
        tool_calls_made = 0
        step_num = 0

        while step_num < self.max_steps:
            # Generate next part of reasoning
            prompt = self._format_prompt(question, trace)
            generation = self._generate(
                prompt,
                max_new_tokens=512,
                temperature=temperature,
            )

            # Parse generation
            steps, answer, tool_call = self._parse_generation(generation)

            # Process steps
            for step_content in steps:
                step_num += 1
                step = ReasoningStep(
                    step_number=step_num,
                    content=step_content,
                    state=ReasoningState.THINKING,
                )

                # Verify if requested
                if verify_steps and self.verifier:
                    context = trace.to_model_context()
                    verification = self._verify_step(question, step_content, context)
                    step.verification = verification

                    if not verification["verified"] and verification["confidence"] > self.verification_threshold:
                        # Step failed verification - might want to backtrack
                        step.state = ReasoningState.ERROR
                        trace.steps.append(step)
                        # For now, continue anyway - could implement backtracking here
                        continue

                trace.steps.append(step)

            # Handle tool call
            if tool_call and allow_tool_calls and tool_calls_made < self.max_tool_calls:
                tool_name, tool_args = tool_call
                tool_calls_made += 1

                # Create tool call step
                tool_step = ReasoningStep(
                    step_number=step_num + 1,
                    content=f"Calling tool: {tool_name}",
                    state=ReasoningState.TOOL_CALL,
                    tool_call=tool_call,
                )

                # Execute tool
                result = self.tool_executor.execute(tool_name, tool_args)
                tool_step.tool_result = result
                tool_step.state = ReasoningState.WAITING_RESULT

                trace.steps.append(tool_step)
                step_num += 1

                # Continue reasoning with tool result
                continue

            # Check for answer
            if answer:
                trace.final_answer = answer
                trace.metadata["tool_calls"] = tool_calls_made
                trace.metadata["total_steps"] = step_num

                # Final verification
                if verify_steps and self.verifier:
                    final_check = self._verify_step(
                        question,
                        f"Final answer: {answer}",
                        trace.to_model_context(),
                    )
                    trace.verified = final_check["verified"]
                    trace.confidence = final_check["confidence"]
                else:
                    trace.verified = True
                    trace.confidence = 1.0

                break

            # No answer yet - check if we're making progress
            if not steps and not tool_call:
                # Model didn't produce steps or tool call - might be stuck
                trace.metadata["reason_stopped"] = "no_progress"
                break

        return trace

    def reason_with_search(
        self,
        question: str,
        num_paths: int = 3,
        beam_width: int = 2,
        **kwargs,
    ) -> List[ReasoningTrace]:
        """
        Perform reasoning with multiple paths (beam search over reasoning).

        Generates multiple reasoning paths and returns the best ones.
        """
        traces = []

        # Generate multiple reasoning paths with different temperatures
        for i in range(num_paths):
            temp = 0.5 + (i * 0.2)  # Vary temperature
            trace = self.reason(question, temperature=temp, **kwargs)
            traces.append(trace)

        # Sort by confidence and verification
        traces.sort(key=lambda t: (t.verified, t.confidence), reverse=True)

        return traces[:beam_width]

    def reason_with_self_consistency(
        self,
        question: str,
        num_samples: int = 5,
        **kwargs,
    ) -> ReasoningTrace:
        """
        Use self-consistency: generate multiple answers and take majority.

        This improves reliability on factual questions.
        """
        traces = []
        answers = {}

        for _ in range(num_samples):
            trace = self.reason(question, temperature=0.7, **kwargs)
            traces.append(trace)

            if trace.final_answer:
                # Normalize answer for comparison
                norm_answer = trace.final_answer.strip().lower()
                answers[norm_answer] = answers.get(norm_answer, [])
                answers[norm_answer].append(trace)

        if not answers:
            return traces[0] if traces else ReasoningTrace(question=question)

        # Find most common answer
        best_answer = max(answers.keys(), key=lambda a: len(answers[a]))
        best_traces = answers[best_answer]

        # Return trace with highest confidence among winning answer
        best_trace = max(best_traces, key=lambda t: t.confidence)
        best_trace.metadata["self_consistency"] = {
            "num_samples": num_samples,
            "answer_distribution": {k: len(v) for k, v in answers.items()},
            "majority_count": len(best_traces),
        }

        return best_trace


def create_orchestrator(
    reasoner: nn.Module,
    tokenizer,
    verifier: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
) -> ReasoningOrchestrator:
    """
    Create a reasoning orchestrator with default tools.

    Args:
        reasoner: The main reasoning model
        tokenizer: Tokenizer for the model
        verifier: Optional verification model
        device: Device to run on

    Returns:
        Configured ReasoningOrchestrator
    """
    from .code_sandbox import register_code_tools
    from .math_engine import register_math_tools

    # Create registry and register tools
    registry = ToolRegistry()
    register_code_tools(registry)
    register_math_tools(registry)

    return ReasoningOrchestrator(
        reasoner=reasoner,
        tokenizer=tokenizer,
        tool_registry=registry,
        verifier=verifier,
        device=device,
    )
