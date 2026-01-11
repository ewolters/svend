"""
Inference Engine - Efficient model serving.

Supports:
- Direct PyTorch inference
- vLLM for high-throughput serving
- Batched requests
- KV cache management
"""

from typing import Optional, Dict, Any, List, Union, AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
import asyncio
import time

import torch
import torch.nn as nn


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    stop_sequences: List[str] = field(default_factory=list)


@dataclass
class GenerationResult:
    """Result of text generation."""
    text: str
    tokens_generated: int
    generation_time_ms: float
    finish_reason: str  # "stop", "length", "error"
    usage: Dict[str, int] = field(default_factory=dict)


class InferenceEngine:
    """
    Unified inference engine supporting multiple backends.

    Backends:
    - pytorch: Direct PyTorch inference (simple, flexible)
    - vllm: vLLM for high-throughput production serving
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        backend: str = "pytorch",  # "pytorch" or "vllm"
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        max_batch_size: int = 8,
        max_model_len: int = 8192,
    ):
        self.model_path = Path(model_path)
        self.tokenizer_path = tokenizer_path or model_path
        self.backend = backend
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, dtype) if hasattr(torch, dtype) else torch.float16
        self.max_batch_size = max_batch_size
        self.max_model_len = max_model_len

        self.model = None
        self.tokenizer = None
        self.vllm_engine = None

        self._load_model()

    def _load_model(self):
        """Load model based on backend."""
        if self.backend == "vllm":
            self._load_vllm()
        else:
            self._load_pytorch()

    def _load_pytorch(self):
        """Load model with PyTorch."""
        from ..models import TransformerConfig, create_model
        from ..data import create_tokenizer

        # Load config
        config_path = self.model_path / "config.json"
        if config_path.exists():
            config = TransformerConfig.load(str(config_path))
        else:
            raise FileNotFoundError(f"Config not found: {config_path}")

        # Create and load model
        self.model = create_model(config)

        model_file = self.model_path / "model.pt"
        if model_file.exists():
            state_dict = torch.load(model_file, map_location="cpu")
            self.model.load_state_dict(state_dict)

        # Move to device and set dtype
        self.model = self.model.to(self.device).to(self.dtype)
        self.model.eval()

        # Load tokenizer
        self.tokenizer = create_tokenizer(
            vocab_size=config.total_vocab_size,
        )

        print(f"Loaded PyTorch model: {config.name} ({config.num_parameters()/1e9:.2f}B params)")

    def _load_vllm(self):
        """Load model with vLLM for high-throughput serving."""
        try:
            from vllm import LLM, SamplingParams
            self._vllm_module = __import__('vllm')
        except ImportError:
            raise ImportError("vLLM required for vllm backend. Install with: pip install vllm")

        # vLLM handles model loading internally
        self.vllm_engine = LLM(
            model=str(self.model_path),
            tokenizer=str(self.tokenizer_path),
            dtype=str(self.dtype).split('.')[-1],
            max_model_len=self.max_model_len,
            gpu_memory_utilization=0.9,
        )

        print(f"Loaded vLLM engine from {self.model_path}")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt
            config: Generation configuration

        Returns:
            GenerationResult with generated text and metadata
        """
        if config is None:
            config = GenerationConfig()

        start_time = time.perf_counter()

        if self.backend == "vllm":
            result = self._generate_vllm(prompt, config)
        else:
            result = self._generate_pytorch(prompt, config)

        result.generation_time_ms = (time.perf_counter() - start_time) * 1000
        return result

    def _generate_pytorch(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> GenerationResult:
        """Generate with PyTorch backend."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]

        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            do_sample=config.do_sample,
        )

        generated_ids = outputs[0][input_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Apply stop sequences
        finish_reason = "length"
        for seq in config.stop_sequences:
            if seq in text:
                text = text[:text.index(seq)]
                finish_reason = "stop"
                break

        return GenerationResult(
            text=text,
            tokens_generated=len(generated_ids),
            generation_time_ms=0,  # Filled by caller
            finish_reason=finish_reason,
            usage={
                "prompt_tokens": input_len,
                "completion_tokens": len(generated_ids),
                "total_tokens": input_len + len(generated_ids),
            },
        )

    def _generate_vllm(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> GenerationResult:
        """Generate with vLLM backend."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
            stop=config.stop_sequences or None,
        )

        outputs = self.vllm_engine.generate([prompt], sampling_params)
        output = outputs[0]

        text = output.outputs[0].text
        finish_reason = output.outputs[0].finish_reason

        return GenerationResult(
            text=text,
            tokens_generated=len(output.outputs[0].token_ids),
            generation_time_ms=0,
            finish_reason=finish_reason,
            usage={
                "prompt_tokens": len(output.prompt_token_ids),
                "completion_tokens": len(output.outputs[0].token_ids),
                "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
            },
        )

    async def generate_async(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Async generation (runs in thread pool for PyTorch)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(prompt, config),
        )

    async def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        """
        Stream generated tokens.

        Note: True streaming requires vLLM backend.
        PyTorch backend simulates streaming by chunking output.
        """
        if config is None:
            config = GenerationConfig()

        if self.backend == "vllm":
            async for chunk in self._stream_vllm(prompt, config):
                yield chunk
        else:
            # Simulate streaming for PyTorch
            result = await self.generate_async(prompt, config)
            words = result.text.split()
            for i, word in enumerate(words):
                yield word + (" " if i < len(words) - 1 else "")
                await asyncio.sleep(0.02)  # Simulate delay

    async def _stream_vllm(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> AsyncIterator[str]:
        """Stream with vLLM."""
        # vLLM streaming would go here
        # For now, fall back to non-streaming
        result = await self.generate_async(prompt, config)
        yield result.text

    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
    ) -> List[GenerationResult]:
        """Generate for multiple prompts in batch."""
        if config is None:
            config = GenerationConfig()

        if self.backend == "vllm":
            return self._generate_batch_vllm(prompts, config)
        else:
            # PyTorch: process sequentially (could be parallelized)
            return [self.generate(p, config) for p in prompts]

    def _generate_batch_vllm(
        self,
        prompts: List[str],
        config: GenerationConfig,
    ) -> List[GenerationResult]:
        """Batch generation with vLLM."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            stop=config.stop_sequences or None,
        )

        outputs = self.vllm_engine.generate(prompts, sampling_params)

        results = []
        for output in outputs:
            results.append(GenerationResult(
                text=output.outputs[0].text,
                tokens_generated=len(output.outputs[0].token_ids),
                generation_time_ms=0,
                finish_reason=output.outputs[0].finish_reason,
                usage={
                    "prompt_tokens": len(output.prompt_token_ids),
                    "completion_tokens": len(output.outputs[0].token_ids),
                    "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
                },
            ))

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = {
            "model_path": str(self.model_path),
            "backend": self.backend,
            "device": self.device,
            "dtype": str(self.dtype),
            "max_model_len": self.max_model_len,
        }

        if self.model is not None:
            info["parameters"] = sum(p.numel() for p in self.model.parameters())

        return info
