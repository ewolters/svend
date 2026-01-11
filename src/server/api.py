"""
FastAPI Server for Svend Reasoning API.

Endpoints:
- POST /v1/completions - OpenAI-compatible completions
- POST /v1/chat/completions - Chat completions
- POST /v1/reason - Multi-step reasoning with tools
- WebSocket /v1/stream - Streaming generation
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
import time
import asyncio
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from .inference import InferenceEngine, GenerationConfig, GenerationResult


# =============================================================================
# Request/Response Models (OpenAI-compatible where possible)
# =============================================================================

class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request."""
    model: str = "svend-13b"
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[List[str]] = None
    stream: bool = False


class Message(BaseModel):
    """Chat message."""
    role: str  # "system", "user", "assistant"
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = "svend-13b"
    messages: List[Message]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[List[str]] = None
    stream: bool = False


class ReasoningRequest(BaseModel):
    """Request for multi-step reasoning."""
    question: str
    max_steps: int = 20
    allow_tools: bool = True
    verify: bool = True
    temperature: float = 0.7
    search_paths: int = 1  # >1 enables beam search


class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionChoice(BaseModel):
    """Single completion choice."""
    index: int
    text: str
    finish_reason: str


class CompletionResponse(BaseModel):
    """OpenAI-compatible completion response."""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo


class ChatChoice(BaseModel):
    """Chat completion choice."""
    index: int
    message: Message
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: UsageInfo


class ReasoningStep(BaseModel):
    """Single reasoning step."""
    step: int
    content: str
    tool_call: Optional[Dict[str, Any]] = None
    tool_result: Optional[str] = None
    verified: Optional[bool] = None


class ReasoningResponse(BaseModel):
    """Response from reasoning endpoint."""
    id: str
    question: str
    steps: List[ReasoningStep]
    answer: Optional[str]
    confidence: float
    verified: bool
    tool_calls_made: int
    generation_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model: str
    backend: str
    uptime_seconds: float


# =============================================================================
# API Server
# =============================================================================

class SvendServer:
    """
    Svend API Server.

    Provides REST API for model inference with:
    - OpenAI-compatible endpoints
    - Multi-step reasoning
    - Streaming support
    - Rate limiting (TODO)
    """

    def __init__(
        self,
        model_path: str,
        backend: str = "pytorch",
        api_key: Optional[str] = None,
    ):
        self.model_path = model_path
        self.backend = backend
        self.api_key = api_key
        self.start_time = time.time()

        # Will be initialized in setup
        self.engine: Optional[InferenceEngine] = None
        self.orchestrator = None

    def create_app(self) -> FastAPI:
        """Create FastAPI application."""
        app = FastAPI(
            title="Svend Reasoning API",
            description="Multi-step reasoning with tool augmentation",
            version="0.1.0",
        )

        # CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Startup/shutdown
        @app.on_event("startup")
        async def startup():
            await self._startup()

        @app.on_event("shutdown")
        async def shutdown():
            await self._shutdown()

        # Routes
        self._register_routes(app)

        return app

    async def _startup(self):
        """Initialize on startup."""
        print(f"Loading model from {self.model_path}...")
        self.engine = InferenceEngine(
            model_path=self.model_path,
            backend=self.backend,
        )
        print("Model loaded successfully!")

    async def _shutdown(self):
        """Cleanup on shutdown."""
        pass

    def _register_routes(self, app: FastAPI):
        """Register API routes."""

        # Health check
        @app.get("/health", response_model=HealthResponse)
        async def health():
            return HealthResponse(
                status="healthy",
                model=str(self.model_path),
                backend=self.backend,
                uptime_seconds=time.time() - self.start_time,
            )

        # Model info
        @app.get("/v1/models")
        async def list_models():
            info = self.engine.get_model_info() if self.engine else {}
            return {
                "object": "list",
                "data": [{
                    "id": "svend-13b",
                    "object": "model",
                    "owned_by": "svend",
                    **info,
                }],
            }

        # Completions (OpenAI-compatible)
        @app.post("/v1/completions", response_model=CompletionResponse)
        async def completions(request: CompletionRequest):
            if self.engine is None:
                raise HTTPException(status_code=503, detail="Model not loaded")

            config = GenerationConfig(
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop_sequences=request.stop or [],
            )

            result = await self.engine.generate_async(request.prompt, config)

            return CompletionResponse(
                id=f"cmpl-{uuid.uuid4().hex[:8]}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    CompletionChoice(
                        index=0,
                        text=result.text,
                        finish_reason=result.finish_reason,
                    )
                ],
                usage=UsageInfo(**result.usage),
            )

        # Chat completions (OpenAI-compatible)
        @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
        async def chat_completions(request: ChatCompletionRequest):
            if self.engine is None:
                raise HTTPException(status_code=503, detail="Model not loaded")

            # Format messages into prompt
            prompt = self._format_chat_prompt(request.messages)

            config = GenerationConfig(
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop_sequences=request.stop or [],
            )

            result = await self.engine.generate_async(prompt, config)

            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatChoice(
                        index=0,
                        message=Message(role="assistant", content=result.text),
                        finish_reason=result.finish_reason,
                    )
                ],
                usage=UsageInfo(**result.usage),
            )

        # Multi-step reasoning
        @app.post("/v1/reason", response_model=ReasoningResponse)
        async def reason(request: ReasoningRequest):
            if self.engine is None:
                raise HTTPException(status_code=503, detail="Model not loaded")

            start = time.perf_counter()

            # For now, use simple generation
            # Full orchestrator integration would go here
            prompt = f"""You are a reasoning assistant. Solve this problem step by step.

Question: {request.question}

Think through this carefully, showing each step of your reasoning. Use tools if needed.
Format your answer as:
<thinking>
<step>First step...</step>
<step>Second step...</step>
</thinking>
<answer>Your final answer</answer>"""

            config = GenerationConfig(
                max_new_tokens=1024,
                temperature=request.temperature,
            )

            result = await self.engine.generate_async(prompt, config)

            # Parse steps from response
            steps = self._parse_reasoning_steps(result.text)
            answer = self._extract_answer(result.text)

            elapsed = (time.perf_counter() - start) * 1000

            return ReasoningResponse(
                id=f"reason-{uuid.uuid4().hex[:8]}",
                question=request.question,
                steps=steps,
                answer=answer,
                confidence=0.8,  # Would come from verifier
                verified=False,  # Would come from verification loop
                tool_calls_made=0,
                generation_time_ms=elapsed,
            )

        # Streaming WebSocket
        @app.websocket("/v1/stream")
        async def websocket_stream(websocket: WebSocket):
            await websocket.accept()

            try:
                while True:
                    data = await websocket.receive_json()
                    prompt = data.get("prompt", "")

                    config = GenerationConfig(
                        max_new_tokens=data.get("max_tokens", 512),
                        temperature=data.get("temperature", 0.7),
                    )

                    async for chunk in self.engine.generate_stream(prompt, config):
                        await websocket.send_json({
                            "type": "chunk",
                            "text": chunk,
                        })

                    await websocket.send_json({
                        "type": "done",
                    })

            except WebSocketDisconnect:
                pass

    def _format_chat_prompt(self, messages: List[Message]) -> str:
        """Format chat messages into model prompt."""
        formatted = []
        for msg in messages:
            if msg.role == "system":
                formatted.append(f"<|system|>\n{msg.content}\n<|/system|>")
            elif msg.role == "user":
                formatted.append(f"<|user|>\n{msg.content}\n<|/user|>")
            elif msg.role == "assistant":
                formatted.append(f"<|assistant|>\n{msg.content}\n<|/assistant|>")

        formatted.append("<|assistant|>")
        return "\n".join(formatted)

    def _parse_reasoning_steps(self, text: str) -> List[ReasoningStep]:
        """Parse reasoning steps from generated text."""
        import re

        steps = []
        pattern = r"<\|?step\|?>(.*?)<\|?/step\|?>"

        for i, match in enumerate(re.finditer(pattern, text, re.DOTALL)):
            steps.append(ReasoningStep(
                step=i + 1,
                content=match.group(1).strip(),
            ))

        return steps

    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract final answer from generated text."""
        import re

        pattern = r"<\|?answer\|?>(.*?)<\|?/answer\|?>"
        match = re.search(pattern, text, re.DOTALL)

        if match:
            return match.group(1).strip()

        return None


def create_app(
    model_path: str,
    backend: str = "pytorch",
    api_key: Optional[str] = None,
) -> FastAPI:
    """Create and configure the API application."""
    server = SvendServer(
        model_path=model_path,
        backend=backend,
        api_key=api_key,
    )
    return server.create_app()


def run_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    backend: str = "pytorch",
):
    """Run the Svend API server."""
    app = create_app(model_path, backend)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Svend API server")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend", type=str, default="pytorch", choices=["pytorch", "vllm"])

    args = parser.parse_args()

    run_server(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        backend=args.backend,
    )
