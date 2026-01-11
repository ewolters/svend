# Svend

**Tool-Augmented Reasoning System**

Train custom reasoning models from scratch with integrated tool use, verification loops, and production-ready serving.

```
╔═══════════════════════════════════════════════════════════════╗
║   ███████╗██╗   ██╗███████╗███╗   ██╗██████╗                  ║
║   ██╔════╝██║   ██║██╔════╝████╗  ██║██╔══██╗                 ║
║   ███████╗██║   ██║█████╗  ██╔██╗ ██║██║  ██║                 ║
║   ╚════██║╚██╗ ██╔╝██╔══╝  ██║╚██╗██║██║  ██║                 ║
║   ███████║ ╚████╔╝ ███████╗██║ ╚████║██████╔╝                 ║
║   ╚══════╝  ╚═══╝  ╚══════╝╚═╝  ╚═══╝╚═════╝                  ║
║                                              svend.ai         ║
╚═══════════════════════════════════════════════════════════════╝
```

## Overview

Svend is a complete framework for building reasoning models that:

- **Think step-by-step** with explicit reasoning chains
- **Use tools** (code execution, math solvers, logic provers)
- **Verify their work** with a separate critic model
- **Search reasoning paths** when problems are hard

## Architecture

```
                    ┌─────────────────┐
                    │  Orchestrator   │
                    │  (routes tasks) │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ┌────▼────┐         ┌─────▼─────┐        ┌─────▼─────┐
   │ Reasoner│         │   Tool    │        │  Verifier │
   │ (13B)   │◄───────►│  Executor │        │   (3B)    │
   └─────────┘         └───────────┘        └───────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
         ┌────▼────┐   ┌─────▼─────┐  ┌─────▼─────┐
         │  Code   │   │   Math    │  │   Logic   │
         │ Sandbox │   │  (SymPy)  │  │   (Z3)    │
         └─────────┘   └───────────┘  └───────────┘
```

## Models

| Model | Size | Purpose | Training Memory | Inference |
|-------|------|---------|-----------------|-----------|
| **svend-reasoner-13b** | 13B | Main reasoning model | ~70 GB | A100/H100 |
| **svend-reasoner-7b** | 7B | Faster reasoning | ~45 GB | RTX 4090+ |
| **svend-verifier-3b** | 3B | Checks reasoning | ~20 GB | RTX 3090+ |
| **svend-reasoner-1b** | 1B | Quick iteration | ~8 GB | Any GPU |

All models use modern architecture:
- **RoPE** positional embeddings (length generalization)
- **GQA** grouped-query attention (memory efficient)
- **SwiGLU** activation (better gradients)
- **8K context** default (32K with RoPE scaling)

## Quick Start

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/reasoning-lab.git
cd reasoning-lab
pip install -r requirements.txt
```

### 2. Train on A100 (Colab Pro+)

```bash
# Full 13B training (~48 hours)
python scripts/train_svend.py --model-size 13b --epochs 3

# Quick 7B training (~24 hours)
python scripts/train_svend.py --model-size 7b --epochs 3 --samples 100000

# Test the pipeline (10 minutes)
python scripts/train_svend.py --model-size 500m --samples 1000 --epochs 1 --no-wandb
```

### 3. Serve the Model

```bash
# PyTorch backend (simple)
python -m src.server.api --model-path checkpoints/svend-13b/final --port 8000

# vLLM backend (production)
python -m src.server.api --model-path checkpoints/svend-13b/final --backend vllm
```

### 4. Use the API

```python
import requests

# Simple completion
response = requests.post("http://localhost:8000/v1/completions", json={
    "prompt": "Solve: What is the derivative of x^3 + 2x?",
    "max_tokens": 512,
    "temperature": 0.7,
})
print(response.json()["choices"][0]["text"])

# Multi-step reasoning with tools
response = requests.post("http://localhost:8000/v1/reason", json={
    "question": "Is the number 1729 a sum of two cubes in two different ways?",
    "allow_tools": True,
    "verify": True,
})
print(response.json())
```

## Tools

Svend models can call external tools for verified computation:

| Tool | Description | Use Case |
|------|-------------|----------|
| `execute_python` | Sandboxed Python execution | Compute, verify, test |
| `symbolic_math` | SymPy symbolic computation | Algebra, calculus, solving |
| `logic_solver` | Z3 SMT solver | Formal logic, proofs, SAT |

The model learns to call tools with special tokens:

```
<|tool_call|><|tool_name|>symbolic_math<|tool_args|>{"operation": "solve", "expression": "x**2 - 4"}<|/tool_call|>
```

## Project Structure

```
reasoning-lab/
├── src/
│   ├── models/           # Transformer architecture
│   │   ├── config.py     # Model configurations
│   │   ├── layers.py     # Attention, FFN, normalization
│   │   └── transformer.py
│   │
│   ├── tools/            # Tool system
│   │   ├── registry.py   # Tool registration
│   │   ├── executor.py   # Execution management
│   │   ├── orchestrator.py # Reasoning coordination
│   │   ├── code_sandbox.py
│   │   └── math_engine.py
│   │
│   ├── training/         # Training infrastructure
│   │   │   ├── trainer.py
│   │   ├── distillation.py
│   │   └── distributed.py
│   │
│   ├── server/           # Production serving
│   │   ├── api.py        # FastAPI endpoints
│   │   └── inference.py  # Inference engine
│   │
│   ├── data/             # Data processing
│   │   ├── datasets.py
│   │   ├── tokenizer.py
│   │   └── synthetic.py
│   │
│   └── evaluation/       # Benchmarking
│       └── benchmarks.py
│
├── scripts/
│   ├── train_svend.py    # Main training script
│   └── evaluate_models.py
│
└── notebooks/
    └── train_on_colab.ipynb
```

## Training Data

Combined from high-quality open sources:

- **OpenMathInstruct-2** - Math reasoning traces
- **SlimOrca** - GPT-4 reasoning examples
- **CodeAlpaca** - Code understanding
- **CoT-Collection** - Chain-of-thought
- **CAMEL** - Science reasoning

Plus synthetic generation via Claude/GPT-4 API for custom domains.

## Hardware Requirements

| Task | GPU | Memory | Time |
|------|-----|--------|------|
| Train 13B | A100 80GB | 70 GB | ~48h |
| Train 7B | A100 40GB | 45 GB | ~24h |
| Train 3B | RTX 4090 | 20 GB | ~12h |
| Inference 13B | A100 | 26 GB | - |
| Inference 7B | RTX 4090 | 14 GB | - |

## Hosting & Pricing

Svend is designed for affordable hosting:

| Provider | GPU | Cost | Notes |
|----------|-----|------|-------|
| **RunPod** | A100 | ~$2/hr | Good for starting |
| **Lambda** | A100 | ~$1.50/hr | Cheapest |
| **Modal** | Various | Pay-per-second | Auto-scaling |

Target: **$3/1M tokens** (competitive with GPT-3.5)

## API Compatibility

The API is OpenAI-compatible:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="svend-13b",
    messages=[{"role": "user", "content": "Explain quantum entanglement"}],
)
```

## Roadmap

- [x] Custom transformer architecture
- [x] Tool calling infrastructure
- [x] Verification loops
- [x] Reasoning tree search
- [x] FastAPI server
- [ ] vLLM integration (in progress)
- [ ] INT4/INT8 quantization
- [ ] Multi-node training
- [ ] Web UI

## License

MIT

---

*svend.ai*
