# Svend - Project Context

## What is this?

Svend is a tool-augmented reasoning system. A small language model trained from scratch to:
- Reason step-by-step through math, logic, chemistry, physics
- Call external tools (Python sandbox, SymPy, Z3) for verified computation
- Verify its own work with a separate critic model

Target: $5/month subscription, launching May 2026.

## Project Structure

```
reasoning-lab/
├── src/
│   ├── models/          # Transformer architecture (RoPE, GQA, SwiGLU)
│   │   ├── config.py    # Model size configs (1B, 3B, 7B, 13B)
│   │   ├── layers.py    # Attention, FFN, RMSNorm
│   │   └── transformer.py
│   │
│   ├── tools/           # Tool calling system
│   │   ├── registry.py  # Tool registration
│   │   ├── executor.py  # Execution management
│   │   ├── orchestrator.py  # Reasoning coordination
│   │   ├── code_sandbox.py  # Sandboxed Python
│   │   ├── math_engine.py   # SymPy integration
│   │   ├── chemistry.py     # Chemistry tools
│   │   └── physics.py       # Physics tools
│   │
│   ├── training/        # Training infrastructure
│   │   ├── trainer.py   # Main training loop
│   │   ├── distillation.py  # Knowledge distillation
│   │   └── distributed.py   # Multi-GPU
│   │
│   ├── pipeline/        # Training pipeline orchestration
│   │   ├── runner.py    # Pipeline execution
│   │   ├── config.py    # Pipeline configs
│   │   ├── validation.py
│   │   └── checkpoints.py
│   │
│   ├── evaluation/      # Benchmarking
│   │   ├── harness.py   # Eval harness
│   │   ├── benchmarks.py
│   │   └── metrics.py
│   │
│   ├── safety/          # Safety infrastructure
│   │   ├── classifier.py  # Safety classifier
│   │   ├── filters.py     # Content filters
│   │   ├── rules.py       # Rule-based checks
│   │   ├── gate.py        # Safety gate
│   │   └── training.py    # Safety model training
│   │
│   ├── server/          # Production serving
│   │   ├── api.py       # FastAPI endpoints (OpenAI-compatible)
│   │   └── inference.py # Inference engine
│   │
│   └── data/            # Data processing
│       ├── datasets.py  # Dataset loaders
│       ├── tokenizer.py # BPE tokenizer
│       └── synthetic.py # Synthetic data generation
│
├── scripts/
│   ├── train_svend.py   # Main training entry point
│   ├── generate_tool_data.py
│   ├── evaluate_models.py
│   └── test_pipeline.py
│
├── notebooks/
│   ├── train_on_colab.ipynb
│   └── train_reasoner_colab.ipynb
│
├── site/                # Landing page (svend.ai)
│   └── index.html       # Static page with email signup (Google Forms)
│
└── docs/
    ├── TRAINING_STRATEGY.md
    ├── SVEND_PRODUCTION_PLAN.md
    └── PIPELINE_GUIDE.md
```

## Key Architecture Decisions

- **Model**: Custom transformer with RoPE, GQA, SwiGLU (modern architecture)
- **Sizes**: 1B (dev), 3B (verifier), 7B (fast), 13B (main)
- **Context**: 8K default, 32K with RoPE scaling
- **Tools**: Special tokens for tool calls: `<|tool_call|>...<|/tool_call|>`
- **Framework**: PyTorch, with optional vLLM for production serving
- **API**: OpenAI-compatible endpoints

## Training

Runs on Google Colab Pro+ (A100) or equivalent:
```bash
# Test pipeline
python scripts/train_svend.py --model-size 500m --samples 1000 --epochs 1 --no-wandb

# Full training
python scripts/train_svend.py --model-size 13b --epochs 3
```

## Current Status

- Core infrastructure built
- Landing page at site/index.html with email capture
- Training pipeline ready for iteration
- Target launch: May 2026

## Working Conventions

- Python 3.10+
- Type hints encouraged
- Tests in scripts/test_*.py
- Configs in src/*/config.py or as dataclasses
