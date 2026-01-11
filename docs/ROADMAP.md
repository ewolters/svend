# Svend Roadmap

## Current State (January 2025)

### Infrastructure Complete

| Component | Status | Location |
|-----------|--------|----------|
| **Transformer Architecture** | Done | `src/models/` |
| - RoPE, GQA, SwiGLU, RMSNorm | Done | `layers.py`, `transformer.py` |
| - Model configs (1B-13B) | Done | `config.py` |
| **Tool System** | Done | `src/tools/` |
| - Symbolic math (10 ops) | Done | `math_engine.py` |
| - Physics (10 ops) | Done | `physics.py` |
| - Chemistry (10 ops) | Done | `chemistry.py` |
| - Logic solver (Z3) | Done | `math_engine.py` |
| - Code sandbox | Done | `code_sandbox.py` |
| - Tool orchestrator | Done | `orchestrator.py`, `specialists.py` |
| - Meta-cognitive tokens | Done | `registry.py` |
| **Training Pipeline** | Done | `src/training/`, `src/pipeline/` |
| - Trainer with checkpointing | Done | `trainer.py` |
| - Distillation support | Done | `distillation.py` |
| - Pipeline runner | Done | `runner.py` |
| **Data Pipeline** | Done | `src/data/` |
| - Dataset loaders (HF) | Done | `datasets.py` |
| - Tokenizer | Done | `tokenizer.py` |
| - Tool trace formatter | Done | `datasets.py` |
| - Failure/clarification formatter | Done | `datasets.py` |
| **Evaluation** | Done | `src/evaluation/` |
| - Benchmark harness | Done | `harness.py`, `benchmarks.py` |
| - Metrics | Done | `metrics.py` |
| **Safety** | Done | `src/safety/` |
| - Classifier | Done | `classifier.py` |
| - Filters & rules | Done | `filters.py`, `rules.py` |
| - Safety gate | Done | `gate.py` |
| **Server** | Done | `src/server/` |
| - FastAPI (OpenAI-compatible) | Done | `api.py` |
| - Inference engine | Done | `inference.py` |
| **Data Generation** | Done | `scripts/` |
| - Tool trace generator | Done | `generate_tool_data.py` |
| - Failure recognition traces | Done | `generate_tool_data.py` |
| - Clarification request traces | Done | `generate_tool_data.py` |
| **Testing** | Done | `tests/` |
| - Tool unit tests (29 tests) | Done | `test_tools.py` |
| - Smoke test script | Done | `scripts/smoke_test.py` |

### What's Actually Missing

1. **Training data** - Need to generate it
2. **Trained models** - Need to train them
3. **End-to-end testing** - Need to validate the full pipeline works
4. **Deployment config** - Modal/RunPod setup

---

## Immediate Next Steps (No Colab Required)

### 1. Test Suite
Create proper tests for the tool system to ensure everything works before training.

```
tests/
    test_tools.py         # Unit tests for all tools
    test_pipeline.py      # Integration tests
    test_safety.py        # Safety system tests
```

### 2. Requirements Validation
Ensure all dependencies are documented and installable.

```
requirements.txt          # Core deps
requirements-dev.txt      # Testing deps
requirements-colab.txt    # Colab-specific deps
```

### 3. Local Smoke Test Script
A script that validates the entire pipeline end-to-end with tiny models.

```python
# scripts/smoke_test.py
# - Load 1M parameter model
# - Process one example through full pipeline
# - Execute tools
# - Verify output format
```

### 4. Configuration Consolidation
Single source of truth for all configs.

```
configs/
    training/
        dev.yaml          # Tiny model for testing
        7b.yaml           # Production 7B
        13b.yaml          # Production 13B
    serving/
        local.yaml        # Local inference
        modal.yaml        # Modal deployment
        runpod.yaml       # RunPod deployment
```

---

## Pre-Training Checklist

Before spending money on Colab/API calls:

- [ ] `py scripts/test_pipeline.py` passes
- [ ] All tools execute correctly (unit tests)
- [ ] Tokenizer handles tool tokens
- [ ] Dataset loaders work with HF datasets
- [ ] Safety classifier loads and runs
- [ ] Checkpoint save/load works
- [ ] WandB integration works (or can be disabled)

---

## Training Plan (When Ready)

### Phase 1: Generate Data (~$60)
```bash
py scripts/generate_tool_data.py --num-examples 12000 --output data/tool_traces.jsonl
```

### Phase 2: Safety Classifier (4h Colab)
- Train lightweight safety classifier
- Validate on red-team examples

### Phase 3: Base Reasoning (24h Colab)
- Fine-tune Qwen2.5-14B or Mistral-7B
- General reasoning without tools

### Phase 4: Tool Integration (18h Colab)
- Continue from Phase 3
- Add tool-calling capability

### Phase 5: Verifier (24h Colab)
- Train 3B verifier model
- Learns to check reasoning chains

---

## Infrastructure Gaps to Fill

### Priority 1: Testing
| Task | Effort | Value |
|------|--------|-------|
| Unit tests for tools | 2h | High - catches bugs before training |
| Integration test | 1h | High - validates full pipeline |
| Smoke test script | 1h | High - quick validation |

### Priority 2: Configuration
| Task | Effort | Value |
|------|--------|-------|
| YAML configs for training | 1h | Medium - cleaner than CLI args |
| Environment setup docs | 30m | Medium - reproducibility |

### Priority 3: Documentation
| Task | Effort | Value |
|------|--------|-------|
| README update | 30m | Low - already have CLAUDE.md |
| Quick start guide | 1h | Medium - helps new contributors |

### Priority 4: Deployment
| Task | Effort | Value |
|------|--------|-------|
| Modal deployment script | 2h | Defer - need trained model first |
| Docker setup | 2h | Defer - need trained model first |

---

## What We Can Build Tonight (Low Risk)

1. **Test suite** - Pure Python, no dependencies on external services
2. **Smoke test script** - Validates infrastructure without training
3. **Requirements files** - Organize dependencies
4. **Config consolidation** - YAML files for different scenarios

---

## Decision Points

### Model Strategy
**Question**: Fine-tune existing model vs train from scratch?

| Option | Pros | Cons |
|--------|------|------|
| Fine-tune Qwen2.5-14B | Strong base, proven | Less control |
| Fine-tune Mistral-7B | Apache 2.0, efficient | Smaller capacity |
| Train from scratch | Full control | Much more compute |

**Current recommendation**: Fine-tune, switch to from-scratch if licensing becomes an issue.

### Hosting Strategy
**Question**: Where to deploy?

| Provider | Cost | Scaling |
|----------|------|---------|
| Modal | Pay-per-second | Auto |
| RunPod | Per-hour | Manual |
| Lambda Labs | Reserved | None |

**Current recommendation**: Modal for initial launch (simplest), migrate to RunPod if traffic justifies.

---

## Success Metrics

Before launch:
- [ ] GSM8K > 50% accuracy
- [ ] Tool calls correct > 90%
- [ ] Safety refusal correct > 95%
- [ ] Latency < 5s p90
- [ ] Fits on A100-40GB

---

## Notes

- Landing page live at svend.ai (Google Forms signup)
- Target launch: May 2026
- Budget: $500 over 30 days of training
