# Svend Training Strategy

## Overview

Train a 13B reasoning model from scratch with tool-calling capabilities, targeting $3/1M token hosting.

## Resources

- **Training**: A100 80GB (Colab Pro+)
- **Local Testing**: ~6GB laptop GPU
- **Synthetic Data**: Claude API ($200 credit)
- **Hosting Target**: RunPod/Lambda/Modal

---

## Phase 1: Pipeline Validation (Local - 6GB GPU)

**Goal**: Verify everything works before burning A100 hours.

```bash
# Run on laptop - tests full pipeline with tiny model
python scripts/test_pipeline.py
```

This runs:
- 50M parameter model (fits in 6GB)
- 100 synthetic examples
- 1 epoch
- Validates: data loading, tokenization, forward/backward pass, checkpointing

---

## Phase 2: Small Scale Training (A100 - 2-4 hours)

**Goal**: Validate architecture scales, tune hyperparameters.

```bash
python scripts/train_svend.py --model-size 1b --samples 10000 --epochs 1
```

Checkpoints to Google Drive for persistence across Colab sessions.

---

## Phase 3: Tool-Calling Data Generation (Claude API)

**Goal**: Generate high-quality tool-use training examples.

Budget: ~$50-100 of API credits for 10-20K examples.

```bash
python scripts/generate_tool_data.py --num-examples 10000 --output data/tool_traces.jsonl
```

Data format:
```json
{
  "question": "What is the derivative of x^3 + 2x?",
  "reasoning": [
    {"step": 1, "content": "I need to find d/dx of x^3 + 2x"},
    {"step": 2, "content": "Using power rule...", "tool_call": {"name": "symbolic_math", "args": {"op": "diff", "expr": "x**3 + 2*x"}}}
  ],
  "tool_result": "3*x**2 + 2",
  "answer": "3x² + 2"
}
```

---

## Phase 4: Full Training (A100 - 24-48 hours)

### 4a: Base Reasoning (no tools)
```bash
python scripts/train_svend.py --model-size 7b --epochs 2 --output-dir checkpoints/svend-7b-base
```

Data: OpenMathInstruct-2 + SlimOrca + CoT-Collection (~150K examples)

### 4b: Tool Integration
```bash
python scripts/train_svend.py --model-size 7b --epochs 1 \
  --resume checkpoints/svend-7b-base/final \
  --data-mix tool_heavy \
  --output-dir checkpoints/svend-7b-tools
```

Data: Generated tool traces + filtered examples with tool use

### 4c: Verifier Training (parallel)
```bash
python scripts/train_svend.py --model-size 3b-verifier --epochs 2
```

---

## Phase 5: Evaluation & Iteration

Benchmarks:
- GSM8K (math reasoning)
- MATH (harder math)
- HumanEval (code)
- Custom tool-use eval

```bash
python scripts/evaluate_models.py --model checkpoints/svend-7b-tools/final --benchmarks gsm8k,math
```

---

## Colab Session Strategy

Colab Pro+ has 24h max sessions. Strategy:

1. **Checkpoint every 500 steps** to Google Drive
2. **Use `--resume` flag** to continue from last checkpoint
3. **Track progress in WandB** - survives session restarts
4. **Save model config with checkpoint** for reproducibility

```bash
# Mount drive first
from google.colab import drive
drive.mount('/content/drive')

# Train with drive output
python scripts/train_svend.py --model-size 7b \
  --output-dir /content/drive/MyDrive/svend-checkpoints/7b \
  --save-steps 500
```

---

## Data Mix Ratios

| Phase | Reasoning | Tool-Use | Code | General |
|-------|-----------|----------|------|---------|
| Base  | 60%       | 0%       | 20%  | 20%     |
| Tools | 30%       | 50%      | 15%  | 5%      |

---

## Hyperparameters (7B baseline)

```yaml
learning_rate: 1e-4
weight_decay: 0.1
warmup_ratio: 0.03
batch_size: 4
grad_accum: 8
effective_batch: 32
max_seq_length: 4096
precision: bf16
optimizer: AdamW
scheduler: cosine
```

---

## Success Criteria

Before deploying:
- [ ] GSM8K accuracy > 50%
- [ ] Tool calls execute correctly > 90% of time
- [ ] Verifier catches obvious errors > 80%
- [ ] Inference latency < 2s for typical query
- [ ] Memory fits on A100 40GB for inference

---

## Cost Estimates

| Item | Cost |
|------|------|
| A100 training (48h) | ~$75-100 |
| Claude API (synth data) | ~$50-100 |
| Initial hosting test | ~$20 |
| **Total to MVP** | **~$150-220** |

---

## Next Steps

1. Run `test_pipeline.py` on laptop
2. Generate 1K tool examples with Claude
3. 1B validation run on Colab
4. Full 7B training
5. Deploy to RunPod for testing
