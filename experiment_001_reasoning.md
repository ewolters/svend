# Experiment 001: Reasoning Fine-tune

**Date:** 2025-01-10
**Objective:** Fine-tune a 1-3B model on chain-of-thought reasoning data

## Setup

- **Platform:** Google Colab (80GB GPU - A100)
- **Base Model:** Qwen2.5-1.5B-Instruct (start small, iterate fast)
- **Method:** QLoRA (4-bit quantization + LoRA adapters)
- **Training Time:** ~1-2 hours

## Why Qwen2.5-1.5B?

1. Strong baseline reasoning for size
2. Apache 2.0 license (fully open)
3. Good tokenizer for code + English
4. Can scale up to 3B/7B with same approach

## Dataset Strategy

Start with existing high-quality reasoning data:

1. **OpenOrca-SlimOrca** - GPT-4 reasoning traces
2. **MetaMathQA** - Mathematical reasoning
3. **Code-Feedback** - Code debugging with explanations

## Training Config

```python
# QLoRA settings
lora_r = 64
lora_alpha = 128
lora_dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training
epochs = 1  # Start with 1, evaluate, iterate
batch_size = 4
gradient_accumulation = 4  # Effective batch = 16
learning_rate = 2e-4
warmup_ratio = 0.03
```

## Success Criteria

After training, test on held-out problems:
- [ ] Can it explain steps for a logic puzzle?
- [ ] Can it debug a simple Python error?
- [ ] Does it show reasoning before answering?

## Next Steps

If this works:
1. Scale to Qwen2.5-3B
2. Curate custom reasoning dataset
3. Add self-correction training
4. Experiment with DPO for preference tuning
