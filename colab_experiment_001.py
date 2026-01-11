# ============================================================
# Experiment 001: Reasoning Fine-tune
# ============================================================
# Run this in Google Colab with A100 GPU
#
# 1. Create new Colab notebook
# 2. Runtime > Change runtime type > A100 GPU
# 3. Copy/paste this entire file into a cell
# 4. Run!
# ============================================================

#@title 1. Install Dependencies
!pip install -q torch transformers datasets accelerate peft bitsandbytes trl wandb

#@title 2. Imports and Setup
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Check GPU
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

#@title 3. Configuration
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # Start small
OUTPUT_DIR = "./reasoning-model-v1"

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# LoRA config
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

#@title 4. Load Model and Tokenizer
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Print trainable params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

#@title 5. Load Dataset
# Using SlimOrca - high quality reasoning traces from GPT-4
print("Loading dataset...")
dataset = load_dataset("Open-Orca/SlimOrca", split="train")

# Sample for faster iteration (remove this line for full training)
dataset = dataset.shuffle(seed=42).select(range(10000))
print(f"Training samples: {len(dataset)}")

# Format for chat template
def format_chat(example):
    """Convert to Qwen chat format."""
    messages = []

    # System message if present
    if example.get("system_prompt"):
        messages.append({"role": "system", "content": example["system_prompt"]})

    # User question
    messages.append({"role": "user", "content": example["question"]})

    # Assistant response (the reasoning trace)
    messages.append({"role": "assistant", "content": example["response"]})

    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

dataset = dataset.map(format_chat, remove_columns=dataset.column_names)
print("Sample:", dataset[0]["text"][:500])

#@title 6. Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",  # Set to "wandb" if you want tracking
)

#@title 7. Train!
print("Starting training...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=True,  # Efficient packing of sequences
)

trainer.train()

#@title 8. Save Model
print("Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")

#@title 9. Test It!
print("\n" + "="*50)
print("TESTING THE MODEL")
print("="*50 + "\n")

# Reload for inference
model.eval()

def ask(question, max_tokens=512):
    """Ask the model a question."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant that thinks step by step."},
        {"role": "user", "content": question},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

# Test questions
test_questions = [
    "If a train travels at 60 mph for 2.5 hours, how far does it go? Explain your reasoning.",
    "Debug this Python code: `def add(a, b): return a - b` - the function should add two numbers.",
    "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?",
]

for q in test_questions:
    print(f"Q: {q}\n")
    print(f"A: {ask(q)}\n")
    print("-" * 50 + "\n")

#@title 10. Download Model (Optional)
# Zip and download to your machine
!zip -r reasoning-model-v1.zip {OUTPUT_DIR}
from google.colab import files
files.download("reasoning-model-v1.zip")
