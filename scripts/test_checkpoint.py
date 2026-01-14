"""Quick test script for trained checkpoint."""

import sys
sys.path.insert(0, 'C:/Users/ewolt/juniperware/reasoning-lab')

import torch
from transformers import AutoTokenizer
from src.models.config import get_config
from src.models.transformer import ReasoningTransformer

# Config
CHECKPOINT_PATH = "C:/Users/ewolt/juniperware/reasoning-lab/checkpoints/final-language-model.pt"
MODEL_SIZE = "500m"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

print("Creating model...")
model_config = get_config(MODEL_SIZE)
model_config.vocab_size = tokenizer.vocab_size

model = ReasoningTransformer(model_config)

print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

# Check if CUDA available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model = model.to(device)
model.eval()

params = sum(p.numel() for p in model.parameters())
print(f"Model loaded: {params/1e6:.1f}M parameters")
print(f"Checkpoint info: {checkpoint.get('global_step', 'N/A')} steps, epoch {checkpoint.get('epoch', 'N/A')}")

# Generation function
@torch.no_grad()
def generate(prompt, max_tokens=150, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        top_p=0.9,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test prompts
print("\n" + "="*60)
print("TESTING GENERATION")
print("="*60)

test_prompts = [
    "Question: What is 15% of 200?\n\nAnswer:",
    "Question: If a train travels at 60 mph for 2.5 hours, how far does it go?\n\nAnswer:",
    "Question: A shirt costs $25. With a 20% discount, what is the final price?\n\nAnswer:",
    "Question: If 3x + 7 = 22, what is x?\n\nAnswer:",
]

for prompt in test_prompts:
    print(f"\n{'-'*40}")
    print(f"PROMPT: {prompt.split('Question: ')[1].split('?')[0]}?")
    response = generate(prompt, max_tokens=100, temperature=0.3)
    # Extract just the answer part
    answer_part = response[len(prompt):].strip()
    print(f"RESPONSE: {answer_part[:200]}")

print("\n" + "="*60)
print("DONE")
print("="*60)
