"""
Run: python eval.py
"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import json
import torch
from unsloth import FastLanguageModel

# Load config
with open("config.json") as f:
    config = json.load(f)

print("="*60)
print("TUNEKIT - MODEL EVALUATION")
print("="*60)
print(f"Loading LoRA adapter from: {config['output_dir']}")
print("="*60)

# Load model with LoRA adapter
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config["output_dir"],  # LoRA adapter path
    max_seq_length=config.get("max_seq_length", 2048),
    dtype=None,
    load_in_4bit=True,
)

# Enable faster inference
FastLanguageModel.for_inference(model)

print("\nModel loaded successfully!\n")

# ============================================================================
# LOAD TEST SAMPLES FROM TRAINING DATA
# ============================================================================

test_samples = []
data_path = config.get("data_path", "./data.jsonl")

if os.path.exists(data_path):
    with open(data_path, "r") as f:
        lines = f.readlines()
        # Take 5 samples from different parts of the dataset
        indices = [0, len(lines)//4, len(lines)//2, 3*len(lines)//4, -1]
        for idx in indices:
            try:
                example = json.loads(lines[idx])
                if "messages" in example:
                    test_samples.append(example["messages"])
            except:
                pass

# Fallback if no data found
if not test_samples:
    test_samples = [
        [{"role": "user", "content": "Hello!"}],
    ]

print("="*60)
print("TESTING MODEL WITH YOUR DATA")
print("="*60)

correct = 0
total = 0

for i, messages in enumerate(test_samples[:5], 1):
    # Get the user input (last user message before assistant)
    user_msg = None
    expected = None

    for j, msg in enumerate(messages):
        if msg["role"] == "user":
            user_msg = msg["content"]
        elif msg["role"] == "assistant" and user_msg:
            expected = msg["content"]
            break

    if not user_msg:
        continue

    total += 1
    print(f"\n--- Test {i} ---")
    print(f"Input: {user_msg[:100]}{'...' if len(user_msg) > 100 else ''}")
    if expected:
        print(f"Expected: {expected[:50]}{'...' if len(expected) > 50 else ''}")

    # Build messages for generation (without the expected response)
    gen_messages = []
    for msg in messages:
        if msg["role"] == "assistant":
            break
        gen_messages.append(msg)

    # Apply chat template with generation prompt
    formatted = tokenizer.apply_chat_template(
        gen_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize
    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")

    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        use_cache=True,
        temperature=0.1,  # Low temperature for more deterministic output
    )

    # Decode only the new tokens
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    print(f"Got: {response[:50]}{'...' if len(response) > 50 else ''}")

    # Check if correct (simple match)
    if expected and expected.lower().strip() in response.lower():
        print("✓ MATCH")
        correct += 1
    elif expected:
        print("✗ MISMATCH")

    print("-"*60)

if total > 0:
    print(f"\nAccuracy: {correct}/{total} ({100*correct/total:.1f}%)")

# ============================================================================
# INTERACTIVE MODE
# ============================================================================

print("\n" + "="*60)
print("INTERACTIVE MODE")
print("="*60)
print("Enter messages to chat with your model (type 'quit' to exit)")
print("="*60)

while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
    
    if not user_input:
        continue
    
    messages = [{"role": "user", "content": user_input}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs, max_new_tokens=300, use_cache=True, temperature=0.7
    )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    print(f"\nAssistant: {response.strip()}")
