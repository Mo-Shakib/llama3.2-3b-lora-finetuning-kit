"""
This script uses Unsloth for 2-5x faster training with 70% less memory.
Your dataset should be in JSONL format with "messages" structure.
"""

import os
# Disable torch dynamo to avoid compilation errors with some models
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import json
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ============================================================================
# LOAD CONFIG
# ============================================================================

with open("config.json") as f:
    config = json.load(f)

print("="*60)
print("TUNEKIT - UNSLOTH LORA TRAINING")
print("="*60)
print(f"Model: {config['base_model']}")
print(f"Data:  {config['data_path']}")
print(f"Output: {config['output_dir']}")
print("="*60)

# ============================================================================
# LOAD MODEL WITH UNSLOTH (4-BIT QUANTIZATION)
# ============================================================================

print("\nLoading model with Unsloth...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config["base_model"],
    max_seq_length=config.get("max_seq_length", 2048),
    dtype=None,  # Auto-detect
    load_in_4bit=True,  # 4-bit quantization for memory efficiency
)

# ============================================================================
# CONFIGURE LORA
# ============================================================================

lora_config = config.get("lora_config", {})

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_config.get("r", 16),
    lora_alpha=lora_config.get("lora_alpha", 16),
    lora_dropout=lora_config.get("lora_dropout", 0),
    target_modules=lora_config.get("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]),
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
    random_state=42,
    use_rslora=False,  # Rank-stabilized LoRA
    loftq_config=None,
)

print("\nLoRA Configuration:")
print(f"  Rank (r): {lora_config.get('r', 16)}")
print(f"  Alpha: {lora_config.get('lora_alpha', 16)}")
print(f"  Dropout: {lora_config.get('lora_dropout', 0)}")

# ============================================================================
# LOAD AND PREPARE DATASET
# ============================================================================

print("\nLoading dataset...")

dataset = load_dataset("json", data_files=config["data_path"], split="train")
print(f"Loaded {len(dataset)} examples")

# Verify format
if len(dataset) == 0:
    raise ValueError("Dataset is empty!")
if "messages" not in dataset[0]:
    raise ValueError('Dataset must have "messages" field.')

# Format using chat template
def format_chat(example):
    """Apply chat template to messages."""
    messages = example["messages"]
    
    # Use tokenizer's chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": text}

print("Formatting with chat template...")
dataset = dataset.map(format_chat, remove_columns=dataset.column_names)

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

training_args_config = config.get("training_args", {})

training_args = TrainingArguments(
    output_dir=config["output_dir"],
    
    # Training params
    num_train_epochs=training_args_config.get("num_train_epochs", 3),
    per_device_train_batch_size=training_args_config.get("per_device_train_batch_size", 2),
    gradient_accumulation_steps=training_args_config.get("gradient_accumulation_steps", 4),
    
    # Optimizer
    learning_rate=training_args_config.get("learning_rate", 2e-4),
    weight_decay=training_args_config.get("weight_decay", 0.01),
    warmup_steps=training_args_config.get("warmup_steps", 5),
    
    # Precision
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    
    # Logging
    logging_steps=training_args_config.get("logging_steps", 10),
    
    # Saving
    save_strategy="epoch",
    save_total_limit=2,
    
    # Optimization
    optim="adamw_8bit",  # 8-bit Adam for memory efficiency
    lr_scheduler_type="linear",
    seed=42,
    
    # Disable unused features
    push_to_hub=False,
    report_to="none",
)

# ============================================================================
# TRAIN
# ============================================================================

print("\nInitializing trainer...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=config.get("max_seq_length", 2048),
    packing=False,  # Can set to True for short sequences
    args=training_args,
)

# Show GPU stats before training
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"\nGPU: {gpu_stats.name}")
print(f"Memory: {start_gpu_memory}GB / {max_memory}GB")

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60 + "\n")

trainer_stats = trainer.train()

# Show final GPU stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_pct = round(used_memory / max_memory * 100, 2)
print(f"\nPeak GPU memory: {used_memory}GB ({used_memory_pct}% of {max_memory}GB)")

# ============================================================================
# SAVE MODEL
# ============================================================================

print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

# Save LoRA adapter
lora_path = config["output_dir"]
model.save_pretrained(lora_path)
tokenizer.save_pretrained(lora_path)
print(f"\n✓ LoRA adapter saved to: {lora_path}")

# Optional: Save merged model (full weights)
# print("\nMerging LoRA with base model...")
# model.save_pretrained_merged(lora_path + "_merged", tokenizer, save_method="merged_16bit")
# print(f"✓ Merged model saved to: {lora_path}_merged")

# ============================================================================
# EXPORT OPTIONS (UNCOMMENT AS NEEDED)
# ============================================================================

# # Save as GGUF for llama.cpp / Ollama
# print("\nExporting to GGUF format...")
# model.save_pretrained_gguf(lora_path + "_gguf", tokenizer, quantization_method="q4_k_m")
# print(f"✓ GGUF model saved to: {lora_path}_gguf")

# # Save as 4-bit quantized
# print("\nExporting 4-bit quantized model...")
# model.save_pretrained_merged(lora_path + "_4bit", tokenizer, save_method="merged_4bit_forced")
# print(f"✓ 4-bit model saved to: {lora_path}_4bit")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"\nNext steps:")
print(f"  1. Test: python eval.py")
print(f"  2. Use LoRA adapter: {lora_path}")
print("="*60)