"""
TuneKit Export Script - Export Model to Various Formats
=========================================================
Run: python export.py

Export your fine-tuned model to:
- GGUF (for llama.cpp, Ollama, LM Studio)
- 16-bit merged (for HuggingFace)
- 4-bit quantized (for low-memory inference)
"""

import json
import os
from unsloth import FastLanguageModel

# Load config
with open("config.json") as f:
    config = json.load(f)

print("="*60)
print("TUNEKIT - MODEL EXPORT")
print("="*60)

# Load the trained LoRA model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config["output_dir"],
    max_seq_length=config.get("max_seq_length", 2048),
    dtype=None,
    load_in_4bit=True,
)

base_output = config["output_dir"]

# ============================================================================
# EXPORT OPTIONS
# ============================================================================

print("\nSelect export format:")
print("  1. GGUF (q4_k_m) - Best for llama.cpp, Ollama, LM Studio")
print("  2. GGUF (q8_0)   - Higher quality, larger file")
print("  3. 16-bit merged - Full precision, HuggingFace compatible")
print("  4. 4-bit merged  - Quantized, memory efficient")
print("  5. All formats")
print("  0. Exit")

choice = input("\nEnter choice (0-5): ").strip()

if choice == "0":
    print("Exiting.")
    exit()

# GGUF q4_k_m
if choice in ["1", "5"]:
    print("\nExporting GGUF (q4_k_m)...")
    output_path = base_output + "_gguf_q4"
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained_gguf(output_path, tokenizer, quantization_method="q4_k_m")
    print(f"✓ Saved to: {output_path}")

# GGUF q8_0
if choice in ["2", "5"]:
    print("\nExporting GGUF (q8_0)...")
    output_path = base_output + "_gguf_q8"
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained_gguf(output_path, tokenizer, quantization_method="q8_0")
    print(f"✓ Saved to: {output_path}")

# 16-bit merged
if choice in ["3", "5"]:
    print("\nExporting 16-bit merged model...")
    output_path = base_output + "_merged_16bit"
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained_merged(output_path, tokenizer, save_method="merged_16bit")
    print(f"✓ Saved to: {output_path}")

# 4-bit merged
if choice in ["4", "5"]:
    print("\nExporting 4-bit merged model...")
    output_path = base_output + "_merged_4bit"
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained_merged(output_path, tokenizer, save_method="merged_4bit_forced")
    print(f"✓ Saved to: {output_path}")

print("\n" + "="*60)
print("EXPORT COMPLETE!")
print("="*60)
print("\nUsage instructions:")
print("  - GGUF: Use with llama.cpp, Ollama, or LM Studio")
print("  - Merged: Use with HuggingFace transformers")
print("="*60)
