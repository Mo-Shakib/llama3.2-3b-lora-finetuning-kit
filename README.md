# Fine-Tuning Training Package

## Overview

This training uses **Unsloth** for fast, memory-efficient LoRA fine-tuning.

| Property | Value |
|----------|-------|
| Base Model | `meta-llama/Llama-3.2-3B-Instruct` |
| Training Method | LoRA with Unsloth (2-5x faster) |
| Data Format | JSONL with "messages" structure |
| Memory Usage | ~70% less than standard training |
| Generated | 2026-01-19 18:17 |

## Why Unsloth?

Unsloth provides significant advantages:
- ⚡ **2-5x faster training** with optimized kernels
- 💾 **70% less memory** usage
- 🔧 **Native 4-bit quantization** for efficiency
- 📦 **Easy export** to GGUF, 16-bit, 4-bit formats
- 🎯 **Optimized gradient checkpointing**

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

Your JSONL file should have this format:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### 3. Train

```bash
python train.py
```

### 4. Evaluate

```bash
python eval.py
```

### 5. Export (Optional)

```bash
python export.py
```

Export to GGUF for llama.cpp/Ollama, or merged formats for HuggingFace.

## Files Included

| File | Description |
|------|-------------|
| `data.jsonl` | Your training dataset (included) |
| `config.json` | Training configuration |
| `train.py` | Unsloth LoRA training script |
| `eval.py` | Evaluation + interactive testing |
| `export.py` | Export to GGUF, 16-bit, 4-bit |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

## LoRA Configuration

```json
{
  "r": 16,
  "lora_alpha": 16,
  "lora_dropout": 0,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}
```

**What these mean:**
- **r (rank)**: Lower = smaller adapter, faster training. 16 is a good default.
- **lora_alpha**: Scaling factor. Usually equals r.
- **target_modules**: Which layers to train. More modules = better quality, more memory.

## Training Configuration

| Setting | Value |
|---------|-------|
| Learning Rate | 0.0002 |
| Batch Size | 4 |
| Gradient Accumulation | 4 |
| Epochs | 3 |
| Max Sequence Length | 2048 |
| Precision | Auto (bf16 if supported, else fp16) |
| Optimizer | AdamW 8-bit |

## Using Your Fine-Tuned Model

### Option 1: With Unsloth (Recommended)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./output",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

messages = [{"role": "user", "content": "Hello!"}]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Option 2: With HuggingFace (after merging)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./output_merged_16bit")
model = AutoModelForCausalLM.from_pretrained("./output_merged_16bit")
```

### Option 3: With Ollama (after GGUF export)

```bash
ollama create my-model -f Modelfile
ollama run my-model
```

## Export Formats

| Format | Use Case | Size |
|--------|----------|------|
| LoRA Adapter | Load with base model | ~50MB |
| GGUF q4_k_m | llama.cpp, Ollama, LM Studio | ~2-4GB |
| GGUF q8_0 | Higher quality GGUF | ~4-8GB |
| 16-bit Merged | HuggingFace, full precision | ~6-14GB |
| 4-bit Merged | Low-memory inference | ~2-4GB |

## Troubleshooting

### Out of Memory

- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps` to 8
- Reduce `max_seq_length` to 1024
- Use a smaller model

### Slow Training

- Ensure you're using a GPU
- Check that Unsloth is properly installed
- Use `load_in_4bit=True` (default)

### Chat Template Issues

Different models use different templates. If output looks wrong:
- Check the model's HuggingFace page for the correct template
- Some models need `add_generation_prompt=True`

## Resources

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [HuggingFace PEFT](https://huggingface.co/docs/peft)
- [TRL Library](https://huggingface.co/docs/trl)