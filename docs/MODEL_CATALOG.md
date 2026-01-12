# L4D2-AI-Architect Model Catalog

> Last Updated: January 10, 2026

This document catalogs all trained models available in this project.

---

## Production Models (Mistral 7B)

### l4d2-mistral-v10plus-lora (RECOMMENDED)

| Property | Value |
|----------|-------|
| Base Model | Mistral-7B-Instruct-v0.3 (4-bit) |
| LoRA Rank | 32 |
| Training Data | V10+ combined dataset |
| Size | 644MB |
| Status | **Production Ready** |
| GGUF Export | `exports/l4d2-v10plus/gguf/` |

**Best for:** General SourcePawn/VScript code generation with high quality.

```bash
# Install to Ollama
ollama create l4d2-code-v10plus -f exports/l4d2-v10plus/gguf/Modelfile

# Use
ollama run l4d2-code-v10plus "Write a function to spawn a Tank"
```

---

## TinyLlama V15 Models (1.1B Parameters)

These lightweight models are trained on the V15 dataset (2,773 examples) and are optimized for fast inference.

### Comparison Table

| Model | LoRA r | Epochs | Adapter Size | Accuracy | Best For |
|-------|--------|--------|--------------|----------|----------|
| lora32-long | 32 | 30 | 96MB | 99.2% | Balanced size/quality |
| lora64 | 64 | 10 | 193MB | ~99% | Quick training baseline |
| lora64-long | 64 | 25 | 193MB | 99.2% | Extended training |
| lora128 | 128 | 15 | 385MB | 99.4% | Higher capacity |
| lora256 | 256 | 20 | 770MB | 99.3% | Maximum capacity |

### Recommended: lora64-long or lora128

- **lora64-long**: Best balance of size (193MB) and accuracy (99.2%)
- **lora128**: Slightly higher accuracy (99.4%) with larger size (385MB)

### Model Details

#### l4d2-tiny-v15-lora32-long
```
Base: TinyLlama/TinyLlama-1.1B-Chat-v1.0
LoRA: r=32, alpha=64
Epochs: 30
Accuracy: 99.2%
Path: model_adapters/l4d2-tiny-v15-lora32-long/
```

#### l4d2-tiny-v15-lora64
```
Base: TinyLlama/TinyLlama-1.1B-Chat-v1.0
LoRA: r=64, alpha=128
Epochs: 10
Path: model_adapters/l4d2-tiny-v15-lora64/
```

#### l4d2-tiny-v15-lora64-long
```
Base: TinyLlama/TinyLlama-1.1B-Chat-v1.0
LoRA: r=64, alpha=128
Epochs: 25
Accuracy: 99.2%
Path: model_adapters/l4d2-tiny-v15-lora64-long/
```

#### l4d2-tiny-v15-lora128
```
Base: TinyLlama/TinyLlama-1.1B-Chat-v1.0
LoRA: r=128, alpha=256
Epochs: 15
Accuracy: 99.4%
Path: model_adapters/l4d2-tiny-v15-lora128/
```

#### l4d2-tiny-v15-lora256
```
Base: TinyLlama/TinyLlama-1.1B-Chat-v1.0
LoRA: r=256, alpha=512
Epochs: 20
Accuracy: 99.3%
Path: model_adapters/l4d2-tiny-v15-lora256/
```

---

## How to Use These Models

### Option 1: Export to GGUF for Ollama (Recommended)

```bash
# Export TinyLlama model to GGUF
python scripts/training/export_gguf_cpu.py \
    --adapter model_adapters/l4d2-tiny-v15-lora64-long

# Install to Ollama
ollama create l4d2-tiny-v15 -f exports/l4d2-tiny-v15/gguf/Modelfile

# Use
ollama run l4d2-tiny-v15 "Write a survivor healing function"
```

### Option 2: Direct Python Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "model_adapters/l4d2-tiny-v15-lora64-long"
)

tokenizer = AutoTokenizer.from_pretrained(
    "model_adapters/l4d2-tiny-v15-lora64-long"
)

# Generate
prompt = "<|user|>\nWrite a Tank spawner plugin\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0]))
```

### Option 3: CLI Tool

```bash
# Using Ollama backend (after export)
python scripts/inference/copilot_cli.py ollama --prompt "Create a special infected spawner"

# Interactive chat
python scripts/inference/copilot_cli.py chat
```

---

## Training Data

All V15 models were trained on `data/processed/l4d2_train_v15.jsonl`:
- **2,773 examples** of SourcePawn/VScript code
- ChatML format with system prompts
- Quality-filtered and augmented

---

## Storage Summary

| Category | Size |
|----------|------|
| TinyLlama V15 Models | ~14.7GB |
| Mistral V10+ Models | ~1GB |
| GGUF Exports | ~14GB |
| Training Data | ~40MB |

---

## Model Selection Guide

| Use Case | Recommended Model |
|----------|-------------------|
| Production API/Server | l4d2-mistral-v10plus-lora |
| Local CPU inference | TinyLlama GGUF export |
| Edge/embedded devices | l4d2-tiny-v15-lora32-long |
| Maximum quality | l4d2-tiny-v15-lora128 |
| Experimentation | l4d2-tiny-v15-lora64 |
