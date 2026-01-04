# L4D2-AI-Architect Training Guide

This guide covers the complete training workflow for fine-tuning LLMs on L4D2 modding code.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Data Collection](#data-collection)
3. [Dataset Preparation](#dataset-preparation)
4. [Model Training](#model-training)
5. [Model Export](#model-export)
6. [Local Inference](#local-inference)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

| Task | Minimum | Recommended |
|------|---------|-------------|
| Data Collection | Any CPU | 4+ cores |
| Fine-tuning (7B) | 16GB VRAM | 24GB+ VRAM |
| Fine-tuning (13B) | 24GB VRAM | 48GB+ VRAM |
| RL Training | 8GB VRAM | 16GB+ VRAM |

### Vultr GPU Instance Recommendations

- **A40 (48GB)**: Best value for 7B-13B models
- **L40S (48GB)**: Newer architecture, faster
- **A100 (80GB)**: For 34B+ models or large batch sizes

### Software Requirements

```bash
# Python 3.10+
python3 --version

# CUDA 11.8+ (Vultr has this pre-installed)
nvcc --version

# Verify GPU
nvidia-smi
```

---

## Data Collection

### 1. GitHub SourceMod Plugins

```bash
# Set your GitHub token (recommended for higher rate limits)
export GITHUB_TOKEN="ghp_your_token_here"

# Run the scraper
python scripts/scrapers/scrape_github_plugins.py \
    --max-repos 500 \
    --output data/raw/github_plugins
```

**Expected output:**
- `data/raw/github_plugins/github_plugins.jsonl` - All collected code
- `data/raw/github_plugins/scrape_stats.json` - Collection statistics

### 2. Valve Developer Wiki

```bash
python scripts/scrapers/scrape_valve_wiki.py \
    --max-pages 200 \
    --output data/raw/valve_wiki
```

**Expected output:**
- `data/raw/valve_wiki/valve_wiki.jsonl` - Wiki pages
- `data/raw/valve_wiki/code_blocks.jsonl` - Extracted code snippets

### 3. Time Estimates

| Source | Pages/Repos | Time |
|--------|-------------|------|
| GitHub (500 repos) | ~500 | 30-60 min |
| Valve Wiki | ~200 pages | 10-20 min |
| **Total** | | **~1 hour** |

---

## Dataset Preparation

### Convert Raw Data to Training Format

```bash
python scripts/training/prepare_dataset.py \
    --input data/raw \
    --output data/processed \
    --format unsloth \
    --min-quality 0.4
```

### Output Files

```
data/processed/
├── combined_train.jsonl      # All training data
├── combined_val.jsonl        # Validation set (10%)
├── sourcepawn_train.jsonl    # SourcePawn only
├── sourcepawn_val.jsonl
├── vscript_train.jsonl       # VScript only
├── vscript_val.jsonl
└── dataset_stats.json        # Statistics
```

### Training Format (ChatML)

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert SourcePawn developer..."},
    {"role": "user", "content": "Write a function to heal all survivors"},
    {"role": "assistant", "content": "public void HealAllSurvivors()\n{\n    for (int i = 1; i <= MaxClients; i++)\n    {\n        if (IsClientInGame(i) && GetClientTeam(i) == 2)\n        {\n            SetEntityHealth(i, 100);\n        }\n    }\n}"}
  ]
}
```

---

## Model Training

### Quick Start

```bash
# Using default config
python scripts/training/train_unsloth.py

# With custom settings
python scripts/training/train_unsloth.py \
    --model unsloth/mistral-7b-instruct-v0.3-bnb-4bit \
    --epochs 3 \
    --batch-size 4 \
    --lr 2e-4
```

### Configuration Options

Edit `configs/unsloth_config.yaml`:

```yaml
model:
  name: "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
  max_seq_length: 2048
  load_in_4bit: true

lora:
  r: 32           # LoRA rank (16-64)
  lora_alpha: 64  # Usually r or r*2
  
training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
```

### GPU Memory Usage

| Model | Batch Size | VRAM Usage |
|-------|------------|------------|
| Mistral-7B | 2 | ~12GB |
| Mistral-7B | 4 | ~16GB |
| Mistral-7B | 8 | ~24GB |
| CodeLlama-34B | 2 | ~28GB |
| CodeLlama-34B | 4 | ~40GB |

### Monitoring Training

```bash
# TensorBoard (in another terminal)
tensorboard --logdir data/training_logs --port 6006

# Watch GPU usage
watch -n 1 nvidia-smi
```

### Training Time Estimates

| Model | Dataset Size | GPU | Time |
|-------|--------------|-----|------|
| 7B | 1000 examples | A40 | 1-2 hours |
| 7B | 5000 examples | A40 | 4-6 hours |
| 13B | 5000 examples | A40 | 8-12 hours |

---

## Model Export

### Export to GGUF (for Ollama/llama.cpp)

```bash
python scripts/training/export_model.py \
    --input model_adapters/l4d2-code-lora/final \
    --format gguf \
    --quantize q4_k_m
```

### Quantization Options

| Method | Size | Quality | Use Case |
|--------|------|---------|----------|
| f16 | 14GB | Best | Highest quality |
| q8_0 | 7GB | Great | Good balance |
| q5_k_m | 5GB | Good | Most users |
| q4_k_m | 4GB | OK | Recommended |
| q3_k_m | 3GB | Lower | Memory constrained |

### Export to Ollama

```bash
# Create model with Ollama
python scripts/training/export_model.py \
    --input model_adapters/l4d2-code-lora/final \
    --format gguf \
    --quantize q4_k_m \
    --install-ollama l4d2-code

# Test it
ollama run l4d2-code "Write a SourcePawn function to spawn a Tank"
```

### Push to HuggingFace Hub

```bash
# Set your token
export HF_TOKEN="hf_your_token_here"

python scripts/training/export_model.py \
    --input model_adapters/l4d2-code-lora/final \
    --format lora \
    --push-to-hub username/l4d2-sourcemod-lora
```

---

## Local Inference

### With Unsloth

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "model_adapters/l4d2-code-lora/final",
    max_seq_length=2048,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

messages = [
    {"role": "system", "content": "You are an expert SourcePawn developer."},
    {"role": "user", "content": "Write a function to heal all survivors"},
]

inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0]))
```

### With Ollama

```bash
ollama run l4d2-code "Write a SourceMod plugin that hooks player_hurt"
```

### With llama.cpp

```bash
./main -m model-q4_k_m.gguf \
    -p "Write a VScript DirectorOptions table for more special infected:" \
    -n 256
```

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
--batch-size 2

# Enable gradient checkpointing (already enabled by default)
# Reduce sequence length
--max-seq-length 1024
```

### PyTorch Not Using GPU

```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show GPU name
```

If False, reinstall PyTorch with CUDA:
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Training Loss Not Decreasing

1. Check learning rate (try 1e-4 or 5e-5)
2. Verify dataset format is correct
3. Increase LoRA rank
4. Check for data quality issues

### Model Generates Garbage

1. Use correct chat template
2. Verify EOS token is correct
3. Check quantization wasn't too aggressive

---

## Best Practices

1. **Start small**: Test with 100 examples before full training
2. **Monitor closely**: Watch loss curves in TensorBoard
3. **Save checkpoints**: Every 100-200 steps
4. **Validate regularly**: Test outputs during training
5. **Backup models**: Download before Vultr credits expire!

---

## Quick Reference Commands

```bash
# Full training pipeline
./run_scraping.sh && python scripts/training/train_unsloth.py

# Monitor GPU
watch -n 1 nvidia-smi

# TensorBoard
tensorboard --logdir data/training_logs --port 6006

# Test model
python scripts/training/train_unsloth.py --test-only model_adapters/l4d2-code-lora/final

# Export to GGUF
python scripts/training/export_model.py --input model_adapters/l4d2-code-lora/final --format gguf

# Download model locally
scp -r root@VULTR_IP:~/L4D2-AI-Architect/model_adapters/l4d2-code-lora ./
```
