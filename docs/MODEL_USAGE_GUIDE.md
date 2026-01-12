# L4D2 AI Model Usage Guide

**Last Updated:** January 2026

This guide documents all trained models and how to use them for Left 4 Dead 2 SourcePawn/VScript code generation.

---

## Available Trained Models

All models are fine-tuned on the V15 dataset (2,773 examples) using TinyLlama-1.1B as the base.

| Model | LoRA Rank | Size | Epochs | Best For |
|-------|-----------|------|--------|----------|
| `l4d2-tiny-v15-lora256` | 256 | 770MB | 25 | **Best quality** - highest capacity adapter |
| `l4d2-tiny-v15-lora128` | 128 | 385MB | 15 | Good balance of quality and size |
| `l4d2-tiny-v15-lora64-long` | 64 | 193MB | 20 | Extended training, good generalization |
| `l4d2-tiny-v15-lora64-a100` | 64 | 193MB | 10 | Trained on A100 GPU |
| `l4d2-tiny-v15-lora64` | 64 | 193MB | 10 | Standard LoRA r=64 |
| `l4d2-tiny-v15-lora32-long` | 32 | 96MB | 20 | Smallest, extended training |
| `l4d2-tiny-v15-lora` | 16 | 96MB | 3 | Fastest inference, baseline |

### Legacy Mistral Models (Older, Larger)

| Model | Size | Notes |
|-------|------|-------|
| `l4d2-mistral-v10plus-lora` | 640MB | Mistral-7B base, older dataset |
| `l4d2-mistral-v10-lora` | 320MB | Mistral-7B base |
| `l4d2-mistral-v9-lora` | 320MB | Mistral-7B base |

---

## Quick Start

### 1. Test a Model (Recommended First Step)

```bash
# Activate environment
source activate.sh

# Test the best model
python scripts/inference/test_lora.py --adapter model_adapters/l4d2-tiny-v15-lora256

# Test a smaller model for faster inference
python scripts/inference/test_lora.py --adapter model_adapters/l4d2-tiny-v15-lora

# Test with Mistral base (requires more VRAM)
python scripts/inference/test_lora.py --adapter model_adapters/l4d2-mistral-v10plus-lora/final --base mistral
```

### 2. Interactive Code Generation

```bash
# Start the copilot CLI
python scripts/inference/copilot_cli.py complete --prompt "Write a SourcePawn plugin that announces Tank spawns"

# Or use the model server
python scripts/inference/model_server.py --adapter model_adapters/l4d2-tiny-v15-lora256
```

### 3. Start the Web UI

```bash
./start_web_ui.sh
# Opens at http://localhost:7860
```

---

## Programmatic Usage

### Basic Inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load LoRA adapter
adapter_path = "model_adapters/l4d2-tiny-v15-lora256"
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

# Generate code
messages = [
    {"role": "system", "content": "You are an expert SourcePawn developer for Left 4 Dead 2."},
    {"role": "user", "content": "Write a function to heal all survivors"}
]

chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(chat_text, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7, do_sample=True)
    
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Using the SDK

```python
from scripts.inference.l4d2_sdk import L4D2CodeGenerator

# Initialize with your preferred adapter
generator = L4D2CodeGenerator(adapter_path="model_adapters/l4d2-tiny-v15-lora256")

# Generate a plugin
code = generator.generate_plugin(
    description="Tank health announcer",
    features=["Announce Tank spawn", "Show remaining health", "Track damage dealers"]
)
print(code)
```

---

## Model Selection Guide

### Choose Based on Your Hardware

| Hardware | Recommended Model | VRAM Required |
|----------|-------------------|---------------|
| Apple Silicon (M1/M2/M3) | `l4d2-tiny-v15-lora256` | 4GB |
| CPU Only | `l4d2-tiny-v15-lora` | 4GB RAM |
| NVIDIA GPU (8GB+) | `l4d2-tiny-v15-lora256` | 4GB |
| NVIDIA GPU (16GB+) | `l4d2-mistral-v10plus-lora` | 14GB |

### Choose Based on Use Case

| Use Case | Recommended Model |
|----------|-------------------|
| Best code quality | `l4d2-tiny-v15-lora256` |
| Fast iteration/prototyping | `l4d2-tiny-v15-lora` |
| Production deployment | `l4d2-tiny-v15-lora128` |
| Resource-constrained | `l4d2-tiny-v15-lora32-long` |

---

## Training Data Overview

The V15 dataset contains:
- **2,773 training examples**
- **70 validation examples**
- Sources: GitHub SourcePawn repositories
- Quality score: 0.948 average
- Focus: L4D2-specific code patterns

### Dataset Location
```
data/processed/l4d2_train_v15.jsonl  # Training data
data/processed/l4d2_val_v15.jsonl    # Validation data
```

---

## Inference Tips

### Prompt Engineering

The models work best with specific, detailed prompts:

```
# Good prompt
"Write a SourcePawn function that detects when a Survivor picks up a health kit and announces it to all players with the player's name and current health."

# Less effective prompt
"health kit plugin"
```

### Temperature Settings

| Temperature | Use Case |
|-------------|----------|
| 0.3 | Deterministic, consistent output |
| 0.7 | Balanced creativity (default) |
| 1.0 | More varied, experimental |

### Max Tokens

- Short functions: 150-200 tokens
- Full plugins: 500-1000 tokens
- Complex systems: 1500+ tokens

---

## Troubleshooting

### Model Not Loading

```bash
# Check adapter files exist
ls -la model_adapters/l4d2-tiny-v15-lora256/

# Required files:
# - adapter_model.safetensors
# - adapter_config.json
# - tokenizer.json
```

### Out of Memory

```bash
# Use a smaller model
python scripts/inference/test_lora.py --adapter model_adapters/l4d2-tiny-v15-lora

# Or reduce max_new_tokens in generation
```

### Poor Output Quality

1. Use the larger `lora256` or `lora128` models
2. Improve your prompt with more context
3. Lower temperature for more focused outputs

---

## File Locations

```
model_adapters/
├── l4d2-tiny-v15-lora/           # Base 3-epoch model
├── l4d2-tiny-v15-lora64/         # LoRA r=64
├── l4d2-tiny-v15-lora64-a100/    # A100 trained
├── l4d2-tiny-v15-lora64-long/    # 20 epochs
├── l4d2-tiny-v15-lora128/        # LoRA r=128
├── l4d2-tiny-v15-lora256/        # Best quality
├── l4d2-tiny-v15-lora32-long/    # Smallest, long training
├── l4d2-mistral-v10plus-lora/    # Legacy Mistral
├── l4d2-mistral-v10-lora/        # Legacy Mistral
└── l4d2-mistral-v9-lora/         # Legacy Mistral
```

---

## Next Steps

1. **Test the models**: `python scripts/inference/test_lora.py`
2. **Try the Web UI**: `./start_web_ui.sh`
3. **Generate plugins**: Use `scripts/inference/plugin_wizard.py`
4. **Evaluate quality**: `python scripts/evaluation/automated_test.py`

For training new models, see `docs/TRAINING_GUIDE.md`.
