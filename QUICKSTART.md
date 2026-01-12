# L4D2-AI-Architect Quick Start

> Get up and running in 5 minutes

## What You Have

You have **7 trained AI models** for generating Left 4 Dead 2 plugin code:
- 1 Mistral 7B model (high quality, larger)
- 6 TinyLlama 1.1B models (fast, lightweight)

All models achieve **99%+ training accuracy** on SourcePawn/VScript code generation.

---

## Fastest Path: Use with Ollama

### Step 1: Export a TinyLlama Model to GGUF

```bash
cd /Users/davidortiz/left4dead-model/L4D2-AI-Architect

# Activate environment
source activate.sh

# Export the best TinyLlama model (r=64, 25 epochs)
python scripts/training/export_gguf_cpu.py \
    --adapter model_adapters/l4d2-tiny-v15-lora64-long
```

### Step 2: Install to Ollama

```bash
ollama create l4d2-tiny -f exports/l4d2-tiny-v15-lora64-long/gguf/Modelfile
```

### Step 3: Use It

```bash
# Direct query
ollama run l4d2-tiny "Write a function that heals all survivors to full health"

# Or use the CLI tool
python scripts/inference/copilot_cli.py ollama --prompt "Create a Tank spawner"

# Interactive mode
python scripts/inference/copilot_cli.py chat
```

---

## Already Exported: Mistral v10plus

The Mistral model is already exported:

```bash
# Install existing Mistral export
ollama create l4d2-code -f exports/l4d2-v10plus/gguf/Modelfile

# Use
ollama run l4d2-code "Write a VScript that spawns witch hordes"
```

---

## Available Models

| Model | Size | Accuracy | Export Status |
|-------|------|----------|---------------|
| Mistral v10plus | 7B params | High | Exported |
| TinyLlama r=32-long | 1.1B | 99.2% | Needs export |
| TinyLlama r=64-long | 1.1B | 99.2% | Needs export |
| TinyLlama r=128 | 1.1B | 99.4% | Needs export |
| TinyLlama r=256 | 1.1B | 99.3% | Needs export |

---

## Example Prompts

```
# SourcePawn
"Write a plugin that spawns a Tank when all survivors are incapacitated"
"Create a heal command that heals the player who typed it"
"Make a plugin that announces special infected spawns in chat"

# VScript
"Write a VScript that creates a custom panic event"
"Create a director script that increases zombie spawns over time"
"Make a mutation that gives survivors unlimited ammo"
```

---

## Project Structure

```
L4D2-AI-Architect/
├── model_adapters/          # Trained LoRA models
│   ├── l4d2-mistral-v10plus-lora/
│   ├── l4d2-tiny-v15-lora64-long/   # Recommended
│   └── ...
├── exports/                 # GGUF exports for Ollama
├── scripts/
│   ├── inference/           # CLI and server tools
│   └── training/            # Training scripts
├── data/processed/          # Training datasets
└── configs/                 # Training configs
```

---

## Next Steps

1. **Export more models**: Try different TinyLlama variants
2. **Compare quality**: Test outputs from r=64 vs r=128 vs r=256
3. **Train new models**: Use `train_unsloth.py` with your own data
4. **Deploy server**: Run `copilot_server.py` for API access

See `docs/MODEL_CATALOG.md` for full model details.
