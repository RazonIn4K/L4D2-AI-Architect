# L4D2-AI-Architect: Complete Architecture Guide

**Last Updated**: January 7, 2026  
**Current Production Model**: V7 (`ft:gpt-4o-mini-2024-07-18:highencodelearning:l4d2-sourcemod-v7:CvTBCVPi`)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Component Deep Dive](#component-deep-dive)
4. [Data Pipeline](#data-pipeline)
5. [Training Pipeline](#training-pipeline)
6. [Inference Pipeline](#inference-pipeline)
7. [Model Version History](#model-version-history)
8. [Quick Start Guide](#quick-start-guide)
9. [Cost Structure](#cost-structure)
10. [Configuration Reference](#configuration-reference)

---

## Project Overview

L4D2-AI-Architect is a comprehensive AI system for Left 4 Dead 2 modding that includes:

| Component | Purpose | Status |
|-----------|---------|--------|
| **SourcePawn Copilot** | Fine-tuned LLM for L4D2 plugin code generation | Production (V7) |
| **Web UI** | Browser-based code generation interface | Ready |
| **API Server** | REST API for IDE/tool integrations | Ready |
| **CLI Tool** | Command-line code generation | Ready |
| **RL Bot Agents** | PPO agents for bot control (Mnemosyne) | Development |
| **AI Director** | Dynamic gameplay sequence generation | Development |

### Key Achievements

- **80% pass rate** on automated test battery
- **100% L4D2 API correctness** (no wrong APIs like `RandomFloat`)
- **$0.002-0.004** per code generation
- **<3 second** average response time

---

## System Architecture

```
+-----------------------------------------------------------------------------------+
|                              L4D2-AI-Architect                                     |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  +------------------+     +------------------+     +------------------+           |
|  |   Data Layer     |     |  Training Layer  |     | Inference Layer  |           |
|  +------------------+     +------------------+     +------------------+           |
|  |                  |     |                  |     |                  |           |
|  | GitHub Scraper   |---->| Dataset Prep     |---->| Web UI           |           |
|  | Valve Wiki       |     | Quality Filter   |     | API Server       |           |
|  | Anti-patterns    |     | OpenAI Fine-tune |     | CLI Tool         |           |
|  |                  |     |                  |     |                  |           |
|  +------------------+     +------------------+     +------------------+           |
|           |                       |                       |                       |
|           v                       v                       v                       |
|  +------------------+     +------------------+     +------------------+           |
|  | data/raw/        |     | data/processed/  |     | OpenAI API       |           |
|  | data/anti_patterns/    | data/openai_finetune/  | (V7 Model)       |           |
|  +------------------+     +------------------+     +------------------+           |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

### Data Flow

```
GitHub Repos ──┐
               ├──> Raw JSONL ──> Quality Filter ──> ChatML JSONL ──> OpenAI Fine-tune
Valve Wiki ────┘                                                            │
                                                                            v
User Prompt ──────────────────────────────────────────────────────> V7 Model ──> Code
```

---

## Component Deep Dive

### 1. Data Collection (`scripts/scrapers/`)

| Script | Purpose | Output |
|--------|---------|--------|
| `scrape_github_plugins.py` | Scrapes SourceMod plugins from GitHub | `data/raw/github_plugins.jsonl` |
| `scrape_valve_wiki.py` | Scrapes Valve Developer Wiki | `data/raw/valve_wiki.jsonl` |

**Usage**:
```bash
# Full scraping pipeline
./run_scraping.sh

# Or individually
python scripts/scrapers/scrape_github_plugins.py --max-repos 500
python scripts/scrapers/scrape_valve_wiki.py --max-pages 200
```

### 2. Dataset Preparation (`scripts/training/`)

| Script | Purpose | Output |
|--------|---------|--------|
| `prepare_dataset.py` | Converts raw data to ChatML format | `data/processed/*.jsonl` |

**Quality Filtering Applied**:
- Removes examples < 10 lines
- Removes documentation-only content
- Removes corrupted/gibberish outputs
- Validates L4D2-specific patterns
- Adds anti-pattern training examples

**Training Data Format** (ChatML):
```json
{
  "messages": [
    {"role": "system", "content": "You are an expert SourcePawn developer..."},
    {"role": "user", "content": "Write a plugin that spawns tanks"},
    {"role": "assistant", "content": "#pragma semicolon 1\n#include <sourcemod>..."}
  ]
}
```

### 3. Training Scripts (`scripts/training/`)

| Script | Purpose | Platform |
|--------|---------|----------|
| `start_openai_finetune.py` | Start OpenAI fine-tuning job | OpenAI API |
| `check_finetune_status.py` | Monitor training progress | OpenAI API |
| `train_unsloth.py` | Local QLoRA training | GPU (A40/A100) |
| `train_runpod.py` | RunPod cloud training | RunPod |
| `export_model.py` | Export to GGUF/Ollama | Local |

### 4. Inference Tools (`scripts/inference/`)

| Tool | Type | Use Case |
|------|------|----------|
| `web_ui.py` | Web Interface | Browser-based generation with presets |
| `copilot_server_openai.py` | REST API | IDE integrations, programmatic access |
| `l4d2_codegen.py` | CLI | Terminal-based generation, batch processing |
| `copilot_cli.py` | CLI (Legacy) | Original CLI interface |

### 5. Evaluation (`scripts/evaluation/`)

| Script | Purpose |
|--------|---------|
| `automated_test.py` | Run 10-prompt test battery |
| `compare_models.py` | Compare model versions |
| `validate_generated_code.py` | Validate SourcePawn syntax |

### 6. Security (`scripts/utils/security.py`)

All file I/O uses security wrappers:
- `safe_path()` - Path traversal prevention
- `safe_read_json/yaml/text()` - Secure file reading
- `safe_write_json/text/jsonl()` - Secure file writing
- `validate_url()` - SSRF prevention for scrapers

---

## Data Pipeline

### Directory Structure

```
data/
├── raw/                          # Scraped data (gitignored)
│   ├── github_plugins.jsonl
│   └── valve_wiki.jsonl
├── processed/                    # Training-ready data
│   ├── combined_train.jsonl      # All training data
│   ├── combined_val.jsonl        # Validation set
│   ├── l4d2_combined_train.jsonl # L4D2-filtered
│   ├── l4d2_combined_train_fixed.jsonl
│   ├── sourcepawn_train.jsonl
│   ├── sourcepawn_val.jsonl
│   ├── synthetic_examples.jsonl  # Hand-crafted examples
│   └── dataset_stats.json        # Statistics
├── anti_patterns/                # Contrastive learning data
│   └── *.jsonl                   # Wrong vs correct examples
├── openai_finetune/              # OpenAI-formatted data
│   ├── train_v7.jsonl
│   └── eval_v7.jsonl
└── training_logs/                # TensorBoard logs
```

### Dataset Statistics (Current)

```json
{
  "total_examples": 676,
  "sourcepawn_examples": 676,
  "anti_pattern_examples": 109,
  "avg_quality": 0.948,
  "game_detection": {
    "generic": 320,
    "l4d2": 356
  }
}
```

---

## Training Pipeline

### OpenAI Fine-tuning (Recommended)

```bash
# 1. Prepare dataset
python scripts/training/prepare_dataset.py

# 2. Start fine-tuning
export OPENAI_API_KEY="sk-..."
python scripts/training/start_openai_finetune.py --version v7

# 3. Monitor progress
python scripts/training/check_finetune_status.py

# 4. Test the model
python scripts/inference/l4d2_codegen.py generate "Test prompt"
```

### Local Training (Vultr/RunPod)

```bash
# On GPU instance
source activate.sh
python scripts/training/train_unsloth.py --config configs/unsloth_config.yaml

# Export to GGUF
python scripts/training/export_model.py \
    --input model_adapters/l4d2-code-lora/final \
    --format gguf \
    --quantize q4_k_m
```

### Training Configurations

| Config | GPU | VRAM | Batch Size | Use Case |
|--------|-----|------|------------|----------|
| `unsloth_config.yaml` | A40/L40S | 48GB | 4 | Default |
| `unsloth_config_a100.yaml` | A100 | 40GB | 8 | Fast training |
| `unsloth_config_gh200.yaml` | GH200 | 96GB | 16 | Large models |
| `unsloth_config_qwen.yaml` | Any | - | - | Qwen models |

---

## Inference Pipeline

### Option 1: Web UI (Easiest)

```bash
cd /Users/davidortiz/left4dead-model/L4D2-AI-Architect

# Set API key
export OPENAI_API_KEY="sk-..."

# Start the server
python scripts/inference/web_ui.py --port 8000

# Open browser to http://localhost:8000
```

**Features**:
- Preset prompts for common plugins
- Syntax highlighting
- Cost tracking
- Chat mode for iterative development

### Option 2: CLI Tool

```bash
# Single generation
python scripts/inference/l4d2_codegen.py generate "Write a tank announcer plugin"

# Interactive chat
python scripts/inference/l4d2_codegen.py chat

# Batch generation
python scripts/inference/l4d2_codegen.py batch prompts.txt --output generated/

# Cost estimation
python scripts/inference/l4d2_codegen.py estimate prompts.txt

# Model info
python scripts/inference/l4d2_codegen.py info
```

### Option 3: REST API

```bash
# Start API server
python scripts/inference/copilot_server_openai.py --port 8000
```

**Endpoints**:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/complete` | POST | Code completion |
| `/v1/chat/completions` | POST | Chat (OpenAI-compatible) |
| `/v1/generate-plugin` | POST | Full plugin generation |

**Example Request**:
```bash
curl -X POST http://localhost:8000/v1/complete \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a function to heal all survivors",
    "temperature": 0.1,
    "max_tokens": 1024
  }'
```

### Option 4: Direct Python

```python
from openai import OpenAI

client = OpenAI()

MODEL_ID = "ft:gpt-4o-mini-2024-07-18:highencodelearning:l4d2-sourcemod-v7:CvTBCVPi"

SYSTEM_PROMPT = """You are an expert SourcePawn developer for L4D2.
CRITICAL: Use GetRandomFloat(), NOT RandomFloat(). Use lunge_pounce, NOT pounce."""

response = client.chat.completions.create(
    model=MODEL_ID,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Write a tank spawn announcer plugin"}
    ],
    max_tokens=2048,
    temperature=0.1
)

print(response.choices[0].message.content)
```

---

## Model Version History

| Version | Date | Pass Rate | API Correctness | Key Changes |
|---------|------|-----------|-----------------|-------------|
| V1 | Jan 5, 2026 | 0% | - | Failed: 69% vague prompts |
| V2 | Jan 6, 2026 | 90% | 85% | Quality filtering, synthetic data |
| V3 | Jan 6, 2026 | 70% | 80% | Comprehensive dataset |
| V5 | Jan 6, 2026 | 80% | 100% | Anti-pattern training |
| V6 | Jan 7, 2026 | 50% | 87% | Regression: contaminated security data |
| **V7** | Jan 7, 2026 | **80%** | **100%** | **Production: sanitized data** |

### V7 Model Details

- **Model ID**: `ft:gpt-4o-mini-2024-07-18:highencodelearning:l4d2-sourcemod-v7:CvTBCVPi`
- **Base Model**: GPT-4o-mini
- **Training Examples**: 627 train + 70 eval
- **Trained Tokens**: 703,947
- **Anti-pattern Signal**: 14.8% of dataset

### Critical L4D2 API Rules (Enforced by V7)

| Wrong API | Correct API |
|-----------|-------------|
| `RandomFloat()` | `GetRandomFloat()` |
| `RandomInt()` | `GetRandomInt()` |
| `pounce` event | `lunge_pounce` event |
| `smoker_tongue_grab` | `tongue_grab` |
| `boomer_vomit` | `player_now_it` |
| `charger_grab` | `charger_carry_start` |
| `m_flSpeed` | `m_flLaggedMovementValue` |

---

## Quick Start Guide

### Prerequisites

```bash
# Python 3.10+
python3 --version

# Install dependencies
cd /Users/davidortiz/left4dead-model/L4D2-AI-Architect
pip install -r requirements.txt
pip install openai fastapi uvicorn
```

### Set Up API Key

```bash
# Option 1: Environment variable
export OPENAI_API_KEY="sk-your-key-here"

# Option 2: .env file
echo "OPENAI_API_KEY=sk-your-key-here" >> .env

# Option 3: Doppler (recommended for teams)
doppler run --project local-mac-work --config dev_personal -- python ...
```

### Start Generating Code

```bash
# Web UI (recommended for beginners)
python scripts/inference/web_ui.py
# Open http://localhost:8000

# CLI (for power users)
python scripts/inference/l4d2_codegen.py chat

# API (for integrations)
python scripts/inference/copilot_server_openai.py
```

---

## Cost Structure

### OpenAI Fine-tuned GPT-4o-mini Pricing

| Usage | Input Cost | Output Cost | Total |
|-------|------------|-------------|-------|
| Per 1M tokens | $0.30 | $1.20 | - |
| Single generation (~1500 tokens) | ~$0.0001 | ~$0.0018 | **~$0.002** |
| 100 generations | ~$0.01 | ~$0.18 | **~$0.20** |
| Batch API (50% off) | - | - | **~$0.10** |

### Typical Session Costs

| Activity | Est. Cost |
|----------|-----------|
| Generate one plugin | $0.002-0.004 |
| 30-minute chat session | $0.02-0.05 |
| Batch generate 50 plugins | $0.10-0.20 |
| Run full test battery | $0.02 |

### Cost Optimization Tips

1. Use **temperature 0.1-0.2** for consistent output (fewer retries)
2. Use **Batch API** for bulk generation (50% discount, 24hr delivery)
3. Cache common responses
4. Use presets for standard plugin types

---

## Configuration Reference

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for inference |
| `GITHUB_TOKEN` | No | For scraping (higher rate limits) |
| `HF_TOKEN` | No | For HuggingFace model upload |

### Config Files

| File | Purpose |
|------|---------|
| `configs/unsloth_config.yaml` | Local training config |
| `configs/director_config.yaml` | AI Director settings |
| `.env` | Environment variables |
| `.env.example` | Template for .env |

### Model Parameters

| Parameter | Recommended | Range | Effect |
|-----------|-------------|-------|--------|
| `temperature` | 0.1 | 0.0-1.0 | Higher = more creative/varied |
| `max_tokens` | 2048 | 256-4096 | Max response length |
| `top_p` | 1.0 | 0.0-1.0 | Nucleus sampling |

---

## Directory Structure Summary

```
L4D2-AI-Architect/
├── configs/                    # Training & system configs
├── data/
│   ├── raw/                   # Scraped data (gitignored)
│   ├── processed/             # Training datasets
│   ├── anti_patterns/         # Contrastive examples
│   ├── openai_finetune/       # OpenAI-formatted data
│   └── training_logs/         # TensorBoard logs
├── docs/                      # Documentation
│   ├── ARCHITECTURE.md        # This file
│   ├── FINAL_MODEL_COMPARISON.md
│   ├── V7_TRAINING_NOTES.md
│   └── ...
├── model_adapters/            # Trained models (gitignored)
├── scripts/
│   ├── scrapers/              # Data collection
│   ├── training/              # Training scripts
│   ├── inference/             # Inference tools
│   │   ├── web_ui.py          # Web interface
│   │   ├── copilot_server_openai.py  # API server
│   │   └── l4d2_codegen.py    # CLI tool
│   ├── evaluation/            # Testing & validation
│   ├── director/              # AI Director (WIP)
│   ├── rl_training/           # RL agents (WIP)
│   └── utils/                 # Utilities & security
├── tests/                     # Unit tests
├── requirements.txt
├── run_scraping.sh
├── run_training.sh
└── setup.sh
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `OPENAI_API_KEY not set` | Set env var or add to .env file |
| `openai package not found` | Run `pip install openai` |
| Wrong API in output | Use V7 model (has 100% API correctness) |
| High variance in output | Lower temperature to 0.1 |
| Empty responses | Check API key permissions |

### Getting Help

1. Check `docs/` for detailed guides
2. Run `python scripts/inference/l4d2_codegen.py info` for model info
3. Check server logs for API errors

---

## Future Roadmap

| Feature | Status | Priority |
|---------|--------|----------|
| VScript support | Planned | Medium |
| VSCode extension | Planned | High |
| Compile validation | Planned | High |
| More anti-patterns | Ongoing | Medium |
| RL bot agents | Development | Low |
| AI Director | Development | Low |

---

*Documentation generated January 7, 2026*
