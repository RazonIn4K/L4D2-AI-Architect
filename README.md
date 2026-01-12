# L4D2-AI-Architect

> **Train specialized AI models for Left 4 Dead 2 modding, bot control, and Director manipulation**

<!-- CI/CD Badges -->
[![CI](https://github.com/RazonIn4K/L4D2-AI-Architect/actions/workflows/ci.yml/badge.svg)](https://github.com/RazonIn4K/L4D2-AI-Architect/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/RazonIn4K/L4D2-AI-Architect/branch/main/graph/badge.svg)](https://codecov.io/gh/RazonIn4K/L4D2-AI-Architect)
[![Security Scan](https://github.com/RazonIn4K/L4D2-AI-Architect/actions/workflows/ci.yml/badge.svg?event=schedule)](https://github.com/RazonIn4K/L4D2-AI-Architect/actions/workflows/ci.yml)

<!-- Project Badges -->
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model Version](https://img.shields.io/badge/model-v13-purple.svg)](https://github.com/RazonIn4K/L4D2-AI-Architect/releases)
[![Ollama Ready](https://img.shields.io/badge/Ollama-ready-orange.svg)](https://ollama.ai)

<!-- Code Quality -->
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

---

## Overview

L4D2-AI-Architect is a comprehensive AI system for Left 4 Dead 2 modding that includes:

| Component | Description | Status |
|-----------|-------------|--------|
| **SourcePawn Copilot** | Fine-tuned LLM for L4D2 plugin code generation | Production |
| **VScript Generator** | AI-powered Squirrel scripting for mutations | Production |
| **RL Bot Agents** | PPO agents for intelligent bot control | Development |
| **AI Director** | Dynamic gameplay sequence generation | Development |

---

## Features at a Glance

```
+-----------------------------------------------------------------------------------+
|                           L4D2-AI-ARCHITECT FEATURES                               |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [CODE GENERATION]          [RL TRAINING]           [AI DIRECTOR]                 |
|  +------------------+       +------------------+    +------------------+          |
|  | SourcePawn       |       | PPO Agents       |    | Dynamic Events   |          |
|  | VScript/Squirrel |       | 5 Personalities  |    | Spawn Control    |          |
|  | Anti-patterns    |       | 14 Bot Actions   |    | Difficulty Mgmt  |          |
|  | L4D2 APIs        |       | 20D Observations |    | Rule + RL Modes  |          |
|  +------------------+       +------------------+    +------------------+          |
|         |                          |                       |                      |
|         v                          v                       v                      |
|  +------------------------------------------------------------------+            |
|  |                      INFERENCE OPTIONS                            |            |
|  |  Web UI  |  CLI Tool  |  REST API  |  Ollama  |  Python SDK      |            |
|  +------------------------------------------------------------------+            |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

---

## Quick Start

### Option 1: Use Pre-trained Model with Ollama (Easiest)

```bash
# Install Ollama (https://ollama.ai)
curl -fsSL https://ollama.ai/install.sh | sh

# Clone and install model
git clone https://github.com/RazonIn4K/L4D2-AI-Architect
cd L4D2-AI-Architect
cd exports/l4d2-v10plus/gguf
ollama create l4d2-code-v10plus -f Modelfile

# Generate code!
ollama run l4d2-code-v10plus "Write a Tank health announcer plugin"
```

### Option 2: Use OpenAI Fine-tuned Model (Recommended for Quality)

```bash
# Clone repository
git clone https://github.com/RazonIn4K/L4D2-AI-Architect
cd L4D2-AI-Architect

# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY="sk-your-key-here"

# Start Web UI
python scripts/inference/web_ui.py --port 8000
# Open http://localhost:8000

# Or use CLI
python scripts/inference/l4d2_codegen.py generate "Write a Tank spawn announcer"
```

### Option 3: Full Setup with Training

```bash
# One-command setup
git clone https://github.com/RazonIn4K/L4D2-AI-Architect
cd L4D2-AI-Architect
./setup.sh

# Activate environment
source activate.sh

# Collect data and train
./run_scraping.sh
./run_training.sh
```

---

## Architecture

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
|  | Anti-patterns    |     | OpenAI/Unsloth   |     | CLI Tool         |           |
|  |                  |     | LoRA Fine-tune   |     | Ollama           |           |
|  +------------------+     +------------------+     +------------------+           |
|           |                       |                       |                       |
|           v                       v                       v                       |
|  +------------------+     +------------------+     +------------------+           |
|  | data/raw/        |     | data/processed/  |     | OpenAI API       |           |
|  | (GitHub plugins) |     | (ChatML JSONL)   |     | Ollama Local     |           |
|  +------------------+     +------------------+     +------------------+           |
|                                                                                   |
|  +-----------------------------------+    +-----------------------------------+   |
|  |         RL Training               |    |         AI Director               |   |
|  +-----------------------------------+    +-----------------------------------+   |
|  |  Mnemosyne Environment            |    |  Rule-Based Mode                  |   |
|  |  PPO with Stable-Baselines3       |    |  RL-Based Mode                    |   |
|  |  5 Bot Personalities              |    |  Hybrid Mode                      |   |
|  |  UDP Game Bridge                  |    |  Event Generation                 |   |
|  +-----------------------------------+    +-----------------------------------+   |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

### Data Flow

```
GitHub Repos ----+
                 |
Valve Wiki ------+---> Raw JSONL --> Quality Filter --> ChatML JSONL
                 |                                            |
Anti-patterns ---+                                            v
                                                    +------------------+
                                                    | Fine-tune Model  |
                                                    | (OpenAI/Unsloth) |
                                                    +------------------+
                                                            |
                            +-------------------------------+
                            v                               v
                    +---------------+               +---------------+
                    | OpenAI API    |               | GGUF/Ollama   |
                    | (Cloud)       |               | (Local)       |
                    +---------------+               +---------------+
                            |                               |
                            +-------------------------------+
                                            |
                                            v
                    +-----------------------------------------+
                    |  Web UI  |  CLI  |  API  |  Python SDK  |
                    +-----------------------------------------+
```

---

## Dataset Statistics (V13)

### Overview

| Metric | Value |
|--------|-------|
| **Total Examples** | 1,010 |
| **SourcePawn Examples** | 856 |
| **VScript Examples** | 154 |
| **Anti-pattern Examples** | 109 |
| **Average Quality Score** | 0.948 |

### Category Distribution

```
SourcePawn Categories:
+---------------------------+-------+
| Category                  | Count |
+---------------------------+-------+
| Event Hooks               |   245 |
| Game Mechanics            |   198 |
| Admin Commands            |   156 |
| Entity Management         |   134 |
| Player State              |    89 |
| Statistics/Tracking       |    34 |
+---------------------------+-------+

VScript Categories:
+---------------------------+-------+
| Category                  | Count |
+---------------------------+-------+
| Director Options          |    52 |
| Mutations                 |    48 |
| Entity Scripts            |    32 |
| Event Triggers            |    22 |
+---------------------------+-------+
```

### Quality Filtering Applied

- Removed examples < 10 lines
- Removed documentation-only content
- Removed corrupted/gibberish outputs
- Validated L4D2-specific patterns
- Added contrastive anti-pattern examples

---

## Model Performance

### Benchmark Results (V13 Model)

| Metric | Score |
|--------|-------|
| **Pass Rate** | 80% |
| **L4D2 API Correctness** | 100% |
| **Syntax Validity** | 95% |
| **Average Response Quality** | 8.4/10 |

### Comparison with Base Models

| Model | Pass Rate | API Correctness | Would Compile |
|-------|-----------|-----------------|---------------|
| GPT-4o-mini (Base) | 15% | 40% | 20% |
| GPT-4o-mini (V1 Fine-tuned) | 0% | 0% | 10% |
| GPT-4o-mini (V7 Fine-tuned) | 80% | 100% | 90% |
| Mistral-7B (LoRA V10+) | 75% | 95% | 85% |

### L4D2 API Accuracy (Critical Corrections)

The model correctly uses these L4D2-specific APIs:

| Correct API | Wrong API (Avoided) |
|-------------|---------------------|
| `GetRandomFloat()` | `RandomFloat()` |
| `GetRandomInt()` | `RandomInt()` |
| `lunge_pounce` event | `pounce` event |
| `tongue_grab` event | `smoker_tongue_grab` |
| `player_now_it` event | `boomer_vomit` |
| `charger_carry_start` | `charger_grab` |
| `m_flLaggedMovementValue` | `m_flSpeed` |

---

## Deployment Options

### 1. Local with Ollama (Free, Offline)

```bash
# Export trained model to GGUF
python scripts/training/export_gguf_cpu.py \
    --adapter model_adapters/l4d2-mistral-v10plus-lora/final

# Install to Ollama
cd exports/l4d2-v10plus/gguf
ollama create l4d2-code-v10plus -f Modelfile

# Use the model
ollama run l4d2-code-v10plus "Your prompt here"

# Or via CLI
python scripts/inference/copilot_cli.py ollama \
    --prompt "Write a Tank health announcer"
```

**Requirements**: 16GB RAM, no GPU required

### 2. OpenAI API (Best Quality)

```bash
# Set API key
export OPENAI_API_KEY="sk-..."

# Web UI
python scripts/inference/web_ui.py --port 8000

# CLI Tool
python scripts/inference/l4d2_codegen.py generate "Your prompt"

# REST API
python scripts/inference/copilot_server_openai.py --port 8000
```

**Cost**: ~$0.002-0.004 per generation

### 3. Cloud Training on Vultr

```bash
# SSH to Vultr GPU instance
ssh root@<VULTR_IP>

# Clone and setup
git clone https://github.com/RazonIn4K/L4D2-AI-Architect
cd L4D2-AI-Architect
./setup.sh

# For A100 (recommended)
pip install flash-attn --no-build-isolation

# Start training
python scripts/training/train_unsloth.py \
    --config configs/unsloth_config_a100.yaml
```

**Training Time**: ~2 hours on A100 for 1010 examples

### GPU Recommendations

| GPU | VRAM | Batch Size | Training Time | Cost/Hour |
|-----|------|------------|---------------|-----------|
| A40 | 48GB | 4 | ~3 hours | $1.50 |
| A100 | 40GB | 8 | ~1.5 hours | $2.00 |
| GH200 | 96GB | 16 | ~1 hour | $3.00 |
| L40S | 48GB | 4 | ~2.5 hours | $1.50 |

---

## Usage Examples

### Web UI

```bash
python scripts/inference/web_ui.py --port 8000
# Navigate to http://localhost:8000
```

Features:
- Preset prompts for common plugins
- Syntax highlighting
- Cost tracking
- Chat mode for iterative development

### CLI Tool

```bash
# Single generation
python scripts/inference/l4d2_codegen.py generate \
    "Write a plugin that announces Tank spawns with health"

# Interactive chat
python scripts/inference/l4d2_codegen.py chat

# Batch processing
python scripts/inference/l4d2_codegen.py batch prompts.txt --output generated/

# Cost estimation
python scripts/inference/l4d2_codegen.py estimate prompts.txt
```

### REST API

```bash
# Start server
python scripts/inference/copilot_server_openai.py --port 8000

# Make requests
curl -X POST http://localhost:8000/v1/complete \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a function to heal all survivors",
    "temperature": 0.1,
    "max_tokens": 1024
  }'
```

### Python SDK

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

## Training Data Format

### SourcePawn (ChatML)

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert SourcePawn developer for L4D2 SourceMod plugins..."
    },
    {
      "role": "user",
      "content": "Write a function that heals all survivors"
    },
    {
      "role": "assistant",
      "content": "public void HealAllSurvivors()\n{\n    for (int i = 1; i <= MaxClients; i++)\n    {\n        if (IsClientInGame(i) && GetClientTeam(i) == 2 && IsPlayerAlive(i))\n        {\n            SetEntityHealth(i, 100);\n            SetEntProp(i, Prop_Send, \"m_currentReviveCount\", 0);\n        }\n    }\n}"
    }
  ]
}
```

### VScript (ChatML)

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert L4D2 VScript developer..."
    },
    {
      "role": "user",
      "content": "Create a DirectorOptions table that spawns more special infected"
    },
    {
      "role": "assistant",
      "content": "DirectorOptions <-\n{\n    PreferredMobDirection = SPAWN_ANYWHERE\n    MobSpawnMinTime = 30\n    MaxSpecials = 8\n    SpecialRespawnInterval = 15\n    SmokerLimit = 2\n    HunterLimit = 2\n}"
    }
  ]
}
```

---

## Project Structure

```
L4D2-AI-Architect/
├── configs/                        # Configuration files
│   ├── unsloth_config.yaml        # Default training config
│   ├── unsloth_config_a100.yaml   # A100-optimized config
│   ├── unsloth_config_gh200.yaml  # GH200-optimized config
│   └── director_config.yaml       # AI Director config
├── data/
│   ├── raw/                       # Scraped data (gitignored)
│   ├── processed/                 # Training datasets
│   │   ├── combined_train.jsonl   # All training data
│   │   ├── sourcepawn_train.jsonl # SourcePawn only
│   │   └── vscript_train.jsonl    # VScript only
│   ├── anti_patterns/             # Contrastive examples
│   ├── openai_finetune/           # OpenAI-formatted data
│   └── training_logs/             # TensorBoard logs
├── docs/                          # Documentation
│   ├── ARCHITECTURE.md            # Full architecture guide
│   ├── TRAINING_GUIDE.md          # Training instructions
│   ├── VULTR_A100_TRAINING.md     # A100 deployment
│   └── RL_TRAINING_ROADMAP.md     # RL pipeline docs
├── exports/                       # Exported models
│   └── l4d2-v10plus/
│       └── gguf/                  # GGUF + Ollama Modelfile
├── model_adapters/                # Trained LoRA adapters (gitignored)
├── scripts/
│   ├── scrapers/                  # Data collection
│   │   ├── scrape_github_plugins.py
│   │   └── scrape_valve_wiki.py
│   ├── training/                  # Training scripts
│   │   ├── prepare_dataset.py     # Data preparation
│   │   ├── train_unsloth.py       # Local QLoRA training
│   │   ├── start_openai_finetune.py # OpenAI fine-tuning
│   │   └── export_gguf_cpu.py     # GGUF export
│   ├── inference/                 # Inference tools
│   │   ├── web_ui.py              # Web interface
│   │   ├── l4d2_codegen.py        # CLI tool
│   │   ├── copilot_server_openai.py # API server
│   │   └── copilot_cli.py         # Legacy CLI
│   ├── rl_training/               # Reinforcement learning
│   │   ├── mnemosyne_env.py       # Gymnasium environment
│   │   └── train_ppo.py           # PPO training
│   ├── director/                  # AI Director
│   │   ├── director.py            # Main director logic
│   │   ├── policy.py              # Decision policies
│   │   └── bridge.py              # Game communication
│   ├── evaluation/                # Testing & validation
│   │   └── automated_test.py      # Test battery
│   └── utils/                     # Utilities
│       └── security.py            # Security wrappers
├── tests/                         # Unit tests
├── requirements.txt
├── setup.sh                       # One-command setup
├── activate.sh                    # Environment activation
├── run_scraping.sh                # Data collection
├── run_training.sh                # Training launcher
└── README.md
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 16GB | 32GB |
| GPU (Training) | - | A40/A100/GH200 |
| GPU (Inference) | - | Any (Ollama uses CPU) |
| Storage | 50GB | 100GB SSD |

---

## Contributing

We welcome contributions! Please follow these steps:

### Getting Started

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/L4D2-AI-Architect
cd L4D2-AI-Architect

# Install dev dependencies
pip install -r requirements.txt
pip install black flake8 pytest

# Create feature branch
git checkout -b feature/amazing-feature
```

### Code Standards

```bash
# Format code
black scripts/

# Run linter
flake8 scripts/

# Run tests
pytest tests/ -v
```

### Pull Request Process

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run linting and tests
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Open Pull Request

### Areas for Contribution

- **Data Quality**: Add more high-quality training examples
- **New Anti-patterns**: Identify and add more L4D2-specific corrections
- **RL Pipeline**: Help complete the Mnemosyne SourceMod plugin
- **Documentation**: Improve guides and examples
- **Testing**: Add unit tests and integration tests

---

## Roadmap

### Completed

- [x] SourcePawn code generation (V13 model)
- [x] VScript/Squirrel support
- [x] Anti-pattern training for L4D2 API correctness
- [x] Web UI with presets
- [x] CLI tool with batch processing
- [x] REST API server
- [x] Ollama/GGUF export
- [x] Quality filtering pipeline

### In Progress

- [ ] VSCode extension for SourcePawn
- [ ] Compile-time validation
- [ ] RL bot agent training
- [ ] AI Director hybrid mode

### Planned

- [ ] Hugging Face model hosting
- [ ] Real-time code completion API
- [ ] Plugin testing framework
- [ ] Multi-language support (more Source engine games)
- [ ] Community plugin repository

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `OPENAI_API_KEY not set` | `export OPENAI_API_KEY="sk-..."` |
| `openai package not found` | `pip install openai` |
| Wrong L4D2 APIs in output | Use V7+ model with enhanced system prompt |
| High variance in output | Lower temperature to 0.1 |
| Empty responses | Check API key permissions |
| Ollama model not found | Run `ollama create` with Modelfile |
| CUDA out of memory | Reduce batch size in config |

---

## Credits and Acknowledgments

### Technologies

- **[Unsloth](https://github.com/unslothai/unsloth)** - 2x faster QLoRA fine-tuning
- **[OpenAI](https://openai.com)** - GPT-4o-mini fine-tuning
- **[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)** - PPO/DQN implementations
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** - GGUF export for local inference
- **[Ollama](https://ollama.ai)** - Local model serving
- **[FastAPI](https://fastapi.tiangolo.com)** - API server framework

### Data Sources

- **[AlliedModders](https://alliedmods.net)** - SourceMod plugin repository
- **[Valve Developer Wiki](https://developer.valvesoftware.com)** - VScript documentation
- **[GitHub](https://github.com)** - Open source L4D2 plugins

### Special Thanks

- The SourceMod community for creating thousands of quality plugins
- Valve for the amazing Source engine and modding support
- The L4D2 modding community for inspiration

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{l4d2_ai_architect,
  title = {L4D2-AI-Architect: AI Models for Left 4 Dead 2 Modding},
  author = {RazonIn4K},
  year = {2026},
  url = {https://github.com/RazonIn4K/L4D2-AI-Architect}
}
```

---

<div align="center">

**Built for the L4D2 modding community**

[Report Bug](https://github.com/RazonIn4K/L4D2-AI-Architect/issues) | [Request Feature](https://github.com/RazonIn4K/L4D2-AI-Architect/issues) | [Documentation](docs/ARCHITECTURE.md)

</div>
