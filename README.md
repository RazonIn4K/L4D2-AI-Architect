# 🎮 L4D2-AI-Architect

> **Train specialized AI models for Left 4 Dead 2 modding, bot control, and Director manipulation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (GPU recommended, not required)
- L4D2 dedicated server with SourceMod

### One-Command Setup
```bash
git clone https://github.com/RazonIn4K/L4D2-AI-Architect
cd L4D2-AI-Architect
./setup.sh
```

### Manual Setup
```bash
# Create environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Unsloth (GPU only)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Copy configuration
cp .env.example .env
# Edit .env with your paths
```

---

## 📖 Usage Guide

### 1. Data Collection
```bash
# Set GitHub token
export GITHUB_TOKEN="your_token_here"

# Collect SourcePawn plugins and wiki documentation
./run_scraping.sh

# Or manually:
python scripts/scrapers/scrape_github_plugins.py --max-repos 500
python scripts/scrapers/scrape_valve_wiki.py --max-pages 200
```

### 2. Train SourcePawn Copilot
```bash
# Prepare dataset
python scripts/training/prepare_dataset.py \
    --input data/raw \
    --output data/processed

# Start training
./run_training.sh --config configs/unsloth_config.yaml

# Export model
python scripts/training/export_model.py \
    --model model_adapters/l4d2-code-lora \
    --format gguf \
    --quantize q4_k_m
```

### 3. Deploy Copilot Service
```bash
# Start inference server
python scripts/inference/copilot_server.py \
    --model-path model_adapters/l4d2-code-lora

# Use CLI tool
python scripts/inference/copilot_cli.py complete \
    --prompt "public void OnPluginStart()" \
    --language sourcepawn

# Interactive chat
python scripts/inference/copilot_cli.py chat
```

### 4. Train RL Bot Agents
```bash
# Start L4D2 server with SourceMod plugin
# See data/l4d2_server/ for plugin installation

# Train PPO agent
python scripts/rl_training/train_ppo.py \
    --episodes 10000 \
    --save-path model_adapters/ppo_agent

# Test trained agent
python scripts/rl_training/test_agent.py \
    --model model_adapters/ppo_agent
```

### 5. Run AI Director
```bash
# Start director (rule-based mode)
python scripts/director/director.py \
    --mode rule \
    --config configs/director_config.yaml

# Or hybrid mode
python scripts/director/director.py \
    --mode hybrid \
    --host localhost \
    --port 27050
```

---

## 🎮 Integration with L4D2

### Server Setup
1. Install L4D2 dedicated server
2. Install SourceMod and Metamod: Source
3. Compile and load `l4d2_ai_bridge.sp`
4. Configure `server.cfg`:

```cfg
// Enable AI bridge
sm_ai_connect 127.0.0.1 27050
sm_ai_director 1

// Bot settings
sb_version 1
sb_allbot_team 2
```

### Connecting Components
```
┌─────────────┐    TCP/UDP    ┌──────────────┐
│   L4D2      │◄────────────►│ AI Director  │
│   Server    │              │   Service    │
└──────▲──────┘              └──────▲───────┘
       │                           │
       │ JSON Commands             │ Game State
       │                           │
┌──────┴──────┐              ┌──────┴───────┐
│ RL Agents   │              │ Copilot     │
│ (Python)    │              │ Inference   │
└─────────────┘              └──────────────┘
```

---

## 🧪 Testing

### Unit Tests
```bash
# Run all tests
python -m pytest tests/

# Test individual components
python tests/test_director.py
python tests/test_copilot.py
python tests/test_bridge.py
```

### Integration Tests
```bash
# Test with mock server
python scripts/director/bridge.py --mock

# Test copilot without GPU
python scripts/inference/copilot_cli.py serve \
    --model-path models/base/mistral-7b-instruct-v0.3-bnb-4bit
```

---

## 📊 Performance

### Hardware Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 16GB | 32GB |
| GPU | - | RTX 4090 / A40 |
| Storage | 50GB | 100GB SSD |

### Benchmarks
- **Data Collection**: 500 repos in ~30 minutes
- **Fine-tuning**: 7B model on A40 (48GB) - ~2 hours
- **Inference**: 100ms latency on RTX 4090
- **RL Training**: 10K episodes in ~4 hours

---

## 🔧 Configuration

### Key Files
- `.env` - Environment variables and paths
- `configs/unsloth_config.yaml` - Model training settings
- `configs/director_config.yaml` - AI Director behavior
- `data/l4d2_server/cfg/server.cfg` - Game server settings

### Environment Variables
```bash
# L4D2 paths
L4D2_INSTALL_PATH=/path/to/l4d2
SRCDS_PATH=/path/to/srcds

# Network
BRIDGE_HOST=localhost
BRIDGE_PORT=27050

# GPU
GPU_ID=0
MIXED_PRECISION=true

# Services
DIRECTOR_ENABLED=true
COPLOT_PORT=8000
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Open Pull Request

### Development Setup
```bash
# Install dev dependencies
pip install -r requirements.txt
pip install black flake8 pytest

# Run linting
black scripts/
flake8 scripts/

# Run tests
pytest tests/ -v
```

## 📁 Repository Structure

```
L4D2-AI-Architect/
├── data/
│   ├── raw/                    # Raw scrapes (gitignored)
│   ├── processed/              # Cleaned JSONL files (gitignored)
│   └── training_logs/          # TensorBoard logs
├── scripts/
│   ├── scrapers/               # Data collection scripts
│   │   ├── scrape_sourcemod.py
│   │   ├── scrape_valve_wiki.py
│   │   └── scrape_github_plugins.py
│   ├── training/               # LLM fine-tuning
│   │   ├── train_unsloth.py
│   │   ├── prepare_dataset.py
│   │   └── export_model.py
│   ├── rl_training/            # Reinforcement learning
│   │   ├── mnemosyne_env.py
│   │   ├── train_ppo.py
│   │   └── train_dqn.py
│   ├── inference/              # Model inference
│   │   └── run_inference.py
│   └── utils/                  # Utilities
│       ├── vultr_setup.sh
│       └── download_artifacts.sh
├── configs/                    # Training configurations
│   ├── unsloth_config.yaml
│   └── rl_config.yaml
├── docs/                       # Documentation
│   ├── VULTR_SETUP.md
│   ├── TRAINING_GUIDE.md
│   └── DATA_COLLECTION.md
├── notebooks/                  # Jupyter notebooks
│   └── training_demo.ipynb
├── model_adapters/             # Saved LoRA adapters (gitignored)
├── tests/                      # Test scripts
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Clone to Vultr GPU Instance

```bash
# SSH into your Vultr instance
ssh root@<VULTR_IP>

# Clone the repository
git clone https://github.com/YOUR_USERNAME/L4D2-AI-Architect.git
cd L4D2-AI-Architect

# Run setup script
chmod +x scripts/utils/vultr_setup.sh
./scripts/utils/vultr_setup.sh
```

### 2. Collect Training Data

```bash
# Scrape SourceMod plugins from GitHub
python scripts/scrapers/scrape_github_plugins.py

# Scrape Valve Developer Wiki
python scripts/scrapers/scrape_valve_wiki.py

# Prepare dataset
python scripts/training/prepare_dataset.py
```

### 3. Fine-Tune the Model

```bash
# Train with Unsloth (QLoRA)
python scripts/training/train_unsloth.py --config configs/unsloth_config.yaml

# Export to GGUF for local use
python scripts/training/export_model.py --format gguf --quantize q4_k_m
```

### 4. Train RL Agents (Optional)

```bash
# Requires L4D2 server with Mnemosyne plugin
python scripts/rl_training/train_ppo.py --episodes 10000
```

## 💰 Vultr Credit Optimization ($250 / 5 Days)

| Task | GPU | Hours | Cost |
|------|-----|-------|------|
| Data Collection | CPU (8GB) | 4h | ~$0.50 |
| Dataset Prep | CPU (8GB) | 2h | ~$0.25 |
| QLoRA Fine-tune | A40/L40S | 12-24h | ~$20-40 |
| RL Training | L40S | 48-72h | ~$80-120 |
| Model Export | A40 | 2h | ~$3 |
| **Total Estimated** | | | **~$105-165** |

### Recommended Schedule

**Day 1:** Data collection + preprocessing (CPU instance)
**Day 2-3:** Fine-tune code model (A40/L40S)
**Day 3-5:** RL bot training (L40S)
**Day 5:** Export models + backup artifacts

## 📊 Dataset Sources

| Source | Content | ~Size |
|--------|---------|-------|
| GitHub sourcemod-plugins | SourcePawn code | 500+ plugins |
| AlliedModders forums | Tutorials, snippets | 1000+ posts |
| Valve Developer Wiki | VScript/Director docs | 200+ pages |
| L4D2 VPK scripts | Official .nut files | 100+ files |
| Your Mnemosyne repo | Custom code | 2000+ lines |

## 🔧 Key Technologies

- **[Unsloth](https://github.com/unslothai/unsloth)** - 2x faster QLoRA fine-tuning
- **[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)** - PPO/DQN implementations
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** - GGUF export for local inference
- **[Ollama](https://ollama.ai)** - Local model serving

## 📝 Training Data Format

### Code Fine-tuning (JSONL)

```json
{"messages": [{"role": "system", "content": "You are an expert SourcePawn developer..."}, {"role": "user", "content": "Write a function that heals all survivors"}, {"role": "assistant", "content": "public void HealAllSurvivors()\n{\n    for (int i = 1; i <= MaxClients; i++)\n    {\n        if (IsClientInGame(i) && GetClientTeam(i) == 2 && IsPlayerAlive(i))\n        {\n            SetEntityHealth(i, 100);\n            SetEntProp(i, Prop_Send, \"m_currentReviveCount\", 0);\n            SetEntProp(i, Prop_Send, \"m_isGoingToDie\", 0);\n        }\n    }\n}"}]}
```

### VScript Fine-tuning (JSONL)

```json
{"messages": [{"role": "system", "content": "You are an expert L4D2 VScript developer..."}, {"role": "user", "content": "Create a DirectorOptions table that spawns more special infected"}, {"role": "assistant", "content": "DirectorOptions <-\n{\n    PreferredMobDirection = SPAWN_ANYWHERE\n    MobSpawnMinTime = 30\n    MobSpawnMaxTime = 60\n    MaxSpecials = 8\n    SpecialRespawnInterval = 15\n    SmokerLimit = 2\n    HunterLimit = 2\n    BoomerLimit = 2\n    ChargerLimit = 2\n}"}]}
```

## 🎓 Model Architecture

### Code Model (SourcePawn Copilot)
- **Base:** Mistral-7B-Instruct or CodeLlama-7B
- **Method:** QLoRA (r=32, alpha=64)
- **Training:** ~1000 examples, 3 epochs
- **Output:** GGUF Q4_K_M (~4GB)

### RL Agent (Bot Controller)
- **Algorithm:** PPO (Stable-Baselines3)
- **State Space:** 32D (position, health, velocity, enemies, teammates)
- **Action Space:** 9 discrete (move, attack, heal, etc.)
- **Reward:** +10 kills, -100 death, +1 damage dealt

## 🔐 Security Notes

- Never commit API keys or tokens
- Model weights are gitignored
- Raw data is gitignored
- Use environment variables for secrets

## 📦 Saving Your Work

Before Vultr credits expire:

```bash
# Save LoRA adapter to HuggingFace
python scripts/training/export_model.py --push-to-hub YOUR_USERNAME/l4d2-sourcemod-lora

# Download GGUF locally
scp root@<VULTR_IP>:~/L4D2-AI-Architect/model_adapters/*.gguf ./

# Backup training logs
tar -czvf training_logs.tar.gz data/training_logs/
```

## 🤝 Integration with Mnemosyne

This project complements the [L4D2_Mnemosyne_Repository](link) by providing:

1. **Code Generation** - Generate SourcePawn plugins for Mnemosyne
2. **Trained Bot Policies** - PPO/DQN models for the Python controller
3. **Director Sequences** - AI-generated gameplay events

## 📄 License

MIT License - See [LICENSE](LICENSE)

---

**Built for the L4D2 modding community** 🧟‍♂️
