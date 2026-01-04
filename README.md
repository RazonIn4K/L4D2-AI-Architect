# 🎮 L4D2-AI-Architect

> **Train specialized AI models for Left 4 Dead 2 modding, bot control, and Director manipulation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## 🎯 Project Goals

1. **Fine-tune a SourcePawn/VScript Copilot** - Generate L4D2 plugin code, gamedata signatures, and bot logic
2. **Train RL Bot Agents** - PPO/DQN agents that learn survival strategies via the Mnemosyne framework
3. **AI Director 2.0** - Generate dynamic gameplay sequences that rival Valve's Director

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
