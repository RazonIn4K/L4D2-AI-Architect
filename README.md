# L4D2-AI-Architect

## Project Overview

**L4D2-AI-Architect** is a comprehensive AI project dedicated to revolutionizing Left 4 Dead 2 modding and gameplay through machine learning. It focuses on three main pillars:

1.  **SourcePawn/VScript Copilot:** A fine-tuned Large Language Model (LLM) utilizing Unsloth/QLoRA to generate high-quality code for L4D2 plugins and VScripts.
2.  **RL Bot Agents:** Reinforcement Learning agents (PPO/DQN) trained via the Mnemosyne framework to act as intelligent bots.
3.  **AI Director 2.0:** Advanced generation of dynamic gameplay sequences.

This repository leverages **Python 3.10+**, **PyTorch**, **Unsloth** for efficient LLM training, and **Stable-Baselines3** for reinforcement learning.

## 📚 Documentation

- **[Codebase Overview](docs/codebase_overview.md)**: Detailed map of all files and scripts.
- **[Server Strategy](docs/server_strategy.md)**: High-level architectural strategy for server/AI integration.
- **[Server Setup Guide](docs/server_setup_guide.md)**: Step-by-step instructions for Local and Cloud (DigitalOcean) deployment.

## 🚀 Quick Start (Local Data Collection)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/RazonIn4K/L4D2-AI-Architect.git
    cd L4D2-AI-Architect
    ```
2.  **Set up Virtual Environment:**

3.  **Set up Virtual Environment:**

    ```bash
    make setup
    source venv/bin/activate
    ```

4.  **Collect Data:**

5.  **Collect Data:**

    ```bash
    # Scrape GitHub SourceMod plugins
    make scrape-github

    # Scrape Valve Wiki
    make scrape-wiki
    ```

6.  **Prepare Dataset:**
    ```bash
    make process-data
    ```

## ☁️ Cloud / Server Deployment

For model training (Linux/GPU required) and Live Server hosting, refer to the **[Server Setup Guide](docs/server_setup_guide.md)**.

**Key Requirements:**

- **Training**: NVIDIA GPU (Linux)
- **Game Server**: High Single-Core CPU (e.g., DigitalOcean Premium CPU-Optimized)
