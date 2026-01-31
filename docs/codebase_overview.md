# L4D2-AI-Architect Codebase Overview

This document provides a detailed breakdown of the file structure and the role of each file in the repository.

## Root Directory

| File / Directory      | Role                                                                                                                                                                         |
| :-------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| `README.md`           | Main project documentation, setup guides, and feature overview.                                                                                                              |
| `GEMINI.md`           | Context file for AI assistants (like Gemini/Claude) to understand the project.                                                                                               |
| `AGENTS.md`           | Specific instructions or context for AI agents working on this codebase.                                                                                                     |
| `QUICKSTART.md`       | Quick setup instructions for new users.                                                                                                                                      |
| `QUICK_DEPLOY.md`     | Deployment guide (likely for inference services).                                                                                                                            |
| `requirements.txt`    | Python dependencies required for the project.                                                                                                                                |
| `pyproject.toml`      | Build system requirements and configuration (mostly for tools like Ruff/Black).                                                                                              |
| `setup.py`            | Package installation script (makes the project installable).                                                                                                                 |
| `setup.sh`            | Main shell script to initialize the environment (installs venv, dependencies).                                                                                               |
| `activate.sh`         | Helper script to quickly activate the virtual environment.                                                                                                                   |
| `LICENSE`             | MIT License file.                                                                                                                                                            |
| `l4d2_ai.py`          | **CLI Entry Point**: Main command-line interface for the L4D2 AI. Handles actions like generating code (`generate`), training (`train`), and evaluating models (`evaluate`). |
| `visual_demo.py`      | **Web UI Demo**: Launches a Gradio web interface to interactively generate SourcePawn scripts and test prompts.                                                              |
| `validate_scripts.py` | **Syntax Validator**: Checks generated SourcePawn scripts for syntax errors using `spcomp` (SourcePawn Compiler).                                                            |
| `preflight_check.py`  | **Environment Checker**: Verifies system requirements (CUDA, GPU memory, dataset existence) before running expensive operations.                                             |     |

## Scripts Directory (`scripts/`)

Contains all executable logic, categorized by function.

### `scripts/scrapers/`

_Data Collection_
| File | Role |
| :--- | :--- |
| `scrape_github_plugins.py` | Scrapes SourceMod plugins from GitHub repositories. |
| `scrape_valve_wiki.py` | **Wiki Scraper**: Crawls the Valve Developer Community Wiki to extract VScript and SourcePawn API documentation, converting it into training examples. |

### `scripts/training/`

_Model Fine-Tuning (Unsloth/QLoRA)_
| File | Role |
| :--- | :--- |
| `train_unsloth.py` | **Fine-Tuning Engine**: Uses Unsloth/QLoRA to fine-tune Llama 3/Mistral models on the L4D2 dataset. Handles model loading, training loop, and saving adapters. |
| `prepare_dataset.py` | **Data Formatter**: Cleans and organizes scraped data (GitHub plugins + Wiki docs) into the specific ChatML JSONL format required for model fine-tuning. |
| `export_model.py` | Exports the trained LoRA adapters to GGUF format for use in Ollama. |
| `analyze_dataset.py` | Provides statistics and insights about the training dataset. |
| `merge_model.py` | Merges LoRA adapters back into the base model. |
| (Various `test_*.py`) | Unit tests for the training pipeline. |

### `scripts/rl_training/`

_Reinforcement Learning (Bot Agents)_
| File | Role |
| :--- | :--- |
| `train_ppo.py` | **RL Trainer**: Implementation of the PPO algorithm to train agents in a simulated L4D2 environment. Handles reward calculation, policy updates, and checkpointing. |
| `l4d2_env.py` | OpenAI Gym environment wrapper for Left 4 Dead 2. |
| `rcon_interface.py` | Handles communication with the L4D2 server via RCON. |
| `game_state.py` | Parses game state from the server for the RL agent. |

### `scripts/inference/`

_Using the Model_
| File | Role |
| :--- | :--- |
| `copilot_cli.py` | **Interactive Interface**: CLI chat interface for asking the model coding questions. |
| `copilot_server.py` | **API Server**: Fast implementation of an API server to serve the model. |
| `l4d2_codegen.py` | **Code Generator**: Specialized utility for generation of L4D2-specific code constructs. |
| `plugin_wizard.py` | **Wizard Tool**: Interactive wizard to scaffold new SourceMod plugins step-by-step. |
| `web_ui.py` | **Web Interface**: A browser-based UI for interacting with the model (likely Gradio/Streamlit based). |

### `scripts/evaluation/`

_Testing Model Quality_
| File | Role |
| :--- | :--- |
| `evaluate_model.py` | Runs benchmarks or test prompts to score model performance. |

### `scripts/bugfixes/`

_Maintenance scripts_
| File | Role |
| :--- | :--- |
| `fix_fine_tuning.py` | Fixes formatting issues in JSONL training files for OpenAI/Unsloth. |
| `upload_fixed_training.sh` | Uploads the corrected dataset to the training target. |

### `scripts/utils/`

_Helpers_
| File | Role |
| :--- | :--- |
| `logger.py` | Centralized logging configuration. |
| (Other utils) | Miscellaneous helper functions. |

## Configuration (`configs/`)

| File                  | Role                                                     |
| :-------------------- | :------------------------------------------------------- |
| `unsloth_config.yaml` | Main config for training (hyperparameters, model paths). |
| `rl_config.yaml`      | Hyperparameters for Reinforcement Learning (PPO).        |

## Data Directory (`data/`)

| Directory        | Role                                                             |
| :--------------- | :--------------------------------------------------------------- |
| `raw/`           | Original scraped files (SourcePawn code, Wiki HTML).             |
| `processed/`     | Cleaned JSONL files ready for training (`combined_train.jsonl`). |
| `training_logs/` | TensorBoard/WandB logs tracking training progress.               |
| `results/`       | Output metrics and evaluation reports.                           |

## Documentation (`docs/`)

Contains deep-dive guides.

- `FINE_TUNING_FIX.md`: Documentation on the dataset formatting fix.
- (Other guides): Project-specific documentation.

---

> [!NOTE]
> This overview is based on the current file structure. As the project evolves, new files may be added. Rerun this analysis if significantly new components are introduced.
