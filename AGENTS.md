# Repository Guidelines

## Project Structure & Module Organization
This repository spans data collection, fine-tuning (local + OpenAI), inference services, RL training, evaluation, and deployment tooling.
- `scripts/` holds runnable entry points across `scrapers/`, `training/`, `rl_training/`, `inference/`, `director/`, `evaluation/`, and `utils/` (security, data filtering, preflight, deployment).
- `configs/` contains YAML configs for Unsloth and Director behavior (`unsloth_config*.yaml`, `director_config.yaml`).
- `data/` stores raw/processed datasets, evaluation artifacts, prompts, and `l4d2_server/` scaffolding; most files are gitignored.
- `model_adapters/` is the output location for LoRA adapters (gitignored).
- `docs/` includes training guides, deployment runbooks (Vultr/GH200/Runpod), evaluation reports, and security references.
- `setup.sh`, `activate.sh`, `run_scraping.sh`, `run_training.sh` are the primary top-level helpers.
- `notebooks/` is for experiments; `tests/` is currently empty.

## Build, Test, and Development Commands
- `./setup.sh` performs full environment setup (venv, deps, directories).
- `./activate.sh` activates the venv and sets `PYTHONPATH` for local runs.
- `./run_scraping.sh` runs GitHub/wiki scrapers and dataset prep; tune with `MAX_REPOS=500 MAX_PAGES=200 ./run_scraping.sh`.
- `./run_training.sh --config configs/unsloth_config.yaml` starts Unsloth fine-tuning (swap in `configs/unsloth_config_a100.yaml`, `configs/unsloth_config_gh200.yaml`, or `configs/unsloth_config_qwen.yaml`).
- `python scripts/training/start_openai_finetune.py --version v5` and `python scripts/training/check_finetune_status.py --version v5` manage OpenAI fine-tunes (swap `v5` for `v6`/`v7` when needed).
- `python scripts/inference/copilot_server.py --model-path model_adapters/...` or `python scripts/inference/copilot_server_openai.py` start inference services; `python scripts/inference/copilot_cli.py complete --prompt "..."` is the CLI client.
- `python scripts/rl_training/train_ppo.py --episodes 10000` launches PPO training.
- `python scripts/evaluation/automated_test.py --model local` or `python scripts/evaluation/evaluate_models.py --compare` runs evaluation suites.
- `python scripts/utils/preflight_check.py` and `./scripts/utils/prepare_deployment.sh --host <IP>` help with deployment validation/packaging.

## Coding Style & Naming Conventions
- Python 3.10+ code uses 4-space indentation and module-level docstrings.
- Prefer `black` and `flake8` (listed in `requirements.txt`) before submitting changes.
- Use `snake_case` for functions/variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.

## Testing Guidelines
- `tests/` exists but is empty; most validation happens via `scripts/evaluation/`.
- For quick smoke checks, use `python scripts/inference/test_lora.py` or `python scripts/evaluation/validate_generated_code.py validate <path>`.
- If you add unit tests, place them under `tests/` with `test_*.py` naming and document `pytest` usage.

## Commit & Pull Request Guidelines
- Git history only includes "Initial project scaffold"; no enforced commit convention yet. Use short, imperative messages (e.g., "Add dataset prep options").
- PRs should explain the change, list commands run, and note any data/model artifacts produced or omitted from version control.

## Security & Configuration Tips
- Never commit tokens or credentials; use environment variables (`GITHUB_TOKEN`, `OPENAI_API_KEY`) or Doppler for OpenAI scripts.
- Scrapers assume HTTPS and validate allowed domains; update `scripts/utils/security.py` allowlists if you add new sources.
- For new HTTP calls, validate URLs with `validate_url`; for file writes/reads use `safe_path`/`safe_read_text`/`safe_write_text`.
- Inference servers restrict hostnames/ports; update allowlists in `scripts/inference` if you expand targets.
- Large artifacts belong in `data/` or `model_adapters/` (gitignored); track only configs, code, and docs.
