# Repository Guidelines

## Project Structure & Module Organization
This repository is organized around data collection, fine-tuning, and RL training workflows.
- `scripts/` holds runnable entry points: `scrapers/`, `training/`, and `rl_training/` plus shared utilities.
- `configs/` contains YAML training configs such as `unsloth_config.yaml`.
- `data/` stores `raw/`, `processed/`, and `training_logs/` outputs (gitignored, with `.gitkeep` placeholders).
- `model_adapters/` is the output location for LoRA adapters (gitignored).
- `docs/` houses operational guidance like `TRAINING_GUIDE.md`.
- `tests/` and `notebooks/` exist for validation scripts and experiments; keep ad-hoc work here.

## Build, Test, and Development Commands
- `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt` sets up the local environment.
- `./activate.sh` activates the venv and sets `PYTHONPATH` for local runs.
- `./run_scraping.sh` runs GitHub/wiki scrapers and dataset prep; tune with `MAX_REPOS=200 MAX_PAGES=100 ./run_scraping.sh`.
- `./run_training.sh --config configs/unsloth_config.yaml` starts Unsloth fine-tuning.
- `python scripts/rl_training/train_ppo.py --episodes 10000` launches PPO training.
- `python scripts/training/export_model.py --format gguf --quantize q4_k_m` exports a GGUF model.

## Coding Style & Naming Conventions
- Python 3.10+ code uses 4-space indentation and module-level docstrings.
- Prefer `black` and `flake8` (listed in `requirements.txt`) before submitting changes.
- Use `snake_case` for functions/variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.

## Testing Guidelines
- `tests/` is present but currently empty; no test framework is configured.
- If you add tests, place them under `tests/`, adopt `test_*.py` naming, and add `pytest` to `requirements.txt` with a `python -m pytest` invocation documented here.

## Commit & Pull Request Guidelines
- Git history only includes "Initial project scaffold"; no enforced commit convention yet. Use short, imperative messages (e.g., "Add dataset prep options").
- PRs should explain the change, list commands run, and note any data/model artifacts produced or omitted from version control.

## Security & Configuration Tips
- Never commit tokens or credentials; use environment variables (e.g., `GITHUB_TOKEN`) for scrapers.
- Large artifacts belong in `data/` or `model_adapters/` (both gitignored); only track configs and code.
