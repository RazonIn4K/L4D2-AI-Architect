# L4D2-AI-Architect – RunPod (A40 48GB) Training Guide

This guide is an **ops-focused, repo-accurate** walkthrough for running the current L4D2-AI-Architect training pipeline on a **RunPod A40 (48GB)**.

It complements `docs/TRAINING_GUIDE.md` (which covers the broader workflow) with RunPod-specific setup notes and a “best next steps” checklist.

---

## 1) Project Goal (What we’re training)

Fine-tune a code assistant for Left 4 Dead 2 modding that can:

- Generate **SourcePawn / SourceMod** plugins
- Generate **VScript / Squirrel** director logic (mutations, DirectorOptions, spawners)
- Feed into later RL / “AI Director” tooling in this repo

The current fine-tuning approach is **Unsloth QLoRA** on a dataset built from:

- GitHub SourceMod/SourcePawn repositories
- Valve Developer Wiki pages + embedded code snippets

---

## 2) What’s in the repo (current pipeline entrypoints)

### 2.1 Data collection

- `scripts/scrapers/scrape_github_plugins.py`
  - Outputs (default):
    - `data/raw/github_plugins/github_plugins.jsonl`
    - `data/raw/github_plugins/scrape_stats.json`

- `scripts/scrapers/scrape_valve_wiki.py`
  - Outputs (default):
    - `data/raw/valve_wiki/valve_wiki.jsonl` (this is what `prepare_dataset.py` reads)
    - `data/raw/valve_wiki/code_blocks.jsonl` (saved for convenience, not currently consumed)

Convenience wrapper:

- `./run_scraping.sh`
  - Activates `./venv` if present
  - Runs both scrapers
  - Runs dataset prep (`scripts/training/prepare_dataset.py`) with default settings

`run_scraping.sh` supports tuning scrape size via environment variables:

- `MAX_REPOS` (default `500`)
- `MAX_PAGES` (default `200`)

### 2.2 Dataset preparation

- `scripts/training/prepare_dataset.py`

Key CLI flags:

- `--input` (default: `data/raw`)
- `--output` (default: `data/processed`)
- `--format` (default: `unsloth`, choices: `unsloth`, `alpaca`, `both`)
- `--min-quality` (default: `0.4`)
- `--train-ratio` (default: `0.9`)

Outputs (in `data/processed/`):

- `combined_train.jsonl`
- `combined_val.jsonl`
- `sourcepawn_train.jsonl`
- `sourcepawn_val.jsonl`
- `vscript_train.jsonl`
- `vscript_val.jsonl`
- `dataset_stats.json`

Notes:

- The “Unsloth” format is **ChatML-style**: each row contains `messages: [{role, content}, ...]`.
- The script uses quality heuristics (`QUALITY_PATTERNS`) and filters aggressively (short/huge files, low quality, missing instruction signals).
- The **combined** dataset is formatted with the **SourcePawn system prompt** for all examples (including VScript/Squirrel examples). If you want the combined dataset to preserve language-specific system prompts, a good next step is to adjust `prepare_dataset.py` to pick the system prompt per-example based on `ex.language`, or train separate adapters (SourcePawn-only + VScript-only).

### 2.3 Training

- `scripts/training/train_unsloth.py`
- Wrapper: `./run_training.sh`

Key behaviors:

- **Hard GPU check:** exits if `torch.cuda.is_available()` is false.
- Uses `configs/unsloth_config.yaml` by default.
- Saves artifacts under `model_adapters/<output_dir>/`:
  - `final/` (model + tokenizer)
  - `lora_adapter/` (LoRA-only adapter export)
  - `training_info.json` (metadata)
  - `checkpoint-*` directories (Trainer checkpoints)

CLI flags supported:

- `--config`
- `--model`
- `--dataset`
- `--epochs`
- `--batch-size`
- `--lr`
- `--resume`
- `--test`
- `--test-only`

Important path note:

- In `configs/unsloth_config.yaml`, `data.train_file` and `data.val_file` are interpreted as **relative to `data/processed/`**.
  - Use `combined_train.jsonl` (not `data/processed/combined_train.jsonl`).
  - If you pass `--dataset`, prefer **just the filename** (e.g. `--dataset sourcepawn_train.jsonl`) unless you’re providing an absolute path inside the repo.

### 2.4 Export

- `scripts/training/export_model.py`

Common export example:

- GGUF export:
  - `--format gguf`
  - `--quantize q4_k_m` (default)

The export script can also:

- Create an Ollama `Modelfile`
- Install into Ollama (`--install-ollama`)
- Push merged/lora formats to HF Hub (`--push-to-hub`)

---

## 3) RunPod A40 setup (recommended)

### 3.1 Pod template

Use a RunPod template that already includes:

- NVIDIA drivers + CUDA userspace
- A working `nvidia-smi`

Verify:

```bash
nvidia-smi
```

### 3.2 Clone the repo

```bash
git clone <YOUR_REPO_URL>
cd L4D2-AI-Architect
```

If you want a single-script bootstrap, `./setup.sh` exists, but note:

- It includes **interactive prompts** (e.g. optional base model download).
- It includes **game server / SourceMod** setup steps that are usually not needed for a training-only RunPod pod.

For RunPod training, the manual venv + CUDA torch steps below are typically the most predictable.

### 3.3 Python environment (venv)

Create and activate the venv:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

#### Install CUDA-enabled PyTorch

This repo does **not** pin/install `torch` in `requirements.txt`.

You must ensure `torch` inside the venv is CUDA-enabled:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)" || true
```

If `torch` is missing or CUDA is false, install the correct CUDA wheel for your image (example for CUDA 11.8):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

If your image has a newer CUDA runtime, install the matching PyTorch wheel instead (the key requirement is that `torch.cuda.is_available()` is `True` inside the venv).

Then verify again:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

#### Install repo dependencies

```bash
pip install -r requirements.txt
```

Install Unsloth:

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

Notes:

- If you use Weights & Biases, set `WANDB_API_KEY` (see `.env.example`).
- If you scrape GitHub heavily, set `GITHUB_TOKEN` to avoid strict rate limits.

---

## 4) Recommended RunPod workflow (A40)

### Step 1: (Optional) Set env vars

```bash
export GITHUB_TOKEN="<your token>"  # optional, recommended
```

### Step 2: Collect raw data (and optionally prepare the dataset)

Fast path:

```bash
./run_scraping.sh
```

`run_scraping.sh` runs:

- `scripts/scrapers/scrape_github_plugins.py`
- `scripts/scrapers/scrape_valve_wiki.py`
- `scripts/training/prepare_dataset.py` (defaults)

Or run each scraper explicitly (gives you output control):

```bash
python scripts/scrapers/scrape_github_plugins.py --max-repos 500
python scripts/scrapers/scrape_valve_wiki.py --max-pages 200
```

### Step 3: Prepare dataset

`run_scraping.sh` already runs this, but if you want to tune quality thresholds:

```bash
python scripts/training/prepare_dataset.py \
  --input data/raw \
  --output data/processed \
  --format unsloth \
  --min-quality 0.4
```

Sanity-check the output:

- `data/processed/combined_train.jsonl` exists and is non-empty
- `data/processed/dataset_stats.json` looks reasonable

### Step 4: Train (A40 defaults)

```bash
./run_training.sh --config configs/unsloth_config.yaml
```

If you want a quick “smoke test” training run first:

- Reduce epochs via CLI:

```bash
./run_training.sh --config configs/unsloth_config.yaml --epochs 1
```

#### Resuming from a checkpoint

Trainer checkpoints are stored under:

- `model_adapters/<output_dir>/checkpoint-*`

Resume example:

```bash
./run_training.sh --config configs/unsloth_config.yaml --resume model_adapters/l4d2-code-lora/checkpoint-200
```

### Step 5: Monitor training

GPU:

```bash
watch -n 1 nvidia-smi
```

TensorBoard logs are written under:

- `data/training_logs/<output_dir>/`

Run TensorBoard (in the pod):

```bash
tensorboard --logdir data/training_logs --host 0.0.0.0 --port 6006
```

Then expose port `6006` in the RunPod UI.

### Step 6: Quick functional test

After training completes:

```bash
python scripts/training/train_unsloth.py --test-only model_adapters/l4d2-code-lora/final \
  --test "Write a SourceMod plugin that hooks player_hurt and prints attacker/victim."
```

---

## 5) Export + backup (before the pod is destroyed)

### 5.1 Export to GGUF

```bash
python scripts/training/export_model.py \
  --input model_adapters/l4d2-code-lora/final \
  --format gguf \
  --quantize q4_k_m
```

If you want exports somewhere predictable, set `--output` (example):

```bash
python scripts/training/export_model.py \
  --input model_adapters/l4d2-code-lora/final \
  --output exports/l4d2-code-lora \
  --format gguf \
  --quantize q4_k_m
```

### 5.2 What to copy off-pod

At minimum:

- `model_adapters/`
- `exports/` (if you used it)
- `data/processed/` (so you can reproduce training later)

---

## 6) Best next steps (practical roadmap)

### 6.1 Short-term (same pod)

- Run **one complete end-to-end pass**:
  - scrape -> prepare -> train -> test-only -> export
- Create a small, fixed **evaluation prompt set** (10–30 prompts) that covers:
  - SourcePawn plugin skeletons
  - event hooks + convars
  - director scripts (DirectorOptions)
  - mutation examples
- Use that prompt set after every run to spot regressions.

### 6.2 Medium-term (quality)

- Improve dataset quality:
  - Increase `--min-quality`
  - Review `data/processed/*_train.jsonl` for “bad instructions” and adjust heuristics in `prepare_dataset.py`
- Train **specialized adapters**:
  - SourcePawn-only (`sourcepawn_train.jsonl` / `sourcepawn_val.jsonl`)
  - VScript-only (`vscript_train.jsonl` / `vscript_val.jsonl`)
  - Compare against the combined adapter

### 6.3 Long-term (productization)

- Establish a “collect → evaluate → fix prompts → retrain” loop.
- Export to GGUF and integrate with a local runner (Ollama / llama.cpp), or wire it into `scripts/inference/`.

---

## 7) Troubleshooting quick hits

- If training exits with `CUDA not available!`:
  - Verify `torch` inside the venv is CUDA-enabled.
- If you hit CUDA OOM:
  - Lower `per_device_train_batch_size` (or pass `--batch-size`)
  - Reduce `model.max_seq_length` in `configs/unsloth_config.yaml`

---

## Reference

- General training workflow: `docs/TRAINING_GUIDE.md`
- Default training config: `configs/unsloth_config.yaml`
- Dataset prep: `scripts/training/prepare_dataset.py`
- Training: `scripts/training/train_unsloth.py`
- Export: `scripts/training/export_model.py`
