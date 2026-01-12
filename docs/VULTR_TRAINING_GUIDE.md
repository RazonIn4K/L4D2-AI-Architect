# L4D2-AI-Architect – Vultr Training Guide (1/2 A100 40GB)

This guide is tailored for running the **current repo training pipeline** on a Vultr Cloud GPU instance with **1/2 NVIDIA A100 (40GB VRAM)**.

It focuses on:

- Getting a working environment quickly on a Vultr VPS
- **Tuning Unsloth QLoRA settings** to make good use of a **40GB A100 slice**
- Correct, repo-accurate commands for scraping → dataset prep → training → export

Related docs:

- General training workflow: `docs/TRAINING_GUIDE.md`
- General Vultr deployment: `docs/VULTR_DEPLOYMENT.md`

---

## 1) Hardware context (what you can expect)

A **1/2 A100 (40GB)** is a great target for `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` fine-tuning:

- A100 generally has **higher training throughput** than A40/L40S.
- You have **40GB VRAM**, which is enough for 7B QLoRA with healthy batch sizes.
- If you tune batch/seq length correctly, you should see **high GPU utilization** while staying under VRAM.

---

## 2) Environment setup (Vultr VPS)

### Option A (recommended): Use the repo’s Vultr setup script

The repo includes a Vultr bootstrap script:

- `scripts/utils/vultr_setup.sh`

It:

- Installs system packages (`tmux`, `build-essential`, etc.)
- Verifies CUDA and drivers
- Installs CUDA-enabled PyTorch
- Installs Unsloth and training dependencies
- Sets up a localhost-bound TensorBoard systemd service

Recommended flow:

1. SSH in:

```bash
ssh root@<YOUR_VULTR_IP>
```

2. Verify GPU is visible:

```bash
nvidia-smi
```

3. Clone your repo into the expected path:

```bash
cd /root
git clone <YOUR_REPO_URL> L4D2-AI-Architect
cd /root/L4D2-AI-Architect
```

4. Run the setup script:

```bash
sudo bash scripts/utils/vultr_setup.sh
```

5. Activate the environment (repo root contains `activate.sh`):

```bash
source ./activate.sh
```

### Option B (manual): venv + CUDA torch + requirements + Unsloth

If you prefer manual control, do this instead:

```bash
cd /root/L4D2-AI-Architect
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip

# install CUDA torch matching your runtime (examples)
# CUDA 11.8:
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

python -m pip install -r requirements.txt
python -m pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

---

## 3) Data pipeline (scrape → prepare)

### 3.1 Scrape + prepare (fastest path)

From the repo root:

```bash
# Defaults: MAX_REPOS=500, MAX_PAGES=200
./run_scraping.sh
```

You can tune scrape volume:

```bash
MAX_REPOS=200 MAX_PAGES=100 ./run_scraping.sh
```

Outputs:

- GitHub scrape:
  - `data/raw/github_plugins/github_plugins.jsonl`
- Valve wiki scrape:
  - `data/raw/valve_wiki/valve_wiki.jsonl`
- Prepared datasets:
  - `data/processed/combined_train.jsonl`
  - `data/processed/combined_val.jsonl`
  - `data/processed/sourcepawn_train.jsonl`, `data/processed/vscript_train.jsonl`, etc.
  - `data/processed/dataset_stats.json`

### 3.2 Dataset preparation (explicit, with knobs)

If you want tighter control over quality filtering:

```bash
python scripts/training/prepare_dataset.py \
  --input data/raw \
  --output data/processed \
  --format unsloth \
  --min-quality 0.4
```

---

## 4) Training on A100 40GB (how to leverage the GPU)

### 4.1 Start with the default config (known-good)

Baseline training:

```bash
./run_training.sh --config configs/unsloth_config.yaml
```

Defaults (from `configs/unsloth_config.yaml`):

- `max_seq_length: 2048`
- `per_device_train_batch_size: 4`
- `gradient_accumulation_steps: 4`
- `bf16: true`

This is a safe first run to confirm the full pipeline works.

### 4.2 Then tune for 40GB (increase batch OR increase context)

You generally tune one of these first:

- **Throughput:** increase `per_device_train_batch_size` while keeping `max_seq_length` at 2048.
- **Context:** increase `max_seq_length` (e.g. 3072/4096) and reduce batch size if needed.

Practical tuning loop:

1. Start training.
2. In a second SSH session:

```bash
watch -n 1 nvidia-smi
```

3. If VRAM usage is well below ~30GB and GPU utilization is not pegged, increase batch size.

Recommended A100 40GB settings to try (edit a copy of the config):

**Profile A (throughput, good starting point):**

- `max_seq_length: 2048`
- `per_device_train_batch_size: 8`
- `gradient_accumulation_steps: 2`

This keeps effective batch size similar to baseline (8×2 = 16 vs 4×4 = 16) but increases parallelism.

**Profile B (more context):**

- `max_seq_length: 4096`
- `per_device_train_batch_size: 4`
- `gradient_accumulation_steps: 2`

This increases context while staying within 40GB more reliably than trying `4096` with batch `8`.

Implementation tip:

- Copy the config so you don’t lose the baseline:

```bash
cp configs/unsloth_config.yaml configs/unsloth_config_a100_40gb.yaml
```

Then edit `configs/unsloth_config_a100_40gb.yaml`.

### 4.3 Optional: enable A100-friendly training options

The training script reads an optional `advanced:` section (if present):

- `tf32` (torch matmul TF32)
- `dataloader_num_workers`
- `dataloader_pin_memory`
- `use_flash_attention_2` (attempts to enable Flash Attention 2 if `flash_attn` is installed)

Example block you can add to your config:

```yaml
advanced:
  tf32: true
  dataloader_num_workers: 4
  dataloader_pin_memory: true
  use_flash_attention_2: true
```

If `flash_attn` is not installed, training will continue with a warning.

**About `flash-attn` installation:**

On some systems it may require a build toolchain and a compatible torch/CUDA combo. Treat it as optional; Unsloth is already optimized.

---

## 5) Monitoring

### 5.1 GPU

```bash
watch -n 1 nvidia-smi
```

### 5.2 TensorBoard

Logs are written under:

- `data/training_logs/<output_dir>/`

You can run TensorBoard manually:

```bash
tensorboard --logdir data/training_logs --host 127.0.0.1 --port 6006
```

Then from your local machine:

```bash
ssh -L 6006:127.0.0.1:6006 root@<YOUR_VULTR_IP>
```

Open: `http://localhost:6006`

---

## 6) Quick validation (don’t wait until the end)

Once you have a trained `final/` directory:

```bash
python scripts/training/train_unsloth.py \
  --test-only model_adapters/l4d2-code-lora/final \
  --test "Write a SourceMod plugin that hooks player_hurt and prints attacker/victim."
```

You can also attempt to test a checkpoint directory:

```bash
python scripts/training/train_unsloth.py --test-only model_adapters/l4d2-code-lora/checkpoint-200
```

---

## 7) Export + backup (do this before credits run out)

### 7.1 Export

GGUF example:

```bash
python scripts/training/export_model.py \
  --input model_adapters/l4d2-code-lora/final \
  --format gguf \
  --quantize q4_k_m
```

### 7.2 Backup artifacts

At minimum copy:

- `model_adapters/`
- `data/training_logs/`
- `data/processed/`

Example:

```bash
tar -czf l4d2-a100-backup.tar.gz model_adapters data/training_logs data/processed
```

From your laptop:

```bash
scp root@<YOUR_VULTR_IP>:/root/L4D2-AI-Architect/l4d2-a100-backup.tar.gz ./
```

---

## 8) Common issues

- **CUDA OOM**
  - Reduce `per_device_train_batch_size`
  - Or reduce `max_seq_length`

- **Low GPU utilization**
  - Increase batch size (first)
  - Add `advanced.dataloader_num_workers: 2-4`

- **Accidentally training on CPU**
  - `train_unsloth.py` will refuse to run if CUDA is unavailable.

---

## 9) Best next steps (high impact)

- Build a small fixed prompt set (10–30 prompts) and rerun it after every training run.
- Improve dataset quality:
  - Raise `--min-quality`
  - Consider adjusting the combined dataset to use language-specific system prompts (SourcePawn vs VScript).
- Train specialized adapters:
  - `sourcepawn_train.jsonl` only
  - `vscript_train.jsonl` only

