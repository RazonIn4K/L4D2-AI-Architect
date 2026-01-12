# Vultr A100 Training Runbook: L4D2 Code Model v11

Complete step-by-step guide to train the L4D2 SourcePawn/VScript code model on Vultr's A100 GPU instance.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Create Vultr A100 Instance](#2-create-vultr-a100-instance)
3. [Instance Setup](#3-instance-setup)
4. [Upload Training Data](#4-upload-training-data)
5. [Run Training](#5-run-training)
6. [Export to GGUF](#6-export-to-gguf)
7. [Download and Local Deployment](#7-download-and-local-deployment)
8. [Cost Summary](#8-cost-summary)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Prerequisites

### What You Need

| Requirement | Details |
|-------------|---------|
| **Vultr Account** | Sign up at [vultr.com](https://www.vultr.com/) |
| **Credits** | Minimum $10 recommended (~4 hours of training buffer) |
| **SSH Key** | Generate with `ssh-keygen -t ed25519` if needed |
| **Local Machine** | macOS/Linux with `scp` and `ssh` available |
| **Git Repository** | Clone URL for L4D2-AI-Architect |

### Instance Specifications

| Component | Specification |
|-----------|---------------|
| **GPU** | 1/2 NVIDIA A100 (40GB HBM2e VRAM) |
| **CPU** | 6 vCPUs |
| **RAM** | 60 GB |
| **Storage** | 1400 GB NVMe |
| **OS** | Ubuntu 22.04 with NVIDIA drivers |

### Estimated Costs

| Task | Time | Cost @ $2.50/hr |
|------|------|-----------------|
| Instance setup | 15 min | ~$0.65 |
| Training (771 samples, 3 epochs) | 1.5-2 hrs | ~$3.75-$5.00 |
| Export to GGUF | 15-30 min | ~$0.65-$1.25 |
| **Total** | **2-3 hours** | **$5.00-$7.50** |

---

## 2. Create Vultr A100 Instance

### Step 2.1: Navigate to Cloud GPU

```
Vultr Dashboard -> Products -> Cloud GPU -> Deploy Instance
```

### Step 2.2: Select Configuration

| Setting | Value |
|---------|-------|
| **Type** | Cloud GPU |
| **Location** | New Jersey (recommended for US) or nearest |
| **GPU** | NVIDIA A100 (1/2 GPU, 40GB) |
| **Plan** | `vcg-a100-6c-60g-40vram` (~$2.50/hr) |
| **OS** | Ubuntu 22.04 LTS x64 |
| **SSH Key** | Add your public key |
| **Hostname** | `l4d2-training` |

### Step 2.3: Deploy and Wait

Click **Deploy Now** and wait 2-5 minutes for provisioning.

Copy the **IP Address** once available.

---

## 3. Instance Setup

### Step 3.1: SSH into Instance

```bash
# Replace with your instance IP
export VULTR_IP="YOUR_INSTANCE_IP"

ssh root@$VULTR_IP
```

### Step 3.2: Verify GPU

```bash
nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xxx       Driver Version: 535.xxx       CUDA Version: 12.x   |
|-------------------------------+----------------------+----------------------+
|   0  NVIDIA A100-SXM...  On   | 00000000:00:00.0 Off |                    0 |
| N/A   32C    P0    52W / 400W |      0MiB / 40960MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

### Step 3.3: Install System Dependencies

```bash
# Update system
apt-get update && apt-get upgrade -y

# Install essentials
apt-get install -y tmux git python3-pip python3-venv htop

# Verify CUDA
nvcc --version
```

### Step 3.4: Clone Repository

```bash
cd /root
git clone https://github.com/YOUR_USERNAME/L4D2-AI-Architect.git
cd L4D2-AI-Architect
```

### Step 3.5: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3.6: Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install Unsloth (optimized training)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install Flash Attention 2 (A100 optimization - takes ~5 min)
pip install flash-attn --no-build-isolation
```

### Step 3.7: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
PyTorch: 2.x.x
CUDA: True
GPU: NVIDIA A100-SXM4-40GB
```

---

## 4. Upload Training Data

### From Your Local Machine

Open a **new terminal** on your local machine:

```bash
# Set your Vultr IP
export VULTR_IP="YOUR_INSTANCE_IP"

# Navigate to your local L4D2-AI-Architect directory
cd /path/to/L4D2-AI-Architect

# Upload the v10 training dataset (latest)
scp data/processed/l4d2_train_v10.jsonl root@$VULTR_IP:/root/L4D2-AI-Architect/data/processed/

# Upload validation data (if available)
scp data/processed/combined_val.jsonl root@$VULTR_IP:/root/L4D2-AI-Architect/data/processed/
```

### Verify Upload (on Vultr instance)

```bash
# Back on the Vultr instance
wc -l /root/L4D2-AI-Architect/data/processed/l4d2_train_v10.jsonl
```

Expected: `771` lines (training examples)

---

## 5. Run Training

### Step 5.1: Create A100 Config for v11

```bash
cd /root/L4D2-AI-Architect

# Create/update config to use v10 dataset
cat > configs/unsloth_config_v11.yaml << 'EOF'
# L4D2-AI-Architect: Unsloth Training Configuration v11
# Optimized for Vultr 1/2 NVIDIA A100 (40GB VRAM)

model:
  name: "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
  max_seq_length: 4096
  dtype: null
  load_in_4bit: true

lora:
  r: 32
  lora_alpha: 64
  lora_dropout: 0
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  bias: "none"
  use_gradient_checkpointing: "unsloth"
  use_rslora: false

training:
  num_train_epochs: 3
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 2
  learning_rate: 2.0e-4
  weight_decay: 0.01
  warmup_steps: 50
  lr_scheduler_type: "cosine"
  optim: "adamw_8bit"
  fp16: false
  bf16: true
  logging_steps: 10
  save_steps: 100
  save_total_limit: 3
  seed: 3407
  max_grad_norm: 1.0

data:
  train_file: "l4d2_train_v10.jsonl"
  val_file: "combined_val.jsonl"
  max_samples: null

output:
  dir: "l4d2-mistral-v11-lora"
  push_to_hub: false
  hub_model_id: null

advanced:
  use_flash_attention_2: true
  dataloader_num_workers: 4
  dataloader_pin_memory: true
  tf32: true

monitoring:
  report_to: "tensorboard"
  logging_dir: "data/training_logs"
  evaluation_strategy: "steps"
  eval_steps: 100
EOF
```

### Step 5.2: Start Training in tmux

**IMPORTANT:** Always use tmux for training to prevent session disconnection from stopping the job.

```bash
# Start tmux session
tmux new -s training

# Inside tmux, activate environment
cd /root/L4D2-AI-Architect
source venv/bin/activate

# Start training
python scripts/training/train_unsloth.py --config configs/unsloth_config_v11.yaml
```

### Tmux Controls

| Command | Action |
|---------|--------|
| `Ctrl+B` then `D` | Detach from tmux (training continues) |
| `tmux attach -t training` | Reattach to session |
| `tmux kill-session -t training` | Stop training |

### Step 5.3: Monitor Training (Optional)

In a **new SSH session**:

```bash
ssh root@$VULTR_IP

# Watch GPU utilization (should be 90%+)
watch -n 2 nvidia-smi

# Or view training logs
tail -f /root/L4D2-AI-Architect/data/training_logs/l4d2-mistral-v11-lora/*.log
```

### Expected Training Progress

| Epoch | Time | Loss | VRAM |
|-------|------|------|------|
| 1/3 | ~30 min | ~1.5 -> 0.8 | ~35GB |
| 2/3 | ~60 min | ~0.8 -> 0.5 | ~35GB |
| 3/3 | ~90 min | ~0.5 -> 0.3 | ~35GB |

---

## 6. Export to GGUF

After training completes, export to GGUF format for Ollama.

### Step 6.1: Verify Training Output

```bash
ls -la /root/L4D2-AI-Architect/model_adapters/l4d2-mistral-v11-lora/
```

Expected files:
```
final/
  adapter_config.json
  adapter_model.safetensors
  tokenizer.json
  ...
lora_adapter/
  ...
training_info.json
```

### Step 6.2: Export to GGUF

```bash
cd /root/L4D2-AI-Architect
source venv/bin/activate

# Export with Q4_K_M quantization (best quality/size balance)
python scripts/training/export_gguf_cpu.py \
    --adapter model_adapters/l4d2-mistral-v11-lora/final \
    --output exports/l4d2-v11 \
    --quantize q4_k_m \
    --create-modelfile
```

### Alternative: Manual GGUF Conversion

If the automatic conversion fails:

```bash
# Clone llama.cpp
git clone https://github.com/ggml-org/llama.cpp /root/llama.cpp
cd /root/llama.cpp
pip install -r requirements.txt

# Convert to GGUF
python convert_hf_to_gguf.py /root/L4D2-AI-Architect/exports/l4d2-v11/merged \
    --outfile /root/L4D2-AI-Architect/exports/l4d2-v11/l4d2-mistral-v11-q4_k_m.gguf

# Quantize (if not already quantized)
./llama-quantize /root/L4D2-AI-Architect/exports/l4d2-v11/l4d2-mistral-v11-f16.gguf \
    /root/L4D2-AI-Architect/exports/l4d2-v11/l4d2-mistral-v11-q4_k_m.gguf q4_k_m
```

### Step 6.3: Verify Export

```bash
ls -lh /root/L4D2-AI-Architect/exports/l4d2-v11/gguf/
```

Expected: `l4d2-mistral-v11-q4_k_m.gguf` (~4-5 GB)

---

## 7. Download and Local Deployment

### Step 7.1: Download Model to Local Machine

From your **local machine**:

```bash
export VULTR_IP="YOUR_INSTANCE_IP"

# Create local exports directory
mkdir -p ~/L4D2-AI-Architect/exports/l4d2-v11

# Download GGUF and Modelfile
scp -r root@$VULTR_IP:/root/L4D2-AI-Architect/exports/l4d2-v11/gguf/ \
    ~/L4D2-AI-Architect/exports/l4d2-v11/

# Download the LoRA adapter (backup)
scp -r root@$VULTR_IP:/root/L4D2-AI-Architect/model_adapters/l4d2-mistral-v11-lora/ \
    ~/L4D2-AI-Architect/model_adapters/
```

### Step 7.2: Install Ollama (if not installed)

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Verify
ollama --version
```

### Step 7.3: Create Ollama Model

```bash
cd ~/L4D2-AI-Architect/exports/l4d2-v11/gguf

# Create the model
ollama create l4d2-code-v11 -f Modelfile
```

### Step 7.4: Test the Model

```bash
# Interactive chat
ollama run l4d2-code-v11

# Single prompt
ollama run l4d2-code-v11 "Write a SourcePawn function to spawn a Tank at a random survivor's position"
```

### Step 7.5: Use with Copilot CLI

```bash
cd ~/L4D2-AI-Architect

# Activate environment
source venv/bin/activate

# Interactive chat mode
python scripts/inference/copilot_cli.py chat --model l4d2-code-v11

# Single prompt
python scripts/inference/copilot_cli.py ollama --model l4d2-code-v11 \
    --prompt "Write a function to heal all survivors within 500 units"
```

---

## 8. Cost Summary

### Breakdown

| Phase | Duration | Cost @ $2.50/hr |
|-------|----------|-----------------|
| Instance provisioning | 5 min | $0.21 |
| Environment setup | 15 min | $0.63 |
| Dependency installation | 10 min | $0.42 |
| Training (771 samples, 3 epochs) | 90 min | $3.75 |
| GGUF export | 20 min | $0.83 |
| Download & cleanup | 10 min | $0.42 |
| **Total** | **~2.5 hours** | **~$6.25** |

### Cost Optimization Tips

1. **Prepare data locally first** - Upload only when ready
2. **Use tmux** - Prevents training restart if SSH disconnects
3. **Export immediately after training** - Don't leave instance idle
4. **Destroy instance when done** - Stop billing immediately

---

## 9. Troubleshooting

### CUDA Out of Memory (OOM)

```bash
# Reduce batch size to 4
python scripts/training/train_unsloth.py \
    --config configs/unsloth_config_v11.yaml \
    --batch-size 4
```

### Training Loss is NaN

Ensure BF16 is enabled (not FP16):

```yaml
training:
  fp16: false
  bf16: true
```

### Flash Attention Not Working

```bash
# Reinstall
pip uninstall flash-attn
pip install flash-attn --no-build-isolation

# Or disable in config
# advanced:
#   use_flash_attention_2: false
```

### SSH Disconnected During Training

Training continues in tmux. Reconnect:

```bash
ssh root@$VULTR_IP
tmux attach -t training
```

### Resume from Checkpoint

If training was interrupted:

```bash
python scripts/training/train_unsloth.py \
    --config configs/unsloth_config_v11.yaml \
    --resume model_adapters/l4d2-mistral-v11-lora/checkpoint-XXX
```

### Slow Training (< 80% GPU Utilization)

```yaml
# Increase dataloader workers
advanced:
  dataloader_num_workers: 4
  dataloader_pin_memory: true
```

---

## Quick Reference Card

```bash
# === VULTR INSTANCE ===
ssh root@$VULTR_IP

# === SETUP (one-time) ===
apt-get update && apt-get install -y tmux git python3-venv
cd /root && git clone <REPO_URL> L4D2-AI-Architect && cd L4D2-AI-Architect
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install flash-attn --no-build-isolation

# === TRAINING ===
tmux new -s training
source venv/bin/activate
python scripts/training/train_unsloth.py --config configs/unsloth_config_v11.yaml
# Detach: Ctrl+B then D

# === MONITOR ===
watch -n 2 nvidia-smi
tmux attach -t training

# === EXPORT ===
python scripts/training/export_gguf_cpu.py \
    --adapter model_adapters/l4d2-mistral-v11-lora/final \
    --output exports/l4d2-v11 \
    --quantize q4_k_m \
    --create-modelfile

# === DOWNLOAD (from local machine) ===
scp -r root@$VULTR_IP:/root/L4D2-AI-Architect/exports/l4d2-v11/gguf/ ./exports/

# === LOCAL OLLAMA ===
cd exports/l4d2-v11/gguf
ollama create l4d2-code-v11 -f Modelfile
ollama run l4d2-code-v11
```

---

## Cleanup

**IMPORTANT:** Destroy the instance when done to stop billing.

```
Vultr Dashboard -> Your Instance -> Settings -> Destroy
```

Or via CLI:
```bash
# List instances
vultr-cli instance list

# Destroy
vultr-cli instance destroy <INSTANCE_ID>
```

---

*Last updated: January 2025*
*Dataset: l4d2_train_v10.jsonl (771 examples)*
*Target model: l4d2-code-v11*
