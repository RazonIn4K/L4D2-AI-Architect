# GH200 Deployment Guide

Complete walkthrough for deploying L4D2-AI-Architect on Vultr's NVIDIA GH200 (Grace Hopper) GPU instance.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Provisioning the Instance](#provisioning-the-instance)
3. [Initial Setup](#initial-setup)
4. [Installing Dependencies](#installing-dependencies)
5. [Running Training](#running-training)
6. [Monitoring](#monitoring)
7. [Handling Preemption](#handling-preemption)
8. [Exporting the Model](#exporting-the-model)
9. [Cost Optimization](#cost-optimization)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Vultr account with GPU access approved
- SSH key configured in Vultr dashboard
- $250 credits (provides ~83 hours at $2.996/hr)
- Training data prepared (`data/processed/combined_train.jsonl`)

## Provisioning the Instance

### 1. Log into Vultr Cloud Compute

Navigate to: https://my.vultr.com/deploy/

### 2. Select GPU Instance

- **Choose Type**: Cloud GPU
- **GPU Type**: NVIDIA GH200 (Grace Hopper)
- **Location**: Atlanta (closest to US Midwest)
- **Image**: Ubuntu 22.04 LTS

### 3. Configure Instance

- **Label**: `l4d2-training`
- **SSH Keys**: Select your key
- **Startup Script**: Leave empty (we'll run setup manually)

### 4. Deploy

Click "Deploy Now" - instance will be ready in ~2-5 minutes.

## Initial Setup

### 1. Connect via SSH

```bash
ssh root@<your-instance-ip>
```

### 2. Clone the Repository

```bash
cd /root
git clone https://github.com/YOUR_USERNAME/L4D2-AI-Architect.git
# Or upload your files via scp:
# scp -r ./L4D2-AI-Architect root@<ip>:/root/
```

### 3. Run GH200 Setup Script

```bash
cd /root/L4D2-AI-Architect
chmod +x scripts/utils/vultr_setup_gh200.sh
sudo ./scripts/utils/vultr_setup_gh200.sh
```

This script will:
- Verify ARM64 architecture and GPU
- Install Docker and nvidia-container-toolkit
- Pull NGC PyTorch container
- Create helper scripts

### 4. Verify GPU Access

```bash
nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xxx       Driver Version: 535.xxx       CUDA Version: 12.x   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GH200 120GB           On | 00000000:XX:00.0 Off |                    0 |
| N/A   30C    P0    70W / 900W |      0MiB / 98304MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

## Installing Dependencies

### 1. Run Container Setup

```bash
./run_in_container.sh --setup
```

This creates a persistent Python virtualenv at:
`/root/L4D2-AI-Architect/venv`

This installs inside the NGC container:
- bitsandbytes (ARM64 preview wheel)
- Unsloth
- TRL, PEFT, Accelerate
- All other training dependencies

### 2. Verify Installation

```bash
./run_in_container.sh python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

import bitsandbytes
print('bitsandbytes: OK')

from unsloth import FastLanguageModel
print('Unsloth: OK')
"
```

## Running Training

### 1. Start tmux Session (CRITICAL)

```bash
tmux new -s training
```

**Why tmux?** GH200 instances can be preempted. tmux keeps your session alive.

### 2. Upload Training Data

If not already present:
```bash
scp data/processed/combined_train.jsonl root@<ip>:/root/L4D2-AI-Architect/data/processed/
```

### 3. Start Training

```bash
./run_training_gh200.sh
```

Or with custom options:
```bash
./run_in_container.sh python scripts/training/train_unsloth.py \
    --config configs/unsloth_config_gh200.yaml \
    --epochs 3 \
    --batch-size 16
```

### 4. Detach from tmux

Press `Ctrl+B` then `D` to detach. Training continues in background.

### 5. Reattach Later

```bash
tmux attach -t training
```

## Monitoring

### TensorBoard

Run TensorBoard via the container wrapper:
```bash
./run_in_container.sh tensorboard --logdir data/training_logs --host 0.0.0.0 --port 6006
```

TensorBoard is exposed on the instance **localhost only**. Access it from your laptop via SSH tunnel:

```bash
ssh -L 6006:127.0.0.1:6006 root@<instance-ip>
```

Then open: `http://localhost:6006`

### GPU Usage

```bash
watch -n 1 nvidia-smi
```

Expected during training:
- Memory: 60-70 GB used
- GPU-Util: 90-100%
- Power: 400-600W

### Training Logs

```bash
tail -f data/training_logs/l4d2-code-lora/events.out.tfevents.*
```

## Handling Preemption

GH200 instances may be preempted with ~30 second warning.

### Automatic Checkpoints

Training saves checkpoints every 100 steps:
```
model_adapters/l4d2-code-lora/
├── checkpoint-100/
├── checkpoint-200/
├── checkpoint-300/
└── ...
```

### Resume After Preemption

1. Provision a new instance (same setup)
2. Upload your project (or re-clone)
3. Resume from last checkpoint:

```bash
./run_training_gh200.sh --resume model_adapters/l4d2-code-lora/checkpoint-300
```

### Backup Checkpoints

Periodically download checkpoints:
```bash
scp -r root@<ip>:/root/L4D2-AI-Architect/model_adapters/l4d2-code-lora/checkpoint-* ./backups/
```

## Exporting the Model

### 1. Export to GGUF (for Ollama)

```bash
./run_in_container.sh python scripts/training/export_model.py \
    --input model_adapters/l4d2-code-lora/final \
    --format gguf \
    --quantize q4_k_m
```

### 2. Download Exported Model

```bash
scp root@<ip>:/root/L4D2-AI-Architect/model_adapters/exports/*.gguf ./
```

### 3. Push to HuggingFace (Optional)

Update config:
```yaml
output:
  push_to_hub: true
  hub_model_id: "your-username/l4d2-code-lora"
```

## Cost Optimization

### GH200 Pricing

- Rate: $2.996/hour
- $250 credits = ~83.4 hours

### Estimated Training Time

| Dataset Size | Epochs | Estimated Time | Cost |
|-------------|--------|----------------|------|
| 5,000 samples | 3 | ~1 hour | ~$3 |
| 10,000 samples | 3 | ~2 hours | ~$6 |
| 50,000 samples | 3 | ~8 hours | ~$24 |

### Tips

1. **Test locally first** - Use small subset to verify config
2. **Export early** - Save model before credits run out
3. **Use checkpoints** - Don't lose progress to preemption
4. **Monitor usage** - Check Vultr billing dashboard

## Troubleshooting

### "CUDA not available" in Container

```bash
# Verify nvidia-container-toolkit
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

If this fails, reinstall nvidia-container-toolkit:
```bash
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
```

### "bitsandbytes" Import Error

ARM64 requires the preview wheel:
```bash
pip install --force-reinstall https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-1.33.7.preview-py3-none-manylinux_2_24_aarch64.whl
```

### Out of Memory (OOM)

Reduce batch size in config:
```yaml
training:
  per_device_train_batch_size: 8  # Reduce from 16
```

### Container Won't Start

Check Docker service:
```bash
systemctl status docker
systemctl restart docker
```

### SSH Connection Lost

1. Re-provision instance if preempted
2. Check Vultr dashboard for instance status
3. Resume from checkpoint on new instance

---

## Quick Reference Card

```
# SSH to instance
ssh root@<ip>

# Start training (always in tmux!)
tmux new -s training
./run_training_gh200.sh

# Detach: Ctrl+B then D
# Reattach: tmux attach -t training

# Resume from checkpoint
./run_training_gh200.sh --resume model_adapters/l4d2-code-lora/checkpoint-XXX

# Interactive container
./run_in_container.sh

# Monitor GPU
watch -n 1 nvidia-smi

# TensorBoard
./run_in_container.sh tensorboard --logdir data/training_logs --host 0.0.0.0

# Access TensorBoard from your laptop
ssh -L 6006:127.0.0.1:6006 root@<ip>

# Backup checkpoint
scp -r root@<ip>:/root/L4D2-AI-Architect/model_adapters/l4d2-code-lora ./backup/
```
