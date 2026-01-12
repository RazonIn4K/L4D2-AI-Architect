# Vultr Credit Burn Command Reference

**The definitive reference for maximizing Vultr GPU credits on L4D2-AI-Architect**

---

## Table of Contents

1. [Pre-Flight Checks](#1-pre-flight-checks)
2. [One-Liner Commands](#2-one-liner-commands)
3. [Cost and Time Estimates](#3-cost-and-time-estimates)
4. [Object Storage Setup](#4-object-storage-setup)
5. [Monitoring Commands](#5-monitoring-commands)
6. [Download Commands](#6-download-commands)
7. [Cleanup Commands](#7-cleanup-commands)
8. [Troubleshooting Quick Fixes](#8-troubleshooting-quick-fixes)

---

## 1. Pre-Flight Checks

### 1.1 Verify Vultr Credits

```bash
# Check credits via Vultr dashboard
open https://my.vultr.com/billing/

# Or via CLI (requires vultr-cli installed)
vultr-cli account

# Expected output shows:
# - Current balance
# - Pending charges
# - Credits remaining
```

### 1.2 Check Dataset Readiness

```bash
# Run from local machine before deploying
cd /Users/davidortiz/left4dead-model/L4D2-AI-Architect

# Count training examples (need at least 500+ for good results)
wc -l data/processed/l4d2_train_v*.jsonl

# Expected output (V13 recommended):
# 771 data/processed/l4d2_train_v10.jsonl
# 771 data/processed/l4d2_train_v11.jsonl
# 830 data/processed/l4d2_train_v12.jsonl
# 1010 data/processed/l4d2_train_v13.jsonl

# Verify JSON validity
python -c "import json; [json.loads(l) for l in open('data/processed/l4d2_train_v13.jsonl')]" && echo "Dataset valid!"

# Check for validation data
ls -la data/processed/*val*.jsonl
```

### 1.3 Verify Configs

```bash
# Check configs exist and are valid YAML
python -c "import yaml; yaml.safe_load(open('configs/unsloth_config_a100.yaml'))" && echo "A100 config OK"
python -c "import yaml; yaml.safe_load(open('configs/unsloth_config_gh200.yaml'))" && echo "GH200 config OK"
python -c "import yaml; yaml.safe_load(open('configs/unsloth_config.yaml'))" && echo "Default config OK"

# View key settings
grep -E "batch_size|num_train_epochs|bf16|learning_rate" configs/unsloth_config_a100.yaml
```

### 1.4 Verify SSH Key

```bash
# Test SSH key is ready
ls -la ~/.ssh/id_ed25519.pub 2>/dev/null || ls -la ~/.ssh/id_rsa.pub

# Generate if missing
ssh-keygen -t ed25519 -C "vultr-l4d2-training"
```

---

## 2. One-Liner Commands

### 2.1 LLM Training Only (Mistral-7B)

**A100 Instance ($2.50/hr):**
```bash
# Set your instance IP
export VULTR_IP="YOUR_INSTANCE_IP"

# SSH and run training in one command (A100)
ssh root@$VULTR_IP 'cd /root && git clone https://github.com/RazonIn4K/L4D2-AI-Architect.git 2>/dev/null; cd L4D2-AI-Architect && tmux new-session -d -s training "source venv/bin/activate 2>/dev/null || (python3 -m venv venv && source venv/bin/activate && pip install -q -r requirements.txt && pip install -q unsloth flash-attn --no-build-isolation) && python scripts/training/train_unsloth.py --config configs/unsloth_config_a100.yaml"'

# Monitor
ssh root@$VULTR_IP 'tmux attach -t training'
```

**Quick LLM Training (minimal setup):**
```bash
ssh root@$VULTR_IP 'tmux new-session -d -s training "cd L4D2-AI-Architect && ./run_vultr_training.sh 3 8"'
```

### 2.2 RL Training Only (PPO Bot Agents)

**Single Personality (500K timesteps, ~15 min):**
```bash
ssh root@$VULTR_IP 'tmux new-session -d -s rl "cd L4D2-AI-Architect && source venv/bin/activate && python scripts/rl_training/train_ppo.py --timesteps 500000 --personality balanced"'
```

**All 5 Personalities (2.5M total timesteps, ~45 min):**
```bash
ssh root@$VULTR_IP 'tmux new-session -d -s rl "cd L4D2-AI-Architect && source venv/bin/activate && python scripts/rl_training/train_all_personalities.py --timesteps 500000"'
```

**Quick RL Test (100K timesteps, ~5 min):**
```bash
ssh root@$VULTR_IP 'python scripts/rl_training/quick_ppo_test.py'
```

### 2.3 Embeddings Only (FAISS Index)

```bash
ssh root@$VULTR_IP 'cd L4D2-AI-Architect && source venv/bin/activate && pip install -q sentence-transformers faiss-cpu && python scripts/training/generate_embeddings.py --data data/processed/l4d2_train_v13.jsonl --model all-MiniLM-L6-v2'
```

### 2.4 All Workloads (Full Pipeline)

**Complete Training Pipeline (~4-6 hours, ~$15-20):**
```bash
ssh root@$VULTR_IP 'tmux new-session -d -s full "cd L4D2-AI-Architect && source venv/bin/activate && \
  echo \"=== LLM Training ===\" && \
  python scripts/training/train_unsloth.py --config configs/unsloth_config_a100.yaml && \
  echo \"=== RL Training ===\" && \
  python scripts/rl_training/train_all_personalities.py --timesteps 500000 && \
  echo \"=== Embeddings ===\" && \
  python scripts/training/generate_embeddings.py --data data/processed/l4d2_train_v13.jsonl && \
  echo \"=== GGUF Export ===\" && \
  python scripts/training/export_gguf_cpu.py --adapter model_adapters/l4d2-mistral-v*-lora/final --create-modelfile && \
  echo \"=== COMPLETE ===\""'
```

### 2.5 Multi-Model Comparison

**Train Multiple Base Models (Mistral + CodeLlama + Llama3):**
```bash
# Mistral (default, best for code)
ssh root@$VULTR_IP 'tmux new-session -d -s mistral "cd L4D2-AI-Architect && source venv/bin/activate && python scripts/training/train_unsloth.py --config configs/unsloth_config.yaml --output model_adapters/l4d2-mistral-compare"'

# CodeLlama (code-specialized)
ssh root@$VULTR_IP 'tmux new-session -d -s codellama "cd L4D2-AI-Architect && source venv/bin/activate && python scripts/training/train_unsloth.py --config configs/unsloth_config_codellama.yaml --output model_adapters/l4d2-codellama-compare"'

# Llama3 (latest architecture)
ssh root@$VULTR_IP 'tmux new-session -d -s llama3 "cd L4D2-AI-Architect && source venv/bin/activate && python scripts/training/train_unsloth.py --config configs/unsloth_config_llama3.yaml --output model_adapters/l4d2-llama3-compare"'
```

---

## 3. Cost and Time Estimates

### 3.1 GPU Instance Pricing

| Instance Type | GPU | VRAM | Hourly Cost | Best For |
|---------------|-----|------|-------------|----------|
| `vcg-a100-6c-60g-40vram` | 1/2 A100 | 40GB | $2.50/hr | LLM training (fastest) |
| `vcg-a40-6c-48g` | A40 | 48GB | $1.80/hr | LLM training (more VRAM) |
| `vcg-l40s-6c-48g` | L40S | 48GB | $1.50/hr | Budget training |
| `vcg-gh200-*` | GH200 | 96GB | $3.50/hr | Large models |

### 3.2 Workload Time and Cost Matrix

| Workload | A100 Time | A100 Cost | A40 Time | A40 Cost |
|----------|-----------|-----------|----------|----------|
| **LLM Training (1K samples, 3 epochs)** | 1.5-2 hrs | $3.75-$5.00 | 3-4 hrs | $5.40-$7.20 |
| **LLM Training (1K samples, 5 epochs)** | 2.5-3 hrs | $6.25-$7.50 | 5-6 hrs | $9.00-$10.80 |
| **RL Training (1 personality, 500K steps)** | 15-20 min | $0.63-$0.83 | 25-35 min | $0.75-$1.05 |
| **RL Training (5 personalities, 2.5M steps)** | 45-60 min | $1.88-$2.50 | 90-120 min | $2.70-$3.60 |
| **Embeddings (1K samples)** | 5-10 min | $0.21-$0.42 | 10-15 min | $0.30-$0.45 |
| **GGUF Export** | 15-30 min | $0.63-$1.25 | 30-45 min | $0.90-$1.35 |
| **Full Pipeline** | 3-4 hrs | $7.50-$10.00 | 5-7 hrs | $9.00-$12.60 |

### 3.3 Budget Planning Calculator

| Credits Available | Recommended Plan |
|-------------------|------------------|
| **$10** | Quick LLM train (3 epochs) + GGUF export |
| **$25** | Full LLM train (5 epochs) + RL (1 personality) + embeddings |
| **$50** | Full LLM + All RL personalities + embeddings + multi-model |
| **$100** | All above + hyperparameter tuning + multiple training runs |
| **$250** | Full experimentation: multiple models, extensive RL, evaluation |

---

## 4. Object Storage Setup

### 4.1 Create Vultr Object Storage

```bash
# Via Vultr Dashboard
open https://my.vultr.com/objectstorage/

# Or via CLI
vultr-cli object-storage create --label "l4d2-models" --cluster ewr1

# Get credentials
vultr-cli object-storage list
```

### 4.2 Configure S3-Compatible Access

```bash
# On Vultr instance, install s3cmd
apt-get install -y s3cmd

# Configure (use Vultr Object Storage credentials)
cat > ~/.s3cfg << 'EOF'
[default]
access_key = YOUR_ACCESS_KEY
secret_key = YOUR_SECRET_KEY
host_base = ewr1.vultrobjects.com
host_bucket = %(bucket)s.ewr1.vultrobjects.com
use_https = True
EOF

# Create bucket
s3cmd mb s3://l4d2-models

# List buckets
s3cmd ls
```

### 4.3 Sync Models to Object Storage

```bash
# Upload trained models
s3cmd sync model_adapters/ s3://l4d2-models/adapters/

# Upload GGUF exports
s3cmd sync exports/ s3://l4d2-models/exports/

# Upload RL agents
s3cmd sync model_adapters/rl_agents/ s3://l4d2-models/rl/

# Upload embeddings
s3cmd sync data/embeddings/ s3://l4d2-models/embeddings/
```

### 4.4 Download from Object Storage

```bash
# From local machine (with s3cmd configured)
s3cmd sync s3://l4d2-models/ ./vultr-backup/

# Or use presigned URLs for specific files
s3cmd signurl s3://l4d2-models/exports/l4d2-v13.gguf +86400
```

---

## 5. Monitoring Commands

### 5.1 GPU Utilization

```bash
# Real-time GPU monitoring (target: >90% utilization)
ssh root@$VULTR_IP 'watch -n 2 nvidia-smi'

# One-shot GPU status
ssh root@$VULTR_IP 'nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv'

# GPU memory history (run during training)
ssh root@$VULTR_IP 'nvidia-smi dmon -s um -d 10'
```

### 5.2 Training Progress

```bash
# Attach to training session
ssh root@$VULTR_IP 'tmux attach -t training'

# View training logs (tail)
ssh root@$VULTR_IP 'tail -f L4D2-AI-Architect/data/training_logs/*/events.*'

# Check checkpoints saved
ssh root@$VULTR_IP 'ls -lh L4D2-AI-Architect/model_adapters/*/checkpoint-*/'

# TensorBoard (run on Vultr, access via SSH tunnel)
ssh -L 6006:localhost:6006 root@$VULTR_IP 'cd L4D2-AI-Architect && tensorboard --logdir data/training_logs --port 6006'
# Then open: http://localhost:6006
```

### 5.3 Disk Usage

```bash
# Check disk space
ssh root@$VULTR_IP 'df -h /'

# Model sizes
ssh root@$VULTR_IP 'du -sh L4D2-AI-Architect/model_adapters/*'

# Total project size
ssh root@$VULTR_IP 'du -sh L4D2-AI-Architect/'

# Find large files
ssh root@$VULTR_IP 'find L4D2-AI-Architect -size +100M -exec ls -lh {} \;'
```

### 5.4 System Resources

```bash
# CPU and memory usage
ssh root@$VULTR_IP 'htop'

# Quick system status
ssh root@$VULTR_IP 'free -h && echo "---" && uptime'

# Process list for training
ssh root@$VULTR_IP 'ps aux | grep -E "python|train"'
```

---

## 6. Download Commands

### 6.1 Download GGUF Models

```bash
export VULTR_IP="YOUR_INSTANCE_IP"

# Create local directories
mkdir -p ~/L4D2-AI-Architect/exports/gguf

# Download all GGUF files
scp -r root@$VULTR_IP:/root/L4D2-AI-Architect/exports/*/gguf/*.gguf ~/L4D2-AI-Architect/exports/gguf/

# Download specific version
scp root@$VULTR_IP:/root/L4D2-AI-Architect/exports/l4d2-v13/gguf/l4d2-mistral-v13-q4_k_m.gguf ~/L4D2-AI-Architect/exports/

# Download Modelfile too
scp root@$VULTR_IP:/root/L4D2-AI-Architect/exports/*/gguf/Modelfile ~/L4D2-AI-Architect/exports/gguf/
```

### 6.2 Download LoRA Adapters

```bash
# All adapters (can be large)
scp -r root@$VULTR_IP:/root/L4D2-AI-Architect/model_adapters/l4d2-*-lora/ ~/L4D2-AI-Architect/model_adapters/

# Just the final adapter (smaller)
scp -r root@$VULTR_IP:/root/L4D2-AI-Architect/model_adapters/l4d2-mistral-v13-lora/final/ ~/L4D2-AI-Architect/model_adapters/l4d2-mistral-v13-lora/

# Compressed download (faster for slow connections)
ssh root@$VULTR_IP 'cd L4D2-AI-Architect && tar -czvf lora-adapters.tar.gz model_adapters/l4d2-*-lora/final/'
scp root@$VULTR_IP:/root/L4D2-AI-Architect/lora-adapters.tar.gz ./
tar -xzvf lora-adapters.tar.gz
```

### 6.3 Download RL Agents

```bash
# All RL agents
scp -r root@$VULTR_IP:/root/L4D2-AI-Architect/model_adapters/rl_agents/ ~/L4D2-AI-Architect/model_adapters/

# Specific personality
scp -r root@$VULTR_IP:/root/L4D2-AI-Architect/model_adapters/rl_agents/ppo_balanced_*/ ~/L4D2-AI-Architect/model_adapters/rl_agents/

# Just the best models
scp -r root@$VULTR_IP:/root/L4D2-AI-Architect/model_adapters/rl_agents/*/best_model/ ~/L4D2-AI-Architect/model_adapters/rl_agents/

# Training results JSON
scp root@$VULTR_IP:/root/L4D2-AI-Architect/model_adapters/rl_agents/*/training_info.json ~/L4D2-AI-Architect/model_adapters/rl_agents/
```

### 6.4 Download Embeddings

```bash
# All embeddings and FAISS index
scp -r root@$VULTR_IP:/root/L4D2-AI-Architect/data/embeddings/ ~/L4D2-AI-Architect/data/

# Verify download
ls -la ~/L4D2-AI-Architect/data/embeddings/
# Should have: prompts.npy, responses.npy, combined.npy, metadata.json, faiss_index.bin
```

### 6.5 Download All Artifacts (Complete Backup)

```bash
# Create timestamped backup
BACKUP_DIR="vultr-backup-$(date +%Y%m%d_%H%M%S)"
mkdir -p ~/$BACKUP_DIR

# Compress on server first (much faster transfer)
ssh root@$VULTR_IP 'cd L4D2-AI-Architect && tar -czvf /tmp/l4d2-all-artifacts.tar.gz \
  model_adapters/ \
  exports/ \
  data/embeddings/ \
  data/training_logs/ \
  --exclude="*.safetensors" \
  --exclude="checkpoint-*"'

# Download compressed archive
scp root@$VULTR_IP:/tmp/l4d2-all-artifacts.tar.gz ~/$BACKUP_DIR/

# Extract locally
cd ~/$BACKUP_DIR && tar -xzvf l4d2-all-artifacts.tar.gz

# Verify
du -sh ~/$BACKUP_DIR/*
```

### 6.6 Rsync for Large Transfers (Resume Support)

```bash
# Rsync with progress and resume capability
rsync -avzP --progress root@$VULTR_IP:/root/L4D2-AI-Architect/exports/ ~/L4D2-AI-Architect/exports/

# Rsync everything important
rsync -avzP --progress \
  --exclude='venv/' \
  --exclude='*.pyc' \
  --exclude='__pycache__/' \
  --exclude='.git/' \
  root@$VULTR_IP:/root/L4D2-AI-Architect/ ~/L4D2-AI-Architect-backup/
```

---

## 7. Cleanup Commands

### 7.1 Delete Instance

**Via Dashboard (Recommended):**
```bash
# Open instance settings
open https://my.vultr.com/compute/

# Click on instance -> Settings -> Destroy Instance
```

**Via CLI:**
```bash
# List instances to get ID
vultr-cli instance list

# Destroy specific instance
vultr-cli instance delete <INSTANCE_ID>

# Confirm deletion
vultr-cli instance list
```

### 7.2 Verify No Charges

```bash
# Check billing immediately after deletion
vultr-cli account

# Or via dashboard
open https://my.vultr.com/billing/

# Expected: Instance should disappear within 1-2 minutes
# Billing stops the moment instance is destroyed
```

### 7.3 Clean Up Object Storage (Optional)

```bash
# List bucket contents
s3cmd ls s3://l4d2-models/ --recursive

# Delete old files
s3cmd del s3://l4d2-models/old-version/

# Delete entire bucket (careful!)
s3cmd rb s3://l4d2-models --force

# Verify deletion
vultr-cli object-storage list
```

### 7.4 Local Cleanup

```bash
# Clean up local temp files
rm -f vultr-backup-*.tar.gz
rm -f lora-adapters.tar.gz

# Verify local models work before deleting Vultr backups
ollama run l4d2-code-v13 "Write a plugin to spawn a Tank"
```

---

## 8. Troubleshooting Quick Fixes

### 8.1 SSH Connection Issues

```bash
# Connection refused
# Wait 2-5 minutes after instance creation

# Permission denied
ssh-add ~/.ssh/id_ed25519

# Slow connection
ssh -o Compression=yes -o TCPKeepAlive=yes root@$VULTR_IP
```

### 8.2 CUDA/GPU Issues

```bash
# GPU not detected
ssh root@$VULTR_IP 'nvidia-smi'
# If fails: Reboot instance

# CUDA version mismatch
ssh root@$VULTR_IP 'pip install torch --index-url https://download.pytorch.org/whl/cu121'

# Out of VRAM
# Reduce batch size in config:
sed -i 's/per_device_train_batch_size: 8/per_device_train_batch_size: 4/' configs/unsloth_config_a100.yaml
```

### 8.3 Training Issues

```bash
# Training stuck at 0%
# Check GPU is being used:
ssh root@$VULTR_IP 'nvidia-smi'

# Loss is NaN
# Switch to BF16:
ssh root@$VULTR_IP 'grep -E "bf16|fp16" L4D2-AI-Architect/configs/unsloth_config_a100.yaml'
# Should show: bf16: true, fp16: false

# Out of memory during training
ssh root@$VULTR_IP 'cd L4D2-AI-Architect && sed -i "s/per_device_train_batch_size: 8/per_device_train_batch_size: 4/" configs/unsloth_config_a100.yaml'

# Resume from checkpoint
ssh root@$VULTR_IP 'python scripts/training/train_unsloth.py --config configs/unsloth_config_a100.yaml --resume model_adapters/l4d2-mistral-v13-lora/checkpoint-XXX'
```

### 8.4 Tmux Session Issues

```bash
# List all tmux sessions
ssh root@$VULTR_IP 'tmux ls'

# Session not found
# May have completed or crashed - check logs:
ssh root@$VULTR_IP 'tail -100 L4D2-AI-Architect/data/training_logs/*/events.*'

# Kill stuck session
ssh root@$VULTR_IP 'tmux kill-session -t training'

# Create new session
ssh root@$VULTR_IP 'tmux new -s training'
```

### 8.5 Disk Space Issues

```bash
# Check disk usage
ssh root@$VULTR_IP 'df -h /'

# Remove old checkpoints (keep only last 3)
ssh root@$VULTR_IP 'cd L4D2-AI-Architect && find model_adapters -name "checkpoint-*" -type d | sort -V | head -n -3 | xargs rm -rf'

# Remove pip cache
ssh root@$VULTR_IP 'rm -rf ~/.cache/pip'

# Remove HuggingFace cache (large model downloads)
ssh root@$VULTR_IP 'rm -rf ~/.cache/huggingface/hub'
```

### 8.6 Download Issues

```bash
# Connection reset during download
# Use rsync instead:
rsync -avzP --partial root@$VULTR_IP:/root/L4D2-AI-Architect/exports/ ./exports/

# Very slow download
# Compress first:
ssh root@$VULTR_IP 'tar -czvf /tmp/exports.tar.gz L4D2-AI-Architect/exports/'
scp root@$VULTR_IP:/tmp/exports.tar.gz ./

# SCP hanging
# Add keep-alive:
scp -o ServerAliveInterval=60 -o ServerAliveCountMax=3 root@$VULTR_IP:/path/to/file ./
```

### 8.7 Python/Dependencies Issues

```bash
# Module not found
ssh root@$VULTR_IP 'cd L4D2-AI-Architect && source venv/bin/activate && pip install -r requirements.txt'

# Unsloth not working
ssh root@$VULTR_IP 'pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --force-reinstall'

# Flash attention fails
ssh root@$VULTR_IP 'pip install flash-attn --no-build-isolation'
# If still fails, disable in config:
ssh root@$VULTR_IP 'sed -i "s/use_flash_attention_2: true/use_flash_attention_2: false/" L4D2-AI-Architect/configs/unsloth_config_a100.yaml'
```

---

## Quick Reference Card

```bash
# === SETUP ===
export VULTR_IP="YOUR_IP_HERE"
ssh root@$VULTR_IP

# === TRAINING ===
tmux new -s training
./run_vultr_training.sh 3 8  # epochs, batch_size
# Detach: Ctrl+B then D

# === MONITOR ===
watch -n 2 nvidia-smi
tmux attach -t training

# === EXPORT ===
python scripts/training/export_gguf_cpu.py \
  --adapter model_adapters/l4d2-mistral-v13-lora/final \
  --create-modelfile

# === DOWNLOAD (from local) ===
scp -r root@$VULTR_IP:/root/L4D2-AI-Architect/exports/*/gguf/ ./exports/

# === LOCAL OLLAMA ===
cd exports/gguf && ollama create l4d2-code-v13 -f Modelfile
ollama run l4d2-code-v13

# === CLEANUP ===
vultr-cli instance delete <ID>
```

---

*Last Updated: January 2026*
*Version: 1.0*
*Target: L4D2-AI-Architect GPU Training*
