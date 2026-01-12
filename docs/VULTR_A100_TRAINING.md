# L4D2-AI-Architect – Vultr A100 Training Guide

This document outlines the training workflow specifically tailored for the **Vultr Cloud GPU (1/2 NVIDIA A100, 40GB)** instance. It covers setup, training configuration, and best practices to maximize your credits.

## 1. Hardware Context

**Instance Type:** `vcg-a100-6c-60g-40vram`

| Component | Specification | Notes |
|-----------|---------------|-------|
| **GPU** | 1/2 NVIDIA A100 (40GB VRAM) | High bandwidth HBM2e, excellent for training |
| **CPU** | 6 vCPUs | Sufficient for data preprocessing |
| **RAM** | 60 GB | Plenty for large datasets |
| **Storage** | 1400 GB NVMe | No storage constraints |

### A100 vs A40 Comparison

| Feature | A100 (40GB) | A40 (48GB) |
|---------|-------------|------------|
| Memory Bandwidth | 1.6 TB/s | 696 GB/s |
| FP16 Performance | 312 TFLOPS | 150 TFLOPS |
| BF16 Support | Native | Limited |
| Tensor Cores | 3rd Gen | 3rd Gen |
| **Training Speed** | **~2x faster** | Baseline |

**Bottom line:** A100 is faster despite having 8GB less VRAM.

---

## 2. Environment Setup

### 2.1 SSH and Verify GPU

```bash
ssh root@<YOUR_VULTR_IP>

# Verify NVIDIA drivers
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

### 2.2 Clone Repository & Setup

```bash
cd /root
git clone https://github.com/YOUR_USERNAME/L4D2-AI-Architect.git
cd L4D2-AI-Architect

# Run setup
chmod +x setup.sh
./setup.sh
```

### 2.3 Install Flash Attention 2 (A100 Optimization)

The A100 benefits significantly from Flash Attention 2:

```bash
source venv/bin/activate
pip install flash-attn --no-build-isolation
```

---

## 3. A100-Optimized Training Configuration

Create `configs/unsloth_config_a100.yaml`:

```yaml
# Unsloth QLoRA Training Configuration - A100 40GB Optimized
#
# Optimized for 1/2 NVIDIA A100 (40GB VRAM)
# Key difference from A40: Higher bandwidth, native BF16, smaller VRAM

model:
  name: "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
  max_seq_length: 4096      # A100 handles large context well
  dtype: null               # Auto-detect (will use bf16)
  load_in_4bit: true

lora:
  r: 32                     # LoRA rank
  lora_alpha: 64            # 2x rank
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

  # A100 40GB: Start with batch_size 8, reduce to 4 if OOM
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 2   # Effective batch: 16

  learning_rate: 2.0e-4
  weight_decay: 0.01
  warmup_steps: 50
  lr_scheduler_type: "cosine"
  optim: "adamw_8bit"

  # CRITICAL: Use BF16 on A100 (better than FP16)
  fp16: false
  bf16: true

  logging_steps: 10
  save_steps: 100
  save_total_limit: 3
  seed: 3407
  max_grad_norm: 1.0

data:
  train_file: "combined_train.jsonl"
  val_file: "combined_val.jsonl"
  max_samples: null

output:
  dir: "l4d2-code-lora-a100"
  push_to_hub: false

advanced:
  use_flash_attention_2: true
  dataloader_num_workers: 4      # Utilize 6 vCPUs
  dataloader_pin_memory: true
  tf32: true                     # A100 supports TF32

monitoring:
  report_to: "tensorboard"
  logging_dir: "data/training_logs"
  evaluation_strategy: "steps"
  eval_steps: 100
```

---

## 4. Training Workflow

### 4.1 Upload Training Data

From your local machine:
```bash
scp data/processed/combined_train.jsonl root@<VULTR_IP>:/root/L4D2-AI-Architect/data/processed/
scp data/processed/combined_val.jsonl root@<VULTR_IP>:/root/L4D2-AI-Architect/data/processed/
```

### 4.2 Start Training in tmux

```bash
# Install tmux if not present
apt-get install tmux -y

# Create persistent session
tmux new -s training

# Activate environment and start training
cd /root/L4D2-AI-Architect
source venv/bin/activate

# Option 1: Use Unsloth (fastest, if compatible)
python scripts/training/train_unsloth.py --config configs/unsloth_config_a100.yaml

# Option 2: Use standard PyTorch training (more compatible)
python scripts/training/train_runpod.py --model mistral --batch-size 8 --epochs 3
```

**Detach:** `Ctrl+B` then `D`
**Reattach:** `tmux attach -t training`

### 4.3 Monitor Training

In a separate SSH session:

```bash
# Watch GPU usage (target: 30-38GB VRAM, >90% utilization)
watch -n 1 nvidia-smi

# View training logs
tail -f data/training_logs/l4d2-code-lora-a100/events.*
```

### 4.4 Early Verification (After ~50 Steps)

Don't wait for full training. Test generation early:

```bash
python scripts/training/train_unsloth.py \
    --test-only model_adapters/l4d2-code-lora-a100/checkpoint-50 \
    --test "Write a SourcePawn function to spawn a Tank"
```

**Success indicators:**
- Generates SourcePawn-like code
- No crashes or NaN losses
- Loss decreasing in TensorBoard

---

## 5. Expected Performance

### Training Time Estimates (971 samples, 3 epochs)

| Configuration | Batch Size | Time | VRAM Usage | Cost |
|--------------|------------|------|------------|------|
| TinyLlama 1.1B | 8 | ~15 min | ~10 GB | ~$0.15 |
| Mistral 7B (4-bit) | 8 | ~1.5 hrs | ~35 GB | ~$1.50 |
| Mistral 7B (4-bit) | 4 | ~2.5 hrs | ~25 GB | ~$2.50 |

### Memory Budget

```
Base Model (4-bit Mistral-7B): ~5 GB
LoRA Adapters:                 ~2 GB
Activations (batch=8):         ~25 GB
Optimizer States:              ~3 GB
─────────────────────────────────────
Total:                         ~35 GB (of 40 GB available)
```

---

## 6. Troubleshooting

### CUDA Out of Memory (OOM)

```bash
# Reduce batch size
python scripts/training/train_runpod.py --model mistral --batch-size 4

# Or edit config
# per_device_train_batch_size: 4
```

### Loss is NaN

```yaml
# Ensure BF16 is enabled (not FP16)
training:
  fp16: false
  bf16: true
```

### Slow Training (< 80% GPU Utilization)

```yaml
# Increase dataloader workers
advanced:
  dataloader_num_workers: 4
  dataloader_pin_memory: true
```

### Flash Attention Not Working

```bash
# Reinstall with CUDA
pip uninstall flash-attn
pip install flash-attn --no-build-isolation

# Or disable in config
advanced:
  use_flash_attention_2: false
```

---

## 7. Export & Backup (CRITICAL)

**Warning:** Vultr suspends instances when credits run out. Back up frequently!

### 7.1 Compress Model

```bash
cd /root/L4D2-AI-Architect
tar -czvf l4d2-lora-a100.tar.gz model_adapters/l4d2-code-lora-a100/
```

### 7.2 Download to Local Machine

From your **local machine**:
```bash
scp root@<VULTR_IP>:/root/L4D2-AI-Architect/l4d2-lora-a100.tar.gz ./
```

### 7.3 Export to GGUF (Optional)

```bash
python scripts/training/export_model.py \
    --input model_adapters/l4d2-code-lora-a100/final \
    --format gguf \
    --quantize q4_k_m
```

---

## 8. Cost Management

### Vultr A100 Pricing

| Resource | Rate |
|----------|------|
| 1/2 A100 Instance | ~$1.00/hr |
| Storage (1.4TB) | Included |

### Estimated Training Costs

| Task | Time | Cost |
|------|------|------|
| Full Mistral-7B training (971 samples) | ~2 hrs | ~$2 |
| With hyperparameter tuning | ~4 hrs | ~$4 |
| Multiple experiments | ~8 hrs | ~$8 |

### Cost-Saving Tips

1. **Test locally first** with TinyLlama before using GPU
2. **Use checkpoints** - resume if interrupted
3. **Export early** - don't wait until credits exhausted
4. **Destroy when done** - stop billing immediately

---

## 9. Quick Reference

```bash
# SSH to instance
ssh root@<VULTR_IP>

# Start training (always in tmux!)
tmux new -s training
cd /root/L4D2-AI-Architect
source venv/bin/activate
python scripts/training/train_runpod.py --model mistral --epochs 3

# Detach: Ctrl+B then D
# Reattach: tmux attach -t training

# Monitor GPU
watch -n 1 nvidia-smi

# Download model
scp root@<VULTR_IP>:/root/L4D2-AI-Architect/model_adapters/l4d2-code-lora-a100/ ./

# Test inference locally
python scripts/inference/test_lora.py --adapter model_adapters/l4d2-code-lora-a100
```

---

## 10. Differences from RunPod

| Aspect | Vultr A100 | RunPod A40 |
|--------|------------|------------|
| Environment | Full VPS (root access) | Container |
| Persistence | Until destroyed | Volume-based |
| Storage | 1.4TB NVMe included | Pay per GB |
| GPU | A100 40GB (faster) | A40 48GB (more VRAM) |
| Best for | Training | Training + experiments |

**Key Vultr advantage:** Persistent storage and full root access make it easier for longer training runs and custom setups.
