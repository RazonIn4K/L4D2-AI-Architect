# L4D2-AI-Architect Deployment Checklist

Complete, copy-paste ready checklist for training the L4D2 code model on Vultr A100.

---

## Pre-Deployment Checklist

### Local Machine Preparation

- [ ] **GitHub repo pushed with latest changes**
  ```bash
  cd /path/to/L4D2-AI-Architect
  git status
  git add -A && git commit -m "Pre-training sync"
  git push origin main
  ```

- [ ] **Vultr account with credits**
  - Minimum recommended: $10 (covers ~4 hours of training buffer)
  - Check balance: [Vultr Billing Dashboard](https://my.vultr.com/billing/)

- [ ] **SSH key configured**
  ```bash
  # Generate if needed
  ssh-keygen -t ed25519 -C "vultr-training"

  # Copy public key to Vultr
  cat ~/.ssh/id_ed25519.pub
  # Paste at: Vultr Dashboard -> Account -> SSH Keys -> Add SSH Key
  ```

- [ ] **Training data verified (2576 examples in v14)**
  ```bash
  wc -l data/processed/l4d2_train_v14.jsonl
  # Expected: 2576

  # Preview first example
  head -1 data/processed/l4d2_train_v14.jsonl | python -m json.tool
  ```

---

## Vultr Instance Setup

### Step 1: Create A100 Instance

- [ ] **Navigate to Cloud GPU**
  ```
  Vultr Dashboard -> Products -> Cloud GPU -> Deploy Instance
  ```

- [ ] **Select configuration**
  | Setting | Value |
  |---------|-------|
  | Type | Cloud GPU |
  | Location | New Jersey (or nearest) |
  | GPU | NVIDIA A100 (1/2 GPU, 40GB) |
  | Plan | `vcg-a100-6c-60g-40vram` (~$2.50/hr) |
  | OS | Ubuntu 22.04 LTS x64 |
  | SSH Key | Select your key |
  | Hostname | `l4d2-training` |

- [ ] **Deploy and wait 2-5 minutes**
  - Copy the IP address when ready

### Step 2: SSH into Instance

- [ ] **Set environment variable and connect**
  ```bash
  export VULTR_IP="YOUR_INSTANCE_IP"
  ssh root@$VULTR_IP
  ```

- [ ] **Verify GPU is detected**
  ```bash
  nvidia-smi
  ```
  Expected: `NVIDIA A100-SXM4-40GB` with ~40960MiB memory

### Step 3: Run Quickstart Setup

- [ ] **Install system dependencies**
  ```bash
  apt-get update && apt-get install -y tmux git python3-pip python3-venv htop
  ```

- [ ] **Clone repository**
  ```bash
  cd /root
  git clone https://github.com/YOUR_USERNAME/L4D2-AI-Architect.git
  cd L4D2-AI-Architect
  ```

- [ ] **Create virtual environment**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

- [ ] **Install Python dependencies**
  ```bash
  pip install --upgrade pip
  pip install -r requirements.txt
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  pip install flash-attn --no-build-isolation
  ```
  Note: Flash Attention takes ~5 minutes to build

- [ ] **Verify PyTorch CUDA**
  ```bash
  python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
  ```
  Expected: `CUDA: True`, `GPU: NVIDIA A100-SXM4-40GB`

### Step 4: Upload Training Data

- [ ] **From local machine (new terminal)**
  ```bash
  export VULTR_IP="YOUR_INSTANCE_IP"

  # Upload v14 training data
  scp data/processed/l4d2_train_v14.jsonl root@$VULTR_IP:/root/L4D2-AI-Architect/data/processed/

  # Upload validation data
  scp data/processed/combined_val.jsonl root@$VULTR_IP:/root/L4D2-AI-Architect/data/processed/
  ```

- [ ] **Verify upload (on Vultr)**
  ```bash
  wc -l /root/L4D2-AI-Architect/data/processed/l4d2_train_v14.jsonl
  # Expected: 2576
  ```

---

## Training

### Step 5: Start Training

- [ ] **Start tmux session (CRITICAL - prevents loss if SSH disconnects)**
  ```bash
  tmux new -s training
  ```

- [ ] **Activate environment and start training**
  ```bash
  cd /root/L4D2-AI-Architect
  source venv/bin/activate

  python scripts/training/train_unsloth.py --config configs/unsloth_config_v14.yaml
  ```

- [ ] **Detach from tmux (training continues in background)**
  ```
  Press: Ctrl+B then D
  ```

---

## Training Monitoring

### GPU Monitoring (Second SSH Session)

- [ ] **Open new terminal and SSH in**
  ```bash
  ssh root@$VULTR_IP
  ```

- [ ] **Watch GPU utilization**
  ```bash
  nvidia-smi -l 1
  ```
  Target: 90%+ GPU utilization, ~35GB VRAM usage

- [ ] **Reattach to training session**
  ```bash
  tmux attach -t training
  ```

### TensorBoard (Optional)

- [ ] **Start TensorBoard on Vultr**
  ```bash
  tensorboard --logdir data/training_logs --host 127.0.0.1 --port 6006
  ```

- [ ] **Create SSH tunnel from local machine**
  ```bash
  ssh -L 6006:127.0.0.1:6006 root@$VULTR_IP
  ```

- [ ] **Open in browser**
  ```
  http://localhost:6006
  ```

### Expected Training Progress

| Epoch | Time | Loss | VRAM |
|-------|------|------|------|
| 1/3 | ~60 min | ~1.5 -> 0.8 | ~35GB |
| 2/3 | ~120 min | ~0.8 -> 0.5 | ~35GB |
| 3/3 | ~180 min | ~0.5 -> 0.3 | ~35GB |

---

## Post-Training

### Step 6: Verify Model Saved

- [ ] **Check training output**
  ```bash
  ls -la /root/L4D2-AI-Architect/model_adapters/l4d2-mistral-v14-lora/
  ```
  Expected files:
  ```
  final/
    adapter_config.json
    adapter_model.safetensors
    tokenizer.json
    tokenizer_config.json
  lora_adapter/
  training_info.json
  ```

- [ ] **Quick model test**
  ```bash
  python scripts/training/train_unsloth.py \
      --test-only model_adapters/l4d2-mistral-v14-lora/final \
      --test "Write a SourcePawn function to spawn a Tank"
  ```

### Step 7: Export to GGUF

- [ ] **Export with Q4_K_M quantization**
  ```bash
  python scripts/training/export_gguf_cpu.py \
      --adapter model_adapters/l4d2-mistral-v14-lora/final \
      --output exports/l4d2-v14 \
      --quantize q4_k_m \
      --create-modelfile
  ```

- [ ] **Verify export**
  ```bash
  ls -lh /root/L4D2-AI-Architect/exports/l4d2-v14/gguf/
  ```
  Expected: `l4d2-mistral-v14-q4_k_m.gguf` (~4-5 GB)

### Step 8: Download to Local Machine

- [ ] **From local machine**
  ```bash
  export VULTR_IP="YOUR_INSTANCE_IP"

  # Create local exports directory
  mkdir -p ~/L4D2-AI-Architect/exports/l4d2-v14

  # Download GGUF and Modelfile
  scp -r root@$VULTR_IP:/root/L4D2-AI-Architect/exports/l4d2-v14/gguf/ \
      ~/L4D2-AI-Architect/exports/l4d2-v14/

  # Backup LoRA adapter
  scp -r root@$VULTR_IP:/root/L4D2-AI-Architect/model_adapters/l4d2-mistral-v14-lora/ \
      ~/L4D2-AI-Architect/model_adapters/
  ```

### Step 9: Install in Ollama

- [ ] **Install Ollama (if not installed)**
  ```bash
  # macOS
  brew install ollama

  # Linux
  curl -fsSL https://ollama.com/install.sh | sh
  ```

- [ ] **Create the model**
  ```bash
  cd ~/L4D2-AI-Architect/exports/l4d2-v14/gguf
  ollama create l4d2-code-v14 -f Modelfile
  ```

- [ ] **Verify installation**
  ```bash
  ollama list
  # Should show: l4d2-code-v14
  ```

### Step 10: Test with copilot_cli.py

- [ ] **Interactive chat mode**
  ```bash
  cd ~/L4D2-AI-Architect
  source venv/bin/activate
  python scripts/inference/copilot_cli.py chat --model l4d2-code-v14
  ```

- [ ] **Single prompt test**
  ```bash
  python scripts/inference/copilot_cli.py ollama --model l4d2-code-v14 \
      --prompt "Write a SourcePawn function to heal all survivors within 500 units"
  ```

- [ ] **Direct Ollama test**
  ```bash
  ollama run l4d2-code-v14 "Write a function to spawn a Tank at a random survivor position"
  ```

---

## Cleanup

### Destroy Vultr Instance (Stop Billing)

- [ ] **Via Dashboard**
  ```
  Vultr Dashboard -> Your Instance -> Settings -> Destroy
  ```

- [ ] **Or via CLI**
  ```bash
  vultr-cli instance list
  vultr-cli instance destroy <INSTANCE_ID>
  ```

---

## Cost Tracking

### Record Times

| Milestone | Time | Notes |
|-----------|------|-------|
| Instance created | ____:____ | |
| Setup complete | ____:____ | |
| Training started | ____:____ | |
| Training finished | ____:____ | |
| Export complete | ____:____ | |
| Instance destroyed | ____:____ | |

### Calculate Actual Cost

| Phase | Duration | Cost @ $2.50/hr |
|-------|----------|-----------------|
| Instance provisioning | ___ min | $_____ |
| Environment setup | ___ min | $_____ |
| Training (2576 samples, 3 epochs) | ___ min | $_____ |
| GGUF export | ___ min | $_____ |
| Download & cleanup | ___ min | $_____ |
| **Total** | **___ hours** | **$_____** |

### Expected Costs (Reference)

| Task | Time | Cost @ $2.50/hr |
|------|------|-----------------|
| Instance setup | 15 min | ~$0.65 |
| Training (2576 samples, 3 epochs) | 3-4 hrs | ~$7.50-$10.00 |
| Export to GGUF | 15-30 min | ~$0.65-$1.25 |
| **Total** | **~3.5-5 hours** | **$8.75-$12.50** |

---

## Troubleshooting Quick Reference

### CUDA Out of Memory (OOM)
```bash
# Reduce batch size in config or via CLI
python scripts/training/train_unsloth.py \
    --config configs/unsloth_config_v14.yaml \
    --batch-size 4
```

### Training Loss is NaN
Ensure BF16 is enabled in config (not FP16):
```yaml
training:
  fp16: false
  bf16: true
```

### SSH Disconnected During Training
Training continues in tmux. Reconnect:
```bash
ssh root@$VULTR_IP
tmux attach -t training
```

### Resume from Checkpoint
```bash
python scripts/training/train_unsloth.py \
    --config configs/unsloth_config_v14.yaml \
    --resume model_adapters/l4d2-mistral-v14-lora/checkpoint-XXX
```

### Flash Attention Not Working
```bash
pip uninstall flash-attn
pip install flash-attn --no-build-isolation

# Or disable in config:
# advanced:
#   use_flash_attention_2: false
```

---

## Quick Reference Card

```bash
# === LOCAL MACHINE ===
export VULTR_IP="YOUR_INSTANCE_IP"

# Upload data
scp data/processed/l4d2_train_v14.jsonl root@$VULTR_IP:/root/L4D2-AI-Architect/data/processed/
scp data/processed/combined_val.jsonl root@$VULTR_IP:/root/L4D2-AI-Architect/data/processed/

# Download model after training
scp -r root@$VULTR_IP:/root/L4D2-AI-Architect/exports/l4d2-v14/gguf/ ./exports/

# === VULTR INSTANCE ===
ssh root@$VULTR_IP

# One-time setup
apt-get update && apt-get install -y tmux git python3-venv
cd /root && git clone <REPO_URL> L4D2-AI-Architect && cd L4D2-AI-Architect
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install flash-attn --no-build-isolation

# Training
tmux new -s training
source venv/bin/activate
python scripts/training/train_unsloth.py --config configs/unsloth_config_v14.yaml
# Detach: Ctrl+B then D

# Monitor
watch -n 2 nvidia-smi
tmux attach -t training

# Export
python scripts/training/export_gguf_cpu.py \
    --adapter model_adapters/l4d2-mistral-v14-lora/final \
    --output exports/l4d2-v14 \
    --quantize q4_k_m \
    --create-modelfile

# === LOCAL OLLAMA ===
cd exports/l4d2-v14/gguf
ollama create l4d2-code-v14 -f Modelfile
ollama run l4d2-code-v14
```

---

*Last updated: January 2026*
*Dataset: l4d2_train_v14.jsonl (2576 examples)*
*Target model: l4d2-code-v14*
