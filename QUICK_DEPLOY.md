# Quick Deploy - Vultr Training

**You have expiring credits. Follow these commands exactly.**

## 1. One-Liner: SSH + Start Training

```bash
ssh root@<VULTR_IP> "cd L4D2-AI-Architect && tmux new -d -s train './run_training.sh --fast' && tmux attach -t train"
```

## 2. If Repo Not Cloned Yet

```bash
ssh root@<VULTR_IP>
git clone <YOUR_REPO> L4D2-AI-Architect && cd L4D2-AI-Architect
./setup.sh
tmux new -s train
./run_training.sh --fast
```

## 3. Fast Mode (Recommended)

```bash
# 1 epoch, smaller batch - finishes in ~30-45 min on A100
./run_training.sh --fast

# Or explicit:
python scripts/training/train_unsloth.py --config configs/unsloth_config_a100.yaml --epochs 1
```

## 4. Check Status

```bash
# Reattach to training session
tmux attach -t train

# Check GPU usage (new terminal)
nvidia-smi

# View logs
tail -f data/training_logs/*/events*
```

## 5. Download Trained Model

```bash
# From your LOCAL machine:
scp -r root@<VULTR_IP>:L4D2-AI-Architect/model_adapters/l4d2-*-lora/final ./

# Or export to GGUF first (on server):
python scripts/training/export_gguf_cpu.py --adapter model_adapters/l4d2-mistral-v10plus-lora/final
scp -r root@<VULTR_IP>:L4D2-AI-Architect/exports/ ./
```

## 6. Emergency: Detach Without Stopping

Press `Ctrl+B` then `D` to detach tmux. Training continues.

---

**GPU Guide:**
- A100 (40GB): `--batch-size 8` (~2 hrs full, ~40 min fast)
- A40 (48GB): `--batch-size 4` (~3 hrs full, ~1 hr fast)
- GH200: Use `./run_training_gh200.sh` instead
