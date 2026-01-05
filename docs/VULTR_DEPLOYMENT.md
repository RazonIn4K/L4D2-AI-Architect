# Vultr GPU Instance Deployment Guide

This guide walks through deploying L4D2-AI-Architect on a Vultr GPU instance.

## Prerequisites

- Vultr account with GPU instance access
- SSH key configured in Vultr
- GitHub personal access token (for data collection)

## Step 1: Create Vultr GPU Instance

### Recommended Instances

| GPU Model | VRAM | Use Case | Price/hr |
|-----------|------|----------|----------|
| RTX A40 | 48GB | Best for 7B-13B models | ~$2.50 |
| L40S | 48GB | Faster inference | ~$3.00 |
| A100 | 80GB | Large models (34B+) | ~$4.50 |

### Instance Setup

1. Log into Vultr Dashboard
2. Click "Deploy New Instance"
3. Select:
   - **Type**: GPU Instance
   - **Location**: Choose nearest datacenter
   - **Image**: Ubuntu 22.04 LTS with NVIDIA Drivers
   - **GPU**: A40 (recommended for cost/performance)
   - **Storage**: 100GB NVMe minimum
   - **SSH Key**: Add your public key

## Step 2: Initial Server Setup

SSH into your instance:
```bash
ssh root@YOUR_VULTR_IP
```

Download and run the setup script:
```bash
# Download setup script
wget https://raw.githubusercontent.com/RazonIn4K/L4D2-AI-Architect/main/scripts/utils/vultr_setup.sh
chmod +x vultr_setup.sh

# Run setup (takes ~10-15 minutes)
sudo ./vultr_setup.sh
```

## Step 3: Configure Environment

After setup completes:

1. Clone the repository:
```bash
cd /root
git clone https://github.com/RazonIn4K/L4D2-AI-Architect.git
cd L4D2-AI-Architect
```

2. Configure environment variables:
```bash
cp .env.example .env
nano .env

# Set these values:
GITHUB_TOKEN=your_github_token_here
WANDB_API_KEY=your_wandb_key_here  # Optional
```

3. Activate the environment:
```bash
source activate.sh
```

## Step 4: Data Collection

Collect training data:
```bash
# Quick collection (100 repos, 50 wiki pages)
MAX_REPOS=100 MAX_PAGES=50 ./run_scraping.sh

# Full collection (500 repos, 200 wiki pages) - takes ~1 hour
./run_scraping.sh
```

Monitor progress:
```bash
tail -f logs/scraping.log
```

## Step 5: Model Training

### Start Training

```bash
# Start in tmux for persistence
tmux new -s training

# Run training
./run_training.sh --config configs/unsloth_config.yaml

# Detach from tmux: Ctrl+B, then D
# Reattach later: tmux attach -t training
```

### Monitor Training

Open a new SSH session:
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Or use the monitor script
./monitor_training.sh
```

### Access TensorBoard

From your local machine:
```bash
# SSH tunnel for TensorBoard
ssh -L 6006:127.0.0.1:6006 root@YOUR_VULTR_IP

# Then open in browser: http://localhost:6006
```

## Step 6: Deploy Services

### Start Copilot Inference Server

```bash
# As systemd service (recommended)
sudo systemctl start l4d2-copilot
sudo systemctl status l4d2-copilot

# Or manually
python scripts/inference/copilot_server.py --host 0.0.0.0 --port 8000
```

Test the API:
```bash
# From the server
curl http://localhost:8000/health

# From your local machine
curl http://YOUR_VULTR_IP:8000/health
```

### Start AI Director

```bash
# As systemd service
sudo systemctl start l4d2-director
sudo systemctl status l4d2-director

# Or manually
python scripts/director/director.py --mode rule --host 0.0.0.0
```

## Step 7: L4D2 Server Setup (Optional)

If you want to test RL agents:

```bash
# Start L4D2 dedicated server in Docker
docker run -d \
  --name l4d2-server \
  -p 27015:27015/udp \
  -p 27050:27050/tcp \
  -v ~/l4d2-server:/home/steam/l4d2-server \
  cm2network/l4d2

# Install SourceMod plugin
docker exec -it l4d2-server bash
cd /home/steam/l4d2-server/left4dead2/addons/sourcemod/scripting
# Copy and compile l4d2_ai_bridge.sp
```

## Step 8: Testing

### Test Data Pipeline
```bash
python -c "
from scripts.scrapers import scrape_github_plugins
from scripts.training import prepare_dataset
print('✓ Imports working')
"
```

### Test GPU
```bash
python -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

### Test Inference
```bash
# Generate a template
python scripts/inference/copilot_cli.py template plugin

# Test completion (after training)
python scripts/inference/copilot_cli.py complete \
  --prompt "public void OnPluginStart()" \
  --language sourcepawn
```

## Step 9: Model Export

After training completes:

```bash
# Export to GGUF format for llama.cpp
python scripts/training/export_model.py \
  --model model_adapters/l4d2-code-lora \
  --format gguf \
  --quantize q4_k_m \
  --output exports/l4d2-copilot-q4.gguf

# Download to local machine
scp root@YOUR_VULTR_IP:/root/L4D2-AI-Architect/exports/l4d2-copilot-q4.gguf .
```

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size in configs/unsloth_config.yaml
per_device_train_batch_size: 2  # Was 4
gradient_accumulation_steps: 8  # Was 4
```

### Training Crashes
```bash
# Check logs
journalctl -xe
tail -f logs/training.log

# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"
```

### Slow Training
```bash
# Check GPU utilization
nvidia-smi dmon -s u

# Should be >90% for efficient training
# If low, increase batch size
```

## Cost Management

### Monitoring Usage
- Check Vultr dashboard for current charges
- GPU instances bill per hour (partial hours rounded up)

### Cost-Saving Tips
1. **Snapshot before destroying**: Save your trained models
2. **Use spot instances**: 50% cheaper but can be interrupted
3. **Schedule training**: Run overnight when you won't need to monitor
4. **Destroy when done**: Don't leave GPU instances idle

### Creating a Snapshot
```bash
# Save model and data
tar -czf l4d2-ai-backup.tar.gz \
  model_adapters/ \
  exports/ \
  data/processed/

# Upload to cloud storage
# Then create Vultr snapshot from dashboard
```

## Security Notes

- The setup script creates a non-root user `l4d2ai`
- Services run with limited privileges
- Firewall configured with UFW
- TensorBoard binds to localhost only (use SSH tunnel)
- API services should use HTTPS in production

## Next Steps

1. **Fine-tune hyperparameters**: Adjust configs/unsloth_config.yaml
2. **Implement RL training**: Connect to L4D2 server
3. **Deploy to production**: Set up HTTPS, authentication
4. **Scale inference**: Use multiple GPUs or cloud endpoints
5. **Monitor performance**: Set up logging and metrics

## Support

- GitHub Issues: https://github.com/RazonIn4K/L4D2-AI-Architect/issues
- Vultr Support: https://www.vultr.com/docs/
- PyTorch Forums: https://discuss.pytorch.org/
