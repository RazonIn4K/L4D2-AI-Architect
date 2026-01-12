# Vultr A100 Training - Quick Start Guide

Deploy a local Mistral-7B model fine-tuned on L4D2 SourcePawn data.

## Cost Estimate
- **A100 40GB**: ~$2.50/hr × 2.5 hours = **~$6.25 total**
- Training: 671 examples, 3 epochs, ~2-2.5 hours

## Prerequisites

### 1. Install Vultr CLI
```bash
brew install vultr-cli
```

### 2. Get Vultr API Key
1. Go to https://my.vultr.com/settings/#settingsapi
2. Enable API Access
3. Copy the API Key

### 3. Configure API Key
```bash
# Option A: Add to Doppler
doppler secrets set VULTR_API_KEY="your-key" --project local-mac-work --config dev_personal

# Option B: Export directly
export VULTR_API_KEY="your-key"
```

## Deploy & Train

### One-Command Deployment
```bash
# With Doppler
doppler run --project local-mac-work --config dev_personal -- ./scripts/vultr/deploy_and_train.sh

# With exported key
./scripts/vultr/deploy_and_train.sh
```

### Manual Deployment

If you prefer to deploy manually via Vultr web console:

1. **Create Instance**
   - Go to https://my.vultr.com/deploy/
   - Select: Cloud GPU → NVIDIA A100 (1/2) 40GB
   - Region: New Jersey (ewr) or closest
   - OS: Ubuntu 24.04 LTS
   - Add your SSH key

2. **SSH to Instance**
   ```bash
   ssh root@<VULTR_IP>
   ```

3. **Clone & Setup**
   ```bash
   git clone https://github.com/yourusername/L4D2-AI-Architect.git
   cd L4D2-AI-Architect

   # Copy training data from local machine
   # (Run this on your LOCAL machine)
   scp data/processed/l4d2_train_v9.jsonl root@<VULTR_IP>:/root/L4D2-AI-Architect/data/processed/
   ```

4. **Start Training**
   ```bash
   ./run_vultr_training.sh
   ```

5. **Monitor**
   ```bash
   tmux attach -t training
   ```

## After Training

### Download Model
```bash
# From local machine
scp root@<VULTR_IP>:/root/L4D2-AI-Architect/exports/l4d2-mistral-v9.gguf ./exports/
```

### Setup Ollama
```bash
# Copy Modelfile template
cp scripts/vultr/Modelfile.template Modelfile

# Create model in Ollama
ollama create l4d2-mistral -f Modelfile

# Test
ollama run l4d2-mistral "Write a function to heal all survivors"
```

### Destroy Instance (Save Costs!)
```bash
./scripts/vultr/deploy_and_train.sh --destroy
# Or via web console
```

## Commands Reference

| Command | Description |
|---------|-------------|
| `./scripts/vultr/deploy_and_train.sh` | Full deploy + train |
| `./scripts/vultr/deploy_and_train.sh --status` | Check training progress |
| `./scripts/vultr/deploy_and_train.sh --ssh` | SSH to instance |
| `./scripts/vultr/deploy_and_train.sh --download` | Download trained model |
| `./scripts/vultr/deploy_and_train.sh --destroy` | Terminate instance |

## Troubleshooting

### "No GPU found"
Ensure you selected A100 GPU plan, not regular compute.

### Training OOM
Reduce batch size in `configs/unsloth_config_a100.yaml`.

### Model not exporting
Check training completed: `tmux attach -t training`
