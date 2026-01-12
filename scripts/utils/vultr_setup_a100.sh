#!/bin/bash
# Vultr A100 Setup Script for L4D2-AI-Architect
#
# Usage: Run this on a fresh Vultr A100 (40GB) instance
#   SECURITY NOTE: Never use 'curl | bash' - always download and inspect scripts first!
#   scp vultr_setup_a100.sh root@<IP>:/root/ && ssh root@<IP> 'bash /root/vultr_setup_a100.sh'

set -e

echo "=============================================="
echo "L4D2-AI-Architect - Vultr A100 Setup"
echo "=============================================="

# Check for A100 GPU
echo "Checking GPU..."
if ! nvidia-smi | grep -qi "A100"; then
    echo "WARNING: A100 GPU not detected. This script is optimized for A100."
    echo "Detected GPU:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ A100 GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# System updates
echo ""
echo "Updating system packages..."
apt-get update -qq
apt-get install -y -qq git tmux htop python3-pip python3-venv > /dev/null 2>&1
echo "✓ System packages installed"

# Clone repository if not exists
REPO_DIR="/root/L4D2-AI-Architect"
if [ ! -d "$REPO_DIR" ]; then
    echo ""
    echo "Cloning repository..."
    # User should replace with their repo URL
    echo "ERROR: Repository not found at $REPO_DIR"
    echo "Please clone the repository first:"
    echo "  git clone <YOUR_REPO_URL> $REPO_DIR"
    exit 1
fi

cd "$REPO_DIR"
echo "✓ Working in $REPO_DIR"

# Create virtual environment
echo ""
echo "Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA..."
pip install --quiet --upgrade pip
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'✓ PyTorch {torch.__version__} with CUDA {torch.version.cuda}')"

# Install dependencies
echo "Installing training dependencies..."
pip install --quiet transformers datasets peft trl bitsandbytes accelerate tensorboard

# Install Flash Attention 2 (A100 optimization)
echo "Installing Flash Attention 2 (this may take a few minutes)..."
pip install --quiet flash-attn --no-build-isolation 2>/dev/null || {
    echo "⚠ Flash Attention 2 install failed - training will work without it"
}

# Verify bitsandbytes
python3 -c "import bitsandbytes; print('✓ bitsandbytes installed')" 2>/dev/null || {
    echo "⚠ bitsandbytes import failed - reinstalling..."
    pip install --quiet --force-reinstall bitsandbytes
}

# Check for training data
echo ""
echo "Checking training data..."
if [ -f "data/processed/combined_train.jsonl" ]; then
    TRAIN_COUNT=$(wc -l < data/processed/combined_train.jsonl)
    echo "✓ Training data found: $TRAIN_COUNT samples"
else
    echo "✗ Training data not found!"
    echo "  Please upload to: data/processed/combined_train.jsonl"
fi

if [ -f "data/processed/combined_val.jsonl" ]; then
    VAL_COUNT=$(wc -l < data/processed/combined_val.jsonl)
    echo "✓ Validation data found: $VAL_COUNT samples"
else
    echo "⚠ Validation data not found (optional)"
fi

# Create output directories
mkdir -p model_adapters data/training_logs

# Print summary
echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "GPU Configuration:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""
echo "Next steps:"
echo ""
echo "1. Upload training data (if not already done):"
echo "   scp data/processed/*.jsonl root@<IP>:$REPO_DIR/data/processed/"
echo ""
echo "2. Start training in tmux (recommended):"
echo "   tmux new -s training"
echo "   source venv/bin/activate"
echo ""
echo "3. Quick validation test (20 steps):"
echo "   python scripts/training/train_runpod.py --model mistral --max-steps 20"
echo ""
echo "4. Full training (~1.5-2 hours):"
echo "   python scripts/training/train_runpod.py --model mistral --batch-size 8 --epochs 3"
echo ""
echo "5. Monitor in another terminal:"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "6. After training, download model:"
echo "   tar -czvf l4d2-lora.tar.gz model_adapters/"
echo ""
