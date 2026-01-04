#!/bin/bash
#
# Vultr GPU Instance Setup Script
# 
# Configures a fresh Vultr GPU instance for L4D2-AI-Architect training.
# Supports A100, A40, and L40S instances.
#
# Usage:
#   chmod +x vultr_setup.sh
#   ./vultr_setup.sh
#
# Note: Run as root on a fresh Vultr instance
#

set -e

echo "=============================================="
echo "L4D2-AI-Architect: Vultr GPU Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    log_error "Please run as root (sudo ./vultr_setup.sh)"
    exit 1
fi

# Detect GPU
log_info "Detecting GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
    log_info "GPU detected: $GPU_INFO"
else
    log_warn "nvidia-smi not found, checking for NVIDIA drivers..."
fi

# Update system
log_info "Updating system packages..."
apt-get update -qq
apt-get upgrade -y -qq

# Install essential packages
log_info "Installing essential packages..."
apt-get install -y -qq \
    git \
    curl \
    wget \
    htop \
    tmux \
    vim \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev

# Check CUDA installation (Vultr GPU images usually have this pre-installed)
if [ -d "/usr/local/cuda" ]; then
    log_info "CUDA found at /usr/local/cuda"
    CUDA_VERSION=$(cat /usr/local/cuda/version.txt 2>/dev/null || nvcc --version | grep release | awk '{print $5}' | cut -d',' -f1)
    log_info "CUDA version: $CUDA_VERSION"
else
    log_warn "CUDA not found - checking if already installed elsewhere..."
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep release | awk '{print $5}' | cut -d',' -f1)
        log_info "CUDA version: $CUDA_VERSION"
    else
        log_error "CUDA not installed! Vultr GPU images should have CUDA pre-installed."
        log_error "Please use a Vultr GPU image or install CUDA manually."
        exit 1
    fi
fi

# Verify NVIDIA drivers
log_info "Verifying NVIDIA drivers..."
if ! nvidia-smi &> /dev/null; then
    log_error "NVIDIA drivers not working properly!"
    exit 1
fi

# Create project directory
PROJECT_DIR="/root/L4D2-AI-Architect"
if [ -d "$PROJECT_DIR" ]; then
    log_info "Project directory exists, updating..."
    cd "$PROJECT_DIR"
    git pull origin main 2>/dev/null || log_warn "Could not pull updates (repo may not be set up)"
else
    log_info "Cloning project repository..."
    # Replace with actual repo URL when available
    # git clone https://github.com/YOUR_USERNAME/L4D2-AI-Architect.git "$PROJECT_DIR"
    mkdir -p "$PROJECT_DIR"
    log_warn "Repository not configured - please clone manually or copy files"
fi

cd "$PROJECT_DIR"

# Create virtual environment
log_info "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
log_info "Installing PyTorch with CUDA support..."
# Detect CUDA version and install appropriate PyTorch
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d'.' -f1)
if [ "$CUDA_MAJOR" -ge 12 ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
elif [ "$CUDA_MAJOR" -ge 11 ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    log_error "CUDA version too old. Need CUDA 11.8 or newer."
    exit 1
fi

# Verify PyTorch CUDA
log_info "Verifying PyTorch CUDA support..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    log_error "PyTorch CUDA not working! Check installation."
    exit 1
fi

# Install Unsloth and dependencies
log_info "Installing Unsloth and training dependencies..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft accelerate bitsandbytes
pip install datasets transformers sentencepiece protobuf
pip install tensorboard wandb

# Install RL dependencies
log_info "Installing RL training dependencies..."
pip install "stable-baselines3[extra]" gymnasium

# Install scraping dependencies
log_info "Installing data collection dependencies..."
pip install requests beautifulsoup4 lxml

# Install additional utilities
pip install pyyaml tqdm

# Create directory structure
log_info "Creating project directories..."
mkdir -p data/{raw,processed,training_logs}
mkdir -p model_adapters/{checkpoints,exports}
mkdir -p logs

# Create .gitkeep files
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/training_logs/.gitkeep
touch model_adapters/.gitkeep

# Setup TensorBoard
log_info "Setting up TensorBoard..."
cat > /etc/systemd/system/tensorboard.service << 'EOF'
[Unit]
Description=TensorBoard
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/L4D2-AI-Architect
ExecStart=/root/L4D2-AI-Architect/venv/bin/tensorboard --logdir=data/training_logs --host=127.0.0.1 --port=6006
Restart=always

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable tensorboard
systemctl start tensorboard

# Print GPU memory info
log_info "GPU Memory Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv

# Print disk space
log_info "Disk Space:"
df -h /

# Create activation script
cat > activate.sh << 'EOF'
#!/bin/bash
source /root/L4D2-AI-Architect/venv/bin/activate
cd /root/L4D2-AI-Architect
export CUDA_VISIBLE_DEVICES=0
echo "Environment activated! GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
EOF
chmod +x activate.sh

# Create convenience scripts
cat > run_training.sh << 'EOF'
#!/bin/bash
source /root/L4D2-AI-Architect/venv/bin/activate
cd /root/L4D2-AI-Architect

echo "Starting Unsloth training..."
python scripts/training/train_unsloth.py "$@"
EOF
chmod +x run_training.sh

cat > run_scraping.sh << 'EOF'
#!/bin/bash
source /root/L4D2-AI-Architect/venv/bin/activate
cd /root/L4D2-AI-Architect

echo "Running data collection..."
python scripts/scrapers/scrape_github_plugins.py --max-repos 500
python scripts/scrapers/scrape_valve_wiki.py --max-pages 200
python scripts/training/prepare_dataset.py
EOF
chmod +x run_scraping.sh

# Print summary
echo ""
echo "=============================================="
echo "SETUP COMPLETE!"
echo "=============================================="
echo ""
echo "Quick Start:"
echo "  1. Activate environment:  source activate.sh"
echo "  2. Run data collection:   ./run_scraping.sh"
echo "  3. Start training:        ./run_training.sh"
echo ""
echo "TensorBoard: http://$(hostname -I | awk '{print $1}'):6006"
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""
echo "Project Directory: $PROJECT_DIR"
echo "Python Environment: $PROJECT_DIR/venv"
echo ""
echo "IMPORTANT: Your Vultr credits expire in 5 days!"
echo "Make sure to save your models before credits expire!"
echo ""
log_info "Run 'tmux' to start a persistent session for training"
