#!/bin/bash
# Vultr GH200 (Grace Hopper) Setup Script
# For ARM64 instances running Ubuntu 22.04
#
# IMPORTANT: GH200 uses ARM64 architecture - standard pip PyTorch gives CPU-only!
# This script uses NGC containers for proper CUDA support on ARM64.

set -e

echo "=============================================="
echo "L4D2-AI-Architect: Vultr GH200 Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
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

log_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    log_error "Please run as root (sudo ./vultr_setup_gh200.sh)"
    exit 1
fi

# Detect architecture
ARCH=$(uname -m)
log_info "Detected architecture: $ARCH"

if [ "$ARCH" != "aarch64" ]; then
    log_error "This script is for GH200 (ARM64/aarch64) only!"
    log_error "Detected: $ARCH"
    log_error "For x86_64 GPUs, use: ./vultr_setup.sh"
    exit 1
fi

# Detect GPU
log_step "Detecting GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
    log_info "GPU detected: $GPU_INFO"

    # Verify it's a GH200/H100
    if [[ "$GPU_INFO" != *"H100"* ]] && [[ "$GPU_INFO" != *"GH200"* ]]; then
        log_warn "Expected GH200/H100 GPU, found: $GPU_INFO"
        log_warn "Continuing anyway..."
    fi
else
    log_error "nvidia-smi not found! NVIDIA drivers may not be installed."
    exit 1
fi

# Update system
log_step "Updating system packages..."
apt-get update -qq
apt-get upgrade -y -qq

# Install essential packages
log_step "Installing essential packages..."
apt-get install -y -qq \
    ca-certificates \
    gnupg \
    git \
    curl \
    wget \
    htop \
    tmux \
    vim \
    docker.io

# Install NVIDIA container toolkit
log_step "Installing NVIDIA container toolkit..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/nvidia-docker/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-docker.gpg
curl -fsSL https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-docker.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-docker.list > /dev/null
apt-get update -qq
apt-get install -y -qq nvidia-container-toolkit

# Configure Docker for NVIDIA
log_step "Configuring Docker for NVIDIA GPUs..."
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# Verify Docker GPU access
log_info "Verifying Docker can access GPU..."
if docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    log_info "Docker GPU access verified!"
else
    log_error "Docker cannot access GPU. Check nvidia-container-toolkit installation."
    exit 1
fi

# Create project directory
PROJECT_DIR="/root/L4D2-AI-Architect"
if [ -d "$PROJECT_DIR" ]; then
    log_info "Project directory exists"
    cd "$PROJECT_DIR"
else
    log_info "Creating project directory..."
    mkdir -p "$PROJECT_DIR"
    log_warn "Please clone or copy your project files to $PROJECT_DIR"
fi

cd "$PROJECT_DIR"

# Pull NGC PyTorch container
log_step "Pulling NGC PyTorch container (this may take a few minutes)..."
if ! docker pull nvcr.io/nvidia/pytorch:24.02-py3; then
    log_error "Failed to pull nvcr.io/nvidia/pytorch:24.02-py3"
    log_error "If you see 'unauthorized: authentication required', log in to NGC:"
    log_error "  docker login nvcr.io"
    log_error "  Username: \$oauthtoken"
    log_error "  Password: <your NGC API key>"
    exit 1
fi

# Create data directories
log_step "Creating project directories..."
mkdir -p data/{raw,processed,training_logs}
mkdir -p model_adapters/{checkpoints,exports}
mkdir -p logs

# Create .gitkeep files
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/training_logs/.gitkeep
touch model_adapters/.gitkeep

# Create container entry script
log_step "Creating container setup script..."
cat > /root/L4D2-AI-Architect/container_setup.sh << 'CONTAINER_EOF'
#!/bin/bash
# Run inside NGC container to install project dependencies

set -e

echo "Installing L4D2-AI-Architect dependencies inside container..."

if [ ! -d "venv" ]; then
    python -m venv --system-site-packages venv
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install bitsandbytes ARM64 preview wheel
echo "Installing bitsandbytes for ARM64..."
python -m pip install --force-reinstall https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-1.33.7.preview-py3-none-manylinux_2_24_aarch64.whl

# Install Unsloth (should work on ARM64 with proper PyTorch)
echo "Installing Unsloth..."
python -m pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
python -m pip install --no-deps trl peft accelerate

echo "Installing xformers (optional)..."
if python -m pip install --no-deps xformers; then
    echo "xformers: OK"
else
    echo "xformers install failed; continuing without it"
fi

# Install other dependencies
python -m pip install datasets transformers sentencepiece protobuf pyyaml
python -m pip install tensorboard wandb
python -m pip install tqdm requests beautifulsoup4 lxml

# Verify installations
echo ""
echo "Verifying installations..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
python -c "import bitsandbytes; print('bitsandbytes: OK')"
python -c "from unsloth import FastLanguageModel; print('Unsloth: OK')"

echo ""
echo "Container setup complete!"
CONTAINER_EOF
chmod +x /root/L4D2-AI-Architect/container_setup.sh

# Create run_in_container.sh wrapper
log_step "Creating container wrapper script..."
cat > /root/L4D2-AI-Architect/run_in_container.sh << 'WRAPPER_EOF'
#!/bin/bash
# Run commands inside the NGC PyTorch container with GPU access
#
# Usage:
#   ./run_in_container.sh                    # Interactive shell
#   ./run_in_container.sh python train.py   # Run specific command
#   ./run_in_container.sh --setup           # Install dependencies

PROJECT_DIR="/root/L4D2-AI-Architect"
CONTAINER_IMAGE="nvcr.io/nvidia/pytorch:24.02-py3"

if [ "$1" == "--setup" ]; then
    echo "Running container setup..."
    docker run --gpus all -it --rm \
        -v "$PROJECT_DIR":/workspace \
        -w /workspace \
        --shm-size=16g \
        "$CONTAINER_IMAGE" \
        bash container_setup.sh
elif [ $# -eq 0 ]; then
    echo "Starting interactive container..."
    docker run --gpus all -it --rm \
        -v "$PROJECT_DIR":/workspace \
        -w /workspace \
        --shm-size=16g \
        -p 127.0.0.1:6006:6006 \
        -p 127.0.0.1:8000:8000 \
        "$CONTAINER_IMAGE" \
        bash -lc 'if [ -f venv/bin/activate ]; then source venv/bin/activate; fi; exec bash'
else
    echo "Running: $@"
    docker run --gpus all -it --rm \
        -v "$PROJECT_DIR":/workspace \
        -w /workspace \
        --shm-size=16g \
        -p 127.0.0.1:6006:6006 \
        -p 127.0.0.1:8000:8000 \
        "$CONTAINER_IMAGE" \
        bash -lc 'if [ -f venv/bin/activate ]; then source venv/bin/activate; fi; exec "$@"' -- "$@"
fi
WRAPPER_EOF
chmod +x /root/L4D2-AI-Architect/run_in_container.sh

# Create training wrapper
log_step "Creating training wrapper..."
cat > /root/L4D2-AI-Architect/run_training_gh200.sh << 'TRAIN_EOF'
#!/bin/bash
# Run training inside container with GH200-optimized config
#
# Usage:
#   ./run_training_gh200.sh                     # Use default GH200 config
#   ./run_training_gh200.sh --resume checkpoint # Resume from checkpoint

PROJECT_DIR="/root/L4D2-AI-Architect"
CONTAINER_IMAGE="nvcr.io/nvidia/pytorch:24.02-py3"

# Default to GH200 config
CONFIG="${CONFIG:-configs/unsloth_config_gh200.yaml}"

echo "Starting GH200-optimized training..."
echo "Config: $CONFIG"
echo ""

if [ ! -f "$PROJECT_DIR/venv/bin/activate" ]; then
    echo "ERROR: venv not found. Run: ./run_in_container.sh --setup"
    exit 1
fi

./run_in_container.sh python scripts/training/train_unsloth.py --config "$CONFIG" "$@"
TRAIN_EOF
chmod +x /root/L4D2-AI-Architect/run_training_gh200.sh

# Create activation script
log_step "Creating activation script..."
cat > /root/L4D2-AI-Architect/activate_gh200.sh << 'ACTIVATE_EOF'
#!/bin/bash
# Quick reference for GH200 environment

echo "=============================================="
echo "L4D2-AI-Architect GH200 Environment"
echo "=============================================="
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""
echo "Quick Commands:"
echo "  ./run_in_container.sh --setup     # Install dependencies (first time)"
echo "  ./run_in_container.sh             # Interactive container shell"
echo "  ./run_training_gh200.sh           # Start training"
echo "  ./run_training_gh200.sh --resume <checkpoint>  # Resume training"
echo ""
echo "Inside Container:"
echo "  python scripts/training/train_unsloth.py --config configs/unsloth_config_gh200.yaml"
echo "  tensorboard --logdir data/training_logs --host 0.0.0.0 --port 6006"
echo ""
echo "IMPORTANT: Always use tmux for long-running tasks!"
echo "  tmux new -s training"
echo "  tmux attach -t training"
echo ""
ACTIVATE_EOF
chmod +x /root/L4D2-AI-Architect/activate_gh200.sh

# Print GPU memory info
log_step "GPU Memory Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv

# Print disk space
log_step "Disk Space:"
df -h /

# Print summary
echo ""
echo "=============================================="
echo "GH200 SETUP COMPLETE!"
echo "=============================================="
echo ""
echo "Architecture: $ARCH (ARM64)"
echo "Container: nvcr.io/nvidia/pytorch:24.02-py3"
echo "Project: $PROJECT_DIR"
echo ""
echo "Next Steps:"
echo "  1. Copy/clone your project files to $PROJECT_DIR"
echo "  2. Run: ./run_in_container.sh --setup"
echo "  3. Start tmux: tmux new -s training"
echo "  4. Run training: ./run_training_gh200.sh"
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""
echo "IMPORTANT: GH200 instances may be preempted!"
echo "- Always use tmux for training sessions"
echo "- Checkpoints saved every 100 steps"
echo "- Resume with: ./run_training_gh200.sh --resume model_adapters/l4d2-code-lora/checkpoint-XXX"
echo ""
log_info "Run './activate_gh200.sh' for quick reference"
