#!/bin/bash
# =============================================================================
# L4D2-AI-Architect: Vultr A100 Quick Start Script
# =============================================================================
#
# Self-contained script for training on Vultr GPU instances.
# Handles complete setup, training, and model export.
#
# Usage (remote - SECURITY WARNING):
#   NEVER use 'curl | bash' - always download and inspect scripts first!
#   Instead: wget <URL> && less vultr_quickstart.sh && bash vultr_quickstart.sh
#
# Usage (local deploy - RECOMMENDED):
#   scp scripts/utils/vultr_quickstart.sh root@VULTR_IP:~/
#   ssh root@VULTR_IP "./vultr_quickstart.sh"
#
# Usage (on the instance):
#   ./vultr_quickstart.sh                    # Full setup + training
#   ./vultr_quickstart.sh --train-only       # Skip setup, start training
#   ./vultr_quickstart.sh --export-only      # Just export existing model
#   ./vultr_quickstart.sh --status           # Check training status
#   ./vultr_quickstart.sh --help             # Show help
#
# Prerequisites:
#   - Vultr Cloud GPU instance (A100 40GB recommended, A40 48GB works too)
#   - Ubuntu 22.04/24.04 with NVIDIA drivers pre-installed
#   - Internet access for git clone and pip install
#
# Cost Estimate (A100 @ ~$2.50/hr):
#   - Setup: ~10 minutes
#   - Training: ~2 hours
#   - Total: ~$6-8
#
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

# Repository settings - UPDATE THESE for your fork
REPO_URL="${L4D2_REPO_URL:-https://github.com/YOUR_USERNAME/L4D2-AI-Architect.git}"
REPO_BRANCH="${L4D2_REPO_BRANCH:-main}"

# Directories
INSTALL_DIR="/root/L4D2-AI-Architect"
VENV_DIR="$INSTALL_DIR/venv"
LOG_FILE="/root/vultr_quickstart.log"

# Training settings
EPOCHS="${L4D2_EPOCHS:-3}"
BATCH_SIZE="${L4D2_BATCH_SIZE:-8}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# =============================================================================
# Helper Functions
# =============================================================================

log_header() {
    echo ""
    echo -e "${BLUE}${BOLD}=============================================${NC}"
    echo -e "${BLUE}${BOLD}  $1${NC}"
    echo -e "${BLUE}${BOLD}=============================================${NC}"
    echo ""
}

log_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_fatal() {
    echo -e "${RED}${BOLD}[FATAL]${NC} $1"
    echo "Check log file: $LOG_FILE"
    exit 1
}

spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    while [ "$(ps a | awk '{print $1}' | grep -w $pid)" ]; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "      \b\b\b\b\b\b"
}

run_with_spinner() {
    local msg="$1"
    shift
    echo -n -e "${CYAN}[...]${NC} $msg"
    "$@" >> "$LOG_FILE" 2>&1 &
    local pid=$!
    spinner $pid
    wait $pid
    local status=$?
    if [ $status -eq 0 ]; then
        echo -e "\r${GREEN}[OK]${NC} $msg"
    else
        echo -e "\r${RED}[FAIL]${NC} $msg"
        return $status
    fi
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        return 1
    fi
    return 0
}

# =============================================================================
# System Checks
# =============================================================================

check_system() {
    log_header "System Check"

    # Check if running as root (recommended on Vultr)
    if [ "$EUID" -ne 0 ]; then
        log_warn "Not running as root. Some operations may fail."
        log_info "Recommended: Run as root on Vultr instances"
    fi

    # Check for NVIDIA GPU
    if ! check_command nvidia-smi; then
        log_fatal "nvidia-smi not found. Is this a GPU instance?"
    fi

    # Detect GPU type
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)

    log_success "GPU detected: $GPU_NAME"
    log_info "GPU Memory: ${GPU_MEM}MB"
    log_info "Driver Version: $DRIVER_VER"

    # Determine optimal config based on GPU
    if echo "$GPU_NAME" | grep -qi "A100"; then
        CONFIG_FILE="configs/unsloth_config_v15.yaml"
        RECOMMENDED_BATCH=8
        log_success "A100 detected - using optimized V15 config"
    elif echo "$GPU_NAME" | grep -qi "GH200\|H100"; then
        CONFIG_FILE="configs/unsloth_config_gh200.yaml"
        RECOMMENDED_BATCH=16
        log_success "GH200/H100 detected - using optimized config"
    elif echo "$GPU_NAME" | grep -qi "A40\|L40"; then
        CONFIG_FILE="configs/unsloth_config.yaml"
        RECOMMENDED_BATCH=4
        log_success "A40/L40 detected - using default config"
    else
        CONFIG_FILE="configs/unsloth_config.yaml"
        RECOMMENDED_BATCH=4
        log_warn "Unknown GPU - using default config"
    fi

    # Check disk space
    DISK_FREE=$(df -BG / | tail -1 | awk '{print $4}' | tr -d 'G')
    if [ "$DISK_FREE" -lt 50 ]; then
        log_warn "Low disk space: ${DISK_FREE}GB free (50GB+ recommended)"
    else
        log_success "Disk space: ${DISK_FREE}GB free"
    fi

    # Check memory
    MEM_TOTAL=$(free -g | awk '/^Mem:/{print $2}')
    log_info "System RAM: ${MEM_TOTAL}GB"

    # Check Python
    if check_command python3; then
        PYTHON_VER=$(python3 --version 2>&1 | cut -d' ' -f2)
        log_success "Python: $PYTHON_VER"
    else
        log_warn "Python3 not found - will install"
    fi

    echo ""
    log_info "Configuration: $CONFIG_FILE"
    log_info "Recommended batch size: $RECOMMENDED_BATCH"
}

# =============================================================================
# System Setup
# =============================================================================

install_system_packages() {
    log_header "Installing System Packages"

    log_step "Updating package lists..."
    apt-get update -qq >> "$LOG_FILE" 2>&1 || log_warn "apt-get update had warnings"

    PACKAGES="git tmux htop nvtop python3 python3-pip python3-venv curl wget"

    for pkg in $PACKAGES; do
        if dpkg -l | grep -q "^ii  $pkg "; then
            log_success "$pkg already installed"
        else
            run_with_spinner "Installing $pkg" apt-get install -y -qq "$pkg"
        fi
    done

    log_success "System packages ready"
}

# =============================================================================
# Repository Setup
# =============================================================================

setup_repository() {
    log_header "Setting Up Repository"

    if [ -d "$INSTALL_DIR" ]; then
        log_info "Repository directory exists, checking..."
        cd "$INSTALL_DIR"

        if [ -d ".git" ]; then
            log_step "Pulling latest changes..."
            git fetch origin >> "$LOG_FILE" 2>&1 || true
            git pull origin "$REPO_BRANCH" >> "$LOG_FILE" 2>&1 || {
                log_warn "Git pull failed - continuing with existing code"
            }
            log_success "Repository updated"
        else
            log_warn "Directory exists but is not a git repo"
            log_info "Using existing files in $INSTALL_DIR"
        fi
    else
        log_step "Cloning repository..."

        if [ "$REPO_URL" = "https://github.com/YOUR_USERNAME/L4D2-AI-Architect.git" ]; then
            log_error "Repository URL not configured!"
            echo ""
            echo "Please set the repository URL:"
            echo "  export L4D2_REPO_URL='https://github.com/YOUR_USER/L4D2-AI-Architect.git'"
            echo "  ./vultr_quickstart.sh"
            echo ""
            echo "Or edit this script and update REPO_URL at the top."
            exit 1
        fi

        git clone --depth 1 --branch "$REPO_BRANCH" "$REPO_URL" "$INSTALL_DIR" >> "$LOG_FILE" 2>&1 || {
            log_fatal "Failed to clone repository from $REPO_URL"
        }
        log_success "Repository cloned"
    fi

    cd "$INSTALL_DIR"

    # Create required directories
    mkdir -p model_adapters exports data/training_logs data/processed

    log_success "Repository ready at $INSTALL_DIR"
}

# =============================================================================
# Python Environment Setup
# =============================================================================

setup_python_environment() {
    log_header "Setting Up Python Environment"

    cd "$INSTALL_DIR"

    # Create virtual environment if needed
    if [ ! -d "$VENV_DIR" ]; then
        log_step "Creating virtual environment..."
        python3 -m venv "$VENV_DIR" >> "$LOG_FILE" 2>&1
        log_success "Virtual environment created"
    else
        log_success "Virtual environment exists"
    fi

    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    log_success "Virtual environment activated"

    # Upgrade pip
    run_with_spinner "Upgrading pip" pip install --upgrade pip

    # Install PyTorch with CUDA
    log_step "Installing PyTorch with CUDA (this may take a few minutes)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 >> "$LOG_FILE" 2>&1 || {
        log_warn "CUDA 12.1 install failed, trying CUDA 11.8..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 >> "$LOG_FILE" 2>&1 || {
            log_fatal "PyTorch installation failed"
        }
    }

    # Verify CUDA
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" >> "$LOG_FILE" 2>&1 || {
        log_fatal "PyTorch CUDA not working"
    }
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
    CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda)")
    log_success "PyTorch $TORCH_VER with CUDA $CUDA_VER"

    # Install core dependencies
    run_with_spinner "Installing transformers & accelerate" \
        pip install transformers datasets accelerate peft trl sentencepiece protobuf

    # Install bitsandbytes
    run_with_spinner "Installing bitsandbytes" pip install bitsandbytes

    # Verify bitsandbytes
    python3 -c "import bitsandbytes" >> "$LOG_FILE" 2>&1 || {
        log_warn "bitsandbytes import failed, reinstalling..."
        pip install --force-reinstall bitsandbytes >> "$LOG_FILE" 2>&1
    }
    log_success "bitsandbytes ready"

    # Install Unsloth
    log_step "Installing Unsloth (optimized fine-tuning)..."
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" >> "$LOG_FILE" 2>&1 || {
        log_warn "Unsloth install failed - trying alternative method"
        pip install unsloth >> "$LOG_FILE" 2>&1 || {
            log_fatal "Unsloth installation failed"
        }
    }
    log_success "Unsloth installed"

    # Install Flash Attention 2 (optional but recommended for A100)
    log_step "Installing Flash Attention 2 (optional, may take a few minutes)..."
    pip install flash-attn --no-build-isolation >> "$LOG_FILE" 2>&1 || {
        log_warn "Flash Attention 2 not installed (training will still work)"
    }

    # Install remaining dependencies
    run_with_spinner "Installing additional dependencies" \
        pip install tensorboard pyyaml tqdm safetensors

    log_success "Python environment ready"
}

# =============================================================================
# Training Data Check
# =============================================================================

check_training_data() {
    log_header "Checking Training Data"

    cd "$INSTALL_DIR"

    # Look for training data in order of preference
    TRAINING_DATA=""
    for version in v15 v14 v13 v12 v11 v10 v9 v8; do
        CANDIDATE="data/processed/l4d2_train_${version}.jsonl"
        if [ -f "$CANDIDATE" ]; then
            TRAINING_DATA="$CANDIDATE"
            break
        fi
    done

    # Fallback to combined_train
    if [ -z "$TRAINING_DATA" ] && [ -f "data/processed/combined_train.jsonl" ]; then
        TRAINING_DATA="data/processed/combined_train.jsonl"
    fi

    if [ -z "$TRAINING_DATA" ]; then
        log_error "No training data found!"
        echo ""
        echo "Expected location: data/processed/l4d2_train_v15.jsonl"
        echo ""
        echo "Options:"
        echo "  1. Upload training data:"
        echo "     scp data/processed/l4d2_train_v15.jsonl root@\$(hostname -I | awk '{print \$1}'):$INSTALL_DIR/data/processed/"
        echo ""
        echo "  2. Or create synthetic data (requires OpenAI API key):"
        echo "     export OPENAI_API_KEY='your-key'"
        echo "     python scripts/training/generate_synthetic_data.py"
        echo ""
        exit 1
    fi

    SAMPLE_COUNT=$(wc -l < "$TRAINING_DATA")
    log_success "Training data found: $TRAINING_DATA"
    log_info "Samples: $SAMPLE_COUNT"

    # Check validation data
    VAL_DATA=""
    if [ -f "data/processed/combined_val.jsonl" ]; then
        VAL_DATA="data/processed/combined_val.jsonl"
        VAL_COUNT=$(wc -l < "$VAL_DATA")
        log_success "Validation data: $VAL_COUNT samples"
    else
        log_warn "No validation data found (optional)"
    fi

    # Show sample
    echo ""
    log_info "Sample from training data:"
    head -1 "$TRAINING_DATA" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"  System: {d['messages'][0]['content'][:80]}...\")" 2>/dev/null || true

    # Export for training script
    export TRAINING_DATA
    export VAL_DATA
}

# =============================================================================
# Start Training
# =============================================================================

start_training() {
    log_header "Starting Training"

    cd "$INSTALL_DIR"
    source "$VENV_DIR/bin/activate"

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="model_adapters/l4d2-mistral-$TIMESTAMP"

    log_info "Output directory: $OUTPUT_DIR"
    log_info "Config: $CONFIG_FILE"
    log_info "Epochs: $EPOCHS"
    log_info "Training data: $TRAINING_DATA"

    # Create training command
    TRAIN_CMD="python scripts/training/train_unsloth.py --config $CONFIG_FILE --epochs $EPOCHS --output $OUTPUT_DIR"

    # Export command for post-training
    EXPORT_CMD="python scripts/training/export_gguf_cpu.py --adapter $OUTPUT_DIR/final --output exports/l4d2-mistral-$TIMESTAMP --create-modelfile"

    echo ""
    echo -e "${BOLD}Starting training in tmux session 'training'${NC}"
    echo ""
    echo "Monitor commands:"
    echo -e "  ${CYAN}tmux attach -t training${NC}     - Watch training output"
    echo -e "  ${CYAN}watch nvidia-smi${NC}            - Monitor GPU usage"
    echo -e "  ${CYAN}tail -f training.log${NC}        - Follow log file"
    echo ""

    # Start training in tmux
    tmux new-session -d -s training "
        cd $INSTALL_DIR && \
        source $VENV_DIR/bin/activate && \
        echo '==================================================' && \
        echo 'L4D2-AI-Architect Training Started' && \
        echo '==================================================' && \
        echo '' && \
        echo 'GPU:' && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && \
        echo '' && \
        echo 'Training data:' && wc -l $TRAINING_DATA && \
        echo '' && \
        echo 'Starting at:' && date && \
        echo '' && \
        $TRAIN_CMD 2>&1 | tee training.log && \
        echo '' && \
        echo '==================================================' && \
        echo 'Training Complete! Starting GGUF Export...' && \
        echo '==================================================' && \
        $EXPORT_CMD 2>&1 | tee -a training.log && \
        echo '' && \
        echo '==================================================' && \
        echo 'ALL DONE!' && \
        echo '==================================================' && \
        echo '' && \
        echo 'Exported model:' && ls -lh exports/l4d2-mistral-$TIMESTAMP/ && \
        echo '' && \
        echo 'Download with:' && \
        echo '  scp -r root@\$(hostname -I | awk \"{print \\\$1}\"):$INSTALL_DIR/exports/l4d2-mistral-$TIMESTAMP ./exports/' && \
        echo '' && \
        echo 'Finished at:' && date
    "

    log_success "Training started in tmux session 'training'"

    echo ""
    echo -e "${GREEN}${BOLD}=============================================${NC}"
    echo -e "${GREEN}${BOLD}  Training is now running!${NC}"
    echo -e "${GREEN}${BOLD}=============================================${NC}"
    echo ""
    echo "Instance IP: $(hostname -I | awk '{print $1}')"
    echo "Estimated time: ~2 hours on A100"
    echo ""
    echo "Quick commands:"
    echo "  tmux attach -t training    # Watch progress"
    echo "  ./vultr_quickstart.sh --status  # Check status"
    echo ""
}

# =============================================================================
# Status Check
# =============================================================================

check_status() {
    log_header "Training Status"

    cd "$INSTALL_DIR" 2>/dev/null || {
        log_error "Project directory not found: $INSTALL_DIR"
        exit 1
    }

    # Check if tmux session exists
    if tmux has-session -t training 2>/dev/null; then
        log_success "Training session is RUNNING"
        echo ""

        # Show GPU status
        echo "GPU Status:"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
        echo ""

        # Show recent log
        if [ -f "training.log" ]; then
            echo "Recent training output:"
            echo "----------------------------------------"
            tail -20 training.log
            echo "----------------------------------------"
        fi

        echo ""
        echo "Attach to session: tmux attach -t training"
    else
        log_warn "Training session not found"
        echo ""

        # Check for completed training
        if ls model_adapters/l4d2-mistral-*/final 2>/dev/null | head -1 > /dev/null; then
            log_success "Trained model found!"
            echo ""
            echo "Available models:"
            ls -d model_adapters/l4d2-mistral-*/final 2>/dev/null
            echo ""
        fi

        # Check for exported models
        if ls exports/l4d2-mistral-*/*.gguf 2>/dev/null | head -1 > /dev/null; then
            log_success "Exported GGUF models:"
            ls -lh exports/l4d2-mistral-*/*.gguf 2>/dev/null
            echo ""
        fi

        # Show log tail if exists
        if [ -f "training.log" ]; then
            echo "Last training log entries:"
            echo "----------------------------------------"
            tail -10 training.log
            echo "----------------------------------------"
        fi
    fi
}

# =============================================================================
# Export Only
# =============================================================================

export_model() {
    log_header "Export Model to GGUF"

    cd "$INSTALL_DIR"
    source "$VENV_DIR/bin/activate"

    # Find latest trained model
    LATEST_MODEL=$(ls -td model_adapters/l4d2-mistral-*/final 2>/dev/null | head -1)

    if [ -z "$LATEST_MODEL" ]; then
        log_error "No trained model found in model_adapters/"
        exit 1
    fi

    log_info "Found model: $LATEST_MODEL"

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="exports/l4d2-mistral-$TIMESTAMP"

    log_step "Exporting to GGUF format..."

    python scripts/training/export_gguf_cpu.py \
        --adapter "$LATEST_MODEL" \
        --output "$OUTPUT_DIR" \
        --quantize q4_k_m \
        --create-modelfile

    log_success "Export complete!"
    echo ""
    echo "Exported files:"
    ls -lh "$OUTPUT_DIR/"
    echo ""
    echo "Download with:"
    echo "  scp -r root@$(hostname -I | awk '{print $1}'):$INSTALL_DIR/$OUTPUT_DIR ./exports/"
}

# =============================================================================
# Help
# =============================================================================

show_help() {
    echo ""
    echo -e "${BOLD}L4D2-AI-Architect Vultr Quick Start${NC}"
    echo ""
    echo "Usage: $0 [option]"
    echo ""
    echo "Options:"
    echo "  (none)         Full setup and start training"
    echo "  --train-only   Skip setup, just start training"
    echo "  --export-only  Export existing trained model to GGUF"
    echo "  --status       Check training status"
    echo "  --help         Show this help"
    echo ""
    echo "Environment Variables:"
    echo "  L4D2_REPO_URL     Repository URL (required for fresh setup)"
    echo "  L4D2_REPO_BRANCH  Branch to checkout (default: main)"
    echo "  L4D2_EPOCHS       Training epochs (default: 3)"
    echo "  L4D2_BATCH_SIZE   Batch size (default: 8)"
    echo ""
    echo "Examples:"
    echo "  # Full setup on fresh Vultr instance"
    echo "  export L4D2_REPO_URL='https://github.com/user/L4D2-AI-Architect.git'"
    echo "  ./vultr_quickstart.sh"
    echo ""
    echo "  # Check training progress"
    echo "  ./vultr_quickstart.sh --status"
    echo ""
    echo "  # Export after training completes"
    echo "  ./vultr_quickstart.sh --export-only"
    echo ""
}

# =============================================================================
# Main
# =============================================================================

main() {
    # Initialize log file
    echo "L4D2 Vultr Quick Start - $(date)" > "$LOG_FILE"

    case "${1:-}" in
        --help|-h)
            show_help
            ;;
        --status)
            check_status
            ;;
        --export-only)
            export_model
            ;;
        --train-only)
            log_header "L4D2-AI-Architect Training (Skip Setup)"
            check_system
            check_training_data
            start_training
            ;;
        *)
            echo ""
            echo -e "${BOLD}=============================================${NC}"
            echo -e "${BOLD}  L4D2-AI-Architect Vultr Quick Start${NC}"
            echo -e "${BOLD}=============================================${NC}"
            echo ""
            echo "This script will:"
            echo "  1. Check GPU and system requirements"
            echo "  2. Clone/update the repository"
            echo "  3. Set up Python environment with all dependencies"
            echo "  4. Verify training data"
            echo "  5. Start training in a tmux session"
            echo "  6. Auto-export to GGUF after training"
            echo ""
            echo "Estimated time: ~10 min setup + ~2 hours training"
            echo "Estimated cost: ~\$6-8 on A100"
            echo ""

            check_system
            install_system_packages
            setup_repository
            setup_python_environment
            check_training_data
            start_training

            echo ""
            echo -e "${GREEN}${BOLD}Setup complete! Training is running.${NC}"
            echo ""
            echo "Next steps:"
            echo "  1. Monitor training: tmux attach -t training"
            echo "  2. Wait ~2 hours for completion"
            echo "  3. Download model: scp -r root@IP:$INSTALL_DIR/exports/ ./exports/"
            echo "  4. Destroy instance to stop billing"
            echo ""
            ;;
    esac
}

main "$@"
