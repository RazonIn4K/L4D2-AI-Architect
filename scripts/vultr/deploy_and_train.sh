#!/bin/bash
# =============================================================================
# L4D2 Vultr GPU Deployment & Training - Complete One-Command Setup
# =============================================================================
# Deploys an A100 instance on Vultr, sets up training, and runs Mistral-7B
# fine-tuning on the V9 dataset with automatic GGUF export.
#
# Usage:
#   ./scripts/vultr/deploy_and_train.sh              # Full deployment + training
#   ./scripts/vultr/deploy_and_train.sh --ssh-only   # Just SSH to existing instance
#   ./scripts/vultr/deploy_and_train.sh --status     # Check training status
#   ./scripts/vultr/deploy_and_train.sh --download   # Download trained model
#
# Prerequisites:
#   - VULTR_API_KEY environment variable set
#   - SSH key uploaded to Vultr (or will be created)
#   - brew install vultr-cli (or cargo install vultr-cli)
#
# Cost Estimate:
#   - A100 (40GB): ~$2.50/hr x 2.5 hours = ~$6.25 total
#   - Training: ~671 examples, 3 epochs, ~2-2.5 hours
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
INSTANCE_LABEL="${VULTR_INSTANCE_LABEL:-l4d2-training-$(date +%Y%m%d)}"
PLAN="${VULTR_PLAN:-vcg-a100-6c-60g-40vram}"  # A100 40GB VRAM - ~$1.20/hr, ideal for training
REGION="${VULTR_REGION:-ewr}"             # New Jersey - good connectivity
OS_ID="${VULTR_OS_ID:-2284}"             # Ubuntu 24.04 LTS
TRAINING_DATA="${VULTR_TRAINING_DATA:-data/processed/l4d2_train_v9.jsonl}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    if [ -z "$VULTR_API_KEY" ]; then
        log_error "VULTR_API_KEY not set. Export it or use Doppler:"
        echo "  export VULTR_API_KEY='your-api-key'"
        echo "  doppler run -- ./scripts/vultr/deploy_and_train.sh"
        exit 1
    fi

    if ! command -v vultr-cli &> /dev/null; then
        log_warn "vultr-cli not found. Installing..."
        if command -v brew &> /dev/null; then
            brew install vultr-cli
        elif command -v cargo &> /dev/null; then
            cargo install vultr-cli
        else
            log_error "Install vultr-cli: brew install vultr-cli"
            exit 1
        fi
    fi

    if [ ! -f "$TRAINING_DATA" ]; then
        log_error "Training data not found: $TRAINING_DATA"
        log_info "Run: python scripts/training/prepare_v9_dataset.py first"
        exit 1
    fi

    log_success "Prerequisites OK"
}

# Get or create SSH key
setup_ssh_key() {
    log_info "Setting up SSH key..."

    SSH_KEY_PATH="$HOME/.ssh/vultr_l4d2"

    if [ ! -f "$SSH_KEY_PATH" ]; then
        log_info "Creating new SSH key for Vultr..."
        ssh-keygen -t ed25519 -f "$SSH_KEY_PATH" -N "" -C "l4d2-vultr-training"
    fi

    # Check if key exists on Vultr
    EXISTING_KEY=$(vultr-cli ssh-key list 2>/dev/null | grep "l4d2-training" | awk '{print $1}' || true)

    if [ -z "$EXISTING_KEY" ]; then
        log_info "Uploading SSH key to Vultr..."
        SSH_KEY_ID=$(vultr-cli ssh-key create \
            --name "l4d2-training" \
            --key "$(cat ${SSH_KEY_PATH}.pub)" \
            2>/dev/null | grep "ID:" | awk '{print $2}' || true)

        if [ -z "$SSH_KEY_ID" ]; then
            # Try to get it if already exists
            SSH_KEY_ID=$(vultr-cli ssh-key list | grep "l4d2-training" | awk '{print $1}')
        fi
    else
        SSH_KEY_ID="$EXISTING_KEY"
        log_info "Using existing SSH key: $SSH_KEY_ID"
    fi

    echo "$SSH_KEY_ID"
}

# Check for existing instance
find_existing_instance() {
    vultr-cli instance list 2>/dev/null | grep "$INSTANCE_LABEL" | awk '{print $1}' || true
}

# Create Vultr instance
create_instance() {
    log_info "Creating Vultr A100 instance..."

    SSH_KEY_ID=$(setup_ssh_key)

    # Create cloud-init script
    CLOUD_INIT=$(cat <<'CLOUDINIT'
#!/bin/bash
apt-get update
apt-get install -y git tmux htop nvtop python3-pip python3-venv
echo "Cloud-init complete" > /root/cloud-init-done
CLOUDINIT
)

    INSTANCE_ID=$(vultr-cli instance create \
        --label "$INSTANCE_LABEL" \
        --plan "$PLAN" \
        --region "$REGION" \
        --os "$OS_ID" \
        --ssh-keys "$SSH_KEY_ID" \
        --userdata "$CLOUD_INIT" \
        2>/dev/null | grep "ID:" | awk '{print $2}')

    if [ -z "$INSTANCE_ID" ]; then
        log_error "Failed to create instance"
        exit 1
    fi

    log_success "Instance created: $INSTANCE_ID"

    # Wait for instance to be ready
    log_info "Waiting for instance to be ready (this takes 2-3 minutes)..."
    for i in {1..60}; do
        STATUS=$(vultr-cli instance get "$INSTANCE_ID" 2>/dev/null | grep "Status" | awk '{print $2}')
        if [ "$STATUS" = "active" ]; then
            break
        fi
        echo -n "."
        sleep 5
    done
    echo ""

    # Get IP address
    INSTANCE_IP=$(vultr-cli instance get "$INSTANCE_ID" 2>/dev/null | grep "Main IP" | awk '{print $3}')

    log_success "Instance ready at $INSTANCE_IP"

    # Wait for SSH
    log_info "Waiting for SSH to be available..."
    for i in {1..30}; do
        if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i "$HOME/.ssh/vultr_l4d2" root@"$INSTANCE_IP" "echo ok" 2>/dev/null; then
            break
        fi
        echo -n "."
        sleep 5
    done
    echo ""

    echo "$INSTANCE_IP"
}

# Upload code and data to instance
upload_to_instance() {
    INSTANCE_IP=$1
    SSH_KEY="$HOME/.ssh/vultr_l4d2"

    log_info "Uploading training code and data to instance..."

    # Create tarball of essential files
    TARBALL="/tmp/l4d2-training-$(date +%s).tar.gz"

    tar -czf "$TARBALL" \
        --exclude='venv' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.git' \
        --exclude='model_adapters' \
        --exclude='data/raw' \
        --exclude='data/training_logs' \
        --exclude='exports' \
        scripts/training/train_unsloth.py \
        scripts/training/export_model.py \
        scripts/utils/ \
        configs/ \
        data/processed/l4d2_train_v9.jsonl \
        requirements.txt \
        run_vultr_training.sh

    log_info "Uploading $(du -h $TARBALL | cut -f1) of training data..."

    scp -o StrictHostKeyChecking=no -i "$SSH_KEY" \
        "$TARBALL" root@"$INSTANCE_IP":/root/training.tar.gz

    # Extract on remote
    ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" root@"$INSTANCE_IP" << 'REMOTE'
mkdir -p /root/L4D2-AI-Architect
cd /root/L4D2-AI-Architect
tar -xzf /root/training.tar.gz
rm /root/training.tar.gz
echo "Files extracted successfully"
ls -la
REMOTE

    rm "$TARBALL"
    log_success "Upload complete"
}

# Start training on instance
start_training() {
    INSTANCE_IP=$1
    SSH_KEY="$HOME/.ssh/vultr_l4d2"

    log_info "Starting training on Vultr instance..."

    ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" root@"$INSTANCE_IP" << 'REMOTE'
cd /root/L4D2-AI-Architect

# Create directories
mkdir -p model_adapters exports data/training_logs

# Setup Python environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -q --upgrade pip
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers datasets accelerate peft bitsandbytes
pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install -q flash-attn --no-build-isolation 2>/dev/null || echo "Flash attention skipped"

# Verify GPU
nvidia-smi --query-gpu=name,memory.total --format=csv

# Start training in tmux
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
tmux new-session -d -s training "
    source venv/bin/activate && \
    echo '=== Starting L4D2 Mistral-7B Training ===' && \
    echo 'Dataset: data/processed/l4d2_train_v9.jsonl' && \
    wc -l data/processed/l4d2_train_v9.jsonl && \
    python scripts/training/train_unsloth.py \
        --config configs/unsloth_config_a100.yaml \
        --epochs 3 \
        --output model_adapters/mistral-l4d2-v9-$TIMESTAMP 2>&1 | tee training.log && \
    echo '' && \
    echo '=== Training Complete! Exporting GGUF ===' && \
    python scripts/training/export_model.py \
        --input model_adapters/mistral-l4d2-v9-$TIMESTAMP \
        --format gguf \
        --quantize q4_k_m \
        --output exports/l4d2-mistral-v9.gguf && \
    echo '' && \
    echo '=== DONE! Model ready for download ===' && \
    ls -lh exports/
"

echo ""
echo "Training started in tmux session 'training'"
echo "Monitor with: tmux attach -t training"
REMOTE

    log_success "Training started!"
    echo ""
    echo "=========================================="
    echo "Training is running on Vultr A100"
    echo "=========================================="
    echo ""
    echo "Instance IP: $INSTANCE_IP"
    echo "SSH Command: ssh -i ~/.ssh/vultr_l4d2 root@$INSTANCE_IP"
    echo "Monitor:     ssh -i ~/.ssh/vultr_l4d2 root@$INSTANCE_IP 'tmux attach -t training'"
    echo ""
    echo "Estimated time: 2-2.5 hours"
    echo "Estimated cost: ~\$6-7"
    echo ""
    echo "When complete, download model with:"
    echo "  ./scripts/vultr/deploy_and_train.sh --download"
}

# Check training status
check_status() {
    INSTANCE_IP=$(get_instance_ip)
    if [ -z "$INSTANCE_IP" ]; then
        log_error "No active training instance found"
        exit 1
    fi

    SSH_KEY="$HOME/.ssh/vultr_l4d2"

    log_info "Checking training status on $INSTANCE_IP..."

    ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" root@"$INSTANCE_IP" << 'REMOTE'
cd /root/L4D2-AI-Architect

if tmux has-session -t training 2>/dev/null; then
    echo "Training is RUNNING"
    echo ""
    echo "Last 20 lines of training.log:"
    tail -20 training.log 2>/dev/null || echo "(no log yet)"
else
    echo "Training session not found (may be complete)"
    echo ""
    if [ -f "exports/l4d2-mistral-v9.gguf" ]; then
        echo "Model is READY for download:"
        ls -lh exports/*.gguf
    else
        echo "No exported model found yet"
    fi
fi
REMOTE
}

# Download trained model
download_model() {
    INSTANCE_IP=$(get_instance_ip)
    if [ -z "$INSTANCE_IP" ]; then
        log_error "No active training instance found"
        exit 1
    fi

    SSH_KEY="$HOME/.ssh/vultr_l4d2"

    log_info "Downloading trained model from $INSTANCE_IP..."

    mkdir -p "$PROJECT_ROOT/exports"

    scp -o StrictHostKeyChecking=no -i "$SSH_KEY" \
        root@"$INSTANCE_IP":/root/L4D2-AI-Architect/exports/*.gguf \
        "$PROJECT_ROOT/exports/"

    log_success "Model downloaded to exports/"
    ls -lh "$PROJECT_ROOT/exports/"*.gguf

    echo ""
    echo "To use with Ollama:"
    echo "  1. Create Modelfile with: FROM ./exports/l4d2-mistral-v9.gguf"
    echo "  2. ollama create l4d2-mistral -f Modelfile"
    echo "  3. ollama run l4d2-mistral"
}

# Get instance IP
get_instance_ip() {
    INSTANCE_ID=$(find_existing_instance)
    if [ -n "$INSTANCE_ID" ]; then
        vultr-cli instance get "$INSTANCE_ID" 2>/dev/null | grep "Main IP" | awk '{print $3}'
    fi
}

# SSH to instance
ssh_to_instance() {
    INSTANCE_IP=$(get_instance_ip)
    if [ -z "$INSTANCE_IP" ]; then
        log_error "No active training instance found"
        exit 1
    fi

    SSH_KEY="$HOME/.ssh/vultr_l4d2"
    log_info "Connecting to $INSTANCE_IP..."
    ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" root@"$INSTANCE_IP"
}

# Destroy instance
destroy_instance() {
    INSTANCE_ID=$(find_existing_instance)
    if [ -n "$INSTANCE_ID" ]; then
        log_warn "Destroying instance $INSTANCE_ID..."
        vultr-cli instance delete "$INSTANCE_ID"
        log_success "Instance destroyed"
    else
        log_info "No instance to destroy"
    fi
}

# Main
main() {
    case "${1:-}" in
        --ssh-only|--ssh)
            ssh_to_instance
            ;;
        --status)
            check_status
            ;;
        --download)
            download_model
            ;;
        --destroy)
            destroy_instance
            ;;
        --help|-h)
            echo "Usage: $0 [option]"
            echo ""
            echo "Options:"
            echo "  (none)      Full deployment + training"
            echo "  --ssh       SSH to existing instance"
            echo "  --status    Check training status"
            echo "  --download  Download trained model"
            echo "  --destroy   Destroy instance (save costs)"
            echo "  --help      Show this help"
            ;;
        *)
            echo "=============================================="
            echo "L4D2 Vultr GPU Training Deployment"
            echo "=============================================="
            echo ""

            check_prerequisites

            # Check for existing instance
            EXISTING=$(find_existing_instance)
            if [ -n "$EXISTING" ]; then
                INSTANCE_IP=$(get_instance_ip)
                log_warn "Existing instance found at $INSTANCE_IP"
                echo ""
                read -p "Use existing instance? [Y/n] " -n 1 -r
                echo ""
                if [[ $REPLY =~ ^[Nn]$ ]]; then
                    destroy_instance
                    INSTANCE_IP=$(create_instance)
                fi
            else
                INSTANCE_IP=$(create_instance)
            fi

            upload_to_instance "$INSTANCE_IP"
            start_training "$INSTANCE_IP"

            # Save instance info
            echo "$INSTANCE_IP" > "$PROJECT_ROOT/.vultr_instance_ip"
            ;;
    esac
}

main "$@"
