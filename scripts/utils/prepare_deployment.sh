#!/bin/bash
# =============================================================================
# L4D2-AI-Architect Vultr Deployment Script
# =============================================================================
# Packages the training code and data, uploads to Vultr, and runs training.
#
# Usage:
#   ./scripts/utils/prepare_deployment.sh --host <VULTR_IP> [OPTIONS]
#
# Options:
#   --host HOST      Vultr instance IP/hostname (required for upload)
#   --user USER      SSH user (default: root)
#   --config CONFIG  Training config to use (default: unsloth_config_a100.yaml)
#   --upload-only    Only upload, don't start training
#   --download-only  Only download results
#   --status         Check training status
#   --resume PATH    Resume from checkpoint
#   --package-only   Only create package, don't upload
#   --help           Show this help message
#
# Examples:
#   ./scripts/utils/prepare_deployment.sh --host 45.32.xxx.xxx
#   ./scripts/utils/prepare_deployment.sh --host 45.32.xxx.xxx --config unsloth_config_qwen.yaml
#   ./scripts/utils/prepare_deployment.sh --host 45.32.xxx.xxx --download-only
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
SSH_USER="root"
CONFIG_FILE="unsloth_config_a100.yaml"
UPLOAD_ONLY=false
DOWNLOAD_ONLY=false
PACKAGE_ONLY=false
CHECK_STATUS=false
RESUME_PATH=""
PROJECT_NAME="L4D2-AI-Architect"
REMOTE_DIR="/root/${PROJECT_NAME}"

# Validate hostname/IP format (prevent command injection)
validate_host() {
    local host="$1"
    # Allow only valid hostnames or IPs (alphanumeric, dots, hyphens)
    if [[ ! "$host" =~ ^[a-zA-Z0-9][a-zA-Z0-9.-]*[a-zA-Z0-9]$|^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        print_error "Invalid host format: $host"
        print_error "Host must be a valid hostname or IP address"
        exit 1
    fi
    # Block localhost and private IPs for safety
    if [[ "$host" == "localhost" || "$host" =~ ^127\. || "$host" =~ ^10\. || "$host" =~ ^192\.168\. || "$host" =~ ^172\.(1[6-9]|2[0-9]|3[01])\. ]]; then
        print_error "Local/private hosts are not allowed: $host"
        exit 1
    fi
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            VULTR_HOST="$2"
            validate_host "$VULTR_HOST"
            shift 2
            ;;
        --user)
            SSH_USER="$2"
            # Validate SSH user (alphanumeric, underscore, hyphen only)
            if [[ ! "$SSH_USER" =~ ^[a-zA-Z_][a-zA-Z0-9_-]*$ ]]; then
                print_error "Invalid SSH user format: $SSH_USER"
                exit 1
            fi
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            # Validate config filename (prevent path traversal)
            if [[ "$CONFIG_FILE" == *".."* || "$CONFIG_FILE" == *"/"* ]]; then
                print_error "Invalid config filename (no path traversal allowed): $CONFIG_FILE"
                exit 1
            fi
            shift 2
            ;;
        --upload-only)
            UPLOAD_ONLY=true
            shift
            ;;
        --download-only)
            DOWNLOAD_ONLY=true
            shift
            ;;
        --package-only)
            PACKAGE_ONLY=true
            shift
            ;;
        --status)
            CHECK_STATUS=true
            shift
            ;;
        --resume)
            RESUME_PATH="$2"
            # Validate resume path (prevent command injection, no shell metacharacters)
            if [[ "$RESUME_PATH" == *".."* || "$RESUME_PATH" =~ [\;\|\&\$\`\(\)\{\}] ]]; then
                print_error "Invalid resume path (no path traversal or shell metacharacters): $RESUME_PATH"
                exit 1
            fi
            shift 2
            ;;
        --help)
            head -30 "$0" | tail -25
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Function to print status
print_status() {
    echo -e "${BLUE}[*]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Create deployment package
create_package() {
    print_status "Creating deployment package..."

    cd "$PROJECT_ROOT"
    PACKAGE_DIR=$(mktemp -d)
    PACKAGE_NAME="l4d2_training_$(date +%Y%m%d_%H%M%S).tar.gz"

    # Files to include
    mkdir -p "$PACKAGE_DIR/$PROJECT_NAME/data"

    # Copy essential directories
    cp -r configs "$PACKAGE_DIR/$PROJECT_NAME/"
    cp -r scripts "$PACKAGE_DIR/$PROJECT_NAME/"

    # Copy training data
    if [ -d "data/processed" ]; then
        cp -r data/processed "$PACKAGE_DIR/$PROJECT_NAME/data/"
    fi
    if [ -d "data/anti_patterns" ]; then
        cp -r data/anti_patterns "$PACKAGE_DIR/$PROJECT_NAME/data/"
    fi

    # Copy essential files
    cp requirements.txt "$PACKAGE_DIR/$PROJECT_NAME/" 2>/dev/null || true
    cp CLAUDE.md "$PACKAGE_DIR/$PROJECT_NAME/" 2>/dev/null || true

    # Create the tarball
    cd "$PACKAGE_DIR"
    tar -czvf "$PROJECT_ROOT/$PACKAGE_NAME" "$PROJECT_NAME" 2>/dev/null

    rm -rf "$PACKAGE_DIR"

    PACKAGE_SIZE=$(du -h "$PROJECT_ROOT/$PACKAGE_NAME" | cut -f1)
    print_success "Created package: $PACKAGE_NAME ($PACKAGE_SIZE)"
    echo "$PROJECT_ROOT/$PACKAGE_NAME"
}

# Upload to Vultr
upload_to_vultr() {
    local package_path="$1"

    print_status "Uploading to Vultr ($VULTR_HOST)..."

    # Test SSH connection
    if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "${SSH_USER}@${VULTR_HOST}" "echo connected" &>/dev/null; then
        print_error "Cannot connect to ${SSH_USER}@${VULTR_HOST}"
        print_warning "Make sure your SSH key is configured"
        exit 1
    fi

    # Upload package
    scp "$package_path" "${SSH_USER}@${VULTR_HOST}:/root/"

    # Extract on remote
    local package_name=$(basename "$package_path")
    ssh "${SSH_USER}@${VULTR_HOST}" << EOF
        cd /root
        rm -rf ${PROJECT_NAME}
        tar -xzf ${package_name}
        rm ${package_name}

        # Create virtual environment if needed
        if [ ! -d "${REMOTE_DIR}/venv" ]; then
            echo "Creating virtual environment..."
            python3 -m venv ${REMOTE_DIR}/venv
        fi

        # Install dependencies
        source ${REMOTE_DIR}/venv/bin/activate
        pip install --upgrade pip -q

        # Install PyTorch with CUDA
        pip install torch --index-url https://download.pytorch.org/whl/cu121 -q

        # Install other requirements
        pip install -r ${REMOTE_DIR}/requirements.txt -q

        # Install unsloth
        pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q

        echo "Setup complete!"
EOF

    print_success "Uploaded and extracted on Vultr"
}

# Start training
start_training() {
    print_status "Starting training on Vultr..."

    local resume_flag=""
    if [[ -n "$RESUME_PATH" ]]; then
        resume_flag="--resume $RESUME_PATH"
    fi

    # Create training command
    local train_cmd="cd ${REMOTE_DIR} && source venv/bin/activate && python scripts/training/train_unsloth.py --config configs/${CONFIG_FILE} ${resume_flag} 2>&1 | tee training.log"

    # Start in tmux for persistence
    ssh "${SSH_USER}@${VULTR_HOST}" << EOF
        # Kill any existing training
        tmux kill-session -t training 2>/dev/null || true

        # Start new tmux session with training
        tmux new-session -d -s training "${train_cmd}"

        echo "Training started in tmux session 'training'"
EOF

    print_success "Training started!"
    echo ""
    echo "Monitor with:"
    echo "  ssh ${SSH_USER}@${VULTR_HOST} -t 'tmux attach -t training'"
}

# Download results
download_results() {
    print_status "Downloading results from Vultr..."

    # Create local results directory
    local results_dir="$PROJECT_ROOT/model_adapters/vultr_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$results_dir"

    # Download model adapters
    scp -r "${SSH_USER}@${VULTR_HOST}:${REMOTE_DIR}/model_adapters/*" "$results_dir/" 2>/dev/null || print_warning "No model adapters found"

    # Download training logs
    mkdir -p "$results_dir/logs"
    scp -r "${SSH_USER}@${VULTR_HOST}:${REMOTE_DIR}/data/training_logs/*" "$results_dir/logs/" 2>/dev/null || print_warning "No training logs found"
    scp "${SSH_USER}@${VULTR_HOST}:${REMOTE_DIR}/training.log" "$results_dir/" 2>/dev/null || true

    print_success "Downloaded to: $results_dir"
}

# Check training status
check_training_status() {
    print_status "Checking training status on $VULTR_HOST..."

    ssh "${SSH_USER}@${VULTR_HOST}" << 'EOF'
        echo ""
        if tmux has-session -t training 2>/dev/null; then
            echo "🟢 Training is RUNNING"
            echo ""
            echo "Last 15 lines of output:"
            echo "---"
            tmux capture-pane -t training -p | tail -15
        else
            echo "🔴 Training is NOT running"
        fi

        echo ""
        echo "GPU Status:"
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null || echo "nvidia-smi not available"

        echo ""
        if [ -d "/root/L4D2-AI-Architect/model_adapters" ]; then
            echo "Available models:"
            ls -la /root/L4D2-AI-Architect/model_adapters/ 2>/dev/null
        fi
EOF
}

# Main execution
main() {
    echo "=============================================="
    echo "L4D2-AI-Architect Vultr Deployment"
    echo "=============================================="
    echo ""

    # Check if host is required
    if [[ -z "$VULTR_HOST" && "$PACKAGE_ONLY" != "true" ]]; then
        print_error "Error: --host is required (or use --package-only)"
        echo "Usage: $0 --host <vultr-ip> [OPTIONS]"
        exit 1
    fi

    # Handle status check
    if [[ "$CHECK_STATUS" == "true" ]]; then
        check_training_status
        exit 0
    fi

    # Handle download only
    if [[ "$DOWNLOAD_ONLY" == "true" ]]; then
        download_results
        exit 0
    fi

    # Create package
    PACKAGE_PATH=$(create_package)

    # Package only mode
    if [[ "$PACKAGE_ONLY" == "true" ]]; then
        print_success "Package created: $PACKAGE_PATH"
        echo ""
        echo "To upload manually:"
        echo "  scp $PACKAGE_PATH root@<VULTR_IP>:/root/"
        exit 0
    fi

    # Upload to Vultr
    upload_to_vultr "$PACKAGE_PATH"

    # Clean up local package
    rm -f "$PACKAGE_PATH"

    # Upload only mode
    if [[ "$UPLOAD_ONLY" == "true" ]]; then
        print_success "Upload complete!"
        echo ""
        echo "To start training manually:"
        echo "  ssh ${SSH_USER}@${VULTR_HOST}"
        echo "  cd ${REMOTE_DIR} && source venv/bin/activate"
        echo "  python scripts/training/train_unsloth.py --config configs/${CONFIG_FILE}"
        exit 0
    fi

    # Start training
    start_training

    echo ""
    echo "=============================================="
    print_success "Deployment complete!"
    echo "=============================================="
    echo ""
    echo "Useful commands:"
    echo "  Monitor:   ssh ${SSH_USER}@${VULTR_HOST} -t 'tmux attach -t training'"
    echo "  Status:    $0 --host ${VULTR_HOST} --status"
    echo "  Download:  $0 --host ${VULTR_HOST} --download-only"
    echo ""
}

# Run main
main
