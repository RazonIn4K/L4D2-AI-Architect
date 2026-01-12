#!/bin/bash
# =============================================================================
# L4D2-AI-Architect: Deploy to Vultr and Start Training
# =============================================================================
#
# One-command deployment to Vultr GPU instance.
#
# Usage:
#   ./deploy_to_vultr.sh <VULTR_IP>                    # Deploy and start training
#   ./deploy_to_vultr.sh <VULTR_IP> --sync-only        # Just sync files
#   ./deploy_to_vultr.sh <VULTR_IP> --start-only       # Just start training
#   ./deploy_to_vultr.sh <VULTR_IP> --status           # Check training status
#   ./deploy_to_vultr.sh <VULTR_IP> --download         # Download trained models
#
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_DIR="/root/L4D2-AI-Architect"

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${CYAN}[STEP]${NC} $1"; }

show_help() {
    echo ""
    echo -e "${BOLD}L4D2-AI-Architect - Deploy to Vultr${NC}"
    echo ""
    echo "Usage: $0 <VULTR_IP> [option]"
    echo ""
    echo "Options:"
    echo "  (none)         Full deploy: sync files and start training"
    echo "  --sync-only    Just sync files, don't start training"
    echo "  --start-only   Just start training (assumes files are synced)"
    echo "  --status       Check training status"
    echo "  --download     Download trained models"
    echo "  --ssh          SSH into the instance"
    echo "  --help         Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 123.45.67.89                # Full deployment"
    echo "  $0 123.45.67.89 --status       # Check progress"
    echo "  $0 123.45.67.89 --download     # Get models"
    echo ""
}

sync_files() {
    local ip=$1
    log_step "Syncing files to ${ip}..."

    # Create remote directory
    ssh -o StrictHostKeyChecking=no root@${ip} "mkdir -p ${REMOTE_DIR}"

    # Sync with rsync (excluding large/unnecessary files)
    rsync -avz --progress \
        --exclude 'venv' \
        --exclude '.git' \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '.DS_Store' \
        --exclude 'node_modules' \
        --exclude '.claude' \
        --exclude 'model_adapters/l4d2-mistral-v9-lora' \
        --exclude 'model_adapters/l4d2-mistral-v10-lora' \
        --exclude 'model_adapters/rl_test' \
        --exclude 'exports/l4d2-v10plus' \
        "${SCRIPT_DIR}/" root@${ip}:${REMOTE_DIR}/

    log_success "Files synced to ${ip}:${REMOTE_DIR}"
}

start_training() {
    local ip=$1
    log_step "Starting training on ${ip}..."

    # Make scripts executable and start training
    ssh -t root@${ip} "
        cd ${REMOTE_DIR} && \
        chmod +x scripts/utils/*.sh run_*.sh && \
        echo '' && \
        echo '============================================' && \
        echo 'Starting Vultr Credit Burn Script' && \
        echo '============================================' && \
        ./scripts/utils/vultr_credit_burn.sh llm
    "

    log_success "Training started on ${ip}"
}

check_status() {
    local ip=$1
    log_step "Checking training status on ${ip}..."

    ssh root@${ip} "
        cd ${REMOTE_DIR} 2>/dev/null || exit 1

        echo ''
        echo '=== GPU Status ==='
        nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv
        echo ''

        echo '=== Training Session ==='
        if tmux has-session -t training 2>/dev/null; then
            echo 'Status: RUNNING'
            echo ''
            echo 'Recent output:'
            tmux capture-pane -t training -p | tail -20
        else
            echo 'Status: NOT RUNNING'
        fi
        echo ''

        echo '=== Trained Models ==='
        ls -la model_adapters/l4d2-mistral-*/final 2>/dev/null || echo 'No trained models yet'
        echo ''

        echo '=== Exported Models ==='
        ls -lh exports/*/gguf/*.gguf 2>/dev/null || echo 'No GGUF exports yet'
        echo ''
    "
}

download_models() {
    local ip=$1
    local local_dir="${SCRIPT_DIR}/downloads_$(date +%Y%m%d_%H%M%S)"

    log_step "Downloading models from ${ip}..."
    mkdir -p "${local_dir}"

    # Download exports
    log_info "Downloading GGUF exports..."
    rsync -avz --progress \
        root@${ip}:${REMOTE_DIR}/exports/ \
        "${local_dir}/exports/" 2>/dev/null || log_warn "No exports found"

    # Download adapters
    log_info "Downloading LoRA adapters..."
    rsync -avz --progress \
        --exclude 'checkpoint-*' \
        root@${ip}:${REMOTE_DIR}/model_adapters/l4d2-mistral-v*/final/ \
        "${local_dir}/model_adapters/" 2>/dev/null || log_warn "No adapters found"

    # Download training logs
    log_info "Downloading training logs..."
    rsync -avz --progress \
        root@${ip}:${REMOTE_DIR}/data/training_logs/ \
        "${local_dir}/training_logs/" 2>/dev/null || log_warn "No logs found"

    # Download embeddings if present
    rsync -avz --progress \
        root@${ip}:${REMOTE_DIR}/data/embeddings/ \
        "${local_dir}/embeddings/" 2>/dev/null || true

    log_success "Models downloaded to: ${local_dir}"
    echo ""
    echo "Contents:"
    ls -la "${local_dir}/"
}

ssh_instance() {
    local ip=$1
    log_info "Connecting to ${ip}..."
    ssh -t root@${ip} "cd ${REMOTE_DIR} && bash"
}

# =============================================================================
# Main
# =============================================================================

if [ $# -lt 1 ]; then
    show_help
    exit 1
fi

VULTR_IP=$1
OPTION=${2:-}

# Validate IP
if ! [[ $VULTR_IP =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    log_error "Invalid IP address: $VULTR_IP"
    exit 1
fi

echo ""
echo -e "${BOLD}============================================${NC}"
echo -e "${BOLD}L4D2-AI-Architect - Vultr Deployment${NC}"
echo -e "${BOLD}============================================${NC}"
echo ""
echo "Target: root@${VULTR_IP}"
echo ""

case "$OPTION" in
    --help|-h)
        show_help
        ;;
    --sync-only)
        sync_files "$VULTR_IP"
        ;;
    --start-only)
        start_training "$VULTR_IP"
        ;;
    --status)
        check_status "$VULTR_IP"
        ;;
    --download)
        download_models "$VULTR_IP"
        ;;
    --ssh)
        ssh_instance "$VULTR_IP"
        ;;
    "")
        # Full deployment
        log_info "Starting full deployment..."
        echo ""

        sync_files "$VULTR_IP"
        echo ""

        log_info "Files synced. Starting training..."
        echo ""

        start_training "$VULTR_IP"
        echo ""

        echo -e "${GREEN}${BOLD}============================================${NC}"
        echo -e "${GREEN}${BOLD}  Deployment Complete!${NC}"
        echo -e "${GREEN}${BOLD}============================================${NC}"
        echo ""
        echo "Training is now running on ${VULTR_IP}"
        echo ""
        echo "Commands:"
        echo "  $0 ${VULTR_IP} --status      # Check progress"
        echo "  $0 ${VULTR_IP} --ssh         # SSH to instance"
        echo "  $0 ${VULTR_IP} --download    # Download models"
        echo ""
        ;;
    *)
        log_error "Unknown option: $OPTION"
        show_help
        exit 1
        ;;
esac
