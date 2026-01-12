#!/bin/bash
# =============================================================================
# L4D2 RunPod GPU Deployment & Training - Complete One-Command Setup
# =============================================================================
# Deploys an A100 or A40 pod on RunPod, sets up training, and runs Mistral-7B
# fine-tuning on the V9 dataset with automatic GGUF export.
#
# Usage:
#   ./scripts/runpod/deploy_and_train.sh              # Full deployment + training
#   ./scripts/runpod/deploy_and_train.sh --ssh-only   # Just SSH to existing pod
#   ./scripts/runpod/deploy_and_train.sh --status     # Check training status
#   ./scripts/runpod/deploy_and_train.sh --download   # Download trained model
#   ./scripts/runpod/deploy_and_train.sh --destroy    # Terminate pod
#
# Prerequisites:
#   - RUNPOD_API_KEY environment variable set
#   - runpodctl CLI installed (or uses curl API fallback)
#   - SSH key configured in RunPod account
#
# Getting RUNPOD_API_KEY:
#   1. Go to https://www.runpod.io/console/user/settings
#   2. Click "API Keys" in the left sidebar
#   3. Click "Create API Key"
#   4. Copy the key and set: export RUNPOD_API_KEY='your-key-here'
#
# Installing runpodctl:
#   - macOS: brew install runpod/runpodctl/runpodctl
#   - Linux: curl -sSL https://raw.githubusercontent.com/runpod/runpodctl/main/install.sh | bash
#   - Or: pip install runpod
#
# Cost Estimate:
#   - A100 80GB: ~$1.99/hr x 2.5 hours = ~$5 total
#   - A40 48GB:  ~$0.79/hr x 3 hours = ~$2.40 total
#   - Training: 671 examples, 3 epochs, ~2-3 hours
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
POD_NAME="l4d2-training-$(date +%Y%m%d)"
GPU_TYPE="${RUNPOD_GPU_TYPE:-NVIDIA A100 80GB PCIe}"  # or "NVIDIA A40"
GPU_COUNT=1
CONTAINER_IMAGE="runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04"
VOLUME_SIZE=50  # GB
TRAINING_DATA="data/processed/l4d2_train_v9.jsonl"

# RunPod API endpoint
RUNPOD_API="https://api.runpod.io/graphql"

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

    if [ -z "$RUNPOD_API_KEY" ]; then
        log_error "RUNPOD_API_KEY not set."
        echo ""
        echo "To get your RunPod API key:"
        echo "  1. Go to https://www.runpod.io/console/user/settings"
        echo "  2. Click 'API Keys' in the left sidebar"
        echo "  3. Click 'Create API Key'"
        echo "  4. Copy the key and run:"
        echo "     export RUNPOD_API_KEY='your-key-here'"
        echo ""
        echo "Or use Doppler:"
        echo "  doppler run -- ./scripts/runpod/deploy_and_train.sh"
        exit 1
    fi

    # Check for runpodctl or runpod Python package
    if command -v runpodctl &> /dev/null; then
        RUNPOD_CLI="runpodctl"
        log_info "Using runpodctl CLI"
    elif python3 -c "import runpod" 2>/dev/null; then
        RUNPOD_CLI="python"
        log_info "Using runpod Python package"
    else
        RUNPOD_CLI="curl"
        log_warn "runpodctl not found, using curl API (limited functionality)"
        echo "  Install runpodctl for better experience:"
        echo "    macOS: brew install runpod/runpodctl/runpodctl"
        echo "    Linux: curl -sSL https://raw.githubusercontent.com/runpod/runpodctl/main/install.sh | bash"
    fi

    if [ ! -f "$TRAINING_DATA" ]; then
        log_error "Training data not found: $TRAINING_DATA"
        log_info "Run: python scripts/training/prepare_v9_dataset.py first"
        exit 1
    fi

    log_success "Prerequisites OK"
}

# Get SSH public key
get_ssh_key() {
    # Try common SSH key locations
    for key in "$HOME/.ssh/id_ed25519.pub" "$HOME/.ssh/id_rsa.pub" "$HOME/.ssh/runpod.pub"; do
        if [ -f "$key" ]; then
            cat "$key"
            return 0
        fi
    done

    # Generate a new key if none exists
    log_info "No SSH key found, generating one..."
    SSH_KEY_PATH="$HOME/.ssh/runpod"
    ssh-keygen -t ed25519 -f "$SSH_KEY_PATH" -N "" -C "l4d2-runpod-training"
    cat "${SSH_KEY_PATH}.pub"
}

# Create pod using runpodctl
create_pod_runpodctl() {
    log_info "Creating RunPod with runpodctl..."

    # Create pod with runpodctl
    POD_ID=$(runpodctl create pod \
        --name "$POD_NAME" \
        --gpuType "$GPU_TYPE" \
        --gpuCount $GPU_COUNT \
        --imageName "$CONTAINER_IMAGE" \
        --volumeSize $VOLUME_SIZE \
        --ports "22/tcp,6006/http" \
        2>&1 | grep -oP 'pod id:\s*\K\S+' || true)

    if [ -z "$POD_ID" ]; then
        # Try alternate parsing
        POD_ID=$(runpodctl get pod 2>/dev/null | grep "$POD_NAME" | awk '{print $1}' | head -1)
    fi

    echo "$POD_ID"
}

# Create pod using curl API
create_pod_curl() {
    log_info "Creating RunPod via API..."

    # Get SSH key
    SSH_KEY=$(get_ssh_key)

    # GraphQL mutation to create pod
    MUTATION=$(cat <<EOF
mutation {
  podFindAndDeployOnDemand(
    input: {
      name: "$POD_NAME"
      imageName: "$CONTAINER_IMAGE"
      gpuTypeId: "NVIDIA A100 80GB PCIe"
      gpuCount: $GPU_COUNT
      volumeInGb: $VOLUME_SIZE
      containerDiskInGb: 20
      minVcpuCount: 8
      minMemoryInGb: 32
      ports: "22/tcp,6006/http"
      startSsh: true
      templateId: null
      volumeMountPath: "/workspace"
    }
  ) {
    id
    name
    runtime {
      ports {
        ip
        isIpPublic
        privatePort
        publicPort
      }
    }
  }
}
EOF
)

    RESPONSE=$(curl -s -X POST "$RUNPOD_API" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -d "{\"query\": \"$(echo "$MUTATION" | tr '\n' ' ' | sed 's/"/\\"/g')\"}")

    POD_ID=$(echo "$RESPONSE" | grep -oP '"id":\s*"\K[^"]+' | head -1)

    if [ -z "$POD_ID" ]; then
        log_error "Failed to create pod. Response:"
        echo "$RESPONSE" | head -20
        exit 1
    fi

    echo "$POD_ID"
}

# Get pod info via API
get_pod_info() {
    POD_ID=$1

    QUERY=$(cat <<EOF
query {
  pod(input: { podId: "$POD_ID" }) {
    id
    name
    desiredStatus
    runtime {
      ports {
        ip
        isIpPublic
        privatePort
        publicPort
      }
      gpus {
        gpuUtilPercent
        memoryUtilPercent
      }
    }
    machine {
      gpuDisplayName
    }
  }
}
EOF
)

    curl -s -X POST "$RUNPOD_API" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -d "{\"query\": \"$(echo "$QUERY" | tr '\n' ' ' | sed 's/"/\\"/g')\"}"
}

# Wait for pod to be ready
wait_for_pod() {
    POD_ID=$1
    log_info "Waiting for pod to be ready (this takes 2-5 minutes)..."

    for i in {1..60}; do
        RESPONSE=$(get_pod_info "$POD_ID")
        STATUS=$(echo "$RESPONSE" | grep -oP '"desiredStatus":\s*"\K[^"]+' || echo "unknown")

        if [ "$STATUS" = "RUNNING" ]; then
            # Get SSH port
            SSH_PORT=$(echo "$RESPONSE" | grep -oP '"publicPort":\s*\K[0-9]+' | head -1)
            SSH_IP=$(echo "$RESPONSE" | grep -oP '"ip":\s*"\K[^"]+' | head -1)

            if [ -n "$SSH_PORT" ] && [ -n "$SSH_IP" ]; then
                log_success "Pod is running!"
                echo "$SSH_IP:$SSH_PORT"
                return 0
            fi
        fi

        echo -n "."
        sleep 5
    done

    log_error "Pod did not become ready in time"
    exit 1
}

# Create RunPod instance
create_pod() {
    log_info "Creating RunPod pod..."

    case "$RUNPOD_CLI" in
        runpodctl)
            POD_ID=$(create_pod_runpodctl)
            ;;
        *)
            POD_ID=$(create_pod_curl)
            ;;
    esac

    if [ -z "$POD_ID" ]; then
        log_error "Failed to create pod"
        exit 1
    fi

    log_success "Pod created: $POD_ID"
    echo "$POD_ID" > "$PROJECT_ROOT/.runpod_pod_id"

    # Wait for pod and get SSH details
    SSH_INFO=$(wait_for_pod "$POD_ID")
    SSH_IP=$(echo "$SSH_INFO" | cut -d: -f1)
    SSH_PORT=$(echo "$SSH_INFO" | cut -d: -f2)

    echo "$SSH_IP" > "$PROJECT_ROOT/.runpod_ssh_ip"
    echo "$SSH_PORT" > "$PROJECT_ROOT/.runpod_ssh_port"

    log_success "Pod ready at $SSH_IP:$SSH_PORT"

    # Wait for SSH to be available
    log_info "Waiting for SSH to be available..."
    for i in {1..30}; do
        if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -p "$SSH_PORT" root@"$SSH_IP" "echo ok" 2>/dev/null; then
            log_success "SSH connection established"
            break
        fi
        echo -n "."
        sleep 5
    done
    echo ""
}

# Upload code and data to pod
upload_to_pod() {
    SSH_IP=$(cat "$PROJECT_ROOT/.runpod_ssh_ip" 2>/dev/null)
    SSH_PORT=$(cat "$PROJECT_ROOT/.runpod_ssh_port" 2>/dev/null)

    if [ -z "$SSH_IP" ] || [ -z "$SSH_PORT" ]; then
        log_error "No pod connection info found. Run deploy first."
        exit 1
    fi

    log_info "Uploading training code and data to pod..."

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
        scripts/training/train_runpod.py \
        scripts/training/export_model.py \
        scripts/utils/ \
        configs/ \
        data/processed/l4d2_train_v9.jsonl \
        requirements.txt \
        2>/dev/null || true

    log_info "Uploading $(du -h "$TARBALL" | cut -f1) of training data..."

    scp -o StrictHostKeyChecking=no -P "$SSH_PORT" \
        "$TARBALL" root@"$SSH_IP":/workspace/training.tar.gz

    # Extract on remote
    ssh -o StrictHostKeyChecking=no -p "$SSH_PORT" root@"$SSH_IP" << 'REMOTE'
mkdir -p /workspace/L4D2-AI-Architect
cd /workspace/L4D2-AI-Architect
tar -xzf /workspace/training.tar.gz
rm /workspace/training.tar.gz
echo "Files extracted successfully"
ls -la
REMOTE

    rm "$TARBALL"
    log_success "Upload complete"
}

# Start training on pod
start_training() {
    SSH_IP=$(cat "$PROJECT_ROOT/.runpod_ssh_ip" 2>/dev/null)
    SSH_PORT=$(cat "$PROJECT_ROOT/.runpod_ssh_port" 2>/dev/null)

    if [ -z "$SSH_IP" ] || [ -z "$SSH_PORT" ]; then
        log_error "No pod connection info found. Run deploy first."
        exit 1
    fi

    log_info "Starting training on RunPod..."

    ssh -o StrictHostKeyChecking=no -p "$SSH_PORT" root@"$SSH_IP" << 'REMOTE'
cd /workspace/L4D2-AI-Architect

# Create directories
mkdir -p model_adapters exports data/training_logs

# Setup Python environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -q --upgrade pip
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers datasets accelerate peft bitsandbytes trl
pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install -q flash-attn --no-build-isolation 2>/dev/null || echo "Flash attention skipped"

# Install additional requirements
pip install -q pyyaml tensorboard

# Verify GPU
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Determine config based on GPU
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if [[ "$GPU_NAME" == *"A100"* ]]; then
    CONFIG="configs/unsloth_config_a100.yaml"
    BATCH_SIZE=8
    echo "Using A100 config"
elif [[ "$GPU_NAME" == *"H100"* ]]; then
    CONFIG="configs/unsloth_config_a100.yaml"  # H100 uses similar settings
    BATCH_SIZE=16
    echo "Using H100-optimized config"
else
    CONFIG="configs/unsloth_config.yaml"
    BATCH_SIZE=4
    echo "Using default config"
fi

# Start training in tmux
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
tmux new-session -d -s training "
    cd /workspace/L4D2-AI-Architect && \
    source venv/bin/activate && \
    echo '=== Starting L4D2 Mistral-7B Training ===' && \
    echo 'Dataset: data/processed/l4d2_train_v9.jsonl' && \
    wc -l data/processed/l4d2_train_v9.jsonl && \
    python scripts/training/train_unsloth.py \
        --config $CONFIG \
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
    echo "Training is running on RunPod"
    echo "=========================================="
    echo ""
    echo "Pod SSH: ssh -p $SSH_PORT root@$SSH_IP"
    echo "Monitor: ssh -p $SSH_PORT root@$SSH_IP 'tmux attach -t training'"
    echo ""
    echo "Estimated time: 2-3 hours"
    echo "Estimated cost: ~\$5-8 (A100) or ~\$2-4 (A40)"
    echo ""
    echo "When complete, download model with:"
    echo "  ./scripts/runpod/deploy_and_train.sh --download"
}

# Check training status
check_status() {
    SSH_IP=$(cat "$PROJECT_ROOT/.runpod_ssh_ip" 2>/dev/null)
    SSH_PORT=$(cat "$PROJECT_ROOT/.runpod_ssh_port" 2>/dev/null)

    if [ -z "$SSH_IP" ] || [ -z "$SSH_PORT" ]; then
        log_error "No pod connection info found"
        exit 1
    fi

    log_info "Checking training status on $SSH_IP:$SSH_PORT..."

    ssh -o StrictHostKeyChecking=no -p "$SSH_PORT" root@"$SSH_IP" << 'REMOTE'
cd /workspace/L4D2-AI-Architect

if tmux has-session -t training 2>/dev/null; then
    echo "Training is RUNNING"
    echo ""
    echo "Last 20 lines of training.log:"
    tail -20 training.log 2>/dev/null || echo "(no log yet)"
else
    echo "Training session not found (may be complete)"
    echo ""
    if ls exports/*.gguf 1> /dev/null 2>&1; then
        echo "Model is READY for download:"
        ls -lh exports/*.gguf
    else
        echo "No exported model found yet"
        echo ""
        echo "Check training log:"
        tail -30 training.log 2>/dev/null || echo "(no log)"
    fi
fi
REMOTE
}

# Download trained model
download_model() {
    SSH_IP=$(cat "$PROJECT_ROOT/.runpod_ssh_ip" 2>/dev/null)
    SSH_PORT=$(cat "$PROJECT_ROOT/.runpod_ssh_port" 2>/dev/null)

    if [ -z "$SSH_IP" ] || [ -z "$SSH_PORT" ]; then
        log_error "No pod connection info found"
        exit 1
    fi

    log_info "Downloading trained model from $SSH_IP:$SSH_PORT..."

    mkdir -p "$PROJECT_ROOT/exports"

    scp -o StrictHostKeyChecking=no -P "$SSH_PORT" \
        root@"$SSH_IP":/workspace/L4D2-AI-Architect/exports/*.gguf \
        "$PROJECT_ROOT/exports/"

    log_success "Model downloaded to exports/"
    ls -lh "$PROJECT_ROOT/exports/"*.gguf

    echo ""
    echo "To use with Ollama:"
    echo "  1. Create Modelfile with: FROM ./exports/l4d2-mistral-v9.gguf"
    echo "  2. ollama create l4d2-mistral -f Modelfile"
    echo "  3. ollama run l4d2-mistral"
}

# SSH to pod
ssh_to_pod() {
    SSH_IP=$(cat "$PROJECT_ROOT/.runpod_ssh_ip" 2>/dev/null)
    SSH_PORT=$(cat "$PROJECT_ROOT/.runpod_ssh_port" 2>/dev/null)

    if [ -z "$SSH_IP" ] || [ -z "$SSH_PORT" ]; then
        log_error "No pod connection info found"
        exit 1
    fi

    log_info "Connecting to $SSH_IP:$SSH_PORT..."
    ssh -o StrictHostKeyChecking=no -p "$SSH_PORT" root@"$SSH_IP"
}

# Destroy pod
destroy_pod() {
    POD_ID=$(cat "$PROJECT_ROOT/.runpod_pod_id" 2>/dev/null)

    if [ -z "$POD_ID" ]; then
        log_info "No pod ID found"
        return
    fi

    log_warn "Destroying pod $POD_ID..."

    if [ "$RUNPOD_CLI" = "runpodctl" ]; then
        runpodctl remove pod "$POD_ID" 2>/dev/null || true
    else
        # Use API to terminate
        MUTATION="mutation { podTerminate(input: { podId: \"$POD_ID\" }) }"
        curl -s -X POST "$RUNPOD_API" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $RUNPOD_API_KEY" \
            -d "{\"query\": \"$MUTATION\"}" > /dev/null
    fi

    rm -f "$PROJECT_ROOT/.runpod_pod_id" "$PROJECT_ROOT/.runpod_ssh_ip" "$PROJECT_ROOT/.runpod_ssh_port"
    log_success "Pod destroyed"
}

# List available GPU types
list_gpus() {
    log_info "Fetching available GPU types..."

    QUERY='query { gpuTypes { id displayName memoryInGb secureCloud communityCloud lowestPrice { minimumBidPrice } } }'

    RESPONSE=$(curl -s -X POST "$RUNPOD_API" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -d "{\"query\": \"$QUERY\"}")

    echo "$RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
gpus = data.get('data', {}).get('gpuTypes', [])
print(f'{'GPU Type':<35} {'VRAM':<10} {'Price/hr':<12} {'Available'}')
print('-' * 70)
for gpu in sorted(gpus, key=lambda x: x.get('lowestPrice', {}).get('minimumBidPrice', 999) or 999):
    name = gpu.get('displayName', gpu.get('id', 'Unknown'))
    mem = gpu.get('memoryInGb', 0)
    price = gpu.get('lowestPrice', {}).get('minimumBidPrice', 'N/A')
    if price != 'N/A':
        price = f'\${price:.2f}'
    secure = 'Yes' if gpu.get('secureCloud') else 'No'
    community = 'Yes' if gpu.get('communityCloud') else 'No'
    avail = f'Secure: {secure}, Community: {community}'
    print(f'{name:<35} {mem:<10} {price:<12} {avail}')
" 2>/dev/null || echo "$RESPONSE"
}

# Show help
show_help() {
    cat << 'HELP'
L4D2 RunPod GPU Training Deployment

Usage: ./scripts/runpod/deploy_and_train.sh [option]

Options:
  (none)        Full deployment + training
  --ssh         SSH to existing pod
  --status      Check training status
  --download    Download trained model
  --destroy     Terminate pod (save costs!)
  --list-gpus   List available GPU types and prices
  --help        Show this help

Environment Variables:
  RUNPOD_API_KEY      Required. Your RunPod API key
  RUNPOD_GPU_TYPE     GPU type (default: "NVIDIA A100 80GB PCIe")

Examples:
  # Full deployment with A100
  export RUNPOD_API_KEY='your-key'
  ./scripts/runpod/deploy_and_train.sh

  # Use A40 instead (cheaper)
  export RUNPOD_GPU_TYPE="NVIDIA A40"
  ./scripts/runpod/deploy_and_train.sh

  # Check status
  ./scripts/runpod/deploy_and_train.sh --status

  # Download model when done
  ./scripts/runpod/deploy_and_train.sh --download

  # IMPORTANT: Destroy pod to stop charges!
  ./scripts/runpod/deploy_and_train.sh --destroy

Getting RUNPOD_API_KEY:
  1. Go to https://www.runpod.io/console/user/settings
  2. Click "API Keys" in the left sidebar
  3. Click "Create API Key"
  4. Copy and export: export RUNPOD_API_KEY='your-key'
HELP
}

# Main
main() {
    case "${1:-}" in
        --ssh-only|--ssh)
            ssh_to_pod
            ;;
        --status)
            check_prerequisites
            check_status
            ;;
        --download)
            check_prerequisites
            download_model
            ;;
        --destroy)
            check_prerequisites
            destroy_pod
            ;;
        --list-gpus|--gpus)
            check_prerequisites
            list_gpus
            ;;
        --help|-h)
            show_help
            ;;
        *)
            echo "=============================================="
            echo "L4D2 RunPod GPU Training Deployment"
            echo "=============================================="
            echo ""

            check_prerequisites

            # Check for existing pod
            if [ -f "$PROJECT_ROOT/.runpod_pod_id" ]; then
                POD_ID=$(cat "$PROJECT_ROOT/.runpod_pod_id")
                SSH_IP=$(cat "$PROJECT_ROOT/.runpod_ssh_ip" 2>/dev/null || echo "")
                SSH_PORT=$(cat "$PROJECT_ROOT/.runpod_ssh_port" 2>/dev/null || echo "")

                log_warn "Existing pod found: $POD_ID"
                if [ -n "$SSH_IP" ] && [ -n "$SSH_PORT" ]; then
                    echo "  SSH: ssh -p $SSH_PORT root@$SSH_IP"
                fi
                echo ""
                read -p "Use existing pod? [Y/n] " -n 1 -r
                echo ""
                if [[ $REPLY =~ ^[Nn]$ ]]; then
                    destroy_pod
                    create_pod
                fi
            else
                create_pod
            fi

            upload_to_pod
            start_training

            echo ""
            echo "=========================================="
            echo "IMPORTANT: Remember to destroy the pod!"
            echo "=========================================="
            echo "./scripts/runpod/deploy_and_train.sh --destroy"
            echo ""
            ;;
    esac
}

main "$@"
