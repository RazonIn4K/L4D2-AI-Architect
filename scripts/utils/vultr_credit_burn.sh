#!/bin/bash
# =============================================================================
# L4D2-AI-Architect: Vultr Credit Burn Master Script
# =============================================================================
#
# Comprehensive script to maximize Vultr GPU credits by running all AI/ML
# workloads: LLM fine-tuning, RL agent training, and embeddings generation.
#
# Usage:
#   ./vultr_credit_burn.sh llm         # Run LLM fine-tuning only
#   ./vultr_credit_burn.sh rl          # Run RL training only
#   ./vultr_credit_burn.sh embeddings  # Run embeddings generation only
#   ./vultr_credit_burn.sh all         # Run all workloads in sequence
#
# Prerequisites:
#   - Vultr A100/A40/GH200 GPU instance
#   - Vultr Object Storage bucket configured
#   - Environment variables set in .env or exported:
#     - VULTR_OBJECT_STORAGE_HOSTNAME
#     - VULTR_OBJECT_STORAGE_ACCESS_KEY
#     - VULTR_OBJECT_STORAGE_SECRET_KEY
#     - VULTR_BUCKET_NAME (default: l4d2-artifacts)
#
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Timestamps and identifiers
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="vultr-burn-${TIMESTAMP}"

# Directories (local)
LOG_DIR="${PROJECT_ROOT}/logs/${RUN_ID}"
CHECKPOINT_DIR="${PROJECT_ROOT}/model_adapters"
EXPORT_DIR="${PROJECT_ROOT}/exports"
EMBEDDINGS_DIR="${PROJECT_ROOT}/data/embeddings"

# Object Storage paths
BUCKET_NAME="${VULTR_BUCKET_NAME:-l4d2-artifacts}"
REMOTE_ARTIFACTS="artifacts"
REMOTE_DATASETS="datasets"
REMOTE_MODELS="models"
REMOTE_EMBEDDINGS="embeddings"
REMOTE_LOGS="logs"
REMOTE_SNAPSHOTS="snapshots"

# Training defaults
LLM_CONFIG="configs/unsloth_config_v15.yaml"
LLM_EPOCHS=3
RL_TIMESTEPS=500000
RL_PERSONALITIES=("aggressive" "medic" "defender")
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

# Auto-upload interval (seconds)
UPLOAD_INTERVAL=1800  # 30 minutes

# Estimated costs (A100 @ $2.50/hr, A40 @ $1.50/hr)
declare -A ESTIMATED_TIMES
ESTIMATED_TIMES[llm]="2-3 hours"
ESTIMATED_TIMES[rl]="4-6 hours"
ESTIMATED_TIMES[embeddings]="30-60 minutes"
ESTIMATED_TIMES[all]="7-10 hours"

declare -A ESTIMATED_COSTS
ESTIMATED_COSTS[llm]="\$5-8"
ESTIMATED_COSTS[rl]="\$10-15"
ESTIMATED_COSTS[embeddings]="\$1-3"
ESTIMATED_COSTS[all]="\$16-26"

# =============================================================================
# COLORS AND LOGGING
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $(date '+%H:%M:%S') $1" | tee -a "${LOG_DIR}/main.log"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $(date '+%H:%M:%S') $1" | tee -a "${LOG_DIR}/main.log"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') $1" | tee -a "${LOG_DIR}/main.log"; }
log_step() { echo -e "${CYAN}[STEP]${NC} $(date '+%H:%M:%S') $1" | tee -a "${LOG_DIR}/main.log"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $(date '+%H:%M:%S') $1" | tee -a "${LOG_DIR}/main.log"; }

# =============================================================================
# CHECKPOINT SYSTEM
# =============================================================================

CHECKPOINT_FILE="${LOG_DIR}/checkpoint.state"

save_checkpoint() {
    local stage=$1
    local status=$2
    echo "${stage}:${status}:$(date +%s)" >> "${CHECKPOINT_FILE}"
    log_info "Checkpoint saved: ${stage} = ${status}"
}

get_checkpoint() {
    local stage=$1
    if [ -f "${CHECKPOINT_FILE}" ]; then
        grep "^${stage}:" "${CHECKPOINT_FILE}" | tail -1 | cut -d: -f2
    fi
}

is_stage_complete() {
    local stage=$1
    [ "$(get_checkpoint ${stage})" = "complete" ]
}

# =============================================================================
# OBJECT STORAGE SETUP
# =============================================================================

setup_object_storage() {
    log_step "Setting up Vultr Object Storage connection..."

    # Load environment variables (with validation)
    if [ -f "${PROJECT_ROOT}/.env" ]; then
        # Security: Only source if .env is not world-writable
        if [ "$(stat -c '%a' "${PROJECT_ROOT}/.env" 2>/dev/null || stat -f '%Lp' "${PROJECT_ROOT}/.env" 2>/dev/null)" != *"2"* ] && \
           [ "$(stat -c '%a' "${PROJECT_ROOT}/.env" 2>/dev/null || stat -f '%Lp' "${PROJECT_ROOT}/.env" 2>/dev/null)" != *"6"* ]; then
            set -a
            # Only load specific expected variables, not arbitrary code
            while IFS='=' read -r key value; do
                # Skip comments and empty lines
                [[ "$key" =~ ^[[:space:]]*# ]] && continue
                [[ -z "$key" ]] && continue
                # Only allow expected variable names (alphanumeric and underscore)
                if [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]]; then
                    # Remove surrounding quotes if present
                    value="${value%\"}"
                    value="${value#\"}"
                    value="${value%\'}"
                    value="${value#\'}"
                    export "$key=$value"
                fi
            done < "${PROJECT_ROOT}/.env"
            set +a
        else
            log_warn ".env file has insecure permissions, skipping"
        fi
    fi

    # Validate credentials
    if [ -z "${VULTR_OBJECT_STORAGE_HOSTNAME}" ] || \
       [ -z "${VULTR_OBJECT_STORAGE_ACCESS_KEY}" ] || \
       [ -z "${VULTR_OBJECT_STORAGE_SECRET_KEY}" ]; then
        log_warn "Object Storage credentials not found in environment"
        log_warn "Auto-upload will be disabled. Set these in .env:"
        log_warn "  VULTR_OBJECT_STORAGE_HOSTNAME"
        log_warn "  VULTR_OBJECT_STORAGE_ACCESS_KEY"
        log_warn "  VULTR_OBJECT_STORAGE_SECRET_KEY"
        STORAGE_AVAILABLE=false
        return 1
    fi

    # Install s3cmd if not available
    if ! command -v s3cmd &> /dev/null; then
        log_info "Installing s3cmd..."
        pip install -q s3cmd
    fi

    # Create s3cmd config with secure permissions
    S3CMD_CONFIG="${HOME}/.s3cfg-vultr"
    # Set umask before creating file with credentials
    (
        umask 077
        cat > "${S3CMD_CONFIG}" << EOF
[default]
access_key = ${VULTR_OBJECT_STORAGE_ACCESS_KEY}
secret_key = ${VULTR_OBJECT_STORAGE_SECRET_KEY}
host_base = ${VULTR_OBJECT_STORAGE_HOSTNAME}
host_bucket = %(bucket)s.${VULTR_OBJECT_STORAGE_HOSTNAME}
use_https = True
signature_v2 = False
EOF
    )
    # Verify permissions are restrictive
    chmod 600 "${S3CMD_CONFIG}"

    # Test connection
    log_info "Testing Object Storage connection..."
    if s3cmd -c "${S3CMD_CONFIG}" ls "s3://${BUCKET_NAME}/" &> /dev/null; then
        log_success "Object Storage connected: s3://${BUCKET_NAME}/"
        STORAGE_AVAILABLE=true
    else
        log_warn "Could not connect to bucket. Creating bucket..."
        if s3cmd -c "${S3CMD_CONFIG}" mb "s3://${BUCKET_NAME}/" 2>/dev/null; then
            log_success "Bucket created: s3://${BUCKET_NAME}/"
            STORAGE_AVAILABLE=true
        else
            log_error "Failed to create bucket. Check credentials."
            STORAGE_AVAILABLE=false
            return 1
        fi
    fi

    # Create folder structure
    log_info "Creating Object Storage folder structure..."
    for folder in "${REMOTE_ARTIFACTS}" "${REMOTE_DATASETS}" "${REMOTE_MODELS}" \
                  "${REMOTE_EMBEDDINGS}" "${REMOTE_LOGS}" "${REMOTE_SNAPSHOTS}"; do
        # s3cmd doesn't need explicit folder creation, but we'll create placeholder files
        echo "Created by vultr_credit_burn.sh at $(date)" | \
            s3cmd -c "${S3CMD_CONFIG}" put - "s3://${BUCKET_NAME}/${folder}/.placeholder" 2>/dev/null || true
    done

    log_success "Object Storage structure created"
    return 0
}

# Upload to Object Storage
upload_to_storage() {
    local local_path=$1
    local remote_path=$2
    local description=${3:-"files"}

    if [ "${STORAGE_AVAILABLE}" != "true" ]; then
        log_warn "Object Storage not available, skipping upload of ${description}"
        return 1
    fi

    if [ ! -e "${local_path}" ]; then
        log_warn "Local path not found: ${local_path}"
        return 1
    fi

    log_info "Uploading ${description} to s3://${BUCKET_NAME}/${remote_path}..."

    if [ -d "${local_path}" ]; then
        s3cmd -c "${S3CMD_CONFIG}" sync --recursive "${local_path}/" "s3://${BUCKET_NAME}/${remote_path}/" 2>&1 | \
            tee -a "${LOG_DIR}/upload.log"
    else
        s3cmd -c "${S3CMD_CONFIG}" put "${local_path}" "s3://${BUCKET_NAME}/${remote_path}/" 2>&1 | \
            tee -a "${LOG_DIR}/upload.log"
    fi

    if [ $? -eq 0 ]; then
        log_success "Uploaded ${description}"
        return 0
    else
        log_error "Failed to upload ${description}"
        return 1
    fi
}

# =============================================================================
# BACKGROUND UPLOAD DAEMON
# =============================================================================

start_upload_daemon() {
    log_info "Starting background upload daemon (every ${UPLOAD_INTERVAL}s)..."

    (
        while true; do
            sleep ${UPLOAD_INTERVAL}

            if [ "${STORAGE_AVAILABLE}" = "true" ]; then
                echo "[$(date '+%H:%M:%S')] Auto-upload triggered" >> "${LOG_DIR}/upload.log"

                # Upload logs
                upload_to_storage "${LOG_DIR}" "${REMOTE_LOGS}/${RUN_ID}" "logs" 2>/dev/null

                # Upload checkpoints
                if [ -d "${CHECKPOINT_DIR}" ]; then
                    upload_to_storage "${CHECKPOINT_DIR}" "${REMOTE_MODELS}/checkpoints-${RUN_ID}" "checkpoints" 2>/dev/null
                fi

                # Upload any exports
                if [ -d "${EXPORT_DIR}" ]; then
                    upload_to_storage "${EXPORT_DIR}" "${REMOTE_MODELS}/exports-${RUN_ID}" "exports" 2>/dev/null
                fi
            fi
        done
    ) &

    UPLOAD_DAEMON_PID=$!
    echo "${UPLOAD_DAEMON_PID}" > "${LOG_DIR}/upload_daemon.pid"
    log_info "Upload daemon started (PID: ${UPLOAD_DAEMON_PID})"
}

stop_upload_daemon() {
    if [ -f "${LOG_DIR}/upload_daemon.pid" ]; then
        local pid=$(cat "${LOG_DIR}/upload_daemon.pid")
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
            log_info "Upload daemon stopped"
        fi
        rm -f "${LOG_DIR}/upload_daemon.pid"
    fi
}

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

setup_environment() {
    log_step "Setting up environment..."

    # Create directories
    mkdir -p "${LOG_DIR}"
    mkdir -p "${CHECKPOINT_DIR}"
    mkdir -p "${EXPORT_DIR}"
    mkdir -p "${EMBEDDINGS_DIR}"

    # Create/activate virtual environment
    if [ ! -d "${PROJECT_ROOT}/venv" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv "${PROJECT_ROOT}/venv"
    fi

    source "${PROJECT_ROOT}/venv/bin/activate"

    # Install base dependencies
    log_info "Installing dependencies..."
    pip install -q --upgrade pip
    pip install -q -r "${PROJECT_ROOT}/requirements.txt"

    # Check GPU
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. Are you on a GPU instance?"
        exit 1
    fi

    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

    log_info "GPU: ${GPU_NAME} (${GPU_MEM}MB)"

    # Determine GPU tier
    if [[ "${GPU_NAME}" == *"A100"* ]] || [ "${GPU_MEM}" -gt 38000 ]; then
        GPU_TIER="a100"
        LLM_CONFIG="configs/unsloth_config_v15.yaml"
    elif [[ "${GPU_NAME}" == *"H100"* ]] || [[ "${GPU_NAME}" == *"GH200"* ]]; then
        GPU_TIER="gh200"
        LLM_CONFIG="configs/unsloth_config_gh200.yaml"
    else
        GPU_TIER="a40"
        LLM_CONFIG="configs/unsloth_config.yaml"
    fi

    log_info "GPU tier: ${GPU_TIER}, using config: ${LLM_CONFIG}"

    # Log system info
    echo "=== System Info ===" >> "${LOG_DIR}/system_info.log"
    uname -a >> "${LOG_DIR}/system_info.log"
    nvidia-smi >> "${LOG_DIR}/system_info.log"
    python --version >> "${LOG_DIR}/system_info.log"
    pip list >> "${LOG_DIR}/system_info.log"

    save_checkpoint "environment" "complete"
}

# =============================================================================
# LLM FINE-TUNING WORKLOAD
# =============================================================================

run_llm_training() {
    log_step "=========================================="
    log_step "LLM FINE-TUNING WORKLOAD"
    log_step "=========================================="

    if is_stage_complete "llm_training"; then
        log_info "LLM training already complete, skipping..."
        return 0
    fi

    save_checkpoint "llm_training" "started"

    # Check for training data
    TRAINING_DATA=""
    for version in v15 v14 v13 v12 v11 v10 v9 v8; do
        if [ -f "${PROJECT_ROOT}/data/processed/l4d2_train_${version}.jsonl" ]; then
            TRAINING_DATA="${PROJECT_ROOT}/data/processed/l4d2_train_${version}.jsonl"
            log_info "Using dataset: l4d2_train_${version}.jsonl"
            break
        fi
    done

    if [ -z "${TRAINING_DATA}" ]; then
        log_error "No training data found in data/processed/"
        save_checkpoint "llm_training" "failed"
        return 1
    fi

    SAMPLE_COUNT=$(wc -l < "${TRAINING_DATA}")
    log_info "Training samples: ${SAMPLE_COUNT}"

    # Install Unsloth
    log_info "Installing Unsloth and dependencies..."
    pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install -q --no-deps trl peft accelerate bitsandbytes
    pip install -q flash-attn --no-build-isolation 2>/dev/null || log_warn "Flash Attention not available"

    # Run training
    LLM_OUTPUT_DIR="${CHECKPOINT_DIR}/l4d2-mistral-${RUN_ID}"

    log_info "Starting LLM training..."
    log_info "Config: ${LLM_CONFIG}"
    log_info "Output: ${LLM_OUTPUT_DIR}"

    python "${PROJECT_ROOT}/scripts/training/train_unsloth.py" \
        --config "${PROJECT_ROOT}/${LLM_CONFIG}" \
        --epochs ${LLM_EPOCHS} \
        2>&1 | tee "${LOG_DIR}/llm_training.log"

    if [ $? -ne 0 ]; then
        log_error "LLM training failed"
        save_checkpoint "llm_training" "failed"
        return 1
    fi

    save_checkpoint "llm_training" "complete"
    log_success "LLM training complete"

    # Export to GGUF
    run_llm_export
}

run_llm_export() {
    log_step "Exporting LLM to GGUF format..."

    if is_stage_complete "llm_export"; then
        log_info "LLM export already complete, skipping..."
        return 0
    fi

    save_checkpoint "llm_export" "started"

    # Find the latest adapter
    LATEST_ADAPTER=$(ls -td "${CHECKPOINT_DIR}"/l4d2-mistral-*/final 2>/dev/null | head -1)

    if [ -z "${LATEST_ADAPTER}" ] || [ ! -d "${LATEST_ADAPTER}" ]; then
        log_error "No trained adapter found for export"
        save_checkpoint "llm_export" "failed"
        return 1
    fi

    log_info "Exporting adapter: ${LATEST_ADAPTER}"

    EXPORT_OUTPUT="${EXPORT_DIR}/l4d2-${RUN_ID}"

    python "${PROJECT_ROOT}/scripts/training/export_gguf_cpu.py" \
        --adapter "${LATEST_ADAPTER}" \
        --output "${EXPORT_OUTPUT}" \
        --quantize q4_k_m \
        --create-modelfile \
        2>&1 | tee "${LOG_DIR}/llm_export.log"

    if [ $? -eq 0 ]; then
        save_checkpoint "llm_export" "complete"
        log_success "LLM export complete: ${EXPORT_OUTPUT}"

        # Upload to Object Storage
        upload_to_storage "${EXPORT_OUTPUT}" "${REMOTE_MODELS}/llm-${RUN_ID}" "LLM model"
    else
        log_warn "LLM export had issues, check logs"
        save_checkpoint "llm_export" "partial"
    fi
}

# =============================================================================
# RL AGENT TRAINING WORKLOAD
# =============================================================================

run_rl_training() {
    log_step "=========================================="
    log_step "RL AGENT TRAINING WORKLOAD"
    log_step "=========================================="

    if is_stage_complete "rl_training"; then
        log_info "RL training already complete, skipping..."
        return 0
    fi

    save_checkpoint "rl_training" "started"

    # Install RL dependencies
    log_info "Installing RL dependencies..."
    pip install -q stable-baselines3[extra] gymnasium tensorboard

    RL_OUTPUT_DIR="${CHECKPOINT_DIR}/rl_agents/${RUN_ID}"
    mkdir -p "${RL_OUTPUT_DIR}"

    # Train each personality
    for personality in "${RL_PERSONALITIES[@]}"; do
        log_info "Training ${personality} agent (${RL_TIMESTEPS} timesteps)..."

        PERSONALITY_DIR="${RL_OUTPUT_DIR}/${personality}"

        # Check if this personality is already trained
        if is_stage_complete "rl_${personality}"; then
            log_info "${personality} agent already trained, skipping..."
            continue
        fi

        save_checkpoint "rl_${personality}" "started"

        # Note: This requires the Mnemosyne environment to be running
        # In practice, this would connect to a L4D2 game server
        # For now, we'll create a mock training run or skip if no server

        python "${PROJECT_ROOT}/scripts/rl_training/train_ppo.py" \
            --mode train \
            --timesteps ${RL_TIMESTEPS} \
            --personality "${personality}" \
            --save-path "${PERSONALITY_DIR}" \
            --n-envs 4 \
            2>&1 | tee "${LOG_DIR}/rl_${personality}.log" || {
                log_warn "RL training for ${personality} failed (likely no game server)"
                save_checkpoint "rl_${personality}" "skipped"
                continue
            }

        save_checkpoint "rl_${personality}" "complete"
        log_success "${personality} agent trained"

        # Upload this agent
        upload_to_storage "${PERSONALITY_DIR}" \
            "${REMOTE_MODELS}/rl-agents/${personality}-${RUN_ID}" \
            "RL agent (${personality})"
    done

    save_checkpoint "rl_training" "complete"
    log_success "RL training complete"

    # Upload all RL agents
    upload_to_storage "${RL_OUTPUT_DIR}" "${REMOTE_MODELS}/rl-agents-${RUN_ID}" "all RL agents"
}

# =============================================================================
# EMBEDDINGS GENERATION WORKLOAD
# =============================================================================

run_embeddings_generation() {
    log_step "=========================================="
    log_step "EMBEDDINGS GENERATION WORKLOAD"
    log_step "=========================================="

    if is_stage_complete "embeddings"; then
        log_info "Embeddings already generated, skipping..."
        return 0
    fi

    save_checkpoint "embeddings" "started"

    # Install embedding dependencies
    log_info "Installing embedding dependencies..."
    pip install -q sentence-transformers faiss-cpu numpy tqdm

    EMBEDDINGS_OUTPUT="${EMBEDDINGS_DIR}/${RUN_ID}"
    mkdir -p "${EMBEDDINGS_OUTPUT}"

    # Create embedding generation script inline
    log_info "Generating embeddings for training data..."

    python << 'EMBED_SCRIPT'
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    print("Installing dependencies...")
    os.system(f"{sys.executable} -m pip install -q sentence-transformers faiss-cpu")
    from sentence_transformers import SentenceTransformer
    import faiss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration from environment
RUN_ID = os.environ.get('RUN_ID', 'default')
PROJECT_ROOT = Path(os.environ.get('PROJECT_ROOT', '.'))
EMBEDDINGS_DIR = PROJECT_ROOT / 'data' / 'embeddings' / RUN_ID
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.environ.get('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

def load_training_data():
    """Load all training data from processed directory."""
    data_dir = PROJECT_ROOT / 'data' / 'processed'
    all_texts = []
    all_metadata = []

    # Find all JSONL files
    jsonl_files = list(data_dir.glob('*.jsonl'))
    logger.info(f"Found {len(jsonl_files)} JSONL files")

    for jsonl_file in jsonl_files:
        logger.info(f"Loading {jsonl_file.name}...")
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    item = json.loads(line)

                    # Extract text based on format
                    if 'messages' in item:
                        # ChatML format
                        text_parts = []
                        for msg in item['messages']:
                            content = msg.get('content', '')
                            if content:
                                text_parts.append(content)
                        text = '\n'.join(text_parts)
                    elif 'text' in item:
                        text = item['text']
                    elif 'instruction' in item:
                        text = f"{item.get('instruction', '')} {item.get('output', '')}"
                    else:
                        continue

                    if text and len(text) > 50:  # Skip very short texts
                        all_texts.append(text[:8192])  # Truncate very long texts
                        all_metadata.append({
                            'source_file': jsonl_file.name,
                            'line_num': line_num,
                            'length': len(text)
                        })
                except json.JSONDecodeError:
                    continue

    logger.info(f"Loaded {len(all_texts)} text samples")
    return all_texts, all_metadata

def generate_embeddings(texts, model_name):
    """Generate embeddings using sentence-transformers."""
    logger.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    logger.info("Generating embeddings...")
    batch_size = 32
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.extend(batch_embeddings)

    return np.array(embeddings)

def create_faiss_index(embeddings):
    """Create a FAISS index for fast similarity search."""
    dimension = embeddings.shape[1]

    # Use IVF index for larger datasets
    if len(embeddings) > 10000:
        nlist = min(100, len(embeddings) // 100)
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        index.train(embeddings)
    else:
        index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)
    return index

def main():
    start_time = datetime.now()

    # Load data
    texts, metadata = load_training_data()

    if not texts:
        logger.error("No texts found to embed")
        sys.exit(1)

    # Generate embeddings
    embeddings = generate_embeddings(texts, MODEL_NAME)

    # Create FAISS index
    logger.info("Creating FAISS index...")
    index = create_faiss_index(embeddings)

    # Save embeddings
    logger.info(f"Saving to {EMBEDDINGS_DIR}...")

    np.save(EMBEDDINGS_DIR / 'embeddings.npy', embeddings)
    faiss.write_index(index, str(EMBEDDINGS_DIR / 'index.faiss'))

    with open(EMBEDDINGS_DIR / 'metadata.json', 'w') as f:
        json.dump({
            'model': MODEL_NAME,
            'num_embeddings': len(embeddings),
            'dimension': embeddings.shape[1],
            'metadata': metadata,
            'created_at': datetime.now().isoformat(),
            'processing_time': str(datetime.now() - start_time)
        }, f, indent=2)

    logger.info(f"Embeddings saved!")
    logger.info(f"  - Shape: {embeddings.shape}")
    logger.info(f"  - Index size: {os.path.getsize(EMBEDDINGS_DIR / 'index.faiss') / 1024 / 1024:.2f} MB")
    logger.info(f"  - Processing time: {datetime.now() - start_time}")

if __name__ == '__main__':
    main()
EMBED_SCRIPT

    if [ $? -eq 0 ]; then
        save_checkpoint "embeddings" "complete"
        log_success "Embeddings generation complete"

        # Upload embeddings
        upload_to_storage "${EMBEDDINGS_OUTPUT}" \
            "${REMOTE_EMBEDDINGS}/${RUN_ID}" \
            "embeddings and FAISS index"
    else
        log_error "Embeddings generation failed"
        save_checkpoint "embeddings" "failed"
        return 1
    fi
}

# =============================================================================
# FINAL UPLOAD AND CLEANUP
# =============================================================================

final_upload() {
    log_step "=========================================="
    log_step "FINAL UPLOAD"
    log_step "=========================================="

    # Stop the upload daemon
    stop_upload_daemon

    if [ "${STORAGE_AVAILABLE}" != "true" ]; then
        log_warn "Object Storage not available, skipping final upload"
        return 1
    fi

    # Upload all logs
    upload_to_storage "${LOG_DIR}" "${REMOTE_LOGS}/${RUN_ID}" "final logs"

    # Upload any remaining models
    if [ -d "${CHECKPOINT_DIR}" ]; then
        upload_to_storage "${CHECKPOINT_DIR}" "${REMOTE_MODELS}/${RUN_ID}" "all model checkpoints"
    fi

    # Upload exports
    if [ -d "${EXPORT_DIR}" ]; then
        upload_to_storage "${EXPORT_DIR}" "${REMOTE_MODELS}/exports-${RUN_ID}" "all exports"
    fi

    # Upload embeddings
    if [ -d "${EMBEDDINGS_DIR}" ]; then
        upload_to_storage "${EMBEDDINGS_DIR}" "${REMOTE_EMBEDDINGS}" "all embeddings"
    fi

    # Create and upload snapshot manifest
    MANIFEST="${LOG_DIR}/manifest.json"
    cat > "${MANIFEST}" << EOF
{
    "run_id": "${RUN_ID}",
    "timestamp": "$(date -Iseconds)",
    "workload": "${WORKLOAD}",
    "gpu": "${GPU_NAME}",
    "stages": {
        "environment": "$(get_checkpoint environment)",
        "llm_training": "$(get_checkpoint llm_training)",
        "llm_export": "$(get_checkpoint llm_export)",
        "rl_training": "$(get_checkpoint rl_training)",
        "embeddings": "$(get_checkpoint embeddings)"
    },
    "artifacts": {
        "logs": "s3://${BUCKET_NAME}/${REMOTE_LOGS}/${RUN_ID}/",
        "models": "s3://${BUCKET_NAME}/${REMOTE_MODELS}/${RUN_ID}/",
        "exports": "s3://${BUCKET_NAME}/${REMOTE_MODELS}/exports-${RUN_ID}/",
        "embeddings": "s3://${BUCKET_NAME}/${REMOTE_EMBEDDINGS}/${RUN_ID}/"
    }
}
EOF

    upload_to_storage "${MANIFEST}" "${REMOTE_SNAPSHOTS}/manifest-${RUN_ID}.json" "manifest"

    log_success "Final upload complete"
}

# =============================================================================
# PRINT SUMMARY
# =============================================================================

print_summary() {
    echo ""
    echo "============================================================"
    echo "VULTR CREDIT BURN SUMMARY"
    echo "============================================================"
    echo ""
    echo "Run ID: ${RUN_ID}"
    echo "Workload: ${WORKLOAD}"
    echo "GPU: ${GPU_NAME:-Unknown} (${GPU_MEM:-?}MB)"
    echo ""
    echo "Stage Status:"
    echo "  Environment:   $(get_checkpoint environment)"
    echo "  LLM Training:  $(get_checkpoint llm_training)"
    echo "  LLM Export:    $(get_checkpoint llm_export)"
    echo "  RL Training:   $(get_checkpoint rl_training)"
    echo "  Embeddings:    $(get_checkpoint embeddings)"
    echo ""

    if [ "${STORAGE_AVAILABLE}" = "true" ]; then
        echo "Object Storage: s3://${BUCKET_NAME}/"
        echo ""
        echo "Artifacts uploaded to:"
        echo "  - Logs:       ${REMOTE_LOGS}/${RUN_ID}/"
        echo "  - Models:     ${REMOTE_MODELS}/${RUN_ID}/"
        echo "  - Exports:    ${REMOTE_MODELS}/exports-${RUN_ID}/"
        echo "  - Embeddings: ${REMOTE_EMBEDDINGS}/${RUN_ID}/"
    else
        echo "Object Storage: Not configured"
        echo ""
        echo "Local artifacts:"
        echo "  - Logs:       ${LOG_DIR}/"
        echo "  - Models:     ${CHECKPOINT_DIR}/"
        echo "  - Exports:    ${EXPORT_DIR}/"
        echo "  - Embeddings: ${EMBEDDINGS_DIR}/"
    fi
    echo ""
    echo "============================================================"
}

print_estimates() {
    local workload=$1
    echo ""
    echo "============================================================"
    echo "ESTIMATED RUNTIME AND COST"
    echo "============================================================"
    echo ""
    echo "Workload: ${workload}"
    echo "Estimated Time: ${ESTIMATED_TIMES[$workload]}"
    echo "Estimated Cost: ${ESTIMATED_COSTS[$workload]} (A100 @ \$2.50/hr)"
    echo ""

    case $workload in
        llm)
            echo "Tasks:"
            echo "  1. LLM fine-tuning with V15 dataset (~3.5-4.5 hours)"
            echo "  2. Export to GGUF format (~15 minutes)"
            ;;
        rl)
            echo "Tasks:"
            echo "  1. Train aggressive agent (500K steps, ~1.5 hours)"
            echo "  2. Train medic agent (500K steps, ~1.5 hours)"
            echo "  3. Train defender agent (500K steps, ~1.5 hours)"
            ;;
        embeddings)
            echo "Tasks:"
            echo "  1. Load all training data"
            echo "  2. Generate embeddings with sentence-transformers"
            echo "  3. Create FAISS index"
            ;;
        all)
            echo "Tasks:"
            echo "  1. LLM fine-tuning + export (~3.5 hours)"
            echo "  2. RL agent training x3 (~4.5 hours)"
            echo "  3. Embeddings generation (~0.5 hours)"
            ;;
    esac
    echo ""
    echo "============================================================"
}

# =============================================================================
# MAIN
# =============================================================================

main() {
    WORKLOAD=${1:-all}

    # Validate workload
    case $WORKLOAD in
        llm|rl|embeddings|all)
            ;;
        *)
            echo "Usage: $0 {llm|rl|embeddings|all}"
            echo ""
            echo "Workloads:"
            echo "  llm        - LLM fine-tuning with V15 dataset, export to GGUF"
            echo "  rl         - Train PPO agents for aggressive, medic, defender"
            echo "  embeddings - Generate embeddings and FAISS index for training data"
            echo "  all        - Run all workloads in sequence"
            exit 1
            ;;
    esac

    # Print estimates
    print_estimates "${WORKLOAD}"

    echo "Starting in 5 seconds... (Ctrl+C to cancel)"
    sleep 5

    echo ""
    log_step "=========================================="
    log_step "VULTR CREDIT BURN - ${WORKLOAD^^}"
    log_step "Run ID: ${RUN_ID}"
    log_step "=========================================="

    # Export environment for Python scripts
    export RUN_ID
    export PROJECT_ROOT
    export EMBEDDING_MODEL

    # Setup
    setup_environment
    setup_object_storage || true  # Continue even if storage fails

    # Start background upload daemon
    if [ "${STORAGE_AVAILABLE}" = "true" ]; then
        start_upload_daemon
    fi

    # Run workloads
    case $WORKLOAD in
        llm)
            run_llm_training
            ;;
        rl)
            run_rl_training
            ;;
        embeddings)
            run_embeddings_generation
            ;;
        all)
            run_llm_training
            run_rl_training
            run_embeddings_generation
            ;;
    esac

    # Final upload
    final_upload

    # Print summary
    print_summary

    log_success "Vultr credit burn complete!"
}

# Cleanup on exit
cleanup() {
    stop_upload_daemon
    log_info "Cleanup complete"
}
trap cleanup EXIT

# Run main
main "$@"
