#!/bin/bash
# =============================================================================
# URGENT CREDIT BURN - Enhanced Training Script v2.0
# =============================================================================
# Maximized training script with:
#   - Preflight checks
#   - Automatic checkpoint detection and resume
#   - Progress logging with timestamps
#   - Estimated completion time
#   - Graceful error handling
#
# Usage:
#   ./run_credit_burn.sh fast      # ~1 hour, 1 epoch
#   ./run_credit_burn.sh full      # ~4 hours, 3 epochs (RECOMMENDED)
#   ./run_credit_burn.sh max       # ~8 hours, LLM + RL
#   ./run_credit_burn.sh rl        # ~4 hours, RL only
#   ./run_credit_burn.sh --status  # Show current status
#   ./run_credit_burn.sh --resume  # Auto-detect and resume latest checkpoint
# =============================================================================

# Exit on error in main script, but allow controlled error handling
set -o pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Log file for progress tracking
LOG_DIR="$SCRIPT_DIR/data/training_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/credit_burn_$(date '+%Y%m%d_%H%M%S').log"
PROGRESS_FILE="$LOG_DIR/.credit_burn_progress"

# Timing tracking
SCRIPT_START_TIME=$(date +%s)

# =============================================================================
# Logging Functions
# =============================================================================

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo -e "${GREEN}${msg}${NC}"
    echo "$msg" >> "$LOG_FILE"
}

warn() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1"
    echo -e "${YELLOW}${msg}${NC}"
    echo "$msg" >> "$LOG_FILE"
}

error() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1"
    echo -e "${RED}${msg}${NC}"
    echo "$msg" >> "$LOG_FILE"
}

step() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [STEP] $1"
    echo -e "${CYAN}${BOLD}${msg}${NC}"
    echo "$msg" >> "$LOG_FILE"
}

info() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1"
    echo -e "${BLUE}${msg}${NC}"
    echo "$msg" >> "$LOG_FILE"
}

# =============================================================================
# Progress and Timing Functions
# =============================================================================

save_progress() {
    local stage="$1"
    local status="$2"
    echo "STAGE=$stage" > "$PROGRESS_FILE"
    echo "STATUS=$status" >> "$PROGRESS_FILE"
    echo "TIMESTAMP=$(date +%s)" >> "$PROGRESS_FILE"
    echo "MODE=$MODE" >> "$PROGRESS_FILE"
}

load_progress() {
    if [ -f "$PROGRESS_FILE" ]; then
        source "$PROGRESS_FILE"
        echo "$STAGE:$STATUS"
    else
        echo "none:none"
    fi
}

format_duration() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(( (seconds % 3600) / 60 ))
    local secs=$((seconds % 60))

    if [ $hours -gt 0 ]; then
        printf "%dh %dm %ds" $hours $minutes $secs
    elif [ $minutes -gt 0 ]; then
        printf "%dm %ds" $minutes $secs
    else
        printf "%ds" $secs
    fi
}

estimate_completion() {
    local mode="$1"
    local elapsed=$(($(date +%s) - SCRIPT_START_TIME))
    local total_estimate

    case "$mode" in
        fast) total_estimate=3600 ;;    # 1 hour
        full) total_estimate=14400 ;;   # 4 hours
        max)  total_estimate=28800 ;;   # 8 hours
        rl)   total_estimate=14400 ;;   # 4 hours
        *)    total_estimate=14400 ;;
    esac

    local remaining=$((total_estimate - elapsed))
    if [ $remaining -lt 0 ]; then remaining=0; fi

    local eta=$(date -d "+${remaining} seconds" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || \
                date -v+${remaining}S '+%Y-%m-%d %H:%M:%S' 2>/dev/null || \
                echo "calculating...")

    echo "Elapsed: $(format_duration $elapsed) | Est. remaining: $(format_duration $remaining) | ETA: $eta"
}

show_status() {
    echo ""
    echo -e "${BOLD}=== Credit Burn Status ===${NC}"
    echo ""

    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${CYAN}GPU Status:${NC}"
        nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
        echo ""
    fi

    # Check progress
    if [ -f "$PROGRESS_FILE" ]; then
        source "$PROGRESS_FILE"
        echo -e "${CYAN}Training Progress:${NC}"
        echo "  Mode: $MODE"
        echo "  Stage: $STAGE"
        echo "  Status: $STATUS"
        echo "  Started: $(date -d @$TIMESTAMP 2>/dev/null || date -r $TIMESTAMP 2>/dev/null || echo 'unknown')"
        echo ""
    fi

    # Check for checkpoints
    echo -e "${CYAN}Available Checkpoints:${NC}"
    find model_adapters -name "checkpoint-*" -type d 2>/dev/null | head -10 || echo "  No checkpoints found"
    echo ""

    # Check for completed models
    echo -e "${CYAN}Completed Models:${NC}"
    find model_adapters -name "final" -type d 2>/dev/null | head -10 || echo "  No completed models"
    echo ""

    # Check training log
    if [ -f "$LOG_FILE" ]; then
        echo -e "${CYAN}Recent Log (last 10 lines):${NC}"
        tail -10 "$LOG_FILE"
    fi
}

# =============================================================================
# Preflight Checks
# =============================================================================

run_preflight() {
    step "Running preflight checks..."

    local errors=0

    # Check for preflight script
    if [ -f "scripts/utils/vultr_preflight.py" ]; then
        info "Running vultr_preflight.py..."
        if python scripts/utils/vultr_preflight.py; then
            log "Preflight check passed"
        else
            warn "Preflight check returned warnings (continuing anyway)"
        fi
    fi

    # GPU Check
    info "Checking GPU..."
    if ! command -v nvidia-smi &> /dev/null; then
        error "nvidia-smi not found. Are you on a GPU instance?"
        ((errors++))
    else
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
        GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | head -1)
        log "GPU: $GPU_NAME ($GPU_MEM) - Temperature: ${GPU_TEMP}C"

        # Temperature warning
        if [ -n "$GPU_TEMP" ] && [ "$GPU_TEMP" -gt 80 ]; then
            warn "GPU temperature is high (${GPU_TEMP}C). Consider cooling before starting."
        fi
    fi

    # Disk space check
    info "Checking disk space..."
    local free_space=$(df -BG . 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')
    if [ -n "$free_space" ] && [ "$free_space" -lt 20 ]; then
        warn "Low disk space: ${free_space}GB free. Recommend at least 20GB."
    else
        log "Disk space OK: ${free_space}GB free"
    fi

    # Memory check
    info "Checking system memory..."
    local total_mem=$(free -g 2>/dev/null | grep Mem | awk '{print $2}')
    if [ -n "$total_mem" ]; then
        log "System memory: ${total_mem}GB"
    fi

    # Check training data exists
    info "Checking training data..."
    if [ -f "data/processed/l4d2_train_v15.jsonl" ]; then
        local sample_count=$(wc -l < "data/processed/l4d2_train_v15.jsonl")
        log "Training data found: $sample_count samples"
    else
        warn "Training data not found at data/processed/l4d2_train_v15.jsonl"
        if [ -f "data/processed/combined_train.jsonl" ]; then
            info "Found alternative: combined_train.jsonl"
        fi
    fi

    # Check config files
    info "Checking config files..."
    for config in configs/unsloth_config_v15.yaml configs/unsloth_config_v15_fast.yaml; do
        if [ -f "$config" ]; then
            log "Found: $config"
        else
            warn "Missing config: $config"
        fi
    done

    # Python and dependencies
    info "Checking Python environment..."
    if python -c "import torch; print(f'PyTorch {torch.__version__} CUDA: {torch.cuda.is_available()}')" 2>/dev/null; then
        log "PyTorch with CUDA available"
    else
        error "PyTorch or CUDA not properly configured"
        ((errors++))
    fi

    if [ $errors -gt 0 ]; then
        error "Preflight check failed with $errors errors"
        return 1
    fi

    log "All preflight checks passed"
    return 0
}

# =============================================================================
# Checkpoint Detection
# =============================================================================

find_latest_checkpoint() {
    local model_dir="$1"
    local latest=""
    local latest_step=0

    # Find all checkpoint directories
    for checkpoint in "$model_dir"/checkpoint-*; do
        if [ -d "$checkpoint" ]; then
            # Extract step number
            local step=$(basename "$checkpoint" | sed 's/checkpoint-//')
            if [ "$step" -gt "$latest_step" ] 2>/dev/null; then
                latest_step=$step
                latest="$checkpoint"
            fi
        fi
    done

    echo "$latest"
}

detect_resume_checkpoint() {
    local config="$1"
    local model_dir=""

    # Determine model directory from config
    case "$config" in
        *v15_fast*) model_dir="model_adapters/l4d2-mistral-v15-fast-lora" ;;
        *v15*) model_dir="model_adapters/l4d2-mistral-v15-lora" ;;
        *) model_dir="model_adapters/l4d2-code-lora" ;;
    esac

    # Check if final model already exists
    if [ -d "$model_dir/final" ]; then
        echo "COMPLETE:$model_dir/final"
        return
    fi

    # Find latest checkpoint
    local checkpoint=$(find_latest_checkpoint "$model_dir")
    if [ -n "$checkpoint" ]; then
        echo "RESUME:$checkpoint"
        return
    fi

    echo "NEW:"
}

# =============================================================================
# Training Functions with Error Handling
# =============================================================================

run_llm_training() {
    local config="$1"
    local batch_size="$2"
    local resume_flag="$3"
    local description="$4"

    step "$description"
    save_progress "llm_training" "running"

    local start_time=$(date +%s)
    local cmd="python scripts/training/train_unsloth.py --config $config --batch-size $batch_size"

    # Check for resume
    if [ "$resume_flag" = "auto" ]; then
        local checkpoint_status=$(detect_resume_checkpoint "$config")
        local status_type="${checkpoint_status%%:*}"
        local checkpoint_path="${checkpoint_status#*:}"

        case "$status_type" in
            COMPLETE)
                log "Training already complete: $checkpoint_path"
                save_progress "llm_training" "complete"
                return 0
                ;;
            RESUME)
                log "Resuming from checkpoint: $checkpoint_path"
                cmd="$cmd --resume $checkpoint_path"
                ;;
            NEW)
                log "Starting fresh training"
                ;;
        esac
    elif [ -n "$resume_flag" ] && [ "$resume_flag" != "auto" ]; then
        cmd="$cmd --resume $resume_flag"
    fi

    info "Running: $cmd"
    info "$(estimate_completion $MODE)"

    if eval "$cmd"; then
        local duration=$(($(date +%s) - start_time))
        log "LLM training completed in $(format_duration $duration)"
        save_progress "llm_training" "complete"
        return 0
    else
        local exit_code=$?
        error "LLM training failed with exit code $exit_code"
        save_progress "llm_training" "failed:$exit_code"
        return $exit_code
    fi
}

run_rl_training() {
    local personality="$1"
    local timesteps="$2"

    info "Training $personality personality ($timesteps timesteps)..."
    save_progress "rl_training_$personality" "running"

    local start_time=$(date +%s)

    if python scripts/rl_training/train_ppo.py \
        --timesteps "$timesteps" \
        --personality "$personality" \
        --save-freq 50000; then

        local duration=$(($(date +%s) - start_time))
        log "RL training ($personality) completed in $(format_duration $duration)"
        save_progress "rl_training_$personality" "complete"
        return 0
    else
        local exit_code=$?
        warn "RL training for $personality failed (exit code $exit_code), continuing..."
        save_progress "rl_training_$personality" "failed:$exit_code"
        return $exit_code
    fi
}

# =============================================================================
# Virtual Environment Setup
# =============================================================================

setup_environment() {
    step "Setting up environment..."

    # Activate virtual environment
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        log "Activated virtual environment"
    elif [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
        log "Activated .venv virtual environment"
    else
        warn "No virtual environment found, using system Python"
    fi

    # Set CUDA environment variables for stability
    export CUDA_LAUNCH_BLOCKING=0
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

    log "Environment setup complete"
}

# =============================================================================
# GPU Detection and Configuration
# =============================================================================

detect_gpu_settings() {
    if ! command -v nvidia-smi &> /dev/null; then
        error "nvidia-smi not found. Are you on a GPU instance?"
        exit 1
    fi

    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)

    # Determine batch size based on GPU
    if [[ "$GPU_NAME" == *"A100"* ]]; then
        BATCH_SIZE=8
        CONFIG_SUFFIX=""
        log "Using A100-optimized settings (batch_size=8)"
    elif [[ "$GPU_NAME" == *"GH200"* ]] || [[ "$GPU_NAME" == *"H100"* ]]; then
        BATCH_SIZE=16
        CONFIG_SUFFIX="_gh200"
        log "Using GH200/H100-optimized settings (batch_size=16)"
    elif [[ "$GPU_NAME" == *"L40"* ]] || [[ "$GPU_NAME" == *"A40"* ]]; then
        BATCH_SIZE=6
        CONFIG_SUFFIX=""
        log "Using L40/A40 settings (batch_size=6)"
    else
        BATCH_SIZE=4
        CONFIG_SUFFIX=""
        log "Using default settings (batch_size=4)"
    fi

    log "GPU: $GPU_NAME ($GPU_MEM)"
}

# =============================================================================
# Mode Execution
# =============================================================================

run_fast_mode() {
    step "FAST MODE: 1 epoch V15 training (~1 hour)"

    local config="configs/unsloth_config_v15_fast.yaml"
    if [ ! -f "$config" ]; then
        warn "Fast config not found, using standard config"
        config="configs/unsloth_config_v15.yaml"
    fi

    run_llm_training "$config" "$BATCH_SIZE" "auto" "Starting V15 fast training..."
}

run_full_mode() {
    step "FULL MODE: 3 epoch V15 training (~4 hours)"

    run_llm_training "configs/unsloth_config_v15.yaml" "$BATCH_SIZE" "auto" "Starting V15 full training..."
}

run_max_mode() {
    step "MAXIMUM MODE: LLM + RL training (~8 hours)"

    local llm_failed=0
    local rl_failed=0

    # Part 1: LLM Training
    log "Part 1/2: Starting V15 LLM training..."
    info "$(estimate_completion max)"

    if ! run_llm_training "configs/unsloth_config_v15.yaml" "$BATCH_SIZE" "auto" "V15 LLM Training"; then
        llm_failed=1
        warn "LLM training failed, continuing with RL training..."
    fi

    # Part 2: RL Training
    log "Part 2/2: Starting RL agent training..."
    local rl_failures=0

    for personality in aggressive medic defender; do
        if ! run_rl_training "$personality" 300000; then
            ((rl_failures++))
        fi
        info "$(estimate_completion max)"
    done

    if [ $rl_failures -gt 0 ]; then
        warn "$rl_failures RL training tasks failed"
    fi

    # Summary
    echo ""
    echo -e "${BOLD}=== MAX MODE SUMMARY ===${NC}"
    if [ $llm_failed -eq 0 ]; then
        log "LLM Training: SUCCESS"
    else
        error "LLM Training: FAILED"
    fi
    log "RL Training: $((3 - rl_failures))/3 personalities completed"
}

run_rl_mode() {
    step "RL ONLY MODE: Training bot agents (~4 hours)"

    local failures=0

    for personality in aggressive medic defender; do
        if ! run_rl_training "$personality" 500000; then
            ((failures++))
        fi
        info "$(estimate_completion rl)"
    done

    if [ $failures -gt 0 ]; then
        warn "$failures RL training tasks failed"
    fi

    log "RL Training: $((3 - failures))/3 personalities completed"
}

# =============================================================================
# Main Script
# =============================================================================

MODE="${1:-full}"

# Handle special flags
case "$MODE" in
    --status|-s)
        show_status
        exit 0
        ;;
    --resume|-r)
        MODE="full"
        FORCE_RESUME=1
        ;;
    --help|-h)
        echo ""
        echo -e "${BOLD}L4D2-AI-Architect Credit Burn Script v2.0${NC}"
        echo ""
        echo "Usage: $0 {fast|full|max|rl|--status|--resume}"
        echo ""
        echo "Modes:"
        echo "  fast      - 1 epoch, ~1 hour, quick validation"
        echo "  full      - 3 epochs, ~4 hours (RECOMMENDED)"
        echo "  max       - LLM + RL, ~8 hours, maximum value"
        echo "  rl        - RL only, ~4 hours, bot agents"
        echo ""
        echo "Options:"
        echo "  --status  - Show current training status"
        echo "  --resume  - Auto-detect and resume interrupted training"
        echo "  --help    - Show this help"
        echo ""
        echo "Features:"
        echo "  - Automatic checkpoint detection and resume"
        echo "  - Progress logging with timestamps"
        echo "  - Estimated completion time"
        echo "  - Graceful error handling"
        echo ""
        echo "Log file: $LOG_FILE"
        echo ""
        exit 0
        ;;
esac

# Start logging
echo ""
echo -e "${BOLD}============================================${NC}"
echo -e "${BOLD}  L4D2-AI-Architect Credit Burn v2.0${NC}"
echo -e "${BOLD}============================================${NC}"
echo ""
log "Starting credit burn in $MODE mode"
log "Log file: $LOG_FILE"
echo ""

# Setup and preflight
setup_environment

if ! run_preflight; then
    error "Preflight checks failed. Fix issues and retry."
    exit 1
fi

echo ""
detect_gpu_settings
echo ""

# Execute selected mode
case "$MODE" in
    fast)
        run_fast_mode
        ;;
    full)
        run_full_mode
        ;;
    max)
        run_max_mode
        ;;
    rl)
        run_rl_mode
        ;;
    *)
        error "Unknown mode: $MODE"
        echo "Usage: $0 {fast|full|max|rl|--status|--resume}"
        echo ""
        echo "  fast  - 1 epoch, ~1 hour, quick validation"
        echo "  full  - 3 epochs, ~4 hours (RECOMMENDED)"
        echo "  max   - LLM + RL, ~8 hours, maximum value"
        echo "  rl    - RL only, ~4 hours, bot agents"
        exit 1
        ;;
esac

# Final summary
echo ""
echo -e "${BOLD}============================================${NC}"
echo -e "${GREEN}${BOLD}  Training Complete!${NC}"
echo -e "${BOLD}============================================${NC}"
echo ""

TOTAL_DURATION=$(($(date +%s) - SCRIPT_START_TIME))
log "Total runtime: $(format_duration $TOTAL_DURATION)"
log "Results saved to: model_adapters/"
log "Full log: $LOG_FILE"

echo ""
echo "=== TRAINING RESULTS ==="
ls -la model_adapters/ 2>/dev/null || echo "No model adapters directory"
echo ""

# Check for exports
if [ -d "exports" ]; then
    echo "=== EXPORTS ==="
    find exports -name "*.gguf" -type f 2>/dev/null | head -5 || echo "No GGUF exports yet"
    echo ""
fi

log "To download results: ./deploy_to_vultr.sh YOUR_IP --download"
save_progress "complete" "success"
