#!/bin/bash
# =============================================================================
# Training Monitor - Real-time training status dashboard
# =============================================================================
#
# Run this in a separate terminal or tmux pane to monitor training progress.
#
# Usage:
#   ./training_monitor.sh              # Full dashboard
#   ./training_monitor.sh --compact    # Compact single-line status
#   ./training_monitor.sh --once       # One-time check (no loop)
#
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$PROJECT_ROOT/data/training_logs"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

get_gpu_info() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1
    else
        echo "N/A,N/A,N/A,N/A"
    fi
}

get_training_status() {
    # Check if training process is running
    if pgrep -f "train_unsloth.py\|train_all_models.py" > /dev/null; then
        echo "RUNNING"
    else
        # Check for recent checkpoints
        latest_ckpt=$(find "$PROJECT_ROOT/model_adapters" -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
        if [ -n "$latest_ckpt" ]; then
            echo "CHECKPOINTED"
        else
            echo "IDLE"
        fi
    fi
}

get_latest_checkpoint() {
    find "$PROJECT_ROOT/model_adapters" -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1 | xargs basename 2>/dev/null || echo "None"
}

get_training_progress() {
    # Try to find progress from training log
    latest_log=$(find "$LOG_DIR" -name "*.log" -type f 2>/dev/null | sort | tail -1)
    if [ -f "$latest_log" ]; then
        # Look for progress indicators
        grep -oE '\d+/\d+.*\d+\.\d+%' "$latest_log" 2>/dev/null | tail -1 || echo "..."
    else
        echo "No log found"
    fi
}

get_elapsed_time() {
    # Find when training started
    training_pid=$(pgrep -f "train_unsloth.py" | head -1)
    if [ -n "$training_pid" ]; then
        ps -o etime= -p "$training_pid" 2>/dev/null | tr -d ' '
    else
        echo "N/A"
    fi
}

get_model_adapters() {
    ls -d "$PROJECT_ROOT/model_adapters"/*/ 2>/dev/null | while read dir; do
        basename "$dir"
        if [ -d "$dir/final" ]; then
            echo "  └─ final (COMPLETE)"
        else
            latest=$(ls -d "$dir/checkpoint-"* 2>/dev/null | sort -V | tail -1 | xargs basename 2>/dev/null)
            if [ -n "$latest" ]; then
                echo "  └─ $latest"
            fi
        fi
    done
}

compact_status() {
    status=$(get_training_status)
    gpu_info=$(get_gpu_info)
    IFS=',' read -r gpu_util mem_used mem_total temp <<< "$gpu_info"

    if [ "$status" = "RUNNING" ]; then
        elapsed=$(get_elapsed_time)
        echo -e "${GREEN}[TRAINING]${NC} GPU: ${gpu_util}% | VRAM: ${mem_used}/${mem_total}MB | Temp: ${temp}C | Elapsed: ${elapsed}"
    elif [ "$status" = "CHECKPOINTED" ]; then
        ckpt=$(get_latest_checkpoint)
        echo -e "${YELLOW}[PAUSED]${NC} Last checkpoint: $ckpt"
    else
        echo -e "${BLUE}[IDLE]${NC} No training in progress"
    fi
}

full_dashboard() {
    clear
    echo -e "${BOLD}============================================================${NC}"
    echo -e "${BOLD}L4D2-AI-ARCHITECT TRAINING MONITOR${NC}"
    echo -e "${BOLD}============================================================${NC}"
    echo ""

    # Training Status
    status=$(get_training_status)
    if [ "$status" = "RUNNING" ]; then
        echo -e "Status: ${GREEN}${BOLD}TRAINING${NC}"
        echo -e "Elapsed: $(get_elapsed_time)"
    elif [ "$status" = "CHECKPOINTED" ]; then
        echo -e "Status: ${YELLOW}${BOLD}PAUSED/CHECKPOINTED${NC}"
    else
        echo -e "Status: ${BLUE}IDLE${NC}"
    fi
    echo ""

    # GPU Info
    echo -e "${CYAN}GPU Status:${NC}"
    gpu_info=$(get_gpu_info)
    IFS=',' read -r gpu_util mem_used mem_total temp <<< "$gpu_info"
    if [ "$gpu_util" != "N/A" ]; then
        echo -e "  Utilization: ${gpu_util}%"
        echo -e "  Memory: ${mem_used}MB / ${mem_total}MB ($(echo "scale=1; $mem_used * 100 / $mem_total" | bc)%)"
        echo -e "  Temperature: ${temp}C"
    else
        echo "  GPU info not available"
    fi
    echo ""

    # Model Adapters
    echo -e "${CYAN}Model Adapters:${NC}"
    adapters=$(get_model_adapters)
    if [ -n "$adapters" ]; then
        echo "$adapters" | while read line; do
            echo "  $line"
        done
    else
        echo "  None found"
    fi
    echo ""

    # Training Logs
    echo -e "${CYAN}Recent Training Output:${NC}"
    if [ "$status" = "RUNNING" ]; then
        # Get last few lines of training output
        latest_log=$(find "$LOG_DIR" -name "*.log" -type f 2>/dev/null | sort | tail -1)
        if [ -f "$latest_log" ]; then
            tail -5 "$latest_log" 2>/dev/null | while read line; do
                echo "  $line"
            done
        else
            # Try to get from tmux
            tmux capture-pane -p -t training 2>/dev/null | tail -5 | while read line; do
                echo "  $line"
            done
        fi
    else
        echo "  No active training"
    fi
    echo ""

    echo -e "${BOLD}============================================================${NC}"
    echo -e "Press Ctrl+C to exit | Refreshing every 10s"
}

main() {
    case "${1:-}" in
        --compact|-c)
            while true; do
                compact_status
                sleep 5
            done
            ;;
        --once|-1)
            full_dashboard
            ;;
        *)
            while true; do
                full_dashboard
                sleep 10
            done
            ;;
    esac
}

main "$@"
