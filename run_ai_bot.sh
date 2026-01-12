#!/bin/bash
# L4D2 AI Bot Controller
# Usage: ./run_ai_bot.sh [personality] [mode] [host] [port]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
source activate.sh

# Default values
PERSONALITY="${1:-aggressive}"
MODE="${2:-demo}"
HOST="${3:-127.0.0.1}"
PORT="${4:-27050}"
ENV="${5:-mock}"

# Available personalities
PERSONALITIES="balanced aggressive medic speedrunner defender"

# Find the best model for the personality
find_model() {
    local p=$1
    local model_path=""

    # Check for retrained models first (longer training) - use ls to expand glob
    model_path=$(ls -td model_adapters/rl_agents/ppo_${p}_*/final_model.zip 2>/dev/null | head -1)
    if [ -n "$model_path" ]; then
        echo "${model_path%.zip}"
        return
    fi

    # Then check batch training
    model_path=$(ls -td model_adapters/rl_agents/all_personalities_*/${p}/final_model.zip 2>/dev/null | head -1)
    if [ -n "$model_path" ]; then
        echo "${model_path%.zip}"
        return
    fi

    echo ""
}

MODEL=$(find_model "$PERSONALITY")

if [ -z "$MODEL" ]; then
    echo "ERROR: No trained model found for personality: $PERSONALITY"
    echo "Available personalities: $PERSONALITIES"
    echo ""
    echo "Train a model first with:"
    echo "  python scripts/rl_training/train_ppo.py --timesteps 500000 --personality $PERSONALITY"
    exit 1
fi

echo "=========================================="
echo "L4D2 AI Bot Controller"
echo "=========================================="
echo "Personality: $PERSONALITY"
echo "Mode: $MODE"
echo "Model: $MODEL"
echo "Environment: $ENV"
if [ "$ENV" = "mnemosyne" ]; then
    echo "Game Server: $HOST:$PORT"
fi
echo "=========================================="
echo ""

# Run the bot
python scripts/rl_training/train_ppo.py \
    --mode "$MODE" \
    --model "$MODEL" \
    --personality "$PERSONALITY" \
    --env "$ENV" \
    --host "$HOST" \
    --port "$PORT"
