#!/bin/bash
# L4D2 AI Director Controller
# Usage: ./run_ai_director.sh [personality] [mode]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
source activate.sh

# Default values
PERSONALITY="${1:-standard}"
MODE="${2:-demo}"

# Available personalities
PERSONALITIES="standard relaxed intense nightmare"

# Find the best model for the personality
find_model() {
    local p=$1
    local model_dir=$(ls -td model_adapters/director_agents/director_${p}_* 2>/dev/null | head -1)

    if [ -n "$model_dir" ] && [ -f "${model_dir}/final_model.zip" ]; then
        echo "${model_dir}/final_model.zip"
    else
        echo ""
    fi
}

MODEL=$(find_model "$PERSONALITY")

if [ -z "$MODEL" ]; then
    echo "ERROR: No trained model found for Director personality: $PERSONALITY"
    echo "Available personalities: $PERSONALITIES"
    echo ""
    echo "Train a model first with:"
    echo "  python -m scripts.director.train_director_rl --personality $PERSONALITY --timesteps 500000"
    exit 1
fi

echo "=========================================="
echo "L4D2 AI Director Controller"
echo "=========================================="
echo "Personality: $PERSONALITY"
echo "Mode: $MODE"
echo "Model: $MODEL"
echo "=========================================="
echo ""

if [ "$MODE" = "demo" ]; then
    # Run the director demo
    python scripts/director/test_director.py --demo
else
    # For eval mode, show model info
    python -c "
from stable_baselines3 import PPO
import json

model = PPO.load('$MODEL')
print(f'Model loaded successfully!')
print(f'Policy: {model.policy.__class__.__name__}')
print(f'Observation space: {model.observation_space}')
print(f'Action space: {model.action_space}')

# Load training info if available
import os
info_path = os.path.dirname('$MODEL') + '/training_info.json'
if os.path.exists(info_path):
    with open(info_path) as f:
        info = json.load(f)
    print(f'Personality: {info.get(\"personality\", \"unknown\")}')
    print(f'Training timesteps: {info.get(\"total_timesteps\", \"unknown\"):,}')
    print(f'Final reward: {info.get(\"final_mean_reward\", 0):.2f}')
"
fi
