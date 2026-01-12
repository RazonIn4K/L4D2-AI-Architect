#!/bin/bash
# =============================================================================
# L4D2 Local Model Training on Vultr GPU
# =============================================================================
# Train a local Mistral-7B model on Vultr A100/A40 GPU instances.
# This gives you a free-to-run local model after training.
#
# Usage:
#   ./run_vultr_training.sh                    # Default: Mistral-7B, 3 epochs
#   ./run_vultr_training.sh --epochs 5         # Train for 5 epochs
#   ./run_vultr_training.sh --export-only      # Just export existing model
#
# Prerequisites:
#   - Vultr A100 (40GB) or A40 (48GB) instance
#   - Ubuntu 22.04 with NVIDIA drivers
#   - This repo cloned to the instance
# =============================================================================

set -e

EPOCHS=${1:-3}
BATCH_SIZE=${2:-4}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo "L4D2 Local Model Training"
echo "============================================================"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Timestamp: $TIMESTAMP"
echo "============================================================"

# Check for GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Are you on a GPU instance?"
    exit 1
fi

echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# Setup virtual environment
if [ ! -d "venv" ]; then
    echo ">>> Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies
echo ">>> Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install -q flash-attn --no-build-isolation 2>/dev/null || echo "Flash attention not available"

# Check for training data (prefer V15, fallback to V14, V13, V12, V11, V10, V9, V8)
TRAINING_DATA=""
if [ -f "data/processed/l4d2_train_v15.jsonl" ]; then
    TRAINING_DATA="data/processed/l4d2_train_v15.jsonl"
    echo ">>> Using V15 dataset (2773 examples - enhanced augmented dataset)"
elif [ -f "data/processed/l4d2_train_v14.jsonl" ]; then
    TRAINING_DATA="data/processed/l4d2_train_v14.jsonl"
    echo ">>> Using V14 dataset (2576 examples - augmented dataset)"
elif [ -f "data/processed/l4d2_train_v13.jsonl" ]; then
    TRAINING_DATA="data/processed/l4d2_train_v13.jsonl"
    echo ">>> Using V13 dataset (1010 examples - comprehensive coverage)"
elif [ -f "data/processed/l4d2_train_v12.jsonl" ]; then
    TRAINING_DATA="data/processed/l4d2_train_v12.jsonl"
    echo ">>> Using V12 dataset (fallback)"
elif [ -f "data/processed/l4d2_train_v11.jsonl" ]; then
    TRAINING_DATA="data/processed/l4d2_train_v11.jsonl"
    echo ">>> Using V11 dataset (fallback)"
elif [ -f "data/processed/l4d2_train_v10.jsonl" ]; then
    TRAINING_DATA="data/processed/l4d2_train_v10.jsonl"
    echo ">>> Using V10 dataset (fallback)"
elif [ -f "data/processed/l4d2_train_v9.jsonl" ]; then
    TRAINING_DATA="data/processed/l4d2_train_v9.jsonl"
    echo ">>> Using V9 dataset (fallback)"
elif [ -f "data/processed/l4d2_train_v8.jsonl" ]; then
    TRAINING_DATA="data/processed/l4d2_train_v8.jsonl"
    echo ">>> Using V8 dataset (fallback)"
else
    echo "ERROR: No training data found."
    echo "Expected: data/processed/l4d2_train_v15.jsonl or earlier versions"
    exit 1
fi

echo ""
echo ">>> Training data stats:"
wc -l "$TRAINING_DATA"
echo ""

# Determine config based on GPU
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$GPU_MEM" -gt 45000 ]; then
    CONFIG="configs/unsloth_config_a100.yaml"
    echo "Using A100 config (${GPU_MEM}MB VRAM)"
elif [ "$GPU_MEM" -gt 90000 ]; then
    CONFIG="configs/unsloth_config_gh200.yaml"
    echo "Using GH200 config (${GPU_MEM}MB VRAM)"
else
    CONFIG="configs/unsloth_config.yaml"
    echo "Using default config (${GPU_MEM}MB VRAM)"
fi

# Start training in tmux if available
if command -v tmux &> /dev/null; then
    echo ""
    echo ">>> Starting training in tmux session 'training'..."
    echo ">>> Use 'tmux attach -t training' to monitor"
    echo ""

    tmux new-session -d -s training "
        source venv/bin/activate && \
        python scripts/training/train_unsloth.py \
            --config $CONFIG \
            --epochs $EPOCHS \
            --output model_adapters/mistral-l4d2-v1-$TIMESTAMP && \
        echo '' && \
        echo '>>> Training complete! Exporting model...' && \
        python scripts/training/export_model.py \
            --input model_adapters/mistral-l4d2-v1-$TIMESTAMP \
            --format gguf \
            --quantize q4_k_m \
            --output exports/l4d2-mistral-$TIMESTAMP.gguf && \
        echo '' && \
        echo '>>> Export complete! Model saved to:' && \
        ls -la exports/l4d2-mistral-$TIMESTAMP.gguf
    "

    echo "Training started in background tmux session."
    echo ""
    echo "Commands:"
    echo "  tmux attach -t training     # Watch training progress"
    echo "  tmux kill-session -t training  # Stop training"
    echo ""
    echo "After training, download the model:"
    echo "  scp root@\$VULTR_IP:/root/L4D2-AI-Architect/exports/*.gguf ./exports/"
else
    echo ""
    echo ">>> Starting training (no tmux, running in foreground)..."
    echo ""

    python scripts/training/train_unsloth.py \
        --config $CONFIG \
        --epochs $EPOCHS \
        --output model_adapters/mistral-l4d2-v1-$TIMESTAMP

    echo ""
    echo ">>> Training complete! Exporting model..."

    python scripts/training/export_model.py \
        --input model_adapters/mistral-l4d2-v1-$TIMESTAMP \
        --format gguf \
        --quantize q4_k_m \
        --output exports/l4d2-mistral-$TIMESTAMP.gguf

    echo ""
    echo ">>> Export complete!"
    ls -la exports/l4d2-mistral-$TIMESTAMP.gguf
fi

echo ""
echo "============================================================"
echo "Vultr Training Script Complete"
echo "============================================================"
echo ""
echo "Estimated training time: ~2-3 hours on A100"
echo "Estimated cost: ~\$6-8 (A100 @ \$2.50/hr)"
echo ""
echo "After training completes:"
echo "1. Download the GGUF model"
echo "2. Run with Ollama: ollama create l4d2-mistral -f Modelfile"
echo "3. Use locally: ollama run l4d2-mistral"
