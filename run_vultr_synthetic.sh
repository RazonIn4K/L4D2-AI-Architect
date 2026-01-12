#!/bin/bash
# =============================================================================
# L4D2 Synthetic Data Generation for Vultr GPU Instance
# =============================================================================
# Run this on a Vultr A100/A40 instance to generate high-quality training data
# before credits expire.
#
# Usage:
#   ./run_vultr_synthetic.sh              # Generate 300 examples (default)
#   ./run_vultr_synthetic.sh 500          # Generate 500 examples
#   ./run_vultr_synthetic.sh 500 true     # Generate 500 + anti-pattern examples
#
# Prerequisites:
#   - OPENAI_API_KEY set in environment or .env file
#   - Python 3.10+ with openai package installed
# =============================================================================

set -e

NUM_EXAMPLES=${1:-300}
INCLUDE_ANTIPATTERNS=${2:-false}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="data/synthetic"

echo "============================================================"
echo "L4D2 Synthetic Data Generation"
echo "============================================================"
echo "Examples to generate: $NUM_EXAMPLES"
echo "Include anti-patterns: $INCLUDE_ANTIPATTERNS"
echo "Timestamp: $TIMESTAMP"
echo "============================================================"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    if [ -f .env ]; then
        export $(grep OPENAI_API_KEY .env | xargs)
    fi
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    echo "Set it via: export OPENAI_API_KEY=sk-..."
    exit 1
fi

# Activate virtual environment if available
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
elif [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

echo ""
echo ">>> Generating priority-weighted examples..."
python scripts/training/generate_synthetic_data.py \
    --num-examples "$NUM_EXAMPLES" \
    --prioritize-gaps \
    --delay 0.3 \
    --output "$OUTPUT_DIR/weighted_${TIMESTAMP}.jsonl"

if [ "$INCLUDE_ANTIPATTERNS" = "true" ]; then
    # Generate additional anti-pattern examples (teaches what NOT to do)
    ANTIPATTERN_COUNT=$((NUM_EXAMPLES / 5))  # 20% anti-patterns
    echo ""
    echo ">>> Generating $ANTIPATTERN_COUNT anti-pattern examples..."
    python scripts/training/generate_synthetic_data.py \
        --num-examples "$ANTIPATTERN_COUNT" \
        --anti-patterns \
        --delay 0.3 \
        --output "$OUTPUT_DIR/antipatterns_${TIMESTAMP}.jsonl"
fi

echo ""
echo "============================================================"
echo "Generation Complete!"
echo "============================================================"
echo "Files created:"
ls -la "$OUTPUT_DIR"/*_${TIMESTAMP}.jsonl 2>/dev/null || echo "  (none)"
echo ""
echo "Next steps:"
echo "1. Review generated examples: head -5 $OUTPUT_DIR/weighted_${TIMESTAMP}.jsonl | jq ."
echo "2. Combine with existing data: cat data/processed/*.jsonl $OUTPUT_DIR/*.jsonl > data/processed/combined_with_synthetic.jsonl"
echo "3. Fine-tune V8: python scripts/training/train_unsloth.py --config configs/unsloth_config_a100.yaml"
echo ""
echo "Estimated API cost for this run: \$$(python3 -c "print(f'{$NUM_EXAMPLES * 2000 * 0.01 / 1000:.2f}')")"
