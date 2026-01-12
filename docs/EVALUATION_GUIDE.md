# OpenAI Evaluation Guide for LoRA Models

Use your **free OpenAI credits** to evaluate your local LoRA models against GPT-4 and other OpenAI models.

---

## Quick Start

### 1. Set Up OpenAI API Key

```bash
# Option 1: Export directly
export OPENAI_API_KEY="your-api-key-here"

# Option 2: Use Doppler (if configured)
doppler run -- python scripts/evaluation/run_lora_evaluation.py
```

### 2. Run Evaluation

```bash
# Activate environment
source activate.sh

# Compare best LoRA against GPT-4o-mini (cheapest)
python scripts/evaluation/run_lora_evaluation.py

# Compare against GPT-4 (better quality, higher cost)
python scripts/evaluation/run_lora_evaluation.py --openai-model gpt-4

# Compare all your LoRA models
python scripts/evaluation/run_lora_evaluation.py --compare-all
```

---

## What Gets Evaluated

The script tests **10 L4D2-specific prompts**:

1. Heal all survivors to full health
2. Announce Tank spawns with health
3. Detect health kit pickups
4. Spawn zombies on timer
5. Give infinite ammo
6. Track Hunter pounce damage
7. Teleport to safe room
8. Spawn Witch at random location
9. Announce top damage dealers
10. Speed boost on special infected kill

### Evaluation Metrics

Each response is scored on:
- **Has Code** (30 points) - Contains code blocks
- **Has SourcePawn Syntax** (30 points) - Uses SourcePawn keywords
- **Has L4D2 APIs** (40 points) - Uses game-specific functions

**Total Score**: 0-100 per prompt

---

## Cost Estimation

Using **GPT-4o-mini** (cheapest):
- ~300 tokens per prompt × 10 prompts = 3,000 tokens
- Input: $0.15 per 1M tokens
- Output: $0.60 per 1M tokens
- **Total cost per run**: ~$0.003 (less than 1 cent)

Using **GPT-4**:
- **Total cost per run**: ~$0.30

With free credits, you can run **hundreds of evaluations** before paying.

---

## Example Output

```
==============================================================
L4D2 LoRA vs OpenAI Evaluation
==============================================================

[1/10] Testing: Write a SourcePawn function to heal all survivors...
  → Local LoRA generating...
  → OpenAI generating...
  ✓ LoRA Score: 100/100
  ✓ OpenAI Score: 100/100

[2/10] Testing: Create a plugin that announces when a Tank spawns...
  → Local LoRA generating...
  → OpenAI generating...
  ✓ LoRA Score: 90/100
  ✓ OpenAI Score: 100/100

...

==============================================================
EVALUATION SUMMARY
==============================================================
LoRA Average Score:   85.0/100
OpenAI Average Score: 95.0/100
Winner: OpenAI

Results saved to: data/evaluation_results/lora_vs_openai_20260109_214530.json
```

---

## Interpreting Results

### LoRA Wins (Score > OpenAI)
Your specialized training is working! The LoRA understands L4D2 patterns better than general-purpose models.

### OpenAI Wins (Score > LoRA)
OpenAI's larger models have better:
- Code structure
- Error handling
- Documentation
- Edge case coverage

**This is expected** - GPT-4 has 1000x more parameters than TinyLlama.

### What Matters
- **Speed**: LoRA runs locally, instant responses
- **Cost**: LoRA is free, OpenAI costs per token
- **Privacy**: LoRA keeps code local
- **Quality**: OpenAI may produce better code

---

## Advanced Usage

### Compare Specific Models

```bash
# Test lora128 vs GPT-4
python scripts/evaluation/run_lora_evaluation.py \
  --lora-adapter model_adapters/l4d2-tiny-v15-lora128 \
  --openai-model gpt-4

# Test smallest LoRA vs cheapest OpenAI
python scripts/evaluation/run_lora_evaluation.py \
  --lora-adapter model_adapters/l4d2-tiny-v15-lora \
  --openai-model gpt-4o-mini
```

### Analyze Results

```bash
# View detailed results
cat data/evaluation_results/lora_vs_openai_*.json | jq '.summary'

# Compare all runs
ls -lt data/evaluation_results/
```

---

## Weekly Evaluation Strategy

Since you get **free credits weekly**, run evaluations to:

1. **Track improvements** - Compare new LoRAs against baseline
2. **Identify weaknesses** - See where LoRA fails vs OpenAI
3. **Optimize training** - Use failures to improve dataset
4. **Benchmark progress** - Weekly scores show training effectiveness

### Recommended Schedule

```bash
# Monday: Baseline evaluation
python scripts/evaluation/run_lora_evaluation.py --compare-all

# After training new models: Compare improvements
python scripts/evaluation/run_lora_evaluation.py \
  --lora-adapter model_adapters/new-model

# Weekly: Full comparison report
python scripts/evaluation/run_lora_evaluation.py \
  --compare-all --openai-model gpt-4
```

---

## Troubleshooting

### "OpenAI API key not found"
```bash
export OPENAI_API_KEY="sk-..."
```

### "Rate limit exceeded"
Wait 60 seconds or use `gpt-4o-mini` instead of `gpt-4`.

### "Model not found"
Check adapter path exists:
```bash
ls -la model_adapters/l4d2-tiny-v15-lora256/
```

---

## Next Steps

After evaluation:
1. Review detailed results in `data/evaluation_results/`
2. Identify patterns where LoRA fails
3. Add those patterns to training data
4. Retrain and re-evaluate

This creates a **continuous improvement loop** using free OpenAI credits for validation.
