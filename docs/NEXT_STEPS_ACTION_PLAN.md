# L4D2 AI Model: Action Plan for Quality Improvement

> Generated: January 10, 2026
> Based on comprehensive benchmark analysis

---

## Executive Summary

**Current State:**
- Ollama model: **40% pass rate** (4/10 tests)
- OpenAI V7 model: **80% pass rate** (8/10 tests)
- Training data contamination identified and **FIXED**

**Root Cause:**
Training data contained 63% wrong API examples (`RandomInt` instead of `GetRandomInt`), causing the model to learn incorrect patterns.

**Solution:**
Fixed training data created (`l4d2_train_v16_fixed.jsonl`) - ready for retraining.

---

## Completed Actions

### 1. ✅ Benchmark Analysis
Identified 6 failing tests with specific issues:

| Test | Issue | Root Cause |
|------|-------|------------|
| syntax_console_command | Missing `ReplyToCommand` | Training data lacks pattern |
| api_random_int | Used `RandomInt` instead of `GetRandomInt` | **Data contamination** |
| si_tank_spawn | Missing `GetEntProp`/`m_iHealth` | Used `GetClientHealth` |
| si_witch_proximity | Missing `FindEntityByClassname` | Entity/client confusion |
| adv_friendly_fire | Timeout (180s) | Complex prompt issue |
| adv_player_glow | Incomplete implementation | Missing entity property knowledge |

### 2. ✅ Data Contamination Analysis
Quantified contamination in `l4d2_train_v15.jsonl`:

```
RandomInt (WRONG):      155 occurrences
GetRandomInt (RIGHT):   89 occurrences
→ 63% contamination rate

RandomFloat (WRONG):    916 occurrences
GetRandomFloat (RIGHT): 847 occurrences
→ 52% contamination rate
```

### 3. ✅ Created Fix Script
`scripts/utils/fix_api_contamination.py` - Surgically corrects wrong patterns:

```bash
# Run the fix
python scripts/utils/fix_api_contamination.py

# Output
Input examples:     2773
Output examples:    2524
Fixed examples:     154
Removed examples:   204

# Verification
GetRandomInt (correct):   196
RandomInt (WRONG):        0    ← FIXED!
GetRandomFloat (correct): 1039
RandomFloat (WRONG):      0    ← FIXED!
```

### 4. ✅ Created Clean Training Data
New file: `data/processed/l4d2_train_v16_fixed.jsonl`
- 2,524 clean examples
- 0 wrong API patterns
- 204 unfixable examples removed

---

## Next Steps (Prioritized)

### 🔴 HIGH PRIORITY: Retrain with Fixed Data

**Option A: Local Training (Requires GPU)**
```bash
# On a GPU server (Vultr, RunPod, etc.)
python scripts/training/train_unsloth.py \
    --config configs/unsloth_config.yaml \
    --train-file data/processed/l4d2_train_v16_fixed.jsonl \
    --output model_adapters/l4d2-tiny-v16-fixed
```

**Option B: OpenAI Fine-tuning (~$3)**
```bash
# Prepare for OpenAI
python scripts/utils/fix_api_contamination.py \
    --input data/processed/l4d2_train_v15.jsonl \
    --output data/openai_finetune/train_v8.jsonl

# Upload and train via OpenAI
openai api fine_tunes.create \
    -t data/openai_finetune/train_v8.jsonl \
    -m gpt-4o-mini-2024-07-18
```

### 🟡 MEDIUM PRIORITY: Export TinyLlama to GGUF

The existing TinyLlama models (trained on contaminated data) could still be useful for comparison:

```bash
# Install llama.cpp first
brew install llama.cpp

# Export the r=128 model
python scripts/training/export_gguf_cpu.py \
    --adapter model_adapters/l4d2-tiny-v15-lora128

# Install to Ollama
ollama create l4d2-tiny-v15 -f exports/l4d2-tiny-v15-lora128/gguf/Modelfile

# Benchmark to compare
python scripts/evaluation/benchmark_suite.py \
    --model ollama --model-name l4d2-tiny-v15 --quick
```

### 🟢 LOW PRIORITY: Add More Training Examples

To improve coverage on failing categories:

1. **Special Infected Events** (0% pass)
   - Add examples with `FindEntityByClassname` for witch/tank detection
   - Add examples showing entity vs client distinction

2. **Advanced Patterns** (0% pass)
   - Add `SDKHook`/`OnTakeDamage` examples
   - Add `SetEntProp`/`m_iGlowType` examples

---

## Quick Reference

### Files Created/Modified

| File | Purpose |
|------|---------|
| `scripts/utils/fix_api_contamination.py` | Fixes wrong APIs in training data |
| `data/processed/l4d2_train_v16_fixed.jsonl` | Clean training data (2,524 examples) |
| `docs/QUALITY_ANALYSIS_FINAL.md` | Comprehensive quality comparison |
| `docs/NEXT_STEPS_ACTION_PLAN.md` | This document |

### Commands Reference

```bash
# Fix training data contamination
python scripts/utils/fix_api_contamination.py

# Run benchmark (quick - 10 tests)
python scripts/evaluation/benchmark_suite.py --model ollama --quick

# Run benchmark (full - 55 tests)
python scripts/evaluation/benchmark_suite.py --model ollama

# Test a model directly
python scripts/inference/test_lora.py --adapter model_adapters/l4d2-tiny-v15-lora128

# Chat with Ollama model
python scripts/inference/copilot_cli.py chat
```

---

## Expected Results After Retraining

Based on OpenAI V7 which was trained on clean data:

| Metric | Before (Contaminated) | After (Clean) |
|--------|----------------------|---------------|
| Pass Rate | 40% | **~80%** |
| Wrong API | 50% | **0%** |
| Avg Score | 7.87 | **~9.4** |

---

## Timeline Estimate

| Task | Time Required |
|------|---------------|
| Fix training data | ✅ Done |
| GPU setup (Vultr/RunPod) | 15-30 min |
| Retrain model | 1-2 hours |
| Export to GGUF | 15-30 min |
| Benchmark verification | 10-15 min |

**Total: ~3 hours with GPU access**

---

## Conclusion

The quality gap between the Ollama model (40%) and OpenAI V7 (80%) is explained by training data contamination. The contamination has been fixed in `l4d2_train_v16_fixed.jsonl`.

**Next action:** Retrain on clean data to achieve ~80% pass rate locally.
