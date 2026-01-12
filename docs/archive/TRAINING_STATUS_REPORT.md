# L4D2 Training Pipeline Status Report

**Date**: January 6, 2026

## Summary

The training pipeline has been significantly improved with game detection, anti-pattern training, and filtered datasets. However, the local TinyLlama LoRA model (1.1B params) is insufficient for complex code generation.

## Completed Improvements

### 1. Game Detection System
- **File**: `scripts/utils/game_detection.py`
- Accurately identifies L4D2, CSGO, TF2, and generic SourcePawn code
- Detection based on includes, events, properties, functions, and entities
- Confidence scoring (high/medium/low)

### 2. Training Data Filtering
- **Results**: Filtered 199 non-L4D2 files from training set
- **Final dataset**: 676 examples (356 L4D2-specific, 320 generic SourcePawn)
- Cross-game contamination reduced from ~60% to <5%

### 3. Anti-Pattern Training
- **File**: `data/anti_patterns/l4d2_anti_patterns.jsonl`
- **Count**: 16 anti-patterns covering common L4D2 mistakes
- Teaches model what NOT to generate alongside corrections

### 4. Reference Implementations
- **Location**: `data/corrected_plugins/`
- **Count**: 17 corrected plugins
- **Validation**: 94.1% pass rate (16/17)
- Single false positive: `m_flSpeed` mentioned in comment

## Model Validation Results

### Corrected Reference Plugins
| Metric | Value |
|--------|-------|
| Total Files | 17 |
| Passed | 16 |
| Failed | 1 (false positive) |
| Pass Rate | 94.1% |
| Avg Score | 10.0/10 |

### TinyLlama LoRA Generated Plugins
| Metric | Value |
|--------|-------|
| Total Files | 3 |
| Passed | 0 |
| Failed | 3 |
| Pass Rate | 0.0% |
| Avg Score | 4.33/10 |

### Issues with TinyLlama Model
1. **Missing structure**: No #pragma/#include directives
2. **Incomplete code**: Generates fragments, not full plugins
3. **Hallucinated APIs**: Invents non-existent functions/variables
4. **Doesn't follow prompts**: Generates template code instead of requested functionality

## Available Models

### 1. Local TinyLlama LoRA (Not Recommended)
- **Path**: `model_adapters/l4d2-lora`
- **Base**: TinyLlama-1.1B-Chat-v1.0
- **Status**: Generates SourcePawn syntax but poor quality
- **Issues**: Too small for complex code generation

### 2. Fine-tuned GPT-4o-mini (Recommended) ✅ TESTED
- **ID**: `ft:gpt-4o-mini-2024-07-18:highencodelearning:l4d2-sourcemod-v2:CuyGSbKT`
- **Training**: 517 quality examples + 15 synthetic
- **Validation Score**: 10/10 (3/3 plugins passed)
- **Status**: Working! Use with Doppler API key

```bash
# Generate code with fine-tuned model
OPENAI_API_KEY=$(doppler secrets get OPENAI_API_KEY --project local-mac-work --config dev_personal --plain) \
  python scripts/inference/l4d2_codegen.py generate "Your prompt here"
```

## Recommendations

### Immediate Actions (DONE ✅)
1. ~~**Set OPENAI_API_KEY** in `.env` to test fine-tuned GPT-4o-mini~~
   - Using Doppler: `--project local-mac-work --config dev_personal`
2. ~~**Validate** GPT-4o-mini output with `validate_generated_code.py`~~
   - Result: 3/3 passed (100%), 10/10 score

### For Better Local Models
1. **Train on GPU server** (Vultr A100/GH200) with:
   - Filtered L4D2 dataset (676 examples)
   - Larger base model (Mistral-7B or CodeLlama-13B)
   - Anti-pattern examples included
2. **Config**: Use `configs/unsloth_config_a100.yaml` or `configs/unsloth_config_gh200.yaml`

### Training Command
```bash
# On Vultr A100
python scripts/training/train_unsloth.py \
    --config configs/unsloth_config_a100.yaml \
    --filter-l4d2 \
    --include-anti-patterns
```

## Key Files Modified/Created

| File | Purpose |
|------|---------|
| `scripts/utils/game_detection.py` | Game detection module |
| `data/anti_patterns/l4d2_anti_patterns.jsonl` | Anti-pattern training data |
| `scripts/training/prepare_dataset.py` | Updated with filtering |
| `docs/L4D2_TRAINING_ARCHITECTURE.md` | Architecture documentation |
| `data/corrected_plugins/*.sp` | 17 reference implementations |
| `scripts/inference/generate_test_plugins.py` | Test plugin generator |

## Anti-Patterns Covered

| Error Pattern | Correct Pattern |
|---------------|-----------------|
| `m_flSpeed` | `m_flLaggedMovementValue` |
| Event damage modification | `SDKHook_OnTakeDamage` |
| `pounce` event | `lunge_pounce` |
| `smoker_tongue_grab` | `tongue_grab` |
| `RandomFloat()` | `GetRandomFloat()` |
| Client index in timers | `GetClientUserId()` |
| `GetClientName()` no buffer | Buffer required |
| `TakeDamage()` | `SDKHooks_TakeDamage()` |
| `RoundFloat()` | `RoundToNearest()` |
| `GetEntityModel()` | `GetEntPropString()` |

## Next Steps

1. Test GPT-4o-mini with API key
2. If local model needed: Train Mistral-7B on Vultr with filtered data
3. Add more reference implementations to training set
4. Consider expanding anti-patterns based on validation failures
