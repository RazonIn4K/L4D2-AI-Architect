# L4D2-AI-Architect Project Statistics

*Generated: 2026-01-08T20:59:40.819985*

## Summary

| Category | Metric | Value |
|----------|--------|-------|
| Code | Python files | 67 |
| Code | Total lines | 44,448 |
| Data | JSONL files | 62 |
| Data | Training examples | 21,091 |
| Models | LoRA adapters | 3 |
| Models | GGUF exports | 1 |
| Models | RL agents | 2 |
| Config | YAML configs | 11 |
| Tests | Test files | 6 |
| Tests | Test functions | 28 |
| Docs | Markdown files | 35 |
| Docs | Total words | 28,245 |

## Code Statistics

- **Total Python files:** 67
- **Total lines:** 44,448
  - Code lines: 31,321
  - Comment lines: 5,965
  - Blank lines: 7,162

### Lines by Module

| Module | Lines | Files |
|--------|-------|-------|
| scripts/training | 18,099 | 19 |
| scripts/utils | 6,906 | 13 |
| scripts/inference | 5,674 | 10 |
| scripts/evaluation | 5,444 | 10 |
| scripts/rl_training | 3,973 | 6 |
| scripts/director | 2,478 | 5 |
| tests | 1,002 | 2 |
| scripts/scrapers | 872 | 2 |

## Data Statistics

- **Total JSONL files:** 62
- **Total training examples:** 21,091

### Examples by Version

| Version | Examples |
|---------|----------|
| v10 | 1,052 |
| v11 | 771 |
| v12 | 835 |
| v13 | 2,783 |
| v14 | 2,576 |
| v2 | 567 |
| v4 | 1,925 |
| v5 | 736 |
| v6 | 775 |
| v7 | 1,394 |
| v8 | 779 |
| v9 | 671 |

### Top JSONL Files

| File | Examples | Size (MB) |
|------|----------|-----------|
| l4d2_train_v14.jsonl | 2,576 | 6.63 |
| l4d2_train_v13_augmented.jsonl | 1,566 | 4.19 |
| l4d2_train_v13.jsonl | 1,010 | 2.44 |
| train.jsonl | 921 | 1.24 |
| l4d2_train_v12.jsonl | 803 | 1.87 |
| l4d2_train_v8.jsonl | 779 | 1.14 |
| l4d2_train_v11.jsonl | 771 | 1.74 |
| l4d2_train_v10.jsonl | 771 | 1.74 |
| train_v7.jsonl | 697 | 1.03 |
| train_v6.jsonl | 697 | 1.03 |
| l4d2_train_v9.jsonl | 671 | 1.01 |
| train_v5.jsonl | 662 | 0.89 |
| l4d2_combined_train.jsonl | 661 | 0.96 |
| l4d2_combined_train_fixed.jsonl | 661 | 0.96 |
| l4d2_train_v4_final.jsonl | 649 | 0.90 |

## Model Statistics

### LoRA Adapters

| Name | LoRA Rank | Size (MB) |
|------|-----------|-----------|
| l4d2-mistral-v9-lora | 32 | 320.06 |
| l4d2-mistral-v10-lora | 32 | 320.06 |
| l4d2-mistral-v10plus-lora | 64 | 640.06 |

### GGUF Exports

| Name | Size (MB) |
|------|-----------|
| l4d2-mistral-v10plus-f16.gguf | 13825.74 |

### RL Agents

| Name | Size (MB) |
|------|-----------|
| final_model | 0.06 |
| final_model | 0.16 |

## Configuration Statistics

| Config File | Model | Sections |
|-------------|-------|----------|
| unsloth_config_gh200.yaml | unsloth/mistral-7b-instruct-v0.3-bnb-... | model, lora, training, data |
| unsloth_config_v14.yaml | unsloth/mistral-7b-instruct-v0.3-bnb-... | model, lora, training, data |
| unsloth_config_v12.yaml | unsloth/mistral-7b-instruct-v0.3-bnb-... | model, lora, training, data |
| unsloth_config.yaml | unsloth/mistral-7b-instruct-v0.3-bnb-... | model, lora, training, data |
| unsloth_config_a100.yaml | unsloth/mistral-7b-instruct-v0.3-bnb-... | model, lora, training, data |
| unsloth_config_v13.yaml | unsloth/mistral-7b-instruct-v0.3-bnb-... | model, lora, training, data |
| unsloth_config_codellama.yaml | unsloth/codellama-7b-bnb-4bit | model, lora, training, data |
| unsloth_config_llama3.yaml | unsloth/llama-3-8b-instruct-bnb-4bit | model, lora, training, data |
| model_server.yaml | - | backends, backend_configs, cache, rate_limit |
| director_config.yaml | - | spawn_rates, stress_factors, flow_control, difficulty |
| unsloth_config_qwen.yaml | unsloth/Qwen2.5-Coder-7B-Instruct-bnb... | model, lora, training, data |

## Test Statistics

- **Test files:** 6
- **Test functions:** 28
- **Test classes:** 7

| File | Functions | Classes | Lines |
|------|-----------|---------|-------|
| conftest.py | 0 | 0 | 359 |
| test_all.py | 28 | 6 | 643 |
| test_enhanced_env.py | 0 | 1 | 486 |
| test_lora.py | 0 | 0 | 115 |
| test_qwen_model.py | 0 | 0 | 308 |
| test_tinyllama.py | 0 | 0 | 38 |

## Documentation Statistics

- **Markdown files:** 35
- **Total words:** 28,245

### Documentation Files

| File | Words | Lines |
|------|-------|-------|
| API_REFERENCE.md | 1,854 | 1226 |
| SECURITY_THREAT_MODEL.md | 1,604 | 325 |
| RL_TRAINING_ROADMAP.md | 1,493 | 676 |
| FINAL_MODEL_COMPARISON.md | 1,322 | 317 |
| ARCHITECTURE.md | 1,294 | 550 |
| EVALUATION_REPORT.md | 1,116 | 216 |
| RUNPOD_TRAINING_GUIDE.md | 1,080 | 392 |
| README.md | 1,069 | 684 |
| ISSUE_REPORT.md | 1,060 | 197 |
| V2_MODEL_VALIDATION.md | 1,012 | 255 |
| MODEL_LEVERAGE_STRATEGY.md | 938 | 320 |
| SECURITY_PATTERNS.md | 884 | 563 |
| VULTR_V11_TRAINING_GUIDE.md | 841 | 608 |
| VULTR_TRAINING_GUIDE.md | 790 | 343 |
| DEPLOYMENT_CHECKLIST.md | 785 | 462 |
| README.md | 738 | 210 |
| README.md | 738 | 210 |
| README.md | 738 | 210 |
| COMPREHENSIVE_ANALYSIS.md | 709 | 141 |
| L4D2_TRAINING_ARCHITECTURE.md | 664 | 242 |
