# L4D2-AI-Architect: Evaluation & Usage Guide

> Complete guide to using, evaluating, and validating your trained models

---

## Quick Status Check

Run these commands to verify everything works:

```bash
# 1. Validate scripts and configs
python validate_scripts.py

# 2. Run the test suite (28 tests)
pytest tests/test_all.py -v

# 3. Pre-flight system check
python preflight_check.py
```

---

## Using Your Trained Models

### Option 1: Ollama (Recommended for Local Use)

**Step 1: Export to GGUF**
```bash
# Export TinyLlama r=64-long (best balance of size/quality)
python scripts/training/export_gguf_cpu.py \
    --adapter model_adapters/l4d2-tiny-v15-lora64-long

# Or export r=128 (highest accuracy)
python scripts/training/export_gguf_cpu.py \
    --adapter model_adapters/l4d2-tiny-v15-lora128
```

**Step 2: Install to Ollama**
```bash
ollama create l4d2-tiny -f exports/l4d2-tiny-v15-lora64-long/gguf/Modelfile
```

**Step 3: Use It**
```bash
# Direct query
ollama run l4d2-tiny "Write a Tank spawner plugin"

# Via CLI tool
python scripts/inference/copilot_cli.py ollama --prompt "Create a heal command"

# Interactive chat
python scripts/inference/copilot_cli.py chat
```

### Option 2: OpenAI API Server

For production use with OpenAI fine-tuned models:

```bash
# Set your API key
export OPENAI_API_KEY=sk-...

# Start the server
python scripts/inference/copilot_server_openai.py --port 8000

# Test it
curl -X POST http://localhost:8000/v1/complete \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Write a function to spawn a witch"}'
```

### Option 3: Local LoRA Server

For testing without exporting:

```bash
python scripts/inference/copilot_server.py \
    --model model_adapters/l4d2-tiny-v15-lora64-long
```

---

## Evaluating Model Quality

### Quick Test (Single Model)

```bash
# Test a specific LoRA adapter
python scripts/inference/test_lora.py \
    --adapter model_adapters/l4d2-tiny-v15-lora64-long \
    --base tiny
```

### Comprehensive Benchmark (55 Tests)

```bash
# Quick test (10 tests)
python scripts/evaluation/benchmark_suite.py --model ollama --model-name l4d2-tiny --quick

# Full benchmark (55 tests)
python scripts/evaluation/benchmark_suite.py --model ollama --model-name l4d2-tiny --output results.json --markdown results.md

# Test specific category
python scripts/evaluation/benchmark_suite.py --model ollama --category "L4D2-Specific"

# List all available tests
python scripts/evaluation/benchmark_suite.py --list-tests
```

### Compare Multiple Models

```bash
python scripts/evaluation/benchmark_suite.py --compare \
    --models "l4d2-tiny,l4d2-code-v10plus,gpt-4o-mini"
```

---

## Validating Generated Code

### Validate a Single File

```bash
python scripts/evaluation/validate_generated_code.py validate path/to/plugin.sp
```

### Validate a Directory

```bash
python scripts/evaluation/validate_generated_code.py validate-dir data/generated/
```

### Generate Validation Report

```bash
python scripts/evaluation/validate_generated_code.py report data/generated/ --output report.json
```

### Validation Stages

The validator runs 3 stages:

1. **Static Analysis**: Checks includes, events, braces, semicolons
2. **Compilation Check**: Uses spcomp if available
3. **Semantic Analysis**: Validates function signatures, detects hallucinations

### Common Hallucinations Detected

The validator catches these common model mistakes:
- Invalid includes (e.g., `l4d2_bile.inc`, `prop_tank.inc`)
- Fake functions (e.g., `RandomFloat`, `GetEntityModel`, `TakeDamage`)
- Wrong event names (e.g., `pounce` instead of `lunge_pounce`)
- Invalid entity properties

---

## Test Categories

### Benchmark Suite Categories

| Category | Tests | Description |
|----------|-------|-------------|
| Basic Syntax | 10 | Plugin structure, commands, timers |
| L4D2-Specific | 15 | Team checks, weapons, SDK hooks |
| Event Handling | 10 | Spawn, death, round events |
| Special Infected | 10 | Tank, Witch, Hunter mechanics |
| Advanced Patterns | 10 | Voting, databases, KeyValues |

### Difficulty Levels

- **Easy**: Basic syntax and structure
- **Medium**: L4D2-specific APIs
- **Hard**: Complex patterns and edge cases

---

## API Endpoints

### OpenAI Server (`copilot_server_openai.py`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/complete` | POST | Code completion |
| `/v1/chat/completions` | POST | OpenAI-compatible chat |
| `/v1/generate-plugin` | POST | Full plugin generation |
| `/health` | GET | Server health check |

### Local Server (`copilot_server.py`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/complete` | POST | Code completion |
| `/chat` | POST | Chat interface |
| `/health` | GET | Server health check |

---

## Model Performance Summary

Based on training metrics:

| Model | Accuracy | Size | Best For |
|-------|----------|------|----------|
| r=32-long | 99.2% | 112MB | Edge devices |
| r=64-long | 99.2% | 208MB | Balanced |
| r=128 | 99.4% | 385MB | Best quality |
| r=256 | 99.3% | 784MB | Experimental |
| Mistral v10+ | High | 644MB | Production |

---

## Troubleshooting

### Model Not Loading

```bash
# Check if adapter files exist
ls -la model_adapters/l4d2-tiny-v15-lora64-long/adapter_model.safetensors

# Check config
cat model_adapters/l4d2-tiny-v15-lora64-long/adapter_config.json
```

### Validation Fails

```bash
# Run with verbose output
python scripts/evaluation/validate_generated_code.py validate file.sp --verbose

# Check for hallucinated includes
grep -E "^#include" file.sp
```

### Ollama Export Fails

```bash
# Check llama.cpp is available
which llama-quantize

# Try different quantization
python scripts/training/export_gguf_cpu.py --adapter ... --quantize q8_0
```

---

## File Locations

| What | Where |
|------|-------|
| Trained Models | `model_adapters/l4d2-tiny-v15-*` |
| GGUF Exports | `exports/` |
| Training Data | `data/processed/l4d2_train_v15.jsonl` |
| Evaluation Results | `data/test_results_*.json` |
| Benchmark Reports | `results/` |

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | For OpenAI server |
| `CUDA_VISIBLE_DEVICES` | GPU selection |
| `HF_HOME` | Hugging Face cache |

---

## Next Steps

1. **Export a model** to GGUF for Ollama
2. **Run benchmarks** to measure quality
3. **Compare models** to find the best one
4. **Validate outputs** before using in production
