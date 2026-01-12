# L4D2 AI Model Quality Analysis

> Final assessment of all trained models - January 10, 2026

---

## Executive Summary

| Model | Type | Pass Rate | Avg Score | Best Use Case |
|-------|------|-----------|-----------|---------------|
| **OpenAI V7** | GPT-4o-mini fine-tuned | **80%** | **9.4/10** | Production, accuracy-critical |
| Ollama (Mistral v10+) | GGUF local | 40% | 7.87/10 | Free local inference |
| TinyLlama (r=128) | LoRA adapter | ~50%* | ~8/10* | Small/edge deployment |

*Estimated based on inference testing; not formally benchmarked.

---

## Model Details

### 1. OpenAI V7 (Production Recommended)

```
Model ID: ft:gpt-4o-mini-2024-07-18:highencodelearning:l4d2-sourcemod-v7:CvTBCVPi
Training: 627 examples, 3 epochs, 703K tokens
Cost: ~$2.80 training, $0.60/1M output tokens
```

**Strengths:**
- ✅ 100% correct L4D2 API usage (GetRandomFloat, not RandomFloat)
- ✅ Correct special infected events (lunge_pounce, tongue_grab, charger_carry_start)
- ✅ 0 critical security vulnerabilities
- ✅ Fast inference (~5s per request)

**Weaknesses:**
- ❌ saferoom_heal test (missing patterns)
- ❌ no_ff_panic test (used forbidden patterns)
- ❌ Requires API key and internet

**Usage:**
```python
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="ft:gpt-4o-mini-2024-07-18:highencodelearning:l4d2-sourcemod-v7:CvTBCVPi",
    messages=[{"role": "user", "content": "Write a Tank spawner plugin"}]
)
```

---

### 2. Ollama (l4d2-code-v10plus)

```
Base: Mistral-7B-Instruct-v0.3
Size: 14 GB (GGUF quantized)
Training: 971 examples, r=32 LoRA
```

**Benchmark Results (10 tests):**
```
Pass Rate:     40% (4/10)
Average Score: 7.87/10

By Category:
  event_handling:    100% pass, 10.00 avg ✓
  basic_syntax:       50% pass,  9.33 avg
  l4d2_api:           50% pass,  7.50 avg
  special_infected:    0% pass,  8.83 avg ✗
  advanced_patterns:   0% pass,  3.67 avg ✗
```

**Common Issues:**
- Missing expected patterns (6 occurrences)
- Used forbidden patterns (2 occurrences)
- Errors (1 occurrence)

**Usage:**
```bash
ollama run l4d2-code-v10plus "Write a Tank spawner plugin"
```

---

### 3. TinyLlama Models (Local Adapters)

Seven variants trained with different LoRA ranks:

| Model | Rank | Epochs | Training Acc | Size |
|-------|------|--------|--------------|------|
| lora32-long | 32 | 30 | 99.2% | 112MB |
| lora64 | 64 | 3 | 98.7% | 208MB |
| lora64-long | 64 | 30 | 99.2% | 208MB |
| lora64-a100 | 64 | 3 | 98.9% | 208MB |
| **lora128** | 128 | 30 | **99.4%** | 385MB |
| lora256 | 256 | 30 | 99.3% | 784MB |

**Inference Quality (r=128):**
- ✅ Valid SourcePawn syntax
- ✅ Correct function signatures
- ⚠️ Sometimes incomplete implementations
- ⚠️ Occasional hallucinated constants

**Not Yet Exported to GGUF** - requires llama.cpp installation

---

## Quality Gap Analysis

### Training Accuracy vs Generation Quality

| Model | Training Accuracy | Generation Pass Rate | Gap |
|-------|-------------------|---------------------|-----|
| TinyLlama r=128 | 99.4% | ~50% | -49% |
| Ollama Mistral | 98%+ | 40% | -58% |
| OpenAI V7 | N/A | 80% | N/A |

**Why the gap?**
1. Training accuracy measures token prediction, not functional correctness
2. Benchmark tests require complete, working implementations
3. Small models struggle with multi-step reasoning
4. L4D2-specific APIs require precise knowledge

---

## Recommendations

### For Production Use
```
OpenAI V7 Model
- 80% pass rate, 9.4/10 quality
- Fast, reliable, no GPU needed
- Cost: ~$0.60/1M tokens
```

### For Local/Offline Development
```
Ollama l4d2-code-v10plus
- 40% pass rate, free
- Good for basic syntax and event handling
- Struggles with advanced patterns
```

### For Edge/Embedded
```
TinyLlama r=128 (after GGUF export)
- 1.1B parameters, ~500MB
- Good syntax, incomplete implementations
- Needs llama.cpp for export
```

---

## How to Export TinyLlama to GGUF

Once llama.cpp is installed:

```bash
# Install llama.cpp (macOS)
brew install llama.cpp

# Export best TinyLlama model
python scripts/training/export_gguf_cpu.py \
    --adapter model_adapters/l4d2-tiny-v15-lora128

# Install to Ollama
ollama create l4d2-tiny-v15 -f exports/l4d2-tiny-v15-lora128/gguf/Modelfile
```

---

## Test Coverage

The benchmark suite tests these capabilities:

| Category | Tests | Ollama | OpenAI |
|----------|-------|--------|--------|
| Basic plugin structure | 2 | ✓ | ✓ |
| Console commands | 2 | ½ | ✓ |
| Random number APIs | 2 | ½ | ✓ |
| Player spawn events | 2 | ✓ | ✓ |
| Death events | 2 | ✓ | ✓ |
| Tank mechanics | 2 | ✗ | ✓ |
| Witch proximity | 1 | ✗ | ✓ |
| Friendly fire | 1 | ✗ | ✗ |
| Player glow effects | 1 | ✗ | ? |

---

## Files Reference

| Purpose | File |
|---------|------|
| OpenAI eval summary | `docs/OPENAI_EVALUATION_SUMMARY.md` |
| Model catalog | `docs/MODEL_CATALOG.md` |
| Usage guide | `docs/EVALUATION_AND_USAGE_GUIDE.md` |
| Benchmark results | `results/benchmark.json` |
| OpenAI test results | `data/test_results_openai_*.json` |

---

## Conclusion

**Best Overall: OpenAI V7** - 2x better pass rate than local models

**Best Free Option: Ollama Mistral** - Works offline, good for basic tasks

**Most Potential: TinyLlama r=128** - Highest training accuracy, needs GGUF export to benchmark properly

The quality gap between training accuracy (99%) and generation quality (40-50%) suggests that smaller models struggle with the multi-step reasoning required for complete, working L4D2 plugins. The OpenAI fine-tuned model benefits from GPT-4o-mini's stronger base capabilities.
