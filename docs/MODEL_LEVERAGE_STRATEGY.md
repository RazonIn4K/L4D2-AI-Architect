# L4D2 Fine-Tuned Model Leverage Strategy

**Date**: January 6, 2026
**Model**: `ft:gpt-4o-mini-2024-07-18:highencodelearning:l4d2-sourcemod-v2:CuyGSbKT`

---

## Executive Summary

Your V2 fine-tuned model is a valuable asset that can be leveraged in multiple ways:
1. **Direct code generation** for L4D2 plugin development
2. **Data augmentation** to bootstrap training data for V3
3. **User-facing tools** for the L4D2 modding community
4. **Cost-optimized batch processing** for high-volume tasks

---

## 1. Model Assets Inventory

### What You Have

| Asset | Description | Value |
|-------|-------------|-------|
| V2 Fine-Tuned Model | GPT-4o-mini trained on 532 L4D2 examples | 9.4/10 validation score |
| Training Pipeline | Scrapers, filters, synthetic generators | Reusable for V3 |
| Evaluation Suite | SourcePawnEvaluator with pattern matching | Quality assurance |
| Documentation | Comparison reports, validation results | Knowledge base |

### Model Performance

```
Average Score: 9.4/10
Pass Rate: 100% (5/5 tests)
L4D2 API Accuracy: 100%
Would Compile: ~90%
```

---

## 2. API Usage & Cost Optimization

### Pricing Structure

| Usage Type | Input (per 1M) | Output (per 1M) | Typical Generation |
|------------|----------------|-----------------|-------------------|
| Direct API | $0.30 | $1.20 | ~$0.003 |
| Batch API | $0.15 | $0.60 | ~$0.0015 |

### Cost Optimization Strategies

#### A. Use Batch API for Non-Urgent Tasks (50% Savings)
```python
# OpenAI Batch API - results within 24 hours at half price
# Ideal for: data augmentation, bulk generation, testing
```

#### B. Efficient Prompt Design
```python
# Good: Specific, concise prompts
"Write a plugin that spawns a tank every 5 minutes with chat announcement"

# Bad: Verbose, redundant prompts
"I need you to write a SourceMod plugin for Left 4 Dead 2 that will create
a new tank special infected entity and spawn it into the game world at a
regular interval of approximately 5 minutes..."
```

#### C. Response Caching
- Cache frequent requests locally
- Build a response database for common patterns
- Reduces redundant API calls

#### D. Temperature Settings
- Use `0.3` for consistent, deterministic output
- Fewer retries needed = lower costs

---

## 3. Immediate Usage Options

### A. CLI Tool (Created)

```bash
# Single generation
python scripts/inference/l4d2_codegen.py generate "Write a tank spawn plugin"

# Batch generation
python scripts/inference/l4d2_codegen.py batch prompts.txt --output generated/

# Interactive chat
python scripts/inference/l4d2_codegen.py chat

# Cost estimation
python scripts/inference/l4d2_codegen.py estimate prompts.txt

# Model info
python scripts/inference/l4d2_codegen.py info
```

### B. Direct API Integration

```python
from openai import OpenAI

client = OpenAI()
MODEL_ID = "ft:gpt-4o-mini-2024-07-18:highencodelearning:l4d2-sourcemod-v2:CuyGSbKT"

response = client.chat.completions.create(
    model=MODEL_ID,
    messages=[
        {"role": "system", "content": "You are an expert SourcePawn developer for L4D2."},
        {"role": "user", "content": "Write a plugin that heals all survivors"}
    ],
    max_tokens=2048,
    temperature=0.3
)
print(response.choices[0].message.content)
```

### C. Batch API for Data Augmentation

```python
from openai import OpenAI

client = OpenAI()

# Create batch file
batch_requests = []
for i, prompt in enumerate(prompts):
    batch_requests.append({
        "custom_id": f"request-{i}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL_ID,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2048
        }
    })

# Save and upload batch file
# Submit via OpenAI Batch API for 50% discount
```

---

## 4. Model Improvement Roadmap

### Phase 1: Data Augmentation (This Week)

**Goal**: Use V2 model to generate training data for V3

1. **Create 100-200 diverse prompts** covering:
   - All special infected (Tank, Witch, Boomer, Hunter, Smoker, Spitter, Charger, Jockey)
   - Survivor mechanics (healing, reviving, weapons, items)
   - Game events (saferoom, finale, versus, survival)
   - Director manipulation
   - Admin commands
   - HUD elements

2. **Run batch generation** (50% cost savings)
   ```bash
   python scripts/inference/l4d2_codegen.py batch prompts/v3_augmentation.txt \
       --output data/v3_synthetic/
   ```

3. **Validate with compiler** (spcomp)
   - Track compilation success rate
   - Use failures to identify model weaknesses

4. **Manual review** for logical correctness

### Phase 2: Expanded Data Collection (1-2 Weeks)

**Sources to scrape**:

| Source | Estimated Examples | Quality |
|--------|-------------------|---------|
| AlliedModders L4D2 Forum | 500+ | High |
| GitHub "l4d2 sourcemod" | 300+ | Medium-High |
| SourceMod Official Plugins | 100+ | Very High |
| Steam Workshop (with source) | 200+ | Variable |

**Target**: 1500-2000 total training examples

### Phase 3: V3 Model Training (2-3 Weeks)

**Improvements over V2**:
- 3-4x more training examples
- Compilation-validated responses
- More diverse prompt formats
- Better coverage of edge cases

**Expected outcome**: 9.4 → 9.7+ validation score

### Phase 4: Deployment (1 Month)

**User-facing tools** (in priority order):

1. **VS Code Extension**
   - Inline code completion
   - "Generate plugin" command
   - Syntax-aware suggestions

2. **Discord Bot**
   - L4D2 modding community integration
   - Rate-limited for cost control
   - Promotes adoption

3. **Web Playground**
   - Browser-based code generation
   - Syntax highlighting
   - Copy/download functionality

---

## 5. The Virtuous Cycle: Model Bootstrapping

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  V2 Model ──→ Generate Synthetic Data ──→ Validate ──┐ │
│     ↑                                                 │ │
│     │                                                 ↓ │
│     └──────────── Train V3 Model ←── Quality Filter ←─┘ │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Key Insight**: Your fine-tuned model pays for itself by generating training data.

- Cost per generation: ~$0.003
- 100 generations: ~$0.30
- Value created: 100 new training examples for V3
- ROI: Massive (training data is expensive to create manually)

---

## 6. Risk Mitigation

### A. The "Confident Hallucination" Problem

**Risk**: Model generates syntactically correct but logically flawed code.

**Mitigation**:
- Always validate with compiler (catches syntax errors)
- Manual review for production use
- Spot-check synthetic data before training V3

### B. Prompt Injection in Tools

**Risk**: Malicious users craft prompts to exploit the system.

**Mitigation**:
- Strong system prompt constraints
- Label all output as "AI-generated"
- Warn users to review before running

### C. Cost Overruns

**Risk**: Unexpected API costs from high usage.

**Mitigation**:
- Use Batch API for bulk operations (50% off)
- Set spending limits in OpenAI dashboard
- Monitor usage via the CLI tool

---

## 7. Quick Start Checklist

- [ ] Test CLI tool: `python scripts/inference/l4d2_codegen.py info`
- [ ] Generate first plugin: `python scripts/inference/l4d2_codegen.py generate "..."`
- [ ] Create prompts file for batch generation
- [ ] Estimate batch costs: `python scripts/inference/l4d2_codegen.py estimate`
- [ ] Run batch generation for V3 data
- [ ] Validate generated code with spcomp
- [ ] Add quality examples to training dataset

---

## 8. Files Reference

| File | Purpose |
|------|---------|
| `scripts/inference/l4d2_codegen.py` | CLI tool for model usage |
| `scripts/evaluation/evaluate_models.py` | Evaluation harness |
| `docs/V2_MODEL_VALIDATION.md` | Validation test results |
| `docs/FINAL_MODEL_COMPARISON.md` | V1 vs V2 comparison |
| `data/openai_finetune/train_v2.jsonl` | V2 training data |

---

## 9. Cost-Benefit Summary

| Action | Cost | Benefit | Priority |
|--------|------|---------|----------|
| CLI tool usage | ~$0.003/gen | Immediate utility | **HIGH** |
| Batch data augmentation | ~$0.15/100 | V3 training data | **HIGH** |
| Data expansion (scraping) | $0 | Better model | **HIGH** |
| V3 training | ~$10-20 | Improved accuracy | MEDIUM |
| VS Code extension | Dev time | User adoption | MEDIUM |

---

## Conclusion

Your V2 fine-tuned model is a **multiplier** for value creation:

1. **Use it now** for L4D2 plugin generation (CLI tool ready)
2. **Use Batch API** for 50% cost savings on bulk tasks
3. **Bootstrap V3 data** by generating synthetic examples
4. **Expand the dataset** from AlliedModders, GitHub, etc.
5. **Deploy user tools** to build community adoption

The model pays for itself by creating training data for the next version.
