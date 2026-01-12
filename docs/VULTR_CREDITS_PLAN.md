# URGENT: Vultr Credits Utilization Plan

**Credits Available**: $250
**Expiry**: <2 days
**Priority**: HIGH
**Last Updated**: January 7, 2026

---

## Latest Evaluation Results (V7 Model)

| Category | Pass Rate | Avg Score | Priority |
|----------|-----------|-----------|----------|
| survivor_mechanics | **83%** | 34.0/40 | **HIGHEST** |
| api_correctness | 100% | **31.3/40** | **HIGH** |
| special_infected | 100% | 34.5/40 | Medium |
| events | 100% | 39.0/40 | Low |
| map_events | 100% | 36.0/40 | Low |

**Overall**: 96.7% pass rate, 34.7/40 average score (3 runs, 30 tests)

---

## Recommended Strategy: Priority-Weighted Synthetic Data

Based on the evaluation gaps, the updated generator now uses weighted category selection:

| Category | Weight | Examples (of 300) | Reason |
|----------|--------|-------------------|--------|
| survivor_mechanics | 3.0 | 81 (27%) | Lowest pass rate |
| api_correctness | 2.5 | 68 (23%) | Lowest score |
| error_handling | 2.0 | 54 (18%) | Common issue in evals |
| special_infected | 1.5 | 40 (14%) | Medium priority |
| events | 1.0 | 27 (9%) | Already good |
| admin_commands | 1.0 | 27 (9%) | Already good |

---

## Quick Start (DO THIS NOW)

### Option 1: Run on Vultr (Recommended)

```bash
# 1. SSH to Vultr (or create new A40 instance)
ssh root@YOUR_VULTR_IP

# 2. Clone repo and setup
git clone https://github.com/RazonIn4K/L4D2-AI-Architect.git
cd L4D2-AI-Architect
pip install -r requirements.txt openai

# 3. Set API key
export OPENAI_API_KEY="sk-your-key"

# 4. Run the script (generates 300 weighted examples)
./run_vultr_synthetic.sh 300

# 5. Or with anti-patterns (teaches what NOT to do)
./run_vultr_synthetic.sh 300 true
```

### Option 2: Run Locally with Doppler

```bash
# Uses your existing Doppler-managed API key
doppler run --project local-mac-work --config dev_personal -- \
    ./run_vultr_synthetic.sh 300 true
```

---

## What Gets Generated

### Priority Categories (based on eval gaps):

1. **survivor_mechanics** (27% of examples)
   - Friendly fire handling
   - Speed modifications using `m_flLaggedMovementValue`
   - Temporary health after events
   - Incap states and revive mechanics

2. **api_correctness** (23% of examples)
   - `GetRandomFloat()` vs wrong `RandomFloat()`
   - `lunge_pounce` vs wrong `pounce` event
   - `tongue_grab` vs wrong `smoker_tongue_grab`
   - Proper `SDKHooks_TakeDamage()` usage

3. **error_handling** (18% of examples)
   - `IsClientInGame()` validation
   - Client index bounds checking
   - Timer callback safety
   - Entity validation before operations

---

## Option 2: Train Local Mistral-7B Model

**Cost**: ~$80-100 (30-40 hours of A40)  
**Output**: Local model for offline inference

### Why?

- Free inference after training (no OpenAI costs)
- Can run on consumer GPUs
- Full control over model

### Step-by-Step

```bash
# 1. Start tmux session
tmux new -s training

# 2. Run Mistral training
python scripts/training/train_unsloth.py \
    --config configs/unsloth_config.yaml \
    --model unsloth/mistral-7b-instruct-v0.3-bnb-4bit \
    --epochs 5 \
    --output model_adapters/mistral-l4d2-v1

# 3. Export to GGUF
python scripts/training/export_model.py \
    --input model_adapters/mistral-l4d2-v1 \
    --format gguf \
    --quantize q4_k_m

# 4. Download model
scp root@VULTR_IP:/root/L4D2-AI-Architect/exports/*.gguf ./exports/
```

---

## Option 3: Run Comprehensive Evaluation

**Cost**: ~$30-40 (12-15 hours)  
**Output**: Detailed model analysis and improvement roadmap

### Step-by-Step

```bash
# Run 5x evaluation on all test cases
python scripts/evaluation/openai_evals.py run --runs 5 --output data/eval_comprehensive.json

# Generate improvement recommendations
python scripts/evaluation/analyze_gaps.py --file data/eval_comprehensive.json
```

---

## Quick Start (Do This NOW)

### 1. Check Existing Vultr Instance

```bash
# Check if you have an existing instance
vultr-cli instance list
```

### 2. Create A40 Instance (if needed)

- Go to: https://my.vultr.com/deploy/
- Select: Cloud GPU → A40 (48GB)
- Region: Atlanta or Dallas
- OS: Ubuntu 22.04 with NVIDIA
- Cost: ~$2.50/hour

### 3. Run Data Generation

SSH in and run:
```bash
cd /root
git clone https://github.com/RazonIn4K/L4D2-AI-Architect.git
cd L4D2-AI-Architect
pip install -r requirements.txt
pip install openai

# Generate synthetic data using GPT-4
export OPENAI_API_KEY="your-key"
python scripts/training/generate_synthetic_data.py --num-examples 300
```

---

## Time-Optimized Schedule

| Time | Task | Cost |
|------|------|------|
| Hour 0-2 | Setup + Start Generation | $5 |
| Hour 2-12 | Generate 300 examples | $25 |
| Hour 12-24 | Train local Mistral model | $30 |
| Hour 24-36 | Run comprehensive evals | $30 |
| Hour 36-48 | Export + Download all artifacts | $5 |

**Total**: ~$95 (leaves $155 buffer)

---

## Files to Download Before Credits Expire

```bash
# CRITICAL - Download these!
scp -r root@VULTR_IP:/root/L4D2-AI-Architect/data/synthetic ./data/
scp -r root@VULTR_IP:/root/L4D2-AI-Architect/model_adapters ./
scp -r root@VULTR_IP:/root/L4D2-AI-Architect/exports ./
scp root@VULTR_IP:/root/L4D2-AI-Architect/data/eval_*.json ./data/
```

---

## Alternative: Quick Win with Credits

If time is very limited, focus on just one high-value task:

### Generate Anti-Pattern Training Data (4 hours, ~$10)

```bash
python scripts/training/generate_antipatterns.py \
    --num-examples 100 \
    --output data/anti_patterns/generated_v3.jsonl
```

This directly addresses the main model weakness (wrong L4D2 APIs).

---

## Post-Credits Next Steps

1. **Merge generated data** into training set
2. **Run OpenAI fine-tune V8** with new data
3. **Evaluate V8** using the evals script
4. **Deploy** improved model to Web UI

---

## Cost Estimates

| Action | OpenAI Cost | Vultr Cost | Total |
|--------|-------------|------------|-------|
| Generate 300 examples | ~$6 (GPT-4o) | $5 (2hr A40) | ~$11 |
| Generate 60 anti-patterns | ~$1.20 | $0 (same session) | ~$1.20 |
| Train V8 fine-tune | ~$5 (est.) | $0 | ~$5 |
| **Total** | **~$12** | **~$5** | **~$17** |

**Remaining Vultr Credits**: ~$233 for additional training or local model export.

---

*Created: January 7, 2026*
*Last Updated: January 7, 2026*
*Priority: URGENT - Credits expire in <48 hours*
