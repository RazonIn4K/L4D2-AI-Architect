# V7 Model Training Notes

## Overview
V7 successfully fixed the V6 regression by sanitizing security pattern training data to remove conflicting API references.

## Key Metrics

| Metric | V5 | V6 (Regression) | V7 (Fixed) |
|--------|-----|-----------------|------------|
| Pass Rate | 80% | 50% | 80% ✅ |
| Wrong API Avoidance | 100% | 86.7% | 100% ✅ |
| Security Score | N/A | 9.9/10 | 9.9/10 ✅ |

## Root Cause of V6 Regression
Security pattern files (`data/anti_patterns/*.jsonl`) contained 62 wrong API references that diluted the model's learning signal:

- `RandomFloat(` instead of `GetRandomFloat(`
- `RandomInt(` instead of `GetRandomInt(`
- `m_flSpeed` instead of `m_flLaggedMovementValue`
- `m_flMaxSpeed` instead of `m_flLaggedMovementValue`
- `"pounce"` instead of `"lunge_pounce"`
- `"smoker_tongue_grab"` instead of `"tongue_grab"`
- `"boomer_vomit"` instead of `"player_now_it"`
- `"charger_grab"` instead of `"charger_carry_start"`

## Solution
Created `scripts/utils/sanitize_security_patterns.py` to:
1. Scan all anti-pattern files for wrong API references
2. Apply regex-based corrections
3. Preserve educational context (explaining WHY wrong APIs are bad)
4. Verify code blocks contain only correct APIs

## Training Data
- Base: 662 examples from V5 (validated clean)
- Added: 35 sanitized security patterns
- Total: 697 examples → 627 train / 70 eval
- Epochs: 3
- Trained tokens: 703,947

## V7 Security Learning Validation

### Random API Functions
```sourcepawn
// V7 correctly generates:
float delay = GetRandomFloat(10.0, 30.0);  // ✅ Correct
int health = GetRandomInt(50, 100);        // ✅ Correct
```

### Speed Modification
```sourcepawn
// V7 correctly uses:
SetEntPropFloat(client, Prop_Send, "m_flLaggedMovementValue", 1.5);  // ✅ Correct
// Comment: "CORRECT: m_flLaggedMovementValue is a MULTIPLIER"
```

### Special Infected Events
```sourcepawn
// V7 correctly hooks:
HookEvent("lunge_pounce", Event_HunterPounce);        // ✅ Correct (not "pounce")
HookEvent("tongue_grab", Event_SmokerGrab);           // ✅ Correct (not "smoker_tongue_grab")
HookEvent("charger_carry_start", Event_ChargerGrab);  // ✅ Correct (not "charger_grab")
HookEvent("player_now_it", Event_BileThrow);          // ✅ Correct (not "boomer_vomit")
```

## Lessons Learned

1. **Contrastive learning signal dilution**: When training with anti-patterns, ensure the "bad" examples don't accidentally teach wrong APIs in code blocks. The model learns patterns from all content, not just the intended lesson.

2. **Data quality over quantity**: V6 added security patterns but introduced conflicting signals. V7 achieved same security learning without regression by sanitizing data first.

3. **Test suite flexibility**: LLMs generate valid code with varying implementations. Test suites should accept alternative valid patterns (e.g., `player_entered_safe_area` vs `player_entered_checkpoint`).

4. **Dual-metric evaluation**: Validating both API correctness AND security scoring provides comprehensive quality assessment.

## Model Details
- Job ID: `ftjob-fDzdTbmDP0jC4ehQweIIlvjm`
- Model ID: `ft:gpt-4o-mini-2024-07-18:highencodelearning:l4d2-sourcemod-v7:CvTBCVPi`
- Base Model: `gpt-4o-mini-2024-07-18`
- Status: Production (updated in `scripts/inference/l4d2_codegen.py`)

## Files Modified/Created
- `scripts/utils/sanitize_security_patterns.py` (new)
- `data/anti_patterns/*.jsonl` (sanitized)
- `data/openai_finetune/train_v7_split.jsonl` (new)
- `data/openai_finetune/eval_v7_split.jsonl` (new)
- `data/finetune_job_v7.json` (new)
- `scripts/inference/l4d2_codegen.py` (MODEL_ID updated)
- `scripts/evaluation/automated_test.py` (added alternative pattern support)
- `data/security_test_prompts.txt` (new)
- `scripts/inference/copilot_server_openai.py` (new - API server)

## Deployment

### Quick Start - CLI Tool
```bash
# Single generation
doppler run --project local-mac-work --config dev_personal -- \
  python scripts/inference/l4d2_codegen.py generate "Write a tank spawn announcer"

# Interactive chat
doppler run --project local-mac-work --config dev_personal -- \
  python scripts/inference/l4d2_codegen.py chat
```

### API Server Deployment
```bash
# Start server (default port 8000)
doppler run --project local-mac-work --config dev_personal -- \
  python scripts/inference/copilot_server_openai.py

# Custom port
doppler run --project local-mac-work --config dev_personal -- \
  python scripts/inference/copilot_server_openai.py --port 8080
```

### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/v1/complete` | POST | Code completion |
| `/v1/chat/completions` | POST | Chat (OpenAI-compatible) |
| `/v1/generate-plugin` | POST | Full plugin generation |

### Example API Request
```bash
curl -X POST http://localhost:8000/v1/complete \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a function to give players a speed boost",
    "max_tokens": 500,
    "temperature": 0.1
  }'
```

### Cost Estimation
- Single completion: ~$0.0004
- Full plugin generation: ~$0.002-0.004
- 100 generations: ~$0.20-0.40
