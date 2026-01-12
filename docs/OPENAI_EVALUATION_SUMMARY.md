# OpenAI Fine-Tuned Models: Evaluation Summary

> Last Updated: January 10, 2026

---

## Production Model (V7)

### Model Details

| Property | Value |
|----------|-------|
| **Model ID** | `ft:gpt-4o-mini-2024-07-18:highencodelearning:l4d2-sourcemod-v7:CvTBCVPi` |
| **Base Model** | GPT-4o-mini |
| **Training Examples** | 627 |
| **Validation Examples** | 70 |
| **Epochs** | 3 |
| **Trained Tokens** | 703,947 |
| **Status** | Production Ready |

### Evaluation Results

| Metric | Score |
|--------|-------|
| **Pass Rate** | 80% (8/10 tests) |
| **Average Score** | 9.4/10 |
| **Security Score** | 9.9/10 |
| **Wrong API Avoidance** | 100% |
| **Forbidden Pattern Avoidance** | 86.7% |
| **Critical Security Issues** | 0 |

### Correct API Usage (Validated)

The V7 model correctly uses these L4D2-specific APIs:

| Correct API | Wrong Alternative (Avoided) |
|-------------|---------------------------|
| `GetRandomFloat` | ~~RandomFloat~~ |
| `GetRandomInt` | ~~RandomInt~~ |
| `m_flLaggedMovementValue` | ~~m_flSpeed~~ |
| `lunge_pounce` | ~~pounce~~ |
| `tongue_grab` | ~~smoker_tongue_grab~~ |
| `charger_carry_start` | ~~charger_grab~~ |
| `player_now_it` | ~~boomer_vomit~~ |

### Test Results Breakdown

| Test Case | Passed | Score | Notes |
|-----------|--------|-------|-------|
| speed_boost | Yes | 10.0 | Correct m_flLaggedMovementValue |
| no_ff_panic | No | 10.0 | Forbidden patterns used |
| kill_tracker | Yes | 10.0 | Correct infected_death event |
| saferoom_heal | No | 4.0 | Missing patterns |
| tank_announce | Yes | 10.0 | Correct tank_spawn event |
| hunter_pounce | Yes | 10.0 | Correct lunge_pounce |
| bile_tracker | Yes | 10.0 | Correct player_now_it |
| smoker_grab | Yes | 10.0 | Correct tongue_grab |
| charger_carry | Yes | 10.0 | Correct charger_carry_start |
| random_timer | Yes | 10.0 | Correct GetRandomFloat |

---

## Model Version History

### V7 (Production) - January 7, 2026
- **Job ID**: `ftjob-fDzdTbmDP0jC4ehQweIIlvjm`
- **Changes**: Sanitized security patterns - fixed 62 wrong API references
- **Result**: 80% pass rate, 100% wrong API avoidance

### V6 - January 7, 2026
- **Job ID**: `ftjob-wL0SrpvsB1WVzhKWB5kbxXoP`
- **Training Examples**: 697
- **Result**: 50% pass rate, 86.7% API avoidance (regression)

### V5 - January 6, 2026
- **Job ID**: `ftjob-ZfZHfwBKeszByjfDrXgOFlKv`
- **Training Examples**: 662
- **Result**: Initial cleaned dataset

---

## How to Use

### API Usage

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="ft:gpt-4o-mini-2024-07-18:highencodelearning:l4d2-sourcemod-v7:CvTBCVPi",
    messages=[
        {"role": "system", "content": "You are an expert SourcePawn developer for L4D2."},
        {"role": "user", "content": "Write a Tank spawner plugin"}
    ]
)
```

### Start the Server

```bash
export OPENAI_API_KEY=sk-...
python scripts/inference/copilot_server_openai.py --port 8000
```

### Run Evaluations

```bash
# Quick evaluation (3 tests)
python scripts/evaluation/openai_evals.py quick

# Full evaluation (18 tests)
python scripts/evaluation/openai_evals.py run --runs 3

# Analyze previous results
python scripts/evaluation/openai_evals.py analyze
```

---

## Evaluation Test Categories

The evaluation suite tests 18 different scenarios:

### Special Infected Events (8 tests)
- tank_spawn, hunter_pounce, smoker_grab, charger_carry
- bile_tracker, jockey_ride, spitter_spit, tank_rock

### Survivor Mechanics (2 tests)
- speed_boost, friendly_fire

### Map Events (5 tests)
- saferoom_heal, witch_proximity, finale_start
- rescue_vehicle, gauntlet_run

### Error Handling (2 tests)
- null_client_check, entity_validation

### API Correctness (1 test)
- random_timer

---

## Cost Analysis

| Model | Training Cost | Per-Request Cost |
|-------|--------------|------------------|
| V7 (703K tokens) | ~$2.80 | $0.15/1M input, $0.60/1M output |
| GPT-4o-mini base | - | $0.15/1M input, $0.60/1M output |

---

## Files Location

| File | Purpose |
|------|---------|
| `data/finetune_job_v7.json` | V7 job metadata |
| `data/finetune_job_v6.json` | V6 job metadata |
| `data/finetune_job_v5.json` | V5 job metadata |
| `data/test_results_openai_*.json` | Individual test results |
| `data/openai_evaluation_results.csv` | Aggregated results |
| `scripts/evaluation/openai_evals.py` | Evaluation script |
| `scripts/inference/copilot_server_openai.py` | Production server |

---

## Key Improvements in V7

1. **Fixed 62 wrong API references** in training data
2. **100% wrong API avoidance** (vs 86.7% in V6)
3. **80% pass rate** (vs 50% in V6)
4. **Security hardened** - no critical vulnerabilities
5. **L4D2-specific events** correctly identified
