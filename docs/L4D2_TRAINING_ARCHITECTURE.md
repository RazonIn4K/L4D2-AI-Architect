# L4D2 SourcePawn Training Architecture

This document describes the training data architecture and quality control measures implemented to ensure the model generates correct L4D2-specific code rather than hallucinating cross-game APIs.

## Problem Statement

Initial training resulted in models that generated plausible-looking SourcePawn code with critical errors:
- **60% Cross-Game Contamination**: Training data mixed CS:GO, TF2, and L4D2 code
- **25% Conceptual Gaps**: Misunderstanding of L4D2-specific patterns
- **15% Data Quality Issues**: Incomplete or incorrect examples

### Examples of Training Failures

| Error Type | Wrong Pattern | Correct L4D2 Pattern |
|------------|---------------|---------------------|
| Invalid Property | `m_flSpeed` | `m_flLaggedMovementValue` (multiplier) |
| Invalid Event | `pounce` | `lunge_pounce` |
| Invalid Function | `RandomFloat()` | `GetRandomFloat()` |
| Wrong API Usage | Event modification to block damage | `SDKHook_OnTakeDamage` |

## Solution Architecture

### 1. Game Detection Module

Located at `scripts/utils/game_detection.py`, this module analyzes SourcePawn code to identify which Source game it targets.

```python
from utils.game_detection import detect_game, is_l4d2_code

result = detect_game(code)
print(f"Game: {result.detected_game}")      # e.g., "l4d2", "csgo", "tf2"
print(f"Confidence: {result.confidence}")    # e.g., "high", "medium"
print(f"L4D2 Compatible: {result.is_l4d2_compatible}")
```

#### Detection Signals

| Category | Weight | L4D2 Examples |
|----------|--------|---------------|
| Includes | 1.0 | `left4dhooks`, `l4d2_stocks`, `l4d2util` |
| Events | 0.8 | `player_incapacitated`, `tongue_grab`, `tank_spawn` |
| Properties | 0.7 | `m_zombieClass`, `m_isIncapacitated`, `m_healthBuffer` |
| Functions | 0.8 | `L4D_GetSurvivorVictim`, `GetZombieClass` |
| Entities | 0.6 | `infected`, `witch`, `weapon_first_aid_kit` |
| Comments | 0.3 | "left 4 dead", "survivor", "infected" |

### 2. Training Data Filtering

The `prepare_dataset.py` script now filters training data:

```bash
# Default: L4D2 filtering enabled
python scripts/training/prepare_dataset.py

# Disable filtering (include all SourceMod code)
python scripts/training/prepare_dataset.py --no-filter-l4d2

# Include anti-patterns (default)
python scripts/training/prepare_dataset.py --include-anti-patterns
```

### 3. Anti-Pattern Training

Anti-patterns teach the model what NOT to generate by showing incorrect patterns alongside corrections.

Located at `data/anti_patterns/l4d2_anti_patterns.jsonl`:

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert SourcePawn developer..."},
    {"role": "user", "content": "How do I set survivor speed?"},
    {"role": "assistant", "content": "Use m_flLaggedMovementValue (multiplier)...\n\n**WRONG**: m_flSpeed does not exist..."}
  ],
  "type": "anti_pattern",
  "error_category": "invalid_property"
}
```

Current anti-patterns cover:
- `m_flSpeed` → `m_flLaggedMovementValue`
- Event damage modification → `SDKHook_OnTakeDamage`
- `pounce` → `lunge_pounce`
- `smoker_tongue_grab` → `tongue_grab`
- `RandomFloat()` → `GetRandomFloat()`
- Client index in timers → `GetClientUserId()`
- `GetClientName()` without buffer
- Incapacitation detection patterns
- `TakeDamage()` → `SDKHooks_TakeDamage()`
- Saferoom detection properties
- Temp health buffer properties
- `RoundFloat()` → `RoundToNearest()`
- `GetEntityModel()` → `GetEntPropString()`
- Tank health event (polling pattern)
- Event callback return types
- Ground height detection (TraceRay)

## L4D2 API Reference

### Correct Properties

| Property | Type | Usage |
|----------|------|-------|
| `m_flLaggedMovementValue` | float | Speed multiplier (1.0 = normal) |
| `m_isIncapacitated` | int | 1 if knocked down |
| `m_zombieClass` | int | 1-6=SI, 8=Tank |
| `m_healthBuffer` | float | Temp health amount |
| `m_healthBufferTime` | float | When temp health was set |
| `m_currentReviveCount` | int | Times revived |
| `m_bIsOnThirdStrike` | int | Black & white state |
| `m_isInMissionStartArea` | int | In start saferoom |

### Correct Events

| Event | When Fired |
|-------|------------|
| `player_incapacitated` | Survivor knocked down |
| `revive_success` | Survivor helped up |
| `tongue_grab` | Smoker grabs |
| `lunge_pounce` | Hunter lands |
| `charger_carry_start` | Charger grabs |
| `jockey_ride` | Jockey mounts |
| `tank_spawn` | Tank spawns |
| `infected_death` | Common infected dies |
| `create_panic_event` | Horde triggered |

### Correct Functions

```sourcepawn
// Random numbers
float f = GetRandomFloat(0.0, 1.0);
int i = GetRandomInt(1, 10);

// Rounding
int rounded = RoundToNearest(3.7);

// Player names (buffer required!)
char name[MAX_NAME_LENGTH];
GetClientName(client, name, sizeof(name));

// Entity models
char model[PLATFORM_MAX_PATH];
GetEntPropString(entity, Prop_Data, "m_ModelName", model, sizeof(model));

// Damage
SDKHooks_TakeDamage(victim, attacker, attacker, 50.0, DMG_GENERIC);

// Timers (use UserID!)
CreateTimer(5.0, Timer_Callback, GetClientUserId(client));
```

## Validation Pipeline

### Stage 1: Syntax Validation
- Bracket matching
- Pragma directives
- Plugin info block

### Stage 2: Include Validation
- Valid SourceMod includes only
- No hallucinated includes (`l4d2_boomer_bile`, etc.)

### Stage 3: Semantic Validation
- No invalid functions (`RandomFloat`, `GetEntityModel`, etc.)
- No invalid events (`pounce`, `smoker_tongue_grab`, etc.)
- No invalid properties (`m_flSpeed`, `m_iHealthBuffer`, etc.)
- Correct API patterns

Run validation:
```bash
python scripts/evaluation/validate_generated_code.py \
    --input data/v3_generated \
    --verbose
```

## Training Configuration

### Recommended Settings

```yaml
# configs/unsloth_config.yaml
model:
  base_model: "mistralai/Mistral-7B-Instruct-v0.2"

training:
  epochs: 3
  batch_size: 4
  learning_rate: 2e-4

dataset:
  filter_l4d2: true           # Enable game filtering
  include_anti_patterns: true  # Include negative examples
  min_quality: 0.4            # Quality threshold
```

### Data Preparation

```bash
# Full pipeline with filtering
python scripts/training/prepare_dataset.py \
    --filter-l4d2 \
    --include-anti-patterns \
    --min-quality 0.4

# Check statistics
cat data/processed/dataset_stats.json
```

## Directory Structure

```
data/
├── raw/                    # Scraped data
├── processed/              # Training datasets
│   ├── sourcepawn_train.jsonl
│   ├── sourcepawn_val.jsonl
│   └── dataset_stats.json
├── anti_patterns/          # Negative examples
│   └── l4d2_anti_patterns.jsonl
├── corrected_plugins/      # Reference implementations
└── v3_generated/           # Generated outputs
    └── ISSUE_REPORT.md
```

## Quality Metrics

After implementing game filtering and anti-patterns:

| Metric | Before | After |
|--------|--------|-------|
| Validation Pass Rate | 91.7% (false) | 16.7% (accurate) |
| Cross-Game Contamination | ~60% | <5% |
| Invalid Function Calls | High | Low |
| Invalid Property Access | High | Low |

## Future Improvements

1. **Compile-Time Validation**: Integrate SourceMod compiler for syntax checking
2. **Runtime Testing**: Test generated plugins in actual L4D2 server
3. **API Coverage**: Expand anti-patterns to cover more edge cases
4. **Confidence Scoring**: Weight training examples by detection confidence
