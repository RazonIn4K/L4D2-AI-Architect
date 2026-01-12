# L4D2 SourcePawn Fine-Tuned Model Evaluation Report

**Date**: January 6, 2026
**Model**: `ft:gpt-4o-mini-2024-07-18:highencodelearning:l4d2-sourcemod:CusA2jFo`
**Base Model**: GPT-4o-mini
**Training**: 921 samples, 50 eval samples, 3 epochs
**Final Train Loss**: 0.021 | **Validation Loss**: 0.195

---

## Executive Summary

The fine-tuned GPT-4o-mini model shows **significant issues with task understanding**. While it learned SourcePawn syntax patterns, it fails to generate contextually appropriate code for the given prompts. The model appears to memorize and reproduce training data snippets rather than understanding the underlying task semantics.

**Overall Score: 1/15 (7%) task-appropriate responses**

---

## Evaluation Dataset

15 test prompts covering common L4D2 SourceMod plugin tasks:

| # | Prompt | Category |
|---|--------|----------|
| 1 | Tank spawn detection + announcement | Event Hook |
| 2 | Speed boost on special infected kill | Game Mechanic |
| 3 | Auto-revive incapacitated players | Player State |
| 4 | Teleport survivors to saferoom | Admin Command |
| 5 | Spawn Witch at nav mesh | Entity Spawn |
| 6 | Damage leaderboard tracking | Statistics |
| 7 | Weapon pickup hook | Event Hook |
| 8 | Tank health scaling by player count | Game Balance |
| 9 | Prevent friendly fire | Damage Mod |
| 10 | Timer-based horde spawns | Timer/Event |
| 11 | Detect all survivors in saferoom | Game State |
| 12 | Headshot bonus points | Statistics |
| 13 | VScript panic event | VScript |
| 14 | Save/restore player inventory | Persistence |
| 15 | Announce special infected spawns | Notifications |

---

## Detailed Results

### Result Categories

| Category | Count | Percentage |
|----------|-------|------------|
| Task-Appropriate Code | 0 | 0% |
| Correct Format, Wrong Task | 2 | 13% |
| Unrelated Code Snippets | 11 | 73% |
| Documentation/Comments Only | 1 | 7% |
| Error Response | 1 | 7% |

### Per-Prompt Analysis

#### 1. Tank Spawn Detection
**Expected**: Event hook for `tank_spawn` event, `PrintToChatAll` announcement
**Actual**: Internal method documentation for `SetAbsBoxMaximums`
**Rating**: FAIL - Completely unrelated

#### 2. Speed Boost Plugin
**Expected**: Complete plugin with `player_death` hook, `SetEntPropFloat` for speed
**Actual**: `TakeItem_Scripted` function with logging
**Rating**: FAIL - Unrelated snippet

#### 3. Auto-Revive Function
**Expected**: Function checking `IsPlayerIncapped`, calling revive logic
**Actual**: Config file constants for scavenge mode
**Rating**: FAIL - Unrelated snippet

#### 4. Teleport Command
**Expected**: Admin command with `RegAdminCmd`, `TeleportEntity`
**Actual**: `ZoomFromC1ToC2` incident handler code
**Rating**: FAIL - Unrelated snippet

#### 5. Witch Spawn at Nav Mesh
**Expected**: Nav mesh query, `SpawnSpecialAtPosition` or equivalent
**Actual**: Documentation about `SetModelAndSkin` method
**Rating**: FAIL - Documentation instead of code

#### 6. Damage Leaderboard
**Expected**: `OnTakeDamage` hook, tracking arrays, display function
**Actual**: Has correct `#pragma semicolon 1` structure but code is for "DZ L4D Locks V4" plugin - completely unrelated content that degenerates into gibberish
**Rating**: PARTIAL - Correct format, wrong task, corrupted output

#### 7. Weapon Pickup Hook
**Expected**: `SDKHook_WeaponEquip` or similar hook
**Actual**: "No TextInput" plugin with menu handling
**Rating**: FAIL - Unrelated plugin

#### 8. Tank Health Scaling
**Expected**: `OnEntityCreated` for Tank, `SetEntityHealth` based on player count
**Actual**: Documentation about entity input triggering
**Rating**: FAIL - Documentation instead of code

#### 9. Friendly Fire Prevention
**Expected**: `OnTakeDamage` hook, team check, damage blocking
**Actual**: Documentation about admin commands and `sm_wipeallitems`
**Rating**: FAIL - Unrelated documentation

#### 10. Horde Spawn Timer
**Expected**: `CreateTimer` with 60.0 interval, `CreateInfected` call
**Actual**: Simple `SetOwner` function (8 lines)
**Rating**: FAIL - Completely unrelated

#### 11. Saferoom Detection
**Expected**: Loop survivors, check trigger_finale or saferoom bounds
**Actual**: `SharedItemSingleton` setup with lobby/rules config
**Rating**: FAIL - Unrelated configuration code

#### 12. Headshot Bonus Points
**Expected**: `OnTakeDamage` or death event, hitgroup check for head
**Actual**: SourceMod base commands header (proper license block) but Give_Command implementation that degenerates into corrupted text
**Rating**: PARTIAL - Correct structure, wrong task, corrupted ending

#### 13. VScript Panic Event
**Expected**: VScript/Squirrel function with `Director.PlayMegaMobWarnSound()` or similar
**Actual**: C++ documentation about `UtlVector::RemoveAll`
**Rating**: FAIL - Wrong language entirely

#### 14. Inventory Save/Restore
**Expected**: `OnMapEnd`/`OnMapStart` hooks, KeyValues or SQL storage
**Actual**: SDK natives for tracking object positions
**Rating**: FAIL - Unrelated SDK code

#### 15. Special Infected Spawn Announcement
**Expected**: `OnEntityCreated` hook, distance calculation, chat message
**Actual**: JSON error: `{"success":false,"error":"item insufficient inputs"}`
**Rating**: ERROR - Model returned invalid response

---

## Root Cause Analysis

### 1. Training Data Quality Issues
- Training data likely contained too many **documentation snippets** and **incomplete code fragments**
- Model learned to reproduce memorized patterns rather than understand task semantics
- Data may have included debug output, comments, and non-code content

### 2. Insufficient Task Diversity
- The 921 training samples may not cover enough diverse task types
- Model hasn't learned the mapping between natural language requests and code patterns

### 3. Prompt Template Mismatch
- Training used specific instruction patterns that evaluation prompts didn't match
- System prompt may be too generic

### 4. Overfitting to Training Data
- Low training loss (0.021) vs higher validation loss (0.195) suggests overfitting
- Model memorizes training examples rather than generalizing

---

## Recommendations

### Immediate Actions

1. **Curate Training Data**
   - Remove documentation snippets, keep only complete working plugins
   - Filter for code that compiles successfully
   - Ensure each example has clear prompt-response mapping

2. **Increase Training Examples**
   - Target 2,000+ high-quality examples
   - Add synthetic examples for common patterns (hooks, timers, commands)

3. **Improve Prompt Engineering**
   - Make evaluation prompts match training format exactly
   - Add few-shot examples in system prompt

### Long-term Improvements

1. **Use Larger Base Model**
   - GPT-4o or GPT-4 Turbo may generalize better
   - Consider Claude fine-tuning when available

2. **Add Retrieval Augmentation**
   - RAG with SourceMod API documentation
   - Reference existing working plugins

3. **Multi-stage Training**
   - First: general code understanding
   - Second: SourcePawn-specific patterns
   - Third: L4D2-specific APIs

---

## Comparison: Fine-tuned vs Base Model

| Metric | Fine-tuned GPT-4o-mini | Base GPT-4o-mini |
|--------|------------------------|------------------|
| SourcePawn Syntax | Knows format | Needs examples |
| Task Understanding | Poor (7%) | Better with few-shot |
| API Accuracy | Hallucinated APIs | Admits uncertainty |
| Output Quality | Memorized snippets | Generic but relevant |

**Conclusion**: The fine-tuned model is currently **worse** than the base model with proper prompting. The base model with few-shot examples would likely outperform this fine-tuned version.

---

## Files

- **Evaluation Dataset**: `data/eval_test_cases.jsonl`
- **OpenAI Dataset ID**: `dset_695cceab14d88190877e67736a458d8d05bca16bfe2d9be4`
- **CSV Export**: `.playwright-mcp/L4D2-SourcePawn-Evaluation.csv`

---

## Next Steps

1. [ ] Clean and curate training dataset (remove docs, keep working code)
2. [ ] Test base GPT-4o-mini with few-shot prompting as baseline
3. [ ] Re-fine-tune with improved dataset
4. [ ] Compare TinyLlama LoRA results against OpenAI model
