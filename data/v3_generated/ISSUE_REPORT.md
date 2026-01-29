# Generated SourcePawn Plugin Issue Report

**Date:** 2026-01-06
**Total Plugins Reviewed:** 24
**Critical Issues Found:** 23/24 plugins have at least one critical issue

## Executive Summary

The generated SourcePawn plugins exhibit structural plausibility but contain fundamental errors in L4D2/SourceMod API usage. Most plugins would not compile or would not function correctly even if they compiled. The model appears to hallucinate:
- Non-existent includes
- Non-existent functions
- Invalid property names
- Wrong event names
- Incorrect API usage patterns

---

## Issue Categories

### Category A: Non-Existent Includes
Files reference includes that don't exist in SourceMod or L4D2:

| Plugin | Invalid Include |
|--------|-----------------|
| generated_001.sp | `l4d2_boomer_bile` |
| generated_005.sp | `l4d2_infected_rides` |
| generated_010.sp | `l4d2_infected_health` |
| generated_018.sp | `l4d2_zombie_spawner` |
| generated_020.sp | `l4d2_timed` |
| generated_022.sp | `l4d2`, `admin-groups` |
| generated_023.sp | `l4d2_infected_random` |
| generated_024.sp | `l4d2` |

### Category B: Non-Existent Functions
Plugins call functions that don't exist:

| Function | Correct Alternative |
|----------|---------------------|
| `RandomFloat()` | `GetRandomFloat()` |
| `RandomInt()` | `GetRandomInt()` |
| `GetEntityModel()` | `GetEntPropString(entity, Prop_Data, "m_ModelName", ...)` |
| `HookEntity()` | `SDKHook()` |
| `GetRandomVector()` | Manual calculation with `Cosine()`/`Sine()` |
| `GetGroundHeight()` | TR_TraceRay + TR_GetEndPosition |
| `TraceLine()` | TR_TraceRay API |
| `GetWorldBounds()` | No direct equivalent, use nav mesh or hardcode |
| `IsPlayerBiled()` | Check `m_bIsOnThirdStrike` or similar |
| `GetBileSource()` | Not available |
| `TakeDamage()` | `SDKHooks_TakeDamage()` |
| `RoundFloat()` | `RoundToNearest()` |
| `CreateConVarEx()` | `CreateConVar()` |
| `SetGlobalFloat()` | Not available |
| `GetMapIndex()` | Not a standard function |
| `Square()` | Use `x * x` or `Pow(x, 2.0)` |

### Category C: Invalid Events
Plugins hook events that don't exist in L4D2:

| Invalid Event | Correct Alternative |
|---------------|---------------------|
| `pounce` | `lunge_pounce` |
| `smoker_tongue_grab` | `tongue_grab` |
| `smoker_tongue_release` | `tongue_release` |
| `spitter_death` | Use `player_death` + class check |
| `tank_health_changed` | No event - use OnGameFrame polling |
| `player_rescue_announce` | Use `player_incapacitated_start` |
| `player_pickup` | `item_pickup` |
| `generator_final_start` | `finale_start` or `gauntlet_finale_start` |
| `player_run_step` | No event - use OnPlayerRunCmd |

### Category D: Invalid Properties
Plugins access entity properties that don't exist:

| Invalid Property | Correct Alternative |
|-----------------|---------------------|
| `m_flSpeed` | `m_flLaggedMovementValue` (multiplier) |
| `m_bExploded` | No direct property |
| `m_flLifetime` | N/A for env_spray |
| `m_flStartTime` | N/A |
| `m_health` | `m_iHealth` |
| `m_bInvulnerable` | Use `m_takedamage` or SDKHooks |
| `m_bCloaked` | Doesn't exist in L4D2 |
| `m_bIsSaferoomDoor` | Check classname instead |
| `m_iHealthBuffer` | `m_healthBuffer` |
| `m_iItemDefinitionIndex` | Not in Source Engine |
| `m_iDamage` | Weapon damage varies by type |
| `m_iArmor` | Doesn't exist for infected |
| `m_iDamageTaken` | Not tracked this way |
| `m_isInSafeRoom` | `m_isInMissionStartArea` |
| `m_bIsSpecialInfected` | Check zombie class instead |

### Category E: Wrong API Usage Patterns
Fundamental misunderstandings of how SourceMod works:

1. **Event modification doesn't prevent actions**
   ```sourcepawn
   // WRONG - events are informational
   event.SetInt("dmg_health", 0);
   return Plugin_Handled;
   ```

2. **Wrong return types from void functions**
   ```sourcepawn
   public void Event_PlayerHurt(...) {
       return Plugin_Handled; // ERROR: void can't return
   }
   ```

3. **Timer with client ID instead of userid**
   ```sourcepawn
   CreateTimer(2.0, Timer_Callback, client); // Client may disconnect!
   CreateTimer(2.0, Timer_Callback, GetClientUserId(client)); // Correct
   ```

4. **GetClientName without buffer**
   ```sourcepawn
   PrintToChat(client, "%s", GetClientName(i)); // WRONG

   char name[MAX_NAME_LENGTH];
   GetClientName(i, name, sizeof(name)); // Correct
   ```

5. **Event.GetString returns void**
   ```sourcepawn
   char item[64] = event.GetString("item"); // WRONG

   event.GetString("item", item, sizeof(item)); // Correct
   ```

---

## Per-Plugin Issue Summary

| # | Plugin | Compiles | Works | Critical Issues |
|---|--------|----------|-------|-----------------|
| 001 | Boomer Bile Detector | No | No | Invalid include, non-existent functions |
| 002 | Hunter Pounce Sound | Yes | No | Wrong event name, missing precache |
| 003 | Smoker Tongue Tracker | No | No | Invalid include, wrong events |
| 004 | Charger Spawner | No | No | Invalid entity name, non-existent functions |
| 005 | Jockey Ride Damage | No | No | Invalid include, non-existent forward |
| 006 | Spitter Acid Pools | No | No | Completely wrong approach |
| 007 | Tank Health Announce | Yes | No | Non-existent event, incomplete functions |
| 008 | Hunter Rescue God Mode | Yes | No | Non-existent event, invalid properties |
| 009 | Saferoom Heal | Yes | No | Wrong event usage, invalid properties |
| 010 | Medkit Delay | No | No | Invalid include, wrong approach |
| 011 | Gnome Speed | No | No | Non-existent functions |
| 012 | Tank Damage Tracker | No | No | Non-existent forward, invalid properties |
| 013 | Drop Weapon | Yes | Partial | Wrong classname check, kills instead of drops |
| 014 | First Aid Kit Heal | Yes | No | Wrong event, invalid property |
| 015 | Defibrillator Announcer | Yes | Partial | GetClientName needs buffer |
| 016 | Prevent Duplicate Weapons | Yes | No | Non-existent event, wrong approach |
| 017 | Panic Timer | No | No | Wrong SDKCall usage, redefines built-in |
| 018 | Finale Zombie Spawner | No | No | Invalid include, non-existent function |
| 019 | Reduced Saferoom Spawns | No | No | Invalid functions, wrong forward |
| 020 | Chapter Time Announcer | No | No | Invalid include, wrong event signatures |
| 021 | Rescue Vehicle Tracker | Yes | No | Non-existent forwards, events not hooked |
| 022 | Saferoom Victory Sound | No | No | Invalid includes, wrong saferoom logic |
| 023 | Random SI on Generator | No | No | Multiple invalid functions, padded switch |
| 024 | Map Explored Percentage | No | No | Invalid include, redefines built-in |

**Summary:** 7/24 would compile, 0/24 would work fully as intended

---

## Recommendations

### For Training Data
1. Add negative examples showing incorrect patterns
2. Include more real L4D2 plugin examples
3. Add explicit API documentation to training context

### For Validation
1. Expand invalid includes list
2. Add non-existent function detection
3. Add property name validation
4. Add event name validation
5. Check function call patterns

### For Model Improvement
1. Reduce hallucination of plausible-sounding includes
2. Improve understanding of SourceMod API
3. Better distinction between CS:GO and L4D2 APIs
4. Understand that events are read-only

---

## Corrected Plugin Status

The following plugins have been corrected in `data/corrected_plugins/`:
- [x] SpeedBoostOnKill.sp
- [x] NoFriendlyFireDuringPanic.sp
- [x] AutoRevive.sp
- [x] ZombieKillTracker.sp
- [x] WitchSpawner.sp

Remaining plugins in v3_generated need review and correction.
