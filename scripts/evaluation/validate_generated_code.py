#!/usr/bin/env python3
"""
Multi-Stage Validation Gauntlet for Generated SourcePawn Code

This script validates generated L4D2 SourcePawn code through three stages:
1. Stage 1: Static Analysis - Check includes, events, basic syntax
2. Stage 2: Compilation Check - Verify with spcomp (if available)
3. Stage 3: Semantic Analysis - Function signatures, heuristic checks

Usage:
    python scripts/evaluation/validate_generated_code.py validate code.sp
    python scripts/evaluation/validate_generated_code.py validate-dir data/generated/
    python scripts/evaluation/validate_generated_code.py report data/generated/
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_path, safe_read_text, safe_write_json

PROJECT_ROOT = Path(__file__).parent.parent.parent

# =============================================================================
# ALLOWLISTS - Valid L4D2/SourceMod constructs
# =============================================================================

VALID_INCLUDES = {
    # Core SourceMod
    "sourcemod", "sdktools", "sdkhooks", "adminmenu", "console", "convars",
    "clients", "entity", "events", "files", "functions", "handles",
    "keyvalues", "logging", "menus", "nextmap", "sorting", "string",
    "textparse", "timers", "topmenus", "usermessages", "vector",

    # L4D2 Specific
    "left4dhooks", "left4downtown", "l4d_stocks", "l4d2_stocks",
    "l4d2_direct", "l4d2util", "l4d2_weapon_stocks",

    # Common extensions
    "colors", "multicolors", "morecolors", "autoexecconfig",
    "clientprefs", "dbi", "regex", "geoip", "tf2", "cstrike",

    # SDK
    "sdktools_client", "sdktools_engine", "sdktools_entinput",
    "sdktools_entoutput", "sdktools_functions", "sdktools_gamerules",
    "sdktools_hooks", "sdktools_sound", "sdktools_stocks",
    "sdktools_stringtables", "sdktools_tempents", "sdktools_trace",
    "sdktools_variant_t", "sdktools_voice",
}

VALID_L4D2_EVENTS = {
    # Player events
    "player_death", "player_hurt", "player_spawn", "player_team",
    "player_disconnect", "player_connect", "player_first_spawn",
    "player_incapacitated", "player_incapacitated_start", "player_ledge_grab",
    "player_ledge_release", "revive_success", "revive_begin", "revive_end",
    "heal_success", "heal_begin", "heal_end", "pills_used", "adrenaline_used",
    "player_now_it", "player_no_longer_it",

    # Special infected events
    "tank_spawn", "tank_killed", "tank_frustrated",
    "witch_spawn", "witch_killed", "witch_harasser_set",
    "tongue_grab", "tongue_release", "tongue_pull_stopped",
    "choke_start", "choke_end", "choke_stopped",
    "lunge_pounce", "pounce_end", "pounce_stopped",
    "jockey_ride", "jockey_ride_end",
    "charger_carry_start", "charger_carry_end", "charger_impact",
    "charger_pummel_start", "charger_pummel_end",
    "spit_burst", "entered_spit",
    "boomer_exploded", "boomer_near", "player_now_it",

    # Game events
    "round_start", "round_end", "map_transition", "finale_start",
    "finale_vehicle_leaving", "finale_vehicle_ready", "finale_escape_start",
    "finale_radio_start", "finale_rush", "finale_win",
    "panic_event_finished", "create_panic_event",
    "scavenge_round_start", "scavenge_round_halftime", "scavenge_round_finished",
    "versus_round_start", "versus_round_end",
    "survival_round_start", "survival_round_end",
    "difficulty_changed", "game_newmap",

    # Item events
    "weapon_fire", "weapon_reload", "weapon_given", "weapon_drop",
    "item_pickup", "ammo_pickup", "upgrade_pack_used",
    "defibrillator_used", "defibrillator_begin",

    # Door/Entity events
    "door_open", "door_close", "door_unlocked",
    "break_breakable", "break_prop", "entity_killed",

    # Checkpoint/Saferoom events
    "player_entered_checkpoint", "player_left_checkpoint",
    "player_entered_start_area", "player_left_start_area",

    # Infected spawning
    "ghost_spawn_time", "spawn_special_infected", "infected_death",
    "zombie_ignited", "zombie_exploded",
}

VALID_FUNCTIONS = {
    # Client functions
    "GetClientCount", "GetMaxClients", "MaxClients", "GetClientName",
    "GetClientUserId", "GetClientOfUserId", "GetClientTeam", "GetClientHealth",
    "IsClientInGame", "IsClientConnected", "IsPlayerAlive", "IsClientObserver",
    "GetClientAbsOrigin", "GetClientAbsAngles", "GetClientEyePosition",
    "GetClientEyeAngles", "GetClientAimTarget", "GetClientWeapon",
    "SetEntityHealth", "SetClientInfo",

    # Entity functions
    "CreateEntityByName", "DispatchSpawn", "DispatchKeyValue", "DispatchKeyValueFloat",
    "DispatchKeyValueVector", "AcceptEntityInput", "RemoveEntity",
    "GetEntProp", "SetEntProp", "GetEntPropFloat", "SetEntPropFloat",
    "GetEntPropEnt", "SetEntPropEnt", "GetEntPropVector", "SetEntPropVector",
    "GetEntPropString", "SetEntPropString", "GetEntDataFloat", "SetEntDataFloat",
    "IsValidEntity", "IsValidEdict", "GetEntityClassname",

    # Event functions
    "HookEvent", "UnhookEvent", "CreateEvent", "FireEvent",

    # Timer functions
    "CreateTimer", "KillTimer", "TriggerTimer", "GetTickedTime",

    # Output functions
    "PrintToChat", "PrintToChatAll", "PrintToConsole", "PrintToServer",
    "PrintHintText", "PrintHintTextToAll", "PrintCenterText", "PrintCenterTextAll",
    "ReplyToCommand", "ShowActivity", "ShowActivity2", "LogMessage",

    # Sound functions
    "EmitSound", "EmitSoundToAll", "EmitSoundToClient", "StopSound",
    "PrecacheSound", "PrecacheModel", "PrecacheGeneric",

    # Vector/Math functions
    "GetVectorDistance", "GetVectorLength", "NormalizeVector", "GetAngleVectors",
    "GetVectorAngles", "MakeVectorFromPoints", "AddVectors", "SubtractVectors",
    "ScaleVector", "NegateVector",

    # String functions
    "Format", "FormatEx", "StrEqual", "StrContains", "StrCat", "strcopy",
    "IntToString", "FloatToString", "StringToInt", "StringToFloat",

    # Menu functions
    "CreateMenu", "DisplayMenu", "AddMenuItem", "SetMenuTitle", "CloseHandle",

    # ConVar functions
    "CreateConVar", "FindConVar", "GetConVarInt", "GetConVarFloat",
    "GetConVarBool", "SetConVarInt", "SetConVarFloat", "SetConVarBool",
    "HookConVarChange",

    # Admin functions
    "CheckCommandAccess", "GetAdminFlag", "GetUserAdmin", "GetUserFlagBits",
    "RegAdminCmd", "RegConsoleCmd", "RegServerCmd",

    # L4D2 specific
    "L4D_GetSurvivorCount", "L4D_GetInfectedCount", "L4D_RespawnPlayer",
    "L4D_CreateRescuableSurvivors", "L4D_GetRandomPZSpawnPosition",
    "L4D2_SpawnTank", "L4D2_SpawnWitch", "L4D2_SpawnSpecial",
    "L4D_GetPlayerZombieClass", "L4D_SetPlayerZombieClass",
    "L4D_IsMissionFinalMap", "L4D_IsFirstMapInScenario",
}

# Known hallucinated/invalid includes
INVALID_INCLUDES = {
    "l4d2_bile", "l4d_tank", "l4d_witch", "l4d_survivor",
    "client", "filesystem", "network", "database",
    "prop_tank", "prop_survivor",
    # Added from code review
    "l4d2_boomer_bile", "l4d2_infected_rides", "l4d2_infected_health",
    "l4d2_zombie_spawner", "l4d2_timed", "l4d2_infected_random",
    "admin-groups", "l4d2",  # l4d2 alone is not valid, should be l4d2_stocks etc
    # Added from GPT-4o-mini output analysis
    "l4d2_infected", "l4d2_infected_boost", "l4d2_survivor_boost",
    "l4d2_tank_boost", "l4d2_witch_boost", "l4d2_boomer",
}

# Non-existent functions that the model hallucinates
INVALID_FUNCTIONS = {
    "RandomFloat", "RandomInt",  # Should be GetRandomFloat, GetRandomInt
    "GetEntityModel",  # Should use GetEntPropString with m_ModelName
    "HookEntity", "UnhookEntity",  # Should be SDKHook, SDKUnhook
    "GetRandomVector",  # Doesn't exist
    "GetGroundHeight",  # Should use TR_TraceRay
    "TraceLine",  # Wrong signature - should use TR_TraceRay
    "GetWorldBounds",  # Doesn't exist
    "IsPlayerBiled",  # Doesn't exist
    "GetBileSource",  # Doesn't exist
    "TakeDamage",  # Should be SDKHooks_TakeDamage
    "RoundFloat",  # Should be RoundToNearest
    "CreateConVarEx",  # Should be CreateConVar
    "SetGlobalFloat",  # Doesn't exist
    "GetMapIndex",  # Not a standard function
    "Square",  # Doesn't exist, use x*x or Pow
    "SpawnRandomZombie",  # Doesn't exist
    "GetRandomSpecialInfected",  # Doesn't exist
    "GetMapName",  # Should be GetCurrentMap
    "IsPlayerTank",  # Should check zombie class
    "OnClientDeath",  # Not a forward
    "OnRoundStart",  # Should be hooked via HookEvent
    "OnRoundEnd",  # Should be hooked via HookEvent
    "OnRescueVehicleCalled",  # Not a forward
    "GetClosestSaferoomDistance",  # Doesn't exist
}

# Invalid events that don't exist in L4D2
INVALID_EVENTS = {
    "pounce",  # Should be lunge_pounce
    "smoker_tongue_grab",  # Should be tongue_grab
    "smoker_tongue_release",  # Should be tongue_release
    "spitter_death",  # Use player_death + class check
    "tank_health_changed",  # Doesn't exist
    "player_rescue_announce",  # Doesn't exist
    "player_pickup",  # Should be item_pickup
    "generator_final_start",  # Should be finale_start
    "player_run_step",  # Doesn't exist
    # Added from GPT-4o-mini output analysis
    "panic_start",  # Should be create_panic_event
    "panic_end",  # Should be panic_event_finished
    "boomer_vomit",  # Should be player_now_it
    "charger_grab",  # Should be charger_carry_start
    "charger_hit",  # Should be charger_impact or charger_carry_start
    "jockey_grab",  # Should be jockey_ride
    "horde_start",  # Should be create_panic_event
    "horde_end",  # Should be panic_event_finished
}

# Invalid entity/prop names
INVALID_PROPERTIES = {
    "m_flSpeed",  # Should be m_flLaggedMovementValue
    "m_bExploded",  # Doesn't exist for boomers
    "m_flLifetime",  # N/A
    "m_flStartTime",  # N/A
    "m_health",  # Should be m_iHealth
    "m_bInvulnerable",  # Doesn't exist in L4D2
    "m_bCloaked",  # Doesn't exist
    "m_bIsSaferoomDoor",  # Doesn't exist
    "m_iHealthBuffer",  # Should be m_healthBuffer
    "m_iHealthBufferCount",  # Doesn't exist
    "m_iItemDefinitionIndex",  # Not in Source Engine
    "m_iDamage",  # Not a weapon property like this
    "m_iArmor",  # Doesn't exist for infected
    "m_iDamageTaken",  # Doesn't exist
    "m_isInSafeRoom",  # Should be m_isInMissionStartArea
    "m_bIsSpecialInfected",  # Doesn't exist
    # Added from GPT-4o-mini output analysis
    "m_flMaxSpeed",  # Should be m_flLaggedMovementValue for L4D2 speed
    "m_flMaxSpeedCrouched",  # Use m_flLaggedMovementValue instead
}

@dataclass
class ValidationResult:
    """Result of validating a single file."""
    file_path: str
    passed: bool = False
    stage1_passed: bool = False
    stage2_passed: bool = False
    stage3_passed: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "passed": self.passed,
            "stage1_passed": self.stage1_passed,
            "stage2_passed": self.stage2_passed,
            "stage3_passed": self.stage3_passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "score": self.score
        }


def extract_includes(code: str) -> List[str]:
    """Extract all #include statements from code."""
    pattern = r'#include\s*[<"]([^>"]+)[>"]'
    return re.findall(pattern, code)


def extract_events(code: str) -> List[str]:
    """Extract all HookEvent event names from code."""
    pattern = r'HookEvent\s*\(\s*"([^"]+)"'
    return re.findall(pattern, code)


def extract_function_calls(code: str) -> List[str]:
    """Extract function calls from code."""
    # Match function calls (word followed by parenthesis)
    pattern = r'\b([A-Z][a-zA-Z0-9_]+)\s*\('
    return re.findall(pattern, code)


def extract_function_definitions(code: str) -> set:
    """Extract locally defined function names from code."""
    # Match function definitions like: void FunctionName(, int FunctionName(, etc.
    patterns = [
        r'(?:public\s+)?(?:void|int|float|bool|char|Handle|Action|any)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(',
        r'(?:public\s+)?stock\s+(?:\w+\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\(',
        r'(?:public\s+)?native\s+(?:\w+\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\(',
    ]
    defined = set()
    for pattern in patterns:
        matches = re.findall(pattern, code)
        defined.update(matches)
    return defined


def check_balanced_braces(code: str) -> Tuple[bool, str]:
    """Check if braces and parentheses are balanced."""
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}

    for i, char in enumerate(code):
        if char in '([{':
            stack.append(char)
        elif char in ')]}':
            if not stack or stack[-1] != pairs[char]:
                return False, f"Unbalanced '{char}' at position {i}"
            stack.pop()

    if stack:
        return False, f"Unclosed '{stack[-1]}'"
    return True, ""


def stage1_static_analysis(code: str, file_path: str) -> ValidationResult:
    """Stage 1: Static analysis - check includes, events, basic syntax."""
    result = ValidationResult(file_path=file_path)

    # Check for basic structure
    if "#pragma" not in code and "#include" not in code:
        result.errors.append("Missing #pragma or #include directives")

    # Check includes
    includes = extract_includes(code)
    for inc in includes:
        # Remove file extension if present
        inc_name = inc.replace(".inc", "").replace(".sp", "")
        if inc_name in INVALID_INCLUDES:
            result.errors.append(f"Invalid/hallucinated include: {inc}")
        elif inc_name not in VALID_INCLUDES:
            result.warnings.append(f"Unknown include (may be valid): {inc}")

    # Check events - STRICT MODE: Unknown L4D2 events are now errors
    # This is critical for ensuring models learn correct L4D2 APIs
    events = extract_events(code)
    for event in events:
        if event in INVALID_EVENTS:
            result.errors.append(f"Invalid/non-existent event: {event}")
        elif event not in VALID_L4D2_EVENTS:
            # Unknown events are errors in strict L4D2 validation
            # Common Source events that may appear in L4D2 plugins are in VALID_L4D2_EVENTS
            result.errors.append(f"Unknown/hallucinated event: {event} (not in L4D2 event list)")

    # Check balanced braces
    balanced, error = check_balanced_braces(code)
    if not balanced:
        result.errors.append(f"Syntax error: {error}")

    # Check for semicolons (basic SourcePawn requirement)
    lines = code.split('\n')
    for i, line in enumerate(lines):
        line = line.strip()
        # Skip comments, preprocessor, empty lines, and block starts/ends
        if (not line or line.startswith('//') or line.startswith('#') or
            line.startswith('/*') or line.startswith('*') or
            line.endswith('{') or line.endswith('}') or
            line == '{' or line == '}'):
            continue
        # Check if statement lines end with semicolon, comma, or are function defs
        if (not line.endswith(';') and not line.endswith(',') and
            not line.endswith(')') and not line.endswith('(') and
            'public' not in line and 'void' not in line and
            'int' not in line and 'float' not in line and 'bool' not in line and
            'char' not in line and 'Handle' not in line and 'Action' not in line):
            # This is a potential issue but not always an error
            pass

    # Score calculation
    error_count = len(result.errors)
    warning_count = len(result.warnings)

    if error_count == 0:
        result.stage1_passed = True
        result.score = max(0, 10 - (warning_count * 0.5))
    else:
        result.score = max(0, 5 - error_count - (warning_count * 0.25))

    return result


def stage2_compilation_check(code: str, result: ValidationResult) -> ValidationResult:
    """Stage 2: Compilation check with spcomp (if available)."""
    # Check if spcomp is available
    spcomp_paths = [
        "/opt/sourcemod/scripting/spcomp",
        "spcomp",
        os.path.expanduser("~/sourcemod/scripting/spcomp"),
    ]

    spcomp = None
    for path in spcomp_paths:
        if os.path.exists(path):
            spcomp = path
            break

    if not spcomp:
        # spcomp not available, assume pass with warning
        result.warnings.append("spcomp not available - compilation check skipped")
        result.stage2_passed = True
        return result

    # Write code to temp file and compile
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sp', delete=False) as f:
            f.write(code)
            temp_path = f.name

        # Run spcomp
        proc = subprocess.run(
            [spcomp, temp_path],
            capture_output=True,
            text=True,
            timeout=30
        )

        if proc.returncode == 0:
            result.stage2_passed = True
            result.score += 2  # Bonus for compilation
        else:
            result.errors.append(f"Compilation failed: {proc.stderr[:500]}")
            result.score -= 2
    except subprocess.TimeoutExpired:
        result.errors.append("Compilation timed out")
    except Exception as e:
        result.warnings.append(f"Compilation check error: {e}")
        result.stage2_passed = True  # Don't fail on internal errors
    finally:
        try:
            os.unlink(temp_path)
            smx_path = temp_path.replace('.sp', '.smx')
            if os.path.exists(smx_path):
                os.unlink(smx_path)
        except:
            pass

    return result


def stage3_semantic_analysis(code: str, result: ValidationResult) -> ValidationResult:
    """Stage 3: Semantic analysis - function signatures, heuristic checks."""

    # Check for common L4D2 patterns
    has_plugin_info = "public Plugin myinfo" in code or "public Plugin:myinfo" in code
    has_on_plugin_start = "OnPluginStart" in code

    if not has_plugin_info:
        result.warnings.append("Missing plugin info block")
    if not has_on_plugin_start:
        result.warnings.append("Missing OnPluginStart function")

    # Check for correct team ID usage
    team_checks = re.findall(r'GetClientTeam\s*\([^)]+\)\s*==\s*(\d+)', code)
    for team_id in team_checks:
        if team_id not in ['1', '2', '3']:  # 1=Spectator, 2=Survivor, 3=Infected
            result.warnings.append(f"Unusual team ID check: {team_id}")

    # Check for proper client validation pattern
    if "GetClientOfUserId" in code:
        # Should have IsClientInGame check nearby
        if "IsClientInGame" not in code:
            result.warnings.append("GetClientOfUserId used without IsClientInGame check")

    # Check for known anti-patterns
    if "FindClass" in code:
        result.errors.append("Invalid function: FindClass (doesn't exist)")
    if "m_nClass" in code and "GetEntData" in code:
        result.warnings.append("Using m_nClass - verify this is correct for L4D2")

    # Check for invalid functions (hallucinations)
    functions_used = extract_function_calls(code)
    locally_defined = extract_function_definitions(code)
    invalid_funcs_found = []
    unknown_functions = []

    for func in functions_used:
        # Skip locally defined functions
        if func in locally_defined:
            continue
        # Check for known invalid functions
        if func in INVALID_FUNCTIONS:
            invalid_funcs_found.append(func)
            continue
        # Skip common patterns that aren't our tracked functions
        if func.startswith("Event_") or func.startswith("Timer_") or func.startswith("Menu_"):
            continue
        if func.startswith("On") or func.startswith("Action_"):
            continue
        if func not in VALID_FUNCTIONS and func not in ["Plugin", "Event", "Handle", "Action", "Timer"]:
            unknown_functions.append(func)

    if invalid_funcs_found:
        unique_invalid = list(set(invalid_funcs_found))[:5]
        result.errors.append(f"Invalid/non-existent functions: {', '.join(unique_invalid)}")

    if unknown_functions:
        unique_unknown = list(set(unknown_functions))[:5]  # Limit to 5
        result.warnings.append(f"Unknown functions (may be valid): {', '.join(unique_unknown)}")

    # Check for invalid properties
    invalid_props_found = []
    for prop in INVALID_PROPERTIES:
        if prop in code:
            invalid_props_found.append(prop)

    if invalid_props_found:
        unique_props = list(set(invalid_props_found))[:5]
        result.errors.append(f"Invalid/non-existent properties: {', '.join(unique_props)}")

    # Score adjustments
    if has_plugin_info:
        result.score += 0.5
    if has_on_plugin_start:
        result.score += 0.5

    # L4D2 SEMANTIC SCORING - Penalize wrong L4D2-specific APIs
    # This catches cases where syntax is correct but L4D2 APIs are wrong
    semantic_violations = check_l4d2_semantic_correctness(code)
    for violation in semantic_violations:
        result.errors.append(f"L4D2 semantic error: {violation}")
        result.score -= 2.0  # Strong penalty for L4D2-specific errors

    # Determine stage 3 pass - check for semantic errors
    stage3_errors = [e for e in result.errors if any(x in e for x in [
        "Invalid function", "Invalid/non-existent functions",
        "Invalid/non-existent properties", "Stage 3", "L4D2 semantic error"
    ])]
    result.stage3_passed = len(stage3_errors) == 0

    return result


def check_l4d2_semantic_correctness(code: str) -> List[str]:
    """
    Check for L4D2-specific semantic violations.
    Returns list of violation messages.

    This catches cases where the code is syntactically valid SourcePawn
    but uses wrong L4D2 APIs that would compile but not work correctly.
    """
    violations = []

    # Pattern: wrong function -> correct function -> message
    L4D2_SEMANTIC_CHECKS = [
        # Random functions (most common failure)
        (r'\bRandomFloat\s*\(', 'GetRandomFloat',
         "Use GetRandomFloat() not RandomFloat()"),
        (r'\bRandomInt\s*\(', 'GetRandomInt',
         "Use GetRandomInt() not RandomInt()"),

        # Wrong events that might pass HookEvent syntax check
        (r'HookEvent\s*\(\s*["\']pounce["\'](?!\s*_)', None,
         "Use 'lunge_pounce' event, not 'pounce' for Hunter"),
        (r'HookEvent\s*\(\s*["\']smoker_tongue_grab["\']', None,
         "Use 'tongue_grab' event, not 'smoker_tongue_grab'"),
        (r'HookEvent\s*\(\s*["\']boomer_vomit["\']', None,
         "Use 'player_now_it' event, not 'boomer_vomit'"),
        (r'HookEvent\s*\(\s*["\']charger_grab["\']', None,
         "Use 'charger_carry_start' event, not 'charger_grab'"),
        (r'HookEvent\s*\(\s*["\']jockey_grab["\']', None,
         "Use 'jockey_ride' event, not 'jockey_grab'"),
        (r'HookEvent\s*\(\s*["\']panic_start["\']', None,
         "Use 'create_panic_event' not 'panic_start'"),
        (r'HookEvent\s*\(\s*["\']panic_end["\']', None,
         "Use 'panic_event_finished' not 'panic_end'"),

        # Wrong properties
        (r'["\']m_flMaxSpeed["\']', 'm_flLaggedMovementValue',
         "Use 'm_flLaggedMovementValue' for L4D2 speed, not 'm_flMaxSpeed'"),
        (r'["\']m_flSpeed["\'](?!Modifier)', 'm_flLaggedMovementValue',
         "Use 'm_flLaggedMovementValue' for L4D2 speed, not 'm_flSpeed'"),

        # Wrong functions
        (r'\bTakeDamage\s*\((?!.*SDKHooks)', 'SDKHooks_TakeDamage',
         "Use SDKHooks_TakeDamage() not TakeDamage()"),
        (r'\bRoundFloat\s*\(', 'RoundToNearest',
         "Use RoundToNearest() not RoundFloat()"),
    ]

    for pattern, correct, message in L4D2_SEMANTIC_CHECKS:
        if re.search(pattern, code):
            violations.append(message)

    return violations


def validate_code(code: str, file_path: str = "code.sp") -> ValidationResult:
    """Run full validation gauntlet on code."""
    # Stage 1
    result = stage1_static_analysis(code, file_path)

    # Only continue if Stage 1 passed
    if result.stage1_passed:
        # Stage 2
        result = stage2_compilation_check(code, result)

    # Stage 3 runs regardless (semantic checks)
    result = stage3_semantic_analysis(code, result)

    # Final pass determination
    result.passed = result.stage1_passed and result.stage3_passed
    result.score = min(10, max(0, result.score))

    return result


def validate_file(file_path: Path) -> ValidationResult:
    """Validate a single SourcePawn file."""
    try:
        code = safe_read_text(file_path, PROJECT_ROOT)
        return validate_code(code, str(file_path))
    except Exception as e:
        result = ValidationResult(file_path=str(file_path))
        result.errors.append(f"Could not read file: {e}")
        return result


def validate_directory(dir_path: Path) -> List[ValidationResult]:
    """Validate all .sp files in a directory."""
    results = []
    sp_files = list(dir_path.glob("*.sp"))

    for sp_file in sp_files:
        result = validate_file(sp_file)
        results.append(result)

    return results


def generate_report(results: List[ValidationResult]) -> dict:
    """Generate a summary report from validation results."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    stage1_passed = sum(1 for r in results if r.stage1_passed)
    stage2_passed = sum(1 for r in results if r.stage2_passed)
    stage3_passed = sum(1 for r in results if r.stage3_passed)

    avg_score = sum(r.score for r in results) / total if total > 0 else 0

    # Collect common errors
    all_errors = []
    all_warnings = []
    for r in results:
        all_errors.extend(r.errors)
        all_warnings.extend(r.warnings)

    # Count error frequencies
    error_counts = {}
    for error in all_errors:
        # Normalize similar errors
        key = error.split(':')[0] if ':' in error else error
        error_counts[key] = error_counts.get(key, 0) + 1

    warning_counts = {}
    for warning in all_warnings:
        key = warning.split(':')[0] if ':' in warning else warning
        warning_counts[key] = warning_counts.get(key, 0) + 1

    return {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_files": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": f"{(passed/total)*100:.1f}%" if total > 0 else "N/A",
            "average_score": round(avg_score, 2),
        },
        "stage_results": {
            "stage1_static": f"{stage1_passed}/{total}",
            "stage2_compilation": f"{stage2_passed}/{total}",
            "stage3_semantic": f"{stage3_passed}/{total}",
        },
        "common_errors": dict(sorted(error_counts.items(), key=lambda x: -x[1])[:10]),
        "common_warnings": dict(sorted(warning_counts.items(), key=lambda x: -x[1])[:10]),
        "file_results": [r.to_dict() for r in results],
    }


def cmd_validate(args):
    """Validate a single file."""
    file_path = safe_path(args.file, PROJECT_ROOT)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    result = validate_file(file_path)

    print(f"\n{'='*60}")
    print(f"Validation Results: {file_path.name}")
    print(f"{'='*60}")
    print(f"Overall: {'PASS' if result.passed else 'FAIL'}")
    print(f"Score: {result.score:.1f}/10")
    print(f"\nStage Results:")
    print(f"  Stage 1 (Static):      {'PASS' if result.stage1_passed else 'FAIL'}")
    print(f"  Stage 2 (Compilation): {'PASS' if result.stage2_passed else 'FAIL'}")
    print(f"  Stage 3 (Semantic):    {'PASS' if result.stage3_passed else 'FAIL'}")

    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for error in result.errors:
            print(f"  - {error}")

    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for warning in result.warnings:
            print(f"  - {warning}")


def cmd_validate_dir(args):
    """Validate all files in a directory."""
    dir_path = safe_path(args.directory, PROJECT_ROOT)
    if not dir_path.exists():
        print(f"Error: Directory not found: {dir_path}")
        sys.exit(1)

    print(f"Validating files in: {dir_path}")
    results = validate_directory(dir_path)

    if not results:
        print("No .sp files found")
        return

    report = generate_report(results)

    print(f"\n{'='*60}")
    print("Validation Summary")
    print(f"{'='*60}")
    print(f"Total files:    {report['summary']['total_files']}")
    print(f"Passed:         {report['summary']['passed']}")
    print(f"Failed:         {report['summary']['failed']}")
    print(f"Pass rate:      {report['summary']['pass_rate']}")
    print(f"Average score:  {report['summary']['average_score']}/10")

    print(f"\nStage Results:")
    print(f"  Stage 1 (Static):      {report['stage_results']['stage1_static']}")
    print(f"  Stage 2 (Compilation): {report['stage_results']['stage2_compilation']}")
    print(f"  Stage 3 (Semantic):    {report['stage_results']['stage3_semantic']}")

    if report['common_errors']:
        print(f"\nMost Common Errors:")
        for error, count in list(report['common_errors'].items())[:5]:
            print(f"  [{count}x] {error}")

    if report['common_warnings']:
        print(f"\nMost Common Warnings:")
        for warning, count in list(report['common_warnings'].items())[:5]:
            print(f"  [{count}x] {warning}")

    # Save report
    if args.output:
        output_path = safe_path(args.output, PROJECT_ROOT, create_parents=True)
        safe_write_json(output_path, report, PROJECT_ROOT)
        print(f"\nReport saved to: {output_path}")


def cmd_report(args):
    """Generate detailed report for a directory."""
    cmd_validate_dir(args)


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Stage Validation Gauntlet for Generated SourcePawn Code',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Validate single file
    val_parser = subparsers.add_parser('validate', help='Validate a single .sp file')
    val_parser.add_argument('file', help='Path to .sp file')
    val_parser.set_defaults(func=cmd_validate)

    # Validate directory
    valdir_parser = subparsers.add_parser('validate-dir', help='Validate all .sp files in directory')
    valdir_parser.add_argument('directory', help='Path to directory')
    valdir_parser.add_argument('--output', '-o', help='Save JSON report to file')
    valdir_parser.set_defaults(func=cmd_validate_dir)

    # Generate report
    report_parser = subparsers.add_parser('report', help='Generate validation report')
    report_parser.add_argument('directory', help='Path to directory')
    report_parser.add_argument('--output', '-o', default='data/validation_report.json',
                              help='Output path for JSON report')
    report_parser.set_defaults(func=cmd_report)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == '__main__':
    main()
