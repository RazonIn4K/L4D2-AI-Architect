#!/usr/bin/env python3
"""
Game Detection Utilities for SourcePawn Code

Identifies which Source game (L4D2, CS:GO, TF2, etc.) a SourcePawn plugin targets
based on includes, events, properties, functions, and other code patterns.

This is critical for preventing cross-game contamination in training data.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class GameType(Enum):
    """Supported Source game types."""
    L4D2 = "l4d2"
    L4D = "l4d"  # Left 4 Dead 1
    CSGO = "csgo"
    CSS = "css"  # Counter-Strike Source
    TF2 = "tf2"
    DODS = "dods"  # Day of Defeat Source
    HL2DM = "hl2dm"  # Half-Life 2 Deathmatch
    GENERIC = "generic"  # Generic SourceMod (all games)
    UNKNOWN = "unknown"


class ConfidenceLevel(Enum):
    """Confidence in game detection."""
    CERTAIN = "certain"      # 3+ strong signals, unambiguous
    HIGH = "high"            # 2+ signals or 1 definitive
    MEDIUM = "medium"        # 1-2 weak signals
    LOW = "low"              # 1 weak signal
    NONE = "none"            # No game-specific signals


@dataclass
class DetectionSignal:
    """A single signal indicating a specific game."""
    game: GameType
    category: str           # "include", "event", "property", "function", "entity", "comment"
    pattern: str            # The matched pattern
    weight: float           # Signal strength (0.0-1.0)
    line_number: Optional[int] = None
    line_context: Optional[str] = None


@dataclass
class GameDetectionResult:
    """Result of game detection analysis."""
    detected_game: GameType = GameType.UNKNOWN
    confidence: ConfidenceLevel = ConfidenceLevel.NONE
    signals: List[DetectionSignal] = field(default_factory=list)
    game_scores: Dict[GameType, float] = field(default_factory=dict)
    is_l4d2_compatible: bool = False

    def add_signal(self, signal: DetectionSignal):
        """Add a detection signal and update scores."""
        self.signals.append(signal)
        if signal.game not in self.game_scores:
            self.game_scores[signal.game] = 0.0
        self.game_scores[signal.game] += signal.weight
        self._recalculate()

    def _recalculate(self):
        """Recalculate detected game and confidence."""
        if not self.game_scores:
            return

        # Find highest scoring game
        best_game = max(self.game_scores.keys(), key=lambda g: self.game_scores[g])
        best_score = self.game_scores[best_game]

        # Determine confidence
        if best_score >= 3.0:
            self.confidence = ConfidenceLevel.CERTAIN
        elif best_score >= 2.0:
            self.confidence = ConfidenceLevel.HIGH
        elif best_score >= 1.0:
            self.confidence = ConfidenceLevel.MEDIUM
        elif best_score > 0:
            self.confidence = ConfidenceLevel.LOW
        else:
            self.confidence = ConfidenceLevel.NONE

        self.detected_game = best_game

        # Check L4D2 compatibility
        l4d2_score = self.game_scores.get(GameType.L4D2, 0) + self.game_scores.get(GameType.L4D, 0)
        other_scores = sum(s for g, s in self.game_scores.items()
                         if g not in (GameType.L4D2, GameType.L4D, GameType.GENERIC, GameType.UNKNOWN))

        # L4D2 compatible if L4D/L4D2 signals dominate or it's generic
        self.is_l4d2_compatible = (
            l4d2_score > other_scores or
            (best_game == GameType.GENERIC and other_scores == 0) or
            best_game in (GameType.L4D2, GameType.L4D)
        )


# =============================================================================
# Game Detection Patterns
# =============================================================================

# Include patterns (high confidence signals)
INCLUDE_PATTERNS: Dict[GameType, Set[str]] = {
    GameType.L4D2: {
        "left4dhooks", "left4downtown", "l4d2_stocks", "l4d2util", "l4d_stocks",
        "l4d2_direct", "l4d2_saferoom_detect", "l4d2_behavior", "l4d2_weapon_stocks",
        "l4d_lib", "l4d2lib", "l4dstocks", "colors_l4d",
    },
    GameType.L4D: {
        "left4dhooks_l4d1", "l4d_stocks",
    },
    GameType.CSGO: {
        "cstrike", "csgo", "csgocolors", "csgoitems", "csgogamestats",
        "csgo_colors", "csweapons",
    },
    GameType.CSS: {
        "cstrike", "csscolors",
    },
    GameType.TF2: {
        "tf2", "tf2_stocks", "tf2items", "tf2attributes", "tf2_morestocks",
        "tf2utils", "tf2itemsinfo", "tf2econ",
    },
    GameType.DODS: {
        "dod", "dods",
    },
}

# Event patterns (very high confidence)
EVENT_PATTERNS: Dict[GameType, Set[str]] = {
    GameType.L4D2: {
        # Infected events
        "player_incapacitated", "player_incapacitated_start", "revive_success", "revive_begin",
        "tongue_grab", "tongue_release", "lunge_pounce", "pounce_end", "pounce_stopped",
        "charger_charge_start", "charger_charge_end", "charger_carry_start", "charger_carry_end",
        "charger_pummel_start", "charger_pummel_end", "charger_impact",
        "jockey_ride", "jockey_ride_end",
        "witch_harasser_set", "witch_spawn", "witch_killed",
        "tank_spawn", "tank_killed", "tank_frustrated",
        # Gameplay events
        "finale_start", "finale_escape_start", "finale_vehicle_ready", "finale_vehicle_leaving",
        "gauntlet_finale_start", "finale_radio_start",
        "infected_death", "infected_hurt", "zombie_ignited",
        "heal_success", "heal_begin", "heal_interrupted",
        "defibrillator_used", "defibrillator_begin", "defibrillator_interrupted",
        "pills_used", "adrenaline_used",
        "weapon_given", "weapon_drop",
        "rescue_door_open", "door_close", "player_entered_checkpoint", "player_left_checkpoint",
        "player_bot_replace", "bot_player_replace",
        "survivor_rescued", "survivor_call_for_help",
        "create_panic_event", "panic_event_finished",
        "scavenge_round_start", "scavenge_round_halftime", "scavenge_round_finished",
        "versus_round_start", "versus_match_finished",
        "gascan_pour_completed", "gascan_dropped",
    },
    GameType.CSGO: {
        "bomb_planted", "bomb_defused", "bomb_exploded", "bomb_begindefuse", "bomb_abortdefuse",
        "bomb_pickup", "bomb_dropped", "bomb_beginplant", "bomb_abortplant",
        "hostage_rescued", "hostage_killed", "hostage_follows", "hostage_stops_following",
        "round_freeze_end", "round_prestart", "round_poststart",
        "cs_win_panel_match", "cs_win_panel_round", "cs_game_disconnected",
        "cs_match_end_restart", "cs_pre_restart",
        "silencer_on", "silencer_off", "buymenu_open", "buymenu_close",
        "player_given_c4", "player_dropped_c4", "gg_leader",
    },
    GameType.TF2: {
        "arena_round_start", "arena_win_panel", "arena_match_maxstreak",
        "mvm_wave_complete", "mvm_wave_failed", "mvm_mission_complete",
        "mvm_creditbonus_wave", "mvm_pickup_currency",
        "pumpkin_lord_summoned", "pumpkin_lord_killed", "eyeball_boss_summoned", "eyeball_boss_killed",
        "merasmus_summoned", "merasmus_killed", "merasmus_escape_warning",
        "player_jarated", "player_jarated_fade", "player_extinguished",
        "player_chargedeployed", "player_teleported",
        "payload_pushed", "training_complete",
        "controlpoint_starttouch", "controlpoint_endtouch",
        "teamplay_point_captured", "teamplay_point_locked", "teamplay_point_unlocked",
        "teamplay_flag_event", "ctf_flag_captured",
    },
}

# Entity property patterns (high confidence)
PROPERTY_PATTERNS: Dict[GameType, Set[str]] = {
    GameType.L4D2: {
        # Survivor properties
        "m_isIncapacitated", "m_currentReviveCount", "m_bIsOnThirdStrike", "m_isGoingToDie",
        "m_healthBuffer", "m_healthBufferTime",
        "m_flLaggedMovementValue",  # This IS the correct L4D2 speed property
        "m_iShovePenalty", "m_fNextShoveTime",
        "m_bAdrenalineActive", "m_fAdrenalineTime",
        # Infected properties
        "m_zombieClass", "m_isGhost", "m_ghostSpawnState", "m_ghostSpawnClockMaxDelay",
        "m_tongueVictim", "m_tongueOwner", "m_pounceVictim", "m_pounceAttacker",
        "m_carryVictim", "m_carryAttacker", "m_pummelVictim", "m_pummelAttacker",
        "m_jockeyVictim", "m_jockeyAttacker",
        # Witch properties
        "m_rage", "m_wanderrage",
        # Tank properties
        "m_frustration", "m_nSequence",
        # Weapon properties
        "m_iClip1", "m_upgradeBitVec", "m_nUpgradedPrimaryAmmoLoaded",
        # Mission tracking
        "m_isInMissionStartArea", "m_checkpointSavedHealth",
    },
    GameType.CSGO: {
        "m_ArmorValue", "m_bHasHelmet", "m_bHasDefuser",
        "m_iCompetitiveRanking", "m_iCompetitiveWins", "m_iCompetitiveRankType",
        "m_nActiveCoinRank", "m_nMusicID",
        "m_iAccount", "m_bInBuyZone", "m_bInBombZone",
        "m_flFlashDuration", "m_flFlashMaxAlpha",
        "m_bIsDefusing", "m_bIsGrabbingHostage",
        "m_iPing", "m_iKills", "m_iAssists", "m_iDeaths", "m_iScore",
        "m_bBombPlanted", "m_flC4Blow", "m_bBombDefused",
    },
    GameType.TF2: {
        "m_iClass", "m_iDesiredPlayerClass", "m_PlayerClass",
        "m_nDisguiseClass", "m_nDisguiseTeam", "m_iDisguiseHealth",
        "m_flCloakMeter", "m_flInvisChangeCompleteTime",
        "m_flRageMeter", "m_bRageDraining",
        "m_flChargeLevel", "m_flChargeMeter",
        "m_nMetalRegenCount", "m_iMetal",
        "m_bCarryingObject", "m_iObjectType", "m_iObjectMode",
        "m_nCurrency", "m_Shared",
    },
}

# Function call patterns (high confidence)
FUNCTION_PATTERNS: Dict[GameType, Set[str]] = {
    GameType.L4D2: {
        # Left4DHooks natives
        "L4D_GetSurvivorVictim", "L4D_GetInfectedAttacker", "L4D_GetPinnedSurvivor",
        "L4D_SetHumanSpec", "L4D_TakeOverBot", "L4D_ChangeClientTeam",
        "L4D_GetTeamScore", "L4D_RestartScenarioFromVote",
        "L4D_GetPlayerSpawnTime", "L4D_SetPlayerSpawnTime",
        "L4D_RespawnPlayer", "L4D_ReplaceTank", "L4D_ForcePanicEvent",
        "L4D_Deafen", "L4D_Dissolve", "L4D_OnITExpired", "L4D_AngularVelocity",
        "L4D_StaggerPlayer", "L4D_GetRandomPZSpawnPosition",
        "L4D_GetNearestNavArea", "L4D_FindRandomSpot",
        "L4D_IsInLastCheckpoint", "L4D_IsFirstMapInScenario",
        "L4D_HasAnySurvivorLeftSafeArea", "L4D_IsAnySurvivorInStartArea",
        "L4D_GetVersusMaxCompletionScore", "L4D_SetVersusMaxCompletionScore",
        # L4D2 specific
        "L4D2_GetCurrentWeaponId", "L4D2_GetIntWeaponAttribute", "L4D2_GetFloatWeaponAttribute",
        "L4D2_SetIntMeleeAttribute", "L4D2_GetIntMeleeAttribute",
        "L4D2_IsValidWeapon", "L4D2_GetMeleeWeaponIndex",
        "L4D2_GetVictimCarry", "L4D2_GetVictimPounce", "L4D2_GetVictimRide",
        "L4D2_SpitterPrj", "L4D2_GetInfectedSpawnTime",
        "L4D2Direct_GetTankPassedCount", "L4D2Direct_GetVSCampaignScore",
        "L4D2Direct_GetPendingMobCount", "L4D2Direct_SetPendingMobCount",
        "L4D2_GetScriptScope", "L4D2_RunScript", "L4D2_DirectorScript",
        # Common L4D2 patterns
        "GetZombieClass", "IsInfectedGhost", "IsSurvivor", "IsInfected",
        "GetSurvivorTempHealth", "SetSurvivorTempHealth",
        "IsIncapacitated", "IsHangingFromLedge", "IsGettingUp",
    },
    GameType.CSGO: {
        "CS_GetWeaponPrice", "CS_GetClientClanTag", "CS_SetClientClanTag",
        "CS_GetClientContributionScore", "CS_SetClientContributionScore",
        "CS_GetClientAssists", "CS_SetClientAssists",
        "CS_GetTeamScore", "CS_SetTeamScore",
        "CS_GetMVPCount", "CS_SetMVPCount",
        "CS_TerminateRound", "CS_RespawnPlayer", "CS_SwitchTeam",
        "CS_DropWeapon", "CS_UpdateClientModel",
        "GivePlayerItem", "GetPlayerWeaponSlot",
    },
    GameType.TF2: {
        "TF2_GetClass", "TF2_SetClass", "TF2_GetPlayerClass",
        "TF2_IsPlayerInCondition", "TF2_AddCondition", "TF2_RemoveCondition",
        "TF2_StunPlayer", "TF2_MakeBleed", "TF2_Burn", "TF2_IgnitePlayer",
        "TF2_DisguisePlayer", "TF2_RemovePlayerDisguise",
        "TF2_RespawnPlayer", "TF2_RegeneratePlayer",
        "TF2_RemoveAllWeapons", "TF2_RemoveWeaponSlot",
        "TF2_GetResourceEntity", "TF2_SetPlayerPowerPlay",
        "TF2Attrib_SetByName", "TF2Attrib_SetByDefIndex", "TF2Attrib_RemoveByName",
        "TF2Items_CreateItem", "TF2Items_GiveNamedItem",
    },
}

# Entity classname patterns
ENTITY_PATTERNS: Dict[GameType, Set[str]] = {
    GameType.L4D2: {
        # Infected
        "infected", "witch", "tank", "boomer", "hunter", "smoker",
        "charger", "jockey", "spitter",
        # Special entities
        "infected_wanderer", "infected_standing",
        "prop_door_rotating_checkpoint", "prop_health_cabinet",
        "weapon_spawn", "weapon_melee_spawn", "weapon_item_spawn",
        "weapon_rifle", "weapon_rifle_ak47", "weapon_rifle_desert", "weapon_rifle_sg552",
        "weapon_smg", "weapon_smg_silenced", "weapon_smg_mp5",
        "weapon_shotgun_chrome", "weapon_shotgun_spas", "weapon_pumpshotgun", "weapon_autoshotgun",
        "weapon_hunting_rifle", "weapon_sniper_military", "weapon_sniper_awp", "weapon_sniper_scout",
        "weapon_grenade_launcher", "weapon_rifle_m60",
        "weapon_pistol", "weapon_pistol_magnum",
        "weapon_chainsaw", "weapon_melee",
        "weapon_molotov", "weapon_pipe_bomb", "weapon_vomitjar",
        "weapon_first_aid_kit", "weapon_defibrillator", "weapon_pain_pills", "weapon_adrenaline",
        "weapon_upgradepack_incendiary", "weapon_upgradepack_explosive",
        "upgrade_ammo_incendiary", "upgrade_ammo_explosive", "upgrade_laser_sight",
        "prop_fuel_barrel", "prop_minigun", "prop_minigun_l4d1", "prop_mounted_machine_gun",
        "info_survivor_position", "info_survivor_rescue",
        "point_viewcontrol_survivor", "terror_player_manager",
        "env_outtro_stats", "env_player_blocker",
    },
    GameType.CSGO: {
        "hostage_entity", "func_hostage_rescue", "func_bomb_target",
        "planted_c4", "weapon_c4",
        "weapon_ak47", "weapon_m4a1", "weapon_awp", "weapon_deagle",
        "weapon_hegrenade", "weapon_flashbang", "weapon_smokegrenade",
        "weapon_molotov", "weapon_incgrenade", "weapon_decoy",
        "weapon_taser", "weapon_knife_t", "weapon_knife_ct",
        "weapon_famas", "weapon_galilar", "weapon_aug", "weapon_sg556",
    },
    GameType.TF2: {
        "tf_player_manager", "tf_gamerules", "tf_objective_resource",
        "obj_sentrygun", "obj_dispenser", "obj_teleporter",
        "tf_projectile_rocket", "tf_projectile_pipe", "tf_projectile_arrow",
        "item_healthkit_small", "item_healthkit_medium", "item_healthkit_full",
        "item_ammopack_small", "item_ammopack_medium", "item_ammopack_full",
        "headless_hatman", "eyeball_boss", "merasmus", "tf_zombie",
    },
}

# Comment/string patterns for game identification
COMMENT_PATTERNS: Dict[GameType, List[str]] = {
    GameType.L4D2: [
        r"left\s*4\s*dead\s*2", r"l4d2", r"left4dead2",
        r"survivor", r"infected", r"special\s+infected",
        r"tank", r"witch", r"boomer", r"hunter", r"smoker", r"charger", r"jockey", r"spitter",
        r"versus\s+mode", r"survival\s+mode", r"scavenge",
        r"safe\s*room", r"rescue\s+vehicle",
    ],
    GameType.CSGO: [
        r"counter[\-\s]*strike", r"cs:?\s*go", r"csgo",
        r"terrorist", r"counter[\-\s]*terrorist", r"ct\b", r"\bt\b",
        r"bomb\s+site", r"defuse", r"hostage",
        r"competitive", r"deathmatch", r"casual",
    ],
    GameType.TF2: [
        r"team\s+fortress\s*2?", r"tf2",
        r"engineer", r"medic", r"heavy", r"scout", r"soldier", r"demoman", r"sniper", r"spy", r"pyro",
        r"sentry", r"dispenser", r"teleporter",
        r"mann\s+vs\s+machine", r"mvm",
        r"payload", r"control\s+point",
    ],
}


def detect_game(code: str) -> GameDetectionResult:
    """
    Detect which Source game a SourcePawn plugin targets.

    Args:
        code: SourcePawn source code

    Returns:
        GameDetectionResult with detected game, confidence, and signals
    """
    result = GameDetectionResult()
    lines = code.split('\n')

    # Check includes (weight: 1.0)
    include_pattern = re.compile(r'#include\s*<([^>]+)>')
    for i, line in enumerate(lines, 1):
        match = include_pattern.search(line)
        if match:
            include_name = match.group(1).lower().strip()
            for game, includes in INCLUDE_PATTERNS.items():
                for inc in includes:
                    if inc.lower() in include_name:
                        result.add_signal(DetectionSignal(
                            game=game,
                            category="include",
                            pattern=include_name,
                            weight=1.0,
                            line_number=i,
                            line_context=line.strip()
                        ))
                        break

    # Check events (weight: 0.8)
    event_pattern = re.compile(r'HookEvent\s*\(\s*"([^"]+)"')
    for i, line in enumerate(lines, 1):
        match = event_pattern.search(line)
        if match:
            event_name = match.group(1)
            for game, events in EVENT_PATTERNS.items():
                if event_name in events:
                    result.add_signal(DetectionSignal(
                        game=game,
                        category="event",
                        pattern=event_name,
                        weight=0.8,
                        line_number=i,
                        line_context=line.strip()
                    ))
                    break

    # Check properties (weight: 0.7)
    prop_pattern = re.compile(r'["\']([a-zA-Z_][a-zA-Z0-9_]*)["\']')
    for i, line in enumerate(lines, 1):
        if 'GetEntProp' in line or 'SetEntProp' in line or 'Prop_Send' in line or 'Prop_Data' in line:
            for match in prop_pattern.finditer(line):
                prop_name = match.group(1)
                if prop_name.startswith('m_'):
                    for game, props in PROPERTY_PATTERNS.items():
                        if prop_name in props:
                            result.add_signal(DetectionSignal(
                                game=game,
                                category="property",
                                pattern=prop_name,
                                weight=0.7,
                                line_number=i,
                                line_context=line.strip()
                            ))
                            break

    # Check function calls (weight: 0.8)
    for i, line in enumerate(lines, 1):
        for game, functions in FUNCTION_PATTERNS.items():
            for func in functions:
                if re.search(rf'\b{re.escape(func)}\s*\(', line):
                    result.add_signal(DetectionSignal(
                        game=game,
                        category="function",
                        pattern=func,
                        weight=0.8,
                        line_number=i,
                        line_context=line.strip()
                    ))

    # Check entity classnames (weight: 0.6)
    entity_pattern = re.compile(r'["\']([a-zA-Z_][a-zA-Z0-9_]*)["\']')
    for i, line in enumerate(lines, 1):
        if 'CreateEntity' in line or 'classname' in line.lower() or 'GetEdictClassname' in line:
            for match in entity_pattern.finditer(line):
                entity_name = match.group(1)
                for game, entities in ENTITY_PATTERNS.items():
                    if entity_name in entities:
                        result.add_signal(DetectionSignal(
                            game=game,
                            category="entity",
                            pattern=entity_name,
                            weight=0.6,
                            line_number=i,
                            line_context=line.strip()
                        ))
                        break

    # Check comments and strings (weight: 0.3)
    for i, line in enumerate(lines, 1):
        line_lower = line.lower()
        for game, patterns in COMMENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, line_lower, re.IGNORECASE):
                    result.add_signal(DetectionSignal(
                        game=game,
                        category="comment",
                        pattern=pattern,
                        weight=0.3,
                        line_number=i,
                        line_context=line.strip()[:80]
                    ))
                    break

    # If no signals found, mark as generic SourceMod
    if not result.signals:
        result.detected_game = GameType.GENERIC
        result.confidence = ConfidenceLevel.LOW
        result.is_l4d2_compatible = True  # Generic code might work

    return result


def is_l4d2_code(code: str, min_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM) -> Tuple[bool, GameDetectionResult]:
    """
    Check if code is L4D2-specific or L4D2-compatible.

    Args:
        code: SourcePawn source code
        min_confidence: Minimum confidence level required

    Returns:
        Tuple of (is_l4d2_compatible, detection_result)
    """
    result = detect_game(code)

    # Confidence hierarchy for comparison
    confidence_order = {
        ConfidenceLevel.NONE: 0,
        ConfidenceLevel.LOW: 1,
        ConfidenceLevel.MEDIUM: 2,
        ConfidenceLevel.HIGH: 3,
        ConfidenceLevel.CERTAIN: 4,
    }

    # Check if detected game is L4D2 or L4D with sufficient confidence
    if result.detected_game in (GameType.L4D2, GameType.L4D):
        if confidence_order[result.confidence] >= confidence_order[min_confidence]:
            return True, result

    # Check if generic (no game-specific code) - might be usable
    if result.detected_game == GameType.GENERIC:
        return True, result

    # Not L4D2 compatible
    return False, result


def filter_l4d2_training_data(examples: List[Dict],
                               min_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM) -> Tuple[List[Dict], List[Dict]]:
    """
    Filter training examples to only include L4D2-compatible code.

    Args:
        examples: List of training examples with 'code' or 'content' field
        min_confidence: Minimum confidence for L4D2 detection

    Returns:
        Tuple of (l4d2_examples, rejected_examples)
    """
    l4d2_examples = []
    rejected_examples = []

    for example in examples:
        # Extract code from different possible fields
        code = example.get('code') or example.get('content') or ''
        if isinstance(example.get('messages'), list):
            # ChatML format - extract from assistant message
            for msg in example['messages']:
                if msg.get('role') == 'assistant':
                    code = msg.get('content', '')
                    break

        is_l4d2, result = is_l4d2_code(code, min_confidence)

        # Add detection metadata
        example_with_meta = example.copy()
        example_with_meta['_game_detection'] = {
            'detected_game': result.detected_game.value,
            'confidence': result.confidence.value,
            'is_l4d2_compatible': result.is_l4d2_compatible,
            'signal_count': len(result.signals),
        }

        if is_l4d2:
            l4d2_examples.append(example_with_meta)
        else:
            rejected_examples.append(example_with_meta)

    return l4d2_examples, rejected_examples


if __name__ == "__main__":
    # Test with sample code
    test_code = '''
#pragma semicolon 1
#include <sourcemod>
#include <left4dhooks>

public void OnPluginStart()
{
    HookEvent("player_incapacitated", Event_Incap);
}

public void Event_Incap(Event event, const char[] name, bool dontBroadcast)
{
    int client = GetClientOfUserId(event.GetInt("userid"));
    int zombieClass = GetEntProp(client, Prop_Send, "m_zombieClass");
}
'''

    result = detect_game(test_code)
    print(f"Detected Game: {result.detected_game.value}")
    print(f"Confidence: {result.confidence.value}")
    print(f"L4D2 Compatible: {result.is_l4d2_compatible}")
    print(f"Signals ({len(result.signals)}):")
    for sig in result.signals:
        print(f"  - {sig.category}: {sig.pattern} (weight: {sig.weight})")
