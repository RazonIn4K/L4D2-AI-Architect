#!/usr/bin/env python3
"""
Comprehensive Benchmark Suite for L4D2 SourcePawn Code Generation Models

This benchmark evaluates models across 55 test cases covering:
- Basic SourcePawn syntax (10 tests)
- L4D2-specific APIs (15 tests)
- Event handling (10 tests)
- Special infected mechanics (10 tests)
- Advanced patterns (10 tests)

Supports testing against:
- Local Ollama models
- OpenAI fine-tuned models
- Base models for comparison

Usage:
    # Run benchmark with Ollama
    python benchmark_suite.py --model ollama --output results/benchmark_v12.json

    # Run benchmark with OpenAI fine-tuned model
    python benchmark_suite.py --model openai --model-id ft:gpt-4o-mini-2024-07-18:...

    # Run benchmark with base model for comparison
    python benchmark_suite.py --model base --model-name gpt-4o-mini

    # Compare multiple models
    python benchmark_suite.py --compare --models ollama,openai

    # Quick test (subset of tests)
    python benchmark_suite.py --model ollama --quick

    # List all test cases
    python benchmark_suite.py --list-tests
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_path, safe_write_json, safe_read_json

PROJECT_ROOT = Path(__file__).parent.parent.parent


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class Difficulty(str, Enum):
    """Test case difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Category(str, Enum):
    """Test case categories."""
    BASIC_SYNTAX = "basic_syntax"
    L4D2_API = "l4d2_api"
    EVENT_HANDLING = "event_handling"
    SPECIAL_INFECTED = "special_infected"
    ADVANCED_PATTERNS = "advanced_patterns"


@dataclass
class TestCase:
    """Definition of a single benchmark test case."""
    id: str
    prompt: str
    expected_patterns: List[str]
    forbidden_patterns: List[str]
    category: Category
    difficulty: Difficulty
    description: str = ""
    expected_patterns_any: Optional[List[List[str]]] = None  # Alternative valid patterns
    min_code_lines: int = 5
    requires_includes: Optional[List[str]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "expected_patterns": self.expected_patterns,
            "forbidden_patterns": self.forbidden_patterns,
            "category": self.category.value,
            "difficulty": self.difficulty.value,
            "description": self.description,
            "expected_patterns_any": self.expected_patterns_any,
            "min_code_lines": self.min_code_lines,
            "requires_includes": self.requires_includes,
        }


@dataclass
class TestResult:
    """Result of running a single test case."""
    test_id: str
    passed: bool
    score: float
    response: str
    expected_found: List[str] = field(default_factory=list)
    expected_missing: List[str] = field(default_factory=list)
    forbidden_found: List[str] = field(default_factory=list)
    has_code: bool = False
    code_lines: int = 0
    has_includes: bool = False
    has_plugin_info: bool = False
    execution_time: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    model_name: str
    model_type: str
    timestamp: str
    total_tests: int
    passed: int
    failed: int
    pass_rate: float
    average_score: float
    by_category: Dict[str, Dict[str, Any]]
    by_difficulty: Dict[str, Dict[str, Any]]
    test_results: List[Dict]
    execution_time: float
    common_issues: List[Tuple[str, int]]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "timestamp": self.timestamp,
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": self.pass_rate,
            "average_score": self.average_score,
            "by_category": self.by_category,
            "by_difficulty": self.by_difficulty,
            "test_results": self.test_results,
            "execution_time": self.execution_time,
            "common_issues": self.common_issues,
        }


# =============================================================================
# BENCHMARK TEST CASES (55 total)
# =============================================================================

BENCHMARK_TESTS: List[TestCase] = [
    # =========================================================================
    # BASIC SYNTAX (10 tests)
    # =========================================================================
    TestCase(
        id="syntax_plugin_base",
        prompt="Write a basic L4D2 SourcePawn plugin with proper plugin info, includes, and OnPluginStart function.",
        expected_patterns=["#include", "public Plugin myinfo", "OnPluginStart"],
        forbidden_patterns=[],
        category=Category.BASIC_SYNTAX,
        difficulty=Difficulty.EASY,
        description="Basic plugin structure with myinfo and OnPluginStart",
        requires_includes=["sourcemod"],
    ),
    TestCase(
        id="syntax_console_command",
        prompt="Write a SourcePawn plugin that registers a console command 'sm_hello' that prints 'Hello World' to the player.",
        expected_patterns=["RegConsoleCmd", "sm_hello", "ReplyToCommand"],
        forbidden_patterns=[],
        category=Category.BASIC_SYNTAX,
        difficulty=Difficulty.EASY,
        description="Console command registration and reply",
    ),
    TestCase(
        id="syntax_admin_command",
        prompt="Write a SourcePawn plugin that registers an admin command 'sm_kick_all' with ADMFLAG_KICK permission.",
        expected_patterns=["RegAdminCmd", "ADMFLAG_KICK"],
        forbidden_patterns=[],
        category=Category.BASIC_SYNTAX,
        difficulty=Difficulty.EASY,
        description="Admin command with permission flag",
    ),
    TestCase(
        id="syntax_convar",
        prompt="Write a SourcePawn plugin that creates a ConVar 'sm_test_enabled' with default value 1 and hooks its change.",
        expected_patterns=["CreateConVar", "HookConVarChange"],
        forbidden_patterns=["CreateConVarEx"],
        category=Category.BASIC_SYNTAX,
        difficulty=Difficulty.EASY,
        description="ConVar creation and change hook",
    ),
    TestCase(
        id="syntax_timer",
        prompt="Write a SourcePawn plugin that creates a repeating timer every 5 seconds using CreateTimer.",
        expected_patterns=["CreateTimer", "TIMER_REPEAT"],
        forbidden_patterns=[],
        category=Category.BASIC_SYNTAX,
        difficulty=Difficulty.EASY,
        description="Timer creation with repeat flag",
    ),
    TestCase(
        id="syntax_array_loop",
        prompt="Write a SourcePawn function that loops through all connected clients and checks if they are alive.",
        expected_patterns=["MaxClients", "IsClientInGame", "IsPlayerAlive"],
        forbidden_patterns=[],
        category=Category.BASIC_SYNTAX,
        difficulty=Difficulty.EASY,
        description="Client iteration with validation",
    ),
    TestCase(
        id="syntax_string_format",
        prompt="Write a SourcePawn function that formats a string with player name and health using Format.",
        expected_patterns=["Format", "GetClientName", "GetClientHealth"],
        forbidden_patterns=[],
        category=Category.BASIC_SYNTAX,
        difficulty=Difficulty.MEDIUM,
        description="String formatting with player data",
    ),
    TestCase(
        id="syntax_methodmap",
        prompt="Write a SourcePawn methodmap for a 'PlayerData' structure that stores kills and deaths.",
        expected_patterns=["methodmap", "property", "get", "set"],
        forbidden_patterns=[],
        category=Category.BASIC_SYNTAX,
        difficulty=Difficulty.MEDIUM,
        description="Methodmap definition with properties",
    ),
    TestCase(
        id="syntax_forward",
        prompt="Write a SourcePawn plugin that defines a forward 'OnPlayerKilledInfected' with client and infected type parameters.",
        expected_patterns=["forward", "OnPlayerKilledInfected"],
        forbidden_patterns=[],
        category=Category.BASIC_SYNTAX,
        difficulty=Difficulty.MEDIUM,
        description="Forward declaration for custom events",
    ),
    TestCase(
        id="syntax_menu",
        prompt="Write a SourcePawn function that creates a menu with 3 options and handles the selection.",
        expected_patterns=["CreateMenu", "AddMenuItem", "DisplayMenu"],
        forbidden_patterns=[],
        category=Category.BASIC_SYNTAX,
        difficulty=Difficulty.MEDIUM,
        description="Menu creation and handling",
    ),

    # =========================================================================
    # L4D2-SPECIFIC APIs (15 tests)
    # =========================================================================
    TestCase(
        id="api_random_float",
        prompt="Write a SourcePawn function that generates a random float between 10.0 and 30.0 for a timer delay.",
        expected_patterns=["GetRandomFloat"],
        forbidden_patterns=["RandomFloat", "RandomInt"],
        category=Category.L4D2_API,
        difficulty=Difficulty.EASY,
        description="Correct random float generation",
    ),
    TestCase(
        id="api_random_int",
        prompt="Write a SourcePawn function that picks a random number between 1 and 100.",
        expected_patterns=["GetRandomInt"],
        forbidden_patterns=["RandomInt", "RandomFloat"],
        category=Category.L4D2_API,
        difficulty=Difficulty.EASY,
        description="Correct random integer generation",
    ),
    TestCase(
        id="api_speed_boost",
        prompt="Write a SourcePawn plugin that increases a survivor's movement speed by 30% using the correct property.",
        expected_patterns=["m_flLaggedMovementValue", "SetEntPropFloat"],
        forbidden_patterns=["m_flSpeed", "m_flMaxSpeed"],
        category=Category.L4D2_API,
        difficulty=Difficulty.MEDIUM,
        description="Correct L4D2 speed modification",
    ),
    TestCase(
        id="api_team_check",
        prompt="Write a SourcePawn function that checks if a player is a survivor (team 2) or infected (team 3).",
        expected_patterns=["GetClientTeam", "== 2", "== 3"],
        forbidden_patterns=["== 1"],  # Spectator check in this context would be wrong
        category=Category.L4D2_API,
        difficulty=Difficulty.EASY,
        description="Team identification for survivors/infected",
    ),
    TestCase(
        id="api_get_zombie_class",
        prompt="Write a SourcePawn function that gets the zombie class of an infected player.",
        expected_patterns_any=[
            ["L4D_GetPlayerZombieClass"],
            ["GetEntProp", "m_zombieClass"],
        ],
        expected_patterns=[],
        forbidden_patterns=[],
        category=Category.L4D2_API,
        difficulty=Difficulty.MEDIUM,
        description="Getting infected player zombie class",
    ),
    TestCase(
        id="api_give_weapon",
        prompt="Write a SourcePawn function that gives a survivor a weapon using GivePlayerItem.",
        expected_patterns=["GivePlayerItem"],
        forbidden_patterns=["CreateEntityByName"],  # For simple giving, GivePlayerItem is preferred
        category=Category.L4D2_API,
        difficulty=Difficulty.EASY,
        description="Giving weapons to players",
    ),
    TestCase(
        id="api_set_health",
        prompt="Write a SourcePawn function that sets a player's health to 100.",
        expected_patterns_any=[
            ["SetEntityHealth"],
            ["SetEntProp", "m_iHealth"],
        ],
        expected_patterns=[],
        forbidden_patterns=["m_health"],  # Wrong property name
        category=Category.L4D2_API,
        difficulty=Difficulty.EASY,
        description="Setting player health",
    ),
    TestCase(
        id="api_teleport",
        prompt="Write a SourcePawn function that teleports a player to specified coordinates.",
        expected_patterns=["TeleportEntity"],
        forbidden_patterns=["SetAbsOrigin"],  # TeleportEntity is the proper way
        category=Category.L4D2_API,
        difficulty=Difficulty.EASY,
        description="Player teleportation",
    ),
    TestCase(
        id="api_sdkhook_damage",
        prompt="Write a SourcePawn plugin that hooks player damage using SDKHook OnTakeDamage.",
        expected_patterns=["SDKHook", "OnTakeDamage"],
        forbidden_patterns=["TakeDamage("],  # Raw TakeDamage is wrong
        category=Category.L4D2_API,
        difficulty=Difficulty.MEDIUM,
        description="SDK hooks for damage",
    ),
    TestCase(
        id="api_emit_sound",
        prompt="Write a SourcePawn function that plays a sound to all players using EmitSoundToAll.",
        expected_patterns=["EmitSoundToAll", "PrecacheSound"],
        forbidden_patterns=[],
        category=Category.L4D2_API,
        difficulty=Difficulty.EASY,
        description="Sound emission and precaching",
    ),
    TestCase(
        id="api_create_entity",
        prompt="Write a SourcePawn function that spawns a prop_physics entity at a location.",
        expected_patterns=["CreateEntityByName", "DispatchSpawn"],
        forbidden_patterns=[],
        category=Category.L4D2_API,
        difficulty=Difficulty.MEDIUM,
        description="Entity creation and spawning",
    ),
    TestCase(
        id="api_trace_ray",
        prompt="Write a SourcePawn function that performs a trace ray from player eye position to find what they're looking at.",
        expected_patterns=["TR_TraceRay", "GetClientEyePosition", "GetClientEyeAngles"],
        forbidden_patterns=["TraceLine"],  # Wrong function name
        category=Category.L4D2_API,
        difficulty=Difficulty.HARD,
        description="Ray tracing for aim detection",
    ),
    TestCase(
        id="api_keyvalue",
        prompt="Write a SourcePawn function that sets entity keyvalues using DispatchKeyValue.",
        expected_patterns=["DispatchKeyValue"],
        forbidden_patterns=[],
        category=Category.L4D2_API,
        difficulty=Difficulty.MEDIUM,
        description="Entity keyvalue configuration",
    ),
    TestCase(
        id="api_finale_check",
        prompt="Write a SourcePawn function that checks if the current map is a finale map.",
        expected_patterns_any=[
            ["L4D_IsMissionFinalMap"],
            ["GetCurrentMap", "finale"],
        ],
        expected_patterns=[],
        forbidden_patterns=[],
        category=Category.L4D2_API,
        difficulty=Difficulty.MEDIUM,
        description="Finale map detection",
    ),
    TestCase(
        id="api_hint_text",
        prompt="Write a SourcePawn function that displays a hint text to a specific player.",
        expected_patterns=["PrintHintText"],
        forbidden_patterns=[],
        category=Category.L4D2_API,
        difficulty=Difficulty.EASY,
        description="Hint text display",
    ),

    # =========================================================================
    # EVENT HANDLING (10 tests)
    # =========================================================================
    TestCase(
        id="event_player_spawn",
        prompt="Write a SourcePawn plugin that hooks the player_spawn event and announces when a player spawns.",
        expected_patterns=["HookEvent", "player_spawn", "GetClientOfUserId"],
        forbidden_patterns=[],
        category=Category.EVENT_HANDLING,
        difficulty=Difficulty.EASY,
        description="Player spawn event handling",
    ),
    TestCase(
        id="event_player_death",
        prompt="Write a SourcePawn plugin that tracks player deaths and announces them.",
        expected_patterns=["HookEvent", "player_death", "GetClientOfUserId"],
        forbidden_patterns=[],
        category=Category.EVENT_HANDLING,
        difficulty=Difficulty.EASY,
        description="Player death event handling",
    ),
    TestCase(
        id="event_round_start",
        prompt="Write a SourcePawn plugin that resets variables when a new round starts.",
        expected_patterns=["HookEvent", "round_start"],
        forbidden_patterns=["OnRoundStart"],  # Should be HookEvent, not a forward
        category=Category.EVENT_HANDLING,
        difficulty=Difficulty.EASY,
        description="Round start event handling",
    ),
    TestCase(
        id="event_map_transition",
        prompt="Write a SourcePawn plugin that saves player data when transitioning to a new map.",
        expected_patterns=["HookEvent", "map_transition"],
        forbidden_patterns=[],
        category=Category.EVENT_HANDLING,
        difficulty=Difficulty.MEDIUM,
        description="Map transition event handling",
    ),
    TestCase(
        id="event_panic_event",
        prompt="Write a SourcePawn plugin that detects when a panic event (horde) is triggered.",
        expected_patterns=["HookEvent", "create_panic_event"],
        forbidden_patterns=["panic_start", "horde_start"],
        category=Category.EVENT_HANDLING,
        difficulty=Difficulty.MEDIUM,
        description="Panic event (horde) detection",
    ),
    TestCase(
        id="event_finale",
        prompt="Write a SourcePawn plugin that detects when the finale starts.",
        expected_patterns=["HookEvent", "finale_start"],
        forbidden_patterns=["generator_final_start"],
        category=Category.EVENT_HANDLING,
        difficulty=Difficulty.MEDIUM,
        description="Finale start detection",
    ),
    TestCase(
        id="event_incap",
        prompt="Write a SourcePawn plugin that detects when a survivor becomes incapacitated.",
        expected_patterns=["HookEvent", "player_incapacitated"],
        forbidden_patterns=[],
        category=Category.EVENT_HANDLING,
        difficulty=Difficulty.MEDIUM,
        description="Player incapacitation event",
    ),
    TestCase(
        id="event_revive",
        prompt="Write a SourcePawn plugin that rewards players when they revive a teammate.",
        expected_patterns=["HookEvent", "revive_success"],
        forbidden_patterns=[],
        category=Category.EVENT_HANDLING,
        difficulty=Difficulty.MEDIUM,
        description="Revive success event handling",
    ),
    TestCase(
        id="event_heal",
        prompt="Write a SourcePawn plugin that logs when a player heals another player.",
        expected_patterns=["HookEvent", "heal_success"],
        forbidden_patterns=[],
        category=Category.EVENT_HANDLING,
        difficulty=Difficulty.MEDIUM,
        description="Heal success event handling",
    ),
    TestCase(
        id="event_saferoom_enter",
        prompt="Write a SourcePawn plugin that detects when a player enters a saferoom checkpoint.",
        expected_patterns=["HookEvent", "player_entered_checkpoint"],
        forbidden_patterns=["m_isInSafeRoom"],
        category=Category.EVENT_HANDLING,
        difficulty=Difficulty.MEDIUM,
        description="Saferoom entry detection",
    ),

    # =========================================================================
    # SPECIAL INFECTED MECHANICS (10 tests)
    # =========================================================================
    TestCase(
        id="si_tank_spawn",
        prompt="Write a SourcePawn plugin that announces when a Tank spawns and displays its health.",
        expected_patterns=["HookEvent", "tank_spawn", "GetEntProp", "m_iHealth"],
        forbidden_patterns=["tank_health_changed"],
        category=Category.SPECIAL_INFECTED,
        difficulty=Difficulty.MEDIUM,
        description="Tank spawn announcement with health",
    ),
    TestCase(
        id="si_witch_proximity",
        prompt="Write a SourcePawn plugin that warns players when they get within 500 units of a Witch.",
        expected_patterns=["witch", "GetVectorDistance", "FindEntityByClassname"],
        forbidden_patterns=[],
        category=Category.SPECIAL_INFECTED,
        difficulty=Difficulty.HARD,
        description="Witch proximity warning system",
    ),
    TestCase(
        id="si_hunter_pounce",
        prompt="Write a SourcePawn plugin that tracks Hunter pounce damage using the correct event.",
        expected_patterns=["HookEvent", "lunge_pounce"],
        forbidden_patterns=["pounce"],  # "pounce" alone is wrong
        category=Category.SPECIAL_INFECTED,
        difficulty=Difficulty.MEDIUM,
        description="Hunter pounce damage tracking",
    ),
    TestCase(
        id="si_smoker_grab",
        prompt="Write a SourcePawn plugin that announces when a Smoker grabs a survivor.",
        expected_patterns=["HookEvent", "tongue_grab"],
        forbidden_patterns=["smoker_tongue_grab", "smoker_grab"],
        category=Category.SPECIAL_INFECTED,
        difficulty=Difficulty.MEDIUM,
        description="Smoker tongue grab detection",
    ),
    TestCase(
        id="si_charger_carry",
        prompt="Write a SourcePawn plugin that tracks Charger carries and impacts.",
        expected_patterns=["HookEvent", "charger_carry_start"],
        forbidden_patterns=["charger_grab", "charger_hit"],
        category=Category.SPECIAL_INFECTED,
        difficulty=Difficulty.MEDIUM,
        description="Charger carry tracking",
    ),
    TestCase(
        id="si_jockey_ride",
        prompt="Write a SourcePawn plugin that measures how long a Jockey rides a survivor.",
        expected_patterns=["HookEvent", "jockey_ride", "jockey_ride_end", "GetGameTime"],
        forbidden_patterns=["jockey_grab"],
        category=Category.SPECIAL_INFECTED,
        difficulty=Difficulty.HARD,
        description="Jockey ride duration tracking",
    ),
    TestCase(
        id="si_boomer_bile",
        prompt="Write a SourcePawn plugin that detects when a survivor gets covered in Boomer bile.",
        expected_patterns=["HookEvent", "player_now_it"],
        forbidden_patterns=["boomer_vomit", "player_biled", "boomer_bile"],
        category=Category.SPECIAL_INFECTED,
        difficulty=Difficulty.MEDIUM,
        description="Boomer bile detection",
    ),
    TestCase(
        id="si_spitter_acid",
        prompt="Write a SourcePawn plugin that tracks damage from Spitter acid pools.",
        expected_patterns=["entered_spit"],
        forbidden_patterns=["spitter_death"],
        category=Category.SPECIAL_INFECTED,
        difficulty=Difficulty.MEDIUM,
        description="Spitter acid damage tracking",
    ),
    TestCase(
        id="si_infected_death",
        prompt="Write a SourcePawn plugin that counts common infected kills per player.",
        expected_patterns=["HookEvent", "infected_death"],
        forbidden_patterns=[],
        category=Category.SPECIAL_INFECTED,
        difficulty=Difficulty.EASY,
        description="Common infected kill counting",
    ),
    TestCase(
        id="si_special_spawn",
        prompt="Write a SourcePawn plugin that announces when any special infected spawns.",
        expected_patterns=["HookEvent", "spawn_special_infected"],
        forbidden_patterns=[],
        category=Category.SPECIAL_INFECTED,
        difficulty=Difficulty.MEDIUM,
        description="Special infected spawn announcements",
    ),

    # =========================================================================
    # ADVANCED PATTERNS (10 tests)
    # =========================================================================
    TestCase(
        id="adv_friendly_fire",
        prompt="Write a SourcePawn plugin that prevents friendly fire damage between survivors using SDKHooks.",
        expected_patterns=["SDKHook", "OnTakeDamage", "GetClientTeam"],
        forbidden_patterns=["TakeDamage("],
        category=Category.ADVANCED_PATTERNS,
        difficulty=Difficulty.HARD,
        description="Friendly fire prevention system",
    ),
    TestCase(
        id="adv_player_glow",
        prompt="Write a SourcePawn function that sets a glowing outline on a player entity.",
        expected_patterns=["m_iGlowType", "m_glowColorOverride", "SetEntProp"],
        forbidden_patterns=[],
        category=Category.ADVANCED_PATTERNS,
        difficulty=Difficulty.HARD,
        description="Player glow/outline effect",
    ),
    TestCase(
        id="adv_spawn_item",
        prompt="Write a SourcePawn function that spawns a first aid kit at a specific location.",
        expected_patterns=["CreateEntityByName", "weapon_first_aid_kit", "DispatchSpawn"],
        forbidden_patterns=[],
        category=Category.ADVANCED_PATTERNS,
        difficulty=Difficulty.MEDIUM,
        description="Item spawning at location",
    ),
    TestCase(
        id="adv_countdown_timer",
        prompt="Write a SourcePawn plugin that implements a 30-second countdown displayed to all players.",
        expected_patterns=["CreateTimer", "PrintCenterTextAll"],
        forbidden_patterns=[],
        category=Category.ADVANCED_PATTERNS,
        difficulty=Difficulty.MEDIUM,
        description="Countdown timer display",
    ),
    TestCase(
        id="adv_player_stats",
        prompt="Write a SourcePawn plugin that tracks and stores kills, deaths, and headshots per player.",
        expected_patterns=["player_death", "headshot"],
        forbidden_patterns=[],
        category=Category.ADVANCED_PATTERNS,
        difficulty=Difficulty.HARD,
        description="Player statistics tracking",
    ),
    TestCase(
        id="adv_vote_system",
        prompt="Write a SourcePawn plugin that implements a vote to restart the map.",
        expected_patterns=["CreateMenu", "vote", "ServerCommand"],
        forbidden_patterns=[],
        category=Category.ADVANCED_PATTERNS,
        difficulty=Difficulty.HARD,
        description="Vote system implementation",
    ),
    TestCase(
        id="adv_entity_cleanup",
        prompt="Write a SourcePawn function that safely removes all entities of a specific classname.",
        expected_patterns=["FindEntityByClassname", "IsValidEntity", "RemoveEntity"],
        forbidden_patterns=[],
        category=Category.ADVANCED_PATTERNS,
        difficulty=Difficulty.MEDIUM,
        description="Safe entity cleanup",
    ),
    TestCase(
        id="adv_client_prefs",
        prompt="Write a SourcePawn plugin that saves player preferences using ClientPrefs cookies.",
        expected_patterns=["RegClientCookie", "SetClientCookie", "GetClientCookie"],
        forbidden_patterns=[],
        category=Category.ADVANCED_PATTERNS,
        difficulty=Difficulty.HARD,
        description="Client preferences storage",
        requires_includes=["clientprefs"],
    ),
    TestCase(
        id="adv_database",
        prompt="Write a SourcePawn plugin that connects to a database and executes a query.",
        expected_patterns=["SQL_Connect", "SQL_Query"],
        forbidden_patterns=[],
        category=Category.ADVANCED_PATTERNS,
        difficulty=Difficulty.HARD,
        description="Database connection and query",
        requires_includes=["dbi"],
    ),
    TestCase(
        id="adv_config_file",
        prompt="Write a SourcePawn plugin that reads configuration from a KeyValues file.",
        expected_patterns=["CreateKeyValues", "FileToKeyValues", "KvGetString"],
        forbidden_patterns=[],
        category=Category.ADVANCED_PATTERNS,
        difficulty=Difficulty.MEDIUM,
        description="KeyValues configuration reading",
    ),
]


# =============================================================================
# MODEL CLIENTS
# =============================================================================

class OllamaClient:
    """Client for Ollama local inference."""

    DEFAULT_MODEL = "l4d2-code-v10plus"

    def __init__(self, model: str = None):
        self.model = model or self.DEFAULT_MODEL
        self._check_ollama()

    def _check_ollama(self) -> bool:
        """Check if Ollama is available."""
        if not shutil.which("ollama"):
            raise RuntimeError("Ollama not found. Install from https://ollama.ai")
        return True

    def is_model_available(self) -> bool:
        """Check if model is available in Ollama."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True
            )
            return self.model in result.stdout
        except Exception:
            return False

    def generate(self, prompt: str, system: str = None) -> Tuple[str, float]:
        """Generate completion using Ollama. Returns (response, execution_time)."""
        if system:
            full_prompt = f"{system}\n\n{prompt}"
        else:
            full_prompt = prompt

        start_time = time.time()
        try:
            result = subprocess.run(
                ["ollama", "run", self.model, full_prompt],
                capture_output=True,
                text=True,
                timeout=180
            )
            execution_time = time.time() - start_time
            return result.stdout.strip(), execution_time
        except subprocess.TimeoutExpired:
            return "Error: Generation timed out", time.time() - start_time
        except Exception as e:
            return f"Error: {e}", time.time() - start_time


class OpenAIClient:
    """Client for OpenAI API."""

    def __init__(self, model_id: str = None, api_key: str = None):
        self.model_id = model_id
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._init_client()

    def _init_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

    def generate(self, prompt: str, system: str = None) -> Tuple[str, float]:
        """Generate completion using OpenAI. Returns (response, execution_time)."""
        if not system:
            system = "You are an expert SourcePawn developer for Left 4 Dead 2."

        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048,
                temperature=0.3
            )
            execution_time = time.time() - start_time
            return response.choices[0].message.content, execution_time
        except Exception as e:
            return f"Error: {e}", time.time() - start_time


class BaseModelClient:
    """Client for testing against base models (non-fine-tuned)."""

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._init_client()

    def _init_client(self):
        """Initialize OpenAI client for base model."""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")

    def generate(self, prompt: str, system: str = None) -> Tuple[str, float]:
        """Generate completion using base model. Returns (response, execution_time)."""
        if not system:
            system = "You are a helpful coding assistant."

        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048,
                temperature=0.3
            )
            execution_time = time.time() - start_time
            return response.choices[0].message.content, execution_time
        except Exception as e:
            return f"Error: {e}", time.time() - start_time


# =============================================================================
# BENCHMARK EVALUATOR
# =============================================================================

class BenchmarkEvaluator:
    """Evaluates model responses against benchmark test cases."""

    SYSTEM_PROMPT = """You are an expert SourcePawn and VScript developer for Left 4 Dead 2 SourceMod plugins.
Write clean, well-documented code with proper error handling.
CRITICAL: Use GetRandomFloat() NOT RandomFloat(). Use GetRandomInt() NOT RandomInt().
Use lunge_pounce NOT pounce for Hunter events.
Use tongue_grab NOT smoker_tongue_grab for Smoker events.
Use player_now_it NOT boomer_vomit for Boomer bile events.
Use charger_carry_start NOT charger_grab for Charger events.
Use m_flLaggedMovementValue for speed, NOT m_flSpeed or m_flMaxSpeed."""

    def __init__(self, client, model_name: str, model_type: str):
        self.client = client
        self.model_name = model_name
        self.model_type = model_type

    def evaluate_test(self, test: TestCase) -> TestResult:
        """Evaluate a single test case."""
        # Generate response
        response, exec_time = self.client.generate(test.prompt, self.SYSTEM_PROMPT)

        # Initialize result
        result = TestResult(
            test_id=test.id,
            passed=False,
            score=0.0,
            response=response,
            execution_time=exec_time
        )

        # Check for errors
        if response.startswith("Error:"):
            result.error = response
            return result

        # Analyze response
        result.has_code = self._has_code(response)
        result.code_lines = self._count_code_lines(response)
        result.has_includes = "#include" in response
        result.has_plugin_info = "public Plugin myinfo" in response or "public Plugin:myinfo" in response

        # Check patterns
        pattern_result = self._check_patterns(response, test)
        result.expected_found = pattern_result["expected_found"]
        result.expected_missing = pattern_result["expected_missing"]
        result.forbidden_found = pattern_result["forbidden_found"]

        # Calculate score
        result.score = self._calculate_score(result, test)

        # Determine pass/fail
        result.passed = self._determine_pass(result, test)

        return result

    def _has_code(self, response: str) -> bool:
        """Check if response contains code."""
        code_indicators = ["#include", "public ", "void ", "int ", "float ", "{", "}"]
        return any(ind in response for ind in code_indicators)

    def _count_code_lines(self, response: str) -> int:
        """Count lines that look like code."""
        code_indicators = ["{", "}", ";", "#include", "public ", "void ", "int ", "float "]
        lines = response.split("\n")
        return sum(1 for line in lines if any(ind in line for ind in code_indicators))

    def _check_patterns(self, response: str, test: TestCase) -> Dict:
        """Check expected and forbidden patterns."""
        result = {
            "expected_found": [],
            "expected_missing": [],
            "forbidden_found": [],
        }

        response_lower = response.lower()

        # Handle expected_patterns_any (alternative valid patterns)
        if test.expected_patterns_any:
            best_match = None
            best_count = -1

            for alt_patterns in test.expected_patterns_any:
                found = [p for p in alt_patterns if p.lower() in response_lower]
                if len(found) > best_count:
                    best_count = len(found)
                    best_match = {
                        "found": found,
                        "missing": [p for p in alt_patterns if p.lower() not in response_lower],
                        "complete": len(found) == len(alt_patterns)
                    }

            if best_match:
                result["expected_found"].extend(best_match["found"])
                if not best_match["complete"]:
                    result["expected_missing"].extend(best_match["missing"])

        # Standard expected patterns
        for pattern in test.expected_patterns:
            if pattern.lower() in response_lower:
                result["expected_found"].append(pattern)
            else:
                result["expected_missing"].append(pattern)

        # Check forbidden patterns with smart matching
        for pattern in test.forbidden_patterns:
            if self._check_forbidden_pattern(response, pattern):
                result["forbidden_found"].append(pattern)

        return result

    def _check_forbidden_pattern(self, response: str, pattern: str) -> bool:
        """Smart check for forbidden patterns to avoid false positives."""
        if pattern == "pounce":
            # Only match HookEvent("pounce" not lunge_pounce
            return bool(re.search(r'HookEvent\s*\(\s*["\']pounce["\']', response, re.IGNORECASE))
        elif pattern == "RandomFloat":
            # Only match RandomFloat( not GetRandomFloat(
            return bool(re.search(r'(?<!Get)RandomFloat\s*\(', response))
        elif pattern == "RandomInt":
            # Only match RandomInt( not GetRandomInt(
            return bool(re.search(r'(?<!Get)RandomInt\s*\(', response))
        elif pattern in ["smoker_tongue_grab", "boomer_vomit", "player_biled",
                         "charger_grab", "charger_hit", "jockey_grab"]:
            # Only match if in HookEvent
            return bool(re.search(rf'HookEvent\s*\(\s*["\']{ re.escape(pattern) }["\']',
                                   response, re.IGNORECASE))
        else:
            return pattern in response

    def _calculate_score(self, result: TestResult, test: TestCase) -> float:
        """Calculate score for test result (0-10 scale)."""
        score = 0.0

        # Has code (3 points)
        if result.has_code:
            score += 3.0

        # Code length bonus (up to 1 point)
        if result.code_lines >= test.min_code_lines:
            score += 1.0
        elif result.code_lines > 0:
            score += 0.5

        # Expected patterns (up to 4 points)
        total_expected = len(test.expected_patterns)
        if test.expected_patterns_any:
            # Use the alternative with best match
            total_expected = max(len(alt) for alt in test.expected_patterns_any)
        if total_expected > 0:
            found_ratio = len(result.expected_found) / total_expected
            score += found_ratio * 4.0
        else:
            score += 4.0  # No expected patterns = full points

        # No forbidden patterns (2 points)
        if not result.forbidden_found:
            score += 2.0

        # Structure bonus
        if result.has_includes:
            score += 0.25
        if result.has_plugin_info:
            score += 0.25

        # Penalty for forbidden patterns
        score -= len(result.forbidden_found) * 1.5

        return max(0.0, min(10.0, score))

    def _determine_pass(self, result: TestResult, test: TestCase) -> bool:
        """Determine if test passed."""
        # Must have code
        if not result.has_code:
            return False

        # Must not have forbidden patterns
        if result.forbidden_found:
            return False

        # Score threshold
        if result.score < 6.0:
            return False

        # Must have most expected patterns
        total_expected = len(test.expected_patterns)
        if test.expected_patterns_any:
            # Check if any alternative is fully satisfied
            for alt in test.expected_patterns_any:
                if all(p in result.expected_found for p in alt):
                    return True
            total_expected = max(len(alt) for alt in test.expected_patterns_any)

        if total_expected > 0:
            found_ratio = len(result.expected_found) / total_expected
            if found_ratio < 0.7:  # Must find at least 70% of expected patterns
                return False

        return True


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

class BenchmarkRunner:
    """Runs the full benchmark suite."""

    def __init__(self, evaluator: BenchmarkEvaluator, tests: List[TestCase] = None):
        self.evaluator = evaluator
        self.tests = tests or BENCHMARK_TESTS

    def run(self, verbose: bool = True) -> BenchmarkReport:
        """Run the full benchmark suite."""
        start_time = time.time()
        results: List[TestResult] = []
        issues: List[str] = []

        total = len(self.tests)
        for i, test in enumerate(self.tests, 1):
            if verbose:
                print(f"[{i}/{total}] {test.id} ({test.category.value}, {test.difficulty.value})...", end=" ", flush=True)

            result = self.evaluator.evaluate_test(test)
            results.append(result)

            if verbose:
                status = "PASS" if result.passed else "FAIL"
                print(f"{status} (score: {result.score:.1f})")

            # Collect issues
            if result.forbidden_found:
                for p in result.forbidden_found:
                    issues.append(f"Used forbidden pattern: {p}")
            if result.expected_missing:
                for p in result.expected_missing:
                    issues.append(f"Missing expected pattern: {p}")
            if result.error:
                issues.append(f"Error: {result.error}")

        execution_time = time.time() - start_time

        # Generate report
        return self._generate_report(results, issues, execution_time)

    def _generate_report(self, results: List[TestResult], issues: List[str],
                         execution_time: float) -> BenchmarkReport:
        """Generate benchmark report from results."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        avg_score = sum(r.score for r in results) / total if total > 0 else 0

        # Group by category
        by_category = {}
        for cat in Category:
            cat_results = [r for r, t in zip(results, self.tests) if t.category == cat]
            if cat_results:
                cat_passed = sum(1 for r in cat_results if r.passed)
                by_category[cat.value] = {
                    "total": len(cat_results),
                    "passed": cat_passed,
                    "failed": len(cat_results) - cat_passed,
                    "pass_rate": cat_passed / len(cat_results) * 100,
                    "avg_score": sum(r.score for r in cat_results) / len(cat_results),
                }

        # Group by difficulty
        by_difficulty = {}
        for diff in Difficulty:
            diff_results = [r for r, t in zip(results, self.tests) if t.difficulty == diff]
            if diff_results:
                diff_passed = sum(1 for r in diff_results if r.passed)
                by_difficulty[diff.value] = {
                    "total": len(diff_results),
                    "passed": diff_passed,
                    "failed": len(diff_results) - diff_passed,
                    "pass_rate": diff_passed / len(diff_results) * 100,
                    "avg_score": sum(r.score for r in diff_results) / len(diff_results),
                }

        # Count common issues
        issue_counts: Dict[str, int] = {}
        for issue in issues:
            key = issue.split(":")[0] if ":" in issue else issue
            issue_counts[key] = issue_counts.get(key, 0) + 1
        common_issues = sorted(issue_counts.items(), key=lambda x: -x[1])[:10]

        return BenchmarkReport(
            model_name=self.evaluator.model_name,
            model_type=self.evaluator.model_type,
            timestamp=datetime.now().isoformat(),
            total_tests=total,
            passed=passed,
            failed=failed,
            pass_rate=passed / total * 100 if total > 0 else 0,
            average_score=avg_score,
            by_category=by_category,
            by_difficulty=by_difficulty,
            test_results=[r.to_dict() for r in results],
            execution_time=execution_time,
            common_issues=common_issues,
        )


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_markdown_report(report: BenchmarkReport, output_path: Path) -> None:
    """Generate a Markdown report from benchmark results."""
    md = f"""# L4D2 SourcePawn Benchmark Report

**Model**: {report.model_name}
**Model Type**: {report.model_type}
**Date**: {report.timestamp}
**Execution Time**: {report.execution_time:.1f}s

## Summary

| Metric | Value |
|--------|-------|
| Total Tests | {report.total_tests} |
| Passed | {report.passed} |
| Failed | {report.failed} |
| **Pass Rate** | **{report.pass_rate:.1f}%** |
| **Average Score** | **{report.average_score:.2f}/10** |

## Results by Category

| Category | Tests | Passed | Failed | Pass Rate | Avg Score |
|----------|-------|--------|--------|-----------|-----------|
"""
    for cat, stats in report.by_category.items():
        md += f"| {cat} | {stats['total']} | {stats['passed']} | {stats['failed']} | {stats['pass_rate']:.1f}% | {stats['avg_score']:.2f} |\n"

    md += """
## Results by Difficulty

| Difficulty | Tests | Passed | Failed | Pass Rate | Avg Score |
|------------|-------|--------|--------|-----------|-----------|
"""
    for diff, stats in report.by_difficulty.items():
        md += f"| {diff} | {stats['total']} | {stats['passed']} | {stats['failed']} | {stats['pass_rate']:.1f}% | {stats['avg_score']:.2f} |\n"

    if report.common_issues:
        md += """
## Common Issues

| Issue | Count |
|-------|-------|
"""
        for issue, count in report.common_issues:
            md += f"| {issue} | {count} |\n"

    md += """
## Detailed Results

"""
    for result in report.test_results:
        status = "PASS" if result["passed"] else "FAIL"
        md += f"""### {result['test_id']}: {status} (Score: {result['score']:.1f}/10)

- **Has Code**: {'Yes' if result['has_code'] else 'No'}
- **Code Lines**: {result['code_lines']}
- **Execution Time**: {result['execution_time']:.2f}s
- **Expected Found**: {', '.join(result['expected_found']) or 'None'}
- **Expected Missing**: {', '.join(result['expected_missing']) or 'None'}
- **Forbidden Found**: {', '.join(result['forbidden_found']) or 'None'}

"""
        if result.get('error'):
            md += f"**Error**: {result['error']}\n\n"

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md)


def generate_comparison_report(reports: List[BenchmarkReport], output_path: Path) -> None:
    """Generate a comparison report for multiple models."""
    md = f"""# L4D2 Model Comparison Report

**Date**: {datetime.now().isoformat()}

## Overall Comparison

| Model | Type | Pass Rate | Avg Score | Time |
|-------|------|-----------|-----------|------|
"""
    for report in reports:
        md += f"| {report.model_name} | {report.model_type} | {report.pass_rate:.1f}% | {report.average_score:.2f} | {report.execution_time:.1f}s |\n"

    md += """
## Comparison by Category

"""
    categories = list(Category)
    for cat in categories:
        md += f"### {cat.value}\n\n"
        md += "| Model | Pass Rate | Avg Score |\n"
        md += "|-------|-----------|----------|\n"
        for report in reports:
            if cat.value in report.by_category:
                stats = report.by_category[cat.value]
                md += f"| {report.model_name} | {stats['pass_rate']:.1f}% | {stats['avg_score']:.2f} |\n"
        md += "\n"

    md += """
## Comparison by Difficulty

"""
    for diff in Difficulty:
        md += f"### {diff.value}\n\n"
        md += "| Model | Pass Rate | Avg Score |\n"
        md += "|-------|-----------|----------|\n"
        for report in reports:
            if diff.value in report.by_difficulty:
                stats = report.by_difficulty[diff.value]
                md += f"| {report.model_name} | {stats['pass_rate']:.1f}% | {stats['avg_score']:.2f} |\n"
        md += "\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md)


# =============================================================================
# CLI
# =============================================================================

def list_tests():
    """Print all test cases."""
    print(f"\n{'='*80}")
    print("L4D2 SourcePawn Benchmark Test Cases")
    print(f"{'='*80}")
    print(f"Total: {len(BENCHMARK_TESTS)} tests\n")

    for cat in Category:
        cat_tests = [t for t in BENCHMARK_TESTS if t.category == cat]
        print(f"\n{cat.value.upper()} ({len(cat_tests)} tests)")
        print("-" * 60)
        for test in cat_tests:
            print(f"  [{test.difficulty.value:6}] {test.id}")
            if test.description:
                print(f"           {test.description}")


def main():
    parser = argparse.ArgumentParser(
        description="L4D2 SourcePawn Model Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--model", choices=["ollama", "openai", "base"],
                        default="ollama", help="Model type to benchmark")
    parser.add_argument("--model-id", type=str,
                        help="Model ID (for OpenAI fine-tuned models)")
    parser.add_argument("--model-name", type=str,
                        help="Model name (for Ollama or base models)")
    parser.add_argument("--output", type=str, default="results/benchmark.json",
                        help="Output file for JSON results")
    parser.add_argument("--markdown", type=str,
                        help="Also generate Markdown report")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick test (10 tests only)")
    parser.add_argument("--category", type=str,
                        help="Run only tests in specific category")
    parser.add_argument("--difficulty", type=str,
                        help="Run only tests of specific difficulty")
    parser.add_argument("--compare", action="store_true",
                        help="Compare multiple models")
    parser.add_argument("--models", type=str,
                        help="Comma-separated list of models for comparison")
    parser.add_argument("--list-tests", action="store_true",
                        help="List all test cases and exit")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Verbose output")
    parser.add_argument("--quiet", action="store_true",
                        help="Quiet output")

    args = parser.parse_args()

    if args.list_tests:
        list_tests()
        return

    # Filter tests
    tests = BENCHMARK_TESTS.copy()

    if args.quick:
        # Take 2 from each category for quick test
        quick_tests = []
        for cat in Category:
            cat_tests = [t for t in tests if t.category == cat]
            quick_tests.extend(cat_tests[:2])
        tests = quick_tests

    if args.category:
        try:
            cat = Category(args.category)
            tests = [t for t in tests if t.category == cat]
        except ValueError:
            print(f"Error: Invalid category '{args.category}'")
            print(f"Valid categories: {[c.value for c in Category]}")
            sys.exit(1)

    if args.difficulty:
        try:
            diff = Difficulty(args.difficulty)
            tests = [t for t in tests if t.difficulty == diff]
        except ValueError:
            print(f"Error: Invalid difficulty '{args.difficulty}'")
            print(f"Valid difficulties: {[d.value for d in Difficulty]}")
            sys.exit(1)

    verbose = args.verbose and not args.quiet

    if args.compare:
        # Run comparison
        if not args.models:
            print("Error: --models required for comparison")
            print("Example: --compare --models ollama,openai,base")
            sys.exit(1)

        reports = []
        for model_spec in args.models.split(","):
            model_spec = model_spec.strip()
            print(f"\n{'='*60}")
            print(f"Benchmarking: {model_spec}")
            print(f"{'='*60}")

            if model_spec == "ollama":
                client = OllamaClient(args.model_name or "l4d2-code-v10plus")
                model_name = client.model
            elif model_spec == "openai":
                if not args.model_id:
                    print("Error: --model-id required for OpenAI")
                    continue
                client = OpenAIClient(args.model_id)
                model_name = args.model_id
            elif model_spec == "base":
                client = BaseModelClient(args.model_name or "gpt-4o-mini")
                model_name = client.model_name
            else:
                print(f"Error: Unknown model type: {model_spec}")
                continue

            evaluator = BenchmarkEvaluator(client, model_name, model_spec)
            runner = BenchmarkRunner(evaluator, tests)
            report = runner.run(verbose=verbose)
            reports.append(report)

            # Save individual report
            output_path = safe_path(f"results/benchmark_{model_spec}.json", PROJECT_ROOT, create_parents=True)
            safe_write_json(str(output_path), report.to_dict(), PROJECT_ROOT)

        # Generate comparison report
        if reports:
            comparison_path = safe_path("results/benchmark_comparison.md", PROJECT_ROOT, create_parents=True)
            generate_comparison_report(reports, comparison_path)
            print(f"\nComparison report saved to: {comparison_path}")

    else:
        # Run single model benchmark
        print(f"\n{'='*60}")
        print(f"L4D2 SourcePawn Benchmark Suite")
        print(f"{'='*60}")
        print(f"Model type: {args.model}")
        print(f"Tests: {len(tests)}")

        # Initialize client
        if args.model == "ollama":
            model_name = args.model_name or "l4d2-code-v10plus"
            try:
                client = OllamaClient(model_name)
                if not client.is_model_available():
                    print(f"Warning: Model '{model_name}' not found in Ollama")
                    print("Available models:")
                    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
                    print(result.stdout)
                    sys.exit(1)
            except RuntimeError as e:
                print(f"Error: {e}")
                sys.exit(1)
            model_type = "ollama"

        elif args.model == "openai":
            if not args.model_id:
                # Try to load from file
                model_files = [
                    PROJECT_ROOT / "data" / "processed" / "v12_model_id.txt",
                    PROJECT_ROOT / "data" / "processed" / "v11_model_id.txt",
                    PROJECT_ROOT / "data" / "processed" / "v10_model_id.txt",
                    PROJECT_ROOT / "data" / "processed" / "v9_model_id.txt",
                ]
                for mf in model_files:
                    if mf.exists():
                        args.model_id = mf.read_text().strip()
                        print(f"Using model from {mf.name}: {args.model_id}")
                        break
                if not args.model_id:
                    print("Error: --model-id required for OpenAI, or create data/processed/v*_model_id.txt")
                    sys.exit(1)

            try:
                client = OpenAIClient(args.model_id)
                model_name = args.model_id
            except RuntimeError as e:
                print(f"Error: {e}")
                sys.exit(1)
            model_type = "openai"

        elif args.model == "base":
            model_name = args.model_name or "gpt-4o-mini"
            try:
                client = BaseModelClient(model_name)
            except RuntimeError as e:
                print(f"Error: {e}")
                sys.exit(1)
            model_type = "base"

        print(f"Model: {model_name}")
        print(f"{'='*60}\n")

        # Run benchmark
        evaluator = BenchmarkEvaluator(client, model_name, model_type)
        runner = BenchmarkRunner(evaluator, tests)
        report = runner.run(verbose=verbose)

        # Print summary
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests:   {report.total_tests}")
        print(f"Passed:        {report.passed}")
        print(f"Failed:        {report.failed}")
        print(f"Pass Rate:     {report.pass_rate:.1f}%")
        print(f"Average Score: {report.average_score:.2f}/10")
        print(f"Time:          {report.execution_time:.1f}s")

        print(f"\nBy Category:")
        for cat, stats in report.by_category.items():
            print(f"  {cat}: {stats['pass_rate']:.1f}% pass, {stats['avg_score']:.2f} avg")

        print(f"\nBy Difficulty:")
        for diff, stats in report.by_difficulty.items():
            print(f"  {diff}: {stats['pass_rate']:.1f}% pass, {stats['avg_score']:.2f} avg")

        if report.common_issues:
            print(f"\nCommon Issues:")
            for issue, count in report.common_issues[:5]:
                print(f"  [{count}x] {issue}")

        # Save results
        output_path = safe_path(args.output, PROJECT_ROOT, create_parents=True)
        safe_write_json(str(output_path), report.to_dict(), PROJECT_ROOT)
        print(f"\nResults saved to: {output_path}")

        # Generate Markdown report if requested
        if args.markdown:
            md_path = safe_path(args.markdown, PROJECT_ROOT, create_parents=True)
            generate_markdown_report(report, md_path)
            print(f"Markdown report saved to: {md_path}")


if __name__ == "__main__":
    main()
