#!/usr/bin/env python3
"""
L4D2 Prompt Engineering Toolkit

Provides 50+ optimized prompt templates for L4D2 code generation including:
- Plugin scaffolding
- Event handlers
- ConVar definitions
- Native declarations
- Forward definitions
- Menu builders
- Database queries
- Timer systems
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.security import safe_write_json

PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class PromptTemplate:
    """A prompt template for L4D2 code generation."""
    name: str
    description: str
    category: str
    system_prompt: str
    user_template: str
    example_output: str
    placeholders: List[str]

    def render(self, **kwargs) -> str:
        """Render the template with provided values."""
        result = self.user_template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


# System prompts for different contexts
SYSTEM_PROMPTS = {
    "sourcepawn": """You are an expert SourcePawn developer specializing in Left 4 Dead 2 plugins.
You write clean, efficient, and well-documented code following SourceMod best practices.
Always use #pragma newdecls required and #pragma semicolon 1.
Include proper error handling and validation.""",

    "vscript": """You are an expert VScript developer for Left 4 Dead 2.
You create efficient Squirrel scripts for Director modifications, custom events, and game logic.
Follow L4D2 VScript conventions and use proper table structures.""",

    "entity": """You are an expert in L4D2 entity manipulation using SourcePawn.
You understand SDKHooks, SDKTools, and entity properties.
Write safe code that validates entities before manipulation.""",

    "database": """You are an expert in SourceMod database programming.
You write secure SQL queries using prepared statements.
Handle connection errors gracefully and use transactions when appropriate.""",

    "menu": """You are an expert in SourceMod menu systems.
You create intuitive, user-friendly menus with proper pagination and callbacks.
Handle menu cancellation and timeout appropriately.""",

    "timer": """You are an expert in SourceMod timer systems.
You create efficient timers with proper cleanup and data management.
Use DataPacks when passing complex data to timer callbacks.""",
}


# Template definitions organized by category
TEMPLATES: List[PromptTemplate] = []


def register_template(
    name: str,
    description: str,
    category: str,
    system_prompt_key: str,
    user_template: str,
    example_output: str,
    placeholders: List[str]
) -> PromptTemplate:
    """Register a new template."""
    template = PromptTemplate(
        name=name,
        description=description,
        category=category,
        system_prompt=SYSTEM_PROMPTS.get(system_prompt_key, SYSTEM_PROMPTS["sourcepawn"]),
        user_template=user_template,
        example_output=example_output,
        placeholders=placeholders
    )
    TEMPLATES.append(template)
    return template


# =============================================================================
# PLUGIN SCAFFOLDING TEMPLATES (1-8)
# =============================================================================

register_template(
    name="plugin_base",
    description="Basic L4D2 plugin skeleton with standard structure",
    category="scaffolding",
    system_prompt_key="sourcepawn",
    user_template="Create a basic L4D2 SourcePawn plugin named '{name}' by '{author}' that {purpose}.",
    example_output='''#include <sourcemod>
#include <sdktools>

#pragma newdecls required
#pragma semicolon 1

#define PLUGIN_VERSION "1.0.0"

public Plugin myinfo = {
    name = "My Plugin",
    author = "Developer",
    description = "Plugin description",
    version = PLUGIN_VERSION,
    url = "https://github.com/developer"
};

public void OnPluginStart() {
    CreateConVar("sm_myplugin_version", PLUGIN_VERSION, "Plugin version", FCVAR_NOTIFY | FCVAR_DONTRECORD);
}''',
    placeholders=["name", "author", "purpose"]
)

register_template(
    name="plugin_full",
    description="Complete plugin with ConVars, commands, and events",
    category="scaffolding",
    system_prompt_key="sourcepawn",
    user_template="Create a complete L4D2 plugin '{name}' with ConVars for configuration, admin commands, and event hooks for {features}.",
    example_output='''#include <sourcemod>
#include <sdktools>
#include <left4dhooks>

#pragma newdecls required
#pragma semicolon 1

#define PLUGIN_VERSION "1.0.0"

ConVar g_cvEnabled;
ConVar g_cvValue;

public Plugin myinfo = {
    name = "Complete Plugin",
    author = "Developer",
    description = "Full featured plugin",
    version = PLUGIN_VERSION,
    url = ""
};

public void OnPluginStart() {
    g_cvEnabled = CreateConVar("sm_plugin_enabled", "1", "Enable plugin");
    g_cvValue = CreateConVar("sm_plugin_value", "100", "Configuration value");

    RegAdminCmd("sm_plugin", Cmd_Plugin, ADMFLAG_GENERIC);

    HookEvent("player_spawn", Event_PlayerSpawn);

    AutoExecConfig(true, "plugin_config");
}

public Action Cmd_Plugin(int client, int args) {
    ReplyToCommand(client, "Plugin command executed");
    return Plugin_Handled;
}

public void Event_PlayerSpawn(Event event, const char[] name, bool dontBroadcast) {
    int client = GetClientOfUserId(event.GetInt("userid"));
    if (IsValidClient(client)) {
        // Handle spawn
    }
}

bool IsValidClient(int client) {
    return client > 0 && client <= MaxClients && IsClientInGame(client);
}''',
    placeholders=["name", "features"]
)

register_template(
    name="plugin_l4d2_specific",
    description="L4D2-specific plugin with team checks and survivor handling",
    category="scaffolding",
    system_prompt_key="sourcepawn",
    user_template="Create an L4D2 plugin that handles {functionality} with proper survivor/infected team checks.",
    example_output='''#include <sourcemod>
#include <sdktools>

#pragma newdecls required
#pragma semicolon 1

#define TEAM_SURVIVOR 2
#define TEAM_INFECTED 3

public void OnPluginStart() {
    HookEvent("player_hurt", Event_PlayerHurt);
}

public void Event_PlayerHurt(Event event, const char[] name, bool dontBroadcast) {
    int victim = GetClientOfUserId(event.GetInt("userid"));
    int attacker = GetClientOfUserId(event.GetInt("attacker"));

    if (!IsValidSurvivor(victim)) return;
    if (!IsValidInfected(attacker)) return;

    int damage = event.GetInt("dmg_health");
    // Handle damage
}

bool IsValidSurvivor(int client) {
    return IsValidClient(client) && GetClientTeam(client) == TEAM_SURVIVOR && IsPlayerAlive(client);
}

bool IsValidInfected(int client) {
    return IsValidClient(client) && GetClientTeam(client) == TEAM_INFECTED && IsPlayerAlive(client);
}

bool IsValidClient(int client) {
    return client > 0 && client <= MaxClients && IsClientInGame(client);
}''',
    placeholders=["functionality"]
)

register_template(
    name="plugin_extension",
    description="Plugin that extends existing functionality",
    category="scaffolding",
    system_prompt_key="sourcepawn",
    user_template="Create a plugin that extends {base_plugin} to add {new_features}.",
    example_output='''#include <sourcemod>
#include <sdktools>

#pragma newdecls required
#pragma semicolon 1

// Forward declarations for base plugin natives
native bool BasePlugin_IsEnabled();
native void BasePlugin_SetValue(int value);

public Plugin myinfo = {
    name = "Plugin Extension",
    author = "Developer",
    description = "Extends base plugin functionality",
    version = "1.0.0",
    url = ""
};

public void OnAllPluginsLoaded() {
    if (!LibraryExists("base_plugin")) {
        SetFailState("Base plugin is required");
    }
}

public void OnLibraryRemoved(const char[] name) {
    if (StrEqual(name, "base_plugin")) {
        SetFailState("Base plugin was unloaded");
    }
}''',
    placeholders=["base_plugin", "new_features"]
)

register_template(
    name="plugin_include",
    description="Include file with natives and forwards",
    category="scaffolding",
    system_prompt_key="sourcepawn",
    user_template="Create an include file for '{plugin_name}' exposing natives for {natives_list} and forwards for {forwards_list}.",
    example_output='''#if defined _myplugin_included
 #endinput
#endif
#define _myplugin_included

/**
 * Check if the plugin is enabled.
 *
 * @return          True if enabled, false otherwise.
 */
native bool MyPlugin_IsEnabled();

/**
 * Get the current value.
 *
 * @param client    Client index.
 * @return          The current value for the client.
 */
native int MyPlugin_GetValue(int client);

/**
 * Set a value for the client.
 *
 * @param client    Client index.
 * @param value     Value to set.
 * @noreturn
 */
native void MyPlugin_SetValue(int client, int value);

/**
 * Called when a value changes.
 *
 * @param client    Client index.
 * @param oldValue  Previous value.
 * @param newValue  New value.
 * @noreturn
 */
forward void MyPlugin_OnValueChanged(int client, int oldValue, int newValue);

public SharedPlugin __pl_myplugin = {
    name = "myplugin",
    file = "myplugin.smx",
#if defined REQUIRE_PLUGIN
    required = 1,
#else
    required = 0,
#endif
};

#if !defined REQUIRE_PLUGIN
public void __pl_myplugin_SetNTVOptional() {
    MarkNativeAsOptional("MyPlugin_IsEnabled");
    MarkNativeAsOptional("MyPlugin_GetValue");
    MarkNativeAsOptional("MyPlugin_SetValue");
}
#endif''',
    placeholders=["plugin_name", "natives_list", "forwards_list"]
)

register_template(
    name="plugin_gamedata",
    description="Plugin with gamedata signatures and offsets",
    category="scaffolding",
    system_prompt_key="sourcepawn",
    user_template="Create a plugin that uses gamedata to {functionality} with proper signature handling.",
    example_output='''#include <sourcemod>
#include <sdktools>

#pragma newdecls required
#pragma semicolon 1

Handle g_hSDKCall;
int g_iOffset;

public void OnPluginStart() {
    GameData hGameData = new GameData("myplugin.games");
    if (hGameData == null) {
        SetFailState("Failed to load gamedata");
    }

    StartPrepSDKCall(SDKCall_Player);
    if (!PrepSDKCall_SetFromConf(hGameData, SDKConf_Signature, "MyFunction")) {
        SetFailState("Failed to find signature");
    }
    PrepSDKCall_SetReturnInfo(SDKType_PlainOldData, SDKPass_Plain);
    g_hSDKCall = EndPrepSDKCall();

    g_iOffset = hGameData.GetOffset("MyOffset");
    if (g_iOffset == -1) {
        SetFailState("Failed to find offset");
    }

    delete hGameData;
}''',
    placeholders=["functionality"]
)

register_template(
    name="plugin_translation",
    description="Plugin with multi-language translation support",
    category="scaffolding",
    system_prompt_key="sourcepawn",
    user_template="Create a plugin with translation support for {languages} that {functionality}.",
    example_output='''#include <sourcemod>

#pragma newdecls required
#pragma semicolon 1

public void OnPluginStart() {
    LoadTranslations("myplugin.phrases");

    RegConsoleCmd("sm_hello", Cmd_Hello);
}

public Action Cmd_Hello(int client, int args) {
    if (client == 0) {
        PrintToServer("%t", "Hello_Message");
    } else {
        PrintToChat(client, "%t", "Hello_Message");
    }
    return Plugin_Handled;
}

// Translation file (translations/myplugin.phrases.txt):
// "Phrases"
// {
//     "Hello_Message"
//     {
//         "en"    "Hello, welcome to the server!"
//         "es"    "Hola, bienvenido al servidor!"
//         "de"    "Hallo, willkommen auf dem Server!"
//     }
// }''',
    placeholders=["languages", "functionality"]
)

register_template(
    name="plugin_api",
    description="Plugin exposing a public API with natives",
    category="scaffolding",
    system_prompt_key="sourcepawn",
    user_template="Create a plugin that exposes a public API for {api_purpose} with natives for {operations}.",
    example_output='''#include <sourcemod>

#pragma newdecls required
#pragma semicolon 1

int g_iClientData[MAXPLAYERS + 1];

GlobalForward g_fwdOnDataChanged;

public APLRes AskPluginLoad2(Handle myself, bool late, char[] error, int err_max) {
    CreateNative("MyAPI_GetData", Native_GetData);
    CreateNative("MyAPI_SetData", Native_SetData);

    g_fwdOnDataChanged = new GlobalForward("MyAPI_OnDataChanged", ET_Ignore, Param_Cell, Param_Cell, Param_Cell);

    RegPluginLibrary("myapi");
    return APLRes_Success;
}

public int Native_GetData(Handle plugin, int numParams) {
    int client = GetNativeCell(1);
    if (!IsValidClient(client)) {
        ThrowNativeError(SP_ERROR_NATIVE, "Invalid client %d", client);
    }
    return g_iClientData[client];
}

public int Native_SetData(Handle plugin, int numParams) {
    int client = GetNativeCell(1);
    int value = GetNativeCell(2);

    if (!IsValidClient(client)) {
        ThrowNativeError(SP_ERROR_NATIVE, "Invalid client %d", client);
    }

    int oldValue = g_iClientData[client];
    g_iClientData[client] = value;

    Call_StartForward(g_fwdOnDataChanged);
    Call_PushCell(client);
    Call_PushCell(oldValue);
    Call_PushCell(value);
    Call_Finish();

    return 0;
}

bool IsValidClient(int client) {
    return client > 0 && client <= MaxClients && IsClientInGame(client);
}''',
    placeholders=["api_purpose", "operations"]
)

# =============================================================================
# EVENT HANDLER TEMPLATES (9-18)
# =============================================================================

register_template(
    name="event_player_spawn",
    description="Handle player spawn events",
    category="events",
    system_prompt_key="sourcepawn",
    user_template="Create an event handler for player spawn that {action}.",
    example_output='''public void OnPluginStart() {
    HookEvent("player_spawn", Event_PlayerSpawn);
}

public void Event_PlayerSpawn(Event event, const char[] name, bool dontBroadcast) {
    int client = GetClientOfUserId(event.GetInt("userid"));

    if (!IsValidClient(client)) return;

    int team = GetClientTeam(client);

    if (team == 2) {
        // Survivor spawned
        PrintToChat(client, "Welcome, Survivor!");
    } else if (team == 3) {
        // Infected spawned
        PrintToChat(client, "You are now infected!");
    }
}

bool IsValidClient(int client) {
    return client > 0 && client <= MaxClients && IsClientInGame(client);
}''',
    placeholders=["action"]
)

register_template(
    name="event_player_death",
    description="Handle player death events with killer tracking",
    category="events",
    system_prompt_key="sourcepawn",
    user_template="Create a player death handler that {action} and tracks {tracking}.",
    example_output='''public void OnPluginStart() {
    HookEvent("player_death", Event_PlayerDeath);
}

public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast) {
    int victim = GetClientOfUserId(event.GetInt("userid"));
    int attacker = GetClientOfUserId(event.GetInt("attacker"));
    bool headshot = event.GetBool("headshot");

    char weapon[64];
    event.GetString("weapon", weapon, sizeof(weapon));

    if (!IsValidClient(victim)) return;

    if (IsValidClient(attacker) && attacker != victim) {
        char attackerName[MAX_NAME_LENGTH];
        char victimName[MAX_NAME_LENGTH];
        GetClientName(attacker, attackerName, sizeof(attackerName));
        GetClientName(victim, victimName, sizeof(victimName));

        if (headshot) {
            PrintToChatAll("%s headshot %s with %s!", attackerName, victimName, weapon);
        }
    }
}

bool IsValidClient(int client) {
    return client > 0 && client <= MaxClients && IsClientInGame(client);
}''',
    placeholders=["action", "tracking"]
)

register_template(
    name="event_round",
    description="Handle round start/end events",
    category="events",
    system_prompt_key="sourcepawn",
    user_template="Create round event handlers for {round_actions}.",
    example_output='''bool g_bRoundActive;

public void OnPluginStart() {
    HookEvent("round_start", Event_RoundStart);
    HookEvent("round_end", Event_RoundEnd);
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast) {
    g_bRoundActive = true;

    for (int i = 1; i <= MaxClients; i++) {
        if (IsClientInGame(i)) {
            ResetClientData(i);
        }
    }

    PrintToChatAll("Round started!");
}

public void Event_RoundEnd(Event event, const char[] name, bool dontBroadcast) {
    g_bRoundActive = false;

    int winner = event.GetInt("winner");
    PrintToChatAll("Round ended! Winner: Team %d", winner);
}

void ResetClientData(int client) {
    // Reset client-specific data
}''',
    placeholders=["round_actions"]
)

register_template(
    name="event_infected_death",
    description="Handle infected death events (common and special)",
    category="events",
    system_prompt_key="sourcepawn",
    user_template="Create an infected death handler that {action}.",
    example_output='''int g_iKillCount[MAXPLAYERS + 1];

public void OnPluginStart() {
    HookEvent("infected_death", Event_InfectedDeath);
    HookEvent("player_death", Event_PlayerDeath);
}

public void Event_InfectedDeath(Event event, const char[] name, bool dontBroadcast) {
    int attacker = GetClientOfUserId(event.GetInt("attacker"));
    bool headshot = event.GetBool("headshot");

    if (!IsValidClient(attacker)) return;
    if (GetClientTeam(attacker) != 2) return;

    g_iKillCount[attacker]++;

    if (headshot) {
        PrintHintText(attacker, "Headshot! Kills: %d", g_iKillCount[attacker]);
    }
}

public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast) {
    int victim = GetClientOfUserId(event.GetInt("userid"));
    int attacker = GetClientOfUserId(event.GetInt("attacker"));

    if (!IsValidClient(victim)) return;
    if (GetClientTeam(victim) != 3) return;

    if (IsValidClient(attacker) && GetClientTeam(attacker) == 2) {
        char className[32];
        GetEntityClassname(victim, className, sizeof(className));
        PrintToChat(attacker, "You killed a %s!", className);
    }
}

bool IsValidClient(int client) {
    return client > 0 && client <= MaxClients && IsClientInGame(client);
}''',
    placeholders=["action"]
)

register_template(
    name="event_weapon",
    description="Handle weapon pickup and fire events",
    category="events",
    system_prompt_key="sourcepawn",
    user_template="Create weapon event handlers for {weapon_actions}.",
    example_output='''public void OnPluginStart() {
    HookEvent("weapon_fire", Event_WeaponFire);
    HookEvent("item_pickup", Event_ItemPickup);
}

public void Event_WeaponFire(Event event, const char[] name, bool dontBroadcast) {
    int client = GetClientOfUserId(event.GetInt("userid"));

    if (!IsValidClient(client)) return;

    char weapon[64];
    event.GetString("weapon", weapon, sizeof(weapon));

    // Track ammo usage or fire rate
    if (StrContains(weapon, "shotgun") != -1) {
        // Shotgun fired
    }
}

public void Event_ItemPickup(Event event, const char[] name, bool dontBroadcast) {
    int client = GetClientOfUserId(event.GetInt("userid"));

    if (!IsValidClient(client)) return;

    char item[64];
    event.GetString("item", item, sizeof(item));

    PrintToChat(client, "You picked up: %s", item);
}

bool IsValidClient(int client) {
    return client > 0 && client <= MaxClients && IsClientInGame(client);
}''',
    placeholders=["weapon_actions"]
)

register_template(
    name="event_heal",
    description="Handle healing and revive events",
    category="events",
    system_prompt_key="sourcepawn",
    user_template="Create healing event handlers that {action}.",
    example_output='''public void OnPluginStart() {
    HookEvent("heal_success", Event_HealSuccess);
    HookEvent("revive_success", Event_ReviveSuccess);
    HookEvent("player_incapacitated", Event_Incapacitated);
}

public void Event_HealSuccess(Event event, const char[] name, bool dontBroadcast) {
    int healer = GetClientOfUserId(event.GetInt("userid"));
    int patient = GetClientOfUserId(event.GetInt("subject"));
    int amount = event.GetInt("health_restored");

    if (!IsValidClient(healer) || !IsValidClient(patient)) return;

    if (healer != patient) {
        char healerName[MAX_NAME_LENGTH];
        GetClientName(healer, healerName, sizeof(healerName));
        PrintToChat(patient, "%s healed you for %d HP!", healerName, amount);
    }
}

public void Event_ReviveSuccess(Event event, const char[] name, bool dontBroadcast) {
    int reviver = GetClientOfUserId(event.GetInt("userid"));
    int victim = GetClientOfUserId(event.GetInt("subject"));

    if (!IsValidClient(reviver) || !IsValidClient(victim)) return;

    PrintToChatAll("%N revived %N!", reviver, victim);
}

public void Event_Incapacitated(Event event, const char[] name, bool dontBroadcast) {
    int victim = GetClientOfUserId(event.GetInt("userid"));

    if (!IsValidClient(victim)) return;

    PrintToChatAll("%N is down!", victim);
}

bool IsValidClient(int client) {
    return client > 0 && client <= MaxClients && IsClientInGame(client);
}''',
    placeholders=["action"]
)

register_template(
    name="event_tank",
    description="Handle Tank spawn and death events",
    category="events",
    system_prompt_key="sourcepawn",
    user_template="Create Tank event handlers that {action}.",
    example_output='''bool g_bTankAlive;
int g_iTankClient;
float g_fTankSpawnTime;

public void OnPluginStart() {
    HookEvent("tank_spawn", Event_TankSpawn);
    HookEvent("player_death", Event_PlayerDeath);
}

public void Event_TankSpawn(Event event, const char[] name, bool dontBroadcast) {
    int tank = GetClientOfUserId(event.GetInt("userid"));

    if (!IsValidClient(tank)) return;

    g_bTankAlive = true;
    g_iTankClient = tank;
    g_fTankSpawnTime = GetGameTime();

    PrintToChatAll("TANK HAS SPAWNED!");
    EmitSoundToAll("ui/pickup_secret01.wav");
}

public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast) {
    int victim = GetClientOfUserId(event.GetInt("userid"));

    if (victim == g_iTankClient) {
        g_bTankAlive = false;
        float surviveTime = GetGameTime() - g_fTankSpawnTime;
        PrintToChatAll("Tank killed! Survived %.1f seconds", surviveTime);
        g_iTankClient = 0;
    }
}

bool IsValidClient(int client) {
    return client > 0 && client <= MaxClients && IsClientInGame(client);
}''',
    placeholders=["action"]
)

register_template(
    name="event_witch",
    description="Handle Witch startle and death events",
    category="events",
    system_prompt_key="sourcepawn",
    user_template="Create Witch event handlers that {action}.",
    example_output='''public void OnPluginStart() {
    HookEvent("witch_spawn", Event_WitchSpawn);
    HookEvent("witch_harasser_set", Event_WitchStartled);
    HookEvent("witch_killed", Event_WitchKilled);
}

public void Event_WitchSpawn(Event event, const char[] name, bool dontBroadcast) {
    int witch = event.GetInt("witchid");

    float pos[3];
    GetEntPropVector(witch, Prop_Send, "m_vecOrigin", pos);

    PrintToChatAll("Witch spawned at %.0f, %.0f, %.0f", pos[0], pos[1], pos[2]);
}

public void Event_WitchStartled(Event event, const char[] name, bool dontBroadcast) {
    int startler = GetClientOfUserId(event.GetInt("userid"));

    if (IsValidClient(startler)) {
        PrintToChatAll("%N startled the Witch!", startler);
    }
}

public void Event_WitchKilled(Event event, const char[] name, bool dontBroadcast) {
    int killer = GetClientOfUserId(event.GetInt("userid"));
    bool oneshot = event.GetBool("oneshot");

    if (IsValidClient(killer)) {
        if (oneshot) {
            PrintToChatAll("%N crowned the Witch!", killer);
        } else {
            PrintToChatAll("%N killed the Witch!", killer);
        }
    }
}

bool IsValidClient(int client) {
    return client > 0 && client <= MaxClients && IsClientInGame(client);
}''',
    placeholders=["action"]
)

register_template(
    name="event_map",
    description="Handle map start/end and transitions",
    category="events",
    system_prompt_key="sourcepawn",
    user_template="Create map event handlers for {map_actions}.",
    example_output='''char g_sCurrentMap[64];

public void OnPluginStart() {
    HookEvent("map_transition", Event_MapTransition);
    HookEvent("finale_start", Event_FinaleStart);
    HookEvent("finale_win", Event_FinaleWin);
}

public void OnMapStart() {
    GetCurrentMap(g_sCurrentMap, sizeof(g_sCurrentMap));
    PrintToServer("Map started: %s", g_sCurrentMap);

    PrecacheResources();
}

public void OnMapEnd() {
    // Cleanup resources
}

public void Event_MapTransition(Event event, const char[] name, bool dontBroadcast) {
    PrintToChatAll("Moving to next map...");
}

public void Event_FinaleStart(Event event, const char[] name, bool dontBroadcast) {
    PrintToChatAll("FINALE STARTED!");
}

public void Event_FinaleWin(Event event, const char[] name, bool dontBroadcast) {
    PrintToChatAll("Survivors escaped!");
}

void PrecacheResources() {
    // Precache models, sounds, etc.
}''',
    placeholders=["map_actions"]
)

register_template(
    name="event_safe_room",
    description="Handle safe room door events",
    category="events",
    system_prompt_key="sourcepawn",
    user_template="Create safe room event handlers that {action}.",
    example_output='''public void OnPluginStart() {
    HookEvent("door_open", Event_DoorOpen);
    HookEvent("door_close", Event_DoorClose);
}

public void Event_DoorOpen(Event event, const char[] name, bool dontBroadcast) {
    int client = GetClientOfUserId(event.GetInt("userid"));
    bool checkpoint = event.GetBool("checkpoint");

    if (!checkpoint) return;

    if (IsValidClient(client)) {
        PrintToChatAll("%N opened the safe room door!", client);
    }
}

public void Event_DoorClose(Event event, const char[] name, bool dontBroadcast) {
    bool checkpoint = event.GetBool("checkpoint");

    if (checkpoint) {
        PrintToChatAll("Safe room door closed!");
    }
}

bool IsValidClient(int client) {
    return client > 0 && client <= MaxClients && IsClientInGame(client);
}''',
    placeholders=["action"]
)

# =============================================================================
# CONVAR TEMPLATES (19-26)
# =============================================================================

register_template(
    name="convar_basic",
    description="Basic ConVar with change callback",
    category="convars",
    system_prompt_key="sourcepawn",
    user_template="Create a ConVar '{cvar_name}' with default '{default}' that controls {purpose}.",
    example_output='''ConVar g_cvEnabled;

public void OnPluginStart() {
    g_cvEnabled = CreateConVar("sm_myplugin_enabled", "1", "Enable the plugin", FCVAR_NOTIFY);
    g_cvEnabled.AddChangeHook(OnConVarChanged);
}

public void OnConVarChanged(ConVar convar, const char[] oldValue, const char[] newValue) {
    if (convar == g_cvEnabled) {
        bool enabled = g_cvEnabled.BoolValue;
        PrintToServer("Plugin %s", enabled ? "enabled" : "disabled");
    }
}''',
    placeholders=["cvar_name", "default", "purpose"]
)

register_template(
    name="convar_range",
    description="ConVar with min/max bounds",
    category="convars",
    system_prompt_key="sourcepawn",
    user_template="Create a ConVar for {purpose} with range {min_val} to {max_val}.",
    example_output='''ConVar g_cvValue;

public void OnPluginStart() {
    g_cvValue = CreateConVar("sm_value", "50", "Configuration value (1-100)", FCVAR_NOTIFY, true, 1.0, true, 100.0);
    g_cvValue.AddChangeHook(OnValueChanged);
}

public void OnValueChanged(ConVar convar, const char[] oldValue, const char[] newValue) {
    int value = g_cvValue.IntValue;
    PrintToServer("Value changed to %d", value);
}''',
    placeholders=["purpose", "min_val", "max_val"]
)

register_template(
    name="convar_multiple",
    description="Multiple related ConVars with configuration",
    category="convars",
    system_prompt_key="sourcepawn",
    user_template="Create multiple ConVars for configuring {feature} including {config_options}.",
    example_output='''ConVar g_cvEnabled;
ConVar g_cvDamage;
ConVar g_cvCooldown;
ConVar g_cvMaxUses;

public void OnPluginStart() {
    g_cvEnabled = CreateConVar("sm_feature_enabled", "1", "Enable the feature");
    g_cvDamage = CreateConVar("sm_feature_damage", "25", "Damage amount", _, true, 0.0, true, 100.0);
    g_cvCooldown = CreateConVar("sm_feature_cooldown", "5.0", "Cooldown in seconds", _, true, 0.0, true, 60.0);
    g_cvMaxUses = CreateConVar("sm_feature_max_uses", "3", "Maximum uses per round", _, true, 1.0, true, 10.0);

    AutoExecConfig(true, "feature_config");
}

void ApplyFeature(int client) {
    if (!g_cvEnabled.BoolValue) return;

    int damage = g_cvDamage.IntValue;
    float cooldown = g_cvCooldown.FloatValue;
    int maxUses = g_cvMaxUses.IntValue;

    // Apply feature with configured values
}''',
    placeholders=["feature", "config_options"]
)

register_template(
    name="convar_string",
    description="String ConVar for text configuration",
    category="convars",
    system_prompt_key="sourcepawn",
    user_template="Create a string ConVar for {purpose}.",
    example_output='''ConVar g_cvPrefix;
ConVar g_cvWebhookURL;

char g_sPrefix[64];
char g_sWebhookURL[256];

public void OnPluginStart() {
    g_cvPrefix = CreateConVar("sm_chat_prefix", "[Server]", "Chat message prefix");
    g_cvWebhookURL = CreateConVar("sm_webhook_url", "", "Discord webhook URL");

    g_cvPrefix.AddChangeHook(OnPrefixChanged);
    g_cvWebhookURL.AddChangeHook(OnWebhookChanged);

    // Get initial values
    g_cvPrefix.GetString(g_sPrefix, sizeof(g_sPrefix));
    g_cvWebhookURL.GetString(g_sWebhookURL, sizeof(g_sWebhookURL));
}

public void OnPrefixChanged(ConVar convar, const char[] oldValue, const char[] newValue) {
    strcopy(g_sPrefix, sizeof(g_sPrefix), newValue);
}

public void OnWebhookChanged(ConVar convar, const char[] oldValue, const char[] newValue) {
    strcopy(g_sWebhookURL, sizeof(g_sWebhookURL), newValue);
}''',
    placeholders=["purpose"]
)

register_template(
    name="convar_cached",
    description="ConVar with cached value for performance",
    category="convars",
    system_prompt_key="sourcepawn",
    user_template="Create cached ConVars for {feature} to optimize performance.",
    example_output='''ConVar g_cvEnabled;
ConVar g_cvMultiplier;

bool g_bEnabled;
float g_fMultiplier;

public void OnPluginStart() {
    g_cvEnabled = CreateConVar("sm_boost_enabled", "1", "Enable damage boost");
    g_cvMultiplier = CreateConVar("sm_boost_multiplier", "1.5", "Damage multiplier", _, true, 1.0, true, 5.0);

    g_cvEnabled.AddChangeHook(OnCvarChanged);
    g_cvMultiplier.AddChangeHook(OnCvarChanged);

    CacheConVars();
}

public void OnCvarChanged(ConVar convar, const char[] oldValue, const char[] newValue) {
    CacheConVars();
}

void CacheConVars() {
    g_bEnabled = g_cvEnabled.BoolValue;
    g_fMultiplier = g_cvMultiplier.FloatValue;
}

int ModifyDamage(int damage) {
    if (!g_bEnabled) return damage;
    return RoundFloat(damage * g_fMultiplier);
}''',
    placeholders=["feature"]
)

register_template(
    name="convar_flags",
    description="ConVar with custom flags",
    category="convars",
    system_prompt_key="sourcepawn",
    user_template="Create a ConVar with flags for {purpose}.",
    example_output='''ConVar g_cvVersion;
ConVar g_cvSecret;
ConVar g_cvProtected;

public void OnPluginStart() {
    // Version ConVar - notify clients, don't save to config
    g_cvVersion = CreateConVar("sm_myplugin_version", "1.0.0", "Plugin version",
        FCVAR_NOTIFY | FCVAR_DONTRECORD);

    // Secret ConVar - hidden from find results
    g_cvSecret = CreateConVar("sm_myplugin_secret", "", "Secret API key",
        FCVAR_PROTECTED | FCVAR_DONTRECORD);

    // Protected ConVar - value hidden in status
    g_cvProtected = CreateConVar("sm_myplugin_password", "", "Server password",
        FCVAR_PROTECTED);
}''',
    placeholders=["purpose"]
)

register_template(
    name="convar_hook_game",
    description="Hook and modify game ConVars",
    category="convars",
    system_prompt_key="sourcepawn",
    user_template="Hook and modify game ConVars for {purpose}.",
    example_output='''ConVar g_cvGameCvar;
float g_fOriginalValue;

public void OnPluginStart() {
    g_cvGameCvar = FindConVar("sv_maxspeed");

    if (g_cvGameCvar != null) {
        g_fOriginalValue = g_cvGameCvar.FloatValue;
        g_cvGameCvar.AddChangeHook(OnGameCvarChanged);
    }
}

public void OnPluginEnd() {
    if (g_cvGameCvar != null) {
        g_cvGameCvar.FloatValue = g_fOriginalValue;
    }
}

public void OnGameCvarChanged(ConVar convar, const char[] oldValue, const char[] newValue) {
    PrintToServer("sv_maxspeed changed from %s to %s", oldValue, newValue);
}

void SetCustomSpeed(float speed) {
    if (g_cvGameCvar != null) {
        g_cvGameCvar.FloatValue = speed;
    }
}''',
    placeholders=["purpose"]
)

register_template(
    name="convar_replicated",
    description="ConVar replicated to clients",
    category="convars",
    system_prompt_key="sourcepawn",
    user_template="Create a replicated ConVar for {purpose} visible to clients.",
    example_output='''ConVar g_cvGameMode;

public void OnPluginStart() {
    g_cvGameMode = CreateConVar("sm_gamemode", "normal", "Current game mode",
        FCVAR_NOTIFY | FCVAR_REPLICATED);

    g_cvGameMode.AddChangeHook(OnGameModeChanged);
}

public void OnGameModeChanged(ConVar convar, const char[] oldValue, const char[] newValue) {
    PrintToChatAll("Game mode changed to: %s", newValue);
    ApplyGameModeSettings(newValue);
}

void ApplyGameModeSettings(const char[] mode) {
    if (StrEqual(mode, "hard")) {
        // Apply hard mode settings
    } else if (StrEqual(mode, "easy")) {
        // Apply easy mode settings
    } else {
        // Normal mode
    }
}''',
    placeholders=["purpose"]
)

# =============================================================================
# NATIVE AND FORWARD TEMPLATES (27-32)
# =============================================================================

register_template(
    name="native_basic",
    description="Basic native function declaration",
    category="natives",
    system_prompt_key="sourcepawn",
    user_template="Create a native function that {functionality}.",
    example_output='''public APLRes AskPluginLoad2(Handle myself, bool late, char[] error, int err_max) {
    CreateNative("MyPlugin_GetHealth", Native_GetHealth);
    CreateNative("MyPlugin_SetHealth", Native_SetHealth);

    RegPluginLibrary("myplugin");
    return APLRes_Success;
}

public int Native_GetHealth(Handle plugin, int numParams) {
    int client = GetNativeCell(1);

    if (!IsValidClient(client)) {
        ThrowNativeError(SP_ERROR_NATIVE, "Invalid client index %d", client);
    }

    return GetClientHealth(client);
}

public int Native_SetHealth(Handle plugin, int numParams) {
    int client = GetNativeCell(1);
    int health = GetNativeCell(2);

    if (!IsValidClient(client)) {
        ThrowNativeError(SP_ERROR_NATIVE, "Invalid client index %d", client);
    }

    SetEntityHealth(client, health);
    return 0;
}

bool IsValidClient(int client) {
    return client > 0 && client <= MaxClients && IsClientInGame(client);
}''',
    placeholders=["functionality"]
)

register_template(
    name="native_string",
    description="Native with string parameters",
    category="natives",
    system_prompt_key="sourcepawn",
    user_template="Create a native that handles strings for {purpose}.",
    example_output='''public APLRes AskPluginLoad2(Handle myself, bool late, char[] error, int err_max) {
    CreateNative("MyPlugin_GetName", Native_GetName);
    CreateNative("MyPlugin_SetName", Native_SetName);

    RegPluginLibrary("myplugin");
    return APLRes_Success;
}

char g_sPlayerName[MAXPLAYERS + 1][64];

public int Native_GetName(Handle plugin, int numParams) {
    int client = GetNativeCell(1);
    int maxlen = GetNativeCell(3);

    if (!IsValidClient(client)) {
        ThrowNativeError(SP_ERROR_NATIVE, "Invalid client index %d", client);
    }

    SetNativeString(2, g_sPlayerName[client], maxlen);
    return strlen(g_sPlayerName[client]);
}

public int Native_SetName(Handle plugin, int numParams) {
    int client = GetNativeCell(1);

    if (!IsValidClient(client)) {
        ThrowNativeError(SP_ERROR_NATIVE, "Invalid client index %d", client);
    }

    GetNativeString(2, g_sPlayerName[client], sizeof(g_sPlayerName[]));
    return 0;
}

bool IsValidClient(int client) {
    return client > 0 && client <= MaxClients && IsClientInGame(client);
}''',
    placeholders=["purpose"]
)

register_template(
    name="forward_basic",
    description="Basic forward declaration and call",
    category="forwards",
    system_prompt_key="sourcepawn",
    user_template="Create a forward that notifies when {event_description}.",
    example_output='''GlobalForward g_fwdOnPlayerAction;

public APLRes AskPluginLoad2(Handle myself, bool late, char[] error, int err_max) {
    g_fwdOnPlayerAction = new GlobalForward("MyPlugin_OnPlayerAction",
        ET_Event, Param_Cell, Param_Cell, Param_String);

    return APLRes_Success;
}

void NotifyPlayerAction(int client, int action, const char[] details) {
    Action result;

    Call_StartForward(g_fwdOnPlayerAction);
    Call_PushCell(client);
    Call_PushCell(action);
    Call_PushString(details);
    Call_Finish(result);

    if (result == Plugin_Handled || result == Plugin_Stop) {
        // Action was blocked by another plugin
        return;
    }

    // Continue with action
}''',
    placeholders=["event_description"]
)

register_template(
    name="forward_private",
    description="Private forward for specific plugins",
    category="forwards",
    system_prompt_key="sourcepawn",
    user_template="Create a private forward for {purpose}.",
    example_output='''PrivateForward g_fwdOnPrivateEvent;

public APLRes AskPluginLoad2(Handle myself, bool late, char[] error, int err_max) {
    CreateNative("MyPlugin_RegisterCallback", Native_RegisterCallback);
    CreateNative("MyPlugin_UnregisterCallback", Native_UnregisterCallback);

    g_fwdOnPrivateEvent = new PrivateForward(ET_Ignore, Param_Cell, Param_Cell);

    return APLRes_Success;
}

public int Native_RegisterCallback(Handle plugin, int numParams) {
    Function callback = GetNativeFunction(1);

    if (!AddToForward(g_fwdOnPrivateEvent, plugin, callback)) {
        ThrowNativeError(SP_ERROR_NATIVE, "Failed to add callback");
    }

    return 0;
}

public int Native_UnregisterCallback(Handle plugin, int numParams) {
    Function callback = GetNativeFunction(1);

    RemoveFromForward(g_fwdOnPrivateEvent, plugin, callback);
    return 0;
}

void FirePrivateEvent(int param1, int param2) {
    Call_StartForward(g_fwdOnPrivateEvent);
    Call_PushCell(param1);
    Call_PushCell(param2);
    Call_Finish();
}''',
    placeholders=["purpose"]
)

register_template(
    name="native_array",
    description="Native with array parameters",
    category="natives",
    system_prompt_key="sourcepawn",
    user_template="Create a native that handles arrays for {purpose}.",
    example_output='''public APLRes AskPluginLoad2(Handle myself, bool late, char[] error, int err_max) {
    CreateNative("MyPlugin_GetPosition", Native_GetPosition);
    CreateNative("MyPlugin_SetPosition", Native_SetPosition);

    RegPluginLibrary("myplugin");
    return APLRes_Success;
}

float g_fPlayerPos[MAXPLAYERS + 1][3];

public int Native_GetPosition(Handle plugin, int numParams) {
    int client = GetNativeCell(1);

    if (!IsValidClient(client)) {
        ThrowNativeError(SP_ERROR_NATIVE, "Invalid client index %d", client);
    }

    SetNativeArray(2, g_fPlayerPos[client], 3);
    return 0;
}

public int Native_SetPosition(Handle plugin, int numParams) {
    int client = GetNativeCell(1);
    float pos[3];

    if (!IsValidClient(client)) {
        ThrowNativeError(SP_ERROR_NATIVE, "Invalid client index %d", client);
    }

    GetNativeArray(2, pos, 3);
    g_fPlayerPos[client] = pos;

    TeleportEntity(client, pos, NULL_VECTOR, NULL_VECTOR);
    return 0;
}

bool IsValidClient(int client) {
    return client > 0 && client <= MaxClients && IsClientInGame(client);
}''',
    placeholders=["purpose"]
)

register_template(
    name="forward_action",
    description="Forward with Action return type",
    category="forwards",
    system_prompt_key="sourcepawn",
    user_template="Create an action forward that can block {action_type}.",
    example_output='''GlobalForward g_fwdOnDamage;

public APLRes AskPluginLoad2(Handle myself, bool late, char[] error, int err_max) {
    g_fwdOnDamage = new GlobalForward("MyPlugin_OnDamage",
        ET_Hook, Param_Cell, Param_Cell, Param_CellByRef, Param_CellByRef);

    return APLRes_Success;
}

Action ProcessDamage(int victim, int attacker, int &damage, int &damagetype) {
    Action result = Plugin_Continue;

    Call_StartForward(g_fwdOnDamage);
    Call_PushCell(victim);
    Call_PushCell(attacker);
    Call_PushCellRef(damage);
    Call_PushCellRef(damagetype);
    Call_Finish(result);

    return result;
}

// Usage in SDKHook
public Action OnTakeDamage(int victim, int &attacker, int &inflictor,
                          float &damage, int &damagetype) {
    int iDamage = RoundFloat(damage);
    Action result = ProcessDamage(victim, attacker, iDamage, damagetype);

    if (result == Plugin_Changed) {
        damage = float(iDamage);
    }

    return result;
}''',
    placeholders=["action_type"]
)

# =============================================================================
# MENU TEMPLATES (33-40)
# =============================================================================

register_template(
    name="menu_basic",
    description="Basic menu with handler",
    category="menus",
    system_prompt_key="menu",
    user_template="Create a basic menu for {purpose}.",
    example_output='''public void OnPluginStart() {
    RegConsoleCmd("sm_menu", Cmd_Menu);
}

public Action Cmd_Menu(int client, int args) {
    if (!IsValidClient(client)) return Plugin_Handled;

    Menu menu = new Menu(MenuHandler_Main);
    menu.SetTitle("Main Menu");

    menu.AddItem("option1", "Option 1");
    menu.AddItem("option2", "Option 2");
    menu.AddItem("option3", "Option 3");

    menu.Display(client, MENU_TIME_FOREVER);
    return Plugin_Handled;
}

public int MenuHandler_Main(Menu menu, MenuAction action, int param1, int param2) {
    switch (action) {
        case MenuAction_Select: {
            char info[32];
            menu.GetItem(param2, info, sizeof(info));

            if (StrEqual(info, "option1")) {
                PrintToChat(param1, "You selected Option 1");
            } else if (StrEqual(info, "option2")) {
                PrintToChat(param1, "You selected Option 2");
            } else if (StrEqual(info, "option3")) {
                PrintToChat(param1, "You selected Option 3");
            }
        }
        case MenuAction_End: {
            delete menu;
        }
    }
    return 0;
}

bool IsValidClient(int client) {
    return client > 0 && client <= MaxClients && IsClientInGame(client);
}''',
    placeholders=["purpose"]
)

register_template(
    name="menu_paginated",
    description="Paginated menu for large lists",
    category="menus",
    system_prompt_key="menu",
    user_template="Create a paginated menu for displaying {items}.",
    example_output='''public Action Cmd_PlayerList(int client, int args) {
    if (!IsValidClient(client)) return Plugin_Handled;

    Menu menu = new Menu(MenuHandler_PlayerList);
    menu.SetTitle("Select a Player");

    char userId[16], name[MAX_NAME_LENGTH];

    for (int i = 1; i <= MaxClients; i++) {
        if (IsClientInGame(i)) {
            IntToString(GetClientUserId(i), userId, sizeof(userId));
            GetClientName(i, name, sizeof(name));
            menu.AddItem(userId, name);
        }
    }

    if (menu.ItemCount == 0) {
        PrintToChat(client, "No players found");
        delete menu;
        return Plugin_Handled;
    }

    menu.ExitButton = true;
    menu.Display(client, MENU_TIME_FOREVER);
    return Plugin_Handled;
}

public int MenuHandler_PlayerList(Menu menu, MenuAction action, int param1, int param2) {
    switch (action) {
        case MenuAction_Select: {
            char info[16];
            menu.GetItem(param2, info, sizeof(info));

            int target = GetClientOfUserId(StringToInt(info));
            if (IsValidClient(target)) {
                PrintToChat(param1, "Selected: %N", target);
            } else {
                PrintToChat(param1, "Player no longer available");
            }
        }
        case MenuAction_End: {
            delete menu;
        }
    }
    return 0;
}

bool IsValidClient(int client) {
    return client > 0 && client <= MaxClients && IsClientInGame(client);
}''',
    placeholders=["items"]
)

register_template(
    name="menu_confirmation",
    description="Confirmation dialog menu",
    category="menus",
    system_prompt_key="menu",
    user_template="Create a confirmation menu for {action}.",
    example_output='''int g_iPendingAction[MAXPLAYERS + 1];

void ShowConfirmMenu(int client, int actionId, const char[] message) {
    g_iPendingAction[client] = actionId;

    Menu menu = new Menu(MenuHandler_Confirm);
    menu.SetTitle("Confirm: %s", message);

    menu.AddItem("yes", "Yes, proceed");
    menu.AddItem("no", "No, cancel");

    menu.ExitButton = false;
    menu.Display(client, 30);
}

public int MenuHandler_Confirm(Menu menu, MenuAction action, int param1, int param2) {
    switch (action) {
        case MenuAction_Select: {
            char info[16];
            menu.GetItem(param2, info, sizeof(info));

            if (StrEqual(info, "yes")) {
                ExecutePendingAction(param1, g_iPendingAction[param1]);
            } else {
                PrintToChat(param1, "Action cancelled");
            }
            g_iPendingAction[param1] = 0;
        }
        case MenuAction_Cancel: {
            PrintToChat(param1, "Action cancelled (timeout)");
            g_iPendingAction[param1] = 0;
        }
        case MenuAction_End: {
            delete menu;
        }
    }
    return 0;
}

void ExecutePendingAction(int client, int actionId) {
    PrintToChat(client, "Executing action %d", actionId);
}''',
    placeholders=["action"]
)

register_template(
    name="menu_submenu",
    description="Menu with submenus",
    category="menus",
    system_prompt_key="menu",
    user_template="Create a menu system with submenus for {categories}.",
    example_output='''public Action Cmd_Settings(int client, int args) {
    ShowMainMenu(client);
    return Plugin_Handled;
}

void ShowMainMenu(int client) {
    Menu menu = new Menu(MenuHandler_Main);
    menu.SetTitle("Settings");

    menu.AddItem("gameplay", "Gameplay Settings");
    menu.AddItem("visual", "Visual Settings");
    menu.AddItem("audio", "Audio Settings");

    menu.Display(client, MENU_TIME_FOREVER);
}

public int MenuHandler_Main(Menu menu, MenuAction action, int param1, int param2) {
    if (action == MenuAction_Select) {
        char info[32];
        menu.GetItem(param2, info, sizeof(info));

        if (StrEqual(info, "gameplay")) {
            ShowGameplayMenu(param1);
        } else if (StrEqual(info, "visual")) {
            ShowVisualMenu(param1);
        } else if (StrEqual(info, "audio")) {
            ShowAudioMenu(param1);
        }
    } else if (action == MenuAction_End) {
        delete menu;
    }
    return 0;
}

void ShowGameplayMenu(int client) {
    Menu menu = new Menu(MenuHandler_Gameplay);
    menu.SetTitle("Gameplay Settings");

    menu.AddItem("difficulty", "Difficulty");
    menu.AddItem("autoaim", "Auto-Aim");
    menu.AddItem("back", "Back");

    menu.ExitBackButton = true;
    menu.Display(client, MENU_TIME_FOREVER);
}

public int MenuHandler_Gameplay(Menu menu, MenuAction action, int param1, int param2) {
    if (action == MenuAction_Select) {
        char info[32];
        menu.GetItem(param2, info, sizeof(info));

        if (StrEqual(info, "back")) {
            ShowMainMenu(param1);
        }
    } else if (action == MenuAction_Cancel && param2 == MenuCancel_ExitBack) {
        ShowMainMenu(param1);
    } else if (action == MenuAction_End) {
        delete menu;
    }
    return 0;
}

void ShowVisualMenu(int client) {
    // Similar implementation
}

void ShowAudioMenu(int client) {
    // Similar implementation
}''',
    placeholders=["categories"]
)

register_template(
    name="menu_voting",
    description="Voting menu system",
    category="menus",
    system_prompt_key="menu",
    user_template="Create a voting system for {vote_topic}.",
    example_output='''bool g_bVoteInProgress;
int g_iVoteYes;
int g_iVoteNo;
bool g_bVoted[MAXPLAYERS + 1];

public Action Cmd_Vote(int client, int args) {
    if (g_bVoteInProgress) {
        ReplyToCommand(client, "A vote is already in progress");
        return Plugin_Handled;
    }

    StartVote();
    return Plugin_Handled;
}

void StartVote() {
    g_bVoteInProgress = true;
    g_iVoteYes = 0;
    g_iVoteNo = 0;

    for (int i = 1; i <= MaxClients; i++) {
        g_bVoted[i] = false;
    }

    Menu menu = new Menu(MenuHandler_Vote);
    menu.SetTitle("Do you want to change the map?");
    menu.AddItem("yes", "Yes");
    menu.AddItem("no", "No");

    menu.ExitButton = false;
    menu.DisplayVoteToAll(30);

    CreateTimer(30.0, Timer_EndVote);
}

public int MenuHandler_Vote(Menu menu, MenuAction action, int param1, int param2) {
    if (action == MenuAction_VoteEnd) {
        // param1 = winning item
        char info[16];
        menu.GetItem(param1, info, sizeof(info));

        if (StrEqual(info, "yes")) {
            PrintToChatAll("Vote passed! Yes: %d, No: %d", g_iVoteYes, g_iVoteNo);
        } else {
            PrintToChatAll("Vote failed! Yes: %d, No: %d", g_iVoteYes, g_iVoteNo);
        }

        g_bVoteInProgress = false;
    } else if (action == MenuAction_VoteCancel) {
        PrintToChatAll("Vote cancelled");
        g_bVoteInProgress = false;
    } else if (action == MenuAction_End) {
        delete menu;
    }
    return 0;
}

public Action Timer_EndVote(Handle timer) {
    if (g_bVoteInProgress) {
        g_bVoteInProgress = false;
    }
    return Plugin_Stop;
}''',
    placeholders=["vote_topic"]
)

register_template(
    name="menu_panel",
    description="Panel for displaying info",
    category="menus",
    system_prompt_key="menu",
    user_template="Create a panel to display {information}.",
    example_output='''void ShowInfoPanel(int client) {
    Panel panel = new Panel();

    panel.SetTitle("Player Statistics");
    panel.DrawText(" ");

    char buffer[128];
    Format(buffer, sizeof(buffer), "Name: %N", client);
    panel.DrawText(buffer);

    Format(buffer, sizeof(buffer), "Health: %d", GetClientHealth(client));
    panel.DrawText(buffer);

    Format(buffer, sizeof(buffer), "Team: %d", GetClientTeam(client));
    panel.DrawText(buffer);

    panel.DrawText(" ");
    panel.DrawItem("Refresh");
    panel.DrawItem("Close");

    panel.Send(client, PanelHandler_Info, 30);
    delete panel;
}

public int PanelHandler_Info(Menu menu, MenuAction action, int param1, int param2) {
    if (action == MenuAction_Select) {
        if (param2 == 1) {
            ShowInfoPanel(param1);
        }
        // param2 == 2 means close, do nothing
    }
    return 0;
}''',
    placeholders=["information"]
)

register_template(
    name="menu_radio",
    description="Radio-style menu for quick selection",
    category="menus",
    system_prompt_key="menu",
    user_template="Create a radio menu for {quick_actions}.",
    example_output='''void ShowQuickMenu(int client) {
    Menu menu = new Menu(MenuHandler_Quick, MENU_ACTIONS_DEFAULT | MenuAction_DisplayItem);
    menu.SetTitle("Quick Actions");
    menu.OptionFlags |= MENUFLAG_NO_SOUND;

    menu.AddItem("heal", "Heal Self");
    menu.AddItem("ammo", "Get Ammo");
    menu.AddItem("teleport", "Teleport to Team");
    menu.AddItem("suicide", "Respawn");

    menu.Display(client, 10);
}

public int MenuHandler_Quick(Menu menu, MenuAction action, int param1, int param2) {
    switch (action) {
        case MenuAction_Select: {
            char info[32];
            menu.GetItem(param2, info, sizeof(info));

            if (StrEqual(info, "heal")) {
                SetEntityHealth(param1, 100);
                PrintToChat(param1, "Healed!");
            } else if (StrEqual(info, "ammo")) {
                GivePlayerAmmo(param1);
                PrintToChat(param1, "Ammo given!");
            } else if (StrEqual(info, "teleport")) {
                TeleportToTeam(param1);
            } else if (StrEqual(info, "suicide")) {
                ForcePlayerSuicide(param1);
            }
        }
        case MenuAction_DisplayItem: {
            char display[64], info[32];
            menu.GetItem(param2, info, sizeof(info), _, display, sizeof(display));

            // Add hotkey numbers
            char newDisplay[64];
            Format(newDisplay, sizeof(newDisplay), "[%d] %s", param2 + 1, display);
            return RedrawMenuItem(newDisplay);
        }
        case MenuAction_End: {
            delete menu;
        }
    }
    return 0;
}

void GivePlayerAmmo(int client) {
    // Implementation
}

void TeleportToTeam(int client) {
    // Implementation
}''',
    placeholders=["quick_actions"]
)

register_template(
    name="menu_dynamic",
    description="Dynamically generated menu based on data",
    category="menus",
    system_prompt_key="menu",
    user_template="Create a dynamic menu populated from {data_source}.",
    example_output='''ArrayList g_alItems;

public void OnPluginStart() {
    g_alItems = new ArrayList(ByteCountToCells(64));

    // Populate with sample data
    g_alItems.PushString("Item 1");
    g_alItems.PushString("Item 2");
    g_alItems.PushString("Item 3");

    RegConsoleCmd("sm_items", Cmd_Items);
}

public Action Cmd_Items(int client, int args) {
    ShowDynamicMenu(client);
    return Plugin_Handled;
}

void ShowDynamicMenu(int client) {
    Menu menu = new Menu(MenuHandler_Dynamic);
    menu.SetTitle("Available Items (%d)", g_alItems.Length);

    char item[64], itemId[16];

    for (int i = 0; i < g_alItems.Length; i++) {
        g_alItems.GetString(i, item, sizeof(item));
        IntToString(i, itemId, sizeof(itemId));
        menu.AddItem(itemId, item);
    }

    if (menu.ItemCount == 0) {
        menu.AddItem("", "No items available", ITEMDRAW_DISABLED);
    }

    menu.Display(client, MENU_TIME_FOREVER);
}

public int MenuHandler_Dynamic(Menu menu, MenuAction action, int param1, int param2) {
    if (action == MenuAction_Select) {
        char info[16];
        menu.GetItem(param2, info, sizeof(info));

        int index = StringToInt(info);
        if (index >= 0 && index < g_alItems.Length) {
            char item[64];
            g_alItems.GetString(index, item, sizeof(item));
            PrintToChat(param1, "Selected: %s", item);
        }
    } else if (action == MenuAction_End) {
        delete menu;
    }
    return 0;
}''',
    placeholders=["data_source"]
)

# =============================================================================
# DATABASE TEMPLATES (41-46)
# =============================================================================

register_template(
    name="database_connect",
    description="Database connection with error handling",
    category="database",
    system_prompt_key="database",
    user_template="Create database connection code for {database_purpose}.",
    example_output='''Database g_hDatabase;

public void OnPluginStart() {
    Database.Connect(OnDatabaseConnect, "storage");
}

public void OnDatabaseConnect(Database db, const char[] error, any data) {
    if (db == null) {
        LogError("Database connection failed: %s", error);
        SetFailState("Could not connect to database");
        return;
    }

    g_hDatabase = db;
    LogMessage("Database connected successfully");

    CreateTables();
}

void CreateTables() {
    char query[] = "CREATE TABLE IF NOT EXISTS player_data ("
        ... "steamid VARCHAR(32) PRIMARY KEY, "
        ... "name VARCHAR(64), "
        ... "points INT DEFAULT 0, "
        ... "last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        ... ")";

    g_hDatabase.Query(OnTableCreated, query);
}

public void OnTableCreated(Database db, DBResultSet results, const char[] error, any data) {
    if (results == null) {
        LogError("Failed to create table: %s", error);
        return;
    }

    LogMessage("Database tables ready");
}''',
    placeholders=["database_purpose"]
)

register_template(
    name="database_query",
    description="Parameterized database query",
    category="database",
    system_prompt_key="database",
    user_template="Create a database query for {query_purpose}.",
    example_output='''void GetPlayerData(int client) {
    if (g_hDatabase == null) return;

    char steamId[32];
    if (!GetClientAuthId(client, AuthId_Steam2, steamId, sizeof(steamId))) {
        return;
    }

    char query[256];
    g_hDatabase.Format(query, sizeof(query),
        "SELECT points, last_seen FROM player_data WHERE steamid = '%s'",
        steamId);

    g_hDatabase.Query(OnPlayerDataLoaded, query, GetClientUserId(client));
}

public void OnPlayerDataLoaded(Database db, DBResultSet results, const char[] error, any data) {
    int client = GetClientOfUserId(data);

    if (results == null) {
        LogError("Query failed: %s", error);
        return;
    }

    if (!IsValidClient(client)) return;

    if (results.FetchRow()) {
        int points = results.FetchInt(0);
        char lastSeen[64];
        results.FetchString(1, lastSeen, sizeof(lastSeen));

        PrintToChat(client, "Welcome back! Points: %d", points);
    } else {
        CreatePlayerData(client);
    }
}

void CreatePlayerData(int client) {
    char steamId[32], name[64], escapedName[129];
    GetClientAuthId(client, AuthId_Steam2, steamId, sizeof(steamId));
    GetClientName(client, name, sizeof(name));
    g_hDatabase.Escape(name, escapedName, sizeof(escapedName));

    char query[256];
    g_hDatabase.Format(query, sizeof(query),
        "INSERT INTO player_data (steamid, name) VALUES ('%s', '%s')",
        steamId, escapedName);

    g_hDatabase.Query(OnPlayerCreated, query);
}

public void OnPlayerCreated(Database db, DBResultSet results, const char[] error, any data) {
    if (results == null) {
        LogError("Insert failed: %s", error);
    }
}

bool IsValidClient(int client) {
    return client > 0 && client <= MaxClients && IsClientInGame(client);
}''',
    placeholders=["query_purpose"]
)

register_template(
    name="database_transaction",
    description="Database transaction for atomic operations",
    category="database",
    system_prompt_key="database",
    user_template="Create a database transaction for {transaction_purpose}.",
    example_output='''void TransferPoints(int from, int to, int amount) {
    if (g_hDatabase == null) return;

    char fromSteamId[32], toSteamId[32];
    GetClientAuthId(from, AuthId_Steam2, fromSteamId, sizeof(fromSteamId));
    GetClientAuthId(to, AuthId_Steam2, toSteamId, sizeof(toSteamId));

    Transaction txn = new Transaction();

    char query[256];

    // Deduct from sender
    g_hDatabase.Format(query, sizeof(query),
        "UPDATE player_data SET points = points - %d WHERE steamid = '%s' AND points >= %d",
        amount, fromSteamId, amount);
    txn.AddQuery(query);

    // Add to receiver
    g_hDatabase.Format(query, sizeof(query),
        "UPDATE player_data SET points = points + %d WHERE steamid = '%s'",
        amount, toSteamId);
    txn.AddQuery(query);

    DataPack pack = new DataPack();
    pack.WriteCell(GetClientUserId(from));
    pack.WriteCell(GetClientUserId(to));
    pack.WriteCell(amount);

    g_hDatabase.Execute(txn, OnTransferSuccess, OnTransferFailure, pack);
}

public void OnTransferSuccess(Database db, any data, int numQueries, DBResultSet[] results, any[] queryData) {
    DataPack pack = view_as<DataPack>(data);
    pack.Reset();

    int from = GetClientOfUserId(pack.ReadCell());
    int to = GetClientOfUserId(pack.ReadCell());
    int amount = pack.ReadCell();

    delete pack;

    if (IsValidClient(from)) {
        PrintToChat(from, "Transferred %d points successfully!", amount);
    }
    if (IsValidClient(to)) {
        PrintToChat(to, "Received %d points!", amount);
    }
}

public void OnTransferFailure(Database db, any data, int numQueries, const char[] error, int failIndex, any[] queryData) {
    DataPack pack = view_as<DataPack>(data);
    pack.Reset();

    int from = GetClientOfUserId(pack.ReadCell());

    delete pack;

    LogError("Transfer failed at query %d: %s", failIndex, error);

    if (IsValidClient(from)) {
        PrintToChat(from, "Transfer failed. Please try again.");
    }
}

bool IsValidClient(int client) {
    return client > 0 && client <= MaxClients && IsClientInGame(client);
}''',
    placeholders=["transaction_purpose"]
)

register_template(
    name="database_prepared",
    description="Prepared statements for repeated queries",
    category="database",
    system_prompt_key="database",
    user_template="Create prepared statements for {statement_purpose}.",
    example_output='''DBStatement g_hInsertStmt;
DBStatement g_hUpdateStmt;
DBStatement g_hSelectStmt;

public void OnDatabaseConnect(Database db, const char[] error, any data) {
    if (db == null) {
        SetFailState("Database connection failed: %s", error);
        return;
    }

    g_hDatabase = db;

    char queryError[256];

    g_hInsertStmt = SQL_PrepareQuery(db,
        "INSERT INTO player_stats (steamid, kills, deaths) VALUES (?, ?, ?)",
        queryError, sizeof(queryError));

    if (g_hInsertStmt == null) {
        LogError("Failed to prepare insert: %s", queryError);
    }

    g_hUpdateStmt = SQL_PrepareQuery(db,
        "UPDATE player_stats SET kills = kills + ?, deaths = deaths + ? WHERE steamid = ?",
        queryError, sizeof(queryError));

    if (g_hUpdateStmt == null) {
        LogError("Failed to prepare update: %s", queryError);
    }
}

void InsertPlayerStats(const char[] steamId, int kills, int deaths) {
    if (g_hInsertStmt == null) return;

    SQL_BindParamString(g_hInsertStmt, 0, steamId, false);
    SQL_BindParamInt(g_hInsertStmt, 1, kills);
    SQL_BindParamInt(g_hInsertStmt, 2, deaths);

    SQL_Execute(g_hInsertStmt);
}

void UpdatePlayerStats(const char[] steamId, int killsDelta, int deathsDelta) {
    if (g_hUpdateStmt == null) return;

    SQL_BindParamInt(g_hUpdateStmt, 0, killsDelta);
    SQL_BindParamInt(g_hUpdateStmt, 1, deathsDelta);
    SQL_BindParamString(g_hUpdateStmt, 2, steamId, false);

    SQL_Execute(g_hUpdateStmt);
}

public void OnPluginEnd() {
    delete g_hInsertStmt;
    delete g_hUpdateStmt;
    delete g_hSelectStmt;
}''',
    placeholders=["statement_purpose"]
)

register_template(
    name="database_sqlite",
    description="SQLite-specific database operations",
    category="database",
    system_prompt_key="database",
    user_template="Create SQLite database operations for {sqlite_purpose}.",
    example_output='''Database g_hDatabase;

public void OnPluginStart() {
    char error[256];
    g_hDatabase = SQLite_UseDatabase("myplugin", error, sizeof(error));

    if (g_hDatabase == null) {
        SetFailState("SQLite error: %s", error);
        return;
    }

    CreateTables();
}

void CreateTables() {
    char query[] = "CREATE TABLE IF NOT EXISTS cache ("
        ... "key TEXT PRIMARY KEY, "
        ... "value TEXT, "
        ... "expires INTEGER"
        ... ")";

    if (!SQL_FastQuery(g_hDatabase, query)) {
        char error[256];
        SQL_GetError(g_hDatabase, error, sizeof(error));
        LogError("Table creation failed: %s", error);
    }
}

void SetCache(const char[] key, const char[] value, int ttl) {
    int expires = GetTime() + ttl;

    char escapedKey[128], escapedValue[512];
    SQL_EscapeString(g_hDatabase, key, escapedKey, sizeof(escapedKey));
    SQL_EscapeString(g_hDatabase, value, escapedValue, sizeof(escapedValue));

    char query[768];
    Format(query, sizeof(query),
        "INSERT OR REPLACE INTO cache (key, value, expires) VALUES ('%s', '%s', %d)",
        escapedKey, escapedValue, expires);

    SQL_FastQuery(g_hDatabase, query);
}

bool GetCache(const char[] key, char[] value, int maxlen) {
    char escapedKey[128];
    SQL_EscapeString(g_hDatabase, key, escapedKey, sizeof(escapedKey));

    char query[256];
    Format(query, sizeof(query),
        "SELECT value FROM cache WHERE key = '%s' AND expires > %d",
        escapedKey, GetTime());

    DBResultSet results = SQL_Query(g_hDatabase, query);

    if (results == null) return false;

    bool found = false;
    if (results.FetchRow()) {
        results.FetchString(0, value, maxlen);
        found = true;
    }

    delete results;
    return found;
}''',
    placeholders=["sqlite_purpose"]
)

register_template(
    name="database_cleanup",
    description="Database cleanup and maintenance",
    category="database",
    system_prompt_key="database",
    user_template="Create database cleanup routines for {cleanup_purpose}.",
    example_output='''public void OnPluginStart() {
    CreateTimer(3600.0, Timer_Cleanup, _, TIMER_REPEAT);
}

public Action Timer_Cleanup(Handle timer) {
    if (g_hDatabase == null) return Plugin_Continue;

    CleanupOldRecords();
    OptimizeDatabase();

    return Plugin_Continue;
}

void CleanupOldRecords() {
    // Delete records older than 30 days
    int cutoff = GetTime() - (30 * 24 * 60 * 60);

    char query[256];
    Format(query, sizeof(query),
        "DELETE FROM player_data WHERE UNIX_TIMESTAMP(last_seen) < %d",
        cutoff);

    g_hDatabase.Query(OnCleanupComplete, query, 1);

    // Delete expired cache entries
    Format(query, sizeof(query),
        "DELETE FROM cache WHERE expires < %d",
        GetTime());

    g_hDatabase.Query(OnCleanupComplete, query, 2);
}

public void OnCleanupComplete(Database db, DBResultSet results, const char[] error, any data) {
    if (results == null) {
        LogError("Cleanup query %d failed: %s", data, error);
        return;
    }

    int affected = results.AffectedRows;
    if (affected > 0) {
        LogMessage("Cleanup %d: Removed %d records", data, affected);
    }
}

void OptimizeDatabase() {
    // For MySQL
    g_hDatabase.Query(OnOptimizeComplete, "OPTIMIZE TABLE player_data");
}

public void OnOptimizeComplete(Database db, DBResultSet results, const char[] error, any data) {
    if (results == null) {
        LogError("Optimize failed: %s", error);
    }
}''',
    placeholders=["cleanup_purpose"]
)

# =============================================================================
# TIMER TEMPLATES (47-52)
# =============================================================================

register_template(
    name="timer_basic",
    description="Basic timer with callback",
    category="timers",
    system_prompt_key="timer",
    user_template="Create a timer that {timer_action} every {interval} seconds.",
    example_output='''Handle g_hTimer;

public void OnPluginStart() {
    g_hTimer = CreateTimer(5.0, Timer_Action, _, TIMER_REPEAT);
}

public void OnPluginEnd() {
    delete g_hTimer;
}

public Action Timer_Action(Handle timer) {
    for (int i = 1; i <= MaxClients; i++) {
        if (IsClientInGame(i) && IsPlayerAlive(i)) {
            // Perform action on each alive player
            PrintHintText(i, "Timer tick!");
        }
    }

    return Plugin_Continue;
}''',
    placeholders=["timer_action", "interval"]
)

register_template(
    name="timer_datapack",
    description="Timer with DataPack for passing data",
    category="timers",
    system_prompt_key="timer",
    user_template="Create a timer that passes {data_type} data to the callback.",
    example_output='''void StartDelayedAction(int client, int value, const char[] message) {
    DataPack pack = new DataPack();
    pack.WriteCell(GetClientUserId(client));
    pack.WriteCell(value);
    pack.WriteString(message);

    CreateTimer(2.0, Timer_DelayedAction, pack, TIMER_DATA_HNDL_CLOSE);
}

public Action Timer_DelayedAction(Handle timer, DataPack pack) {
    pack.Reset();

    int client = GetClientOfUserId(pack.ReadCell());
    int value = pack.ReadCell();
    char message[256];
    pack.ReadString(message, sizeof(message));

    if (!IsValidClient(client)) {
        return Plugin_Stop;
    }

    PrintToChat(client, "%s (Value: %d)", message, value);
    return Plugin_Stop;
}

bool IsValidClient(int client) {
    return client > 0 && client <= MaxClients && IsClientInGame(client);
}''',
    placeholders=["data_type"]
)

register_template(
    name="timer_per_client",
    description="Per-client timer management",
    category="timers",
    system_prompt_key="timer",
    user_template="Create per-client timers for {client_action}.",
    example_output='''Handle g_hClientTimer[MAXPLAYERS + 1];
int g_iClientData[MAXPLAYERS + 1];

public void OnClientDisconnect(int client) {
    StopClientTimer(client);
    g_iClientData[client] = 0;
}

void StartClientTimer(int client, float interval) {
    StopClientTimer(client);

    g_hClientTimer[client] = CreateTimer(interval, Timer_ClientAction,
        GetClientUserId(client), TIMER_REPEAT | TIMER_FLAG_NO_MAPCHANGE);
}

void StopClientTimer(int client) {
    if (g_hClientTimer[client] != null) {
        delete g_hClientTimer[client];
        g_hClientTimer[client] = null;
    }
}

public Action Timer_ClientAction(Handle timer, int userId) {
    int client = GetClientOfUserId(userId);

    if (!IsValidClient(client)) {
        return Plugin_Stop;
    }

    g_iClientData[client]++;
    PrintHintText(client, "Count: %d", g_iClientData[client]);

    if (g_iClientData[client] >= 10) {
        g_hClientTimer[client] = null;
        return Plugin_Stop;
    }

    return Plugin_Continue;
}

bool IsValidClient(int client) {
    return client > 0 && client <= MaxClients && IsClientInGame(client);
}''',
    placeholders=["client_action"]
)

register_template(
    name="timer_countdown",
    description="Countdown timer with display",
    category="timers",
    system_prompt_key="timer",
    user_template="Create a countdown timer for {countdown_purpose}.",
    example_output='''int g_iCountdown;
Handle g_hCountdownTimer;

void StartCountdown(int seconds) {
    StopCountdown();

    g_iCountdown = seconds;
    g_hCountdownTimer = CreateTimer(1.0, Timer_Countdown, _, TIMER_REPEAT | TIMER_FLAG_NO_MAPCHANGE);

    DisplayCountdown();
}

void StopCountdown() {
    if (g_hCountdownTimer != null) {
        delete g_hCountdownTimer;
        g_hCountdownTimer = null;
    }
}

public Action Timer_Countdown(Handle timer) {
    g_iCountdown--;

    if (g_iCountdown <= 0) {
        g_hCountdownTimer = null;
        OnCountdownFinished();
        return Plugin_Stop;
    }

    DisplayCountdown();
    return Plugin_Continue;
}

void DisplayCountdown() {
    char display[32];
    int minutes = g_iCountdown / 60;
    int seconds = g_iCountdown % 60;

    if (minutes > 0) {
        Format(display, sizeof(display), "%d:%02d", minutes, seconds);
    } else {
        Format(display, sizeof(display), "%d", seconds);
    }

    PrintCenterTextAll("%s", display);

    if (g_iCountdown <= 10) {
        EmitSoundToAll("buttons/button17.wav");
    }
}

void OnCountdownFinished() {
    PrintToChatAll("Countdown finished!");
    EmitSoundToAll("ambient/alarms/klaxon1.wav");
}''',
    placeholders=["countdown_purpose"]
)

register_template(
    name="timer_frame",
    description="RequestFrame for next-frame execution",
    category="timers",
    system_prompt_key="timer",
    user_template="Create frame callbacks for {frame_action}.",
    example_output='''void ScheduleNextFrame(int client, int action) {
    DataPack pack = new DataPack();
    pack.WriteCell(GetClientUserId(client));
    pack.WriteCell(action);

    RequestFrame(Frame_ExecuteAction, pack);
}

public void Frame_ExecuteAction(DataPack pack) {
    pack.Reset();

    int client = GetClientOfUserId(pack.ReadCell());
    int action = pack.ReadCell();

    delete pack;

    if (!IsValidClient(client)) return;

    switch (action) {
        case 1: {
            // Teleport after spawn
            float pos[3] = {0.0, 0.0, 100.0};
            TeleportEntity(client, pos, NULL_VECTOR, NULL_VECTOR);
        }
        case 2: {
            // Give weapons after spawn
            GivePlayerItem(client, "weapon_autoshotgun");
        }
        case 3: {
            // Set health after damage
            SetEntityHealth(client, 100);
        }
    }
}

// Multi-frame delay
void DelayedAction(int client, int frames) {
    DataPack pack = new DataPack();
    pack.WriteCell(GetClientUserId(client));
    pack.WriteCell(frames);

    RequestFrame(Frame_DelayCounter, pack);
}

public void Frame_DelayCounter(DataPack pack) {
    pack.Reset();

    int userId = pack.ReadCell();
    int remaining = pack.ReadCell();

    if (remaining > 1) {
        pack.Reset();
        pack.WriteCell(userId);
        pack.WriteCell(remaining - 1);
        RequestFrame(Frame_DelayCounter, pack);
    } else {
        delete pack;

        int client = GetClientOfUserId(userId);
        if (IsValidClient(client)) {
            PrintToChat(client, "Delayed action complete!");
        }
    }
}

bool IsValidClient(int client) {
    return client > 0 && client <= MaxClients && IsClientInGame(client);
}''',
    placeholders=["frame_action"]
)

register_template(
    name="timer_cooldown",
    description="Cooldown system with timers",
    category="timers",
    system_prompt_key="timer",
    user_template="Create a cooldown system for {ability_name}.",
    example_output='''float g_fCooldownEnd[MAXPLAYERS + 1];

bool IsOnCooldown(int client) {
    return GetGameTime() < g_fCooldownEnd[client];
}

float GetRemainingCooldown(int client) {
    float remaining = g_fCooldownEnd[client] - GetGameTime();
    return remaining > 0.0 ? remaining : 0.0;
}

void StartCooldown(int client, float duration) {
    g_fCooldownEnd[client] = GetGameTime() + duration;

    CreateTimer(duration, Timer_CooldownEnd, GetClientUserId(client), TIMER_FLAG_NO_MAPCHANGE);
}

public Action Timer_CooldownEnd(Handle timer, int userId) {
    int client = GetClientOfUserId(userId);

    if (IsValidClient(client)) {
        PrintToChat(client, "Ability ready!");
        EmitSoundToClient(client, "buttons/button14.wav");
    }

    return Plugin_Stop;
}

public Action Cmd_UseAbility(int client, int args) {
    if (IsOnCooldown(client)) {
        PrintToChat(client, "On cooldown! %.1f seconds remaining", GetRemainingCooldown(client));
        return Plugin_Handled;
    }

    // Use ability
    PerformAbility(client);

    // Start 30 second cooldown
    StartCooldown(client, 30.0);
    PrintToChat(client, "Ability used! Cooldown: 30 seconds");

    return Plugin_Handled;
}

void PerformAbility(int client) {
    // Ability implementation
}

public void OnClientDisconnect(int client) {
    g_fCooldownEnd[client] = 0.0;
}

bool IsValidClient(int client) {
    return client > 0 && client <= MaxClients && IsClientInGame(client);
}''',
    placeholders=["ability_name"]
)

# =============================================================================
# VSCRIPT TEMPLATES (53-58)
# =============================================================================

register_template(
    name="vscript_director",
    description="Director script for custom difficulty",
    category="vscript",
    system_prompt_key="vscript",
    user_template="Create a Director script that modifies {director_settings}.",
    example_output='''// Custom Director Settings
DirectorOptions <-
{
    cm_CommonLimit = 30
    cm_DominatorLimit = 4
    cm_MaxSpecials = 8
    cm_TankLimit = 1
    cm_WitchLimit = 2

    SpecialInitialSpawnDelayMin = 30
    SpecialInitialSpawnDelayMax = 60

    SpecialRespawnInterval = 20.0

    MobSpawnMinTime = 60
    MobSpawnMaxTime = 120
    MobMinSize = 10
    MobMaxSize = 30

    TankHitDamageModifierCoop = 1.0

    SmokerLimit = 2
    BoomerLimit = 2
    HunterLimit = 2
    SpitterLimit = 2
    JockeyLimit = 2
    ChargerLimit = 2
}

function OnGameplayStart()
{
    printl("Custom Director activated")
}''',
    placeholders=["director_settings"]
)

register_template(
    name="vscript_mutation",
    description="Custom mutation/game mode script",
    category="vscript",
    system_prompt_key="vscript",
    user_template="Create a mutation script for {mutation_name} that {mutation_behavior}.",
    example_output='''MutationOptions <-
{
    ActiveChallenge = 1

    cm_CommonLimit = 0
    cm_DominatorLimit = 8
    cm_MaxSpecials = 8

    TempHealthDecayRate = 0.0

    function AllowTakeDamage(damageTable)
    {
        if (damageTable.Attacker && damageTable.Attacker.IsSurvivor())
        {
            damageTable.DamageDone = damageTable.DamageDone * 2
        }
        return true
    }

    function AllowBash(basher)
    {
        return true
    }
}

MutationState <-
{
    RoundStarted = false
    SpecialKills = 0
}

function OnGameEvent_round_start(params)
{
    MutationState.RoundStarted = true
    MutationState.SpecialKills = 0

    foreach (player in Players.AliveSurvivors())
    {
        player.SetHealthBuffer(50)
    }
}

function OnGameEvent_player_death(params)
{
    local victim = GetPlayerFromUserID(params.userid)

    if (victim && victim.IsInfected())
    {
        MutationState.SpecialKills++

        if (MutationState.SpecialKills % 10 == 0)
        {
            foreach (survivor in Players.AliveSurvivors())
            {
                survivor.GiveItem("pain_pills")
            }
        }
    }
}''',
    placeholders=["mutation_name", "mutation_behavior"]
)

register_template(
    name="vscript_spawn",
    description="Custom spawn control script",
    category="vscript",
    system_prompt_key="vscript",
    user_template="Create a spawn script for {spawn_type}.",
    example_output='''SpawnLocations <- []

function InitSpawnLocations()
{
    // Define spawn locations
    SpawnLocations.append(Vector(100, 200, 0))
    SpawnLocations.append(Vector(500, 600, 0))
    SpawnLocations.append(Vector(900, 1000, 0))
}

function SpawnZombieAtRandom(zombieType)
{
    if (SpawnLocations.len() == 0)
    {
        InitSpawnLocations()
    }

    local index = RandomInt(0, SpawnLocations.len() - 1)
    local pos = SpawnLocations[index]

    ZSpawn({ type = zombieType, pos = pos })
}

function SpawnHorde(count)
{
    for (local i = 0; i < count; i++)
    {
        SpawnZombieAtRandom(ZOMBIE_NORMAL)
    }
}

function SpawnSpecialWave()
{
    local specials = [ZOMBIE_HUNTER, ZOMBIE_SMOKER, ZOMBIE_BOOMER, ZOMBIE_CHARGER]

    foreach (special in specials)
    {
        SpawnZombieAtRandom(special)
    }
}

function Think()
{
    // Called every frame
    if (RandomInt(0, 1000) < 5)
    {
        SpawnZombieAtRandom(ZOMBIE_NORMAL)
    }
}''',
    placeholders=["spawn_type"]
)

register_template(
    name="vscript_entity",
    description="Entity manipulation in VScript",
    category="vscript",
    system_prompt_key="vscript",
    user_template="Create VScript for manipulating {entity_type} entities.",
    example_output='''function FindAllDoors()
{
    local doors = []
    local ent = null

    while ((ent = Entities.FindByClassname(ent, "prop_door_rotating")) != null)
    {
        doors.append(ent)
    }

    return doors
}

function LockAllDoors()
{
    local doors = FindAllDoors()

    foreach (door in doors)
    {
        door.Lock()
        door.SetFriction(1000)
    }

    printl("Locked " + doors.len() + " doors")
}

function UnlockAllDoors()
{
    local doors = FindAllDoors()

    foreach (door in doors)
    {
        door.Unlock()
        door.SetFriction(1)
    }
}

function SpawnItem(classname, position)
{
    local item = SpawnEntityFromTable(classname, {
        origin = position,
        angles = Vector(0, 0, 0)
    })

    return item
}

function RemoveAllWeapons(survivor)
{
    local weapons = ["weapon_rifle", "weapon_shotgun", "weapon_pistol"]

    foreach (weapon in weapons)
    {
        local ent = survivor.FirstChild()
        while (ent)
        {
            if (ent.GetClassname() == weapon)
            {
                ent.Destroy()
            }
            ent = ent.Next()
        }
    }
}''',
    placeholders=["entity_type"]
)

register_template(
    name="vscript_event",
    description="Event handling in VScript",
    category="vscript",
    system_prompt_key="vscript",
    user_template="Create VScript event handlers for {event_types}.",
    example_output='''// Event handlers
function OnGameEvent_player_hurt(params)
{
    local victim = GetPlayerFromUserID(params.userid)
    local attacker = GetPlayerFromUserID(params.attacker)
    local damage = params.dmg_health

    if (victim && victim.IsSurvivor())
    {
        if (damage > 20)
        {
            ClientPrint(victim, HUD_PRINTTALK, "Heavy hit! -" + damage + " HP")
        }
    }
}

function OnGameEvent_witch_harasser_set(params)
{
    local harasser = GetPlayerFromUserID(params.userid)

    if (harasser)
    {
        ClientPrint(null, HUD_PRINTTALK, harasser.GetPlayerName() + " startled the Witch!")
    }
}

function OnGameEvent_tank_spawn(params)
{
    local tank = GetPlayerFromUserID(params.userid)

    foreach (survivor in Players.AliveSurvivors())
    {
        ClientPrint(survivor, HUD_PRINTCENTER, "TANK INCOMING!")
    }
}

function OnGameEvent_mission_lost(params)
{
    printl("Mission failed!")

    // Reset state
    ResetMutationState()
}

function OnGameEvent_finale_win(params)
{
    printl("Survivors escaped!")

    foreach (survivor in Players.AliveSurvivors())
    {
        ClientPrint(survivor, HUD_PRINTCENTER, "VICTORY!")
    }
}

function ResetMutationState()
{
    // Reset any custom state
}''',
    placeholders=["event_types"]
)

register_template(
    name="vscript_logic",
    description="Logic_script entity integration",
    category="vscript",
    system_prompt_key="vscript",
    user_template="Create a logic_script for {logic_purpose}.",
    example_output='''// This script should be attached to a logic_script entity
// Name the entity: logic_custom

ScriptScope <- {}

function Precache()
{
    // Precache resources
    PrecacheModel("models/props/cs_assault/box_stack1.mdl")
    PrecacheSound("ambient/alarms/klaxon1.wav")
}

function Activate()
{
    printl("Logic script activated")

    // Initialize
    ScriptScope.Active <- true
    ScriptScope.Counter <- 0
}

function OnTimer()
{
    if (!ScriptScope.Active)
        return

    ScriptScope.Counter++

    if (ScriptScope.Counter >= 10)
    {
        TriggerEvent()
        ScriptScope.Counter = 0
    }
}

function TriggerEvent()
{
    // Trigger output
    DoEntFire("!self", "FireUser1", "", 0, null, null)

    // Play sound
    EmitSoundOn("ambient/alarms/klaxon1.wav", self)
}

function EnableScript()
{
    ScriptScope.Active = true
}

function DisableScript()
{
    ScriptScope.Active = false
}

// Input handlers
function InputEnable()
{
    EnableScript()
}

function InputDisable()
{
    DisableScript()
}''',
    placeholders=["logic_purpose"]
)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_template(name: str) -> Optional[PromptTemplate]:
    """Get a template by name."""
    for template in TEMPLATES:
        if template.name == name:
            return template
    return None


def get_templates_by_category(category: str) -> List[PromptTemplate]:
    """Get all templates in a category."""
    return [t for t in TEMPLATES if t.category == category]


def get_all_categories() -> List[str]:
    """Get all unique categories."""
    return list(set(t.category for t in TEMPLATES))


def list_templates(category: Optional[str] = None) -> None:
    """List all templates, optionally filtered by category."""
    categories = get_all_categories()

    if category:
        if category not in categories:
            print(f"Unknown category: {category}")
            print(f"Available categories: {', '.join(sorted(categories))}")
            return
        categories = [category]

    for cat in sorted(categories):
        templates = get_templates_by_category(cat)
        print(f"\n=== {cat.upper()} ({len(templates)} templates) ===")
        for t in sorted(templates, key=lambda x: x.name):
            print(f"  {t.name}: {t.description}")
            if t.placeholders:
                print(f"    Placeholders: {', '.join(t.placeholders)}")


def use_template(template_name: str, **kwargs) -> str:
    """Use a template with provided values."""
    template = get_template(template_name)
    if not template:
        raise ValueError(f"Template '{template_name}' not found")

    # Check for missing required placeholders
    missing = [p for p in template.placeholders if p not in kwargs]
    if missing:
        raise ValueError(f"Missing required placeholders: {', '.join(missing)}")

    return template.render(**kwargs)


def export_templates(output_path: str = None) -> Path:
    """Export all templates to JSON."""
    if output_path is None:
        output_path = "data/prompts/templates.json"

    data = {
        "version": "1.0.0",
        "template_count": len(TEMPLATES),
        "categories": get_all_categories(),
        "system_prompts": SYSTEM_PROMPTS,
        "templates": [t.to_dict() for t in TEMPLATES]
    }

    return safe_write_json(output_path, data, PROJECT_ROOT, indent=2)


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="L4D2 Prompt Engineering Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prompt_templates.py list
  python prompt_templates.py list --category menus
  python prompt_templates.py use plugin_base --name "My Plugin" --author "Developer" --purpose "manages player stats"
  python prompt_templates.py export
  python prompt_templates.py show plugin_base
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List available templates")
    list_parser.add_argument("--category", "-c", help="Filter by category")

    # Use command
    use_parser = subparsers.add_parser("use", help="Use a template")
    use_parser.add_argument("template", help="Template name")
    use_parser.add_argument("--output", "-o", help="Output file")
    # Dynamic arguments for placeholders will be parsed from remaining args

    # Show command
    show_parser = subparsers.add_parser("show", help="Show template details")
    show_parser.add_argument("template", help="Template name")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export templates to JSON")
    export_parser.add_argument("--output", "-o", default="data/prompts/templates.json",
                               help="Output file path")

    # Count command
    subparsers.add_parser("count", help="Show template count")

    # Parse known args first to get the command
    args, remaining = parser.parse_known_args()

    if args.command == "list":
        list_templates(args.category)
        print(f"\nTotal: {len(TEMPLATES)} templates")

    elif args.command == "use":
        template = get_template(args.template)
        if not template:
            print(f"Error: Template '{args.template}' not found")
            sys.exit(1)

        # Parse placeholder arguments from remaining args
        kwargs = {}
        i = 0
        while i < len(remaining):
            if remaining[i].startswith("--"):
                key = remaining[i][2:]
                if i + 1 < len(remaining) and not remaining[i + 1].startswith("--"):
                    kwargs[key] = remaining[i + 1]
                    i += 2
                else:
                    i += 1
            else:
                i += 1

        try:
            result = use_template(args.template, **kwargs)

            print("\n--- System Prompt ---")
            print(template.system_prompt)
            print("\n--- User Prompt ---")
            print(result)

            if args.output:
                from utils.security import safe_write_text
                safe_write_text(args.output, result, PROJECT_ROOT)
                print(f"\nWritten to {args.output}")

        except ValueError as e:
            print(f"Error: {e}")
            print(f"\nRequired placeholders for '{args.template}':")
            for p in template.placeholders:
                print(f"  --{p} <value>")
            sys.exit(1)

    elif args.command == "show":
        template = get_template(args.template)
        if not template:
            print(f"Error: Template '{args.template}' not found")
            sys.exit(1)

        print(f"Name: {template.name}")
        print(f"Category: {template.category}")
        print(f"Description: {template.description}")
        print(f"Placeholders: {', '.join(template.placeholders) if template.placeholders else 'None'}")
        print(f"\n--- System Prompt ---")
        print(template.system_prompt)
        print(f"\n--- User Template ---")
        print(template.user_template)
        print(f"\n--- Example Output ---")
        print(template.example_output)

    elif args.command == "export":
        output = export_templates(args.output)
        print(f"Exported {len(TEMPLATES)} templates to {output}")

    elif args.command == "count":
        print(f"Total templates: {len(TEMPLATES)}")
        for cat in sorted(get_all_categories()):
            count = len(get_templates_by_category(cat))
            print(f"  {cat}: {count}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
