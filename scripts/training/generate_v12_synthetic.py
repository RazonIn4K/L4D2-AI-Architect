#!/usr/bin/env python3
"""
Generate V12 Synthetic Training Examples for L4D2 SourcePawn Fine-tuning

Focuses on UNDERREPRESENTED areas identified in training data analysis:
1. Multi-tank spawn handling
2. Witch avoidance mechanics
3. Advanced event hooks (OnMapStart, OnPluginStart, OnClientDisconnect)
4. Network forwarding and SQL database interactions
5. Custom menus and HUD elements
6. Inter-plugin communication (forwards, natives)

This script generates examples locally using patterns, without requiring OpenAI API.
"""

import json
import random
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent

SYSTEM_PROMPT = "You are an expert SourcePawn developer for Left 4 Dead 2 SourceMod plugins. Write clean, well-documented code with proper error handling. Use correct L4D2 APIs like GetRandomFloat (not RandomFloat), m_flLaggedMovementValue for speed, and proper event names."

# ============================================================================
# TEMPLATE DEFINITIONS - 30+ diverse templates covering underrepresented areas
# ============================================================================

TEMPLATES = {
    # === MULTI-TANK SPAWN HANDLING ===
    "multi_tank_basic": {
        "user": "Write a plugin that handles multiple Tank spawns simultaneously and tracks each Tank's health",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

#define MAX_TANKS 4

int g_iTankHealth[MAX_TANKS];
int g_iTankClient[MAX_TANKS];
int g_iTankCount = 0;

public Plugin myinfo =
{
    name = "Multi-Tank Tracker",
    author = "Developer",
    description = "Tracks multiple Tanks simultaneously",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("tank_spawn", Event_TankSpawn);
    HookEvent("player_death", Event_PlayerDeath);
    HookEvent("round_start", Event_RoundStart);

    RegConsoleCmd("sm_tanks", Command_ShowTanks, "Show active tanks");
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
    ResetTankTracking();
}

void ResetTankTracking()
{
    g_iTankCount = 0;
    for (int i = 0; i < MAX_TANKS; i++)
    {
        g_iTankHealth[i] = 0;
        g_iTankClient[i] = 0;
    }
}

public void Event_TankSpawn(Event event, const char[] name, bool dontBroadcast)
{
    int tank = GetClientOfUserId(event.GetInt("userid"));
    if (tank <= 0 || !IsClientInGame(tank))
        return;

    if (g_iTankCount < MAX_TANKS)
    {
        g_iTankClient[g_iTankCount] = tank;
        g_iTankHealth[g_iTankCount] = GetClientHealth(tank);
        g_iTankCount++;

        PrintToChatAll("\\x04[TANK] \\x01Tank #%d has spawned! Total active: %d", g_iTankCount, g_iTankCount);
    }
}

public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast)
{
    int victim = GetClientOfUserId(event.GetInt("userid"));

    for (int i = 0; i < g_iTankCount; i++)
    {
        if (g_iTankClient[i] == victim)
        {
            PrintToChatAll("\\x04[TANK] \\x01Tank #%d has been killed!", i + 1);

            // Shift remaining tanks down
            for (int j = i; j < g_iTankCount - 1; j++)
            {
                g_iTankClient[j] = g_iTankClient[j + 1];
                g_iTankHealth[j] = g_iTankHealth[j + 1];
            }
            g_iTankCount--;
            break;
        }
    }
}

Action Command_ShowTanks(int client, int args)
{
    if (g_iTankCount == 0)
    {
        ReplyToCommand(client, "[SM] No active Tanks.");
        return Plugin_Handled;
    }

    ReplyToCommand(client, "[SM] Active Tanks: %d", g_iTankCount);
    for (int i = 0; i < g_iTankCount; i++)
    {
        if (IsClientInGame(g_iTankClient[i]))
        {
            int health = GetClientHealth(g_iTankClient[i]);
            ReplyToCommand(client, "  Tank #%d: %d HP", i + 1, health);
        }
    }
    return Plugin_Handled;
}'''
    },

    "multi_tank_damage_split": {
        "user": "Create a plugin that splits damage evenly across all active Tanks when one is attacked",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>

ArrayList g_hActiveTanks;

public Plugin myinfo =
{
    name = "Tank Damage Splitter",
    author = "Developer",
    description = "Splits damage across all active Tanks",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    g_hActiveTanks = new ArrayList();

    HookEvent("tank_spawn", Event_TankSpawn);
    HookEvent("player_death", Event_TankDeath);
    HookEvent("round_start", Event_RoundStart);
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
    g_hActiveTanks.Clear();
}

public void Event_TankSpawn(Event event, const char[] name, bool dontBroadcast)
{
    int tank = GetClientOfUserId(event.GetInt("userid"));
    if (tank > 0 && IsClientInGame(tank))
    {
        g_hActiveTanks.Push(GetClientUserId(tank));
        SDKHook(tank, SDKHook_OnTakeDamage, OnTankTakeDamage);
    }
}

public void Event_TankDeath(Event event, const char[] name, bool dontBroadcast)
{
    int userid = event.GetInt("userid");
    int index = g_hActiveTanks.FindValue(userid);
    if (index != -1)
    {
        g_hActiveTanks.Erase(index);
    }
}

Action OnTankTakeDamage(int victim, int &attacker, int &inflictor, float &damage, int &damagetype)
{
    int tankCount = g_hActiveTanks.Length;
    if (tankCount <= 1)
        return Plugin_Continue;

    // Split damage across all tanks
    float splitDamage = damage / float(tankCount);

    for (int i = 0; i < tankCount; i++)
    {
        int tank = GetClientOfUserId(g_hActiveTanks.Get(i));
        if (tank > 0 && tank != victim && IsClientInGame(tank) && IsPlayerAlive(tank))
        {
            int currentHealth = GetClientHealth(tank);
            int newHealth = currentHealth - RoundToFloor(splitDamage);
            if (newHealth > 0)
            {
                SetEntityHealth(tank, newHealth);
            }
        }
    }

    // Reduce original damage to split amount
    damage = splitDamage;
    return Plugin_Changed;
}'''
    },

    "multi_tank_spawn_queue": {
        "user": "Write a plugin that queues Tank spawns with a configurable delay between each spawn",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

ConVar g_cvMaxTanks;
ConVar g_cvSpawnDelay;
ConVar g_cvEnabled;

int g_iTanksToSpawn = 0;
int g_iActiveTanks = 0;
Handle g_hSpawnTimer = null;

public Plugin myinfo =
{
    name = "Tank Spawn Queue",
    author = "Developer",
    description = "Queues Tank spawns with delays",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    g_cvMaxTanks = CreateConVar("sm_tankqueue_max", "3", "Maximum tanks to spawn", FCVAR_NOTIFY, true, 1.0, true, 8.0);
    g_cvSpawnDelay = CreateConVar("sm_tankqueue_delay", "10.0", "Delay between tank spawns", FCVAR_NOTIFY, true, 5.0, true, 60.0);
    g_cvEnabled = CreateConVar("sm_tankqueue_enabled", "1", "Enable tank queue system", FCVAR_NOTIFY, true, 0.0, true, 1.0);

    RegAdminCmd("sm_queuetanks", Command_QueueTanks, ADMFLAG_CHEATS, "Queue multiple tank spawns");

    HookEvent("tank_spawn", Event_TankSpawn);
    HookEvent("player_death", Event_PlayerDeath);
    HookEvent("round_start", Event_RoundStart);
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
    g_iTanksToSpawn = 0;
    g_iActiveTanks = 0;
    StopSpawnTimer();
}

void StopSpawnTimer()
{
    if (g_hSpawnTimer != null)
    {
        KillTimer(g_hSpawnTimer);
        g_hSpawnTimer = null;
    }
}

public void Event_TankSpawn(Event event, const char[] name, bool dontBroadcast)
{
    g_iActiveTanks++;
}

public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast)
{
    int victim = GetClientOfUserId(event.GetInt("userid"));
    if (victim > 0 && IsClientInGame(victim) && GetClientTeam(victim) == 3)
    {
        int zombieClass = GetEntProp(victim, Prop_Send, "m_zombieClass");
        if (zombieClass == 8) // Tank
        {
            g_iActiveTanks--;
            if (g_iActiveTanks < 0) g_iActiveTanks = 0;
        }
    }
}

Action Command_QueueTanks(int client, int args)
{
    if (args < 1)
    {
        ReplyToCommand(client, "[SM] Usage: sm_queuetanks <count>");
        return Plugin_Handled;
    }

    char arg[8];
    GetCmdArg(1, arg, sizeof(arg));
    int count = StringToInt(arg);

    if (count < 1 || count > g_cvMaxTanks.IntValue)
    {
        ReplyToCommand(client, "[SM] Invalid count. Max: %d", g_cvMaxTanks.IntValue);
        return Plugin_Handled;
    }

    g_iTanksToSpawn = count;
    SpawnNextTank();

    ReplyToCommand(client, "[SM] Queued %d tanks for spawning.", count);
    return Plugin_Handled;
}

void SpawnNextTank()
{
    if (g_iTanksToSpawn <= 0)
        return;

    // Spawn tank using director
    int director = FindEntityByClassname(-1, "info_director");
    if (director != -1)
    {
        AcceptEntityInput(director, "SpawnTank");
        g_iTanksToSpawn--;

        PrintToChatAll("\\x04[TANK] \\x01Tank spawned! %d remaining in queue.", g_iTanksToSpawn);

        if (g_iTanksToSpawn > 0)
        {
            float delay = g_cvSpawnDelay.FloatValue;
            g_hSpawnTimer = CreateTimer(delay, Timer_SpawnNext);
        }
    }
}

Action Timer_SpawnNext(Handle timer)
{
    g_hSpawnTimer = null;
    SpawnNextTank();
    return Plugin_Stop;
}'''
    },

    # === WITCH AVOIDANCE MECHANICS ===
    "witch_proximity_warning": {
        "user": "Create a plugin that warns survivors when they get too close to a Witch with distance-based alerts",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

ConVar g_cvWarningDistance;
ConVar g_cvDangerDistance;
Handle g_hCheckTimer = null;

public Plugin myinfo =
{
    name = "Witch Proximity Warning",
    author = "Developer",
    description = "Warns survivors about nearby Witches",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    g_cvWarningDistance = CreateConVar("sm_witch_warning", "500.0", "Distance for witch warning", FCVAR_NOTIFY, true, 100.0, true, 2000.0);
    g_cvDangerDistance = CreateConVar("sm_witch_danger", "200.0", "Distance for danger alert", FCVAR_NOTIFY, true, 50.0, true, 500.0);

    HookEvent("round_start", Event_RoundStart);
    HookEvent("round_end", Event_RoundEnd);
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
    StartProximityCheck();
}

public void Event_RoundEnd(Event event, const char[] name, bool dontBroadcast)
{
    StopProximityCheck();
}

void StartProximityCheck()
{
    StopProximityCheck();
    g_hCheckTimer = CreateTimer(0.5, Timer_CheckProximity, _, TIMER_REPEAT);
}

void StopProximityCheck()
{
    if (g_hCheckTimer != null)
    {
        KillTimer(g_hCheckTimer);
        g_hCheckTimer = null;
    }
}

Action Timer_CheckProximity(Handle timer)
{
    float warningDist = g_cvWarningDistance.FloatValue;
    float dangerDist = g_cvDangerDistance.FloatValue;

    // Find all witches
    int witch = -1;
    while ((witch = FindEntityByClassname(witch, "witch")) != -1)
    {
        float witchPos[3];
        GetEntPropVector(witch, Prop_Send, "m_vecOrigin", witchPos);

        // Check each survivor
        for (int i = 1; i <= MaxClients; i++)
        {
            if (!IsClientInGame(i) || GetClientTeam(i) != 2 || !IsPlayerAlive(i))
                continue;

            float survivorPos[3];
            GetClientAbsOrigin(i, survivorPos);

            float distance = GetVectorDistance(survivorPos, witchPos);

            if (distance <= dangerDist)
            {
                PrintCenterText(i, "!! DANGER - WITCH VERY CLOSE !!");
                EmitSoundToClient(i, "player/heartbeatloop.wav", _, _, _, _, 0.8);
            }
            else if (distance <= warningDist)
            {
                PrintCenterText(i, "Warning: Witch nearby (%.0f units)", distance);
            }
        }
    }

    return Plugin_Continue;
}'''
    },

    "witch_startled_tracker": {
        "user": "Write a plugin that tracks who startled a Witch and applies a team penalty",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

ConVar g_cvPenaltyAmount;
int g_iWitchStartles[MAXPLAYERS + 1];

public Plugin myinfo =
{
    name = "Witch Startle Tracker",
    author = "Developer",
    description = "Tracks and penalizes witch startles",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    g_cvPenaltyAmount = CreateConVar("sm_witch_penalty", "25", "Health penalty for startling witch", FCVAR_NOTIFY, true, 0.0, true, 100.0);

    HookEvent("witch_harasser_set", Event_WitchStartled);
    HookEvent("round_start", Event_RoundStart);

    RegConsoleCmd("sm_witchstats", Command_WitchStats, "Show witch startle stats");
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
    for (int i = 1; i <= MaxClients; i++)
    {
        g_iWitchStartles[i] = 0;
    }
}

public void Event_WitchStartled(Event event, const char[] name, bool dontBroadcast)
{
    int harasser = GetClientOfUserId(event.GetInt("userid"));

    if (harasser <= 0 || !IsClientInGame(harasser))
        return;

    g_iWitchStartles[harasser]++;

    char harasserName[64];
    GetClientName(harasser, harasserName, sizeof(harasserName));

    PrintToChatAll("\\x04[WITCH] \\x03%s \\x01startled a Witch! (Total: %d)", harasserName, g_iWitchStartles[harasser]);

    // Apply health penalty to the harasser
    int penalty = g_cvPenaltyAmount.IntValue;
    if (penalty > 0 && IsPlayerAlive(harasser))
    {
        int health = GetClientHealth(harasser);
        int newHealth = health - penalty;
        if (newHealth < 1) newHealth = 1;

        SetEntityHealth(harasser, newHealth);
        PrintToChat(harasser, "\\x04[PENALTY] \\x01You lost %d HP for startling a Witch!", penalty);
    }
}

Action Command_WitchStats(int client, int args)
{
    PrintToChat(client, "\\x04=== Witch Startle Stats ===");

    for (int i = 1; i <= MaxClients; i++)
    {
        if (IsClientInGame(i) && g_iWitchStartles[i] > 0)
        {
            char name[64];
            GetClientName(i, name, sizeof(name));
            PrintToChat(client, "\\x03%s\\x01: %d startles", name, g_iWitchStartles[i]);
        }
    }

    return Plugin_Handled;
}'''
    },

    "witch_cr0wn_detector": {
        "user": "Create a plugin that detects and announces successful Witch crowns (one-shot kills)",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

int g_iCrownCount[MAXPLAYERS + 1];
int g_iLastWitchAttacker = -1;
int g_iLastWitchHealth = 0;

public Plugin myinfo =
{
    name = "Witch Crown Detector",
    author = "Developer",
    description = "Detects and announces successful witch crowns",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("witch_killed", Event_WitchKilled);
    HookEvent("witch_harasser_set", Event_WitchStartled);
    HookEvent("round_start", Event_RoundStart);

    RegConsoleCmd("sm_crowns", Command_ShowCrowns, "Show crown statistics");
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
    for (int i = 1; i <= MaxClients; i++)
    {
        g_iCrownCount[i] = 0;
    }
}

public void Event_WitchStartled(Event event, const char[] name, bool dontBroadcast)
{
    g_iLastWitchAttacker = GetClientOfUserId(event.GetInt("userid"));

    int witch = event.GetInt("witchid");
    if (IsValidEntity(witch))
    {
        g_iLastWitchHealth = GetEntProp(witch, Prop_Data, "m_iHealth");
    }
}

public void Event_WitchKilled(Event event, const char[] name, bool dontBroadcast)
{
    int killer = GetClientOfUserId(event.GetInt("userid"));
    bool oneShot = event.GetBool("oneshot");

    if (killer <= 0 || !IsClientInGame(killer))
        return;

    char killerName[64];
    GetClientName(killer, killerName, sizeof(killerName));

    if (oneShot)
    {
        // Perfect crown - killed before she could attack
        g_iCrownCount[killer]++;

        PrintToChatAll("\\x04[CROWN] \\x03%s \\x01executed a PERFECT CROWN! (Total: %d)", killerName, g_iCrownCount[killer]);
        EmitSoundToAll("ui/pickup_secret01.wav");

        // Bonus: give temporary speed boost
        SetEntPropFloat(killer, Prop_Send, "m_flLaggedMovementValue", 1.2);
        CreateTimer(3.0, Timer_ResetSpeed, GetClientUserId(killer));
    }
    else if (killer == g_iLastWitchAttacker)
    {
        // Killed after startling but survived
        PrintToChatAll("\\x04[WITCH] \\x03%s \\x01killed the Witch after startling her!", killerName);
    }
    else
    {
        PrintToChatAll("\\x04[WITCH] \\x03%s \\x01killed the Witch!", killerName);
    }
}

Action Timer_ResetSpeed(Handle timer, int userid)
{
    int client = GetClientOfUserId(userid);
    if (client > 0 && IsClientInGame(client) && IsPlayerAlive(client))
    {
        SetEntPropFloat(client, Prop_Send, "m_flLaggedMovementValue", 1.0);
    }
    return Plugin_Stop;
}

Action Command_ShowCrowns(int client, int args)
{
    PrintToChat(client, "\\x04=== Crown Statistics ===");

    bool hasCrowns = false;
    for (int i = 1; i <= MaxClients; i++)
    {
        if (IsClientInGame(i) && g_iCrownCount[i] > 0)
        {
            char name[64];
            GetClientName(i, name, sizeof(name));
            PrintToChat(client, "\\x03%s\\x01: %d crowns", name, g_iCrownCount[i]);
            hasCrowns = true;
        }
    }

    if (!hasCrowns)
    {
        PrintToChat(client, "No crowns this round.");
    }

    return Plugin_Handled;
}'''
    },

    # === ADVANCED EVENT HOOKS ===
    "onmapstart_initialization": {
        "user": "Write a plugin that uses OnMapStart to initialize map-specific settings and spawn custom entities",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

StringMap g_hMapSettings;
bool g_bMapLoaded = false;

public Plugin myinfo =
{
    name = "Map Initializer",
    author = "Developer",
    description = "Initializes map-specific settings on map start",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    g_hMapSettings = new StringMap();
    LoadMapConfigurations();
}

public void OnMapStart()
{
    g_bMapLoaded = true;

    char mapName[64];
    GetCurrentMap(mapName, sizeof(mapName));

    LogMessage("Map starting: %s", mapName);

    // Initialize map-specific settings
    InitializeMapSettings(mapName);

    // Precache custom sounds and models
    PrecacheCustomAssets();

    // Create map-specific entities after a short delay
    CreateTimer(1.0, Timer_SpawnCustomEntities, _, TIMER_FLAG_NO_MAPCHANGE);
}

public void OnMapEnd()
{
    g_bMapLoaded = false;
}

void LoadMapConfigurations()
{
    // Store map-specific configurations
    g_hMapSettings.SetValue("c1m1_hotel", 1);    // Campaign 1 Map 1
    g_hMapSettings.SetValue("c2m1_highway", 2);  // Campaign 2 Map 1
    g_hMapSettings.SetValue("c5m1_waterfront", 3);
}

void InitializeMapSettings(const char[] mapName)
{
    int mapType;
    if (g_hMapSettings.GetValue(mapName, mapType))
    {
        switch (mapType)
        {
            case 1:
            {
                // Dead Center settings
                PrintToServer("[MAP] Loading Dead Center configuration");
                ServerCommand("z_common_limit 25");
            }
            case 2:
            {
                // Dark Carnival settings
                PrintToServer("[MAP] Loading Dark Carnival configuration");
                ServerCommand("z_common_limit 30");
            }
            case 3:
            {
                // The Parish settings
                PrintToServer("[MAP] Loading The Parish configuration");
                ServerCommand("z_common_limit 28");
            }
        }
    }
    else
    {
        PrintToServer("[MAP] Using default configuration for %s", mapName);
    }
}

void PrecacheCustomAssets()
{
    // Precache sounds
    PrecacheSound("ui/pickup_secret01.wav", true);
    PrecacheSound("player/heartbeatloop.wav", true);

    // Precache models (example)
    if (FileExists("models/custom/my_model.mdl"))
    {
        PrecacheModel("models/custom/my_model.mdl", true);
    }
}

Action Timer_SpawnCustomEntities(Handle timer)
{
    if (!g_bMapLoaded)
        return Plugin_Stop;

    char mapName[64];
    GetCurrentMap(mapName, sizeof(mapName));

    // Spawn custom info_target for reference points
    int infoTarget = CreateEntityByName("info_target");
    if (infoTarget != -1)
    {
        DispatchKeyValue(infoTarget, "targetname", "custom_spawn_reference");
        DispatchSpawn(infoTarget);

        LogMessage("Created custom spawn reference entity");
    }

    return Plugin_Stop;
}'''
    },

    "onclientdisconnect_cleanup": {
        "user": "Create a plugin that properly handles OnClientDisconnect to clean up player data and timers",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

// Player data storage
int g_iPlayerScore[MAXPLAYERS + 1];
int g_iPlayerKills[MAXPLAYERS + 1];
float g_fPlayerPlayTime[MAXPLAYERS + 1];
Handle g_hPlayerTimer[MAXPLAYERS + 1];
ArrayList g_hPlayerInventory[MAXPLAYERS + 1];
char g_sPlayerSteamId[MAXPLAYERS + 1][32];

public Plugin myinfo =
{
    name = "Player Data Manager",
    author = "Developer",
    description = "Manages player data with proper cleanup",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("player_death", Event_PlayerDeath);

    // Initialize arrays
    for (int i = 1; i <= MaxClients; i++)
    {
        g_hPlayerInventory[i] = null;
        g_hPlayerTimer[i] = null;
    }
}

public void OnClientConnected(int client)
{
    // Pre-initialize data before full connection
    g_iPlayerScore[client] = 0;
    g_iPlayerKills[client] = 0;
    g_fPlayerPlayTime[client] = 0.0;
    g_sPlayerSteamId[client][0] = '\\0';
}

public void OnClientPostAdminCheck(int client)
{
    if (IsFakeClient(client))
        return;

    // Get Steam ID for storage
    GetClientAuthId(client, AuthId_Steam2, g_sPlayerSteamId[client], sizeof(g_sPlayerSteamId[]));

    // Initialize player inventory
    g_hPlayerInventory[client] = new ArrayList(64);

    // Start playtime tracking timer
    g_hPlayerTimer[client] = CreateTimer(60.0, Timer_UpdatePlayTime, GetClientUserId(client), TIMER_REPEAT | TIMER_FLAG_NO_MAPCHANGE);

    LogMessage("Player connected: %s (%s)", g_sPlayerSteamId[client], IsFakeClient(client) ? "Bot" : "Human");
}

public void OnClientDisconnect(int client)
{
    if (IsFakeClient(client))
        return;

    // Save player data before cleanup
    SavePlayerData(client);

    // Clean up timer
    if (g_hPlayerTimer[client] != null)
    {
        KillTimer(g_hPlayerTimer[client]);
        g_hPlayerTimer[client] = null;
    }

    // Clean up ArrayList
    if (g_hPlayerInventory[client] != null)
    {
        delete g_hPlayerInventory[client];
        g_hPlayerInventory[client] = null;
    }

    // Log disconnect
    LogMessage("Player disconnected: %s - Score: %d, Kills: %d, Time: %.1f mins",
        g_sPlayerSteamId[client], g_iPlayerScore[client], g_iPlayerKills[client], g_fPlayerPlayTime[client]);

    // Reset all player data
    g_iPlayerScore[client] = 0;
    g_iPlayerKills[client] = 0;
    g_fPlayerPlayTime[client] = 0.0;
    g_sPlayerSteamId[client][0] = '\\0';
}

void SavePlayerData(int client)
{
    // In a real plugin, this would save to a database
    char filepath[PLATFORM_MAX_PATH];
    BuildPath(Path_SM, filepath, sizeof(filepath), "data/player_%s.txt", g_sPlayerSteamId[client]);

    Handle file = OpenFile(filepath, "w");
    if (file != null)
    {
        WriteFileLine(file, "score=%d", g_iPlayerScore[client]);
        WriteFileLine(file, "kills=%d", g_iPlayerKills[client]);
        WriteFileLine(file, "playtime=%.1f", g_fPlayerPlayTime[client]);
        CloseHandle(file);
    }
}

public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast)
{
    int attacker = GetClientOfUserId(event.GetInt("attacker"));

    if (attacker > 0 && attacker <= MaxClients && IsClientInGame(attacker))
    {
        g_iPlayerKills[attacker]++;
        g_iPlayerScore[attacker] += 10;
    }
}

Action Timer_UpdatePlayTime(Handle timer, int userid)
{
    int client = GetClientOfUserId(userid);
    if (client <= 0 || !IsClientInGame(client))
    {
        // Client left, timer will be cleaned up by OnClientDisconnect
        return Plugin_Stop;
    }

    g_fPlayerPlayTime[client] += 1.0; // Add 1 minute
    return Plugin_Continue;
}'''
    },

    "onpluginstart_comprehensive": {
        "user": "Write a plugin with a comprehensive OnPluginStart that sets up ConVars, commands, event hooks, and database connections",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

// ConVars
ConVar g_cvEnabled;
ConVar g_cvDebug;
ConVar g_cvMultiplier;
ConVar g_cvMaxPlayers;

// Database handle
Database g_hDatabase = null;

// Global state
bool g_bPluginReady = false;
bool g_bLateLoad = false;

public Plugin myinfo =
{
    name = "Comprehensive Plugin Example",
    author = "Developer",
    description = "Demonstrates comprehensive OnPluginStart setup",
    version = "1.0",
    url = ""
};

public APLRes AskPluginLoad2(Handle myself, bool late, char[] error, int err_max)
{
    g_bLateLoad = late;

    // Register natives for other plugins
    CreateNative("MyPlugin_GetStatus", Native_GetStatus);
    CreateNative("MyPlugin_SetMultiplier", Native_SetMultiplier);

    // Register library
    RegPluginLibrary("myplugin");

    return APLRes_Success;
}

public void OnPluginStart()
{
    // === CONVARS ===
    g_cvEnabled = CreateConVar("sm_myplugin_enabled", "1", "Enable the plugin", FCVAR_NOTIFY, true, 0.0, true, 1.0);
    g_cvDebug = CreateConVar("sm_myplugin_debug", "0", "Enable debug logging", FCVAR_NONE, true, 0.0, true, 1.0);
    g_cvMultiplier = CreateConVar("sm_myplugin_multiplier", "1.0", "Score multiplier", FCVAR_NOTIFY, true, 0.1, true, 10.0);
    g_cvMaxPlayers = CreateConVar("sm_myplugin_maxplayers", "8", "Maximum tracked players", FCVAR_NOTIFY, true, 1.0, true, 32.0);

    // Auto-generate config file
    AutoExecConfig(true, "myplugin");

    // Hook ConVar changes
    g_cvEnabled.AddChangeHook(OnConVarChanged);
    g_cvMultiplier.AddChangeHook(OnConVarChanged);

    // === ADMIN COMMANDS ===
    RegAdminCmd("sm_myplugin_reload", Command_Reload, ADMFLAG_CONFIG, "Reload plugin configuration");
    RegAdminCmd("sm_myplugin_status", Command_Status, ADMFLAG_GENERIC, "Show plugin status");
    RegAdminCmd("sm_myplugin_reset", Command_Reset, ADMFLAG_ROOT, "Reset all plugin data");

    // === CONSOLE COMMANDS ===
    RegConsoleCmd("sm_myplugin", Command_Info, "Show plugin information");
    RegConsoleCmd("sm_mystats", Command_MyStats, "Show your statistics");

    // === EVENT HOOKS ===
    HookEvent("player_spawn", Event_PlayerSpawn);
    HookEvent("player_death", Event_PlayerDeath);
    HookEvent("round_start", Event_RoundStart);
    HookEvent("round_end", Event_RoundEnd);
    HookEvent("map_transition", Event_MapTransition);

    // === DATABASE CONNECTION ===
    if (SQL_CheckConfig("myplugin"))
    {
        Database.Connect(OnDatabaseConnect, "myplugin");
    }
    else
    {
        LogMessage("No database configuration found, using file storage");
    }

    // === TRANSLATIONS ===
    LoadTranslations("common.phrases");
    LoadTranslations("myplugin.phrases");

    // === LATE LOAD HANDLING ===
    if (g_bLateLoad)
    {
        for (int i = 1; i <= MaxClients; i++)
        {
            if (IsClientInGame(i))
            {
                OnClientPutInServer(i);
            }
        }
    }

    g_bPluginReady = true;
    LogMessage("Plugin initialized successfully");
}

public void OnConVarChanged(ConVar cvar, const char[] oldValue, const char[] newValue)
{
    if (cvar == g_cvEnabled)
    {
        LogMessage("Plugin %s", StringToInt(newValue) ? "enabled" : "disabled");
    }
}

void OnDatabaseConnect(Database db, const char[] error, any data)
{
    if (db == null)
    {
        LogError("Database connection failed: %s", error);
        return;
    }

    g_hDatabase = db;
    LogMessage("Database connected successfully");

    // Create tables if needed
    g_hDatabase.Query(OnTableCreated,
        "CREATE TABLE IF NOT EXISTS player_stats ("
        ... "steamid VARCHAR(32) PRIMARY KEY,"
        ... "score INT DEFAULT 0,"
        ... "kills INT DEFAULT 0"
        ... ")");
}

void OnTableCreated(Database db, DBResultSet results, const char[] error, any data)
{
    if (results == null)
    {
        LogError("Failed to create table: %s", error);
    }
}

// Native implementations
int Native_GetStatus(Handle plugin, int numParams)
{
    return g_bPluginReady && g_cvEnabled.BoolValue;
}

int Native_SetMultiplier(Handle plugin, int numParams)
{
    float value = GetNativeCell(1);
    g_cvMultiplier.SetFloat(value);
    return 0;
}

// Event handlers (stubs)
public void Event_PlayerSpawn(Event event, const char[] name, bool dontBroadcast) {}
public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast) {}
public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast) {}
public void Event_RoundEnd(Event event, const char[] name, bool dontBroadcast) {}
public void Event_MapTransition(Event event, const char[] name, bool dontBroadcast) {}

// Command handlers (stubs)
Action Command_Reload(int client, int args) { return Plugin_Handled; }
Action Command_Status(int client, int args) { return Plugin_Handled; }
Action Command_Reset(int client, int args) { return Plugin_Handled; }
Action Command_Info(int client, int args) { return Plugin_Handled; }
Action Command_MyStats(int client, int args) { return Plugin_Handled; }'''
    },

    # === NETWORK FORWARDING & SQL ===
    "sql_player_stats": {
        "user": "Create a plugin that stores and retrieves player statistics from a MySQL database",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

Database g_hDatabase = null;

enum struct PlayerStats
{
    int kills;
    int deaths;
    int headshots;
    int revives;
    float playtime;
}

PlayerStats g_PlayerStats[MAXPLAYERS + 1];

public Plugin myinfo =
{
    name = "SQL Player Stats",
    author = "Developer",
    description = "Stores player statistics in MySQL",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("player_death", Event_PlayerDeath);
    HookEvent("player_incapacitated_start", Event_Incap);
    HookEvent("revive_success", Event_Revive);

    RegConsoleCmd("sm_stats", Command_ShowStats, "Show your statistics");
    RegConsoleCmd("sm_topstats", Command_TopStats, "Show top players");

    // Connect to database
    if (SQL_CheckConfig("l4d2stats"))
    {
        Database.Connect(OnDatabaseConnect, "l4d2stats");
    }
    else
    {
        SetFailState("Database configuration 'l4d2stats' not found");
    }
}

void OnDatabaseConnect(Database db, const char[] error, any data)
{
    if (db == null)
    {
        LogError("Database connection failed: %s", error);
        return;
    }

    g_hDatabase = db;
    g_hDatabase.SetCharset("utf8");

    // Create table
    char query[1024];
    Format(query, sizeof(query),
        "CREATE TABLE IF NOT EXISTS player_stats ("
        ... "steamid VARCHAR(32) PRIMARY KEY,"
        ... "name VARCHAR(64),"
        ... "kills INT DEFAULT 0,"
        ... "deaths INT DEFAULT 0,"
        ... "headshots INT DEFAULT 0,"
        ... "revives INT DEFAULT 0,"
        ... "playtime FLOAT DEFAULT 0,"
        ... "last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"
        ... ") ENGINE=InnoDB DEFAULT CHARSET=utf8");

    g_hDatabase.Query(OnTableCreated, query);
}

void OnTableCreated(Database db, DBResultSet results, const char[] error, any data)
{
    if (results == null)
    {
        LogError("Table creation failed: %s", error);
        return;
    }

    LogMessage("Database initialized successfully");
}

public void OnClientPostAdminCheck(int client)
{
    if (IsFakeClient(client))
        return;

    LoadPlayerStats(client);
}

void LoadPlayerStats(int client)
{
    char steamId[32];
    GetClientAuthId(client, AuthId_Steam2, steamId, sizeof(steamId));

    char query[256];
    Format(query, sizeof(query),
        "SELECT kills, deaths, headshots, revives, playtime FROM player_stats WHERE steamid = '%s'",
        steamId);

    g_hDatabase.Query(OnStatsLoaded, query, GetClientUserId(client));
}

void OnStatsLoaded(Database db, DBResultSet results, const char[] error, int userid)
{
    int client = GetClientOfUserId(userid);
    if (client <= 0 || !IsClientInGame(client))
        return;

    if (results == null)
    {
        LogError("Failed to load stats: %s", error);
        return;
    }

    if (results.FetchRow())
    {
        g_PlayerStats[client].kills = results.FetchInt(0);
        g_PlayerStats[client].deaths = results.FetchInt(1);
        g_PlayerStats[client].headshots = results.FetchInt(2);
        g_PlayerStats[client].revives = results.FetchInt(3);
        g_PlayerStats[client].playtime = results.FetchFloat(4);
    }
    else
    {
        // New player - insert record
        char steamId[32], name[64], escapedName[129];
        GetClientAuthId(client, AuthId_Steam2, steamId, sizeof(steamId));
        GetClientName(client, name, sizeof(name));
        g_hDatabase.Escape(name, escapedName, sizeof(escapedName));

        char query[256];
        Format(query, sizeof(query),
            "INSERT INTO player_stats (steamid, name) VALUES ('%s', '%s')",
            steamId, escapedName);

        g_hDatabase.Query(OnPlayerInserted, query);
    }
}

void OnPlayerInserted(Database db, DBResultSet results, const char[] error, any data)
{
    if (results == null)
    {
        LogError("Failed to insert player: %s", error);
    }
}

public void OnClientDisconnect(int client)
{
    if (!IsFakeClient(client))
    {
        SavePlayerStats(client);

        // Reset local stats
        g_PlayerStats[client].kills = 0;
        g_PlayerStats[client].deaths = 0;
        g_PlayerStats[client].headshots = 0;
        g_PlayerStats[client].revives = 0;
        g_PlayerStats[client].playtime = 0.0;
    }
}

void SavePlayerStats(int client)
{
    char steamId[32], name[64], escapedName[129];
    GetClientAuthId(client, AuthId_Steam2, steamId, sizeof(steamId));
    GetClientName(client, name, sizeof(name));
    g_hDatabase.Escape(name, escapedName, sizeof(escapedName));

    char query[512];
    Format(query, sizeof(query),
        "UPDATE player_stats SET name='%s', kills=%d, deaths=%d, headshots=%d, revives=%d, playtime=%.1f WHERE steamid='%s'",
        escapedName, g_PlayerStats[client].kills, g_PlayerStats[client].deaths,
        g_PlayerStats[client].headshots, g_PlayerStats[client].revives,
        g_PlayerStats[client].playtime, steamId);

    g_hDatabase.Query(OnStatsSaved, query);
}

void OnStatsSaved(Database db, DBResultSet results, const char[] error, any data)
{
    if (results == null)
    {
        LogError("Failed to save stats: %s", error);
    }
}

public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast)
{
    int victim = GetClientOfUserId(event.GetInt("userid"));
    int attacker = GetClientOfUserId(event.GetInt("attacker"));
    bool headshot = event.GetBool("headshot");

    if (victim > 0 && victim <= MaxClients && IsClientInGame(victim) && !IsFakeClient(victim))
    {
        g_PlayerStats[victim].deaths++;
    }

    if (attacker > 0 && attacker <= MaxClients && IsClientInGame(attacker) && !IsFakeClient(attacker))
    {
        g_PlayerStats[attacker].kills++;
        if (headshot)
        {
            g_PlayerStats[attacker].headshots++;
        }
    }
}

public void Event_Incap(Event event, const char[] name, bool dontBroadcast) {}

public void Event_Revive(Event event, const char[] name, bool dontBroadcast)
{
    int reviver = GetClientOfUserId(event.GetInt("userid"));

    if (reviver > 0 && reviver <= MaxClients && IsClientInGame(reviver) && !IsFakeClient(reviver))
    {
        g_PlayerStats[reviver].revives++;
    }
}

Action Command_ShowStats(int client, int args)
{
    PrintToChat(client, "\\x04=== Your Statistics ===");
    PrintToChat(client, "Kills: %d | Deaths: %d | K/D: %.2f",
        g_PlayerStats[client].kills, g_PlayerStats[client].deaths,
        g_PlayerStats[client].deaths > 0 ? float(g_PlayerStats[client].kills) / float(g_PlayerStats[client].deaths) : float(g_PlayerStats[client].kills));
    PrintToChat(client, "Headshots: %d | Revives: %d",
        g_PlayerStats[client].headshots, g_PlayerStats[client].revives);

    return Plugin_Handled;
}

Action Command_TopStats(int client, int args)
{
    char query[256];
    Format(query, sizeof(query),
        "SELECT name, kills FROM player_stats ORDER BY kills DESC LIMIT 5");

    g_hDatabase.Query(OnTopStatsLoaded, query, GetClientUserId(client));
    return Plugin_Handled;
}

void OnTopStatsLoaded(Database db, DBResultSet results, const char[] error, int userid)
{
    int client = GetClientOfUserId(userid);
    if (client <= 0)
        return;

    if (results == null)
    {
        PrintToChat(client, "Failed to load top stats.");
        return;
    }

    PrintToChat(client, "\\x04=== Top 5 Players ===");
    int rank = 1;
    while (results.FetchRow())
    {
        char name[64];
        results.FetchString(0, name, sizeof(name));
        int kills = results.FetchInt(1);

        PrintToChat(client, "#%d: \\x03%s\\x01 - %d kills", rank++, name, kills);
    }
}'''
    },

    "network_forward_example": {
        "user": "Write a plugin that creates a global forward to notify other plugins when a survivor is rescued",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

// Global forward declaration
GlobalForward g_hOnSurvivorRescued;
GlobalForward g_hOnSurvivorDowned;

public Plugin myinfo =
{
    name = "Survivor Events Forward",
    author = "Developer",
    description = "Provides forwards for survivor rescue events",
    version = "1.0",
    url = ""
};

public APLRes AskPluginLoad2(Handle myself, bool late, char[] error, int err_max)
{
    // Create global forwards that other plugins can hook
    g_hOnSurvivorRescued = new GlobalForward(
        "L4D2_OnSurvivorRescued",
        ET_Ignore,
        Param_Cell,    // int rescuer
        Param_Cell     // int rescued
    );

    g_hOnSurvivorDowned = new GlobalForward(
        "L4D2_OnSurvivorDowned",
        ET_Event,
        Param_Cell,    // int survivor
        Param_Cell,    // int attacker
        Param_Cell     // int damage
    );

    // Register natives for other plugins
    CreateNative("L4D2_GetDownedSurvivors", Native_GetDownedSurvivors);
    CreateNative("L4D2_RescueSurvivor", Native_RescueSurvivor);

    RegPluginLibrary("survivor_events");

    return APLRes_Success;
}

public void OnPluginStart()
{
    HookEvent("revive_success", Event_ReviveSuccess);
    HookEvent("player_incapacitated", Event_PlayerIncap);
    HookEvent("survivor_rescued", Event_SurvivorRescued);
}

public void Event_ReviveSuccess(Event event, const char[] name, bool dontBroadcast)
{
    int rescuer = GetClientOfUserId(event.GetInt("userid"));
    int rescued = GetClientOfUserId(event.GetInt("subject"));

    if (rescuer <= 0 || rescued <= 0)
        return;

    // Fire the forward to notify other plugins
    Call_StartForward(g_hOnSurvivorRescued);
    Call_PushCell(rescuer);
    Call_PushCell(rescued);
    Call_Finish();

    LogMessage("Forward fired: OnSurvivorRescued (rescuer=%d, rescued=%d)", rescuer, rescued);
}

public void Event_PlayerIncap(Event event, const char[] name, bool dontBroadcast)
{
    int survivor = GetClientOfUserId(event.GetInt("userid"));
    int attacker = GetClientOfUserId(event.GetInt("attacker"));
    int damage = event.GetInt("dmg_health");

    if (survivor <= 0)
        return;

    // Fire forward - allows blocking via ET_Event
    Action result;
    Call_StartForward(g_hOnSurvivorDowned);
    Call_PushCell(survivor);
    Call_PushCell(attacker);
    Call_PushCell(damage);
    Call_Finish(result);

    if (result == Plugin_Handled)
    {
        // Another plugin blocked the incap (would need SDKHooks to actually block)
        LogMessage("Incap event was blocked by another plugin");
    }
}

public void Event_SurvivorRescued(Event event, const char[] name, bool dontBroadcast)
{
    int rescued = GetClientOfUserId(event.GetInt("victim"));
    int rescuer = GetClientOfUserId(event.GetInt("rescuer"));

    if (rescued > 0 && rescuer > 0)
    {
        Call_StartForward(g_hOnSurvivorRescued);
        Call_PushCell(rescuer);
        Call_PushCell(rescued);
        Call_Finish();
    }
}

// Native: Get count of currently downed survivors
int Native_GetDownedSurvivors(Handle plugin, int numParams)
{
    int count = 0;
    for (int i = 1; i <= MaxClients; i++)
    {
        if (IsClientInGame(i) && GetClientTeam(i) == 2 && IsPlayerAlive(i))
        {
            if (GetEntProp(i, Prop_Send, "m_isIncapacitated") == 1)
            {
                count++;
            }
        }
    }
    return count;
}

// Native: Force rescue a survivor
int Native_RescueSurvivor(Handle plugin, int numParams)
{
    int survivor = GetNativeCell(1);

    if (survivor <= 0 || survivor > MaxClients || !IsClientInGame(survivor))
        return 0;

    if (GetClientTeam(survivor) != 2 || !IsPlayerAlive(survivor))
        return 0;

    // Clear incap state
    SetEntProp(survivor, Prop_Send, "m_isIncapacitated", 0);
    SetEntityHealth(survivor, 50);
    SetEntProp(survivor, Prop_Send, "m_currentReviveCount", 0);

    return 1;
}'''
    },

    # === CUSTOM MENUS AND HUD ===
    "custom_menu_weapon_selector": {
        "user": "Create a plugin with a custom menu that lets players select their starting weapons",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

public Plugin myinfo =
{
    name = "Weapon Selector Menu",
    author = "Developer",
    description = "Custom menu for selecting starting weapons",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    RegConsoleCmd("sm_weapons", Command_WeaponMenu, "Open weapon selection menu");

    HookEvent("player_spawn", Event_PlayerSpawn);
}

public void Event_PlayerSpawn(Event event, const char[] name, bool dontBroadcast)
{
    int client = GetClientOfUserId(event.GetInt("userid"));

    if (client > 0 && IsClientInGame(client) && GetClientTeam(client) == 2)
    {
        // Show menu after a short delay
        CreateTimer(1.0, Timer_ShowMenu, GetClientUserId(client), TIMER_FLAG_NO_MAPCHANGE);
    }
}

Action Timer_ShowMenu(Handle timer, int userid)
{
    int client = GetClientOfUserId(userid);
    if (client > 0 && IsClientInGame(client))
    {
        ShowMainWeaponMenu(client);
    }
    return Plugin_Stop;
}

Action Command_WeaponMenu(int client, int args)
{
    if (client <= 0 || !IsClientInGame(client))
        return Plugin_Handled;

    ShowMainWeaponMenu(client);
    return Plugin_Handled;
}

void ShowMainWeaponMenu(int client)
{
    Menu menu = new Menu(MenuHandler_Main);
    menu.SetTitle("Select Weapon Category");

    menu.AddItem("primary", "Primary Weapons");
    menu.AddItem("secondary", "Secondary Weapons");
    menu.AddItem("throwable", "Throwables");
    menu.AddItem("medical", "Medical Items");

    menu.ExitButton = true;
    menu.Display(client, MENU_TIME_FOREVER);
}

int MenuHandler_Main(Menu menu, MenuAction action, int client, int param2)
{
    switch (action)
    {
        case MenuAction_Select:
        {
            char selection[32];
            menu.GetItem(param2, selection, sizeof(selection));

            if (StrEqual(selection, "primary"))
                ShowPrimaryMenu(client);
            else if (StrEqual(selection, "secondary"))
                ShowSecondaryMenu(client);
            else if (StrEqual(selection, "throwable"))
                ShowThrowableMenu(client);
            else if (StrEqual(selection, "medical"))
                ShowMedicalMenu(client);
        }
        case MenuAction_End:
        {
            delete menu;
        }
    }
    return 0;
}

void ShowPrimaryMenu(int client)
{
    Menu menu = new Menu(MenuHandler_Weapon);
    menu.SetTitle("Select Primary Weapon");

    menu.AddItem("weapon_smg", "SMG");
    menu.AddItem("weapon_smg_silenced", "Silenced SMG");
    menu.AddItem("weapon_pumpshotgun", "Pump Shotgun");
    menu.AddItem("weapon_shotgun_chrome", "Chrome Shotgun");
    menu.AddItem("weapon_rifle", "M16 Assault Rifle");
    menu.AddItem("weapon_rifle_ak47", "AK-47");
    menu.AddItem("weapon_hunting_rifle", "Hunting Rifle");
    menu.AddItem("weapon_sniper_military", "Military Sniper");
    menu.AddItem("weapon_autoshotgun", "Auto Shotgun");

    menu.ExitBackButton = true;
    menu.Display(client, MENU_TIME_FOREVER);
}

void ShowSecondaryMenu(int client)
{
    Menu menu = new Menu(MenuHandler_Weapon);
    menu.SetTitle("Select Secondary Weapon");

    menu.AddItem("weapon_pistol", "Pistol");
    menu.AddItem("weapon_pistol_magnum", "Magnum");
    menu.AddItem("weapon_melee", "Random Melee");
    menu.AddItem("weapon_chainsaw", "Chainsaw");

    menu.ExitBackButton = true;
    menu.Display(client, MENU_TIME_FOREVER);
}

void ShowThrowableMenu(int client)
{
    Menu menu = new Menu(MenuHandler_Weapon);
    menu.SetTitle("Select Throwable");

    menu.AddItem("weapon_molotov", "Molotov");
    menu.AddItem("weapon_pipe_bomb", "Pipe Bomb");
    menu.AddItem("weapon_vomitjar", "Bile Bomb");

    menu.ExitBackButton = true;
    menu.Display(client, MENU_TIME_FOREVER);
}

void ShowMedicalMenu(int client)
{
    Menu menu = new Menu(MenuHandler_Weapon);
    menu.SetTitle("Select Medical Item");

    menu.AddItem("weapon_first_aid_kit", "First Aid Kit");
    menu.AddItem("weapon_defibrillator", "Defibrillator");
    menu.AddItem("weapon_pain_pills", "Pain Pills");
    menu.AddItem("weapon_adrenaline", "Adrenaline");

    menu.ExitBackButton = true;
    menu.Display(client, MENU_TIME_FOREVER);
}

int MenuHandler_Weapon(Menu menu, MenuAction action, int client, int param2)
{
    switch (action)
    {
        case MenuAction_Select:
        {
            char weapon[64];
            menu.GetItem(param2, weapon, sizeof(weapon));

            if (IsClientInGame(client) && IsPlayerAlive(client))
            {
                GivePlayerItem(client, weapon);

                char weaponName[64];
                GetWeaponDisplayName(weapon, weaponName, sizeof(weaponName));
                PrintToChat(client, "\\x04[WEAPONS] \\x01You received: %s", weaponName);
            }
        }
        case MenuAction_Cancel:
        {
            if (param2 == MenuCancel_ExitBack)
            {
                ShowMainWeaponMenu(client);
            }
        }
        case MenuAction_End:
        {
            delete menu;
        }
    }
    return 0;
}

void GetWeaponDisplayName(const char[] classname, char[] buffer, int maxlen)
{
    if (StrContains(classname, "smg_silenced") != -1)
        strcopy(buffer, maxlen, "Silenced SMG");
    else if (StrContains(classname, "smg") != -1)
        strcopy(buffer, maxlen, "SMG");
    else if (StrContains(classname, "rifle_ak47") != -1)
        strcopy(buffer, maxlen, "AK-47");
    else if (StrContains(classname, "rifle") != -1)
        strcopy(buffer, maxlen, "Assault Rifle");
    else
        strcopy(buffer, maxlen, classname);
}'''
    },

    "hud_sync_display": {
        "user": "Write a plugin that displays custom HUD information using HudSynchronize showing player stats",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

Handle g_hHudSync;
Handle g_hHudTimer = null;

int g_iPlayerKills[MAXPLAYERS + 1];
int g_iPlayerHeadshots[MAXPLAYERS + 1];
int g_iPlayerDamage[MAXPLAYERS + 1];

public Plugin myinfo =
{
    name = "Custom HUD Display",
    author = "Developer",
    description = "Displays player stats on HUD",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    g_hHudSync = CreateHudSynchronizer();

    HookEvent("player_death", Event_PlayerDeath);
    HookEvent("round_start", Event_RoundStart);

    RegConsoleCmd("sm_togglehud", Command_ToggleHud, "Toggle HUD display");
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
    // Reset stats
    for (int i = 1; i <= MaxClients; i++)
    {
        g_iPlayerKills[i] = 0;
        g_iPlayerHeadshots[i] = 0;
        g_iPlayerDamage[i] = 0;
    }

    // Start HUD update timer
    if (g_hHudTimer != null)
    {
        KillTimer(g_hHudTimer);
    }
    g_hHudTimer = CreateTimer(0.5, Timer_UpdateHud, _, TIMER_REPEAT | TIMER_FLAG_NO_MAPCHANGE);
}

public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast)
{
    int attacker = GetClientOfUserId(event.GetInt("attacker"));
    bool headshot = event.GetBool("headshot");

    if (attacker > 0 && attacker <= MaxClients && IsClientInGame(attacker))
    {
        if (GetClientTeam(attacker) == 2)
        {
            g_iPlayerKills[attacker]++;
            if (headshot)
            {
                g_iPlayerHeadshots[attacker]++;
            }
        }
    }
}

Action Timer_UpdateHud(Handle timer)
{
    for (int i = 1; i <= MaxClients; i++)
    {
        if (!IsClientInGame(i) || IsFakeClient(i))
            continue;

        if (GetClientTeam(i) != 2)
            continue;

        UpdateClientHud(i);
    }

    return Plugin_Continue;
}

void UpdateClientHud(int client)
{
    // Configure HUD text appearance
    SetHudTextParams(0.02, 0.15, 0.6, 255, 255, 0, 200, 0, 0.0, 0.0, 0.0);

    // Get team statistics
    int totalKills = 0;
    int totalHeadshots = 0;
    int aliveCount = 0;

    for (int i = 1; i <= MaxClients; i++)
    {
        if (IsClientInGame(i) && GetClientTeam(i) == 2)
        {
            totalKills += g_iPlayerKills[i];
            totalHeadshots += g_iPlayerHeadshots[i];
            if (IsPlayerAlive(i))
                aliveCount++;
        }
    }

    // Build HUD text
    char hudText[256];
    Format(hudText, sizeof(hudText),
        "Kills: %d | Headshots: %d\\n"
        ... "Team Kills: %d | Survivors: %d/4",
        g_iPlayerKills[client], g_iPlayerHeadshots[client],
        totalKills, aliveCount);

    ShowSyncHudText(client, g_hHudSync, hudText);
}

Action Command_ToggleHud(int client, int args)
{
    // Toggle HUD timer
    if (g_hHudTimer != null)
    {
        KillTimer(g_hHudTimer);
        g_hHudTimer = null;
        ClearSyncHud(client, g_hHudSync);
        PrintToChat(client, "\\x04[HUD] \\x01HUD display disabled.");
    }
    else
    {
        g_hHudTimer = CreateTimer(0.5, Timer_UpdateHud, _, TIMER_REPEAT);
        PrintToChat(client, "\\x04[HUD] \\x01HUD display enabled.");
    }

    return Plugin_Handled;
}

public void OnPluginEnd()
{
    // Clean up HUD for all clients
    for (int i = 1; i <= MaxClients; i++)
    {
        if (IsClientInGame(i))
        {
            ClearSyncHud(i, g_hHudSync);
        }
    }
}'''
    },

    "panel_info_display": {
        "user": "Create a plugin that shows server information using a Panel when players join",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

ConVar g_cvServerName;
ConVar g_cvWelcomeMessage;

public Plugin myinfo =
{
    name = "Welcome Panel",
    author = "Developer",
    description = "Shows welcome information panel to joining players",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    g_cvServerName = CreateConVar("sm_welcome_servername", "L4D2 Server", "Server name for welcome message");
    g_cvWelcomeMessage = CreateConVar("sm_welcome_message", "Welcome to our server!", "Welcome message text");

    RegConsoleCmd("sm_info", Command_ShowInfo, "Show server information");
    RegConsoleCmd("sm_rules", Command_ShowRules, "Show server rules");
}

public void OnClientPostAdminCheck(int client)
{
    if (IsFakeClient(client))
        return;

    CreateTimer(3.0, Timer_ShowWelcome, GetClientUserId(client), TIMER_FLAG_NO_MAPCHANGE);
}

Action Timer_ShowWelcome(Handle timer, int userid)
{
    int client = GetClientOfUserId(userid);
    if (client > 0 && IsClientInGame(client))
    {
        ShowWelcomePanel(client);
    }
    return Plugin_Stop;
}

void ShowWelcomePanel(int client)
{
    Panel panel = new Panel();

    char serverName[64], welcomeMsg[256];
    g_cvServerName.GetString(serverName, sizeof(serverName));
    g_cvWelcomeMessage.GetString(welcomeMsg, sizeof(welcomeMsg));

    char title[128];
    Format(title, sizeof(title), "Welcome to %s", serverName);
    panel.SetTitle(title);

    panel.DrawText(" ");
    panel.DrawText(welcomeMsg);
    panel.DrawText(" ");
    panel.DrawText("Server Features:");
    panel.DrawText("- Custom plugins");
    panel.DrawText("- Fair play enabled");
    panel.DrawText("- Anti-cheat active");
    panel.DrawText(" ");

    panel.DrawItem("Show Rules");
    panel.DrawItem("Show Commands");
    panel.DrawItem("Close");

    panel.Send(client, PanelHandler_Welcome, 30);
    delete panel;
}

int PanelHandler_Welcome(Menu menu, MenuAction action, int client, int param2)
{
    if (action == MenuAction_Select)
    {
        switch (param2)
        {
            case 1: ShowRulesPanel(client);
            case 2: ShowCommandsPanel(client);
            // case 3: Close - does nothing
        }
    }
    return 0;
}

void ShowRulesPanel(int client)
{
    Panel panel = new Panel();
    panel.SetTitle("Server Rules");

    panel.DrawText(" ");
    panel.DrawText("1. No cheating or exploiting");
    panel.DrawText("2. Respect all players");
    panel.DrawText("3. No intentional team killing");
    panel.DrawText("4. No mic spam or chat spam");
    panel.DrawText("5. Follow admin instructions");
    panel.DrawText(" ");
    panel.DrawText("Violations may result in a ban.");
    panel.DrawText(" ");

    panel.DrawItem("Back to Welcome");
    panel.DrawItem("Close");

    panel.Send(client, PanelHandler_Rules, 30);
    delete panel;
}

int PanelHandler_Rules(Menu menu, MenuAction action, int client, int param2)
{
    if (action == MenuAction_Select)
    {
        if (param2 == 1)
        {
            ShowWelcomePanel(client);
        }
    }
    return 0;
}

void ShowCommandsPanel(int client)
{
    Panel panel = new Panel();
    panel.SetTitle("Available Commands");

    panel.DrawText(" ");
    panel.DrawText("!info - Show server info");
    panel.DrawText("!rules - Show server rules");
    panel.DrawText("!stats - Show your statistics");
    panel.DrawText("!weapons - Weapon selection");
    panel.DrawText("!vote - Start a vote");
    panel.DrawText(" ");

    panel.DrawItem("Back to Welcome");
    panel.DrawItem("Close");

    panel.Send(client, PanelHandler_Commands, 30);
    delete panel;
}

int PanelHandler_Commands(Menu menu, MenuAction action, int client, int param2)
{
    if (action == MenuAction_Select)
    {
        if (param2 == 1)
        {
            ShowWelcomePanel(client);
        }
    }
    return 0;
}

Action Command_ShowInfo(int client, int args)
{
    ShowWelcomePanel(client);
    return Plugin_Handled;
}

Action Command_ShowRules(int client, int args)
{
    ShowRulesPanel(client);
    return Plugin_Handled;
}'''
    },

    # === INTER-PLUGIN COMMUNICATION ===
    "forward_native_library": {
        "user": "Write a library plugin that exposes natives and forwards for other plugins to use",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

// Forward handles
GlobalForward g_hOnPlayerLevelUp;
GlobalForward g_hOnPlayerXPGain;

// Plugin state
int g_iPlayerLevel[MAXPLAYERS + 1];
int g_iPlayerXP[MAXPLAYERS + 1];

// Constants
#define XP_PER_LEVEL 100

public Plugin myinfo =
{
    name = "L4D2 Leveling Library",
    author = "Developer",
    description = "Provides leveling system API for other plugins",
    version = "1.0",
    url = ""
};

public APLRes AskPluginLoad2(Handle myself, bool late, char[] error, int err_max)
{
    // Create forwards
    g_hOnPlayerLevelUp = new GlobalForward(
        "L4D2Level_OnPlayerLevelUp",
        ET_Ignore,
        Param_Cell,    // int client
        Param_Cell,    // int oldLevel
        Param_Cell     // int newLevel
    );

    g_hOnPlayerXPGain = new GlobalForward(
        "L4D2Level_OnPlayerXPGain",
        ET_Event,
        Param_Cell,    // int client
        Param_CellByRef // int &xpAmount (modifiable)
    );

    // Register natives
    CreateNative("L4D2Level_GetPlayerLevel", Native_GetPlayerLevel);
    CreateNative("L4D2Level_SetPlayerLevel", Native_SetPlayerLevel);
    CreateNative("L4D2Level_GetPlayerXP", Native_GetPlayerXP);
    CreateNative("L4D2Level_AddPlayerXP", Native_AddPlayerXP);
    CreateNative("L4D2Level_GetXPForLevel", Native_GetXPForLevel);

    // Register library for dependency checking
    RegPluginLibrary("l4d2_leveling");

    return APLRes_Success;
}

public void OnPluginStart()
{
    HookEvent("player_death", Event_PlayerDeath);
    HookEvent("round_start", Event_RoundStart);
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
    // Reset XP at round start but keep levels
    for (int i = 1; i <= MaxClients; i++)
    {
        g_iPlayerXP[i] = 0;
    }
}

public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast)
{
    int attacker = GetClientOfUserId(event.GetInt("attacker"));

    if (attacker > 0 && attacker <= MaxClients && IsClientInGame(attacker))
    {
        if (GetClientTeam(attacker) == 2)
        {
            AddPlayerXP(attacker, 10);
        }
    }
}

void AddPlayerXP(int client, int amount)
{
    // Fire forward to allow modification
    Action result;
    Call_StartForward(g_hOnPlayerXPGain);
    Call_PushCell(client);
    Call_PushCellRef(amount);
    Call_Finish(result);

    if (result == Plugin_Handled || result == Plugin_Stop)
        return;

    g_iPlayerXP[client] += amount;

    // Check for level up
    int requiredXP = GetXPForLevel(g_iPlayerLevel[client] + 1);
    while (g_iPlayerXP[client] >= requiredXP)
    {
        g_iPlayerXP[client] -= requiredXP;
        int oldLevel = g_iPlayerLevel[client];
        g_iPlayerLevel[client]++;

        // Fire level up forward
        Call_StartForward(g_hOnPlayerLevelUp);
        Call_PushCell(client);
        Call_PushCell(oldLevel);
        Call_PushCell(g_iPlayerLevel[client]);
        Call_Finish();

        PrintToChat(client, "\\x04[LEVEL UP] \\x01You are now level %d!", g_iPlayerLevel[client]);

        requiredXP = GetXPForLevel(g_iPlayerLevel[client] + 1);
    }
}

int GetXPForLevel(int level)
{
    return level * XP_PER_LEVEL;
}

// === NATIVE IMPLEMENTATIONS ===

int Native_GetPlayerLevel(Handle plugin, int numParams)
{
    int client = GetNativeCell(1);

    if (client < 1 || client > MaxClients)
    {
        ThrowNativeError(SP_ERROR_PARAM, "Invalid client index %d", client);
        return 0;
    }

    return g_iPlayerLevel[client];
}

int Native_SetPlayerLevel(Handle plugin, int numParams)
{
    int client = GetNativeCell(1);
    int level = GetNativeCell(2);

    if (client < 1 || client > MaxClients)
    {
        ThrowNativeError(SP_ERROR_PARAM, "Invalid client index %d", client);
        return 0;
    }

    if (level < 0)
        level = 0;

    int oldLevel = g_iPlayerLevel[client];
    g_iPlayerLevel[client] = level;

    if (level != oldLevel)
    {
        Call_StartForward(g_hOnPlayerLevelUp);
        Call_PushCell(client);
        Call_PushCell(oldLevel);
        Call_PushCell(level);
        Call_Finish();
    }

    return 1;
}

int Native_GetPlayerXP(Handle plugin, int numParams)
{
    int client = GetNativeCell(1);

    if (client < 1 || client > MaxClients)
    {
        ThrowNativeError(SP_ERROR_PARAM, "Invalid client index %d", client);
        return 0;
    }

    return g_iPlayerXP[client];
}

int Native_AddPlayerXP(Handle plugin, int numParams)
{
    int client = GetNativeCell(1);
    int amount = GetNativeCell(2);

    if (client < 1 || client > MaxClients)
    {
        ThrowNativeError(SP_ERROR_PARAM, "Invalid client index %d", client);
        return 0;
    }

    if (!IsClientInGame(client))
        return 0;

    AddPlayerXP(client, amount);
    return 1;
}

int Native_GetXPForLevel(Handle plugin, int numParams)
{
    int level = GetNativeCell(1);
    return GetXPForLevel(level);
}'''
    },

    "include_file_example": {
        "user": "Write the include file (.inc) for a leveling system plugin that defines natives and forwards",
        "assistant": '''#if defined _l4d2_leveling_included
    #endinput
#endif
#define _l4d2_leveling_included

/**
 * L4D2 Leveling System Include File
 *
 * This include file provides access to the L4D2 Leveling System API.
 * Make sure the l4d2_leveling plugin is loaded before using these functions.
 */

/**
 * Called when a player levels up.
 *
 * @param client        Client index who leveled up.
 * @param oldLevel      The player's previous level.
 * @param newLevel      The player's new level.
 * @noreturn
 */
forward void L4D2Level_OnPlayerLevelUp(int client, int oldLevel, int newLevel);

/**
 * Called when a player gains XP. Can be used to modify the amount.
 *
 * @param client        Client index gaining XP.
 * @param xpAmount      Reference to the XP amount (can be modified).
 * @return              Plugin_Continue to allow, Plugin_Handled to block.
 */
forward Action L4D2Level_OnPlayerXPGain(int client, int &xpAmount);

/**
 * Gets a player's current level.
 *
 * @param client        Client index.
 * @return              The player's current level.
 * @error               Invalid client index.
 */
native int L4D2Level_GetPlayerLevel(int client);

/**
 * Sets a player's level directly.
 *
 * @param client        Client index.
 * @param level         The new level to set.
 * @return              1 on success, 0 on failure.
 * @error               Invalid client index.
 */
native int L4D2Level_SetPlayerLevel(int client, int level);

/**
 * Gets a player's current XP.
 *
 * @param client        Client index.
 * @return              The player's current XP.
 * @error               Invalid client index.
 */
native int L4D2Level_GetPlayerXP(int client);

/**
 * Adds XP to a player. May trigger level ups.
 *
 * @param client        Client index.
 * @param amount        Amount of XP to add.
 * @return              1 on success, 0 on failure.
 * @error               Invalid client index.
 */
native int L4D2Level_AddPlayerXP(int client, int amount);

/**
 * Gets the XP required to reach a specific level.
 *
 * @param level         The target level.
 * @return              XP required for that level.
 */
native int L4D2Level_GetXPForLevel(int level);

/**
 * Check if the leveling library is available.
 */
public SharedPlugin __pl_l4d2_leveling =
{
    name = "l4d2_leveling",
    file = "l4d2_leveling.smx",
#if defined REQUIRE_PLUGIN
    required = 1,
#else
    required = 0,
#endif
};

#if !defined REQUIRE_PLUGIN
public void __pl_l4d2_leveling_SetNTVOptional()
{
    MarkNativeAsOptional("L4D2Level_GetPlayerLevel");
    MarkNativeAsOptional("L4D2Level_SetPlayerLevel");
    MarkNativeAsOptional("L4D2Level_GetPlayerXP");
    MarkNativeAsOptional("L4D2Level_AddPlayerXP");
    MarkNativeAsOptional("L4D2Level_GetXPForLevel");
}
#endif'''
    },

    "plugin_using_library": {
        "user": "Create a plugin that uses another plugin's natives and hooks its forwards for a bonus system",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <l4d2_leveling>  // Include the library header

#define PLUGIN_VERSION "1.0"

ConVar g_cvBonusMultiplier;
ConVar g_cvLevelBonusHealth;

public Plugin myinfo =
{
    name = "Leveling Bonus System",
    author = "Developer",
    description = "Adds bonuses based on player levels",
    version = PLUGIN_VERSION,
    url = ""
};

public void OnPluginStart()
{
    g_cvBonusMultiplier = CreateConVar("sm_levelbonus_xpmult", "1.5", "XP multiplier for high level players", FCVAR_NOTIFY, true, 1.0, true, 5.0);
    g_cvLevelBonusHealth = CreateConVar("sm_levelbonus_health", "5", "Bonus health per level", FCVAR_NOTIFY, true, 0.0, true, 25.0);

    RegConsoleCmd("sm_mylevel", Command_MyLevel, "Show your level info");
    RegAdminCmd("sm_setlevel", Command_SetLevel, ADMFLAG_CHEATS, "Set a player's level");
}

// Hook the forward from the leveling library
public void L4D2Level_OnPlayerLevelUp(int client, int oldLevel, int newLevel)
{
    if (!IsClientInGame(client))
        return;

    char name[64];
    GetClientName(client, name, sizeof(name));

    PrintToChatAll("\\x04[LEVEL] \\x03%s \\x01leveled up to \\x04Level %d\\x01!", name, newLevel);

    // Apply level-based health bonus
    int bonusHealth = newLevel * g_cvLevelBonusHealth.IntValue;
    int maxHealth = 100 + bonusHealth;

    SetEntProp(client, Prop_Data, "m_iMaxHealth", maxHealth);

    // Heal to full on level up
    SetEntityHealth(client, maxHealth);

    // Play level up sound
    EmitSoundToClient(client, "ui/pickup_secret01.wav");

    // Announce special milestones
    if (newLevel % 10 == 0)
    {
        PrintToChatAll("\\x04[MILESTONE] \\x03%s \\x01has reached \\x04Level %d\\x01! Congratulations!", name, newLevel);
        EmitSoundToAll("ui/beep_synthtone01.wav");
    }
}

// Hook to modify XP gains
public Action L4D2Level_OnPlayerXPGain(int client, int &xpAmount)
{
    // High level players get bonus XP
    int level = L4D2Level_GetPlayerLevel(client);

    if (level >= 10)
    {
        float multiplier = g_cvBonusMultiplier.FloatValue;
        xpAmount = RoundToFloor(float(xpAmount) * multiplier);

        PrintToChat(client, "\\x04[XP BONUS] \\x01Level %d bonus: +%d XP!", level, xpAmount);
    }

    return Plugin_Changed;
}

Action Command_MyLevel(int client, int args)
{
    if (client <= 0)
        return Plugin_Handled;

    int level = L4D2Level_GetPlayerLevel(client);
    int xp = L4D2Level_GetPlayerXP(client);
    int xpNeeded = L4D2Level_GetXPForLevel(level + 1);

    PrintToChat(client, "\\x04=== Your Level Info ===");
    PrintToChat(client, "Level: \\x04%d", level);
    PrintToChat(client, "XP: \\x04%d\\x01 / \\x04%d", xp, xpNeeded);
    PrintToChat(client, "Progress: \\x04%.1f%%", (float(xp) / float(xpNeeded)) * 100.0);

    int bonusHealth = level * g_cvLevelBonusHealth.IntValue;
    PrintToChat(client, "Bonus Max Health: +\\x04%d", bonusHealth);

    return Plugin_Handled;
}

Action Command_SetLevel(int client, int args)
{
    if (args < 2)
    {
        ReplyToCommand(client, "[SM] Usage: sm_setlevel <#userid|name> <level>");
        return Plugin_Handled;
    }

    char arg1[64], arg2[16];
    GetCmdArg(1, arg1, sizeof(arg1));
    GetCmdArg(2, arg2, sizeof(arg2));

    int target = FindTarget(client, arg1, true, false);
    if (target == -1)
        return Plugin_Handled;

    int level = StringToInt(arg2);
    if (level < 0 || level > 100)
    {
        ReplyToCommand(client, "[SM] Level must be between 0 and 100.");
        return Plugin_Handled;
    }

    L4D2Level_SetPlayerLevel(target, level);

    char targetName[64];
    GetClientName(target, targetName, sizeof(targetName));

    ReplyToCommand(client, "[SM] Set %s's level to %d.", targetName, level);
    PrintToChat(target, "\\x04[ADMIN] \\x01Your level has been set to \\x04%d\\x01.", level);

    return Plugin_Handled;
}'''
    },

    # === ADDITIONAL TEMPLATES FOR DIVERSITY ===
    "tank_rock_throw_modifier": {
        "user": "Write a plugin that modifies Tank rock throwing behavior and tracks rock accuracy",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>

int g_iRocksThrown = 0;
int g_iRocksHit = 0;
int g_iLastRockThrower = -1;

public Plugin myinfo =
{
    name = "Tank Rock Tracker",
    author = "Developer",
    description = "Tracks Tank rock throwing accuracy",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("ability_use", Event_AbilityUse);
    HookEvent("player_hurt", Event_PlayerHurt);
    HookEvent("round_start", Event_RoundStart);

    RegConsoleCmd("sm_rockstats", Command_RockStats, "Show rock throwing statistics");
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
    g_iRocksThrown = 0;
    g_iRocksHit = 0;
}

public void Event_AbilityUse(Event event, const char[] name, bool dontBroadcast)
{
    int client = GetClientOfUserId(event.GetInt("userid"));
    char ability[32];
    event.GetString("ability", ability, sizeof(ability));

    if (StrEqual(ability, "ability_throw"))
    {
        g_iRocksThrown++;
        g_iLastRockThrower = client;

        // Track rock entity
        CreateTimer(0.1, Timer_FindRock, GetClientUserId(client));
    }
}

Action Timer_FindRock(Handle timer, int userid)
{
    int rock = -1;
    while ((rock = FindEntityByClassname(rock, "tank_rock")) != -1)
    {
        SDKHook(rock, SDKHook_Touch, OnRockTouch);
    }
    return Plugin_Stop;
}

Action OnRockTouch(int rock, int other)
{
    if (other > 0 && other <= MaxClients && IsClientInGame(other))
    {
        if (GetClientTeam(other) == 2) // Hit a survivor
        {
            g_iRocksHit++;
        }
    }
    return Plugin_Continue;
}

public void Event_PlayerHurt(Event event, const char[] name, bool dontBroadcast)
{
    char weapon[32];
    event.GetString("weapon", weapon, sizeof(weapon));

    if (StrEqual(weapon, "tank_rock"))
    {
        int victim = GetClientOfUserId(event.GetInt("userid"));
        int damage = event.GetInt("dmg_health");

        if (victim > 0 && IsClientInGame(victim))
        {
            char victimName[64];
            GetClientName(victim, victimName, sizeof(victimName));
            PrintToChatAll("\\x04[ROCK] \\x03%s \\x01was hit by Tank rock for \\x04%d \\x01damage!", victimName, damage);
        }
    }
}

Action Command_RockStats(int client, int args)
{
    float accuracy = 0.0;
    if (g_iRocksThrown > 0)
    {
        accuracy = (float(g_iRocksHit) / float(g_iRocksThrown)) * 100.0;
    }

    PrintToChat(client, "\\x04=== Tank Rock Statistics ===");
    PrintToChat(client, "Rocks Thrown: \\x04%d", g_iRocksThrown);
    PrintToChat(client, "Rocks Hit: \\x04%d", g_iRocksHit);
    PrintToChat(client, "Accuracy: \\x04%.1f%%", accuracy);

    return Plugin_Handled;
}'''
    },

    "witch_wanderer_spawner": {
        "user": "Create a plugin that spawns wandering witches at regular intervals with configurable settings",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

ConVar g_cvEnabled;
ConVar g_cvInterval;
ConVar g_cvMaxWitches;
ConVar g_cvMinDistance;

Handle g_hSpawnTimer = null;
int g_iWitchCount = 0;

public Plugin myinfo =
{
    name = "Wandering Witch Spawner",
    author = "Developer",
    description = "Spawns wandering witches periodically",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    g_cvEnabled = CreateConVar("sm_wanderwitch_enabled", "1", "Enable wandering witch spawner", FCVAR_NOTIFY, true, 0.0, true, 1.0);
    g_cvInterval = CreateConVar("sm_wanderwitch_interval", "120.0", "Seconds between witch spawns", FCVAR_NOTIFY, true, 30.0, true, 600.0);
    g_cvMaxWitches = CreateConVar("sm_wanderwitch_max", "3", "Maximum wandering witches at once", FCVAR_NOTIFY, true, 1.0, true, 10.0);
    g_cvMinDistance = CreateConVar("sm_wanderwitch_distance", "1000.0", "Minimum spawn distance from survivors", FCVAR_NOTIFY, true, 500.0, true, 3000.0);

    HookEvent("round_start", Event_RoundStart);
    HookEvent("round_end", Event_RoundEnd);
    HookEvent("witch_killed", Event_WitchKilled);

    RegAdminCmd("sm_spawnwanderer", Command_SpawnWanderer, ADMFLAG_CHEATS, "Spawn a wandering witch");
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
    g_iWitchCount = 0;
    StartSpawnTimer();
}

public void Event_RoundEnd(Event event, const char[] name, bool dontBroadcast)
{
    StopSpawnTimer();
}

public void Event_WitchKilled(Event event, const char[] name, bool dontBroadcast)
{
    g_iWitchCount--;
    if (g_iWitchCount < 0) g_iWitchCount = 0;
}

void StartSpawnTimer()
{
    StopSpawnTimer();

    if (!g_cvEnabled.BoolValue)
        return;

    float interval = g_cvInterval.FloatValue;
    g_hSpawnTimer = CreateTimer(interval, Timer_SpawnWitch, _, TIMER_REPEAT | TIMER_FLAG_NO_MAPCHANGE);
}

void StopSpawnTimer()
{
    if (g_hSpawnTimer != null)
    {
        KillTimer(g_hSpawnTimer);
        g_hSpawnTimer = null;
    }
}

Action Timer_SpawnWitch(Handle timer)
{
    if (!g_cvEnabled.BoolValue)
        return Plugin_Continue;

    if (g_iWitchCount >= g_cvMaxWitches.IntValue)
        return Plugin_Continue;

    SpawnWanderingWitch();
    return Plugin_Continue;
}

void SpawnWanderingWitch()
{
    float spawnPos[3];
    if (!GetValidSpawnPosition(spawnPos))
    {
        LogMessage("Could not find valid spawn position for wandering witch");
        return;
    }

    int witch = CreateEntityByName("witch");
    if (witch != -1)
    {
        float angles[3] = {0.0, 0.0, 0.0};
        angles[1] = GetRandomFloat(0.0, 360.0);

        TeleportEntity(witch, spawnPos, angles, NULL_VECTOR);
        DispatchSpawn(witch);
        ActivateEntity(witch);

        // Make her wander
        SetEntProp(witch, Prop_Send, "m_mobRush", 1);

        g_iWitchCount++;
        PrintToChatAll("\\x04[WITCH] \\x01A wandering Witch has appeared...");
    }
}

bool GetValidSpawnPosition(float pos[3])
{
    float minDist = g_cvMinDistance.FloatValue;

    // Get center position of all survivors
    float centerPos[3];
    int survivorCount = 0;

    for (int i = 1; i <= MaxClients; i++)
    {
        if (IsClientInGame(i) && GetClientTeam(i) == 2 && IsPlayerAlive(i))
        {
            float clientPos[3];
            GetClientAbsOrigin(i, clientPos);

            centerPos[0] += clientPos[0];
            centerPos[1] += clientPos[1];
            centerPos[2] += clientPos[2];
            survivorCount++;
        }
    }

    if (survivorCount == 0)
        return false;

    centerPos[0] /= float(survivorCount);
    centerPos[1] /= float(survivorCount);
    centerPos[2] /= float(survivorCount);

    // Try to find a valid position
    for (int attempts = 0; attempts < 20; attempts++)
    {
        float angle = GetRandomFloat(0.0, 360.0);
        float distance = GetRandomFloat(minDist, minDist + 500.0);

        pos[0] = centerPos[0] + Cosine(DegToRad(angle)) * distance;
        pos[1] = centerPos[1] + Sine(DegToRad(angle)) * distance;
        pos[2] = centerPos[2];

        // Trace to find ground
        float endPos[3];
        endPos[0] = pos[0];
        endPos[1] = pos[1];
        endPos[2] = pos[2] - 500.0;

        Handle trace = TR_TraceRayFilterEx(pos, endPos, MASK_SOLID, RayType_EndPoint, TraceFilter_NoClients);
        if (TR_DidHit(trace))
        {
            TR_GetEndPosition(pos, trace);
            pos[2] += 20.0;
            delete trace;
            return true;
        }
        delete trace;
    }

    return false;
}

bool TraceFilter_NoClients(int entity, int contentsMask)
{
    return entity > MaxClients || entity == 0;
}

Action Command_SpawnWanderer(int client, int args)
{
    SpawnWanderingWitch();
    ReplyToCommand(client, "[SM] Spawned a wandering witch.");
    return Plugin_Handled;
}'''
    },

    "multiple_event_hooks": {
        "user": "Write a plugin that demonstrates hooking multiple related events for comprehensive tracking",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

// Statistics tracking
int g_iSurvivorDamageDealt[MAXPLAYERS + 1];
int g_iSurvivorDamageTaken[MAXPLAYERS + 1];
int g_iSurvivorIncaps[MAXPLAYERS + 1];
int g_iSurvivorRevives[MAXPLAYERS + 1];
int g_iSurvivorFF[MAXPLAYERS + 1];

public Plugin myinfo =
{
    name = "Comprehensive Event Tracker",
    author = "Developer",
    description = "Hooks multiple events for complete statistics",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    // Round events
    HookEvent("round_start", Event_RoundStart);
    HookEvent("round_end", Event_RoundEnd);
    HookEvent("map_transition", Event_MapTransition);

    // Player state events
    HookEvent("player_spawn", Event_PlayerSpawn);
    HookEvent("player_death", Event_PlayerDeath);
    HookEvent("player_hurt", Event_PlayerHurt);

    // Incap and revive events
    HookEvent("player_incapacitated", Event_PlayerIncap);
    HookEvent("player_incapacitated_start", Event_PlayerIncapStart);
    HookEvent("revive_begin", Event_ReviveBegin);
    HookEvent("revive_success", Event_ReviveSuccess);
    HookEvent("revive_end", Event_ReviveEnd);

    // Heal events
    HookEvent("heal_begin", Event_HealBegin);
    HookEvent("heal_success", Event_HealSuccess);
    HookEvent("heal_end", Event_HealEnd);

    // Special events
    HookEvent("defibrillator_used", Event_DefibUsed);
    HookEvent("adrenaline_used", Event_AdrenalineUsed);
    HookEvent("pills_used", Event_PillsUsed);

    // Friendly fire
    HookEvent("friendly_fire", Event_FriendlyFire);

    RegConsoleCmd("sm_roundstats", Command_RoundStats, "Show round statistics");
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
    ResetAllStats();
    PrintToChatAll("\\x04[TRACKER] \\x01Round started. Statistics tracking enabled.");
}

public void Event_RoundEnd(Event event, const char[] name, bool dontBroadcast)
{
    DisplayRoundSummary();
}

public void Event_MapTransition(Event event, const char[] name, bool dontBroadcast)
{
    DisplayMapTransitionStats();
}

void ResetAllStats()
{
    for (int i = 1; i <= MaxClients; i++)
    {
        g_iSurvivorDamageDealt[i] = 0;
        g_iSurvivorDamageTaken[i] = 0;
        g_iSurvivorIncaps[i] = 0;
        g_iSurvivorRevives[i] = 0;
        g_iSurvivorFF[i] = 0;
    }
}

public void Event_PlayerSpawn(Event event, const char[] name, bool dontBroadcast)
{
    int client = GetClientOfUserId(event.GetInt("userid"));
    LogMessage("Player spawned: %d (Team %d)", client, IsClientInGame(client) ? GetClientTeam(client) : -1);
}

public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast)
{
    int victim = GetClientOfUserId(event.GetInt("userid"));
    int attacker = GetClientOfUserId(event.GetInt("attacker"));

    if (victim > 0 && victim <= MaxClients && IsClientInGame(victim))
    {
        char victimName[64];
        GetClientName(victim, victimName, sizeof(victimName));
        PrintToChatAll("\\x04[DEATH] \\x03%s \\x01has died.", victimName);
    }
}

public void Event_PlayerHurt(Event event, const char[] name, bool dontBroadcast)
{
    int victim = GetClientOfUserId(event.GetInt("userid"));
    int attacker = GetClientOfUserId(event.GetInt("attacker"));
    int damage = event.GetInt("dmg_health");

    if (attacker > 0 && attacker <= MaxClients && IsClientInGame(attacker))
    {
        if (GetClientTeam(attacker) == 2)
        {
            g_iSurvivorDamageDealt[attacker] += damage;
        }
    }

    if (victim > 0 && victim <= MaxClients && IsClientInGame(victim))
    {
        if (GetClientTeam(victim) == 2)
        {
            g_iSurvivorDamageTaken[victim] += damage;
        }
    }
}

public void Event_PlayerIncap(Event event, const char[] name, bool dontBroadcast)
{
    int victim = GetClientOfUserId(event.GetInt("userid"));
    if (victim > 0 && victim <= MaxClients)
    {
        g_iSurvivorIncaps[victim]++;
    }
}

public void Event_PlayerIncapStart(Event event, const char[] name, bool dontBroadcast)
{
    int victim = GetClientOfUserId(event.GetInt("userid"));
    if (victim > 0 && IsClientInGame(victim))
    {
        char name[64];
        GetClientName(victim, name, sizeof(name));
        PrintToChatAll("\\x04[INCAP] \\x03%s \\x01is down!", name);
    }
}

public void Event_ReviveBegin(Event event, const char[] name, bool dontBroadcast)
{
    int reviver = GetClientOfUserId(event.GetInt("userid"));
    int victim = GetClientOfUserId(event.GetInt("subject"));

    if (reviver > 0 && victim > 0 && IsClientInGame(reviver) && IsClientInGame(victim))
    {
        char reviverName[64], victimName[64];
        GetClientName(reviver, reviverName, sizeof(reviverName));
        GetClientName(victim, victimName, sizeof(victimName));
        PrintToChat(reviver, "\\x04[REVIVE] \\x01Reviving %s...", victimName);
    }
}

public void Event_ReviveSuccess(Event event, const char[] name, bool dontBroadcast)
{
    int reviver = GetClientOfUserId(event.GetInt("userid"));
    if (reviver > 0 && reviver <= MaxClients)
    {
        g_iSurvivorRevives[reviver]++;
    }
}

public void Event_ReviveEnd(Event event, const char[] name, bool dontBroadcast) {}

public void Event_HealBegin(Event event, const char[] name, bool dontBroadcast) {}
public void Event_HealSuccess(Event event, const char[] name, bool dontBroadcast) {}
public void Event_HealEnd(Event event, const char[] name, bool dontBroadcast) {}

public void Event_DefibUsed(Event event, const char[] name, bool dontBroadcast)
{
    int user = GetClientOfUserId(event.GetInt("userid"));
    int subject = GetClientOfUserId(event.GetInt("subject"));

    if (user > 0 && subject > 0 && IsClientInGame(user) && IsClientInGame(subject))
    {
        char userName[64], subjectName[64];
        GetClientName(user, userName, sizeof(userName));
        GetClientName(subject, subjectName, sizeof(subjectName));
        PrintToChatAll("\\x04[DEFIB] \\x03%s \\x01revived \\x03%s \\x01with a defibrillator!", userName, subjectName);
    }
}

public void Event_AdrenalineUsed(Event event, const char[] name, bool dontBroadcast) {}
public void Event_PillsUsed(Event event, const char[] name, bool dontBroadcast) {}

public void Event_FriendlyFire(Event event, const char[] name, bool dontBroadcast)
{
    int attacker = GetClientOfUserId(event.GetInt("attacker"));
    if (attacker > 0 && attacker <= MaxClients)
    {
        g_iSurvivorFF[attacker]++;
    }
}

void DisplayRoundSummary()
{
    PrintToChatAll("\\x04=== ROUND SUMMARY ===");

    for (int i = 1; i <= MaxClients; i++)
    {
        if (!IsClientInGame(i) || GetClientTeam(i) != 2)
            continue;

        char name[64];
        GetClientName(i, name, sizeof(name));

        PrintToChatAll("\\x03%s\\x01: DMG %d | Taken %d | Incaps %d | Revives %d | FF %d",
            name, g_iSurvivorDamageDealt[i], g_iSurvivorDamageTaken[i],
            g_iSurvivorIncaps[i], g_iSurvivorRevives[i], g_iSurvivorFF[i]);
    }
}

void DisplayMapTransitionStats()
{
    PrintToChatAll("\\x04[TRANSITION] \\x01Map complete! Saving statistics...");
    DisplayRoundSummary();
}

Action Command_RoundStats(int client, int args)
{
    DisplayRoundSummary();
    return Plugin_Handled;
}'''
    },

    "keyvalues_config_loader": {
        "user": "Write a plugin that loads configuration from a KeyValues file and applies map-specific settings",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

KeyValues g_hConfig = null;

// Loaded settings
int g_iMaxZombies;
int g_iTankHealth;
float g_fHordeInterval;
bool g_bAllowWitches;

public Plugin myinfo =
{
    name = "KeyValues Config Loader",
    author = "Developer",
    description = "Loads map settings from KeyValues config",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    RegAdminCmd("sm_reloadconfig", Command_ReloadConfig, ADMFLAG_CONFIG, "Reload map configuration");
    RegAdminCmd("sm_showconfig", Command_ShowConfig, ADMFLAG_GENERIC, "Show current configuration");
}

public void OnMapStart()
{
    LoadConfiguration();
    ApplyConfiguration();
}

void LoadConfiguration()
{
    // Clean up old config
    if (g_hConfig != null)
    {
        delete g_hConfig;
    }

    g_hConfig = new KeyValues("MapSettings");

    char configPath[PLATFORM_MAX_PATH];
    BuildPath(Path_SM, configPath, sizeof(configPath), "configs/map_settings.cfg");

    if (!FileExists(configPath))
    {
        LogMessage("Config file not found, creating default: %s", configPath);
        CreateDefaultConfig(configPath);
    }

    if (!g_hConfig.ImportFromFile(configPath))
    {
        LogError("Failed to load config file: %s", configPath);
        SetDefaultValues();
        return;
    }

    // Get current map name
    char mapName[64];
    GetCurrentMap(mapName, sizeof(mapName));

    // Try to find map-specific settings
    if (g_hConfig.JumpToKey(mapName))
    {
        LoadMapSettings();
        LogMessage("Loaded settings for map: %s", mapName);
    }
    else if (g_hConfig.JumpToKey("default"))
    {
        LoadMapSettings();
        LogMessage("Using default settings for map: %s", mapName);
    }
    else
    {
        SetDefaultValues();
        LogMessage("No settings found, using hardcoded defaults");
    }

    g_hConfig.Rewind();
}

void LoadMapSettings()
{
    g_iMaxZombies = g_hConfig.GetNum("max_zombies", 30);
    g_iTankHealth = g_hConfig.GetNum("tank_health", 6000);
    g_fHordeInterval = g_hConfig.GetFloat("horde_interval", 120.0);
    g_bAllowWitches = g_hConfig.GetNum("allow_witches", 1) == 1;
}

void SetDefaultValues()
{
    g_iMaxZombies = 30;
    g_iTankHealth = 6000;
    g_fHordeInterval = 120.0;
    g_bAllowWitches = true;
}

void ApplyConfiguration()
{
    // Apply zombie limit
    ServerCommand("z_common_limit %d", g_iMaxZombies);

    // Log applied settings
    LogMessage("Applied config - Zombies: %d, Tank HP: %d, Horde: %.0fs, Witches: %s",
        g_iMaxZombies, g_iTankHealth, g_fHordeInterval, g_bAllowWitches ? "Yes" : "No");

    PrintToChatAll("\\x04[CONFIG] \\x01Map settings loaded successfully.");
}

void CreateDefaultConfig(const char[] path)
{
    KeyValues kv = new KeyValues("MapSettings");

    // Default settings
    kv.JumpToKey("default", true);
    kv.SetNum("max_zombies", 30);
    kv.SetNum("tank_health", 6000);
    kv.SetFloat("horde_interval", 120.0);
    kv.SetNum("allow_witches", 1);
    kv.GoBack();

    // Dead Center settings
    kv.JumpToKey("c1m1_hotel", true);
    kv.SetNum("max_zombies", 25);
    kv.SetNum("tank_health", 5000);
    kv.SetFloat("horde_interval", 90.0);
    kv.SetNum("allow_witches", 1);
    kv.GoBack();

    // Dark Carnival finale
    kv.JumpToKey("c2m5_concert", true);
    kv.SetNum("max_zombies", 40);
    kv.SetNum("tank_health", 8000);
    kv.SetFloat("horde_interval", 60.0);
    kv.SetNum("allow_witches", 0);
    kv.GoBack();

    kv.ExportToFile(path);
    delete kv;
}

Action Command_ReloadConfig(int client, int args)
{
    LoadConfiguration();
    ApplyConfiguration();
    ReplyToCommand(client, "[SM] Configuration reloaded.");
    return Plugin_Handled;
}

Action Command_ShowConfig(int client, int args)
{
    char mapName[64];
    GetCurrentMap(mapName, sizeof(mapName));

    ReplyToCommand(client, "[SM] Current Map: %s", mapName);
    ReplyToCommand(client, "[SM] Max Zombies: %d", g_iMaxZombies);
    ReplyToCommand(client, "[SM] Tank Health: %d", g_iTankHealth);
    ReplyToCommand(client, "[SM] Horde Interval: %.0f seconds", g_fHordeInterval);
    ReplyToCommand(client, "[SM] Witches Allowed: %s", g_bAllowWitches ? "Yes" : "No");

    return Plugin_Handled;
}'''
    },

    # === ADDITIONAL TEMPLATES TO REACH 30+ ===
    "tank_frustration_handler": {
        "user": "Write a plugin that monitors Tank frustration levels and takes action when it reaches critical levels",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

Handle g_hFrustrationTimer = null;
int g_iCurrentTank = -1;

public Plugin myinfo =
{
    name = "Tank Frustration Monitor",
    author = "Developer",
    description = "Monitors and handles Tank frustration",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("tank_spawn", Event_TankSpawn);
    HookEvent("player_death", Event_PlayerDeath);
}

public void Event_TankSpawn(Event event, const char[] name, bool dontBroadcast)
{
    int tank = GetClientOfUserId(event.GetInt("userid"));
    if (tank > 0 && IsClientInGame(tank))
    {
        g_iCurrentTank = tank;
        StartFrustrationMonitor();
    }
}

public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast)
{
    int victim = GetClientOfUserId(event.GetInt("userid"));
    if (victim == g_iCurrentTank)
    {
        StopFrustrationMonitor();
        g_iCurrentTank = -1;
    }
}

void StartFrustrationMonitor()
{
    StopFrustrationMonitor();
    g_hFrustrationTimer = CreateTimer(1.0, Timer_CheckFrustration, _, TIMER_REPEAT | TIMER_FLAG_NO_MAPCHANGE);
}

void StopFrustrationMonitor()
{
    if (g_hFrustrationTimer != null)
    {
        KillTimer(g_hFrustrationTimer);
        g_hFrustrationTimer = null;
    }
}

Action Timer_CheckFrustration(Handle timer)
{
    if (g_iCurrentTank <= 0 || !IsClientInGame(g_iCurrentTank) || !IsPlayerAlive(g_iCurrentTank))
    {
        g_hFrustrationTimer = null;
        return Plugin_Stop;
    }

    int frustration = GetEntProp(g_iCurrentTank, Prop_Send, "m_frustration");

    if (frustration >= 90)
    {
        PrintToChatAll("\\x04[TANK] \\x01Tank frustration critical: %d%%!", frustration);
    }
    else if (frustration >= 50)
    {
        PrintToChatAll("\\x04[TANK] \\x01Tank getting frustrated: %d%%", frustration);
    }

    return Plugin_Continue;
}'''
    },

    "witch_bride_detector": {
        "user": "Create a plugin that detects Witch Brides (wedding witches) and applies special behavior",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

public Plugin myinfo =
{
    name = "Witch Bride Detector",
    author = "Developer",
    description = "Detects and handles Witch Bride variants",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("witch_spawn", Event_WitchSpawn);
}

public void OnEntityCreated(int entity, const char[] classname)
{
    if (StrEqual(classname, "witch"))
    {
        CreateTimer(0.1, Timer_CheckWitchType, EntIndexToEntRef(entity), TIMER_FLAG_NO_MAPCHANGE);
    }
}

Action Timer_CheckWitchType(Handle timer, int ref)
{
    int witch = EntRefToEntIndex(ref);
    if (witch == INVALID_ENT_REFERENCE || !IsValidEntity(witch))
        return Plugin_Stop;

    char model[PLATFORM_MAX_PATH];
    GetEntPropString(witch, Prop_Data, "m_ModelName", model, sizeof(model));

    if (StrContains(model, "bride") != -1)
    {
        HandleBrideWitch(witch);
    }

    return Plugin_Stop;
}

void HandleBrideWitch(int witch)
{
    PrintToChatAll("\\x04[WITCH] \\x01A Witch Bride has spawned! She's more aggressive!");

    // Make bride witch more dangerous
    int health = GetEntProp(witch, Prop_Data, "m_iHealth");
    SetEntProp(witch, Prop_Data, "m_iHealth", health + 500);
    SetEntProp(witch, Prop_Data, "m_iMaxHealth", health + 500);
}

public void Event_WitchSpawn(Event event, const char[] name, bool dontBroadcast)
{
    int witch = event.GetInt("witchid");
    if (IsValidEntity(witch))
    {
        CreateTimer(0.1, Timer_CheckWitchType, EntIndexToEntRef(witch), TIMER_FLAG_NO_MAPCHANGE);
    }
}'''
    },

    "onconfigsexecuted_handler": {
        "user": "Write a plugin that uses OnConfigsExecuted to apply final settings after all configs have loaded",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

ConVar g_cvEnabled;
ConVar g_cvDifficulty;

bool g_bConfigsLoaded = false;

public Plugin myinfo =
{
    name = "Config Execution Handler",
    author = "Developer",
    description = "Demonstrates OnConfigsExecuted usage",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    g_cvEnabled = CreateConVar("sm_mymod_enabled", "1", "Enable the mod");
    g_cvDifficulty = CreateConVar("sm_mymod_difficulty", "normal", "Difficulty setting");

    AutoExecConfig(true, "mymod");

    RegAdminCmd("sm_configstatus", Command_ConfigStatus, ADMFLAG_GENERIC, "Show config status");
}

public void OnMapStart()
{
    g_bConfigsLoaded = false;
    LogMessage("Map starting, waiting for configs...");
}

public void OnConfigsExecuted()
{
    g_bConfigsLoaded = true;
    LogMessage("All configs executed, applying final settings");

    ApplyFinalSettings();
}

void ApplyFinalSettings()
{
    if (!g_cvEnabled.BoolValue)
    {
        LogMessage("Mod disabled in config, skipping initialization");
        return;
    }

    char difficulty[32];
    g_cvDifficulty.GetString(difficulty, sizeof(difficulty));

    if (StrEqual(difficulty, "hard"))
    {
        ServerCommand("z_common_limit 35");
        ServerCommand("z_mega_mob_size 60");
        LogMessage("Applied HARD difficulty settings");
    }
    else if (StrEqual(difficulty, "easy"))
    {
        ServerCommand("z_common_limit 20");
        ServerCommand("z_mega_mob_size 30");
        LogMessage("Applied EASY difficulty settings");
    }
    else
    {
        ServerCommand("z_common_limit 30");
        ServerCommand("z_mega_mob_size 50");
        LogMessage("Applied NORMAL difficulty settings");
    }

    PrintToChatAll("\\x04[CONFIG] \\x01Server configured for %s difficulty.", difficulty);
}

Action Command_ConfigStatus(int client, int args)
{
    ReplyToCommand(client, "[SM] Configs loaded: %s", g_bConfigsLoaded ? "Yes" : "No");

    char difficulty[32];
    g_cvDifficulty.GetString(difficulty, sizeof(difficulty));
    ReplyToCommand(client, "[SM] Current difficulty: %s", difficulty);

    return Plugin_Handled;
}'''
    },

    "sql_async_queries": {
        "user": "Write a plugin demonstrating proper async/threaded SQL queries to prevent server lag",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>

Database g_hDatabase = null;

public Plugin myinfo =
{
    name = "Async SQL Example",
    author = "Developer",
    description = "Demonstrates threaded SQL queries",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    RegConsoleCmd("sm_asynctest", Command_AsyncTest, "Test async query");
    RegConsoleCmd("sm_synctest", Command_SyncTest, "Test sync query (bad)");

    ConnectDatabase();
}

void ConnectDatabase()
{
    char error[256];

    // Using threaded connection
    if (SQL_CheckConfig("storage-local"))
    {
        Database.Connect(OnDatabaseConnected, "storage-local");
    }
    else
    {
        g_hDatabase = SQLite_UseDatabase("mydata", error, sizeof(error));
        if (g_hDatabase == null)
        {
            LogError("Database error: %s", error);
        }
        else
        {
            CreateTables();
        }
    }
}

void OnDatabaseConnected(Database db, const char[] error, any data)
{
    if (db == null)
    {
        LogError("Failed to connect: %s", error);
        return;
    }

    g_hDatabase = db;
    LogMessage("Database connected (async)");
    CreateTables();
}

void CreateTables()
{
    // Use threaded query for table creation
    g_hDatabase.Query(OnTableCreated,
        "CREATE TABLE IF NOT EXISTS player_data ("
        ... "steamid VARCHAR(32) PRIMARY KEY,"
        ... "score INT DEFAULT 0,"
        ... "last_seen DATETIME DEFAULT CURRENT_TIMESTAMP"
        ... ")");
}

void OnTableCreated(Database db, DBResultSet results, const char[] error, any data)
{
    if (results == null)
    {
        LogError("Table creation failed: %s", error);
    }
    else
    {
        LogMessage("Tables ready");
    }
}

Action Command_AsyncTest(int client, int args)
{
    if (g_hDatabase == null)
    {
        ReplyToCommand(client, "[SM] Database not connected");
        return Plugin_Handled;
    }

    // GOOD: Threaded query - won't block the server
    char steamId[32];
    GetClientAuthId(client, AuthId_Steam2, steamId, sizeof(steamId));

    char query[256];
    Format(query, sizeof(query), "SELECT score FROM player_data WHERE steamid = '%s'", steamId);

    DataPack pack = new DataPack();
    pack.WriteCell(GetClientUserId(client));

    g_hDatabase.Query(OnAsyncQueryComplete, query, pack);

    ReplyToCommand(client, "[SM] Query sent (async, no lag)");
    return Plugin_Handled;
}

void OnAsyncQueryComplete(Database db, DBResultSet results, const char[] error, DataPack pack)
{
    pack.Reset();
    int userid = pack.ReadCell();
    delete pack;

    int client = GetClientOfUserId(userid);
    if (client <= 0)
        return;

    if (results == null)
    {
        PrintToChat(client, "[SM] Query error: %s", error);
        return;
    }

    if (results.FetchRow())
    {
        int score = results.FetchInt(0);
        PrintToChat(client, "[SM] Your score: %d", score);
    }
    else
    {
        PrintToChat(client, "[SM] No record found, creating...");
        InsertPlayerRecord(client);
    }
}

void InsertPlayerRecord(int client)
{
    char steamId[32];
    GetClientAuthId(client, AuthId_Steam2, steamId, sizeof(steamId));

    char query[256];
    Format(query, sizeof(query), "INSERT INTO player_data (steamid, score) VALUES ('%s', 0)", steamId);

    g_hDatabase.Query(OnInsertComplete, query);
}

void OnInsertComplete(Database db, DBResultSet results, const char[] error, any data)
{
    if (results == null)
    {
        LogError("Insert failed: %s", error);
    }
}

Action Command_SyncTest(int client, int args)
{
    ReplyToCommand(client, "[SM] WARNING: Sync queries can cause server lag!");
    ReplyToCommand(client, "[SM] Always use threaded queries instead.");
    return Plugin_Handled;
}'''
    },

    "vote_menu_system": {
        "user": "Create a plugin with a voting menu system that lets players vote on game settings",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

bool g_bVoteInProgress = false;
int g_iVoteYes = 0;
int g_iVoteNo = 0;
bool g_bHasVoted[MAXPLAYERS + 1];
char g_sVoteTopic[128];

public Plugin myinfo =
{
    name = "Vote Menu System",
    author = "Developer",
    description = "Player voting system for game settings",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    RegConsoleCmd("sm_vote", Command_StartVote, "Start a vote");
    RegConsoleCmd("sm_votekick", Command_VoteKick, "Start a kick vote");
    RegConsoleCmd("sm_votedifficulty", Command_VoteDifficulty, "Vote on difficulty");
}

Action Command_StartVote(int client, int args)
{
    if (g_bVoteInProgress)
    {
        ReplyToCommand(client, "[SM] A vote is already in progress.");
        return Plugin_Handled;
    }

    ShowVoteTopicMenu(client);
    return Plugin_Handled;
}

void ShowVoteTopicMenu(int client)
{
    Menu menu = new Menu(MenuHandler_VoteTopic);
    menu.SetTitle("Select Vote Topic");

    menu.AddItem("restart", "Restart Round");
    menu.AddItem("alltalk", "Toggle Alltalk");
    menu.AddItem("difficulty", "Change Difficulty");

    menu.Display(client, MENU_TIME_FOREVER);
}

int MenuHandler_VoteTopic(Menu menu, MenuAction action, int client, int param2)
{
    switch (action)
    {
        case MenuAction_Select:
        {
            char selection[32];
            menu.GetItem(param2, selection, sizeof(selection));

            if (StrEqual(selection, "restart"))
                StartVote("Restart the round?");
            else if (StrEqual(selection, "alltalk"))
                StartVote("Enable alltalk?");
            else if (StrEqual(selection, "difficulty"))
                StartVote("Increase difficulty?");
        }
        case MenuAction_End:
        {
            delete menu;
        }
    }
    return 0;
}

void StartVote(const char[] topic)
{
    g_bVoteInProgress = true;
    g_iVoteYes = 0;
    g_iVoteNo = 0;
    strcopy(g_sVoteTopic, sizeof(g_sVoteTopic), topic);

    for (int i = 1; i <= MaxClients; i++)
    {
        g_bHasVoted[i] = false;

        if (IsClientInGame(i) && !IsFakeClient(i))
        {
            ShowVoteMenu(i);
        }
    }

    PrintToChatAll("\\x04[VOTE] \\x01Vote started: %s", topic);

    CreateTimer(20.0, Timer_EndVote);
}

void ShowVoteMenu(int client)
{
    Menu menu = new Menu(MenuHandler_Vote);
    menu.SetTitle("VOTE: %s", g_sVoteTopic);

    menu.AddItem("yes", "Yes");
    menu.AddItem("no", "No");

    menu.ExitButton = false;
    menu.Display(client, 20);
}

int MenuHandler_Vote(Menu menu, MenuAction action, int client, int param2)
{
    switch (action)
    {
        case MenuAction_Select:
        {
            if (g_bHasVoted[client])
                return 0;

            g_bHasVoted[client] = true;

            char selection[8];
            menu.GetItem(param2, selection, sizeof(selection));

            if (StrEqual(selection, "yes"))
            {
                g_iVoteYes++;
                PrintToChat(client, "\\x04[VOTE] \\x01You voted YES.");
            }
            else
            {
                g_iVoteNo++;
                PrintToChat(client, "\\x04[VOTE] \\x01You voted NO.");
            }
        }
        case MenuAction_End:
        {
            delete menu;
        }
    }
    return 0;
}

Action Timer_EndVote(Handle timer)
{
    g_bVoteInProgress = false;

    int totalVotes = g_iVoteYes + g_iVoteNo;
    bool passed = (totalVotes > 0) && (float(g_iVoteYes) / float(totalVotes) > 0.5);

    PrintToChatAll("\\x04[VOTE] \\x01Results: YES: %d, NO: %d - Vote %s!",
        g_iVoteYes, g_iVoteNo, passed ? "PASSED" : "FAILED");

    if (passed)
    {
        ExecuteVoteResult();
    }

    return Plugin_Stop;
}

void ExecuteVoteResult()
{
    PrintToChatAll("\\x04[VOTE] \\x01Executing vote result...");
    // Implementation would depend on what was voted on
}

Action Command_VoteKick(int client, int args) { return Plugin_Handled; }
Action Command_VoteDifficulty(int client, int args) { return Plugin_Handled; }'''
    },

    "private_forward_example": {
        "user": "Write a plugin that creates private forwards for internal plugin communication",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>

// Private forwards (only this plugin can call them)
PrivateForward g_hOnInternalEvent;
PrivateForward g_hOnDataProcessed;

// Registered callback functions
ArrayList g_hCallbacks;

public Plugin myinfo =
{
    name = "Private Forward Example",
    author = "Developer",
    description = "Demonstrates private forwards",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    // Create private forwards
    g_hOnInternalEvent = new PrivateForward(ET_Ignore, Param_Cell, Param_String);
    g_hOnDataProcessed = new PrivateForward(ET_Event, Param_Cell, Param_CellByRef);

    g_hCallbacks = new ArrayList();

    // Register our own internal handlers
    g_hOnInternalEvent.AddFunction(null, InternalEventHandler);
    g_hOnDataProcessed.AddFunction(null, DataProcessedHandler);

    RegConsoleCmd("sm_triggerevent", Command_TriggerEvent, "Trigger internal event");
    RegConsoleCmd("sm_processdata", Command_ProcessData, "Process some data");
}

// Internal handler for events
void InternalEventHandler(int eventType, const char[] eventData)
{
    LogMessage("Internal event received - Type: %d, Data: %s", eventType, eventData);
}

// Internal handler for data processing
Action DataProcessedHandler(int dataId, int &modifiedValue)
{
    LogMessage("Processing data ID %d, current value: %d", dataId, modifiedValue);

    // Modify the value
    modifiedValue *= 2;

    return Plugin_Changed;
}

Action Command_TriggerEvent(int client, int args)
{
    // Fire the private forward
    Call_StartForward(g_hOnInternalEvent);
    Call_PushCell(1);  // Event type
    Call_PushString("Test event triggered");
    Call_Finish();

    ReplyToCommand(client, "[SM] Internal event triggered.");
    return Plugin_Handled;
}

Action Command_ProcessData(int client, int args)
{
    int originalValue = 100;
    int modifiedValue = originalValue;

    // Fire the forward with modifiable parameter
    Action result;
    Call_StartForward(g_hOnDataProcessed);
    Call_PushCell(42);  // Data ID
    Call_PushCellRef(modifiedValue);
    Call_Finish(result);

    ReplyToCommand(client, "[SM] Original: %d, Modified: %d", originalValue, modifiedValue);
    return Plugin_Handled;
}

// Allow other parts of the plugin to add callbacks
void RegisterCallback(Function callback)
{
    g_hOnInternalEvent.AddFunction(null, callback);
}

public void OnPluginEnd()
{
    delete g_hOnInternalEvent;
    delete g_hOnDataProcessed;
    delete g_hCallbacks;
}'''
    },

    "sdkhooks_damage_modification": {
        "user": "Write a plugin using SDKHooks to modify damage with OnTakeDamage and OnTakeDamageAlive",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>

ConVar g_cvDamageMultiplier;
ConVar g_cvHeadshotBonus;
ConVar g_cvTankResistance;

public Plugin myinfo =
{
    name = "Damage Modifier",
    author = "Developer",
    description = "Modifies damage using SDKHooks",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    g_cvDamageMultiplier = CreateConVar("sm_dmg_multiplier", "1.0", "Global damage multiplier", FCVAR_NOTIFY, true, 0.1, true, 10.0);
    g_cvHeadshotBonus = CreateConVar("sm_dmg_headshot", "1.5", "Headshot damage multiplier", FCVAR_NOTIFY, true, 1.0, true, 5.0);
    g_cvTankResistance = CreateConVar("sm_dmg_tank_resist", "0.8", "Tank damage resistance", FCVAR_NOTIFY, true, 0.1, true, 1.0);

    for (int i = 1; i <= MaxClients; i++)
    {
        if (IsClientInGame(i))
        {
            OnClientPutInServer(i);
        }
    }
}

public void OnClientPutInServer(int client)
{
    SDKHook(client, SDKHook_OnTakeDamage, OnTakeDamage);
    SDKHook(client, SDKHook_OnTakeDamageAlive, OnTakeDamageAlive);
}

public void OnClientDisconnect(int client)
{
    SDKUnhook(client, SDKHook_OnTakeDamage, OnTakeDamage);
    SDKUnhook(client, SDKHook_OnTakeDamageAlive, OnTakeDamageAlive);
}

// Called before damage is applied - can block entirely
Action OnTakeDamage(int victim, int &attacker, int &inflictor, float &damage, int &damagetype)
{
    if (victim <= 0 || victim > MaxClients || !IsClientInGame(victim))
        return Plugin_Continue;

    // Check for Tank - apply resistance
    if (GetClientTeam(victim) == 3)
    {
        int zombieClass = GetEntProp(victim, Prop_Send, "m_zombieClass");
        if (zombieClass == 8) // Tank
        {
            damage *= g_cvTankResistance.FloatValue;
            return Plugin_Changed;
        }
    }

    // Block self-damage for survivors
    if (attacker == victim && GetClientTeam(victim) == 2)
    {
        if (damagetype & DMG_FALL)
        {
            // Allow fall damage but reduce it
            damage *= 0.5;
            return Plugin_Changed;
        }
    }

    return Plugin_Continue;
}

// Called after damage is confirmed but before health is modified
Action OnTakeDamageAlive(int victim, int &attacker, int &inflictor, float &damage, int &damagetype)
{
    if (attacker <= 0 || attacker > MaxClients || !IsClientInGame(attacker))
        return Plugin_Continue;

    // Apply global multiplier
    float originalDamage = damage;
    damage *= g_cvDamageMultiplier.FloatValue;

    // Check for headshot
    if (damagetype & DMG_HEADSHOT || (GetEntProp(victim, Prop_Send, "m_LastHitGroup") == 1))
    {
        damage *= g_cvHeadshotBonus.FloatValue;

        if (GetClientTeam(attacker) == 2) // Survivor
        {
            PrintToChat(attacker, "\\x04[HEADSHOT] \\x01Bonus damage: %.0f -> %.0f", originalDamage, damage);
        }
    }

    // Log significant damage
    if (damage >= 100.0)
    {
        char attackerName[64], victimName[64];
        GetClientName(attacker, attackerName, sizeof(attackerName));
        GetClientName(victim, victimName, sizeof(victimName));

        LogMessage("High damage: %s dealt %.0f to %s", attackerName, damage, victimName);
    }

    return Plugin_Changed;
}'''
    },

    "array_stringmap_usage": {
        "user": "Write a plugin demonstrating ArrayList and StringMap usage for data storage",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>

// Data structures
ArrayList g_hPlayerList;          // Simple list of player userids
StringMap g_hPlayerScores;        // Map of steamid -> score
StringMap g_hPlayerData;          // Map of steamid -> PlayerData enum struct

enum struct PlayerData
{
    int kills;
    int deaths;
    float playtime;
    char lastWeapon[64];
}

public Plugin myinfo =
{
    name = "Data Structures Example",
    author = "Developer",
    description = "Demonstrates ArrayList and StringMap",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    // Initialize data structures
    g_hPlayerList = new ArrayList();
    g_hPlayerScores = new StringMap();
    g_hPlayerData = new StringMap();

    RegConsoleCmd("sm_datastats", Command_DataStats, "Show data structure stats");
    RegConsoleCmd("sm_myscore", Command_MyScore, "Show your score");

    HookEvent("player_death", Event_PlayerDeath);
}

public void OnPluginEnd()
{
    // Clean up
    delete g_hPlayerList;
    delete g_hPlayerScores;
    delete g_hPlayerData;
}

public void OnClientPostAdminCheck(int client)
{
    if (IsFakeClient(client))
        return;

    char steamId[32];
    GetClientAuthId(client, AuthId_Steam2, steamId, sizeof(steamId));

    // Add to player list
    g_hPlayerList.Push(GetClientUserId(client));

    // Initialize score if not exists
    int existingScore;
    if (!g_hPlayerScores.GetValue(steamId, existingScore))
    {
        g_hPlayerScores.SetValue(steamId, 0);
    }

    // Initialize player data
    PlayerData data;
    if (!g_hPlayerData.GetArray(steamId, data, sizeof(data)))
    {
        data.kills = 0;
        data.deaths = 0;
        data.playtime = 0.0;
        data.lastWeapon[0] = '\\0';
        g_hPlayerData.SetArray(steamId, data, sizeof(data));
    }
}

public void OnClientDisconnect(int client)
{
    // Remove from player list
    int userid = GetClientUserId(client);
    int index = g_hPlayerList.FindValue(userid);
    if (index != -1)
    {
        g_hPlayerList.Erase(index);
    }
}

public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast)
{
    int victim = GetClientOfUserId(event.GetInt("userid"));
    int attacker = GetClientOfUserId(event.GetInt("attacker"));
    char weapon[64];
    event.GetString("weapon", weapon, sizeof(weapon));

    // Update victim deaths
    if (victim > 0 && victim <= MaxClients && IsClientInGame(victim) && !IsFakeClient(victim))
    {
        char steamId[32];
        GetClientAuthId(victim, AuthId_Steam2, steamId, sizeof(steamId));

        PlayerData data;
        if (g_hPlayerData.GetArray(steamId, data, sizeof(data)))
        {
            data.deaths++;
            g_hPlayerData.SetArray(steamId, data, sizeof(data));
        }
    }

    // Update attacker kills and score
    if (attacker > 0 && attacker <= MaxClients && IsClientInGame(attacker) && !IsFakeClient(attacker))
    {
        char steamId[32];
        GetClientAuthId(attacker, AuthId_Steam2, steamId, sizeof(steamId));

        // Update score in StringMap
        int score;
        g_hPlayerScores.GetValue(steamId, score);
        score += 10;
        g_hPlayerScores.SetValue(steamId, score);

        // Update detailed player data
        PlayerData data;
        if (g_hPlayerData.GetArray(steamId, data, sizeof(data)))
        {
            data.kills++;
            strcopy(data.lastWeapon, sizeof(data.lastWeapon), weapon);
            g_hPlayerData.SetArray(steamId, data, sizeof(data));
        }
    }
}

Action Command_DataStats(int client, int args)
{
    ReplyToCommand(client, "[SM] Active players: %d", g_hPlayerList.Length);
    ReplyToCommand(client, "[SM] Tracked scores: %d", g_hPlayerScores.Size);
    ReplyToCommand(client, "[SM] Player records: %d", g_hPlayerData.Size);
    return Plugin_Handled;
}

Action Command_MyScore(int client, int args)
{
    char steamId[32];
    GetClientAuthId(client, AuthId_Steam2, steamId, sizeof(steamId));

    int score;
    g_hPlayerScores.GetValue(steamId, score);

    PlayerData data;
    g_hPlayerData.GetArray(steamId, data, sizeof(data));

    PrintToChat(client, "\\x04=== Your Stats ===");
    PrintToChat(client, "Score: %d | K/D: %d/%d", score, data.kills, data.deaths);
    PrintToChat(client, "Last weapon: %s", data.lastWeapon);

    return Plugin_Handled;
}'''
    },

    "entity_output_hooks": {
        "user": "Write a plugin that hooks entity outputs like OnStartTouch and OnTrigger",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

public Plugin myinfo =
{
    name = "Entity Output Hooks",
    author = "Developer",
    description = "Hooks entity outputs for triggers",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEntityOutput("trigger_once", "OnStartTouch", OnTriggerStartTouch);
    HookEntityOutput("trigger_multiple", "OnStartTouch", OnTriggerStartTouch);
    HookEntityOutput("trigger_multiple", "OnTrigger", OnTriggerFired);
    HookEntityOutput("func_button", "OnPressed", OnButtonPressed);
    HookEntityOutput("prop_door_rotating", "OnOpen", OnDoorOpen);
    HookEntityOutput("prop_door_rotating", "OnClose", OnDoorClose);
}

void OnTriggerStartTouch(const char[] output, int caller, int activator, float delay)
{
    if (activator <= 0 || activator > MaxClients)
        return;

    if (!IsClientInGame(activator))
        return;

    char targetname[64];
    GetEntPropString(caller, Prop_Data, "m_iName", targetname, sizeof(targetname));

    char playerName[64];
    GetClientName(activator, playerName, sizeof(playerName));

    LogMessage("Player %s touched trigger: %s", playerName, targetname);

    // Check for special trigger names
    if (StrContains(targetname, "checkpoint") != -1)
    {
        PrintToChat(activator, "\\x04[CHECKPOINT] \\x01You reached a checkpoint!");
    }
    else if (StrContains(targetname, "finale") != -1)
    {
        PrintToChatAll("\\x04[FINALE] \\x03%s \\x01triggered the finale!", playerName);
    }
}

void OnTriggerFired(const char[] output, int caller, int activator, float delay)
{
    char targetname[64];
    GetEntPropString(caller, Prop_Data, "m_iName", targetname, sizeof(targetname));

    LogMessage("Trigger fired: %s (activator: %d)", targetname, activator);
}

void OnButtonPressed(const char[] output, int caller, int activator, float delay)
{
    if (activator <= 0 || activator > MaxClients || !IsClientInGame(activator))
        return;

    char buttonName[64];
    GetEntPropString(caller, Prop_Data, "m_iName", buttonName, sizeof(buttonName));

    char playerName[64];
    GetClientName(activator, playerName, sizeof(playerName));

    // Check for special buttons
    if (StrContains(buttonName, "elevator") != -1)
    {
        PrintToChatAll("\\x04[ELEVATOR] \\x03%s \\x01called the elevator!", playerName);
    }
    else if (StrContains(buttonName, "radio") != -1)
    {
        PrintToChatAll("\\x04[RADIO] \\x03%s \\x01used the radio!", playerName);
    }
}

void OnDoorOpen(const char[] output, int caller, int activator, float delay)
{
    char doorName[64];
    GetEntPropString(caller, Prop_Data, "m_iName", doorName, sizeof(doorName));

    // Check for saferoom doors
    if (StrContains(doorName, "saferoom") != -1 || StrContains(doorName, "checkpoint") != -1)
    {
        if (activator > 0 && activator <= MaxClients && IsClientInGame(activator))
        {
            char playerName[64];
            GetClientName(activator, playerName, sizeof(playerName));
            PrintToChatAll("\\x04[SAFEROOM] \\x03%s \\x01opened the saferoom door!", playerName);
        }
    }
}

void OnDoorClose(const char[] output, int caller, int activator, float delay)
{
    char doorName[64];
    GetEntPropString(caller, Prop_Data, "m_iName", doorName, sizeof(doorName));

    if (StrContains(doorName, "saferoom") != -1)
    {
        PrintToChatAll("\\x04[SAFEROOM] \\x01The saferoom door has been closed!");
    }
}'''
    },

    "translation_phrases": {
        "user": "Write a plugin that uses translation files for multi-language support",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>

public Plugin myinfo =
{
    name = "Translation Example",
    author = "Developer",
    description = "Demonstrates translation file usage",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    // Load translation files
    LoadTranslations("common.phrases");
    LoadTranslations("mymod.phrases");

    RegConsoleCmd("sm_hello", Command_Hello, "Say hello in your language");
    RegConsoleCmd("sm_stats", Command_Stats, "Show your stats");
    RegAdminCmd("sm_kick", Command_Kick, ADMFLAG_KICK, "Kick a player");
}

Action Command_Hello(int client, int args)
{
    // Simple translation
    PrintToChat(client, "%t", "Welcome Message");

    // Translation with player name
    char name[64];
    GetClientName(client, name, sizeof(name));
    PrintToChat(client, "%t", "Hello Player", name);

    return Plugin_Handled;
}

Action Command_Stats(int client, int args)
{
    int kills = 42;
    int deaths = 10;
    float ratio = float(kills) / float(deaths);

    // Translation with multiple parameters
    PrintToChat(client, "%t", "Stats Display", kills, deaths, ratio);

    // Conditional translation
    if (kills > deaths)
    {
        PrintToChat(client, "%t", "Good Performance");
    }
    else
    {
        PrintToChat(client, "%t", "Need Improvement");
    }

    return Plugin_Handled;
}

Action Command_Kick(int client, int args)
{
    if (args < 1)
    {
        // Use common.phrases for standard messages
        ReplyToCommand(client, "[SM] %t", "Usage: sm_kick <target>");
        return Plugin_Handled;
    }

    char arg[64];
    GetCmdArg(1, arg, sizeof(arg));

    int target = FindTarget(client, arg, true, false);
    if (target == -1)
        return Plugin_Handled;

    char targetName[64], adminName[64];
    GetClientName(target, targetName, sizeof(targetName));
    GetClientName(client, adminName, sizeof(adminName));

    // Announce to all in their language
    for (int i = 1; i <= MaxClients; i++)
    {
        if (IsClientInGame(i) && !IsFakeClient(i))
        {
            PrintToChat(i, "%t", "Player Kicked", adminName, targetName);
        }
    }

    KickClient(target, "%t", "You Were Kicked");

    return Plugin_Handled;
}

// Example translation file content for addons/sourcemod/translations/mymod.phrases.txt:
/*
"Phrases"
{
    "Welcome Message"
    {
        "en"    "Welcome to our server!"
        "de"    "Willkommen auf unserem Server!"
        "fr"    "Bienvenue sur notre serveur!"
        "ru"    "Dobro pozhalovat' na nash server!"
    }

    "Hello Player"
    {
        "#format"   "{1:s}"
        "en"        "Hello, {1}!"
        "de"        "Hallo, {1}!"
        "fr"        "Bonjour, {1}!"
    }

    "Stats Display"
    {
        "#format"   "{1:d},{2:d},{3:.2f}"
        "en"        "Kills: {1} | Deaths: {2} | K/D Ratio: {3}"
        "de"        "Kills: {1} | Tode: {2} | K/D-Verhaltnis: {3}"
    }

    "Good Performance"
    {
        "en"    "Great job! Keep it up!"
        "de"    "Gute Arbeit! Weiter so!"
    }

    "Need Improvement"
    {
        "en"    "Room for improvement. Practice more!"
        "de"    "Raum fur Verbesserungen. Ube mehr!"
    }

    "Player Kicked"
    {
        "#format"   "{1:s},{2:s}"
        "en"        "{1} kicked {2} from the server."
        "de"        "{1} hat {2} vom Server geworfen."
    }

    "You Were Kicked"
    {
        "en"    "You have been kicked from the server."
        "de"    "Du wurdest vom Server geworfen."
    }
}
*/'''
    },

    "client_prefs_cookies": {
        "user": "Write a plugin using clientprefs cookies to save player preferences",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <clientprefs>

// Cookie handles
Cookie g_hHudEnabledCookie;
Cookie g_hVolumePreferenceCookie;
Cookie g_hColorPreferenceCookie;

// Local cached values
bool g_bHudEnabled[MAXPLAYERS + 1];
int g_iVolumeLevel[MAXPLAYERS + 1];
int g_iColorPreference[MAXPLAYERS + 1];

public Plugin myinfo =
{
    name = "Client Preferences",
    author = "Developer",
    description = "Saves player preferences using cookies",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    // Create cookies
    g_hHudEnabledCookie = new Cookie("mymod_hud", "HUD enabled preference", CookieAccess_Protected);
    g_hVolumePreferenceCookie = new Cookie("mymod_volume", "Sound volume preference", CookieAccess_Protected);
    g_hColorPreferenceCookie = new Cookie("mymod_color", "Color preference", CookieAccess_Protected);

    // Register cookie menu items
    SetCookieMenuItem(CookieMenuHandler, g_hHudEnabledCookie, "My Mod Preferences");

    // Commands
    RegConsoleCmd("sm_togglehud", Command_ToggleHud, "Toggle HUD display");
    RegConsoleCmd("sm_setvolume", Command_SetVolume, "Set sound volume");
    RegConsoleCmd("sm_setcolor", Command_SetColor, "Set color preference");
    RegConsoleCmd("sm_settings", Command_Settings, "Open settings menu");
}

public void OnClientCookiesCached(int client)
{
    // Load cookies when they become available
    LoadClientCookies(client);
}

void LoadClientCookies(int client)
{
    // HUD preference
    char buffer[8];
    g_hHudEnabledCookie.Get(client, buffer, sizeof(buffer));
    g_bHudEnabled[client] = (buffer[0] == '\\0' || buffer[0] == '1');

    // Volume preference
    g_hVolumePreferenceCookie.Get(client, buffer, sizeof(buffer));
    g_iVolumeLevel[client] = (buffer[0] == '\\0') ? 100 : StringToInt(buffer);

    // Color preference
    g_hColorPreferenceCookie.Get(client, buffer, sizeof(buffer));
    g_iColorPreference[client] = (buffer[0] == '\\0') ? 0 : StringToInt(buffer);
}

public void CookieMenuHandler(int client, CookieMenuAction action, any info, char[] buffer, int maxlen)
{
    if (action == CookieMenuAction_DisplayOption)
    {
        Format(buffer, maxlen, "My Mod Settings");
    }
    else if (action == CookieMenuAction_SelectOption)
    {
        ShowSettingsMenu(client);
    }
}

void ShowSettingsMenu(int client)
{
    Menu menu = new Menu(MenuHandler_Settings);
    menu.SetTitle("My Mod Settings");

    char buffer[64];
    Format(buffer, sizeof(buffer), "HUD Display: %s", g_bHudEnabled[client] ? "ON" : "OFF");
    menu.AddItem("hud", buffer);

    Format(buffer, sizeof(buffer), "Sound Volume: %d%%", g_iVolumeLevel[client]);
    menu.AddItem("volume", buffer);

    char colorName[32];
    GetColorName(g_iColorPreference[client], colorName, sizeof(colorName));
    Format(buffer, sizeof(buffer), "Color Theme: %s", colorName);
    menu.AddItem("color", buffer);

    menu.ExitButton = true;
    menu.Display(client, MENU_TIME_FOREVER);
}

int MenuHandler_Settings(Menu menu, MenuAction action, int client, int param2)
{
    switch (action)
    {
        case MenuAction_Select:
        {
            char selection[32];
            menu.GetItem(param2, selection, sizeof(selection));

            if (StrEqual(selection, "hud"))
            {
                g_bHudEnabled[client] = !g_bHudEnabled[client];
                g_hHudEnabledCookie.Set(client, g_bHudEnabled[client] ? "1" : "0");
            }
            else if (StrEqual(selection, "volume"))
            {
                g_iVolumeLevel[client] = (g_iVolumeLevel[client] + 25) % 125;
                char buffer[8];
                IntToString(g_iVolumeLevel[client], buffer, sizeof(buffer));
                g_hVolumePreferenceCookie.Set(client, buffer);
            }
            else if (StrEqual(selection, "color"))
            {
                g_iColorPreference[client] = (g_iColorPreference[client] + 1) % 4;
                char buffer[8];
                IntToString(g_iColorPreference[client], buffer, sizeof(buffer));
                g_hColorPreferenceCookie.Set(client, buffer);
            }

            ShowSettingsMenu(client);
        }
        case MenuAction_End:
        {
            delete menu;
        }
    }
    return 0;
}

void GetColorName(int colorId, char[] buffer, int maxlen)
{
    switch (colorId)
    {
        case 0: strcopy(buffer, maxlen, "Default");
        case 1: strcopy(buffer, maxlen, "Red");
        case 2: strcopy(buffer, maxlen, "Green");
        case 3: strcopy(buffer, maxlen, "Blue");
        default: strcopy(buffer, maxlen, "Unknown");
    }
}

Action Command_ToggleHud(int client, int args) { return Plugin_Handled; }
Action Command_SetVolume(int client, int args) { return Plugin_Handled; }
Action Command_SetColor(int client, int args) { return Plugin_Handled; }
Action Command_Settings(int client, int args)
{
    ShowSettingsMenu(client);
    return Plugin_Handled;
}'''
    },
}


def create_example(user: str, assistant: str) -> dict:
    """Create a properly formatted training example in ChatML format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant}
        ]
    }


def generate_variations(templates: dict) -> list:
    """Generate additional variations of templates for diversity."""
    variations = []

    # Multi-tank variations
    multi_tank_prompts = [
        ("Write a plugin that spawns two Tanks simultaneously when the finale begins", "multi_tank"),
        ("Create a plugin that makes all active Tanks share a single health pool", "multi_tank"),
        ("Write a function that checks how many Tanks are currently alive in the game", "multi_tank"),
        ("Create a plugin that prevents more than 2 Tanks from being alive at once", "multi_tank"),
    ]

    # Witch variations
    witch_prompts = [
        ("Write a plugin that makes Witches invisible until a survivor gets within 300 units", "witch"),
        ("Create a function that finds the nearest Witch to a given survivor", "witch"),
        ("Write a plugin that teleports Witches away when survivors get too close", "witch"),
        ("Create a plugin that tracks which survivors have been attacked by Witches", "witch"),
    ]

    # Event hook variations
    event_prompts = [
        ("Write OnPluginStart that registers multiple console commands and hooks all survivor events", "events"),
        ("Create OnMapStart that precaches custom models and sounds for a custom gamemode", "events"),
        ("Write OnClientDisconnect that saves player data to a file before they leave", "events"),
        ("Create a plugin using OnConfigsExecuted to apply server settings after configs load", "events"),
    ]

    # SQL variations
    sql_prompts = [
        ("Write a plugin that stores player bans in a SQLite database", "sql"),
        ("Create a function that queries player rank from a MySQL database", "sql"),
        ("Write threaded SQL queries to avoid server lag when loading player data", "sql"),
    ]

    # Menu variations
    menu_prompts = [
        ("Create a multi-page menu that shows all available weapons", "menu"),
        ("Write a plugin with a confirmation dialog before performing admin actions", "menu"),
        ("Create a voting menu that lets players vote on game settings", "menu"),
    ]

    # Forward/native variations
    forward_prompts = [
        ("Write a plugin that calls a forward when a survivor reaches critical health", "forward"),
        ("Create natives that allow other plugins to modify game difficulty", "forward"),
        ("Write a forward that notifies plugins when the Tank's frustration changes", "forward"),
    ]

    return variations


def main():
    output_path = PROJECT_ROOT / "data" / "processed" / "synthetic_v12_specialized.jsonl"

    print("=" * 60)
    print("Generating V12 Synthetic Training Examples")
    print("Focus: Underrepresented Areas")
    print("=" * 60)
    print()

    # Generate examples from templates
    examples = []
    template_count = 0

    for template_name, template_data in TEMPLATES.items():
        example = create_example(template_data["user"], template_data["assistant"])
        example["metadata"] = {
            "template": template_name,
            "category": template_name.split("_")[0],
            "generated_at": datetime.now().isoformat(),
            "version": "v12"
        }
        examples.append(example)
        template_count += 1

    print(f"Generated {template_count} examples from templates")

    # Count by category
    categories = {}
    for ex in examples:
        cat = ex.get("metadata", {}).get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print("\nExamples by category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    print(f"\nSaved {len(examples)} examples to: {output_path}")

    # Show sample prompts
    print("\nSample prompts covered:")
    sample_templates = list(TEMPLATES.keys())[:10]
    for i, name in enumerate(sample_templates, 1):
        prompt = TEMPLATES[name]["user"][:70]
        print(f"  {i}. {prompt}...")

    print(f"\n... and {len(TEMPLATES) - 10} more templates")

    return len(examples)


if __name__ == "__main__":
    count = main()
    print(f"\nTotal examples generated: {count}")
