#!/usr/bin/env python3
"""
Generate high-quality synthetic training examples for L4D2 SourcePawn fine-tuning.

These examples cover common plugin patterns with clear task-to-code mapping.
"""

import json
from pathlib import Path

SYSTEM_PROMPT = "You are an expert SourcePawn and VScript developer for Left 4 Dead 2 SourceMod plugins. Write clean, well-documented code with proper error handling."

SYNTHETIC_EXAMPLES = [
    # === EVENT HOOKS ===
    {
        "user": "Write a SourcePawn function to detect when a Tank spawns and announce it to all players",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

public Plugin myinfo =
{
    name = "Tank Spawn Announcer",
    author = "Developer",
    description = "Announces when a Tank spawns",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("tank_spawn", Event_TankSpawn);
}

public void Event_TankSpawn(Event event, const char[] name, bool dontBroadcast)
{
    int tank = GetClientOfUserId(event.GetInt("userid"));
    if (tank > 0 && IsClientInGame(tank))
    {
        PrintToChatAll("\\x04[WARNING] \\x01A Tank has spawned!");
        EmitSoundToAll("ui/pickup_secret01.wav");
    }
}'''
    },
    {
        "user": "Create a plugin that gives survivors a temporary speed boost when they kill a special infected",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>

#define SPEED_BOOST 1.3
#define BOOST_DURATION 3.0

public Plugin myinfo =
{
    name = "Kill Speed Boost",
    author = "Developer",
    description = "Gives survivors speed boost on special infected kills",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("player_death", Event_PlayerDeath);
}

public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast)
{
    int victim = GetClientOfUserId(event.GetInt("userid"));
    int attacker = GetClientOfUserId(event.GetInt("attacker"));

    // Check if a survivor killed a special infected
    if (attacker > 0 && attacker <= MaxClients && IsClientInGame(attacker))
    {
        if (GetClientTeam(attacker) == 2) // Survivor team
        {
            if (victim > 0 && victim <= MaxClients && IsClientInGame(victim))
            {
                if (GetClientTeam(victim) == 3) // Infected team
                {
                    ApplySpeedBoost(attacker);
                }
            }
        }
    }
}

void ApplySpeedBoost(int client)
{
    SetEntPropFloat(client, Prop_Send, "m_flLaggedMovementValue", SPEED_BOOST);
    PrintToChat(client, "\\x04[BOOST] \\x01Speed boost activated!");
    CreateTimer(BOOST_DURATION, Timer_RemoveBoost, GetClientUserId(client));
}

Action Timer_RemoveBoost(Handle timer, int userid)
{
    int client = GetClientOfUserId(userid);
    if (client > 0 && IsClientInGame(client) && IsPlayerAlive(client))
    {
        SetEntPropFloat(client, Prop_Send, "m_flLaggedMovementValue", 1.0);
        PrintToChat(client, "\\x04[BOOST] \\x01Speed boost ended.");
    }
    return Plugin_Continue;
}'''
    },
    {
        "user": "Write a function to check if a player is incapacitated and revive them automatically",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

public Plugin myinfo =
{
    name = "Auto Revive",
    author = "Developer",
    description = "Automatically revives incapacitated players",
    version = "1.0",
    url = ""
};

bool IsPlayerIncapacitated(int client)
{
    if (!IsClientInGame(client) || !IsPlayerAlive(client))
        return false;

    return GetEntProp(client, Prop_Send, "m_isIncapacitated") == 1;
}

void RevivePlayer(int client)
{
    if (!IsPlayerIncapacitated(client))
        return;

    // Set incapacitated to false
    SetEntProp(client, Prop_Send, "m_isIncapacitated", 0);

    // Restore some health
    SetEntityHealth(client, 50);

    // Clear the revive count
    SetEntProp(client, Prop_Send, "m_currentReviveCount", 0);

    PrintToChat(client, "\\x04[REVIVE] \\x01You have been automatically revived!");
}

void CheckAndReviveAll()
{
    for (int i = 1; i <= MaxClients; i++)
    {
        if (IsClientInGame(i) && GetClientTeam(i) == 2) // Survivor
        {
            if (IsPlayerIncapacitated(i))
            {
                RevivePlayer(i);
            }
        }
    }
}'''
    },
    # === ADMIN COMMANDS ===
    {
        "user": "Create a SourcePawn command that teleports all survivors to the saferoom",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

public Plugin myinfo =
{
    name = "Teleport to Saferoom",
    author = "Developer",
    description = "Teleports all survivors to the end saferoom",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    RegAdminCmd("sm_tosaferoom", Command_TeleportToSaferoom, ADMFLAG_CHEATS, "Teleport all survivors to the saferoom");
}

Action Command_TeleportToSaferoom(int client, int args)
{
    // Find the checkpoint (end saferoom door)
    int checkpoint = FindEntityByClassname(-1, "info_changelevel");
    if (checkpoint == -1)
    {
        checkpoint = FindEntityByClassname(-1, "trigger_changelevel");
    }

    if (checkpoint == -1)
    {
        ReplyToCommand(client, "[SM] Could not find saferoom location.");
        return Plugin_Handled;
    }

    float saferoomPos[3];
    GetEntPropVector(checkpoint, Prop_Send, "m_vecOrigin", saferoomPos);

    // Teleport all survivors
    int count = 0;
    for (int i = 1; i <= MaxClients; i++)
    {
        if (IsClientInGame(i) && GetClientTeam(i) == 2 && IsPlayerAlive(i))
        {
            TeleportEntity(i, saferoomPos, NULL_VECTOR, NULL_VECTOR);
            count++;
        }
    }

    PrintToChatAll("\\x04[ADMIN] \\x01Teleported %d survivors to the saferoom.", count);
    return Plugin_Handled;
}'''
    },
    # === TIMERS ===
    {
        "user": "Create a SourcePawn timer that spawns hordes every 60 seconds",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

ConVar g_cvHordeInterval;
Handle g_hHordeTimer = null;

public Plugin myinfo =
{
    name = "Periodic Horde Spawner",
    author = "Developer",
    description = "Spawns zombie hordes at regular intervals",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    g_cvHordeInterval = CreateConVar("sm_horde_interval", "60.0", "Seconds between horde spawns", FCVAR_NOTIFY, true, 10.0, true, 300.0);

    RegAdminCmd("sm_starthorde", Command_StartHorde, ADMFLAG_CHEATS, "Start periodic horde spawning");
    RegAdminCmd("sm_stophorde", Command_StopHorde, ADMFLAG_CHEATS, "Stop periodic horde spawning");

    HookEvent("round_start", Event_RoundStart);
    HookEvent("round_end", Event_RoundEnd);
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
    StartHordeTimer();
}

public void Event_RoundEnd(Event event, const char[] name, bool dontBroadcast)
{
    StopHordeTimer();
}

void StartHordeTimer()
{
    StopHordeTimer();
    float interval = g_cvHordeInterval.FloatValue;
    g_hHordeTimer = CreateTimer(interval, Timer_SpawnHorde, _, TIMER_REPEAT);
}

void StopHordeTimer()
{
    if (g_hHordeTimer != null)
    {
        KillTimer(g_hHordeTimer);
        g_hHordeTimer = null;
    }
}

Action Timer_SpawnHorde(Handle timer)
{
    SpawnHorde();
    return Plugin_Continue;
}

void SpawnHorde()
{
    // Use director to spawn a panic event
    int director = FindEntityByClassname(-1, "info_director");
    if (director != -1)
    {
        AcceptEntityInput(director, "ForcePanicEvent");
        PrintToChatAll("\\x04[HORDE] \\x01A zombie horde is approaching!");
    }
}

Action Command_StartHorde(int client, int args)
{
    StartHordeTimer();
    ReplyToCommand(client, "[SM] Periodic horde spawning started.");
    return Plugin_Handled;
}

Action Command_StopHorde(int client, int args)
{
    StopHordeTimer();
    ReplyToCommand(client, "[SM] Periodic horde spawning stopped.");
    return Plugin_Handled;
}'''
    },
    # === SDK HOOKS (DAMAGE) ===
    {
        "user": "Write a plugin that prevents friendly fire damage between survivors",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>

ConVar g_cvEnabled;

public Plugin myinfo =
{
    name = "No Friendly Fire",
    author = "Developer",
    description = "Prevents friendly fire between survivors",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    g_cvEnabled = CreateConVar("sm_noff_enabled", "1", "Enable friendly fire prevention", FCVAR_NOTIFY, true, 0.0, true, 1.0);
}

public void OnClientPutInServer(int client)
{
    SDKHook(client, SDKHook_OnTakeDamage, OnTakeDamage);
}

Action OnTakeDamage(int victim, int &attacker, int &inflictor, float &damage, int &damagetype)
{
    if (!g_cvEnabled.BoolValue)
        return Plugin_Continue;

    // Check if both victim and attacker are valid players
    if (victim < 1 || victim > MaxClients || !IsClientInGame(victim))
        return Plugin_Continue;

    if (attacker < 1 || attacker > MaxClients || !IsClientInGame(attacker))
        return Plugin_Continue;

    // Check if both are survivors (team 2)
    if (GetClientTeam(victim) == 2 && GetClientTeam(attacker) == 2)
    {
        // Block friendly fire damage
        return Plugin_Handled;
    }

    return Plugin_Continue;
}'''
    },
    {
        "user": "Create a function to modify the Tank's health based on player count",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

#define TANK_BASE_HEALTH 6000
#define HEALTH_PER_PLAYER 1000

ConVar g_cvBaseHealth;
ConVar g_cvHealthPerPlayer;

public Plugin myinfo =
{
    name = "Tank Health Scaler",
    author = "Developer",
    description = "Scales Tank health based on survivor count",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    g_cvBaseHealth = CreateConVar("sm_tank_base_health", "6000", "Base Tank health", FCVAR_NOTIFY, true, 1000.0, true, 100000.0);
    g_cvHealthPerPlayer = CreateConVar("sm_tank_health_per_player", "1000", "Additional health per survivor", FCVAR_NOTIFY, true, 0.0, true, 10000.0);

    HookEvent("tank_spawn", Event_TankSpawn);
}

int GetSurvivorCount()
{
    int count = 0;
    for (int i = 1; i <= MaxClients; i++)
    {
        if (IsClientInGame(i) && GetClientTeam(i) == 2 && IsPlayerAlive(i))
        {
            count++;
        }
    }
    return count;
}

int CalculateTankHealth()
{
    int survivors = GetSurvivorCount();
    int baseHealth = g_cvBaseHealth.IntValue;
    int perPlayer = g_cvHealthPerPlayer.IntValue;

    return baseHealth + (survivors * perPlayer);
}

void SetTankHealth(int tank, int health)
{
    SetEntProp(tank, Prop_Data, "m_iHealth", health);
    SetEntProp(tank, Prop_Data, "m_iMaxHealth", health);
}

public void Event_TankSpawn(Event event, const char[] name, bool dontBroadcast)
{
    int tank = GetClientOfUserId(event.GetInt("userid"));
    if (tank > 0 && IsClientInGame(tank))
    {
        int newHealth = CalculateTankHealth();
        SetTankHealth(tank, newHealth);

        PrintToChatAll("\\x04[TANK] \\x01Tank spawned with %d health!", newHealth);
    }
}'''
    },
    # === ENTITY DETECTION ===
    {
        "user": "Write a function to detect when all survivors are inside the saferoom",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

Handle g_hCheckTimer = null;

public Plugin myinfo =
{
    name = "Saferoom Detector",
    author = "Developer",
    description = "Detects when all survivors are in the saferoom",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("round_start", Event_RoundStart);
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
    // Start checking periodically
    if (g_hCheckTimer != null)
        KillTimer(g_hCheckTimer);

    g_hCheckTimer = CreateTimer(1.0, Timer_CheckSaferoom, _, TIMER_REPEAT);
}

bool IsPlayerInEndSaferoom(int client)
{
    if (!IsClientInGame(client) || !IsPlayerAlive(client))
        return false;

    float clientPos[3];
    GetClientAbsOrigin(client, clientPos);

    // Find checkpoint/saferoom trigger
    int entity = -1;
    while ((entity = FindEntityByClassname(entity, "trigger_changelevel")) != -1)
    {
        float mins[3], maxs[3], origin[3];
        GetEntPropVector(entity, Prop_Send, "m_vecOrigin", origin);
        GetEntPropVector(entity, Prop_Send, "m_vecMins", mins);
        GetEntPropVector(entity, Prop_Send, "m_vecMaxs", maxs);

        // Expand the check area
        for (int i = 0; i < 3; i++)
        {
            mins[i] = origin[i] + mins[i] - 200.0;
            maxs[i] = origin[i] + maxs[i] + 200.0;
        }

        if (clientPos[0] >= mins[0] && clientPos[0] <= maxs[0] &&
            clientPos[1] >= mins[1] && clientPos[1] <= maxs[1] &&
            clientPos[2] >= mins[2] && clientPos[2] <= maxs[2])
        {
            return true;
        }
    }

    return false;
}

bool AreAllSurvivorsInSaferoom()
{
    int total = 0;
    int inSaferoom = 0;

    for (int i = 1; i <= MaxClients; i++)
    {
        if (IsClientInGame(i) && GetClientTeam(i) == 2 && IsPlayerAlive(i))
        {
            total++;
            if (IsPlayerInEndSaferoom(i))
            {
                inSaferoom++;
            }
        }
    }

    return (total > 0 && total == inSaferoom);
}

Action Timer_CheckSaferoom(Handle timer)
{
    if (AreAllSurvivorsInSaferoom())
    {
        PrintToChatAll("\\x04[SAFEROOM] \\x01All survivors are in the saferoom!");
        // Could trigger map change or bonus here
    }
    return Plugin_Continue;
}'''
    },
    # === ENTITY SPAWNING ===
    {
        "user": "Write a function to spawn a Witch at a random navigation mesh location",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

public Plugin myinfo =
{
    name = "Witch Spawner",
    author = "Developer",
    description = "Spawns Witches at random locations",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    RegAdminCmd("sm_spawnwitch", Command_SpawnWitch, ADMFLAG_CHEATS, "Spawn a Witch at a random location");
}

bool GetRandomNavPosition(float pos[3])
{
    // Get a random survivor's position as a starting point
    int randomSurvivor = -1;
    for (int i = 1; i <= MaxClients; i++)
    {
        if (IsClientInGame(i) && GetClientTeam(i) == 2 && IsPlayerAlive(i))
        {
            randomSurvivor = i;
            break;
        }
    }

    if (randomSurvivor == -1)
        return false;

    float survivorPos[3];
    GetClientAbsOrigin(randomSurvivor, survivorPos);

    // Generate a random offset (500-1500 units away)
    float angle = GetRandomFloat(0.0, 360.0);
    float distance = GetRandomFloat(500.0, 1500.0);

    pos[0] = survivorPos[0] + Cosine(DegToRad(angle)) * distance;
    pos[1] = survivorPos[1] + Sine(DegToRad(angle)) * distance;
    pos[2] = survivorPos[2];

    // Trace down to find ground
    float endPos[3];
    endPos[0] = pos[0];
    endPos[1] = pos[1];
    endPos[2] = pos[2] - 500.0;

    Handle trace = TR_TraceRayFilterEx(pos, endPos, MASK_SOLID, RayType_EndPoint, TraceFilter_NoPlayers);
    if (TR_DidHit(trace))
    {
        TR_GetEndPosition(pos, trace);
        pos[2] += 10.0; // Slight offset from ground
        delete trace;
        return true;
    }

    delete trace;
    return false;
}

bool TraceFilter_NoPlayers(int entity, int contentsMask)
{
    return entity > MaxClients || entity == 0;
}

void SpawnWitchAtPosition(float pos[3])
{
    // Use z_spawn_old command through director
    int director = CreateEntityByName("info_zombie_spawn");
    if (director != -1)
    {
        DispatchKeyValue(director, "population", "witch");
        TeleportEntity(director, pos, NULL_VECTOR, NULL_VECTOR);
        DispatchSpawn(director);
        AcceptEntityInput(director, "SpawnZombie");
        AcceptEntityInput(director, "Kill"); // Clean up spawner
    }
}

Action Command_SpawnWitch(int client, int args)
{
    float spawnPos[3];
    if (GetRandomNavPosition(spawnPos))
    {
        SpawnWitchAtPosition(spawnPos);
        ReplyToCommand(client, "[SM] Witch spawned at %.0f, %.0f, %.0f", spawnPos[0], spawnPos[1], spawnPos[2]);
    }
    else
    {
        ReplyToCommand(client, "[SM] Could not find valid spawn position.");
    }
    return Plugin_Handled;
}'''
    },
    # === STATISTICS/TRACKING ===
    {
        "user": "Create a plugin that tracks player damage dealt to infected and displays a leaderboard",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>

int g_iDamageDealt[MAXPLAYERS + 1];

public Plugin myinfo =
{
    name = "Damage Leaderboard",
    author = "Developer",
    description = "Tracks damage dealt to infected",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    RegConsoleCmd("sm_damage", Command_ShowDamage, "Show damage leaderboard");

    HookEvent("round_start", Event_RoundStart);
    HookEvent("round_end", Event_RoundEnd);
}

public void OnClientPutInServer(int client)
{
    g_iDamageDealt[client] = 0;
    SDKHook(client, SDKHook_OnTakeDamageAlive, OnTakeDamageAlive);
}

public void OnClientDisconnect(int client)
{
    g_iDamageDealt[client] = 0;
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
    ResetDamage();
}

public void Event_RoundEnd(Event event, const char[] name, bool dontBroadcast)
{
    ShowLeaderboard();
}

void ResetDamage()
{
    for (int i = 1; i <= MaxClients; i++)
    {
        g_iDamageDealt[i] = 0;
    }
}

Action OnTakeDamageAlive(int victim, int &attacker, int &inflictor, float &damage, int &damagetype)
{
    // Check if attacker is a survivor damaging infected
    if (attacker > 0 && attacker <= MaxClients && IsClientInGame(attacker))
    {
        if (GetClientTeam(attacker) == 2) // Survivor
        {
            if (victim > 0 && IsClientInGame(victim) && GetClientTeam(victim) == 3) // Infected
            {
                g_iDamageDealt[attacker] += RoundToFloor(damage);
            }
        }
    }
    return Plugin_Continue;
}

void ShowLeaderboard()
{
    // Sort players by damage
    int sorted[MAXPLAYERS + 1][2];
    int count = 0;

    for (int i = 1; i <= MaxClients; i++)
    {
        if (IsClientInGame(i) && GetClientTeam(i) == 2 && g_iDamageDealt[i] > 0)
        {
            sorted[count][0] = i;
            sorted[count][1] = g_iDamageDealt[i];
            count++;
        }
    }

    // Simple bubble sort
    for (int i = 0; i < count - 1; i++)
    {
        for (int j = 0; j < count - i - 1; j++)
        {
            if (sorted[j][1] < sorted[j + 1][1])
            {
                int temp0 = sorted[j][0];
                int temp1 = sorted[j][1];
                sorted[j][0] = sorted[j + 1][0];
                sorted[j][1] = sorted[j + 1][1];
                sorted[j + 1][0] = temp0;
                sorted[j + 1][1] = temp1;
            }
        }
    }

    // Display leaderboard
    PrintToChatAll("\\x04=== DAMAGE LEADERBOARD ===");
    for (int i = 0; i < count && i < 5; i++)
    {
        char name[64];
        GetClientName(sorted[i][0], name, sizeof(name));
        PrintToChatAll("\\x01#%d: \\x03%s \\x01- %d damage", i + 1, name, sorted[i][1]);
    }
}

Action Command_ShowDamage(int client, int args)
{
    ShowLeaderboard();
    return Plugin_Handled;
}'''
    },
    {
        "user": "Create a plugin that gives bonus points for headshot kills on infected",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>

#define HEADSHOT_BONUS 50

int g_iHeadshotKills[MAXPLAYERS + 1];
int g_iBonusPoints[MAXPLAYERS + 1];

public Plugin myinfo =
{
    name = "Headshot Bonus",
    author = "Developer",
    description = "Awards bonus points for headshot kills",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("player_death", Event_PlayerDeath);
    HookEvent("infected_death", Event_InfectedDeath);
    HookEvent("round_start", Event_RoundStart);

    RegConsoleCmd("sm_headshots", Command_ShowHeadshots, "Show your headshot stats");
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
    for (int i = 1; i <= MaxClients; i++)
    {
        g_iHeadshotKills[i] = 0;
        g_iBonusPoints[i] = 0;
    }
}

public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast)
{
    int attacker = GetClientOfUserId(event.GetInt("attacker"));
    bool headshot = event.GetBool("headshot");
    int victim = GetClientOfUserId(event.GetInt("userid"));

    if (attacker > 0 && attacker <= MaxClients && IsClientInGame(attacker))
    {
        if (GetClientTeam(attacker) == 2) // Survivor
        {
            if (victim > 0 && IsClientInGame(victim) && GetClientTeam(victim) == 3) // Infected
            {
                if (headshot)
                {
                    AwardHeadshotBonus(attacker);
                }
            }
        }
    }
}

public void Event_InfectedDeath(Event event, const char[] name, bool dontBroadcast)
{
    int attacker = GetClientOfUserId(event.GetInt("attacker"));
    bool headshot = event.GetBool("headshot");

    if (attacker > 0 && attacker <= MaxClients && IsClientInGame(attacker))
    {
        if (GetClientTeam(attacker) == 2 && headshot)
        {
            AwardHeadshotBonus(attacker);
        }
    }
}

void AwardHeadshotBonus(int client)
{
    g_iHeadshotKills[client]++;
    g_iBonusPoints[client] += HEADSHOT_BONUS;

    PrintToChat(client, "\\x04[HEADSHOT] \\x01+%d bonus points! Total: %d", HEADSHOT_BONUS, g_iBonusPoints[client]);

    // Play a subtle sound
    EmitSoundToClient(client, "buttons/bell1.wav", _, _, SNDLEVEL_NORMAL, _, 0.5);
}

Action Command_ShowHeadshots(int client, int args)
{
    PrintToChat(client, "\\x04[STATS] \\x01Headshot kills: %d | Bonus points: %d",
        g_iHeadshotKills[client], g_iBonusPoints[client]);
    return Plugin_Handled;
}'''
    },
    # === WEAPON HOOKS ===
    {
        "user": "Write a SourcePawn hook for when a survivor picks up a weapon",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>

public Plugin myinfo =
{
    name = "Weapon Pickup Hook",
    author = "Developer",
    description = "Hooks weapon pickup events",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("item_pickup", Event_ItemPickup);
}

public void OnClientPutInServer(int client)
{
    SDKHook(client, SDKHook_WeaponEquip, OnWeaponEquip);
}

Action OnWeaponEquip(int client, int weapon)
{
    if (!IsValidEntity(weapon))
        return Plugin_Continue;

    if (GetClientTeam(client) != 2) // Not a survivor
        return Plugin_Continue;

    char classname[64];
    GetEntityClassname(weapon, classname, sizeof(classname));

    char weaponName[64];
    GetWeaponDisplayName(classname, weaponName, sizeof(weaponName));

    PrintToChat(client, "\\x04[PICKUP] \\x01You picked up: %s", weaponName);

    // Log the pickup
    char clientName[64];
    GetClientName(client, clientName, sizeof(clientName));
    LogMessage("%s picked up %s", clientName, classname);

    return Plugin_Continue;
}

public void Event_ItemPickup(Event event, const char[] name, bool dontBroadcast)
{
    int client = GetClientOfUserId(event.GetInt("userid"));
    char item[64];
    event.GetString("item", item, sizeof(item));

    if (client > 0 && IsClientInGame(client))
    {
        // Handle special items (pills, medkits, etc.)
        if (StrContains(item, "pain_pills") != -1)
        {
            PrintToChat(client, "\\x04[ITEM] \\x01You grabbed some pills!");
        }
        else if (StrContains(item, "first_aid_kit") != -1)
        {
            PrintToChat(client, "\\x04[ITEM] \\x01You grabbed a medkit!");
        }
    }
}

void GetWeaponDisplayName(const char[] classname, char[] buffer, int maxlen)
{
    if (StrContains(classname, "smg") != -1)
        strcopy(buffer, maxlen, "SMG");
    else if (StrContains(classname, "rifle") != -1)
        strcopy(buffer, maxlen, "Assault Rifle");
    else if (StrContains(classname, "shotgun") != -1)
        strcopy(buffer, maxlen, "Shotgun");
    else if (StrContains(classname, "sniper") != -1)
        strcopy(buffer, maxlen, "Sniper Rifle");
    else if (StrContains(classname, "pistol") != -1)
        strcopy(buffer, maxlen, "Pistol");
    else if (StrContains(classname, "melee") != -1)
        strcopy(buffer, maxlen, "Melee Weapon");
    else
        strcopy(buffer, maxlen, classname);
}'''
    },
    # === PLAYER INVENTORY ===
    {
        "user": "Create a SourcePawn function to save and restore player inventory between maps",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

#define MAX_WEAPONS 5

enum struct PlayerInventory
{
    char primaryWeapon[64];
    char secondaryWeapon[64];
    char throwable[64];
    char kit[64];
    char pills[64];
    int health;
    bool isValid;
}

PlayerInventory g_SavedInventory[MAXPLAYERS + 1];

public Plugin myinfo =
{
    name = "Inventory Saver",
    author = "Developer",
    description = "Saves and restores player inventory between maps",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("map_transition", Event_MapTransition);
    HookEvent("player_spawn", Event_PlayerSpawn);
}

void SavePlayerInventory(int client)
{
    if (!IsClientInGame(client) || GetClientTeam(client) != 2)
        return;

    PlayerInventory inv;
    inv.isValid = true;
    inv.health = GetClientHealth(client);

    // Get weapon slots
    int weapon;
    char classname[64];

    // Slot 0 - Primary weapon
    weapon = GetPlayerWeaponSlot(client, 0);
    if (weapon != -1)
    {
        GetEntityClassname(weapon, classname, sizeof(classname));
        strcopy(inv.primaryWeapon, sizeof(inv.primaryWeapon), classname);
    }

    // Slot 1 - Secondary weapon (pistol/melee)
    weapon = GetPlayerWeaponSlot(client, 1);
    if (weapon != -1)
    {
        GetEntityClassname(weapon, classname, sizeof(classname));
        strcopy(inv.secondaryWeapon, sizeof(inv.secondaryWeapon), classname);
    }

    // Slot 2 - Throwable
    weapon = GetPlayerWeaponSlot(client, 2);
    if (weapon != -1)
    {
        GetEntityClassname(weapon, classname, sizeof(classname));
        strcopy(inv.throwable, sizeof(inv.throwable), classname);
    }

    // Slot 3 - First aid kit
    weapon = GetPlayerWeaponSlot(client, 3);
    if (weapon != -1)
    {
        GetEntityClassname(weapon, classname, sizeof(classname));
        strcopy(inv.kit, sizeof(inv.kit), classname);
    }

    // Slot 4 - Pills/adrenaline
    weapon = GetPlayerWeaponSlot(client, 4);
    if (weapon != -1)
    {
        GetEntityClassname(weapon, classname, sizeof(classname));
        strcopy(inv.pills, sizeof(inv.pills), classname);
    }

    g_SavedInventory[client] = inv;

    char name[64];
    GetClientName(client, name, sizeof(name));
    LogMessage("Saved inventory for %s", name);
}

void RestorePlayerInventory(int client)
{
    if (!g_SavedInventory[client].isValid)
        return;

    PlayerInventory inv = g_SavedInventory[client];

    // Strip current weapons first
    for (int slot = 0; slot < 5; slot++)
    {
        int weapon = GetPlayerWeaponSlot(client, slot);
        if (weapon != -1)
        {
            RemovePlayerItem(client, weapon);
            AcceptEntityInput(weapon, "Kill");
        }
    }

    // Give saved weapons
    if (strlen(inv.primaryWeapon) > 0)
        GivePlayerItem(client, inv.primaryWeapon);

    if (strlen(inv.secondaryWeapon) > 0)
        GivePlayerItem(client, inv.secondaryWeapon);

    if (strlen(inv.throwable) > 0)
        GivePlayerItem(client, inv.throwable);

    if (strlen(inv.kit) > 0)
        GivePlayerItem(client, inv.kit);

    if (strlen(inv.pills) > 0)
        GivePlayerItem(client, inv.pills);

    // Restore health
    SetEntityHealth(client, inv.health);

    PrintToChat(client, "\\x04[INVENTORY] \\x01Your inventory has been restored!");

    // Clear saved data
    g_SavedInventory[client].isValid = false;
}

public void Event_MapTransition(Event event, const char[] name, bool dontBroadcast)
{
    for (int i = 1; i <= MaxClients; i++)
    {
        if (IsClientInGame(i) && GetClientTeam(i) == 2)
        {
            SavePlayerInventory(i);
        }
    }
}

public void Event_PlayerSpawn(Event event, const char[] name, bool dontBroadcast)
{
    int client = GetClientOfUserId(event.GetInt("userid"));
    if (client > 0)
    {
        CreateTimer(1.0, Timer_RestoreInventory, GetClientUserId(client));
    }
}

Action Timer_RestoreInventory(Handle timer, int userid)
{
    int client = GetClientOfUserId(userid);
    if (client > 0 && IsClientInGame(client))
    {
        RestorePlayerInventory(client);
    }
    return Plugin_Continue;
}'''
    },
    # === SPECIAL INFECTED ANNOUNCEMENTS ===
    {
        "user": "Write a plugin that announces special infected spawns with their distance from survivors",
        "assistant": '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

public Plugin myinfo =
{
    name = "Special Infected Announcer",
    author = "Developer",
    description = "Announces special infected spawns with distance",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("player_spawn", Event_PlayerSpawn);
}

public void Event_PlayerSpawn(Event event, const char[] name, bool dontBroadcast)
{
    int client = GetClientOfUserId(event.GetInt("userid"));

    if (client < 1 || !IsClientInGame(client))
        return;

    // Check if it's a special infected (team 3, not common)
    if (GetClientTeam(client) != 3)
        return;

    // Get the infected class
    int zombieClass = GetEntProp(client, Prop_Send, "m_zombieClass");

    char className[32];
    GetZombieClassName(zombieClass, className, sizeof(className));

    if (strlen(className) == 0)
        return; // Common infected, skip

    // Find nearest survivor and calculate distance
    float infectedPos[3];
    GetClientAbsOrigin(client, infectedPos);

    int nearestSurvivor = -1;
    float nearestDistance = 99999.0;

    for (int i = 1; i <= MaxClients; i++)
    {
        if (IsClientInGame(i) && GetClientTeam(i) == 2 && IsPlayerAlive(i))
        {
            float survivorPos[3];
            GetClientAbsOrigin(i, survivorPos);

            float distance = GetVectorDistance(infectedPos, survivorPos);
            if (distance < nearestDistance)
            {
                nearestDistance = distance;
                nearestSurvivor = i;
            }
        }
    }

    // Announce to all survivors
    char distanceStr[32];
    if (nearestDistance < 500.0)
        strcopy(distanceStr, sizeof(distanceStr), "VERY CLOSE");
    else if (nearestDistance < 1000.0)
        strcopy(distanceStr, sizeof(distanceStr), "NEARBY");
    else if (nearestDistance < 2000.0)
        strcopy(distanceStr, sizeof(distanceStr), "MEDIUM RANGE");
    else
        strcopy(distanceStr, sizeof(distanceStr), "FAR AWAY");

    PrintToChatAll("\\x04[SI SPAWN] \\x03%s \\x01spawned! Distance: \\x04%s \\x01(%.0f units)",
        className, distanceStr, nearestDistance);
}

void GetZombieClassName(int zombieClass, char[] buffer, int maxlen)
{
    switch (zombieClass)
    {
        case 1: strcopy(buffer, maxlen, "Smoker");
        case 2: strcopy(buffer, maxlen, "Boomer");
        case 3: strcopy(buffer, maxlen, "Hunter");
        case 4: strcopy(buffer, maxlen, "Spitter");
        case 5: strcopy(buffer, maxlen, "Jockey");
        case 6: strcopy(buffer, maxlen, "Charger");
        case 8: strcopy(buffer, maxlen, "Tank");
        default: buffer[0] = '\\0'; // Unknown or common
    }
}'''
    },
    # === VSCRIPT EXAMPLE ===
    {
        "user": "Write a VScript function to create a custom panic event in L4D2",
        "assistant": '''// VScript function to create a custom panic event
// Place this in scripts/vscripts/custom_panic.nut

/**
 * Triggers a custom panic event with configurable parameters
 * @param intensity - Horde intensity (1-10)
 * @param duration - How long the panic lasts in seconds
 */
function TriggerCustomPanicEvent(intensity = 5, duration = 30.0)
{
    // Get the director
    local director = Entities.FindByClassname(null, "info_director");

    if (director == null)
    {
        printl("[PANIC] Error: Could not find info_director");
        return false;
    }

    // Scale mob count based on intensity
    local mobCount = intensity * 10;

    // Trigger the panic event
    EntFire("info_director", "ForcePanicEvent", "", 0.0);

    // Play warning sound to all players
    EmitSoundOnClient("UI/Pickup_Secret01.wav", null);

    // Print warning to all players
    ClientPrint(null, 3, "[PANIC] A massive horde is approaching!");

    // Schedule end of panic
    local endPanicFunc = function() {
        EntFire("info_director", "ReleaseSurvivorPositions", "", 0.0);
        ClientPrint(null, 3, "[PANIC] The horde has subsided.");
    };

    DoEntFire("!self", "CallScriptFunction", "EndPanic", duration, null, null);

    return true;
}

/**
 * Creates a wandering witch at the specified position
 */
function SpawnWanderingWitch(position)
{
    local witch = SpawnEntityFromTable("witch", {
        origin = position,
        angles = Vector(0, RandomFloat(0, 360), 0)
    });

    if (witch != null)
    {
        // Make her wander
        DoEntFire("!self", "StartWandering", "", 0.1, null, witch);
        printl("[SPAWN] Wandering witch spawned at " + position);
        return witch;
    }

    return null;
}

/**
 * Spawns a mini-tank event (multiple tanks at once)
 * @param count - Number of tanks to spawn
 */
function SpawnMiniTankEvent(count = 2)
{
    local survivors = [];
    local ent = null;

    // Find all survivors
    while ((ent = Entities.FindByClassname(ent, "player")) != null)
    {
        if (ent.GetTeam() == 2 && ent.IsAlive())
        {
            survivors.append(ent);
        }
    }

    if (survivors.len() == 0)
    {
        printl("[TANK] No survivors found");
        return;
    }

    // Spawn tanks near random survivors
    for (local i = 0; i < count; i++)
    {
        local survivor = survivors[RandomInt(0, survivors.len() - 1)];
        local survPos = survivor.GetOrigin();

        // Offset spawn position
        local angle = RandomFloat(0, 360);
        local distance = RandomFloat(800, 1200);
        local spawnPos = survPos + Vector(
            cos(angle * 0.0174533) * distance,
            sin(angle * 0.0174533) * distance,
            0
        );

        // Use director to spawn
        EntFire("info_director", "SpawnTank", "", i * 2.0);
    }

    ClientPrint(null, 3, "[ALERT] Multiple Tanks incoming!");
}

// Register these functions to be callable
::TriggerCustomPanicEvent <- TriggerCustomPanicEvent;
::SpawnWanderingWitch <- SpawnWanderingWitch;
::SpawnMiniTankEvent <- SpawnMiniTankEvent;'''
    },
]


def create_example(user: str, assistant: str) -> dict:
    """Create a properly formatted training example."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant}
        ]
    }


def main():
    project_root = Path(__file__).parent.parent.parent
    output_path = project_root / "data" / "processed" / "synthetic_examples.jsonl"

    print("=" * 60)
    print("Generating Synthetic Training Examples")
    print("=" * 60)
    print()

    examples = []
    for item in SYNTHETIC_EXAMPLES:
        example = create_example(item["user"], item["assistant"])
        examples.append(example)

    print(f"Generated {len(examples)} synthetic examples")

    # Save to file
    with open(output_path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

    print(f"Saved to: {output_path}")

    # Show sample
    print("\nSample prompts covered:")
    for i, item in enumerate(SYNTHETIC_EXAMPLES[:5], 1):
        print(f"  {i}. {item['user'][:60]}...")

    return examples


if __name__ == "__main__":
    main()
