/**
 * =============================================================================
 * L4D2 Speed Boost on Kill - CORRECTED REFERENCE IMPLEMENTATION
 * =============================================================================
 * Demonstrates proper usage of:
 * - m_flLaggedMovementValue (speed multiplier, NOT m_flSpeed)
 * - player_death event for special infected
 * - Timer with GetClientUserId (NOT raw client index)
 */

#include <sourcemod>
#include <sdktools>

#pragma semicolon 1
#pragma newdecls required

#define PLUGIN_VERSION "1.0"
#define BOOST_MULTIPLIER 1.4  // 40% faster (NOT absolute speed!)
#define BOOST_DURATION 5.0

public Plugin myinfo =
{
    name = "L4D2 Speed Boost on Kill",
    author = "Corrected Example",
    description = "Gives survivors speed boost on special infected kill",
    version = PLUGIN_VERSION,
    url = ""
};

// Track if player has active boost (prevents stacking issues)
bool g_bHasBoost[MAXPLAYERS + 1];

public void OnPluginStart()
{
    HookEvent("player_death", Event_PlayerDeath);
}

public void OnClientDisconnect(int client)
{
    g_bHasBoost[client] = false;
}

public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast)
{
    int victim = GetClientOfUserId(event.GetInt("userid"));
    int attacker = GetClientOfUserId(event.GetInt("attacker"));

    // Validate victim is a special infected (Team 3)
    // Note: player_death fires for special infected, NOT common zombies
    if (!IsValidInfected(victim))
        return;

    // Validate attacker is a survivor
    if (!IsValidSurvivor(attacker))
        return;

    // Give speed boost
    GiveSpeedBoost(attacker);
}

void GiveSpeedBoost(int client)
{
    // Don't stack boosts
    if (g_bHasBoost[client])
        return;

    g_bHasBoost[client] = true;

    // CORRECT: m_flLaggedMovementValue is a MULTIPLIER
    // 1.0 = normal speed, 1.4 = 40% faster, 0.5 = 50% slower
    SetEntPropFloat(client, Prop_Send, "m_flLaggedMovementValue", BOOST_MULTIPLIER);

    // Notify player
    PrintToChat(client, "\x04[Speed Boost]\x01 You feel a surge of adrenaline!");

    // Schedule reset - CRITICAL: Use GetClientUserId, NOT raw client index!
    CreateTimer(BOOST_DURATION, Timer_ResetSpeed, GetClientUserId(client));
}

public Action Timer_ResetSpeed(Handle timer, int userid)
{
    int client = GetClientOfUserId(userid);

    if (client > 0 && IsClientInGame(client))
    {
        // Reset to normal speed (1.0 multiplier)
        SetEntPropFloat(client, Prop_Send, "m_flLaggedMovementValue", 1.0);
        g_bHasBoost[client] = false;
        PrintToChat(client, "\x04[Speed Boost]\x01 Speed boost has worn off.");
    }

    return Plugin_Continue;
}

bool IsValidSurvivor(int client)
{
    return (client > 0 && client <= MaxClients &&
            IsClientInGame(client) && GetClientTeam(client) == 2 &&
            IsPlayerAlive(client));
}

bool IsValidInfected(int client)
{
    return (client > 0 && client <= MaxClients &&
            IsClientInGame(client) && GetClientTeam(client) == 3);
}
