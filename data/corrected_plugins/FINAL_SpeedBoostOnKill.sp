#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

#define PLUGIN_VERSION "1.2"
#define BOOST_MULTIPLIER 1.4  // 40% faster (NOT absolute speed!)
#define BOOST_DURATION 5.0

public Plugin myinfo =
{
    name = "L4D2 Speed Boost on Kill",
    author = "Optimized Version",
    description = "Gives survivors speed boost on infected kill",
    version = PLUGIN_VERSION,
    url = ""
};

// Track if player has active boost (prevents stacking issues)
bool g_bHasBoost[MAXPLAYERS + 1];

public void OnPluginStart()
{
    HookEvent("player_death", Event_PlayerDeath);
    HookEvent("infected_death", Event_InfectedDeath);
    HookEvent("round_start", Event_RoundStart);
}

public void OnClientDisconnect(int client)
{
    g_bHasBoost[client] = false;
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
    // Clear all boosts on round start
    for (int i = 1; i <= MaxClients; i++)
    {
        if (IsClientInGame(i) && IsPlayerAlive(i))
        {
            SetEntPropFloat(i, Prop_Send, "m_flLaggedMovementValue", 1.0);
        }
        g_bHasBoost[i] = false;
    }
}

public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast)
{
    int victim = GetClientOfUserId(event.GetInt("userid"));
    int attacker = GetClientOfUserId(event.GetInt("attacker"));
    
    // Validate victim is a special infected (Team 3)
    if (!IsValidInfected(victim))
        return;
    
    // Validate attacker is a survivor
    if (!IsValidSurvivor(attacker))
        return;
    
    // Give speed boost
    GiveSpeedBoost(attacker);
}

public void Event_InfectedDeath(Event event, const char[] name, bool dontBroadcast)
{
    int attacker = GetClientOfUserId(event.GetInt("attacker"));
    
    // Also give boost for common infected kills
    if (IsValidSurvivor(attacker))
    {
        GiveSpeedBoost(attacker);
    }
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
    
    // Schedule reset
    CreateTimer(BOOST_DURATION, Timer_ResetSpeed, GetClientUserId(client));
}

public Action Timer_ResetSpeed(Handle timer, int userid)
{
    int client = GetClientOfUserId(userid);
    
    if (client > 0 && IsClientInGame(client) && g_bHasBoost[client])
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
