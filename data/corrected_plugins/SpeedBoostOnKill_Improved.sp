#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

#define BOOST_SPEED 1.4  // 40% speed boost
#define BOOST_DURATION 5.0

bool g_bHasSpeedBoost[MAXPLAYERS + 1];
Handle g_hSpeedBoostTimer[MAXPLAYERS + 1] = {null, ...};

public Plugin myinfo =
{
    name = "Speed Boost On Kill",
    author = "Developer",
    description = "Gives survivors a temporary speed boost when killing infected",
    version = "1.1",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("player_death", Event_PlayerDeath);
    HookEvent("infected_death", Event_InfectedDeath);
    HookEvent("round_start", Event_RoundStart);
    HookEvent("player_disconnect", Event_PlayerDisconnect);
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
    // Clear all speed boosts on round start
    for (int i = 1; i <= MaxClients; i++)
    {
        ClearSpeedBoost(i, true);
    }
}

public void Event_PlayerDisconnect(Event event, const char[] name, bool dontBroadcast)
{
    int client = GetClientOfUserId(event.GetInt("userid"));
    if (client > 0)
    {
        ClearSpeedBoost(client, true);
    }
}

public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast)
{
    int attacker = GetClientOfUserId(event.GetInt("attacker"));
    int victim = GetClientOfUserId(event.GetInt("userid"));
    
    if (attacker > 0 && attacker <= MaxClients && IsClientInGame(attacker) && GetClientTeam(attacker) == 2)
    {
        if (victim > 0 && victim <= MaxClients && IsClientInGame(victim) && GetClientTeam(victim) == 3)
        {
            ApplySpeedBoost(attacker);
        }
    }
}

public void Event_InfectedDeath(Event event, const char[] name, bool dontBroadcast)
{
    int attacker = GetClientOfUserId(event.GetInt("attacker"));
    
    if (attacker > 0 && attacker <= MaxClients && IsClientInGame(attacker) && GetClientTeam(attacker) == 2)
    {
        ApplySpeedBoost(attacker);
    }
}

void ApplySpeedBoost(int client)
{
    if (!IsPlayerAlive(client))
        return;
    
    // Clear existing boost if any
    if (g_bHasSpeedBoost[client])
    {
        ClearSpeedBoost(client, false);
    }
    
    // Apply new speed boost
    SetEntPropFloat(client, Prop_Send, "m_flLaggedMovementValue", BOOST_SPEED);
    g_bHasSpeedBoost[client] = true;
    
    PrintToChat(client, "\x04[Speed Boost] \x01You received a speed boost!");
    
    // Create timer to reset speed
    g_hSpeedBoostTimer[client] = CreateTimer(BOOST_DURATION, Timer_ResetSpeed, GetClientUserId(client));
}

public Action Timer_ResetSpeed(Handle timer, int userid)
{
    int client = GetClientOfUserId(userid);
    if (client > 0 && client <= MaxClients)
    {
        g_hSpeedBoostTimer[client] = null;
        ClearSpeedBoost(client, false);
    }
    return Plugin_Continue;
}

void ClearSpeedBoost(int client, bool silent)
{
    if (!g_bHasSpeedBoost[client])
        return;
    
    // Reset speed to normal if client is in game and alive
    if (IsClientInGame(client) && IsPlayerAlive(client))
    {
        SetEntPropFloat(client, Prop_Send, "m_flLaggedMovementValue", 1.0);
        if (!silent)
        {
            PrintToChat(client, "\x04[Speed Boost] \x01Speed boost expired!");
        }
    }
    
    g_bHasSpeedBoost[client] = false;
    
    // Kill timer if it exists
    if (g_hSpeedBoostTimer[client] != null)
    {
        KillTimer(g_hSpeedBoostTimer[client]);
        g_hSpeedBoostTimer[client] = null;
    }
}
