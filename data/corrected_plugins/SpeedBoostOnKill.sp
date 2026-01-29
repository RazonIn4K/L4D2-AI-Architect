#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

#define BOOST_SPEED 1.4  // 40% speed boost
#define BOOST_DURATION 5.0

public Plugin myinfo =
{
    name = "Speed Boost On Kill",
    author = "Developer",
    description = "Gives survivors a temporary speed boost when killing infected",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("player_death", Event_PlayerDeath);
    HookEvent("infected_death", Event_InfectedDeath);
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
    
    // Apply speed boost (m_flLaggedMovementValue is a multiplier, 1.0 = normal speed)
    SetEntPropFloat(client, Prop_Send, "m_flLaggedMovementValue", BOOST_SPEED);
    
    PrintToChat(client, "\x04[Speed Boost] \x01You received a speed boost!");
    
    // Create timer to reset speed
    CreateTimer(BOOST_DURATION, Timer_ResetSpeed, GetClientUserId(client));
}

public Action Timer_ResetSpeed(Handle timer, int userid)
{
    int client = GetClientOfUserId(userid);
    if (client > 0 && IsClientInGame(client) && IsPlayerAlive(client))
    {
        SetEntPropFloat(client, Prop_Send, "m_flLaggedMovementValue", 1.0);
        PrintToChat(client, "\x04[Speed Boost] \x01Speed boost expired!");
    }
    return Plugin_Continue;
}
