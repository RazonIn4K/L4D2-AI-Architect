#pragma semicolon 1
#pragma newdecls required

#include <sdktools>
#include <sdkhooks>

#define SPEED_BOOST 1.30  // 30% increase
#define BOOST_DURATION 5.0

public Plugin myinfo =
{
    name = "Speed Boost on Kills",
    author = "Developer",
    description = "Gives survivors a 30% speed boost for 5 secs on SI kill",
    version = "1.0",
    url = ""
};

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

void ApplySpeedBoost(int client)
{
    if (!g_bHasBoost[client])
    {
        g_bHasBoost[client] = true;
        SetEntPropFloat(client, Prop_Send, "m_flLaggedMovementValue", SPEED_BOOST);
        
        PrintToChat(client, "\x04[Speed Boost]\x01 You received a speed boost!");
        
        CreateTimer(BOOST_DURATION, Timer_ResetSpeed, GetClientUserId(client));
    }
}

public Action Timer_ResetSpeed(Handle timer, int userid)
{
    int client = GetClientOfUserId(userid);
    
    if (client > 0 && client <= MaxClients && IsClientInGame(client))
    {
        g_bHasBoost[client] = false;
        SetEntPropFloat(client, Prop_Send, "m_flLaggedMovementValue", 1.0);
        PrintToChat(client, "\x04[Speed Boost]\x01 Speed boost has worn off.");
    }
    
    return Plugin_Continue;
}