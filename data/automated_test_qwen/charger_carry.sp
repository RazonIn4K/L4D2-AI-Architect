#pragma semicolon 1
#pragma newdecls required

#include <sdktools>
#include <left4dhooks>

#define PLUGIN_VERSION "1.0"

public Plugin myinfo =
{
    name = "Charger Carry Tracker",
    author = "Developer",
    description = "Tracks and displays Charger carry status in chat",
    version = PLUGIN_VERSION,
    url = ""
};

bool g_bIsBeingCarried[MAXPLAYERS + 1];

public void OnPluginStart()
{
    HookEvent("charger_carry_start", Event_CarryStart);
    HookEvent("charger_carry_end", Event_CarryEnd);
}

public void OnClientDisconnect(int client)
{
    g_bIsBeingCarried[client] = false;
}

public void Event_CarryStart(Event event, const char[] name, bool dontBroadcast)
{
    int attacker = GetClientOfUserId(event.GetInt("attacker"));
    int victim = GetClientOfUserId(event.GetInt("userid"));

    if (attacker > 0 && attacker <= MaxClients && IsClientInGame(attacker) && GetClientTeam(attacker) == 3)
    {
        if (victim > 0 && victim <= MaxClients && IsClientInGame(victim) && GetClientTeam(victim) == 2)
        {
            g_bIsBeingCarried[victim] = true;
            PrintToChatAll("%N is now being carried by %N!", victim, attacker);
        }
    }
}

public void Event_CarryEnd(Event event, const char[] name, bool dontBroadcast)
{
    int victim = GetClientOfUserId(event.GetInt("userid"));

    if (victim > 0 && victim <= MaxClients && IsClientInGame(victim))
    {
        g_bIsBeingCarried[victim] = false;
        PrintToChatAll("%N is no longer being carried.", victim);
    }
}

// Alternative: Check carry state via left4dhooks
public Action OnPlayerRunCmd(int client, int &buttons, int &impulse, float vel[3], float angles[3], int &weapon, int mouse[2])
{
    if (!g_bLeft4DHooksAvailable)
        return Plugin_Continue;

    if (GetEntProp(client, Prop_Send, "m_isGhost") != 1)
        return Plugin_Continue;

    int carrier = GetInfectedGhostOwner(client);
    if (carrier > 0 && carrier <= MaxClients && IsClientInGame(carrier) && GetClientTeam(carrier) == 3)
    {
        if (!g_bIsBeingCarried[client])
        {
            g_bIsBeingCarried[client] = true;
            PrintToChatAll("%N is now being carried by %N!", client, carrier);
        }
    }
    else
    {
        g_bIsBeingCarried[client] = false;
    }

    return Plugin_Continue;
}