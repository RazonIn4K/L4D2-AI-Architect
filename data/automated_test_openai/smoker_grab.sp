#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>
#include <left4dhooks>

public Plugin myinfo =
{
    name = "Smoker Grab Announcer",
    author = "ChatGPT",
    description = "Announces when a Smoker grabs a player",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    // Hook the tongue grab event
    HookEvent("tongue_grab", Event_TongueGrab);
    HookEvent("tongue_release", Event_TongueRelease);
}

public void OnMapStart()
{
    // Ensure all players are updated in case they were grabbed before
    for (int i = 1; i <= MaxClients; i++)
    {
        if (IsClientInGame(i) && GetClientTeam(i) == 2)
        {
            UpdatePlayerState(i);
        }
    }
}

public void Event_TongueGrab(Event event, const char[] name, bool dontBroadcast)
{
    int smoker = GetClientOfUserId(event.GetInt("userid"));
    int victim = GetClientOfUserId(event.GetInt("victim"));

    if (smoker > 0 && victim > 0 && IsPlayerAlive(smoker) && IsPlayerAlive(victim))
    {
        char smokerName[64];
        char victimName[64];
        GetClientName(smoker, smokerName, sizeof(smokerName));
        GetClientName(victim, victimName, sizeof(victimName));

        PrintToChatAll("\x04[Smoker] \x01%s \x1egrabbed\x01 %s!", smokerName, victimName);
    }
}

public void Event_TongueRelease(Event event, const char[] name, bool dontBroadcast)
{
    int smoker = GetClientOfUserId(event.GetInt("userid"));

    if (smoker > 0 && IsPlayerAlive(smoker))
    {
        char smokerName[64];
        GetClientName(smoker, smokerName, sizeof(smokerName));

        PrintToChatAll("\x04[Smoker] \x01%s\x01 released!", smokerName);
    }
}

// Utility functions
bool IsPlayerAlive(int client)
{
    return (client > 0 && client <= MaxClients && IsClientInGame(client) && GetClientTeam(client) == 2 && GetEntProp(client, Prop_Send, "m_lifeState") == LIFESTATE_ALIVE);
}

void UpdatePlayerState(int client)
{
    // Custom logic to update player state if needed
}
