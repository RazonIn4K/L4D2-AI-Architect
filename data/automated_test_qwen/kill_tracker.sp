#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

public Plugin myinfo =
{
    name = "Zombie Kill Tracker",
    author = "Developer",
    description = "Tracks and displays zombie kill counts per player",
    version = "1.0",
    url = ""
};

int g_iKillCount[MAXPLAYERS + 1];

public void OnPluginStart()
{
    HookEvent("infected_death", Event_IncKillCount);
    HookEvent("round_start", Event_RoundStart);
}

public void OnClientDisconnect(int client)
{
    g_iKillCount[client] = 0;
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
    // Clear all kill counts on round start
    for (int i = 1; i <= MaxClients; i++)
    {
        g_iKillCount[i] = 0;
    }
}

public void Event_IncKillCount(Event event, const char[] name, bool dontBroadcast)
{
    int victim = GetClientOfUserId(event.GetInt("userid"));
    
    if (victim > 0 && IsClientInGame(victim) && GetClientTeam(victim) == 3)
    {
        g_iKillCount[victim]++;
        
        PrintToChatAll("%N killed a zombie! (%d/%d)", victim, g_iKillCount[victim], GetTotalZombies());
    }
}

// Helper to get total number of zombies spawned (visible + invisible)
int GetTotalZombies()
{
    return FindEntityByClassname(-1, "zombie") - 1 +
           FindEntityByClassname(-1, "hidden_zombie") - 1;
}