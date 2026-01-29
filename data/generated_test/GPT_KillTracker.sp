#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>
#include <l4d2_infected>

#define PLUGIN_VERSION "1.0"

public Plugin myinfo =
{
	name = "Zombie Kill Tracker",
	author = "ChatGPT",
	description = "Tracks the number of zombies killed by each survivor",
	version = PLUGIN_VERSION,
	url = ""
};

#define MAX_PLAYERS 64

int g_iZombiesKilled[MAX_PLAYERS + 1];

public void OnPluginStart()
{
    HookEvent("infected_death", Event_InfectedDeath);
    HookEvent("player_death", Event_PlayerDeath);
}

public void OnMapStart()
{
    for (int i = 1; i <= MaxClients; i++)
    {
        g_iZombiesKilled[i] = 0;
    }
}

public void Event_InfectedDeath(Event event, const char[] name, bool dontBroadcast)
{
    // Check if the infected was killed by a survivor
    int attacker = event.GetInt("attacker");
    int victim = event.GetInt("userid");

    if (attacker > 0 && attacker <= MaxClients && GetClientTeam(attacker) == 2)
    {
        // Infected was killed by a survivor
        if (victim > 0 && IsClientInGame(victim))
        {
            int victimIndex = GetClientOfUserId(victim);
            if (victimIndex > 0 && GetClientTeam(victimIndex) == 3)
            {
                // Survivor killed an infected
                g_iZombiesKilled[attacker]++;
                PrintToChat(attacker, "\x04[ZK] \x01You have killed %d zombies!", g_iZombiesKilled[attacker]);
            }
        }
    }
}

public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast)
{
    // Reset zombie kill count on survivor death
    int victim = event.GetInt("userid");
    int victimIndex = GetClientOfUserId(victim);

    if (victimIndex > 0 && IsClientInGame(victimIndex) && GetClientTeam(victimIndex) == 3)
    {
        // Survivor died
        g_iZombiesKilled[victimIndex] = 0;
    }
}

public int GetZombiesKilled(int client)
{
    if (client < 1 || client > MaxClients)
        return 0;

    return g_iZombiesKilled[client];
}