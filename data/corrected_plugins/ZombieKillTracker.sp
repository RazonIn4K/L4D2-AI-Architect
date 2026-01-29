#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

int g_iZombieKills[MAXPLAYERS + 1];

public Plugin myinfo =
{
    name = "Zombie Kill Tracker",
    author = "Developer",
    description = "Tracks and displays zombie kill counts for each player",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("player_death", Event_PlayerDeath);
    HookEvent("infected_death", Event_InfectedDeath);
    HookEvent("round_start", Event_RoundStart);
    HookEvent("round_end", Event_RoundEnd);
    
    RegConsoleCmd("sm_kills", Cmd_ShowKills);
    RegConsoleCmd("sm_zombiekills", Cmd_ShowKills);
}

public void OnClientDisconnect(int client)
{
    g_iZombieKills[client] = 0;
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
    for (int i = 0; i <= MaxClients; i++)
    {
        g_iZombieKills[i] = 0;
    }
}

public void Event_RoundEnd(Event event, const char[] name, bool dontBroadcast)
{
    ShowKillStats();
}

public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast)
{
    int attacker = GetClientOfUserId(event.GetInt("attacker"));
    int victim = GetClientOfUserId(event.GetInt("userid"));
    
    if (attacker > 0 && attacker <= MaxClients && IsClientInGame(attacker) && GetClientTeam(attacker) == 2)
    {
        if (victim > 0 && victim <= MaxClients && IsClientInGame(victim) && GetClientTeam(victim) == 3)
        {
            g_iZombieKills[attacker]++;
        }
    }
}

public void Event_InfectedDeath(Event event, const char[] name, bool dontBroadcast)
{
    int attacker = GetClientOfUserId(event.GetInt("attacker"));
    
    if (attacker > 0 && attacker <= MaxClients && IsClientInGame(attacker) && GetClientTeam(attacker) == 2)
    {
        g_iZombieKills[attacker]++;
    }
}

public Action Cmd_ShowKills(int client, int args)
{
    if (client == 0)
    {
        PrintToServer("Zombie Kill Stats:");
        for (int i = 1; i <= MaxClients; i++)
        {
            if (IsClientInGame(i) && GetClientTeam(i) == 2)
            {
                char name[MAX_NAME_LENGTH];
                GetClientName(i, name, sizeof(name));
                PrintToServer("- %s: %d", name, g_iZombieKills[i]);
            }
        }
    }
    else
    {
        PrintToChat(client, "\x04[Zombie Kills] \x01Kill Statistics:");
        for (int i = 1; i <= MaxClients; i++)
        {
            if (IsClientInGame(i) && GetClientTeam(i) == 2)
            {
                char name[MAX_NAME_LENGTH];
                GetClientName(i, name, sizeof(name));
                PrintToChat(client, "- %s: \x05%d\x01", name, g_iZombieKills[i]);
            }
        }
    }
    
    return Plugin_Handled;
}

void ShowKillStats()
{
    PrintToChatAll("\x04[Zombie Kills] \x01Round End Statistics:");
    
    for (int i = 1; i <= MaxClients; i++)
    {
        if (IsClientInGame(i) && GetClientTeam(i) == 2)
        {
            char name[MAX_NAME_LENGTH];
            GetClientName(i, name, sizeof(name));
            PrintToChatAll("- %s: \x05%d\x01", name, g_iZombieKills[i]);
        }
    }
}
