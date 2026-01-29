#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

#define MAX_KILLS 2147483647  // Maximum int value

int g_iZombieKills[MAXPLAYERS + 1];

public Plugin myinfo =
{
    name = "Zombie Kill Tracker",
    author = "Developer",
    description = "Tracks and displays zombie kill counts for each player",
    version = "1.1",
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
    RegConsoleCmd("sm_resetkills", Cmd_ResetKills, ADMFLAG_ROOT);
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
            AddKill(attacker);
        }
    }
}

public void Event_InfectedDeath(Event event, const char[] name, bool dontBroadcast)
{
    int attacker = GetClientOfUserId(event.GetInt("attacker"));
    
    if (attacker > 0 && attacker <= MaxClients && IsClientInGame(attacker) && GetClientTeam(attacker) == 2)
    {
        AddKill(attacker);
    }
}

void AddKill(int client)
{
    // Prevent integer overflow
    if (g_iZombieKills[client] < MAX_KILLS)
    {
        g_iZombieKills[client]++;
    }
    else
    {
        // Reset to 0 if we hit the max (unlikely but safe)
        g_iZombieKills[client] = 0;
        PrintToChat(client, "\x04[Zombie Kills] \x01Kill count reset due to maximum limit!");
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
        
        // Show player's own kills first
        if (GetClientTeam(client) == 2)
        {
            char clientName[MAX_NAME_LENGTH];
            GetClientName(client, clientName, sizeof(clientName));
            PrintToChat(client, "- \x05%s\x01: \x04%d\x01 (you)", clientName, g_iZombieKills[client]);
        }
        
        // Show other players' kills
        for (int i = 1; i <= MaxClients; i++)
        {
            if (i != client && IsClientInGame(i) && GetClientTeam(i) == 2)
            {
                char name[MAX_NAME_LENGTH];
                GetClientName(i, name, sizeof(name));
                PrintToChat(client, "- %s: \x05%d\x01", name, g_iZombieKills[i]);
            }
        }
    }
    
    return Plugin_Handled;
}

public Action Cmd_ResetKills(int client, int args)
{
    if (client == 0)
    {
        for (int i = 1; i <= MaxClients; i++)
        {
            g_iZombieKills[i] = 0;
        }
        PrintToServer("All kill counts reset!");
    }
    else
    {
        for (int i = 1; i <= MaxClients; i++)
        {
            g_iZombieKills[i] = 0;
        }
        PrintToChatAll("\x04[Zombie Kills] \x01All kill counts have been reset by admin!");
    }
    
    return Plugin_Handled;
}

void ShowKillStats()
{
    // Find top killer
    int topKiller = -1;
    int topKills = 0;
    int totalKills = 0;
    
    for (int i = 1; i <= MaxClients; i++)
    {
        if (IsClientInGame(i) && GetClientTeam(i) == 2)
        {
            totalKills += g_iZombieKills[i];
            if (g_iZombieKills[i] > topKills)
            {
                topKills = g_iZombieKills[i];
                topKiller = i;
            }
        }
    }
    
    PrintToChatAll("\x04[Zombie Kills] \x01Round End Statistics:");
    PrintToChatAll("- \x05Total Kills\x01: %d", totalKills);
    
    if (topKiller > 0)
    {
        char name[MAX_NAME_LENGTH];
        GetClientName(topKiller, name, sizeof(name));
        PrintToChatAll("- \x05Top Killer\x01: %s with %d kills!", name, topKills);
    }
    
    // Show individual stats
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
