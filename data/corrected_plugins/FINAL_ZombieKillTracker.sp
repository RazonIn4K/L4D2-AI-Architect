#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

#define PLUGIN_VERSION "1.2"

public Plugin myinfo =
{
    name = "L4D2 Zombie Kill Tracker",
    author = "Optimized Version",
    description = "Tracks all zombie kills by survivors",
    version = PLUGIN_VERSION,
    url = ""
};

int g_iCommonKills[MAXPLAYERS + 1];
int g_iSpecialKills[MAXPLAYERS + 1];

public void OnPluginStart()
{
    RegConsoleCmd("sm_kills", Cmd_ShowKills, "Displays zombie kill counts");
    RegAdminCmd("sm_resetkills", Cmd_ResetKills, ADMFLAG_ROOT, "Reset all kill counts");
    
    // CORRECT: Different events for different infected types
    HookEvent("infected_death", Event_InfectedDeath);  // Common zombies
    HookEvent("player_death", Event_PlayerDeath);       // Special infected
    HookEvent("round_start", Event_RoundStart);
}

public void OnClientDisconnect(int client)
{
    g_iCommonKills[client] = 0;
    g_iSpecialKills[client] = 0;
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
    // Reset all kill counts
    for (int i = 1; i <= MaxClients; i++)
    {
        g_iCommonKills[i] = 0;
        g_iSpecialKills[i] = 0;
    }
}

// Common infected deaths (NOT players, so different event)
public void Event_InfectedDeath(Event event, const char[] name, bool dontBroadcast)
{
    int attacker = GetClientOfUserId(event.GetInt("attacker"));
    
    if (IsValidSurvivor(attacker))
    {
        g_iCommonKills[attacker]++;
    }
}

// Special infected deaths (they ARE players, so player_death)
public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast)
{
    int victim = GetClientOfUserId(event.GetInt("userid"));
    int attacker = GetClientOfUserId(event.GetInt("attacker"));
    
    // Check victim was infected team
    if (!IsValidClient(victim) || GetClientTeam(victim) != 3)
        return;
    
    // Check attacker was survivor team
    if (!IsValidSurvivor(attacker))
        return;
    
    g_iSpecialKills[attacker]++;
}

public Action Cmd_ShowKills(int client, int args)
{
    PrintToChat(client, "\x04=== Zombie Kill Counts ===");
    
    // Find top killer
    int topKiller = -1;
    int topKills = 0;
    int totalKills = 0;
    
    for (int i = 1; i <= MaxClients; i++)
    {
        if (!IsClientInGame(i) || GetClientTeam(i) != 2)
            continue;
        
        int total = g_iCommonKills[i] + g_iSpecialKills[i];
        totalKills += total;
        
        if (total > topKills)
        {
            topKills = total;
            topKiller = i;
        }
    }
    
    // Show total and top killer
    PrintToChat(client, "\x01Total Kills: \x05%d", totalKills);
    if (topKiller > 0)
    {
        char name[MAX_NAME_LENGTH];
        GetClientName(topKiller, name, sizeof(name));
        PrintToChat(client, "\x01Top Killer: \x04%s \x05(%d kills)", name, topKills);
    }
    
    PrintToChat(client, "\x01--------------------");
    
    // Show individual stats
    for (int i = 1; i <= MaxClients; i++)
    {
        if (!IsClientInGame(i) || GetClientTeam(i) != 2)
            continue;
        
        // CORRECT: GetClientName requires a buffer!
        char playerName[MAX_NAME_LENGTH];
        GetClientName(i, playerName, sizeof(playerName));
        
        int total = g_iCommonKills[i] + g_iSpecialKills[i];
        PrintToChat(client, "\x01%s: \x05%d common\x01, \x03%d special\x01 (\x04%d total\x01)", 
                    playerName, g_iCommonKills[i], g_iSpecialKills[i], total);
    }
    
    return Plugin_Handled;
}

public Action Cmd_ResetKills(int client, int args)
{
    for (int i = 1; i <= MaxClients; i++)
    {
        g_iCommonKills[i] = 0;
        g_iSpecialKills[i] = 0;
    }
    
    if (client == 0)
    {
        PrintToServer("All kill counts reset!");
    }
    else
    {
        PrintToChatAll("\x04[Kill Tracker]\x01 All kill counts have been reset!");
    }
    
    return Plugin_Handled;
}

bool IsValidSurvivor(int client)
{
    return (client > 0 && client <= MaxClients && 
            IsClientInGame(client) && GetClientTeam(client) == 2);
}

bool IsValidClient(int client)
{
    return (client > 0 && client <= MaxClients && IsClientInGame(client));
}
