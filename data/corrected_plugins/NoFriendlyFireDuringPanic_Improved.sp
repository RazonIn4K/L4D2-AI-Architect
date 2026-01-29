#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>

bool g_bPanicActive = false;
bool g_bIsSurvivor[MAXPLAYERS + 1];

public Plugin myinfo =
{
    name = "No Friendly Fire During Panic",
    author = "Developer",
    description = "Disables friendly fire while panic event is active",
    version = "1.1",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("create_panic_event", Event_PanicStart);
    HookEvent("panic_event_finished", Event_PanicEnd);
    HookEvent("round_start", Event_RoundStart);
    HookEvent("player_team", Event_PlayerTeam);
    HookEvent("player_disconnect", Event_PlayerDisconnect);
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
    g_bPanicActive = false;
    UpdateTeamCache();
}

public void Event_PlayerTeam(Event event, const char[] name, bool dontBroadcast)
{
    int userid = event.GetInt("userid");
    int client = GetClientOfUserId(userid);
    if (client > 0 && client <= MaxClients)
    {
        // Update after a short delay to ensure team change is processed
        CreateTimer(0.1, Timer_UpdateClientTeam, userid);
    }
}

public void Event_PlayerDisconnect(Event event, const char[] name, bool dontBroadcast)
{
    int userid = event.GetInt("userid");
    int client = GetClientOfUserId(userid);
    if (client > 0 && client <= MaxClients)
    {
        g_bIsSurvivor[client] = false;
    }
}

public Action Timer_UpdateClientTeam(Handle timer, int userid)
{
    int client = GetClientOfUserId(userid);
    if (client > 0 && client <= MaxClients && IsClientInGame(client))
    {
        g_bIsSurvivor[client] = (GetClientTeam(client) == 2);
    }
    return Plugin_Continue;
}

void UpdateTeamCache()
{
    for (int i = 1; i <= MaxClients; i++)
    {
        g_bIsSurvivor[i] = (IsClientInGame(i) && GetClientTeam(i) == 2);
    }
}

public void OnClientPutInServer(int client)
{
    SDKHook(client, SDKHook_OnTakeDamage, OnTakeDamage);
    g_bIsSurvivor[client] = false;
}

public void Event_PanicStart(Event event, const char[] name, bool dontBroadcast)
{
    g_bPanicActive = true;
    PrintToChatAll("\x04[Panic] \x01Panic event started! Friendly fire disabled.");
}

public void Event_PanicEnd(Event event, const char[] name, bool dontBroadcast)
{
    g_bPanicActive = false;
    PrintToChatAll("\x04[Panic] \x01Panic event ended! Friendly fire re-enabled.");
}

public Action OnTakeDamage(int victim, int &attacker, int &inflictor, float &damage, int &damagetype)
{
    if (!g_bPanicActive)
        return Plugin_Continue;
    
    // Use cached team information for performance
    if (attacker > 0 && attacker <= MaxClients && g_bIsSurvivor[attacker])
    {
        if (victim > 0 && victim <= MaxClients && g_bIsSurvivor[victim])
        {
            damage = 0.0;
            return Plugin_Changed;
        }
    }
    
    return Plugin_Continue;
}
