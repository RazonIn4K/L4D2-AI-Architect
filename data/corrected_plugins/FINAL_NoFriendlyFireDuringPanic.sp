#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>  // REQUIRED for damage interception!

#define PLUGIN_VERSION "1.2"

public Plugin myinfo =
{
    name = "L4D2 No FF During Panic",
    author = "Optimized Version",
    description = "Prevents friendly fire during panic events",
    version = PLUGIN_VERSION,
    url = ""
};

bool g_bPanicActive = false;

public void OnPluginStart()
{
    // Hook panic events - these are L4D2 specific!
    HookEvent("create_panic_event", Event_PanicStart);
    HookEvent("panic_event_finished", Event_PanicEnd);
    HookEvent("round_start", Event_RoundStart);
    
    // Late load support
    for (int i = 1; i <= MaxClients; i++)
    {
        if (IsClientInGame(i))
        {
            OnClientPutInServer(i);
        }
    }
}

public void OnClientPutInServer(int client)
{
    // CRITICAL: SDKHook is the ONLY way to intercept/modify damage
    // Events are INFORMATIONAL ONLY - you cannot block damage via events!
    SDKHook(client, SDKHook_OnTakeDamage, OnTakeDamage);
}

public void OnClientDisconnect(int client)
{
    SDKUnhook(client, SDKHook_OnTakeDamage, OnTakeDamage);
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
    g_bPanicActive = false;
}

public void Event_PanicStart(Event event, const char[] name, bool dontBroadcast)
{
    g_bPanicActive = true;
    PrintToChatAll("\x04[FF Protection]\x01 Panic event started - friendly fire disabled!");
}

public void Event_PanicEnd(Event event, const char[] name, bool dontBroadcast)
{
    g_bPanicActive = false;
    PrintToChatAll("\x04[FF Protection]\x01 Panic event ended - friendly fire re-enabled.");
}

// CORRECT: This is how you actually prevent damage in Source games
public Action OnTakeDamage(int victim, int &attacker, int &inflictor, 
                           float &damage, int &damagetype, int &weapon,
                           float damageForce[3], float damagePosition[3])
{
    // Only process during panic
    if (!g_bPanicActive)
        return Plugin_Continue;
    
    // Check if this is survivor-on-survivor damage (friendly fire)
    if (IsValidSurvivor(victim) && IsValidSurvivor(attacker) && victim != attacker)
    {
        // Block the damage by setting it to 0 and returning Plugin_Changed
        damage = 0.0;
        return Plugin_Changed;
    }
    
    return Plugin_Continue;
}

bool IsValidSurvivor(int client)
{
    return (client > 0 && client <= MaxClients && 
            IsClientInGame(client) && GetClientTeam(client) == 2);
}
