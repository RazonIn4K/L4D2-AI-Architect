#pragma semicolon 1
#pragma newdecls required

#include <sdkhooks>
#include <left4dhooks>  // For L4D_IsPanicEventActive()

bool g_bPanicActive = false;

public void OnPluginStart()
{
    HookEvent("panic_event_start", Event_PanicStart);
    HookEvent("panic_event_end", Event_PanicEnd);
    
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
    SDKHook(client, SDKHook_OnTakeDamage, OnTakeDamage);
}

public void Event_PanicStart(Event event, const char[] name, bool dontBroadcast)
{
    g_bPanicActive = true;
}

public void Event_PanicEnd(Event event, const char[] name, bool dontBroadcast)
{
    g_bPanicActive = false;
}

// CORRECT: Using SDKHook_OnTakeDamage for damage interception
public Action OnTakeDamage(int victim, int &attacker, int &inflictor, float &damage, int &damagetype)
{
    // Only process during panic events
    if (!g_bPanicActive)
        return Plugin_Continue;
    
    // Check if this is survivor-on-survivor damage
    if (victim > 0 && victim <= MaxClients && IsClientInGame(victim) && GetClientTeam(victim) == 2)
    {
        if (attacker > 0 && attacker <= MaxClients && IsClientInGame(attacker) && GetClientTeam(attacker) == 2)
        {
            // Block the damage by setting it to 0 and returning Plugin_Changed
            damage = 0.0;
            return Plugin_Changed;
        }
    }
    
    return Plugin_Continue;
}