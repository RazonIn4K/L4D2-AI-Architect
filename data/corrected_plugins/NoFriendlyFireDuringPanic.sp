#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>

public Plugin myinfo =
{
    name = "No Friendly Fire During Panic",
    author = "Developer",
    description = "Disables friendly fire while panic event is active",
    version = "1.0",
    url = ""
};

bool g_bPanicActive = false;

public void OnPluginStart()
{
    HookEvent("create_panic_event", Event_PanicStart);
    HookEvent("panic_event_finished", Event_PanicEnd);
}

public void OnClientPutInServer(int client)
{
    SDKHook(client, SDKHook_OnTakeDamage, OnTakeDamage);
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
    
    if (attacker > 0 && attacker <= MaxClients && IsClientInGame(attacker) && GetClientTeam(attacker) == 2)
    {
        if (victim > 0 && victim <= MaxClients && IsClientInGame(victim) && GetClientTeam(victim) == 2)
        {
            damage = 0.0;
            return Plugin_Changed;
        }
    }
    
    return Plugin_Continue;
}
