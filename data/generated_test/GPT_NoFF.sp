#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdkhooks>
#include <sdktools>

#define PLUGIN_VERSION "1.0"

public Plugin myinfo =
{
    name = "Disable Friendly Fire in Panic Events",
    author = "Developer",
    description = "Disables friendly fire during panic events",
    version = PLUGIN_VERSION,
    url = "https://example.com"
};

bool g_bInPanic = false;

public void OnPluginStart()
{
    HookEvent("panic_start", Event_PanicStart, EventHookMode_PostNoCopy);
    HookEvent("panic_end", Event_PanicEnd, EventHookMode_PostNoCopy);
    HookEvent("round_start", Event_RoundStart, EventHookMode_PostNoCopy);
    HookEvent("round_end", Event_RoundEnd, EventHookMode_PostNoCopy);
}

public void SDKHook_OnTakeDamage(int victim, int &attacker, int &inflictor, float &damage, int &damagetype)
{
    // Block friendly fire damage during panic events
    if (g_bInPanic && attacker != victim && IsPlayerAlive(attacker) && IsPlayerAlive(victim))
    {
        // Check if the damage is from a survivor to another survivor
        if (GetClientTeam(attacker) == 2 && GetClientTeam(victim) == 2)
        {
            // Block the damage
            return Plugin_Handled;
        }
    }
    return Plugin_Continue;
}

void Event_PanicStart(Event event, const char[] name, bool dontBroadcast)
{
    g_bInPanic = true;
}

void Event_PanicEnd(Event event, const char[] name, bool dontBroadcast)
{
    g_bInPanic = false;
}

void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
    // Also disable panic events when round starts
    g_bInPanic = false;
}

void Event_RoundEnd(Event event, const char[] name, bool dontBroadcast)
{
    // Also disable panic events when round ends
    g_bInPanic = false;
}