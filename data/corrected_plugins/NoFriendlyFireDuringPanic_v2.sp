/**
 * =============================================================================
 * L4D2 No Friendly Fire During Panic - CORRECTED REFERENCE IMPLEMENTATION
 * =============================================================================
 * Demonstrates proper usage of:
 * - SDKHooks for damage interception (NOT event modification!)
 * - create_panic_event / panic_event_finished events
 * - Action return type for SDKHook callbacks
 *
 * CRITICAL LESSON: Events are INFORMATIONAL ONLY!
 * You CANNOT prevent/modify damage by changing event values.
 * You MUST use SDKHook_OnTakeDamage to intercept damage.
 */

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>  // REQUIRED for damage interception!

#pragma semicolon 1
#pragma newdecls required

#define PLUGIN_VERSION "1.0"

public Plugin myinfo =
{
    name = "L4D2 No FF During Panic",
    author = "Corrected Example",
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
// Note: Returns Action, NOT void like event callbacks!
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

/*
 * =============================================================================
 * WRONG APPROACH (for reference - DO NOT USE)
 * =============================================================================
 * This does NOT work because events are informational only!
 *
 * public void Event_PlayerHurt(Event event, const char[] name, bool dontBroadcast)
 * {
 *     // This has NO EFFECT - events are read-only notifications!
 *     event.SetInt("dmg_health", 0);
 *
 *     // ERROR: void function cannot return a value!
 *     return Plugin_Handled;
 * }
 */
