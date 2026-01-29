#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdkhooks>
#include <left4dhooks>  // For L4D2 specific SDK functions

#define PLUGIN_VERSION "1.0"

public Plugin myinfo =
{
    name = "Panic FF Control",
    author = "Developer",
    description = "Disables friendly fire during horde panic events",
    version = PLUGIN_VERSION,
    url = "http://steamcommunity.com/profiles/1234567890/"
};

bool g_bInPanic = false;

// Panic event constants
#define PANIC_START_EVENT "horde_panic_start"
#define PANIC_END_EVENT   "horde_panic_end"

public void OnPluginStart()
{
    // Hook panic events
    HookEvent(PANIC_START_EVENT, PanicStart_Event, EventHookMode_PostNoCopy);
    HookEvent(PANIC_END_EVENT, PanicEnd_Event, EventHookMode_PostNoCopy);
    
    // Print to server console
    PrintToServer("[Panic FF Control] Plugin loaded. Version: %s", PLUGIN_VERSION);
}

public void OnMapStart()
{
    // Reset panic state on map start
    g_bInPanic = false;
}

public void PanicStart_Event(Event event, const char[] name, bool dontBroadcast)
{
    g_bInPanic = true;
    
    // Optional: Print to console
    PrintToServer("[Panic FF Control] Horde panic started!");
}

public void PanicEnd_Event(Event event, const char[] name, bool dontBroadcast)
{
    g_bInPanic = false;
    
    // Optional: Print to console
    PrintToServer("[Panic FF Control] Horde panic ended!");
}

// SDKHook_OnTakeDamage callback
public Action OnTakeDamage(int victim, int &attacker, int &inflictor, int &damage, int &damagetype, int &weapon, int &ammo)
{
    // Check if victim and attacker are players
    if (!IsClientInGame(victim) || !IsClientInGame(attacker))
        return Plugin_Continue;
    
    // Check if we are in panic mode
    if (!g_bInPanic)
        return Plugin_Continue;
    
    // Check if damage is friendly fire
    if (GetClientTeam(victim) == GetClientTeam(attacker))
    {
        // Print debug info (optional)
        // PrintToServer("[Panic FF Control] FF damage blocked: Victim %d (%s), Attacker %d (%s)", 
        //               victim, GetClientName(victim), attacker, GetClientName(attacker));
        
        // Block damage
        damage = 0;
        return Plugin_Handled;
    }
    
    return Plugin_Continue;
}

// Utility function to get client team
int GetClientTeam(int client)
{
    if (client <= 0 || client > MaxClients)
        return -1;
    
    if (!IsClientInGame(client))
        return -1;
    
    return GetEntProp(client, Prop_Send, "m_iTeamNum");
}
