#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>
#include <left4dhooks>

#define PLUGIN_VERSION "1.0"

public Plugin myinfo =
{
    name = "Hunter Pounce Damage Tracker",
    author = "Developer",
    description = "Tracks and displays damage dealt by Hunter pounces",
    version = PLUGIN_VERSION,
    url = ""
};

public void OnPluginStart()
{
    // CORRECT: lunge_pounce for Hunter pounces
    HookEvent("lunge_pounce", Event_HunterPounce);
    HookEvent("pounce_end", Event_PounceEnd);
    
    CreateConVar("l4d2_hunter_pounce_tracker_version", PLUGIN_VERSION, "Hunter Pounce Tracker Version", FCVAR_NOTIFY | FCVAR_DONTRECORD);
}

public void Event_HunterPounce(Event event, const char[] name, bool dontBroadcast)
{
    int hunter = GetClientOfUserId(event.GetInt("userid"));
    int victim = GetClientOfUserId(event.GetInt("victim"));
    int damage = event.GetInt("damage");
    
    if (hunter > 0 && IsClientInGame(hunter) && GetClientTeam(hunter) == 3)
    {
        PrintToChatAll("Hunter %N pounced %N for %d damage!", hunter, victim, damage);
    }
}

public void Event_PounceEnd(Event event, const char[] name, bool dontBroadcast)
{
    int hunter = GetClientOfUserId(event.GetInt("userid"));
    int victim = GetClientOfUserId(event.GetInt("victim"));
    
    if (hunter > 0 && IsClientInGame(hunter) && GetClientTeam(hunter) == 3)
    {
        PrintToChatAll("Hunter %N pounce ended on %N", hunter, victim);
    }
}
