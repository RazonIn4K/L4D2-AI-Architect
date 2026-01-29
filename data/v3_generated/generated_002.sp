#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>

#define SOUND_HUNTER_POUNCE "ui/hunter_pounce.wav"

public Plugin myinfo =
{
    name = "Hunter Pounce Sound",
    author = "Developer",
    description = "Plays a sound when Hunter pounces on a survivor",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("pounce", Event_HunterPounce);
}

public void Event_HunterPounce(Event event, const char[] name, bool dontBroadcast)
{
    int attacker = GetClientOfUserId(event.GetInt("userid"));
    int victim = GetClientOfUserId(event.GetInt("victim"));

    if (attacker > 0 && attacker <= MaxClients && IsClientInGame(attacker) && GetClientTeam(attacker) == 3)
    {
        if (victim > 0 && victim <= MaxClients && IsClientInGame(victim) && GetClientTeam(victim) == 2)
        {
            // Play sound for the victim
            EmitSoundToClient(victim, SOUND_HUNTER_POUNCE);
        }
    }
}