#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>

public Plugin myinfo =
{
    name = "Hunter Rescue God Mode",
    author = "Developer",
    description = "Gives survivors temporary god mode after being rescued from a Hunter",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("player_rescue_announce", Event_PlayerRescueAnnounce);
}

public void Event_PlayerRescueAnnounce(Event event, const char[] name, bool dontBroadcast)
{
    int survivor = GetClientOfUserId(event.GetInt("player"));
    if (survivor > 0 && survivor <= MaxClients && IsClientInGame(survivor) && GetClientTeam(survivor) == 2)
    {
        GiveGodMode(survivor, true);
        CreateTimer(2.0, Timer_RemoveGodMode, survivor);
    }
}

void GiveGodMode(int client, bool state)
{
    SetEntProp(client, Prop_Send, "m_bInvulnerable", state);
    SetEntProp(client, Prop_Send, "m_bCloaked", state);
}

Action Timer_RemoveGodMode(Handle timer, int survivor)
{
    if (survivor > 0 && IsClientInGame(survivor) && GetClientTeam(survivor) == 2)
    {
        GiveGodMode(survivor, false);
    }
    return Plugin_Continue;
}