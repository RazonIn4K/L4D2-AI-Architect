#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

public Plugin myinfo =
{
    name = "Smoker Grab Announcer",
    author = "Developer",
    description = "Announces when a Smoker grabs a survivor",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("smoker_grab", Event_SmokerGrab);
}

public void Event_SmokerGrab(Event event, const char[] name, bool dontBroadcast)
{
    int smokerIndex = GetClientOfUserId(event.GetInt("userid"));
    int victimIndex = GetClientOfUserId(event.GetInt("victimid"));
    
    if (smokerIndex > 0 && smokerIndex <= MaxClients && IsClientInGame(smokerIndex) && GetClientTeam(smokerIndex) == 3)
    {
        if (victimIndex > 0 && victimIndex <= MaxClients && IsClientInGame(victimIndex) && GetClientTeam(victimIndex) == 2)
        {
            PrintToChatAll("\x04[Smoker Action]\x01 Smoker %N grabbed %N!", smokerIndex, victimIndex);
        }
    }
}