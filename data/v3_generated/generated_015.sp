#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>

public Plugin myinfo =
{
    name = "Defibrillator Pickup Announcer",
    author = "Developer",
    description = "Announces when a survivor picks up a defibrillator",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("item_pickup", Event_ItemPickup);
}

public void Event_ItemPickup(Event event, const char[] name, bool dontBroadcast)
{
    int client = GetClientOfUserId(event.GetInt("userid"));
    char item[64];
    
    if (client > 0 && IsClientInGame(client) && GetClientTeam(client) == 2) // Survivor team
    {
        StrCopy(item, sizeof(item), event.GetString("item"));
        
        if (StrContains(item, "defibrillator") != -1)
        {
            PrintToChatAll("\x04[ANNOUNCE] \x01%s \x00has picked up a defibrillator!", GetClientName(client));
            EmitSoundToAll("ui/achievement_earned.wav");
        }
    }
}