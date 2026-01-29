#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

#define REVIVE_DELAY 3.0
#define REVIVE_HEALTH 50

public Plugin myinfo =
{
    name = "Auto Revive",
    author = "Developer",
    description = "Automatically revives players after they become incapacitated",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("player_incapacitated", Event_PlayerIncapacitated);
}

public void Event_PlayerIncapacitated(Event event, const char[] name, bool dontBroadcast)
{
    int survivor = GetClientOfUserId(event.GetInt("userid"));
    
    if (survivor > 0 && survivor <= MaxClients && IsClientInGame(survivor) && GetClientTeam(survivor) == 2)
    {
        if (IsClientIncapacitated(survivor))
        {
            PrintToChat(survivor, "\x04[Auto Revive] \x01You will be revived in %d seconds!", RoundToFloor(REVIVE_DELAY));
            CreateTimer(REVIVE_DELAY, Timer_RevivePlayer, GetClientUserId(survivor));
        }
    }
}

public Action Timer_RevivePlayer(Handle timer, int userid)
{
    int survivor = GetClientOfUserId(userid);
    if (survivor > 0 && IsClientInGame(survivor) && IsClientIncapacitated(survivor))
    {
        ReviveClient(survivor);
        PrintToChat(survivor, "\x04[Auto Revive] \x01You have been revived!");
    }
    return Plugin_Continue;
}

bool IsClientIncapacitated(int client)
{
    return GetEntProp(client, Prop_Send, "m_isIncapacitated") == 1;
}

void ReviveClient(int client)
{
    // Method 1: Using cheat command (requires sv_cheats or command flags)
    int flags = GetCommandFlags("give");
    SetCommandFlags("give", flags & ~FCVAR_CHEAT);
    FakeClientCommand(client, "give health");
    SetCommandFlags("give", flags);
    
    // Alternative: Direct prop manipulation (may not work fully)
    // SetEntProp(client, Prop_Send, "m_isIncapacitated", 0);
    // SetEntProp(client, Prop_Send, "m_iHealth", REVIVE_HEALTH);
}
