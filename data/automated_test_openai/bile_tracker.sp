#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>

#define BOOMER_BILE_DURATION 30.0

public Plugin myinfo =
{
    name = "Bile Tracker",
    author = "ChatGPT",
    description = "Tracks when players get covered in Boomer bile",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("player_now_it", Event_PlayerBiled);
    HookEvent("player_no_longer_it", Event_BileWoreOff);
}

public void OnClientDisconnect(int client)
{
    // Clean up any timers if player disconnects
    DeleteBileTimer(client);
}

void Event_PlayerBiled(Event event, const char[] name, bool dontBroadcast)
{
    int victim = GetClientOfUserId(event.GetInt("userid"));
    int attacker = GetClientOfUserId(event.GetInt("attacker"));

    if (victim > 0 && IsClientInGame(victim) && IsPlayerAlive(victim))
    {
        PrintToChatAll("%N got covered in bile!", victim);

        // Start timer to track bile duration
        CreateTimer(BOOMER_BILE_DURATION, Timer_BileWoreOff, GetClientUserId(victim), TIMER_FLAG_NO_MAPCHANGE);
    }
}

void Event_BileWoreOff(Event event, const char[] name, bool dontBroadcast)
{
    int victim = GetClientOfUserId(event.GetInt("userid"));

    if (victim > 0 && IsClientInGame(victim) && IsPlayerAlive(victim))
    {
        PrintToChatAll("%N's bile wore off!", victim);
    }
}

public Action Timer_BileWoreOff(Handle timer, int userid)
{
    int client = GetClientOfUserId(userid);
    if (client > 0 && IsClientInGame(client) && IsPlayerAlive(client))
    {
        PrintToChatAll("%N's bile wore off!", client);
    }
    return Plugin_Continue;
}
