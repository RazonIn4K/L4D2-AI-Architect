#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>
#include <left4dhooks>

#define PLUGIN_VERSION "1.0"

public Plugin myinfo =
{
    name = "Charger Carry Tracker",
    author = "ChatGPT",
    description = "Tracks and logs Charger carries",
    version = PLUGIN_VERSION,
    url = ""
};

public void OnPluginStart()
{
    // Charger carry events
    HookEvent("charger_carry_start", Event_ChargerGrab);
    HookEvent("charger_carry_end", Event_ChargerRelease);
    HookEvent("charger_pummel_end", Event_PummelEnd);
}

public void Event_ChargerGrab(Event event, const char[] name, bool dontBroadcast)
{
    int charger = GetClientOfUserId(event.GetInt("userid"));
    int victim = GetClientOfUserId(event.GetInt("victim"));

    if (charger > 0 && IsClientInGame(charger) && IsPlayerAlive(charger))
    {
        PrintToChatAll("Charger %N grabbed %N!", charger, victim);
    }
}

public void Event_ChargerRelease(Event event, const char[] name, bool dontBroadcast)
{
    int charger = GetClientOfUserId(event.GetInt("userid"));

    if (charger > 0 && IsClientInGame(charger) && IsPlayerAlive(charger))
    {
        PrintToChatAll("Charger %N released!", charger);
    }
}

public void Event_PummelEnd(Event event, const char[] name, bool dontBroadcast)
{
    int charger = GetClientOfUserId(event.GetInt("userid"));

    if (charger > 0 && IsClientInGame(charger) && IsPlayerAlive(charger))
    {
        PrintToChatAll("Charger %N finished pummeling!", charger);
    }
}
