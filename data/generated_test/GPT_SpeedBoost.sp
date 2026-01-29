#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>
#include <left4dhooks>

#define PLUGIN_VERSION "1.0"

public Plugin myinfo =
{
    name = "Kill Speed Boost",
    author = "Developer",
    description = "Gives survivors a speed boost on killing special infected",
    version = PLUGIN_VERSION,
    url = ""
};

#define BOOST_DURATION 5.0
#define SPEED_BOOST 1.3

public void OnPluginStart()
{
    HookEvent("player_death", Event_PlayerDeath);
}

public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast)
{
    int victim = GetClientOfUserId(event.GetInt("userid"));
    int attacker = GetClientOfUserId(event.GetInt("attacker"));

    // Check if a survivor killed a special infected
    if (attacker > 0 && attacker <= MaxClients && IsClientInGame(attacker))
    {
        if (GetClientTeam(attacker) == 2) // Survivor team
        {
            if (victim > 0 && victim <= MaxClients && IsClientInGame(victim))
            {
                if (GetClientTeam(victim) == 3) // Infected team
                {
                    ApplySpeedBoost(attacker);
                }
            }
        }
    }
}

void ApplySpeedBoost(int client)
{
    float currentSpeed = GetEntPropFloat(client, Prop_Send, "m_flMaxSpeed");
    float newSpeed = currentSpeed * SPEED_BOOST;

    SetEntPropFloat(client, Prop_Send, "m_flMaxSpeed", newSpeed);
    SetEntPropFloat(client, Prop_Send, "m_flMaxSpeedCrouched", newSpeed * 0.5);

    PrintToChat(client, "\x04[BOOST] \x01Speed boost activated!");

    CreateTimer(BOOST_DURATION, Timer_RemoveSpeedBoost, GetClientUserId(client));
}

Action Timer_RemoveSpeedBoost(Handle timer, int userid)
{
    int client = GetClientOfUserId(userid);
    if (client > 0 && IsClientInGame(client) && IsPlayerAlive(client))
    {
        float currentSpeed = GetEntPropFloat(client, Prop_Send, "m_flMaxSpeed");
        float normalSpeed = 0.0;

        // Find normal speed
        for (int i = 0; i < 10; i++)
        {
            normalSpeed = GetEntPropFloat(client, Prop_Send, "m_flMaxSpeed", i);
            if (normalSpeed > 0.0)
                break;
        }

        SetEntPropFloat(client, Prop_Send, "m_flMaxSpeed", normalSpeed);
        SetEntPropFloat(client, Prop_Send, "m_flMaxSpeedCrouched", normalSpeed * 0.5);

        PrintToChat(client, "\x04[BOOST] \x01Speed boost ended.");
    }
    return Plugin_Continue;
}