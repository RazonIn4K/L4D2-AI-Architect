#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>
#include <l4d2_infected_health>

public Plugin myinfo =
{
    name = "Medkit Delay",
    author = "Developer",
    description = "Prevents survivors from using medkits until below 40 health",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("player_use", Event_PlayerUse);
}

public void Event_PlayerUse(Event event, const char[] name, bool dontBroadcast)
{
    int userId = event.GetInt("userid");
    int client = GetClientOfUserId(userId);

    if (client > 0 && client <= MaxClients && IsClientInGame(client) && GetClientTeam(client) == 2)
    {
        // Check if the player is trying to use a medkit
        int weapon = GetPlayerWeaponSlot(client, 0);
        if (weapon > 0 && GetEntProp(weapon, Prop_Send, "m_iItemDefinitionIndex") == 3) // 3 = Medkit
        {
            int health = GetClientHealth(client);
            if (health > 40)
            {
                // Block the medkit usage
                event.SetInt("userid", 0); // Prevent the medkit from being used
                PrintToChat(client, "{red}You must be below 40 health to use a medkit!");
            }
        }
    }
}