#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>

#define PLUGIN_VERSION "1.0"

public Plugin myinfo =
{
    name = "First Aid Kit Heal Nearby Survivors",
    author = "Developer",
    description = "Heals nearby survivors when using a first aid kit",
    version = PLUGIN_VERSION,
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

    if (client > 0 && client <= MaxClients && IsClientInGame(client))
    {
        // Check if the client is a survivor (team 2) and is alive
        if (GetClientTeam(client) == 2 && IsPlayerAlive(client))
        {
            // Check if the client is using a first aid kit
            int weapon = GetPlayerWeaponSlot(client, 0);
            if (weapon > 0 && GetEntProp(weapon, Prop_Send, "m_iItemDefinitionIndex") == 4) // 4 = First Aid Kit
            {
                // Heal nearby survivors
                HealNearbySurvivors(client);
            }
        }
    }
}

void HealNearbySurvivors(int client)
{
    float clientPos[3];
    GetClientAbsOrigin(client, clientPos);

    int healedCount = 0;
    for (int i = 1; i <= MaxClients; i++)
    {
        if (i != client && IsClientInGame(i) && GetClientTeam(i) == 2 && IsPlayerAlive(i))
        {
            float survivorPos[3];
            GetClientAbsOrigin(i, survivorPos);

            // Check if within 250 units
            if (GetVectorDistance(clientPos, survivorPos) <= 250.0)
            {
                // Heal survivor
                SetEntProp(i, Prop_Send, "m_iHealth", 100);
                healedCount++;
            }
        }
    }

    if (healedCount > 0)
    {
        PrintToChat(client, "\x04[HEAL] \x01Healed %d nearby survivor(s)!", healedCount);
    }
}