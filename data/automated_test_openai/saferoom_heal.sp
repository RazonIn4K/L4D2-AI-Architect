#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>
#include <left4dhooks>

#define PLUGIN_VERSION "1.0"

public Plugin myinfo =
{
    name = "Saferoom Heal",
    author = "Developer",
    description = "Heals survivors when they enter the saferoom",
    version = PLUGIN_VERSION,
    url = "http://steamcommunity.com/profiles/1234567890/"
};

public void OnPluginStart()
{
    // Use saferoom trigger touch for all campaigns
    HookEvent("player_entered_safe_area", Event_PlayerEnteredSaferoom, EventHookMode_PostNoCopy);
    HookEvent("player_left_safe_area", Event_PlayerLeftSaferoom, EventHookMode_PostNoCopy);
}

public void Event_PlayerEnteredSaferoom(Event event, const char[] name, bool dontBroadcast)
{
    int client = GetClientOfUserId(event.GetInt("userid"));
    if (!IsClientInGame(client) || !IsPlayerAlive(client))
        return;

    // Check if we need to heal
    if (ShouldHeal(client))
    {
        // Perform heal
        PrintToChat(client, "\x04[Saferoom Heal]\x01 You have been healed!");
        SetEntProp(client, Prop_Send, "m_iHealth", 100);
        SetEntProp(client, Prop_Send, "m_iHealthMax", 100);
        
        // Optional: Play heal sound
        EmitSoundToClient(client, "items/medshotno1.wav");
    }
}

public void Event_PlayerLeftSaferoom(Event event, const char[] name, bool dontBroadcast)
{
    int client = GetClientOfUserId(event.GetInt("userid"));
    if (!IsClientInGame(client) || !IsPlayerAlive(client))
        return;

    // Reset any heal state if needed
}

bool ShouldHeal(int client)
{
    // Check if client is below 50% health
    int health = GetClientHealth(client);
    if (health >= 50)
        return false;

    // Check if client has no medkit or is using one
    if (HasClientLethal(client))
        return false;

    return true;
}

bool HasClientLethal(int client)
{
    int weapon = GetPlayerWeaponSlot(client, 0);
    if (weapon <= 0)
        return false;

    char weaponName[64];
    GetEdictClassname(EntIndexToEntRef(weapon), weaponName, sizeof(weaponName));
    return (StrContains(weaponName, "first_aid") != -1);
}
