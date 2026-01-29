#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>

#define MAX_WEAPONS 8

public Plugin myinfo =
{
    name = "Prevent Duplicate Weapons",
    author = "Developer",
    description = "Prevents survivors from picking up duplicate weapons",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("player_pickup", Event_PlayerPickup);
}

public void Event_PlayerPickup(Event event, const char[] name, bool dontBroadcast)
{
    int client = GetClientOfUserId(event.GetInt("userid"));
    int weapon = GetEntPropEnt(client, Prop_Send, "m_hActiveWeapon");

    if (client > 0 && client <= MaxClients && IsClientInGame(client) && GetClientTeam(client) == 2)
    {
        if (weapon > 0 && IsValidEntity(weapon))
        {
            int weaponID = GetWeaponID(weapon);
            if (weaponID != -1)
            {
                int weaponSlot = GetWeaponSlot(weapon);
                if (weaponSlot != -1)
                {
                    // Check if the player already has this weapon
                    for (int i = 0; i < MAX_WEAPONS; i++)
                    {
                        int currentWeapon = GetEntPropEnt(client, Prop_Send, "m_hMyWeapons", i);
                        if (currentWeapon == weapon)
                        {
                            // Block the pickup
                            event.SetInt("blocked", 1);
                            return;
                        }
                    }
                }
            }
        }
    }
}

int GetWeaponID(int weapon)
{
    char classname[32];
    GetEntityClassname(weapon, classname, sizeof(classname));

    if (strcmp(classname, "weapon_pistol") == 0) return 1;
    if (strcmp(classname, "weapon_shotgun") == 0) return 2;
    if (strcmp(classname, "weapon_smg") == 0) return 3;
    if (strcmp(classname, "weapon_rifle") == 0) return 4;
    if (strcmp(classname, "weapon_sniper") == 0) return 5;
    if (strcmp(classname, "weapon_grenade_launcher") == 0) return 6;
    if (strcmp(classname, "weapon_autoshotgun") == 0) return 7;

    return -1;
}

int GetWeaponSlot(int weapon)
{
    int weaponID = GetWeaponID(weapon);
    if (weaponID == -1) return -1;

    // Slots: 0 - Primary, 1 - Secondary
    return (weaponID <= 7) ? 0 : 1;
}