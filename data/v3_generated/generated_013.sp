#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>

#define MAX_DROPS 3

public Plugin myinfo =
{
    name = "Drop Weapon",
    author = "Developer",
    description = "Allows survivors to drop their primary weapon",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    RegConsoleCmd("sm_dropweapon", DropWeaponCmd, "Drop your primary weapon");
}

Action DropWeaponCmd(int client, int args)
{
    if (!IsClientInGame(client) || GetClientTeam(client) != 2)
    {
        ReplyToCommand(client, "[SM] You must be a survivor to use this command.");
        return Plugin_Handled;
    }

    int weapon = GetPlayerWeaponSlot(client, 0);
    if (weapon == -1)
    {
        ReplyToCommand(client, "[SM] You are not holding any weapon.");
        return Plugin_Handled;
    }

    char weaponName[64];
    GetEntityClassname(weapon, weaponName, sizeof(weaponName));

    // Check if the weapon is a primary weapon (not a melee or secondary)
    if (StrContains(weaponName, "primary") == -1)
    {
        ReplyToCommand(client, "[SM] You can only drop primary weapons.");
        return Plugin_Handled;
    }

    // Drop the weapon
    DropPlayerWeapon(client, weapon);
    ReplyToCommand(client, "[SM] You dropped your weapon.");

    return Plugin_Handled;
}

void DropPlayerWeapon(int client, int weapon)
{
    // Check for max drops
    static int dropCount[MAXPLAYERS + 1];
    if (dropCount[client] >= MAX_DROPS)
    {
        PrintToChat(client, "[SM] You can only drop %d weapons.", MAX_DROPS);
        return;
    }

    // Drop the weapon
    AcceptEntityInput(weapon, "Kill");
    dropCount[client]++;

    // Play sound
    EmitSoundToClient(client, "items/ammocrate_open.wav");
}