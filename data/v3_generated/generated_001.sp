#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>
#include <l4d2_boomer_bile>

public Plugin myinfo =
{
    name = "Boomer Bile Detector",
    author = "Developer",
    description = "Counts survivors hit by Boomer bile",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("infected_death", Event_InfectedDeath);
}

public void Event_InfectedDeath(Event event, const char[] name, bool dontBroadcast)
{
    // Check if the infected that died was a Boomer
    int entity = GetClientOfUserId(event.GetInt("userid"));
    if (entity > 0 && IsValidEntity(entity) && GetClientTeam(entity) == 3)
    {
        int classID = GetEntProp(entity, Prop_Send, "m_iClassID");
        if (classID == 88) // Boomer class ID
        {
            // Check if the Boomer was killed by a shotgun
            int weapon = GetPlayerWeaponSlot(entity, 0);
            if (weapon > 0 && IsValidEntity(weapon))
            {
                int weaponClass = GetEntPropEnt(weapon, Prop_Send, "m_weaponID");
                if (weaponClass == -1) // Invalid weapon class
                {
                    return;
                }

                // Check if the weapon is a shotgun
                if (weaponClass >= 0 && weaponClass <= 5)
                {
                    // Check if the Boomer exploded
                    if (GetEntProp(entity, Prop_Send, "m_bExploded"))
                    {
                        // Count survivors hit by bile
                        int count = CountSurvivorsHitByBile(entity);
                        if (count > 0)
                        {
                            PrintToChatAll("\x04[BOOMER BILE] \x01%d survivor(s) hit by bile!", count);
                        }
                    }
                }
            }
        }
    }
}

int CountSurvivorsHitByBile(int boomer)
{
    int count = 0;
    for (int i = 1; i <= MaxClients; i++)
    {
        if (IsClientInGame(i) && GetClientTeam(i) == 2 && IsPlayerAlive(i))
        {
            if (IsPlayerBiled(i) && GetBileSource(i) == boomer)
            {
                count++;
            }
        }
    }
    return count;
}