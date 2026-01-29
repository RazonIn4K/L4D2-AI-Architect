#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>

public Plugin myinfo =
{
    name = "Tank Spawn Announcer",
    author = "Developer",
    description = "Announces when a Tank spawns with its health",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("tank_spawn", Event_TankSpawn);
}

public void Event_TankSpawn(Event event, const char[] name, bool dontBroadcast)
{
    int tank = GetClientOfUserId(event.GetInt("userid"));
    if (tank > 0 && IsClientInGame(tank) && GetClientTeam(tank) == 3)
    {
        int health = GetEntProp(tank, Prop_Send, "m_iHealth");
        PrintToChatAll("\x04[WARNING] \x01A Tank has spawned! Health: %d", health);
    }
}
