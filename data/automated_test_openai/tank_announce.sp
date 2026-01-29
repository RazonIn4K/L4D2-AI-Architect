#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>

public Plugin myinfo =
{
    name = "Tank Spawn Announcer",
    author = "ChatGPT",
    description = "Announces when a Tank spawns",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    // Hook the tank_spawn event
    HookEvent("tank_spawn", Event_TankSpawn);
}

public void OnAllPluginsLoaded()
{
    // Precache sound
    PrecacheSound("ambient/levels/labs/electric_explosion1.wav");
}

public void Event_TankSpawn(Event event, const char[] name, bool dontBroadcast)
{
    // Get the tank's client index
    int tank = GetClientOfUserId(event.GetInt("userid"));
    
    // Check if valid player
    if (tank > 0 && tank <= MaxClients && IsClientInGame(tank) && GetClientTeam(tank) == 3)
    {
        // Get tank's name
        char tankName[64];
        GetClientName(tank, tankName, sizeof(tankName));
        
        // Format message
        char message[256];
        Format(message, sizeof(message), "The mighty Tank %s has entered the battle!", tankName);
        
        // Play sound to all players
        EmitSoundToAll("ambient/levels/labs/electric_explosion1.wav");
        
        // Send chat message
        PrintToChatAll(message);
    }
}
