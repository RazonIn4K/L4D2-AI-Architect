#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

public Plugin myinfo =
{
    name = "Tank Spawn Announcer",
    author = "Developer",
    description = "Announces when a Tank spawns in the game",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    // Hook Tank spawn event - CORRECT event name is 'tank_spawn' (NOT 'tank_death')
    HookEvent("tank_spawn", Event_TankSpawn);
    
    PrintToServer("[Tank Announce] Plugin loaded and ready.");
}

// Function called when Tank spawns
void Event_TankSpawn(Event event, const char[] name, bool dontBroadcast)
{
    int tankEntity = GetClientOfUserId(event.GetInt("userid"));
    
    if (tankEntity > 0 && IsClientInGame(tankEntity))
    {
        // Get player's name
        char playerName[MAX_NAME_LENGTH];
        GetClientName(tankEntity, playerName, sizeof(playerName));
        
        // Broadcast announcement to all players
        char message[256];
        Format(message, sizeof(message), "\x04[Tank Spawn]\x01 %s has spawned as the Tank!", playerName);
        
        // Use Hint Box for clear announcement
        CreateHintText(message, HintType_CenterPrint, 3.0);
    }
}