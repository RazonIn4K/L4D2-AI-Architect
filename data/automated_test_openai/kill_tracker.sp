#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>

#define MAX_PLAYERS 100

// Plugin version
#define PLUGIN_VERSION "1.0"

// Data structure for tracking kills
struct KillData
{
    int totalKills;
    int[] specialKills;
};

// Global variables
KillData g_killData[MAX_PLAYERS];
int g_zombieModels[10];

// Plugin initialization
public Plugin myinfo =
{
    name = "Zombie Kill Tracker",
    author = "Developer",
    description = "Tracks zombie kills by players",
    version = PLUGIN_VERSION,
    url = ""
};

public void OnPluginStart()
{
    // Load models for zombie types
    g_zombieModels[0] = LoadEntityModel("models/zombie/zombie.mdl"); // Common
    g_zombieModels[1] = LoadEntityModel("models/zombie/hunter.mdl"); // Hunter
    g_zombieModels[2] = LoadEntityModel("models/zombie/smoker.mdl"); // Smoker
    g_zombieModels[3] = LoadEntityModel("models/zombie/boomer.mdl"); // Boomer
    g_zombieModels[4] = LoadEntityModel("models/zombie/charger.mdl"); // Charger
    g_zombieModels[5] = LoadEntityModel("models/zombie/spitter.mdl"); // Spitter
    g_zombieModels[6] = LoadEntityModel("models/zombie/jockey.mdl"); // Jockey
    g_zombieModels[7] = LoadEntityModel("models/zombie/witch.mdl"); // Witch
    g_zombieModels[8] = LoadEntityModel("models/zombie/riot.mdl"); // Riot
    g_zombieModels[9] = LoadEntityModel("models/zombie/flagellant.mdl"); // Flagellant
    
    // Initialize kill data
    for (int i = 0; i < MAX_PLAYERS; i++)
    {
        g_killData[i].totalKills = 0;
        g_killData[i].specialKills = new int[7]; // 6 specials + 1 common
    }
    
    // Hook events
    HookEvent("infected_death", Event_InfectedDeath);
    HookEvent("player_death", Event_PlayerDeath);
}

// Event: Infected death
public void Event_InfectedDeath(Event event, const char[] name, bool dontBroadcast)
{
    int attacker = GetClientOfUserId(event.GetInt("attacker"));
    int victim = GetClientOfUserId(event.GetInt("userid"));
    
    // Validate attacker
    if (!IsClientInGame(attacker) || GetClientTeam(attacker) != 2)
        return;
    
    // Validate victim
    if (!IsClientInGame(victim) || GetClientTeam(victim) != 3)
        return;
    
    // Track common zombie kills
    g_killData[attacker].totalKills++;
    
    // Track special zombie kills
    int modelIndex = GetEntityModel(victim);
    if (modelIndex != -1)
    {
        if (modelIndex == g_zombieModels[1]) // Hunter
            g_killData[attacker].specialKills[0]++;
        else if (modelIndex == g_zombieModels[2]) // Smoker
            g_killData[attacker].specialKills[1]++;
        else if (modelIndex == g_zombieModels[3]) // Boomer
            g_killData[attacker].specialKills[2]++;
        else if (modelIndex == g_zombieModels[4]) // Charger
            g_killData[attacker].specialKills[3]++;
        else if (modelIndex == g_zombieModels[5]) // Spitter
            g_killData[attacker].specialKills[4]++;
        else if (modelIndex == g_zombieModels[6]) // Jockey
            g_killData[attacker].specialKills[5]++;
        else // Common
            g_killData[attacker].specialKills[6]++;
    }
}

// Event: Player death (reset data)
public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast)
{
    int victim = GetClientOfUserId(event.GetInt("userid"));
    
    // Reset kill data on player death
    if (IsClientInGame(victim) && GetClientTeam(victim) == 2)
    {
        g_killData[victim].totalKills = 0;
        for (int i = 0; i < 7; i++)
            g_killData[victim].specialKills[i] = 0;
    }
}

// Get total kills for a player
public int GetTotalKills(int client)
{
    if (!IsClientInGame(client) || GetClientTeam(client) != 2)
        return 0;
    
    return g_killData[client].totalKills;
}

// Get special kills for a player
public int GetSpecialKills(int client, int type)
{
    if (!IsClientInGame(client) || GetClientTeam(client) != 2)
        return 0;
    
    return g_killData[client].specialKills[type];
}

// Get entity model index
int GetEntityModel(int entity)
{
    char model[PLATFORM_MAX_PATH];
    if (GetEntPropString(entity, Prop_Data, "m_ModelName", model, sizeof(model)) != -1)
    {
        for (int i = 0; i < 10; i++)
        {
            char zombieModel[PLATFORM_MAX_PATH];
            Format(zombieModel, sizeof(zombieModel), "models/zombie/%s", GetFileName(model));
            if (StrEqual(zombieModel, model))
                return i;
        }
    }
    return -1;
}

// Get file name from path
void GetFileName(const char[] path, char[] fileName)
{
    int lastSlash = -1;
    for (int i = 0; i < strlen(path); i++)
    {
        if (path[i] == '/' || path[i] == '\\')
            lastSlash = i;
    }
    if (lastSlash == -1)
        StrCopy(fileName, strlen(path), path);
    else
        StrCopy(fileName, strlen(path) - lastSlash, path + lastSlash + 1);
}