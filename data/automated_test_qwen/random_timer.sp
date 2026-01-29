#pragma semicolon 1
#pragma newdecls required

#include <sdktools>
#include <left4dhooks>

#define SPAWN_MIN_TIME 30.0  // Minimum spawn interval in seconds
#define SPAWN_MAX_TIME 60.0  // Maximum spawn interval in seconds

public Plugin myinfo =
{
    name = "Witch Spawner",
    author = "Developer",
    description = "Spawns a witch at random intervals",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    RegConsoleCmd("sm_spawnwitch", Cmd_SpawnWitch);
}

public Action Cmd_SpawnWitch(int client)
{
    if (client == 0)  // Server console
    {
        SpawnWitchAtRandom();
        ReplyToCommand(client, "Witch spawned!");
    }
    else
    {
        ReplyToCommand(client, "This command can only be used by server.");
    }
    
    return Plugin_Handled;
}

void SpawnWitchAtRandom()
{
    // Calculate random interval
    float minTime = SPAWN_MIN_TIME;
    float maxTime = SPAWN_MAX_TIME;
    float interval = RandomFloat(minTime, maxTime);
    
    // Schedule next spawn
    CreateTimer(interval, Timer_SpawnWitch, _, TIMER_FLAG_NO_MAPCHANGE);
    
    // Spawn the witch immediately
    Do_SpawnWitch();
}

public Action Timer_SpawnWitch(Handle timer)
{
    Do_SpawnWitch();
    return Plugin_Continue;
}

void Do_SpawnWitch()
{
    // Method 1: Using left4dhooks extension (requires l4dhooks)
    // This is the recommended method if available
    L4DSpawnSpecialInfected(L4D2Infected_Witch);
    
    // Method 2: Using z_spawn command (fallback)
    // Note: May not work properly in all L4D2 versions or custom maps
    /*
    char cmd[64];
    Format(cmd, sizeof(cmd), "z_spawn witch");
    DispatchKeyValue(0, "command", cmd);
    */
}

// Alternative spawn method using entity manipulation (more advanced)
/*
void SpawnWitchEntity()
{
    int witch = CreateEntityByName("witch");
    if (witch != INVALID_ENT_REFERENCE)
    {
        TeleportEntity(witch, GetRandomPosition(), NULL_VECTOR, NULL_VECTOR);
        AcceptEntityInput(witch, "TurnOn");
    }
}
*/

// Helper to get a random position near survivors
float[] GetRandomPosition()
{
    static float pos[3];
    GetClientAbsOrigin(GetRandomSurvivorIndex(), pos);
    pos[2] += 50.0;  // Raise it slightly
    
    // Add some random offset
    float angle = RandomFloat(0.0, 6.28318);  // 0 to 2π radians
    float distance = RandomFloat(1000.0, 2000.0);
    pos[0] += Cosine(angle) * distance;
    pos[1] += Sine(angle) * distance;
    
    return pos;
}

// Helper to get a random survivor index
int GetRandomSurvivorIndex()
{
    int count = 0;
    for (int i = 1; i <= MaxClients; i++)
    {
        if (IsClientInGame(i) && GetClientTeam(i) == 2 && IsPlayerAlive(i))
        {
            count++;
        }
    }
    
    if (count == 0) return -1;
    
    int index = RandomInt(1, count);
    int found = 0;
    
    for (int i = 1; i <= MaxClients; i++)
    {
        if (IsClientInGame(i) && GetClientTeam(i)