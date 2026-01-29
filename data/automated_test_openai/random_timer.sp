#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>

public Plugin myinfo =
{
    name = "Random Witch Spawner",
    author = "ChatGPT",
    description = "Spawns a Witch at random intervals",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    // Schedule first witch spawn
    float randomTime = GetRandomFloat(30.0, 60.0);
    CreateTimer(randomTime, Timer_SpawnWitch, _, TIMER_FLAG_NO_MAPCHANGE);
}

public void OnMapEnd()
{
    // Clear all timers on map change
    KillAllTimers();
}

public Action Timer_SpawnWitch(Handle timer)
{
    // Spawn witch
    if (GetCurrentGameState() != GAMESTATE_PLAYING)
        return Plugin_Continue;
    
    float spawnX, spawnY, spawnZ;
    if (!GetRandomSpawnPosition(spawnX, spawnY, spawnZ))
    {
        // If no valid position, try again later
        float randomTime = GetRandomFloat(30.0, 60.0);
        CreateTimer(randomTime, Timer_SpawnWitch, _, TIMER_FLAG_NO_MAPCHANGE);
        return Plugin_Continue;
    }
    
    // Spawn witch using L4D2 API
    char model[PLATFORM_MAX_PATH];
    Format(model, sizeof(model), "models/witch.mdl");
    
    int witch = CreateEntityByName("witch");
    if (witch == INVALID_ENT_REFERENCE)
    {
        PrintToChatAll("Failed to create witch entity!");
        return Plugin_Continue;
    }
    
    DispatchKeyValue(witch, "model", model);
    DispatchKeyValueFloat(witch, "origin", spawnX, spawnY, spawnZ);
    DispatchSpawn(witch);
    
    // Schedule next witch spawn
    float randomTime = GetRandomFloat(30.0, 60.0);
    CreateTimer(randomTime, Timer_SpawnWitch, _, TIMER_FLAG_NO_MAPCHANGE);
    
    return Plugin_Continue;
}

/**
 * Get a random valid spawn position away from players
 * @param spawnX Output random X
 * @param spawnY Output random Y
 * @param spawnZ Output random Z
 * @return True if valid position found
 */
bool GetRandomSpawnPosition(float &spawnX, float &spawnY, float &spawnZ)
{
    int playerCount = GetClientCount();
    if (playerCount < 1)
        return false;
    
    // Get average player position
    float avgX = 0.0, avgY = 0.0, avgZ = 0.0;
    int validPlayers = 0;
    for (int i = 1; i <= MaxClients; i++)
    {
        if (!IsClientInGame(i) || !IsPlayerAlive(i))
            continue;
        
        float pos[3];
        GetClientAbsOrigin(i, pos);
        avgX += pos[0];
        avgY += pos[1];
        avgZ += pos[2];
        validPlayers++;
    }
    
    if (validPlayers == 0)
        return false;
    
    avgX /= validPlayers;
    avgY /= validPlayers;
    avgZ /= validPlayers;
    
    // Generate random offset
    float offset = GetRandomFloat(500.0, 1000.0);
    float angle = GetRandomFloat(0.0, 360.0);
    float offsetX = offset * Cosine(angle);
    float offsetY = offset * Sine(angle);
    
    spawnX = avgX + offsetX;
    spawnY = avgY + offsetY;
    spawnZ = avgZ; // Keep same Z for simplicity
    
    // Check if position is valid
    char buffer[512];
    TraceLine(spawnX, spawnY, spawnZ + 50.0, spawnX, spawnY, spawnZ - 50.0, MASK_SOLID, TRACE_IGNORE_NODRAW_OPAQUE, _, buffer);
    float hitPos[3];
    GetVectorFromString(buffer, hitPos);
    
    if (GetVectorDistanceToVector(spawnX, spawnY, spawnZ, hitPos) < 50.0)
        return false;
    
    return true;
}

/**
 * Get the current game state
 * @return GameState enum value
 */
GameState GetCurrentGameState()
{
    int state = GameRules_GetProp("m_nGameState");
    return (GameState)state;
}

/**
 * Get the distance between two 3D vectors
 * @param x1 X1
 * @param y1 Y1
 * @param z1 Z1
 * @param x2 X2
 * @param y2 Y2
 * @param z2 Z2
 * @return Distance in units
 */
float GetVectorDistance(float x1, float y1, float z1, float x2, float y2, float z2)
{
    float dx = x2 - x1;
    float dy = y2 - y1;
    float dz = z2 - z1;
    return SquareRoot(dx * dx + dy * dy + dz * dz);
}

/**
 * Get distance between vector and point
 * @param x X coordinate
 * @param y Y coordinate
 * @param z Z coordinate
 * @param point Vector point
 * @return Distance in units
 */
float GetVectorDistanceToVector(float x, float y, float z, float point[3])
{
    return GetVectorDistance(x, y, z, point[0], point[1], point[2]);
}

/**
 * Get distance between two entities
 * @param entity1 Entity index 1
 * @param entity2 Entity index 2
 * @return Distance in units
 */
float GetEntityDistance(int entity1, int entity2)
{
    float pos1[3], pos2[3];
    GetEntPropVector(entity1, Prop_Send, "m_vecOrigin", pos1);
    GetEntPropVector(entity2, Prop_Send, "m_vecOrigin", pos2);
    return GetVectorDistance(pos1[0], pos1[1], pos1[2], pos2[0], pos2[1], pos2[2]);
}
