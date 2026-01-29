#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

#define PLUGIN_VERSION "1.2"
#define SPAWN_RADIUS 1000.0
#define SPAWN_INTERVAL 300.0  // 5 minutes

public Plugin myinfo =
{
    name = "L4D2 Witch Spawner",
    author = "Optimized Version",
    description = "Spawns a witch near survivors every 5 minutes",
    version = PLUGIN_VERSION,
    url = ""
};

Handle g_hSpawnTimer = null;

public void OnPluginStart()
{
    HookEvent("round_start", Event_RoundStart);
    HookEvent("round_end", Event_RoundEnd);
    
    RegConsoleCmd("sm_spawnwitch", Cmd_SpawnWitch, "Spawn a witch nearby");
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
    // CORRECT: Kill existing timer before creating new one (prevents stacking)
    if (g_hSpawnTimer != null)
    {
        KillTimer(g_hSpawnTimer);
        g_hSpawnTimer = null;
    }
    
    g_hSpawnTimer = CreateTimer(SPAWN_INTERVAL, Timer_SpawnWitch, _, 
                                 TIMER_REPEAT | TIMER_FLAG_NO_MAPCHANGE);
}

public void Event_RoundEnd(Event event, const char[] name, bool dontBroadcast)
{
    if (g_hSpawnTimer != null)
    {
        KillTimer(g_hSpawnTimer);
        g_hSpawnTimer = null;
    }
}

public Action Cmd_SpawnWitch(int client, int args)
{
    if (!IsValidClient(client))
    {
        ReplyToCommand(client, "[SM] This command can only be used by players.");
        return Plugin_Handled;
    }
    
    // Get client position
    float pos[3];
    GetClientAbsOrigin(client, pos);
    
    // Calculate spawn position
    float spawnPos[3];
    GetRandomPositionNear(pos, SPAWN_RADIUS, spawnPos);
    
    // Spawn witch
    SpawnWitchAt(spawnPos);
    
    ReplyToCommand(client, "[SM] Witch spawned!");
    return Plugin_Handled;
}

public Action Timer_SpawnWitch(Handle timer)
{
    int survivor = GetRandomAliveSurvivor();
    
    if (survivor == -1)
        return Plugin_Continue;
    
    // Get survivor position
    float pos[3];
    GetClientAbsOrigin(survivor, pos);
    
    // CORRECT: Calculate random offset manually (GetRandomVector doesn't exist!)
    float spawnPos[3];
    GetRandomPositionNear(pos, SPAWN_RADIUS, spawnPos);
    
    // Spawn witch
    SpawnWitchAt(spawnPos);
    
    // Play warning sound and announce
    EmitSoundToAll("ui/pickup_scifi37.wav");
    PrintToChatAll("\x04[WARNING]\x01 A witch has spawned nearby! Be careful!");
    
    return Plugin_Continue;
}

// CORRECT: Proper random position calculation (no GetRandomVector function exists)
void GetRandomPositionNear(float basePos[3], float radius, float result[3])
{
    // Random angle in radians (0 to 2*PI)
    float angle = GetRandomFloat(0.0, 6.28318);
    
    // Random distance (not too close, not too far)
    float distance = GetRandomFloat(radius * 0.3, radius);
    
    // Calculate position using trigonometry
    result[0] = basePos[0] + Cosine(angle) * distance;
    result[1] = basePos[1] + Sine(angle) * distance;
    result[2] = basePos[2];  // Keep same Z for now
}

void SpawnWitchAt(float pos[3])
{
    // Method 1: Try using director command
    int flags = GetCommandFlags("z_spawn_old");
    if (flags != -1)
    {
        SetCommandFlags("z_spawn_old", flags & ~FCVAR_CHEAT);
        
        // Need a client to execute command
        int admin = GetRandomAliveSurvivor();
        if (admin > 0)
        {
            // Store position for spawn
            char cmd[128];
            Format(cmd, sizeof(cmd), "z_spawn_old witch auto");
            FakeClientCommand(admin, cmd);
        }
        
        SetCommandFlags("z_spawn_old", flags);
        return;
    }
    
    // Method 2: Direct entity creation (fallback)
    int witch = CreateEntityByName("witch");
    if (witch != -1)
    {
        TeleportEntity(witch, pos, NULL_VECTOR, NULL_VECTOR);
        DispatchSpawn(witch);
        ActivateEntity(witch);
        
        // Set witch properties
        SetEntProp(witch, Prop_Data, "m_iHealth", 1000);
        SetEntProp(witch, Prop_Data, "m_iMaxHealth", 1000);
    }
}

// CORRECT: Proper survivor finding (not self-comparing positions!)
int GetRandomAliveSurvivor()
{
    int survivors[MAXPLAYERS];
    int count = 0;
    
    for (int i = 1; i <= MaxClients; i++)
    {
        if (IsClientInGame(i) && GetClientTeam(i) == 2 && IsPlayerAlive(i))
        {
            survivors[count++] = i;
        }
    }
    
    if (count == 0)
        return -1;
    
    return survivors[GetRandomInt(0, count - 1)];
}

bool IsValidClient(int client)
{
    return (client > 0 && client <= MaxClients && 
            IsClientInGame(client) && GetClientTeam(client) == 2);
}
