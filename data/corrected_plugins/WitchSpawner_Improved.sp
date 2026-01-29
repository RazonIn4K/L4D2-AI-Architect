#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

#define SPAWN_INTERVAL 300.0  // 5 minutes
#define SPAWN_RADIUS 1500.0
#define MAX_WITCHES 10  // Maximum witches per round

Handle g_hSpawnTimer = null;
int g_iWitchCount = 0;

public Plugin myinfo =
{
    name = "Witch Spawner",
    author = "Developer",
    description = "Spawns witches periodically near survivors",
    version = "1.1",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("round_start", Event_RoundStart);
    HookEvent("round_end", Event_RoundEnd);
    HookEvent("witch_killed", Event_WitchKilled);
    
    RegConsoleCmd("sm_spawnwitch", Cmd_SpawnWitch);
    RegAdminCmd("sm_spawnwitch_force", Cmd_SpawnWitchForce, ADMFLAG_ROOT);
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
    if (g_hSpawnTimer != null)
    {
        KillTimer(g_hSpawnTimer);
        g_hSpawnTimer = null;
    }
    
    g_iWitchCount = 0;
    g_hSpawnTimer = CreateTimer(SPAWN_INTERVAL, Timer_SpawnWitch, _, TIMER_REPEAT);
}

public void Event_RoundEnd(Event event, const char[] name, bool dontBroadcast)
{
    if (g_hSpawnTimer != null)
    {
        KillTimer(g_hSpawnTimer);
        g_hSpawnTimer = null;
    }
    g_iWitchCount = 0;
}

public void Event_WitchKilled(Event event, const char[] name, bool dontBroadcast)
{
    g_iWitchCount--;
}

public Action Cmd_SpawnWitch(int client, int args)
{
    if (client == 0 || !IsClientInGame(client))
    {
        ReplyToCommand(client, "[SM] This command can only be used by players in game.");
        return Plugin_Handled;
    }
    
    if (g_iWitchCount >= MAX_WITCHES)
    {
        ReplyToCommand(client, "[SM] Maximum witches reached (%d)!", MAX_WITCHES);
        return Plugin_Handled;
    }
    
    if (SpawnWitchNearSurvivor(client))
    {
        ReplyToCommand(client, "[SM] Witch spawned!");
    }
    else
    {
        ReplyToCommand(client, "[SM] Failed to spawn witch!");
    }
    
    return Plugin_Handled;
}

public Action Cmd_SpawnWitchForce(int client, int args)
{
    if (SpawnWitchNearSurvivor(client))
    {
        ReplyToCommand(client, "[SM] Witch force spawned!");
    }
    else
    {
        ReplyToCommand(client, "[SM] Failed to spawn witch!");
    }
    
    return Plugin_Handled;
}

public Action Timer_SpawnWitch(Handle timer)
{
    if (g_iWitchCount >= MAX_WITCHES)
        return Plugin_Continue;
    
    int survivor = FindRandomSurvivor();
    if (survivor != -1)
    {
        if (SpawnWitchNearSurvivor(survivor))
        {
            g_iWitchCount++;
        }
    }
    return Plugin_Continue;
}

bool SpawnWitchNearSurvivor(int survivor)
{
    float pos[3];
    GetClientAbsOrigin(survivor, pos);
    
    // Calculate random offset
    float angle = GetRandomFloat(0.0, 6.28318);  // 0 to 2π radians
    float distance = GetRandomFloat(500.0, SPAWN_RADIUS);
    
    pos[0] += Cosine(angle) * distance;
    pos[1] += Sine(angle) * distance;
    pos[2] = GetGroundHeight(pos[0], pos[1]) + 10.0;
    
    // Method 1: Try using director to spawn witch
    if (SpawnWitchDirector(pos))
    {
        g_iWitchCount++;
        PrintToChatAll("\x04[Witch Spawner] \x01A witch has spawned!");
        return true;
    }
    
    // Method 2: Try using z_spawn command with admin flags
    if (SpawnWitchCommand(pos))
    {
        g_iWitchCount++;
        PrintToChatAll("\x04[Witch Spawner] \x01A witch has spawned!");
        return true;
    }
    
    // Method 3: Direct entity creation (last resort)
    if (SpawnWitchEntity(pos))
    {
        g_iWitchCount++;
        PrintToChatAll("\x04[Witch Spawner] \x01A witch has spawned!");
        return true;
    }
    
    return false;
}

bool SpawnWitchDirector(float pos[3])
{
    // Try to use the AI director to spawn a witch
    int director = CreateEntityByName("info_director");
    if (director == -1)
        return false;
    
    DispatchSpawn(director);
    TeleportEntity(director, pos, NULL_VECTOR, NULL_VECTOR);
    
    // Force witch spawn through director
    AcceptEntityInput(director, "BeginScript");
    AcceptEntityInput(director, "EndScript");
    
    // Create witch at position
    int witch = CreateEntityByName("witch");
    if (witch == -1)
    {
        AcceptEntityInput(director, "Kill");
        return false;
    }
    
    DispatchSpawn(witch);
    TeleportEntity(witch, pos, NULL_VECTOR, NULL_VECTOR);
    ActivateEntity(witch);
    
    AcceptEntityInput(director, "Kill");
    return true;
}

bool SpawnWitchCommand(float pos[3])
{
    // Try using z_spawn with command flags
    int flags = GetCommandFlags("z_spawn_old");
    if (flags == -1)
        return false;
    
    // Remove cheat flag temporarily
    SetCommandFlags("z_spawn_old", flags & ~FCVAR_CHEAT);
    
    // Execute command on server (not client)
    char cmd[128];
    Format(cmd, sizeof(cmd), "z_spawn_old witch %f %f %f", pos[0], pos[1], pos[2]);
    ServerCommand(cmd);
    
    // Restore flags
    SetCommandFlags("z_spawn_old", flags);
    
    return true;
}

bool SpawnWitchEntity(float pos[3])
{
    // Direct entity creation
    int witch = CreateEntityByName("witch");
    if (witch == -1)
        return false;
    
    DispatchKeyValueVector(witch, "origin", pos);
    DispatchSpawn(witch);
    ActivateEntity(witch);
    
    // Set witch properties
    SetEntProp(witch, Prop_Data, "m_iHealth", 1000);
    SetEntProp(witch, Prop_Data, "m_iMaxHealth", 1000);
    
    return true;
}

int FindRandomSurvivor()
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
    
    if (count > 0)
        return survivors[GetRandomInt(0, count - 1)];
    return -1;
}

float GetGroundHeight(float x, float y)
{
    float start[3], end[3];
    start[0] = x;
    start[1] = y;
    start[2] = 1000.0;
    
    end[0] = x;
    end[1] = y;
    end[2] = -1000.0;
    
    Handle trace = TR_TraceRayFilterEx(start, end, MASK_SOLID, RayType_EndPoint, TraceFilter_NoPlayers);
    float height;
    
    if (TR_DidHit(trace))
    {
        float pos[3];
        TR_GetEndPosition(pos, trace);
        height = pos[2];
    }
    else
    {
        height = 0.0;
    }
    
    delete trace;
    return height;
}

public bool TraceFilter_NoPlayers(int entity, int contentsMask)
{
    return !IsClient(entity);
}
