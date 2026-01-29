#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

#define SPAWN_INTERVAL 300.0  // 5 minutes
#define SPAWN_RADIUS 1500.0

Handle g_hSpawnTimer = null;

public Plugin myinfo =
{
    name = "Witch Spawner",
    author = "Developer",
    description = "Spawns witches periodically near survivors",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("round_start", Event_RoundStart);
    HookEvent("round_end", Event_RoundEnd);
    
    RegConsoleCmd("sm_spawnwitch", Cmd_SpawnWitch);
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
    if (g_hSpawnTimer != null)
    {
        KillTimer(g_hSpawnTimer);
        g_hSpawnTimer = null;
    }
    
    g_hSpawnTimer = CreateTimer(SPAWN_INTERVAL, Timer_SpawnWitch, _, TIMER_REPEAT);
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
    if (client == 0 || !IsClientInGame(client))
    {
        ReplyToCommand(client, "[SM] This command can only be used by players in game.");
        return Plugin_Handled;
    }
    
    SpawnWitchNearSurvivor(client);
    ReplyToCommand(client, "[SM] Witch spawned!");
    return Plugin_Handled;
}

public Action Timer_SpawnWitch(Handle timer)
{
    int survivor = FindRandomSurvivor();
    if (survivor != -1)
    {
        SpawnWitchNearSurvivor(survivor);
    }
    return Plugin_Continue;
}

void SpawnWitchNearSurvivor(int survivor)
{
    float pos[3];
    GetClientAbsOrigin(survivor, pos);
    
    // Calculate random offset
    float angle = GetRandomFloat(0.0, 6.28318);  // 0 to 2π radians
    float distance = GetRandomFloat(500.0, SPAWN_RADIUS);
    
    pos[0] += Cosine(angle) * distance;
    pos[1] += Sine(angle) * distance;
    pos[2] = GetGroundHeight(pos[0], pos[1]) + 10.0;
    
    // Spawn witch using z_spawn command
    int flags = GetCommandFlags("z_spawn_old");
    SetCommandFlags("z_spawn_old", flags & ~FCVAR_CHEAT);
    
    char cmd[64];
    Format(cmd, sizeof(cmd), "z_spawn_old witch %f %f %f", pos[0], pos[1], pos[2]);
    
    // Create a temporary bot at position to spawn witch
    int bot = CreateFakeClient("WitchSpawner");
    if (bot != 0)
    {
        float eyeAngles[3] = {0.0, 0.0, 0.0};
        TeleportEntity(bot, pos, eyeAngles, NULL_VECTOR);
        FakeClientCommand(bot, "z_spawn_old witch auto");
        KickClient(bot);
    }
    
    SetCommandFlags("z_spawn_old", flags);
    
    PrintToChatAll("\x04[Witch Spawner] \x01A witch has spawned!");
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
