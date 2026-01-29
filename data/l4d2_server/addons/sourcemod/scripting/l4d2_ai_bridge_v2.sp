/**
 * L4D2 AI Bridge v2.0
 *
 * Enhanced bridge for AI-controlled bots and director.
 * Communicates with Python via TCP socket using simple JSON protocol.
 *
 * Requirements:
 *   - SourceMod 1.11+
 *   - Socket extension (https://forums.alliedmods.net/showthread.php?t=67640)
 *
 * Commands:
 *   sm_ai_connect [host] [port] - Connect to Python AI server
 *   sm_ai_disconnect           - Disconnect from Python
 *   sm_ai_director             - Toggle AI director control
 *   sm_ai_status               - Show connection status
 *   sm_ai_debug                - Toggle debug output
 */

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>
#include <socket>

#pragma newdecls required
#pragma semicolon 1

#define PLUGIN_VERSION "2.0.0"
#define BUFFER_SIZE 8192
#define MAX_SURVIVORS 4
#define UPDATE_RATE 10  // Hz

// Plugin info
public Plugin myinfo = {
    name = "L4D2 AI Bridge",
    author = "L4D2-AI-Architect",
    description = "Bridge for AI-controlled bots and director",
    version = PLUGIN_VERSION,
    url = "https://github.com/RazonIn4K/L4D2-AI-Architect"
};

// Global state
Handle g_hSocket = null;
Handle g_hStateTimer = null;
bool g_bConnected = false;
bool g_bDirectorEnabled = true;
bool g_bDebug = false;

// Bot control state
int g_iBotActions[MAXPLAYERS + 1];  // Pending actions per bot
float g_fBotMoveDir[MAXPLAYERS + 1][3];  // Movement direction

// Statistics
int g_iMessagesSent = 0;
int g_iMessagesReceived = 0;
int g_iActionsExecuted = 0;
float g_fConnectTime = 0.0;

// Action constants (must match Python)
enum BotAction {
    ACTION_IDLE = 0,
    ACTION_MOVE_FORWARD = 1,
    ACTION_MOVE_BACKWARD = 2,
    ACTION_MOVE_LEFT = 3,
    ACTION_MOVE_RIGHT = 4,
    ACTION_ATTACK = 5,
    ACTION_USE = 6,
    ACTION_RELOAD = 7,
    ACTION_CROUCH = 8,
    ACTION_JUMP = 9,
    ACTION_SHOVE = 10,
    ACTION_HEAL_SELF = 11,
    ACTION_HEAL_OTHER = 12,
    ACTION_THROW_ITEM = 13
}

// ============================================================================
// PLUGIN LIFECYCLE
// ============================================================================

public void OnPluginStart() {
    // Register commands
    RegAdminCmd("sm_ai_connect", Cmd_Connect, ADMFLAG_ROOT, "Connect to Python AI server");
    RegAdminCmd("sm_ai_disconnect", Cmd_Disconnect, ADMFLAG_ROOT, "Disconnect from Python");
    RegAdminCmd("sm_ai_director", Cmd_ToggleDirector, ADMFLAG_ROOT, "Toggle AI director");
    RegAdminCmd("sm_ai_status", Cmd_Status, ADMFLAG_ROOT, "Show connection status");
    RegAdminCmd("sm_ai_debug", Cmd_Debug, ADMFLAG_ROOT, "Toggle debug output");

    // Hook game events
    HookEvent("round_start", Event_RoundStart);
    HookEvent("round_end", Event_RoundEnd);
    HookEvent("player_death", Event_PlayerDeath);
    HookEvent("player_incapacitated", Event_PlayerIncap);
    HookEvent("revive_success", Event_ReviveSuccess);
    HookEvent("heal_success", Event_HealSuccess);
    HookEvent("infected_death", Event_InfectedDeath);
    HookEvent("witch_spawn", Event_WitchSpawn);
    HookEvent("tank_spawn", Event_TankSpawn);

    // Start state timer
    g_hStateTimer = CreateTimer(1.0 / UPDATE_RATE, Timer_SendState, _, TIMER_REPEAT);

    PrintToServer("[L4D2-AI] Plugin v%s loaded", PLUGIN_VERSION);
}

public void OnPluginEnd() {
    DisconnectFromPython();

    if (g_hStateTimer != null) {
        KillTimer(g_hStateTimer);
        g_hStateTimer = null;
    }

    PrintToServer("[L4D2-AI] Plugin unloaded");
}

public void OnClientPutInServer(int client) {
    g_iBotActions[client] = ACTION_IDLE;
    g_fBotMoveDir[client] = view_as<float>({0.0, 0.0, 0.0});

    // Hook bot movement
    if (IsFakeClient(client)) {
        SDKHook(client, SDKHook_PreThink, Hook_BotPreThink);
    }
}

public void OnClientDisconnect(int client) {
    SDKUnhook(client, SDKHook_PreThink, Hook_BotPreThink);
}

// ============================================================================
// COMMANDS
// ============================================================================

public Action Cmd_Connect(int client, int args) {
    char host[64] = "127.0.0.1";
    int port = 27050;

    if (args >= 1) {
        GetCmdArg(1, host, sizeof(host));
    }
    if (args >= 2) {
        char portStr[16];
        GetCmdArg(2, portStr, sizeof(portStr));
        port = StringToInt(portStr);
    }

    ConnectToPython(host, port);
    ReplyToCommand(client, "[L4D2-AI] Connecting to %s:%d...", host, port);
    return Plugin_Handled;
}

public Action Cmd_Disconnect(int client, int args) {
    DisconnectFromPython();
    ReplyToCommand(client, "[L4D2-AI] Disconnected from Python");
    return Plugin_Handled;
}

public Action Cmd_ToggleDirector(int client, int args) {
    g_bDirectorEnabled = !g_bDirectorEnabled;
    ReplyToCommand(client, "[L4D2-AI] Director %s", g_bDirectorEnabled ? "ENABLED" : "DISABLED");
    return Plugin_Handled;
}

public Action Cmd_Status(int client, int args) {
    float uptime = g_bConnected ? (GetGameTime() - g_fConnectTime) : 0.0;

    ReplyToCommand(client, "=== L4D2 AI Bridge Status ===");
    ReplyToCommand(client, "Connected: %s", g_bConnected ? "YES" : "NO");
    ReplyToCommand(client, "Director: %s", g_bDirectorEnabled ? "ENABLED" : "DISABLED");
    ReplyToCommand(client, "Uptime: %.1f seconds", uptime);
    ReplyToCommand(client, "Messages Sent: %d", g_iMessagesSent);
    ReplyToCommand(client, "Messages Received: %d", g_iMessagesReceived);
    ReplyToCommand(client, "Actions Executed: %d", g_iActionsExecuted);
    return Plugin_Handled;
}

public Action Cmd_Debug(int client, int args) {
    g_bDebug = !g_bDebug;
    ReplyToCommand(client, "[L4D2-AI] Debug %s", g_bDebug ? "ON" : "OFF");
    return Plugin_Handled;
}

// ============================================================================
// SOCKET MANAGEMENT
// ============================================================================

void ConnectToPython(const char[] host, int port) {
    DisconnectFromPython();

    g_hSocket = SocketCreate(SOCKET_TCP, OnSocketError);
    SocketConnect(g_hSocket, OnSocketConnected, OnSocketReceive, OnSocketDisconnected, host, port);
}

void DisconnectFromPython() {
    if (g_hSocket != null) {
        SocketDisconnect(g_hSocket);
        CloseHandle(g_hSocket);
        g_hSocket = null;
    }
    g_bConnected = false;
}

public void OnSocketConnected(Handle socket, any arg) {
    g_bConnected = true;
    g_fConnectTime = GetGameTime();
    g_iMessagesSent = 0;
    g_iMessagesReceived = 0;
    g_iActionsExecuted = 0;

    PrintToServer("[L4D2-AI] Connected to Python!");

    // Send handshake
    char handshake[256];
    Format(handshake, sizeof(handshake),
        "{\"type\":\"handshake\",\"version\":\"%s\",\"map\":\"%s\"}\n",
        PLUGIN_VERSION, GetCurrentMap());
    SocketSend(socket, handshake, strlen(handshake));
    g_iMessagesSent++;
}

public void OnSocketReceive(Handle socket, char[] data, const int size, any arg) {
    g_iMessagesReceived++;

    if (g_bDebug) {
        PrintToServer("[L4D2-AI] Received: %s", data);
    }

    // Parse simple JSON commands
    // Format: {"type":"bot_action","bot":1,"action":5}
    // Format: {"type":"director","cmd":"spawn_common","count":10}

    if (StrContains(data, "\"type\":\"bot_action\"") != -1) {
        ParseBotAction(data);
    } else if (StrContains(data, "\"type\":\"director\"") != -1) {
        ParseDirectorCommand(data);
    } else if (StrContains(data, "\"type\":\"reset\"") != -1) {
        ResetEpisode();
    }
}

public void OnSocketDisconnected(Handle socket, any arg) {
    g_bConnected = false;
    PrintToServer("[L4D2-AI] Disconnected from Python");
}

public void OnSocketError(Handle socket, const int errorType, const int errorNum, any arg) {
    g_bConnected = false;
    PrintToServer("[L4D2-AI] Socket error: type=%d num=%d", errorType, errorNum);
}

// ============================================================================
// STATE TRANSMISSION
// ============================================================================

public Action Timer_SendState(Handle timer) {
    if (!g_bConnected || g_hSocket == null) {
        return Plugin_Continue;
    }

    char buffer[BUFFER_SIZE];
    BuildGameState(buffer, sizeof(buffer));

    SocketSend(g_hSocket, buffer, strlen(buffer));
    g_iMessagesSent++;

    return Plugin_Continue;
}

void BuildGameState(char[] buffer, int maxlen) {
    // Start JSON object
    Format(buffer, maxlen, "{\"type\":\"state\",\"time\":%.2f,\"survivors\":[", GetGameTime());

    // Add survivor data
    int survivorCount = 0;
    for (int i = 1; i <= MaxClients; i++) {
        if (IsClientInGame(i) && GetClientTeam(i) == 2) {  // Team 2 = Survivors
            if (survivorCount > 0) {
                StrCat(buffer, maxlen, ",");
            }

            // Get survivor state
            float pos[3], ang[3];
            GetClientAbsOrigin(i, pos);
            GetClientAbsAngles(i, ang);

            int health = IsPlayerAlive(i) ? GetClientHealth(i) : 0;
            bool alive = IsPlayerAlive(i);
            bool incapped = GetEntProp(i, Prop_Send, "m_isIncapacitated") == 1;
            bool isBot = IsFakeClient(i);

            char weapon[32];
            GetClientWeapon(i, weapon, sizeof(weapon));

            char survivorData[512];
            Format(survivorData, sizeof(survivorData),
                "{\"id\":%d,\"health\":%d,\"alive\":%s,\"incapped\":%s,\"bot\":%s,"
                ..."\"pos\":[%.1f,%.1f,%.1f],\"ang\":[%.1f,%.1f],\"weapon\":\"%s\"}",
                i, health,
                alive ? "true" : "false",
                incapped ? "true" : "false",
                isBot ? "true" : "false",
                pos[0], pos[1], pos[2],
                ang[0], ang[1],
                weapon);

            StrCat(buffer, maxlen, survivorData);
            survivorCount++;
        }
    }

    StrCat(buffer, maxlen, "],");

    // Add infected counts
    int common = CountCommonInfected();
    int witches = CountWitches();
    int tanks = CountTanks();

    char infectedData[256];
    Format(infectedData, sizeof(infectedData),
        "\"infected\":{\"common\":%d,\"witches\":%d,\"tanks\":%d},",
        common, witches, tanks);
    StrCat(buffer, maxlen, infectedData);

    // Add director state
    char directorData[128];
    Format(directorData, sizeof(directorData),
        "\"director\":{\"enabled\":%s}}\n",
        g_bDirectorEnabled ? "true" : "false");
    StrCat(buffer, maxlen, directorData);
}

int CountCommonInfected() {
    int count = 0;
    int entity = -1;
    while ((entity = FindEntityByClassname(entity, "infected")) != -1) {
        count++;
    }
    return count;
}

int CountWitches() {
    int count = 0;
    int entity = -1;
    while ((entity = FindEntityByClassname(entity, "witch")) != -1) {
        count++;
    }
    return count;
}

int CountTanks() {
    int count = 0;
    for (int i = 1; i <= MaxClients; i++) {
        if (IsClientInGame(i) && GetClientTeam(i) == 3) {
            int class = GetEntProp(i, Prop_Send, "m_zombieClass");
            if (class == 8) {  // Tank
                count++;
            }
        }
    }
    return count;
}

// ============================================================================
// COMMAND PARSING
// ============================================================================

void ParseBotAction(const char[] data) {
    // Extract bot ID and action from JSON
    // Format: {"type":"bot_action","bot":1,"action":5}

    int botId = ExtractInt(data, "\"bot\":");
    int action = ExtractInt(data, "\"action\":");

    if (botId > 0 && botId <= MaxClients && IsClientInGame(botId)) {
        g_iBotActions[botId] = action;
        g_iActionsExecuted++;

        if (g_bDebug) {
            PrintToServer("[L4D2-AI] Bot %d action: %d", botId, action);
        }
    }
}

void ParseDirectorCommand(const char[] data) {
    if (!g_bDirectorEnabled) {
        return;
    }

    // Extract command type
    // Format: {"type":"director","cmd":"spawn_common","count":10}

    if (StrContains(data, "\"cmd\":\"spawn_common\"") != -1) {
        int count = ExtractInt(data, "\"count\":");
        if (count <= 0) count = 5;
        SpawnCommonInfected(count);
    }
    else if (StrContains(data, "\"cmd\":\"spawn_witch\"") != -1) {
        SpawnWitch();
    }
    else if (StrContains(data, "\"cmd\":\"spawn_tank\"") != -1) {
        SpawnTank();
    }
    else if (StrContains(data, "\"cmd\":\"panic\"") != -1) {
        TriggerPanicEvent();
    }

    g_iActionsExecuted++;
}

int ExtractInt(const char[] data, const char[] key) {
    int pos = StrContains(data, key);
    if (pos == -1) return 0;

    pos += strlen(key);
    char numStr[16];
    int numPos = 0;

    while (data[pos] != '\0' && numPos < 15) {
        if (data[pos] >= '0' && data[pos] <= '9') {
            numStr[numPos++] = data[pos];
        } else if (numPos > 0) {
            break;
        }
        pos++;
    }
    numStr[numPos] = '\0';

    return StringToInt(numStr);
}

// ============================================================================
// BOT CONTROL
// ============================================================================

public void Hook_BotPreThink(int client) {
    if (!g_bConnected || !IsFakeClient(client) || !IsPlayerAlive(client)) {
        return;
    }

    int action = g_iBotActions[client];
    if (action == ACTION_IDLE) {
        return;
    }

    // Execute action
    switch (action) {
        case ACTION_MOVE_FORWARD, ACTION_MOVE_BACKWARD, ACTION_MOVE_LEFT, ACTION_MOVE_RIGHT: {
            ExecuteMovement(client, action);
        }
        case ACTION_ATTACK: {
            ExecuteAttack(client);
        }
        case ACTION_USE: {
            ClientCommand(client, "+use");
            CreateTimer(0.1, Timer_ReleaseUse, client);
        }
        case ACTION_RELOAD: {
            ClientCommand(client, "+reload");
            CreateTimer(0.1, Timer_ReleaseReload, client);
        }
        case ACTION_JUMP: {
            ExecuteJump(client);
        }
        case ACTION_SHOVE: {
            ClientCommand(client, "+attack2");
            CreateTimer(0.1, Timer_ReleaseAttack2, client);
        }
        case ACTION_HEAL_SELF: {
            UseHealItem(client, client);
        }
        case ACTION_HEAL_OTHER: {
            UseHealOnTeammate(client);
        }
    }

    // Clear action after execution
    g_iBotActions[client] = ACTION_IDLE;
}

void ExecuteMovement(int client, int action) {
    float velocity[3];
    float speed = 220.0;  // Normal move speed

    float ang[3];
    GetClientAbsAngles(client, ang);

    float forward[3], right[3];
    GetAngleVectors(ang, forward, right, NULL_VECTOR);

    switch (action) {
        case ACTION_MOVE_FORWARD: {
            velocity[0] = forward[0] * speed;
            velocity[1] = forward[1] * speed;
        }
        case ACTION_MOVE_BACKWARD: {
            velocity[0] = -forward[0] * speed;
            velocity[1] = -forward[1] * speed;
        }
        case ACTION_MOVE_LEFT: {
            velocity[0] = -right[0] * speed;
            velocity[1] = -right[1] * speed;
        }
        case ACTION_MOVE_RIGHT: {
            velocity[0] = right[0] * speed;
            velocity[1] = right[1] * speed;
        }
    }

    TeleportEntity(client, NULL_VECTOR, NULL_VECTOR, velocity);
}

void ExecuteAttack(int client) {
    ClientCommand(client, "+attack");
    CreateTimer(0.1, Timer_ReleaseAttack, client);
}

void ExecuteJump(int client) {
    float velocity[3];
    GetEntPropVector(client, Prop_Data, "m_vecVelocity", velocity);

    if (GetEntityFlags(client) & FL_ONGROUND) {
        velocity[2] = 300.0;
        TeleportEntity(client, NULL_VECTOR, NULL_VECTOR, velocity);
    }
}

void UseHealItem(int client, int target) {
    // Switch to medkit and use
    ClientCommand(client, "slot3");
    CreateTimer(0.5, Timer_UseHeal, target);
}

void UseHealOnTeammate(int client) {
    // Find nearest alive teammate
    float myPos[3];
    GetClientAbsOrigin(client, myPos);

    int nearestTarget = -1;
    float nearestDist = 9999999.0;

    for (int i = 1; i <= MaxClients; i++) {
        if (i != client && IsClientInGame(i) && GetClientTeam(i) == 2 && IsPlayerAlive(i)) {
            float theirPos[3];
            GetClientAbsOrigin(i, theirPos);
            float dist = GetVectorDistance(myPos, theirPos);

            if (dist < nearestDist && dist < 100.0) {  // Within 100 units
                nearestDist = dist;
                nearestTarget = i;
            }
        }
    }

    if (nearestTarget > 0) {
        UseHealItem(client, nearestTarget);
    }
}

// Timer callbacks for releasing buttons
public Action Timer_ReleaseAttack(Handle timer, int client) {
    if (IsClientInGame(client)) ClientCommand(client, "-attack");
    return Plugin_Stop;
}

public Action Timer_ReleaseAttack2(Handle timer, int client) {
    if (IsClientInGame(client)) ClientCommand(client, "-attack2");
    return Plugin_Stop;
}

public Action Timer_ReleaseUse(Handle timer, int client) {
    if (IsClientInGame(client)) ClientCommand(client, "-use");
    return Plugin_Stop;
}

public Action Timer_ReleaseReload(Handle timer, int client) {
    if (IsClientInGame(client)) ClientCommand(client, "-reload");
    return Plugin_Stop;
}

public Action Timer_UseHeal(Handle timer, int client) {
    if (IsClientInGame(client)) ClientCommand(client, "+attack");
    CreateTimer(0.1, Timer_ReleaseAttack, client);
    return Plugin_Stop;
}

// ============================================================================
// DIRECTOR COMMANDS
// ============================================================================

void SpawnCommonInfected(int count) {
    // Find a spawn point near survivors
    float spawnPos[3];
    if (!GetSpawnPosition(spawnPos)) {
        return;
    }

    for (int i = 0; i < count; i++) {
        int infected = CreateEntityByName("infected");
        if (infected != -1) {
            // Add some randomness to position
            float pos[3];
            pos[0] = spawnPos[0] + GetRandomFloat(-100.0, 100.0);
            pos[1] = spawnPos[1] + GetRandomFloat(-100.0, 100.0);
            pos[2] = spawnPos[2];

            TeleportEntity(infected, pos, NULL_VECTOR, NULL_VECTOR);
            DispatchSpawn(infected);
            ActivateEntity(infected);
        }
    }

    if (g_bDebug) {
        PrintToServer("[L4D2-AI] Spawned %d common infected", count);
    }
}

void SpawnWitch() {
    float spawnPos[3];
    if (!GetSpawnPosition(spawnPos)) {
        return;
    }

    int witch = CreateEntityByName("witch");
    if (witch != -1) {
        TeleportEntity(witch, spawnPos, NULL_VECTOR, NULL_VECTOR);
        DispatchSpawn(witch);
        ActivateEntity(witch);

        if (g_bDebug) {
            PrintToServer("[L4D2-AI] Spawned witch");
        }
    }
}

void SpawnTank() {
    // Use director command to spawn tank
    int flags = GetCommandFlags("z_spawn_old");
    SetCommandFlags("z_spawn_old", flags & ~FCVAR_CHEAT);
    ServerCommand("z_spawn_old tank auto");
    SetCommandFlags("z_spawn_old", flags);

    if (g_bDebug) {
        PrintToServer("[L4D2-AI] Spawned tank");
    }
}

void TriggerPanicEvent() {
    int flags = GetCommandFlags("director_force_panic_event");
    SetCommandFlags("director_force_panic_event", flags & ~FCVAR_CHEAT);
    ServerCommand("director_force_panic_event");
    SetCommandFlags("director_force_panic_event", flags);

    if (g_bDebug) {
        PrintToServer("[L4D2-AI] Triggered panic event");
    }
}

void ResetEpisode() {
    ServerCommand("mp_restartgame 1");

    if (g_bDebug) {
        PrintToServer("[L4D2-AI] Episode reset");
    }
}

bool GetSpawnPosition(float pos[3]) {
    // Find a position away from but near survivors
    int survivor = GetRandomSurvivor();
    if (survivor <= 0) {
        return false;
    }

    float survivorPos[3], survivorAng[3];
    GetClientAbsOrigin(survivor, survivorPos);
    GetClientAbsAngles(survivor, survivorAng);

    // Spawn behind/beside survivor
    float forward[3];
    GetAngleVectors(survivorAng, forward, NULL_VECTOR, NULL_VECTOR);

    pos[0] = survivorPos[0] - forward[0] * 500.0 + GetRandomFloat(-200.0, 200.0);
    pos[1] = survivorPos[1] - forward[1] * 500.0 + GetRandomFloat(-200.0, 200.0);
    pos[2] = survivorPos[2];

    return true;
}

int GetRandomSurvivor() {
    int survivors[MAX_SURVIVORS];
    int count = 0;

    for (int i = 1; i <= MaxClients; i++) {
        if (IsClientInGame(i) && GetClientTeam(i) == 2 && IsPlayerAlive(i)) {
            survivors[count++] = i;
        }
    }

    if (count == 0) {
        return -1;
    }

    return survivors[GetRandomInt(0, count - 1)];
}

// ============================================================================
// EVENT HANDLERS
// ============================================================================

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast) {
    if (g_bConnected) {
        SendEvent("round_start", "{}");
    }
}

public void Event_RoundEnd(Event event, const char[] name, bool dontBroadcast) {
    if (g_bConnected) {
        SendEvent("round_end", "{}");
    }
}

public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast) {
    if (g_bConnected) {
        int victim = GetClientOfUserId(event.GetInt("userid"));
        char data[64];
        Format(data, sizeof(data), "{\"victim\":%d,\"team\":%d}", victim, GetClientTeam(victim));
        SendEvent("player_death", data);
    }
}

public void Event_PlayerIncap(Event event, const char[] name, bool dontBroadcast) {
    if (g_bConnected) {
        int victim = GetClientOfUserId(event.GetInt("userid"));
        char data[64];
        Format(data, sizeof(data), "{\"victim\":%d}", victim);
        SendEvent("player_incap", data);
    }
}

public void Event_ReviveSuccess(Event event, const char[] name, bool dontBroadcast) {
    if (g_bConnected) {
        int subject = GetClientOfUserId(event.GetInt("subject"));
        int helper = GetClientOfUserId(event.GetInt("userid"));
        char data[64];
        Format(data, sizeof(data), "{\"subject\":%d,\"helper\":%d}", subject, helper);
        SendEvent("revive", data);
    }
}

public void Event_HealSuccess(Event event, const char[] name, bool dontBroadcast) {
    if (g_bConnected) {
        int subject = GetClientOfUserId(event.GetInt("subject"));
        int healer = GetClientOfUserId(event.GetInt("userid"));
        char data[64];
        Format(data, sizeof(data), "{\"subject\":%d,\"healer\":%d}", subject, healer);
        SendEvent("heal", data);
    }
}

public void Event_InfectedDeath(Event event, const char[] name, bool dontBroadcast) {
    if (g_bConnected) {
        int attacker = GetClientOfUserId(event.GetInt("attacker"));
        char data[64];
        Format(data, sizeof(data), "{\"attacker\":%d}", attacker);
        SendEvent("infected_death", data);
    }
}

public void Event_WitchSpawn(Event event, const char[] name, bool dontBroadcast) {
    if (g_bConnected) {
        SendEvent("witch_spawn", "{}");
    }
}

public void Event_TankSpawn(Event event, const char[] name, bool dontBroadcast) {
    if (g_bConnected) {
        SendEvent("tank_spawn", "{}");
    }
}

void SendEvent(const char[] eventType, const char[] data) {
    if (g_hSocket == null) {
        return;
    }

    char buffer[512];
    Format(buffer, sizeof(buffer), "{\"type\":\"event\",\"event\":\"%s\",\"data\":%s}\n", eventType, data);
    SocketSend(g_hSocket, buffer, strlen(buffer));
    g_iMessagesSent++;
}

// ============================================================================
// UTILITIES
// ============================================================================

char[] GetCurrentMap() {
    static char mapName[64];
    GetCurrentMap(mapName, sizeof(mapName));
    return mapName;
}
