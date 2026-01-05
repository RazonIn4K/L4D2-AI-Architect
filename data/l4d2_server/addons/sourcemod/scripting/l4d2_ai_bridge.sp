#include <sourcemod>
#include <sdktools>
#include <socket>

#pragma newdecls required
#pragma semicolon 1

#define PLUGIN_VERSION "1.0.0"
#define BUFFER_SIZE 4096

public Plugin myinfo = {
    name = "L4D2 AI Bridge",
    author = "L4D2-AI-Architect",
    description = "Bridge for AI-controlled bots and director",
    version = PLUGIN_VERSION,
    url = "https://github.com/RazonIn4K/L4D2-AI-Architect"
};

// Global variables
Handle g_hSocket = null;
Handle g_hStateTimer = null;
Handle g_hDirectorTimer = null;
bool g_bDirectorEnabled = true;
int g_iUpdateRate = 10;  // Hz

// Game state structure
enum GameState {
    Float: gameTime,
    Float: survivorHealth[4],
    Float: survivorPos[4][3],
    Float: survivorAngle[4][3],
    infectedCount,
    specialInfected[5],  // smoker, boomer, hunter, spitter, jockey
    witchCount,
    tankCount,
    commonCount,
    panicActive,
    tankActive
}

// Director commands
enum DirectorCommand {
    spawnCommon,
    spawnSpecial,
    spawnWitch,
    spawnTank,
    triggerPanic,
    spawnItem
}

public void OnPluginStart() {
    // Create socket for Python communication
    g_hSocket = SocketCreate(SOCKET_TCP, OnSocketError);
    
    // Hook events
    HookEvent("round_start", Event_RoundStart);
    HookEvent("round_end", Event_RoundEnd);
    HookEvent("player_death", Event_PlayerDeath);
    HookEvent("infected_death", Event_InfectedDeath);
    HookEvent("witch_killed", Event_WitchKilled);
    HookEvent("tank_killed", Event_TankKilled);
    
    // Register commands
    RegAdminCmd("sm_ai_connect", Cmd_ConnectToPython, ADMFLAG_ROOT);
    RegAdminCmd("sm_ai_disconnect", Cmd_DisconnectFromPython, ADMFLAG_ROOT);
    RegAdminCmd("sm_ai_director", Cmd_ToggleDirector, ADMFLAG_ROOT);
    
    // Start state update timer
    g_hStateTimer = CreateTimer(1.0 / g_iUpdateRate, Timer_UpdateState, _, TIMER_REPEAT);
    
    // Start director timer
    g_hDirectorTimer = CreateTimer(1.0 / 10.0, Timer_DirectorUpdate, _, TIMER_REPEAT);
    
    PrintToServer("[L4D2-AI] Plugin loaded");
}

public void OnPluginEnd() {
    if (g_hSocket != null) {
        SocketClose(g_hSocket);
    }
    if (g_hStateTimer != null) {
        KillTimer(g_hStateTimer);
    }
    if (g_hDirectorTimer != null) {
        KillTimer(g_hDirectorTimer);
    }
    PrintToServer("[L4D2-AI] Plugin unloaded");
}

public Action Cmd_ConnectToPython(int client, int args) {
    if (g_hSocket != null) {
        SocketClose(g_hSocket);
    }
    
    char host[64];
    int port = 27050;
    
    if (args >= 1) {
        GetCmdArg(1, host, sizeof(host));
    } else {
        strcopy(host, sizeof(host), "127.0.0.1");
    }
    
    if (args >= 2) {
        char portStr[16];
        GetCmdArg(2, portStr, sizeof(portStr));
        port = StringToInt(portStr);
    }
    
    g_hSocket = SocketCreate(SOCKET_TCP, OnSocketError);
    SocketConnect(g_hSocket, OnSocketConnected, OnSocketReceive, OnSocketDisconnected, host, port);
    
    ReplyToCommand(client, "[L4D2-AI] Connecting to %s:%d...", host, port);
    return Plugin_Handled;
}

public Action Cmd_DisconnectFromPython(int client, int args) {
    if (g_hSocket != null) {
        SocketClose(g_hSocket);
        g_hSocket = null;
    }
    ReplyToCommand(client, "[L4D2-AI] Disconnected from Python");
    return Plugin_Handled;
}

public Action Cmd_ToggleDirector(int client, int args) {
    g_bDirectorEnabled = !g_bDirectorEnabled;
    ReplyToCommand(client, "[L4D2-AI] Director %s", g_bDirectorEnabled ? "enabled" : "disabled");
    return Plugin_Handled;
}

public void OnSocketConnected(Handle socket, any arg) {
    PrintToServer("[L4D2-AI] Connected to Python bridge");
    
    // Send initial handshake
    char handshake[256];
    Format(handshake, sizeof(handshake), "{\"type\":\"handshake\",\"version\":\"%s\"}", PLUGIN_VERSION);
    SocketSend(socket, handshake, strlen(handshake));
}

public void OnSocketReceive(Handle socket, char[] data, const int size, any arg) {
    // Parse JSON commands from Python
    Handle json = json_load(data);
    
    if (json == null) {
        LogError("[L4D2-AI] Invalid JSON received: %s", data);
        return;
    }
    
    char commandType[64];
    json_object_get_string(json, "type", commandType, sizeof(commandType));
    
    if (StrEqual(commandType, "bot_action")) {
        HandleBotCommand(json);
    } else if (StrEqual(commandType, "director_command")) {
        HandleDirectorCommand(json);
    } else if (StrEqual(commandType, "reset_episode")) {
        ResetEpisode();
    }
    
    CloseHandle(json);
}

public void OnSocketDisconnected(Handle socket, any arg) {
    PrintToServer("[L4D2-AI] Disconnected from Python bridge");
    g_hSocket = null;
}

public void OnSocketError(Handle socket, const int errorType, const int errorNum, const char[] errorStr) {
    LogError("[L4D2-AI] Socket error: %s", errorStr);
}

public Action Timer_UpdateState(Handle timer) {
    if (g_hSocket == null) {
        return Plugin_Continue;
    }
    
    // Collect game state
    char stateJson[BUFFER_SIZE];
    CollectGameState(stateJson, sizeof(stateJson));
    
    // Send to Python
    SocketSend(g_hSocket, stateJson, strlen(stateJson));
    
    return Plugin_Continue;
}

public Action Timer_DirectorUpdate(Handle timer) {
    if (!g_bDirectorEnabled || g_hSocket == null) {
        return Plugin_Continue;
    }
    
    // Director logic runs at lower frequency
    // Python will send director commands based on state
    return Plugin_Continue;
}

void CollectGameState(char[] buffer, int bufferSize) {
    // This is a simplified state collection
    // In production, you'd collect much more detailed information
    
    Handle json = json_object();
    
    // Basic game info
    json_object_set_number(json, "gameTime", GetGameTime());
    json_object_set_number(json, "roundTime", GetRoundTime());
    
    // Survivor info
    Handle survivors = json_array();
    for (int i = 1; i <= 4; i++) {
        if (IsClientInGame(i) && GetClientTeam(i) == 2) {  // Team 2 = survivors
            Handle survivor = json_object();
            
            // Health
            int health = GetClientHealth(i);
            int tempHealth = GetEntProp(i, Prop_Send, "m_tempHealth");
            json_object_set_number(survivor, "health", health);
            json_object_set_number(survivor, "tempHealth", tempHealth);
            
            // Position
            float pos[3];
            GetClientAbsOrigin(i, pos);
            Handle posArray = json_array();
            json_array_append_number(posArray, pos[0]);
            json_array_append_number(posArray, pos[1]);
            json_array_append_number(posArray, pos[2]);
            json_object_set(survivor, "position", posArray);
            CloseHandle(posArray);
            
            // Angle
            float ang[3];
            GetClientAbsAngles(i, ang);
            Handle angArray = json_array();
            json_array_append_number(angArray, ang[0]);
            json_array_append_number(angArray, ang[1]);
            json_array_append_number(angArray, ang[2]);
            json_object_set(survivor, "angle", angArray);
            CloseHandle(angArray);
            
            // Current weapon
            char weapon[64];
            GetClientWeapon(i, weapon, sizeof(weapon));
            json_object_set_string(survivor, "weapon", weapon);
            
            json_array_append(survivors, survivor);
            CloseHandle(survivor);
        }
    }
    json_object_set(json, "survivors", survivors);
    CloseHandle(survivors);
    
    // Infected counts
    int common = GetEntProp(FindEntityByClassname(-1, "infected"), Prop_Data, "m_iCount");
    json_object_set_number(json, "commonInfected", common);
    
    // Count special infected
    int specials[5] = {0, 0, 0, 0, 0};
    int entity = -1;
    while ((entity = FindEntityByClassname(entity, "infected")) != -1) {
        int class = GetEntProp(entity, Prop_Send, "m_zombieClass");
        if (class >= 1 && class <= 5) {
            specials[class - 1]++;
        }
    }
    
    Handle specialArray = json_array();
    for (int i = 0; i < 5; i++) {
        json_array_append_number(specialArray, specials[i]);
    }
    json_object_set(json, "specialInfected", specialArray);
    CloseHandle(specialArray);
    
    // Convert to string
    json_stringify(json, buffer, bufferSize);
    CloseHandle(json);
}

void HandleBotCommand(Handle json) {
    int botId = json_object_get_int(json, "bot_id");
    char action[32];
    json_object_get_string(json, "action", action, sizeof(action));
    
    if (!IsClientInGame(botId) || GetClientTeam(botId) != 2) {
        return;
    }
    
    // Execute bot action
    if (StrEqual(action, "move_forward")) {
        MoveBot(botId, 0.0, 100.0);
    } else if (StrEqual(action, "move_backward")) {
        MoveBot(botId, 180.0, 100.0);
    } else if (StrEqual(action, "move_left")) {
        MoveBot(botId, 90.0, 100.0);
    } else if (StrEqual(action, "move_right")) {
        MoveBot(botId, -90.0, 100.0);
    } else if (StrEqual(action, "attack")) {
        ClientCommand(botId, "+attack; -attack");
    } else if (StrEqual(action, "use_item")) {
        ClientCommand(botId, "+use; -use");
    }
}

void HandleDirectorCommand(Handle json) {
    char command[32];
    json_object_get_string(json, "command", command, sizeof(command));
    
    if (StrEqual(command, "spawn_common")) {
        int count = json_object_get_int(json, "count");
        float pos[3];
        json_object_get_vector(json, "position", pos);
        SpawnCommonInfected(pos, count);
    } else if (StrEqual(command, "spawn_special")) {
        char type[32];
        json_object_get_string(json, "type", type, sizeof(type));
        float pos[3];
        json_object_get_vector(json, "position", pos);
        SpawnSpecialInfected(type, pos);
    } else if (StrEqual(command, "spawn_witch")) {
        float pos[3];
        json_object_get_vector(json, "position", pos);
        SpawnWitch(pos);
    } else if (StrEqual(command, "spawn_tank")) {
        float pos[3];
        json_object_get_vector(json, "position", pos);
        SpawnTank(pos);
    } else if (StrEqual(command, "trigger_panic")) {
        CTimer_Start(CreateTimer(0.1, Timer_TriggerPanic), 1.0);
    }
}

// Helper functions
void MoveBot(int client, float angleOffset, float distance) {
    float ang[3], pos[3], forward[3];
    GetClientAbsAngles(client, ang);
    ang[1] += angleOffset;
    
    GetAngleVectors(ang, forward, NULL_VECTOR, NULL_VECTOR);
    GetClientAbsOrigin(client, pos);
    
    pos[0] += forward[0] * distance;
    pos[1] += forward[1] * distance;
    pos[2] += forward[2] * distance;
    
    TeleportEntity(client, pos, ang, NULL_VECTOR);
}

void SpawnCommonInfected(float pos[3], int count) {
    for (int i = 0; i < count; i++) {
        int infected = CreateEntityByName("infected");
        if (infected != -1) {
            TeleportEntity(infected, pos, NULL_VECTOR, NULL_VECTOR);
            DispatchSpawn(infected);
            ActivateEntity(infected);
        }
    }
}

void SpawnSpecialInfected(char[] type, float pos[3]) {
    char classname[64];
    Format(classname, sizeof(classname), "%s", type);
    
    int infected = CreateEntityByName(classname);
    if (infected != -1) {
        TeleportEntity(infected, pos, NULL_VECTOR, NULL_VECTOR);
        DispatchSpawn(infected);
        ActivateEntity(infected);
    }
}

void SpawnWitch(float pos[3]) {
    int witch = CreateEntityByName("witch");
    if (witch != -1) {
        TeleportEntity(witch, pos, NULL_VECTOR, NULL_VECTOR);
        DispatchSpawn(witch);
        ActivateEntity(witch);
    }
}

void SpawnTank(float pos[3]) {
    SpawnSpecialInfected("tank", pos);
}

void ResetEpisode() {
    // Force round restart
    ServerCommand("mp_restartgame 1");
}

// Event handlers
public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast) {
    PrintToServer("[L4D2-AI] Round started");
}

public void Event_RoundEnd(Event event, const char[] name, bool dontBroadcast) {
    PrintToServer("[L4D2-AI] Round ended");
}

public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast) {
    int victim = GetClientOfUserId(event.GetInt("userid"));
    if (victim > 0 && IsClientInGame(victim)) {
        // Send death event to Python
        if (g_hSocket != null) {
            char deathJson[256];
            Format(deathJson, sizeof(deathJson), 
                "{\"type\":\"player_death\",\"victim\":%d,\"team\":%d}", 
                victim, GetClientTeam(victim));
            SocketSend(g_hSocket, deathJson, strlen(deathJson));
        }
    }
}

public void Event_InfectedDeath(Event event, const char[] name, bool dontBroadcast) {
    // Common infected death
}

public void Event_WitchKilled(Event event, const char[] name, bool dontBroadcast) {
    // Witch killed
}

public void Event_TankKilled(Event event, const char[] name, bool dontBroadcast) {
    // Tank killed
}

public Action Timer_TriggerPanic(Handle timer) {
    CTimer_Start(timer, 0.1);
    return Plugin_Continue;
}
