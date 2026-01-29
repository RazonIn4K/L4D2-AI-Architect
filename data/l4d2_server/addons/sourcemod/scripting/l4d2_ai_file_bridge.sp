/**
 * L4D2 AI File Bridge v1.0
 *
 * Simple file-based communication for AI bot control.
 * No socket extension needed - just reads/writes files.
 *
 * Files:
 *   /tmp/l4d2_state.json     - Game state (written by plugin)
 *   /tmp/l4d2_commands.txt   - Bot commands (read by plugin)
 */

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>

#pragma newdecls required
#pragma semicolon 1

#define PLUGIN_VERSION "1.0.1"
#define UPDATE_RATE 10.0  // Hz

char g_sStateFile[PLATFORM_MAX_PATH];
char g_sCommandFile[PLATFORM_MAX_PATH];

public Plugin myinfo = {
    name = "L4D2 AI File Bridge",
    author = "L4D2-AI-Architect",
    description = "File-based AI bot control bridge",
    version = PLUGIN_VERSION,
    url = "https://github.com/RazonIn4K/L4D2-AI-Architect"
};

// Action constants (must match Python)
enum {
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

Handle g_hUpdateTimer = null;
bool g_bEnabled = true;
bool g_bDebug = false;
int g_iPendingActions[MAXPLAYERS + 1];

public void OnPluginStart() {
    // Build paths in SourceMod's data directory
    BuildPath(Path_SM, g_sStateFile, sizeof(g_sStateFile), "data/l4d2_state.json");
    BuildPath(Path_SM, g_sCommandFile, sizeof(g_sCommandFile), "data/l4d2_commands.txt");

    RegAdminCmd("sm_aibridge_toggle", Cmd_Toggle, ADMFLAG_ROOT, "Toggle AI bridge");
    RegAdminCmd("sm_aibridge_debug", Cmd_Debug, ADMFLAG_ROOT, "Toggle debug mode");
    RegAdminCmd("sm_aibridge_status", Cmd_Status, ADMFLAG_ROOT, "Show bridge status");

    g_hUpdateTimer = CreateTimer(1.0 / UPDATE_RATE, Timer_Update, _, TIMER_REPEAT);

    PrintToServer("[AI-Bridge] File-based bridge v%s loaded", PLUGIN_VERSION);
    PrintToServer("[AI-Bridge] State file: %s", g_sStateFile);
    PrintToServer("[AI-Bridge] Command file: %s", g_sCommandFile);
}

public void OnPluginEnd() {
    if (g_hUpdateTimer != null) {
        KillTimer(g_hUpdateTimer);
    }
}

public void OnClientPutInServer(int client) {
    g_iPendingActions[client] = ACTION_IDLE;

    if (IsFakeClient(client)) {
        SDKHook(client, SDKHook_PreThink, Hook_BotThink);
    }
}

public void OnClientDisconnect(int client) {
    SDKUnhook(client, SDKHook_PreThink, Hook_BotThink);
}

// ============================================================================
// COMMANDS
// ============================================================================

public Action Cmd_Toggle(int client, int args) {
    g_bEnabled = !g_bEnabled;
    ReplyToCommand(client, "[AI-Bridge] %s", g_bEnabled ? "ENABLED" : "DISABLED");
    return Plugin_Handled;
}

public Action Cmd_Debug(int client, int args) {
    g_bDebug = !g_bDebug;
    ReplyToCommand(client, "[AI-Bridge] Debug %s", g_bDebug ? "ON" : "OFF");
    return Plugin_Handled;
}

public Action Cmd_Status(int client, int args) {
    ReplyToCommand(client, "=== AI File Bridge Status ===");
    ReplyToCommand(client, "Enabled: %s", g_bEnabled ? "YES" : "NO");
    ReplyToCommand(client, "Debug: %s", g_bDebug ? "ON" : "OFF");
    ReplyToCommand(client, "State: %s", g_sStateFile);
    ReplyToCommand(client, "Commands: %s", g_sCommandFile);
    return Plugin_Handled;
}

// ============================================================================
// UPDATE TIMER
// ============================================================================

public Action Timer_Update(Handle timer) {
    if (!g_bEnabled) {
        return Plugin_Continue;
    }

    // Write game state
    WriteGameState();

    // Read and execute commands
    ReadCommands();

    return Plugin_Continue;
}

// ============================================================================
// STATE WRITING
// ============================================================================

void WriteGameState() {
    Handle file = OpenFile(g_sStateFile, "w");
    if (file == null) {
        if (g_bDebug) PrintToServer("[AI-Bridge] Failed to open state file");
        return;
    }

    // Start JSON
    WriteFileLine(file, "{");
    WriteFileLine(file, "  \"time\": %.2f,", GetGameTime());

    // Map info
    char mapName[64];
    GetCurrentMap(mapName, sizeof(mapName));
    WriteFileLine(file, "  \"map\": \"%s\",", mapName);

    // Survivors array
    WriteFileLine(file, "  \"survivors\": [");

    bool firstSurvivor = true;
    for (int i = 1; i <= MaxClients; i++) {
        if (!IsClientInGame(i) || GetClientTeam(i) != 2) continue;

        if (!firstSurvivor) WriteFileLine(file, ",");
        firstSurvivor = false;

        float pos[3], ang[3], vel[3];
        GetClientAbsOrigin(i, pos);
        GetClientAbsAngles(i, ang);
        GetEntPropVector(i, Prop_Data, "m_vecVelocity", vel);

        int health = IsPlayerAlive(i) ? GetClientHealth(i) : 0;
        bool alive = IsPlayerAlive(i);
        bool incapped = GetEntProp(i, Prop_Send, "m_isIncapacitated") == 1;
        bool isBot = IsFakeClient(i);

        char weapon[32];
        GetClientWeapon(i, weapon, sizeof(weapon));

        char name[64];
        GetClientName(i, name, sizeof(name));

        WriteFileLine(file, "    {");
        WriteFileLine(file, "      \"id\": %d,", i);
        WriteFileLine(file, "      \"name\": \"%s\",", name);
        WriteFileLine(file, "      \"health\": %d,", health);
        WriteFileLine(file, "      \"alive\": %s,", alive ? "true" : "false");
        WriteFileLine(file, "      \"incapped\": %s,", incapped ? "true" : "false");
        WriteFileLine(file, "      \"bot\": %s,", isBot ? "true" : "false");
        WriteFileLine(file, "      \"pos\": [%.1f, %.1f, %.1f],", pos[0], pos[1], pos[2]);
        WriteFileLine(file, "      \"ang\": [%.1f, %.1f],", ang[0], ang[1]);
        WriteFileLine(file, "      \"vel\": [%.1f, %.1f, %.1f],", vel[0], vel[1], vel[2]);
        WriteFileLine(file, "      \"weapon\": \"%s\"", weapon);
        WriteFileString(file, "    }", false);
    }

    WriteFileLine(file, "");
    WriteFileLine(file, "  ],");

    // Infected counts
    int commonCount = CountCommonInfected();
    int witchCount = CountWitches();
    int tankCount = CountTanks();

    WriteFileLine(file, "  \"infected\": {");
    WriteFileLine(file, "    \"common\": %d,", commonCount);
    WriteFileLine(file, "    \"witches\": %d,", witchCount);
    WriteFileLine(file, "    \"tanks\": %d", tankCount);
    WriteFileLine(file, "  }");

    WriteFileLine(file, "}");

    CloseHandle(file);
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
            if (class == 8) count++;
        }
    }
    return count;
}

// ============================================================================
// COMMAND READING
// ============================================================================

void ReadCommands() {
    if (!FileExists(g_sCommandFile)) {
        return;
    }

    Handle file = OpenFile(g_sCommandFile, "r");
    if (file == null) {
        return;
    }

    char line[256];
    while (ReadFileLine(file, line, sizeof(line))) {
        TrimString(line);
        if (strlen(line) == 0) continue;

        // Parse: bot_id,action
        // Example: "2,5" means bot 2, action 5 (attack)
        char parts[2][16];
        ExplodeString(line, ",", parts, 2, 16);

        int botId = StringToInt(parts[0]);
        int action = StringToInt(parts[1]);

        if (botId > 0 && botId <= MaxClients) {
            g_iPendingActions[botId] = action;

            if (g_bDebug) {
                PrintToServer("[AI-Bridge] Bot %d -> Action %d", botId, action);
            }
        }
    }

    CloseHandle(file);

    // Clear the command file after reading
    DeleteFile(g_sCommandFile);
}

// ============================================================================
// BOT CONTROL
// ============================================================================

public void Hook_BotThink(int client) {
    if (!g_bEnabled || !IsFakeClient(client) || !IsPlayerAlive(client)) {
        return;
    }

    int action = g_iPendingActions[client];
    if (action == ACTION_IDLE) {
        return;
    }

    ExecuteAction(client, action);
    g_iPendingActions[client] = ACTION_IDLE;
}

void ExecuteAction(int client, int action) {
    switch (action) {
        case ACTION_MOVE_FORWARD, ACTION_MOVE_BACKWARD, ACTION_MOVE_LEFT, ACTION_MOVE_RIGHT: {
            ExecuteMovement(client, action);
        }
        case ACTION_ATTACK: {
            ClientCommand(client, "+attack");
            CreateTimer(0.1, Timer_ReleaseAttack, client);
        }
        case ACTION_USE: {
            ClientCommand(client, "+use");
            CreateTimer(0.1, Timer_ReleaseUse, client);
        }
        case ACTION_RELOAD: {
            ClientCommand(client, "+reload");
            CreateTimer(0.2, Timer_ReleaseReload, client);
        }
        case ACTION_JUMP: {
            ExecuteJump(client);
        }
        case ACTION_SHOVE: {
            ClientCommand(client, "+attack2");
            CreateTimer(0.1, Timer_ReleaseAttack2, client);
        }
        case ACTION_CROUCH: {
            ClientCommand(client, "+duck");
            CreateTimer(0.3, Timer_ReleaseDuck, client);
        }
        case ACTION_HEAL_SELF: {
            ClientCommand(client, "slot3");  // Switch to medkit
            CreateTimer(0.5, Timer_UseHeal, client);
        }
    }
}

void ExecuteMovement(int client, int action) {
    float velocity[3];
    float speed = 220.0;

    float ang[3];
    GetClientAbsAngles(client, ang);

    float fwdVec[3], rightVec[3];
    GetAngleVectors(ang, fwdVec, rightVec, NULL_VECTOR);

    switch (action) {
        case ACTION_MOVE_FORWARD: {
            velocity[0] = fwdVec[0] * speed;
            velocity[1] = fwdVec[1] * speed;
        }
        case ACTION_MOVE_BACKWARD: {
            velocity[0] = -fwdVec[0] * speed;
            velocity[1] = -fwdVec[1] * speed;
        }
        case ACTION_MOVE_LEFT: {
            velocity[0] = -rightVec[0] * speed;
            velocity[1] = -rightVec[1] * speed;
        }
        case ACTION_MOVE_RIGHT: {
            velocity[0] = rightVec[0] * speed;
            velocity[1] = rightVec[1] * speed;
        }
    }

    TeleportEntity(client, NULL_VECTOR, NULL_VECTOR, velocity);
}

void ExecuteJump(int client) {
    if (GetEntityFlags(client) & FL_ONGROUND) {
        float velocity[3];
        GetEntPropVector(client, Prop_Data, "m_vecVelocity", velocity);
        velocity[2] = 300.0;
        TeleportEntity(client, NULL_VECTOR, NULL_VECTOR, velocity);
    }
}

// Timer callbacks
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

public Action Timer_ReleaseDuck(Handle timer, int client) {
    if (IsClientInGame(client)) ClientCommand(client, "-duck");
    return Plugin_Stop;
}

public Action Timer_UseHeal(Handle timer, int client) {
    if (IsClientInGame(client)) {
        ClientCommand(client, "+attack");
        CreateTimer(0.1, Timer_ReleaseAttack, client);
    }
    return Plugin_Stop;
}
