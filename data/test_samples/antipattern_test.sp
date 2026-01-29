/**
 * Test plugin containing intentional anti-patterns for testing
 * the detect_antipatterns.py system.
 *
 * DO NOT USE THIS CODE IN PRODUCTION - it contains bugs!
 */

#include <sourcemod>
#include <sdktools>

// Missing #pragma semicolon 1 and #pragma newdecls required

// Global variable without g_ prefix (STY003)
int playerKills[MAXPLAYERS + 1];
Handle hTimer;  // Timer handle that gets overwritten (CORR007)

// Magic numbers (STY001)
public void OnClientPutInServer(int client)
{
    // Missing client validation (CORR001)
    char name[64];
    GetClientName(client, name, sizeof(name));  // No IsClientInGame check!

    // Using wrong random function (L4D001)
    float delay = RandomFloat(1.0, 5.0);  // Should be GetRandomFloat!

    // Timer handle overwrite without killing (CORR007)
    hTimer = CreateTimer(delay, Timer_Announce, client);
    hTimer = CreateTimer(2.0, Timer_Announce, client);  // Overwrites without killing!
}

public Action Timer_Announce(Handle timer, any client)
{
    // No client validation after timer delay
    PrintToChat(client, "Welcome!");  // Client may have disconnected!
    return Plugin_Continue;
}

// SQL Injection vulnerability (SEC001)
public void SavePlayerData(int client, const char[] input)
{
    char query[512];
    char name[64];
    GetClientName(client, name, sizeof(name));

    // Direct format into SQL without escaping!
    Format(query, sizeof(query), "INSERT INTO players (name, data) VALUES ('%s', '%s')", name, input);
    SQL_Query(g_hDatabase, query);  // SQL Injection vulnerability!
}

// Blocking operation in callback (PERF001)
public void Event_PlayerSpawn(Event event, const char[] name, bool dontBroadcast)
{
    int client = GetClientOfUserId(event.GetInt("userid"));
    // No validation of client!

    // Blocking SQL in event callback - should use SQL_TQuery!
    char query[256];
    Format(query, sizeof(query), "SELECT * FROM players WHERE id = %d", client);
    SQL_Query(g_hDatabase, query);  // Blocking operation in callback!

    // Also blocking file operation
    Handle file = OpenFile("data.txt", "r");  // Blocking file I/O
    ReadFile(file, buffer, 256);
    // Handle never closed! (MEM001)
}

// Unclosed handles (MEM001)
public void ShowMenu(int client)
{
    Handle menu = CreateMenu(MenuHandler);
    AddMenuItem(menu, "option1", "Option 1");
    AddMenuItem(menu, "option2", "Option 2");
    DisplayMenu(menu, client, 30);
    // Menu handle created but never closed!
}

// Infinite loop risk (CORR002)
public void ProcessData()
{
    while(true)
    {
        // No break condition - infinite loop!
        DoSomething();
    }
}

// Wrong L4D2 event names (L4D003)
public void OnPluginStart()
{
    HookEvent("pounce", Event_HunterPounce);  // Wrong! Should be "lunge_pounce"
    HookEvent("smoker_tongue_grab", Event_SmokerGrab);  // Wrong! Should be "tongue_grab"
    HookEvent("boomer_vomit", Event_BoomerVomit);  // Wrong! Should be "player_now_it"
    HookEvent("charger_grab", Event_ChargerGrab);  // Wrong! Should be "charger_carry_start"
    HookEvent("panic_start", Event_PanicStart);  // Wrong! Should be "create_panic_event"
}

// Wrong speed property (L4D002)
public void SpeedBoost(int client)
{
    // Wrong property for L4D2!
    SetEntPropFloat(client, Prop_Send, "m_flMaxSpeed", 300.0);  // Should be m_flLaggedMovementValue!
    SetEntPropFloat(client, Prop_Data, "m_flSpeed", 1.5);  // Also wrong!
}

// Off-by-one error in client loop (CORR004)
public void CheckAllClients()
{
    // Wrong: starting at 0 instead of 1
    for (int i = 0; i <= MaxClients; i++)
    {
        if (IsClientInGame(i))  // i=0 is invalid!
        {
            ProcessClient(i);
        }
    }
}

// Entity index stored without reference (CORR005)
int g_iMyEntity;

public void SpawnEntity()
{
    g_iMyEntity = CreateEntityByName("prop_physics");
    // Should use EntIndexToEntRef for persistent storage!
    // Also return value not checked (CORR006)
}

// Command injection (SEC002)
public Action Command_Execute(int client, int args)
{
    char arg[256];
    GetCmdArg(1, arg, sizeof(arg));
    // Direct user input to ServerCommand - dangerous!
    ServerCommand("exec %s", arg);
    return Plugin_Handled;
}

// Old declaration style (DEP001)
new String:oldStyleString[64];
new Float:oldStyleFloat;
new Handle:oldStyleHandle;

// FindEntityByClassname in OnGameFrame (PERF002)
public void OnGameFrame()
{
    // Expensive operation every frame!
    int entity = -1;
    while ((entity = FindEntityByClassname(entity, "infected")) != -1)
    {
        // Process each infected every single frame...
        GetEntProp(entity, Prop_Send, "m_iHealth");
    }

    // Also expensive trace every frame (PERF006)
    TR_TraceRay(origin, angles, MASK_SOLID, RayType_Infinite);
}

// DataPack without ResetPack (MEM002)
public void UseDataPack(Handle pack)
{
    // Missing ResetPack before reading!
    int value = ReadPackCell(pack);  // May read garbage!
}

// Race condition potential (CORR003)
int g_GlobalCounter;

public void Event_PlayerJoin(Event event, const char[] name, bool dontBroadcast)
{
    g_GlobalCounter++;  // Modified in callback
}

public void Event_PlayerLeave(Event event, const char[] name, bool dontBroadcast)
{
    g_GlobalCounter--;  // Also modified here - race condition!
}

public void Timer_CheckCounter(Handle timer)
{
    g_GlobalCounter = 0;  // And here too!
}

// String operation in hot path (PERF003)
public Action OnPlayerRunCmd(int client, int &buttons)
{
    char buffer[256];
    Format(buffer, sizeof(buffer), "Client %d pressed buttons", client);  // Every frame!
    return Plugin_Continue;
}

// GetMaxClients instead of MaxClients (DEP004)
public void CountPlayers()
{
    int max = GetMaxClients();  // Should use MaxClients constant
}

// Hardcoded zombie class (L4D005)
public void CheckZombieClass(int client)
{
    int class = GetEntProp(client, Prop_Send, "m_zombieClass");
    if (class == 8)  // Magic number! Should use L4D2_ZOMBIECLASS_TANK
    {
        // Tank handling
    }
}

// Menu handle created in loop (MEM006)
public void CreateManyMenus()
{
    for (int i = 1; i <= MaxClients; i++)
    {
        Handle menu = CreateMenu(MenuHandler);  // Created in loop without cleanup!
        AddMenuItem(menu, "opt", "Option");
        DisplayMenu(menu, i, 30);
        // Previous menu handle leaked!
    }
}

// Path traversal risk (SEC006)
public Action Command_LoadFile(int client, int args)
{
    char path[256];
    GetCmdArg(1, path, sizeof(path));
    // User can specify "../../../etc/passwd"
    BuildPath(Path_SM, buffer, sizeof(buffer), "configs/%s", path);
    Handle file = OpenFile(buffer, "r");
    return Plugin_Handled;
}

// Excessive timer creation (PERF004)
public void Event_Damage(Event event, const char[] name, bool dontBroadcast)
{
    int victim = GetClientOfUserId(event.GetInt("userid"));
    // Creates new timer on EVERY damage event!
    CreateTimer(1.0, Timer_RegenHealth, victim);
}

// GetEntProp in loop without caching (PERF005)
public void ProcessEntities()
{
    for (int i = 0; i < 100; i++)
    {
        // Accessing same property repeatedly in loop
        int health = GetEntProp(entity, Prop_Send, "m_iHealth");
        int maxHealth = GetEntProp(entity, Prop_Send, "m_iMaxHealth");
    }
}
