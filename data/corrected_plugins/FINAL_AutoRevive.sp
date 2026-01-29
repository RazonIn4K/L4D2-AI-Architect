#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

#define PLUGIN_VERSION "1.2"
#define REVIVE_DELAY 30.0
#define CHECK_RADIUS 500.0

public Plugin myinfo =
{
    name = "L4D2 Auto Revive",
    author = "Optimized Version",
    description = "Auto-revives incapacitated survivors after 30s if alone",
    version = PLUGIN_VERSION,
    url = ""
};

Handle g_hReviveTimer[MAXPLAYERS + 1];

public void OnPluginStart()
{
    // CORRECT: Use player_incapacitated, NOT player_death!
    // In L4D2: Incapacitated = knocked down but alive
    //          Death = actually dead (bled out or finished off)
    HookEvent("player_incapacitated", Event_PlayerIncap);
    HookEvent("revive_success", Event_ReviveSuccess);
    HookEvent("player_death", Event_PlayerDeath);
    HookEvent("round_start", Event_RoundStart);
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
    // Clear all timers on round start
    for (int i = 1; i <= MaxClients; i++)
    {
        ClearReviveTimer(i);
    }
}

public void OnClientDisconnect(int client)
{
    ClearReviveTimer(client);
}

public void Event_PlayerIncap(Event event, const char[] name, bool dontBroadcast)
{
    int client = GetClientOfUserId(event.GetInt("userid"));
    
    if (!IsValidClient(client))
        return;
    
    // Clear any existing timer
    ClearReviveTimer(client);
    
    // Start auto-revive countdown
    g_hReviveTimer[client] = CreateTimer(REVIVE_DELAY, Timer_CheckAutoRevive, 
                                          GetClientUserId(client), TIMER_FLAG_NO_MAPCHANGE);
    
    PrintToChat(client, "\x04[Auto-Revive]\x01 You will be auto-revived in %.0f seconds if no one is nearby.", REVIVE_DELAY);
}

public void Event_ReviveSuccess(Event event, const char[] name, bool dontBroadcast)
{
    int client = GetClientOfUserId(event.GetInt("subject"));
    ClearReviveTimer(client);
}

public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast)
{
    int client = GetClientOfUserId(event.GetInt("userid"));
    ClearReviveTimer(client);
}

public Action Timer_CheckAutoRevive(Handle timer, int userid)
{
    int client = GetClientOfUserId(userid);
    g_hReviveTimer[client] = null;
    
    if (!IsValidClient(client))
        return Plugin_Continue;
    
    // CORRECT: Check incapacitation via netprop
    if (!IsClientIncapacitated(client))
        return Plugin_Continue;
    
    // Check if any other survivor is nearby
    if (IsOtherSurvivorNearby(client, CHECK_RADIUS))
    {
        PrintToChat(client, "\x04[Auto-Revive]\x01 A teammate is nearby - waiting for manual revive.");
        return Plugin_Continue;
    }
    
    // Perform the revive
    RevivePlayer(client);
    
    return Plugin_Continue;
}

bool IsOtherSurvivorNearby(int client, float radius)
{
    float clientPos[3];
    GetClientAbsOrigin(client, clientPos);
    
    for (int i = 1; i <= MaxClients; i++)
    {
        if (i == client)
            continue;
        
        if (!IsValidClient(i) || !IsPlayerAlive(i))
            continue;
        
        if (IsClientIncapacitated(i))  // Can't help if also down
            continue;
        
        float otherPos[3];
        GetClientAbsOrigin(i, otherPos);
        
        if (GetVectorDistance(clientPos, otherPos) <= radius)
            return true;
    }
    
    return false;
}

void RevivePlayer(int client)
{
    // Method 1: Try Left4DHooks if available
    if (GetFeatureStatus(FeatureType_Native, "L4D_ReviveSurvivor") == FeatureStatus_Available)
    {
        L4D_ReviveSurvivor(client);
        AnnounceRevive(client);
        return;
    }
    
    // Method 2: Use cheat command with temporary flag removal
    int flags = GetCommandFlags("give");
    if (flags != -1)
    {
        SetCommandFlags("give", flags & ~FCVAR_CHEAT);
        FakeClientCommand(client, "give health");
        SetCommandFlags("give", flags);
        
        // Set to reasonable health (not full)
        SetEntityHealth(client, 30);
        
        // Mark as having been revived (for black & white status)
        int currentRevives = GetEntProp(client, Prop_Send, "m_currentReviveCount");
        SetEntProp(client, Prop_Send, "m_currentReviveCount", currentRevives + 1);
        
        AnnounceRevive(client);
        return;
    }
    
    // Method 3: Direct prop manipulation (fallback)
    SetEntProp(client, Prop_Send, "m_isIncapacitated", 0);
    SetEntProp(client, Prop_Send, "m_iHealth", 30);
    SetEntDataFloat(client, FindSendPropInfo("CTerrorPlayer", "m_flHealthBuffer"), 0.0);
    
    AnnounceRevive(client);
}

void AnnounceRevive(int client)
{
    char name[MAX_NAME_LENGTH];
    GetClientName(client, name, sizeof(name));
    PrintToChatAll("\x04[Auto-Revive]\x01 %s has been automatically revived!", name);
}

// CORRECT: Check incapacitation via entity property
bool IsClientIncapacitated(int client)
{
    return GetEntProp(client, Prop_Send, "m_isIncapacitated") == 1;
}

void ClearReviveTimer(int client)
{
    if (g_hReviveTimer[client] != null)
    {
        KillTimer(g_hReviveTimer[client]);
        g_hReviveTimer[client] = null;
    }
}

bool IsValidClient(int client)
{
    return (client > 0 && client <= MaxClients && 
            IsClientInGame(client) && GetClientTeam(client) == 2);
}

// Native declaration for Left4DHooks (if available)
native void L4D_ReviveSurvivor(int client);
