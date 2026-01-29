#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>
#include <l4d2>

#define PLUGIN_VERSION "1.0"

public Plugin myinfo =
{
    name = "Map Explored Percentage",
    author = "Developer",
    description = "Announces the percentage of the map explored by survivors",
    version = PLUGIN_VERSION,
    url = ""
};

float g_fMapSize = 0.0;
float g_fMapExplored = 0.0;

public void OnPluginStart()
{
    HookEvent("round_start", Event_RoundStart, EventHookMode_PostNoCopy);
    HookEvent("round_end", Event_RoundEnd, EventHookMode_PostNoCopy);
    HookEvent("player_run_step", Event_PlayerRunStep, EventHookMode_PostNoCopy);
    HookEvent("map_transition", Event_MapTransition, EventHookMode_PostNoCopy); // when a map is changed in a campaign (not including round_end)
    HookEvent("mission_lost", Event_MissionLost, EventHookMode_PostNoCopy); // when a map is changed in a campaign (including round_end)
    HookEvent("finale_vehicle_leaving", Event_FinaleVehicleLeaving, EventHookMode_PostNoCopy); // when a finale is completed and the vehicle leaves (not including round_end)
    
    RegConsoleCmd("sm_mapexplored", Cmd_MapExplored);
}

public void OnMapStart()
{
    // Reset values on map start
    g_fMapSize = 0.0;
    g_fMapExplored = 0.0;
}

public void OnMapEnd()
{
    // Reset values on map end
    g_fMapSize = 0.0;
    g_fMapExplored = 0.0;
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
    // Reset explored percentage on round start
    g_fMapExplored = 0.0;
}

public void Event_RoundEnd(Event event, const char[] name, bool dontBroadcast)
{
    // Announce explored percentage on round end
    AnnounceExploredPercentage();
}

public void Event_PlayerRunStep(Event event, const char[] name, bool dontBroadcast)
{
    int client = GetClientOfUserId(event.GetInt("userid"));
    if (client > 0 && IsClientInGame(client) && IsPlayerAlive(client))
    {
        // Check if the player is a survivor
        if (GetClientTeam(client) == 2) // Survivor team
        {
            // Update explored percentage
            UpdateExploredPercentage(client);
        }
    }
}

public void Event_MapTransition(Event event, const char[] name, bool dontBroadcast)
{
    // Announce explored percentage on map transition (final maps)
    AnnounceExploredPercentage();
}

public void Event_MissionLost(Event event, const char[] name, bool dontBroadcast)
{
    // Announce explored percentage on mission lost (all maps)
    AnnounceExploredPercentage();
}

public void Event_FinaleVehicleLeaving(Event event, const char[] name, bool dontBroadcast)
{
    // Announce explored percentage on finale vehicle leaving (final maps)
    AnnounceExploredPercentage();
}

void UpdateExploredPercentage(int client)
{
    // Get the current position of the survivor
    float position[3];
    GetClientAbsOrigin(client, position);

    // Get the distance from the origin to the current position
    float distance = GetDistanceToOrigin(position);

    // Calculate the explored percentage based on the distance
    float explored = (distance / g_fMapSize) * 100.0;
    if (explored > g_fMapExplored)
    {
        g_fMapExplored = explored;
    }

    // Print debug info
    // PrintToChatAll("[DEBUG] Client %d: Explored %.2f%%", client, g_fMapExplored);
}

void AnnounceExploredPercentage()
{
    // Calculate the explored percentage
    float exploredPercentage = g_fMapExplored;
    if (exploredPercentage > 100.0)
    {
        exploredPercentage = 100.0;
    }

    // Announce the explored percentage to all players
    PrintToChatAll("\x04[MAP] \x01Explored %.2f%% of the map!", exploredPercentage);
}

float GetDistanceToOrigin(const float position[3])
{
    float origin[3] = {0.0, 0.0, 0.0};
    GetEntPropVector(FindEntityByClassname(-1, "info_map_transition"), Prop_Send, "origin", origin);

    return GetVectorDistance(position, origin);
}

float GetVectorDistance(const float vec1[3], const float vec2[3])
{
    float delta[3];
    delta[0] = vec1[0] - vec2[0];
    delta[1] = vec1[1] - vec2[1];
    delta[2] = vec1[2] - vec2[2];

    return SquareRoot(Square(delta[0]) + Square(delta[1]) + Square(delta[2]));
}

Action Cmd_MapExplored(int client, int args)
{
    // Print the explored percentage to the player
    PrintToChat(client, "\x04[MAP] \x01Explored %.2f%% of the map!", g_fMapExplored);
    return Plugin_Handled;
}