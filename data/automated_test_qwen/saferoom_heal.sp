#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

#define CHECKPOINT_MODEL "models/props_interiors/chair_padded01a.mdl"

public Plugin myinfo =
{
    name = "Checkpoint Healer",
    author = "Developer",
    description = "Heals survivors when entering checkpoint",
    version = "1.0",
    url = ""
};

bool g_bHasLeft[32];
int g_iCheckpoints[MAX_MAPS][2]; // [map index][0=enabled, 1=modelindex]

public void OnPluginStart()
{
    LoadTranslations("checkpoint_healer.phrases");
    
    RegConsoleCmd("sm_enable", Cmd_Enable);
    RegAdminCmd("sm_forceheal", Cmd_ForceHeal, ADMFLAG_ROOT);
    
    HookEvent("player_bot_replace", Event_BotSwap);
    HookEvent("bot_player_replace", Event_BotSwap);
    
    HookEvent("round_start", Event_RoundStart);
    HookEvent("player_spawn", Event_PlayerSpawn);
    HookEvent("player_death", Event_PlayerDeath);
    HookEvent("player_team", Event_PlayerTeam);
    
    char sMap[64];
    GetCurrentMap(sMap, sizeof(sMap));
    int mapIndex = FindMapInArray(sMap);
    
    if (mapIndex != -1)
        LoadCheckpoints(mapIndex);
}

void LoadCheckpoints(int mapIndex)
{
    Handle hFile = OpenHandle(FormatEx(sFile, sizeof(sFile), "configs/checkpoints/%s.txt", maps_array[mapIndex]));
    if (!hFile) { 
        PrintToServer("[Checkpoint Healer] File not found: configs/checkpoints/%s.txt", maps_array[mapIndex]);
        return; 
    }
    
    int iLine;
    char sModel[PLATFORM_MAX_PATH];
    while ((iLine = ReadFileLine(hFile, sModel, sizeof(sModel))) != -1)
    {
        if (StringToInt(sModel) == 1)
            g_iCheckpoints[mapIndex][0] = true;
        
        if (StringToInt(sModel) == 2)
        {
            g_iCheckpoints[mapIndex][1] = AddEntityListener(ListenType_Created, "*", EntityCreated);
        }
    }
    
    CloseHandle(hFile);
}

public void OnMapEnd()
{
    ClearAll();
}

void ClearAll()
{
    for (int i = 0; i < MAX_MAPS; i++)
    {
        g_iCheckpoints[i][0] = false;
        g_iCheckpoints[i][1] = 0;
    }
    
    for (int i = 1; i <= MaxClients; i++)
    {
        g_bHasLeft[i] = false;
        SetEntProp(i, Prop_Send, "m_hasLeft", 0);
    }
}

public void OnClientDisconnect(int client)
{
    g_bHasLeft[client] = false;
    SetEntProp(client, Prop_Send, "m_hasLeft", 0);
}

public Action Command_CheckpointCheck(int client, int args)
{
    if (client == 0 || !IsClientInGame(client))
        return Plugin_Handled;
    
    if (GetClientTeam(client) != 2)
        return Plugin_Handled;
    
    if (g_bHasLeft[client])
        ReplyToCommand(client, "[SM] You have already left the checkpoint.");
    else
        ReplyToCommand(client, "[SM] Press 'C' to use your checkpoint.");
    
    return Plugin_Handled;
}

Action EntityCreated(int entity, const char[] classname)
{
    if (strcmp(classname, "prop_physics_override") == 0 && IsModelOfEntity(entity, CHECKPOINT_MODEL))
    {
        CreateConVarExt("sm