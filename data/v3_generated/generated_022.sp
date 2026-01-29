#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>
#include <l4d2>
#include <admin-groups>

public Plugin myinfo =
{
    name = "Saferoom Victory Sound",
    author = "Developer",
    description = "Plays a victory sound when all survivors reach the saferoom",
    version = "1.0",
    url = ""
};

#define VICTORY_SOUND "ambient/levels/lake/river_splash1.wav"

public void OnPluginStart()
{
    HookEvent("round_end", Event_RoundEnd, EventHookMode_PostNoCopy);
}

public void Event_RoundEnd(Event event, const char[] name, bool dontBroadcast)
{
    // Play sound for all survivors when round ends
    PlayVictorySound();
}

void PlayVictorySound()
{
    int aliveCount = 0;
    int client;

    for (int i = 1; i <= MaxClients; i++)
    {
        if (IsClientInGame(i) && GetClientTeam(i) == 2 && IsPlayerAlive(i))
        {
            if (IsPlayerInSaferoom(i))
            {
                aliveCount++;
            }
        }
    }

    // Check if all survivors are in the saferoom
    if (aliveCount == GetSurvivorCount())
    {
        for (int i = 1; i <= MaxClients; i++)
        {
            if (IsClientInGame(i) && GetClientTeam(i) == 2 && IsPlayerAlive(i))
            {
                EmitSoundToClient(i, VICTORY_SOUND, SOUND_FROM_PLAYER, i, SNDLEVEL_NORMAL, 0.5, 0.5);
            }
        }
    }
}

bool IsPlayerInSaferoom(int client)
{
    float pos[3];
    GetClientAbsOrigin(client, pos);

    return (GetClosestSaferoomDistance(pos) <= 500.0); // 500 units
}

float GetClosestSaferoomDistance(const float pos[3])
{
    float saferoomPos[3];
    float closestDistance = 99999.0;

    for (int i = 1; i <= MaxClients; i++)
    {
        if (IsClientInGame(i) && GetClientTeam(i) == 3 && IsPlayerAlive(i))
        {
            GetClientAbsOrigin(i, saferoomPos);
            float distance = GetVectorDistance(pos, saferoomPos);

            if (distance < closestDistance)
            {
                closestDistance = distance;
            }
        }
    }

    return closestDistance;
}

int GetSurvivorCount()
{
    int count = 0;

    for (int i = 1; i <= MaxClients; i++)
    {
        if (IsClientInGame(i) && GetClientTeam(i) == 2 && IsPlayerAlive(i))
        {
            count++;
        }
    }

    return count;
}