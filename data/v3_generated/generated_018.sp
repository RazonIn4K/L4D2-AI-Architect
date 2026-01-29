#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>
#include <l4d2_zombie_spawner>

public Plugin myinfo =
{
    name = "Finale Zombie Spawner",
    author = "Developer",
    description = "Spawns extra zombies during finales",
    version = "1.0",
    url = ""
};

ConVar g_cvarExtraZombies;
int g_iCvarExtraZombies;

public void OnPluginStart()
{
    g_cvarExtraZombies = CreateConVar("l4d2_finale_extra_zombies", "10", "Number of extra zombies to spawn during finales", FCVAR_NOTIFY, true, 0.0);
    AutoExecConfig(true, "l4d2_finale_extra_zombies");

    HookConVarChange(g_cvarExtraZombies, OnCvarChanged);
    OnCvarChanged(null, "", "");

    HookEvent("finale_start", Event_FinaleStart);
}

void OnCvarChanged(ConVar hCvar, const char[] sOldVal, const char[] sNewVal)
{
    g_iCvarExtraZombies = g_cvarExtraZombies.IntValue;
}

void Event_FinaleStart(Event hEvent, const char[] sEventName, bool bDontBroadcast)
{
    CreateTimer(5.0, Timer_SpawnExtraZombies, _, TIMER_FLAG_NO_MAPCHANGE);
}

Action Timer_SpawnExtraZombies(Handle hTimer)
{
    if (IsMapFinaleInProgress())
    {
        for (int i = 0; i < g_iCvarExtraZombies; i++)
        {
            SpawnRandomZombie();
        }
    }
    return Plugin_Continue;
}

bool IsMapFinaleInProgress()
{
    return FindEntityByClassname(-1, "info_finale") != -1;
}
