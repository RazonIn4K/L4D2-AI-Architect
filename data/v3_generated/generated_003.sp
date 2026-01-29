#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>
#include <colors>

#define PLUGIN_VERSION "1.0"

public Plugin myinfo =
{
	name = "Smoker Tongue Grab Tracker",
	description = "Tracks Smoker tongue grab duration",
	author = "Developer",
	version = PLUGIN_VERSION,
	url = ""
};

#define MAX_GRAB_DURATION 60.0

float g_fLongestGrab[MAXPLAYERS + 1];
int g_iLongestSmoker[MAXPLAYERS + 1];
bool g_bInGrab[MAXPLAYERS + 1];

public void OnPluginStart()
{
	HookEvent("round_start", Event_RoundStart);
	HookEvent("round_end", Event_RoundEnd);
	HookEvent("player_death", Event_PlayerDeath);
	HookEvent("player_spawn", Event_PlayerSpawn);
	HookEvent("smoker_tongue_grab", Event_TongueGrab);
	HookEvent("smoker_tongue_release", Event_TongueRelease);
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
	ClearArrays();
}

public void Event_RoundEnd(Event event, const char[] name, bool dontBroadcast)
{
	AnnounceLongestGrab();
	ClearArrays();
}

public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast)
{
	int victim = GetClientOfUserId(event.GetInt("userid"));
	if (victim > 0 && IsClientInGame(victim) && IsPlayerAlive(victim))
	{
		g_bInGrab[victim] = false;
	}
}

public void Event_PlayerSpawn(Event event, const char[] name, bool dontBroadcast)
{
	int client = GetClientOfUserId(event.GetInt("userid"));
	if (client > 0 && IsClientInGame(client))
	{
		g_bInGrab[client] = false;
	}
}

public void Event_TongueGrab(Event event, const char[] name, bool dontBroadcast)
{
	int smoker = GetClientOfUserId(event.GetInt("userid"));
	int survivor = GetClientOfUserId(event.GetInt("attacker"));
	if (survivor > 0 && IsClientInGame(survivor) && IsPlayerAlive(survivor))
	{
		g_bInGrab[survivor] = true;
		g_fLongestGrab[survivor] = GetEngineTime();
	}
}

public void Event_TongueRelease(Event event, const char[] name, bool dontBroadcast)
{
	int survivor = GetClientOfUserId(event.GetInt("userid"));
	if (survivor > 0 && IsClientInGame(survivor) && IsPlayerAlive(survivor) && g_bInGrab[survivor])
	{
		float fGrabDuration = GetEngineTime() - g_fLongestGrab[survivor];
		if (fGrabDuration > g_fLongestGrab[g_iLongestSmoker[0]])
		{
			g_fLongestGrab[g_iLongestSmoker[0]] = fGrabDuration;
			g_iLongestSmoker[0] = survivor;
			g_iLongestSmoker[1] = GetClientOfUserId(event.GetInt("userid"));
		}
		g_bInGrab[survivor] = false;
	}
}

void AnnounceLongestGrab()
{
	if (g_iLongestSmoker[0] > 0)
	{
		char buffer[256];
		Format(buffer, sizeof(buffer), "%T %s", "has the longest Smoker grab", g_fLongestGrab[g_iLongestSmoker[0]], g_iLongestSmoker[0]);
		PrintToChatAll("%s", buffer);
	}
}

void ClearArrays()
{
	for (int i = 0; i <= MaxClients; i++)
	{
		g_fLongestGrab[i] = 0.0;
		g_iLongestSmoker[i] = 0;
		g_bInGrab[i] = false;
	}
}