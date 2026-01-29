#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>
#include <left4dhooks>

#define PLUGIN_VERSION "1.0"

public Plugin myinfo =
{
	name = "Panic Timer",
	description = "Triggers a panic event if survivors stay in one area too long",
	version = PLUGIN_VERSION,
	author = "AI Developer",
};

#define AREA_CHECK_INTERVAL 30.0 // seconds
#define PANIC_THRESHOLD 120.0 // seconds

float g_fLastCheckTime = 0.0;
float g_fAreaStayTime = 0.0;

bool g_bInPanic = false;

public void OnPluginStart()
{
	HookEvent("round_start", Event_RoundStart, EventHookMode_PostNoCopy);
	HookEvent("round_end", Event_RoundEnd, EventHookMode_PostNoCopy);
	HookEvent("map_transition", Event_RoundEnd, EventHookMode_PostNoCopy); // Coop only, when leaving a map in a campaign
	HookEvent("mission_lost", Event_RoundEnd, EventHookMode_PostNoCopy); // Survival/Scavenge only, when ending a round
	HookEvent("finale_vehicle_leaving", Event_RoundEnd, EventHookMode_PostNoCopy); // Finale only, when the rescue vehicle is leaving

	CreateTimer(AREA_CHECK_INTERVAL, Timer_CheckArea, _, TIMER_REPEAT);
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
	g_fLastCheckTime = 0.0;
	g_fAreaStayTime = 0.0;
	g_bInPanic = false;
}

public void Event_RoundEnd(Event event, const char[] name, bool dontBroadcast)
{
	g_fLastCheckTime = 0.0;
	g_fAreaStayTime = 0.0;
	g_bInPanic = false;
}

Action Timer_CheckArea(Handle timer)
{
	if (GetGameTime() - g_fLastCheckTime < AREA_CHECK_INTERVAL)
		return Plugin_Continue;

	int iSurvivorsInArea = 0;

	for (int i = 1; i <= MaxClients; i++)
	{
		if (IsClientInGame(i) && GetClientTeam(i) == 2 && IsPlayerAlive(i))
		{
			float fPos[3];
			GetClientAbsOrigin(i, fPos);

			if (IsInArea(fPos))
			{
				iSurvivorsInArea++;
			}
		}
	}

	if (iSurvivorsInArea > 0)
	{
		g_fAreaStayTime += AREA_CHECK_INTERVAL;

		if (g_fAreaStayTime >= PANIC_THRESHOLD && !g_bInPanic)
		{
			TriggerPanic();
		}
	}
	else
	{
		g_fAreaStayTime = 0.0;
	}

	g_fLastCheckTime = GetGameTime();

	return Plugin_Continue;
}

bool IsInArea(float fPos[3])
{
	// Define the area where you want to check for survivors
	float fAreaCenter[3] = { -1000.0, -1000.0, 0.0 }; // Example area center
	float fAreaRadius = 500.0; // Example area radius

	float fDist = GetVectorDistance(fPos, fAreaCenter);
	return (fDist <= fAreaRadius);
}

void TriggerPanic()
{
	g_bInPanic = true;

	// Trigger the panic event
	SDKCall(SDKHook_GetEvent("panic_start"), NULL, NULL);

	// Play panic sound
	EmitSoundToAll("ambient/levels/labs/electric_explosion1.wav");

	// Print to chat
	PrintToChatAll("\x04[WARNING] \x01A panic event has been triggered!");

	// Add more effects if needed
}

float GetVectorDistance(float vec1[3], float vec2[3])
{
	float fDist = 0.0;
	for (int i = 0; i < 3; i++)
	{
		float fDiff = vec1[i] - vec2[i];
		fDist += fDiff * fDiff;
	}
	return SquareRoot(fDist);
}