#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>

#define PLUGIN_VERSION "1.0"

public Plugin myinfo =
{
	name = "Rescue Vehicle Tracker",
	author = "Developer",
	description = "Tracks how many times the rescue vehicle has been called",
	version = PLUGIN_VERSION,
	url = ""
};

int g_iRescueCalledCount = 0;

public void OnPluginStart()
{
	LoadTranslations("l4d2_rescue_tracker.phrases");
}

public void OnMapStart()
{
	g_iRescueCalledCount = 0;
}

public void OnRoundStart()
{
	g_iRescueCalledCount = 0;
}

public void OnRoundEnd()
{
	// Optionally, report the count at round end
	// PrintToChatAll("Rescue vehicle called %d times this round.", g_iRescueCalledCount);
}

public void OnRescueVehicleCalled()
{
	g_iRescueCalledCount++;
}

public int GetRescueCalledCount()
{
	return g_iRescueCalledCount;
}

public void PrintRescueStats()
{
	PrintToChatAll("Rescue vehicle has been called %d times.", g_iRescueCalledCount);
}

public Action Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast)
{
	int client = GetClientOfUserId(event.GetInt("userid"));
	if (client > 0 && IsClientInGame(client))
	{
		// Reset count on player death
		g_iRescueCalledCount = 0;
	}
	return Plugin_Continue;
}

public Action Event_RescueEnd(Event event, const char[] name, bool dontBroadcast)
{
	// Called when the rescue vehicle leaves
	OnRescueVehicleCalled();
	return Plugin_Continue;
}

public Action Event_RescueVehicleReady(Event event, const char[] name, bool dontBroadcast)
{
	// Called when the rescue vehicle is ready
	// This is a good time to track the call
	OnRescueVehicleCalled();
	return Plugin_Continue;
}