#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>
#include <l4d2_infected_rides>

#define PLUGIN_VERSION "1.0"

public Plugin myinfo =
{
	name = "Jockey Ride Damage",
	author = "AI",
	description = "Increases damage based on ride duration",
	version = PLUGIN_VERSION,
	url = "https://example.com"
};

ConVar g_hCvarRideDamageFactor;

public void OnPluginStart()
{
	g_hCvarRideDamageFactor = CreateConVar("jockey_ride_damage_factor", "1.0", "Factor to multiply ride duration for damage", FCVAR_NOTIFY, true, 0.0);
}

public void OnClientPutInServer(int client)
{
	g_hCvarRideDamageFactor.AddChangeHook(ConVarChanged_Cvars);
}

void ConVarChanged_Cvars(ConVar hCvar, const char[] sOldVal, const char[] sNewVal)
{
	GetCvars();
}

void GetCvars()
{
	g_hCvarRideDamageFactor.FloatValue;
}

public Action OnRideEnd(int jockey, int victim, float rideDuration)
{
	if (rideDuration <= 0.0)
		return Plugin_Continue;

	float damageFactor = g_hCvarRideDamageFactor.FloatValue;
	float damage = RoundFloat(rideDuration * damageFactor);

	if (damage > 0.0)
	{
		// Apply damage to survivor
		TakeDamage(victim, jockey, damage, DMG_SLASH);
	}
	
	return Plugin_Continue;
}