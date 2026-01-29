#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>
#include <left4dhooks>

#define PLUGIN_VERSION "1.0"

public Plugin myinfo =
{
	name = "Charger Spawner",
	author = "Developer",
	description = "Spawns a Charger behind the last survivor every 3 minutes",
	version = PLUGIN_VERSION,
	url = ""
};

#define SPAWN_INTERVAL 180.0 // 3 minutes

public void OnPluginStart()
{
	RegConsoleCmd("sm_spawncharger", Cmd_SpawnCharger);
}

public void OnMapStart()
{
	CreateTimer(SPAWN_INTERVAL, Timer_SpawnCharger, _, TIMER_REPEAT);
}

Action Cmd_SpawnCharger(int client, int args)
{
	if (client == 0 || !IsClientInGame(client) || GetClientTeam(client) != 2)
	{
		ReplyToCommand(client, "[SM] This command can only be used by infected players.");
		return Plugin_Handled;
	}

	SpawnCharger(client);
	return Plugin_Handled;
}

void SpawnCharger(int survivor)
{
	if (!IsClientInGame(survivor) || GetClientTeam(survivor) != 2 || !IsPlayerAlive(survivor))
		return;

	float pos[3];
	GetClientAbsOrigin(survivor, pos);
	pos[0] += RandomFloat(-500.0, -300.0);
	pos[1] += RandomFloat(-500.0, 500.0);
	pos[2] = GetGroundHeight(pos[0], pos[1]) + 1.0;

	int charger = CreateEntityByName("infcharger");
	if (charger == -1)
	{
		LogError("Failed to create Charger entity.");
		return;
	}

	DispatchKeyValue(charger, "origin", Format("%f %f %f", pos[0], pos[1], pos[2]));
	DispatchSpawn(charger);
	AcceptEntityInput(charger, "setteam", 3);
	AcceptEntityInput(charger, "sethealth", "100");
	AcceptEntityInput(charger, "setmaxhealth", "100");

	int target = FindTargetForCharger(charger, survivor);
	if (target != -1)
	{
		AcceptEntityInput(charger, "settarget", target);
	}

	EmitSoundToAll("ambient/levels/labs/electric_explosion1.wav");
}

int FindTargetForCharger(int charger, int survivor)
{
	int infectedCount = 0;
	int infected[32];

	for (int i = 1; i <= MaxClients; i++)
	{
		if (IsClientInGame(i) && GetClientTeam(i) == 3 && IsPlayerAlive(i))
		{
			infected[infectedCount++] = i;
		}
	}

	if (infectedCount == 0)
	{
		return -1;
	}

	int target = infected[RandomInt(0, infectedCount - 1)];
	return target;
}

Action Timer_SpawnCharger(Handle timer)
{
	int lastSurvivor = GetLastSurvivor();
	if (lastSurvivor != -1)
	{
		SpawnCharger(lastSurvivor);
	}
	return Plugin_Continue;
}

int GetLastSurvivor()
{
	int last = -1;
	for (int i = 1; i <= MaxClients; i++)
	{
		if (IsClientInGame(i) && GetClientTeam(i) == 2 && IsPlayerAlive(i))
		{
			last = i;
		}
	}
	return last;
}

float GetGroundHeight(float x, float y)
{
	float z = 0.0;
	TraceLine(x, y, x, y - 1000.0, z, z);
	return z;
}