#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>

#define PLUGIN_VERSION "1.0"

public Plugin myinfo =
{
	name = "Tank Damage Tracker",
	author = "Developer",
	description = "Tracks which survivor has dealt the most damage to Tanks",
	version = PLUGIN_VERSION,
	url = ""
};

int g_iMostDamageSurvivor = -1;
float g_fMostDamage = 0.0;

public void OnPluginStart()
{
	HookEvent("tank_killed", Event_TankKilled);
}

public void OnClientPutInServer(int client)
{
	if (IsClientInGame(client) && IsPlayerAlive(client) && GetClientTeam(client) == 2)
	{
		g_iMostDamageSurvivor = client;
		g_fMostDamage = 0.0;
	}
}

public void OnClientDisconnect(int client)
{
	if (client == g_iMostDamageSurvivor)
	{
		g_iMostDamageSurvivor = -1;
		g_fMostDamage = 0.0;
	}
}

public void OnClientDeath(int client, int killer, int weapon)
{
	if (GetClientTeam(client) == 2 && IsPlayerTank(killer))
	{
		g_iMostDamageSurvivor = -1;
		g_fMostDamage = 0.0;
	}
}

public void OnPlayerRunCmd(int client, int &buttons)
{
	if (buttons & IN_ATTACK && buttons & IN_ATTACK2)
	{
		if (GetClientTeam(client) == 2 && IsPlayerAlive(client))
		{
			float damage = GetClientWeaponDamage(client);
			if (damage > 0.0)
			{
				DealTankDamage(client, damage);
			}
		}
	}
}

float GetClientWeaponDamage(int client)
{
	int weapon = GetPlayerWeaponSlot(client, 0);
	if (weapon == -1) return 0.0;

	float damage = GetEntPropFloat(weapon, Prop_Send, "m_iDamage");
	return damage;
}

void DealTankDamage(int client, float damage)
{
	int tank = GetClientNearestTank(client);
	if (tank != -1)
	{
		float tankHealth = GetEntPropFloat(tank, Prop_Data, "m_iHealth");
		float tankArmor = GetEntPropFloat(tank, Prop_Data, "m_iArmor");

		if (tankHealth > 0.0 || tankArmor > 0.0)
		{
			if (tankHealth > damage)
			{
				SetEntPropFloat(tank, Prop_Data, "m_iHealth", tankHealth - damage);
			}
			else
			{
				SetEntPropFloat(tank, Prop_Data, "m_iHealth", 0.0);
			}

			if (tankArmor > damage)
			{
				SetEntPropFloat(tank, Prop_Data, "m_iArmor", tankArmor - damage);
			}
			else
			{
				SetEntPropFloat(tank, Prop_Data, "m_iArmor", 0.0);
			}

			float totalDamage = GetTankDamage(tank);
			if (totalDamage > g_fMostDamage)
			{
				g_fMostDamage = totalDamage;
				g_iMostDamageSurvivor = client;

				PrintToChatAll("\x04[Damage Tracker] \x01Most Damage: \x04%s \x01(%.1f)", GetClientName(g_iMostDamageSurvivor), g_fMostDamage);
			}
		}
	}
}

int GetClientNearestTank(int client)
{
	int tank = -1;
	float nearestDistance = 9999.0;

	for (int i = 1; i <= MaxClients; i++)
	{
		if (IsClientInGame(i) && GetClientTeam(i) == 3 && IsPlayerAlive(i))
		{
			float tankPosition[3];
			GetClientAbsOrigin(i, tankPosition);

			float clientPosition[3];
			GetClientAbsOrigin(client, clientPosition);

			float distance = GetVectorDistance(tankPosition, clientPosition);
			if (distance < nearestDistance)
			{
				nearestDistance = distance;
				tank = i;
			}
		}
	}

	return tank;
}

float GetTankDamage(int tank)
{
	return GetEntPropFloat(tank, Prop_Data, "m_iDamageTaken");
}

void Event_TankKilled(Event event, const char[] name, bool dontBroadcast)
{
	int tank = GetClientOfUserId(event.GetInt("userid"));
	if (tank != -1)
	{
		float totalDamage = GetTankDamage(tank);
		if (totalDamage > 0.0)
		{
			PrintToChatAll("\x04[Damage Tracker] \x01Tank killed! Most Damage: \x04%s \x01(%.1f)", GetClientName(g_iMostDamageSurvivor), g_fMostDamage);
		}
	}
}