#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <left4dhooks>

public Plugin myinfo =
{
	name = "Kill Tracker",
	author = "Harry Potter",
	description = "Track kills by in-game client.",
	version = "1.0",
	url = "https://steamcommunity.com/profiles/76561198026784913/"
}

ConVar g_hCvarEnable;
int g_iClientKills[MAXPLAYERS + 1];

void OnPluginStart()
{
	g_hCvarEnable = CreateConVar("killtracker_enable", "1", "Enable Kill Tracker? (0=Off, 1=On)", CVAR_FLAGS, true, 0.0, true, 1.0);
}

stock void OnClientPostAdminCheck(int client)
{
	if (!IsClientInGame(client)) { return; }

	if (g_bLateLoad) { return; }

	// Kill tracker is off, skip.
	if (!g_hCvarEnable.BoolValue) {
		return;
	}

	// Kill tracker is on, start tracking.
	g_hCvarEnable.BoolValue = false;
	g_bLateLoad = true;
	CreateTimer(0.5, Timer_ClientKills, _, TIMER_REPEAT);
}

stock Action Timer_ClientKills(Handle timer)
{
	g_iClientKills[GetClientUserId(GetClientOfUserId(timer))]++;
	return Plugin_Continue;
}

stock bool IsClientInGame(int client)
{
	return (GetClientTeam(client) == TEAM_SURVIVORS && IsClientConnected(client));
}

stock void AddToClientKills(int client, int kill)
{
	if (IsClientBot(client)) { return; }

	if (!IsClientInGame(client)) { return; }

	if (g_iClientKills[client] == 0) {
		g_iClientKills[client] = 1;
		OnPlayerConnect(client);
	} else {
		g_iClientKills[client]++;
	}
}

stock void RemoveFromClientKills(int client)
{
	if (IsClientBot(client))