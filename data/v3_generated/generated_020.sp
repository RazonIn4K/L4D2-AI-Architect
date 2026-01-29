#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>
#include <l4d2_timed>

#define PLUGIN_VERSION "1.0"

public Plugin myinfo =
{
	name = "Chapter Time Announcer",
	author = "Developer",
	description = "Announces the time taken to complete each chapter",
	version = PLUGIN_VERSION,
	url = ""
};

float g_fChapterStartTime[MAP_MAXSLOTS];
int g_iChapterNumber[MAP_MAXSLOTS];

public void OnPluginStart()
{
	HookEvent("round_end", Event_RoundEnd);
	HookEvent("map_transition", Event_MapTransition);
	HookEvent("player_spawn", Event_PlayerSpawn);
	HookEvent("player_death", Event_PlayerDeath);
	HookEvent("player_disconnect", Event_PlayerDisconnect);
	HookEvent("player_bot_replace", Event_PlayerBotReplace);
	HookEvent("bot_player_replace", Event_BotPlayerReplace);
}

public void OnMapStart()
{
	int mapIndex = GetMapIndex();
	g_fChapterStartTime[mapIndex] = GetEngineTime();
	g_iChapterNumber[mapIndex] = 0;
}

public void Event_RoundEnd(Event event, const char[] name, bool dontBroadcast)
{
	int mapIndex = GetMapIndex();
	g_fChapterStartTime[mapIndex] = GetEngineTime();
	g_iChapterNumber[mapIndex]++;
}

public void Event_MapTransition(Event event, const char[] name, bool dontBroadcast)
{
	int mapIndex = GetMapIndex();
	float fEndTime = GetEngineTime();
	float fStartTime = g_fChapterStartTime[mapIndex];
	float fDuration = fEndTime - fStartTime;

	char buffer[64];
	Format(buffer, sizeof(buffer), "Chapter %d completed in %.2f seconds.", g_iChapterNumber[mapIndex], fDuration);
	PrintToChatAll("\x04[CHAPTER TIME] \x01%s", buffer);
}

public void Event_PlayerSpawn(Event event)
{
	int mapIndex = GetMapIndex();
	g_fChapterStartTime[mapIndex] = GetEngineTime();
}

public void Event_PlayerDeath(Event event)
{
	int mapIndex = GetMapIndex();
	g_fChapterStartTime[mapIndex] = GetEngineTime();
}

public void Event_PlayerDisconnect(Event event)
{
	int mapIndex = GetMapIndex();
	g_fChapterStartTime[mapIndex] = GetEngineTime();
}

public void Event_PlayerBotReplace(Event event)
{
	int mapIndex = GetMapIndex();
	g_fChapterStartTime[mapIndex] = GetEngineTime();
}

public void Event_BotPlayerReplace(Event event)
{
	int mapIndex = GetMapIndex();
	g_fChapterStartTime[mapIndex] = GetEngineTime();
}