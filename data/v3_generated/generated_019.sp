#if defined _reduced_infected_spawn_rates_in_saferooms_included
	#endinput
#endif
#define _reduced_infected_spawn_rates_in_saferooms_included

#define SPAWN_DELAY 5.0

// 0 = normal, 1 = saferoom
static int g_iSpawnDelayMode[2];

void _reduced_infected_spawn_rates_in_saferooms_OnPluginStart()
{
	g_iSpawnDelayMode[0] = CreateConVarEx("sm_spawn_delay_mode_in_saferooms", "0", "Turn on saferoom spawn delay", _, true, 0.0, true, 1.0);
	g_iSpawnDelayMode[1] = CreateConVarEx("sm_saferoom_spawn_delay", "5.0", "Change the saferoom spawn delay", _, true, 0.0);
}

void _reduced_infected_spawn_rates_in_saferooms_OnMapStart()
{
	g_iSpawnDelayMode[0].IntValue = g_iSpawnDelayMode[0].IntValue;
	g_iSpawnDelayMode[1].FloatValue = g_iSpawnDelayMode[1].FloatValue;
}

void _reduced_infected_spawn_rates_in_saferooms_OnConVarChanged(ConVar hCvar, const char[] sOldVal, const char[] sNewVal)
{
	g_iSpawnDelayMode[0].IntValue = g_iSpawnDelayMode[0].IntValue;
	g_iSpawnDelayMode[1].FloatValue = g_iSpawnDelayMode[1].FloatValue;
}

void _reduced_infected_spawn_rates_in_saferooms_OnClientPutInServer(int iClient)
{
	if (IsFakeClient(iClient))
		return;

	g_iSpawnDelayMode[0].IntValue = g_iSpawnDelayMode[0].IntValue;
	g_iSpawnDelayMode[1].FloatValue = g_iSpawnDelayMode[1].FloatValue;
}

public void OnPluginEnd()
{
	// Restore default values
	g_iSpawnDelayMode[0].IntValue = 0;
	g_iSpawnDelayMode[1].FloatValue = 5.0;
}

public Action OnSpawningInfectedPost(int iBot, int iType)
{
	if (g_iSpawnDelayMode[0].IntValue == 0)
		return Plugin_Continue;

	if (IsClientInSaferoom(iBot))
	{
		SetGlobalFloat("z_spawn_delay", g_iSpawnDelayMode[1].FloatValue + GetGameTime());
	}
	
	return Plugin_Continue;
}

bool IsClientInSaferoom(int iClient)
{
	if (!IsClientInGame(iClient) || IsFakeClient(iClient))
		return false;

	if (GetClientTeam(iClient) != 2)
		return false;

	return GetEntProp(iClient, Prop_Send, "m_isInSafeRoom") == 1;
}
