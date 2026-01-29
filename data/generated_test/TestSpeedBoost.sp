void OnPluginStart()
{

	LoadTranslations("common.phrases");
	LoadGameData();
	g_hCvarEnable = CreateConVar(PLUGIN_NAME ... "_enable", "1", "0=Plugin off, 1=Plugin on.", CVAR_FLAGS, true, 0.0, true, 1.0);
	CreateConVar(PLUGIN_NAME ... "_version", PLUGIN_VERSION, PLUGIN_NAME ... " version", FCVAR_NOTIFY | FCVAR_DONTRECORD);
	AutoExecConfig(true, PLUGIN_NAME);

	GetCvars();
	g_hCvarEnable.AddChangeHook(ConVarChanged_Cvars);
	CreateTimer(0.1, Timer_Cvars);

	// Get the right gamemode from the given client's team
	for (int i = 1; i <= MaxClients; i++) {
		if (!IsClientInGame(i)) continue;
		if (IsFakeClient(i) || !IsPlayerAlive(i)) continue;

		char SteamID[MAX_LINE_WIDTH];
		GetClientRankAuthString(i, SteamID, sizeof(SteamID));

		if (strcmp(SteamID, "custom_player") == 0) {
			// Custom player
			break;
		
}