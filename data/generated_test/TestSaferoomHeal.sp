void OnPluginStart()
{

	g_hCvarEnable = CreateConVar("l4d_heal_survivors", "1", "Turn on this plugin? (0=Off, 1=On)", CVAR_FLAGS, true, 0.0, true, 1.0);
	g_hCvarHealing = FindConVar("z_heal");
	g_hCvarHealing.AddChangeHook(ConVarChanged_Healing);
	g_hCvarRespawnTime = FindConVar("respawn_time");
	g_hCvarRespawnTime.AddChangeHook(ConVarChanged_RespawnTime);

	GetCvars();
	g_hCvarEnable.AddChangeHook(ConVarChanged_Enable);
	g_hCvarHealing.AddChangeHook(ConVarChanged_Healing);
	g_hCvarRespawnTime.AddChangeHook(ConVarChanged_RespawnTime);

	AutoExecConfig(true, "l4d_heal_survivors");

}