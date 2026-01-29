#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>
#include <left4dhooks>

public Plugin myinfo =
{
    name = "Spitter Acid Pools Duration Modifier",
    author = "Developer",
    description = "Increases the duration of Spitter acid pools",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("spitter_death", Event_SpitterDeath);
}

public void Event_SpitterDeath(Event event, const char[] name, bool dontBroadcast)
{
    int client = GetClientOfUserId(event.GetInt("userid"));
    if (client > 0 && IsClientInGame(client) && GetClientTeam(client) == 3)
    {
        // Spitter died, increase acid pool duration
        int acidPool = CreateEntityByName("env_spray");
        if (acidPool != -1)
        {
            DispatchSpawn(acidPool);
            SetVariantString("acid");
            AcceptEntityInput(acidPool, "SetTexture");
            SetEntPropFloat(acidPool, Prop_Send, "m_flLifetime", 12.0); // Increase lifetime by 50%
            SetEntPropFloat(acidPool, Prop_Send, "m_flStartTime", GetGameTime());
            SetEntPropFloat(acidPool, Prop_Send, "m_flLastTime", GetGameTime() + 12.0);
            TeleportEntity(acidPool, GetClientAbsOrigin(client), NULL_VECTOR, NULL_VECTOR);
            AcceptEntityInput(acidPool, "StartAnimation");
        }
    }
}