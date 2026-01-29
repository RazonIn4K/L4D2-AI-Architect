#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

public Plugin myinfo =
{
    name = "Hunter Pounce Damage Tracker",
    author = "Developer",
    description = "Tracks and logs Hunter pounce damage in L4D2",
    version = "1.0",
    url = ""
};

float g_fPounceDamage[MAXPLAYERS + 1] = {0.0...0.0};
bool g_bInPounce[MAXPLAYERS + 1] = {false...false};

public void OnPluginStart()
{
    HookEvent("ability_use", Event_AbilityUse);
    HookEvent("player_hurt_concise", Event_PlayerHurtConcise);
}

public void OnClientDisconnect(int client)
{
    g_fPounceDamage[client] = 0.0;
    g_bInPounce[client] = false;
}

public void Event_AbilityUse(Event event, const char[] name, bool dontBroadcast)
{
    int victim = GetClientOfUserId(event.GetInt("userid"));
    int attacker = GetClientOfUserId(event.GetInt("entityid"));
    
    if (victim > 0 && victim <= MaxClients && IsClientInGame(victim) && GetClientTeam(victim) == 3)
    {
        if (attacker > 0 && attacker <= MaxClients && IsClientInGame(attacker) && GetClientTeam(attacker) == 2)
        {
            // Check ability name - note: use official L4D2 ability names
            char abilityName[64];
            event.GetString("ability", abilityName, sizeof(abilityName));
            
            if (strcmp(abilityName, "ability_lunge") == 0)
            {
                g_bInPounce[victim] = true;
                g_fPounceDamage[victim] = 0.0;  // Reset on new pounce
            }
        }
    }
}

public void Event_PlayerHurtConcise(Event event, const char[] name, bool dontBroadcast)
{
    int victim = GetClientOfUserId(event.GetInt("userid"));
    int attacker = GetClientOfUserId(event.GetInt("attacker"));
    float damage = event.GetFloat("dmg_health");
    
    if (victim > 0 && victim <= MaxClients && IsClientInGame(victim) && GetClientTeam(victim) == 3)
    {
        if (g_bInPounce[victim])
        {
            g_fPounceDamage[victim] += damage;
            PrintToChatAll("Hunter pounced %N for %.2f damage (Total: %.2f)", victim, damage, g_fPounceDamage[victim]);
        }
    }
}