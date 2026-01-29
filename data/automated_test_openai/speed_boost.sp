#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>
#include <left4dhooks>

#define BOOST_MULTIPLIER 1.3  // 30% speed boost
#define BOOST_DURATION 5.0

public Plugin myinfo =
{
    name = "Kill Speed Boost",
    author = "Developer",
    description = "Gives speed boost when killing special infected",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    // Use left4dhooks for special infected death detection
    L4D_OnInfectedDeath_Post(DetectKill);
    
    // Alternatively, you could hook all player deaths and check the attacker
    // HookEvent("player_death", Event_PlayerDeath);
}

public void DetectKill(int victim, int attacker, int weapon)
{
    // Check if victim is a special infected
    if (!IsValidEntIndex(victim) || GetClientTeam(victim) != 3)
        return;
    
    // Check if attacker is a survivor
    if (!IsValidEntIndex(attacker) || GetClientTeam(attacker) != 2)
        return;
    
    // Give speed boost
    GiveSpeedBoost(attacker);
}

void GiveSpeedBoost(int client)
{
    if (!IsClientInGame(client) || !IsPlayerAlive(client))
        return;
    
    // m_flLaggedMovementValue is a multiplier
    float original = GetEntPropFloat(client, Prop_Send, "m_flLaggedMovementValue");
    SetEntPropFloat(client, Prop_Send, "m_flLaggedMovementValue", original * BOOST_MULTIPLIER);
    
    PrintToChat(client, "\x04[Speed Boost]\x01 You feel a surge of adrenaline!");
    
    // Schedule to reset speed
    CreateTimer(BOOST_DURATION, Timer_ResetSpeed, GetClientUserId(client));
}

public Action Timer_ResetSpeed(Handle timer, int userid)
{
    int client = GetClientOfUserId(userid);
    if (client > 0 && IsClientInGame(client) && IsPlayerAlive(client))
    {
        SetEntPropFloat(client, Prop_Send, "m_flLaggedMovementValue", 1.0);
        PrintToChat(client, "\x04[Speed Boost]\x01 Speed boost has worn off.");
    }
    return Plugin_Continue;
}

// Optional: Hook all player deaths
public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast)
{
    int victim = GetClientOfUserId(event.GetInt("userid"));
    int attacker = GetClientOfUserId(event.GetInt("attacker"));
    
    // Only care about infected deaths
    if (victim <= 0 || GetClientTeam(victim) != 3)
        return;
    
    // Check if attacker is a survivor
    if (attacker <= 0 || GetClientTeam(attacker) != 2)
        return;
    
    // Check if we can determine the weapon
    int weapon = GetEntPropEnt(event.GetInt("weapon"), Prop_Send, "m_iEntityID");
    if (weapon <= 0)
        return;
    
    // Call our detection function
    DetectKill(victim, attacker, weapon);
}
