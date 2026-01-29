#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>

public Plugin myinfo =
{
    name = "Tank Health Announcement",
    author = "Developer",
    description = "Announces Tank health thresholds",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("tank_spawn", Event_TankSpawn);
    HookEvent("tank_health_changed", Event_TankHealthChanged);
}

float g_fTankMaxHealth = 0.0;
float g_fTankCurrentHealth = 0.0;
bool g_bTankSpawned = false;

void Event_TankSpawn(Event event, const char[] name, bool dontBroadcast)
{
    g_bTankSpawned = true;
    g_fTankMaxHealth = GetTankMaxHealth();
    g_fTankCurrentHealth = g_fTankMaxHealth;

    // Reset health announcement
    ResetHealthAnnouncement();
}

void Event_TankHealthChanged(Event event, const char[] name, bool dontBroadcast)
{
    if (!g_bTankSpawned) return;

    float fNewHealth = event.GetFloat("new_health");
    if (fNewHealth < 0.0) fNewHealth = 0.0;

    // Update current health
    g_fTankCurrentHealth = fNewHealth;

    // Check health thresholds
    CheckHealthThresholds();
}

void CheckHealthThresholds()
{
    float fHealthPercentage = (g_fTankCurrentHealth / g_fTankMaxHealth) * 100.0;

    if (fHealthPercentage <= 25.0 && !HasAnnounced(25.0))
    {
        PrintToChatAll("\x04[TANK] \x01Tank is at 25%% health!");
        SetAnnounced(25.0);
    }
    else if (fHealthPercentage <= 50.0 && !HasAnnounced(50.0))
    {
        PrintToChatAll("\x04[TANK] \x01Tank is at 50%% health!");
        SetAnnounced(50.0);
    }
    else if (fHealthPercentage <= 75.0 && !HasAnnounced(75.0))
    {
        PrintToChatAll("\x04[TANK] \x01Tank is at 75%% health!");
        SetAnnounced(75.0);
    }
}

bool HasAnnounced(float percentage)
{
    return (percentage == 25.0 && GetEntPropFloat(GetTank(), Prop_Send, "m_health") <= 25.0) ||
           (percentage == 50.0 && GetEntPropFloat(GetTank(), Prop_Send, "m_health") <= 50.0) ||
           (percentage == 75.0 && GetEntPropFloat(GetTank(), Prop_Send, "m_health") <= 75.0);
}

void SetAnnounced(float percentage)
{
    // Store the announced percentage
}

void ResetHealthAnnouncement()
{
    // Reset the health announcement status
}

float GetTankMaxHealth()
{
    // Get the tank's max health from the cvars or game data
    return 3000.0; // Default value
}

int GetTank()
{
    // Get the tank entity index
    return -1; // Default value
}