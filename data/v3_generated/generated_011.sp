#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>

#define GNOME_MODEL "models/props_l4d/gnome.mdl"
#define SPEED_BONUS 1.5

public Plugin myinfo =
{
    name = "Gnome Speed",
    author = "Developer",
    description = "Increases speed while carrying a gnome",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    RegConsoleCmd("sm_speedgnome", Cmd_SpeedGnome, "Toggle speed bonus on gnome carry");
}

public Action Cmd_SpeedGnome(int client, int args)
{
    if (client <= 0 || !IsClientInGame(client) || IsFakeClient(client))
        return Plugin_Handled;

    if (args == 0)
    {
        PrintToChat(client, "[SM] Usage: sm_speedgnome <0|1>");
        return Plugin_Handled;
    }

    int toggle = GetCmdArgInt(1);
    if (toggle == 1)
    {
        SetSpeedBonus(client, true);
        PrintToChat(client, "[SM] Speed bonus enabled.");
    }
    else if (toggle == 0)
    {
        SetSpeedBonus(client, false);
        PrintToChat(client, "[SM] Speed bonus disabled.");
    }
    else
    {
        PrintToChat(client, "[SM] Invalid argument. Use 0 to disable, 1 to enable.");
    }

    return Plugin_Handled;
}

void SetSpeedBonus(int client, bool enable)
{
    if (enable)
    {
        HookEntity(client, SDKHook_OnTakeDamage, OnTakeDamage);
        HookEntity(client, SDKHook_OnEntityCreated, OnEntityCreated);
    }
    else
    {
        UnhookEntity(client, SDKHook_OnTakeDamage, OnTakeDamage);
        UnhookEntity(client, SDKHook_OnEntityCreated, OnEntityCreated);
    }
}

void OnEntityCreated(int entity)
{
    char model[PLATFORM_MAX_PATH];
    GetEntityModel(entity, model, sizeof(model));

    if (strcmp(model, GNOME_MODEL) == 0)
    {
        int owner = GetEntPropEnt(entity, Prop_Send, "m_hOwnerEntity");
        if (owner > 0 && IsClientInGame(owner))
        {
            SetEntPropFloat(entity, Prop_Send, "m_flLaggedMovementValue", SPEED_BONUS);
            SetEntPropFloat(entity, Prop_Send, "m_flMaxSpeed", 320.0 * SPEED_BONUS);
        }
    }
}

Action OnTakeDamage(int victim, int &attacker, int &inflictor, float &damage, int &damagetype)
{
    if (victim > 0 && IsClientInGame(victim))
    {
        if (IsCarryingGnome(victim))
        {
            SetEntPropFloat(victim, Prop_Send, "m_flLaggedMovementValue", SPEED_BONUS);
            SetEntPropFloat(victim, Prop_Send, "m_flMaxSpeed", 320.0 * SPEED_BONUS);
        }
    }

    return Plugin_Continue;
}

bool IsCarryingGnome(int client)
{
    int weapon = GetPlayerWeaponSlot(client, 0);
    if (weapon <= 0) return false;

    char model[PLATFORM_MAX_PATH];
    GetEntityModel(weapon, model, sizeof(model));

    return strcmp(model, GNOME_MODEL) == 0;
}