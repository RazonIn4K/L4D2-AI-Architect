#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>

public Plugin myinfo =
{
    name = "Saferoom Heal",
    author = "Developer",
    description = "Automatically heals survivors to full health when they enter a saferoom",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("round_start", Event_RoundStart, EventHookMode_PostNoCopy);
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
    // Hook the saferoom door opening event
    HookEvent("door_closed", Event_DoorClosed);
}

public void Event_DoorClosed(Event event, const char[] name, bool dontBroadcast)
{
    // Check if the door is a saferoom door
    int entity = GetClientOfUserId(event.GetInt("userid"));
    if (entity > 0 && IsClientInGame(entity) && GetClientTeam(entity) == 2)
    {
        // Check if the entity is a door
        int door = EntIndexToEntRef(entity);
        if (door != INVALID_ENT_REFERENCE && IsValidEntity(door))
        {
            // Check if the door is a saferoom door
            if (GetEntProp(door, Prop_Send, "m_bIsSaferoomDoor"))
            {
                // Heal all survivors in the saferoom
                HealAllSurvivors();
            }
        }
    }
}

void HealAllSurvivors()
{
    for (int i = 1; i <= MaxClients; i++)
    {
        if (IsClientInGame(i) && GetClientTeam(i) == 2 && IsPlayerAlive(i))
        {
            SetEntProp(i, Prop_Send, "m_iHealth", 100);
            SetEntProp(i, Prop_Send, "m_iMaxHealth", 100);
            SetEntProp(i, Prop_Send, "m_iHealthBuffer", 0);
            SetEntProp(i, Prop_Send, "m_iHealthBufferCount", 0);

            // Send a heal sound
            EmitSoundToClient(i, "ui/survivor_heal.wav");
            PrintToChat(i, "\x04[HEAL] \x01You have been healed to full health!");
        }
    }
}