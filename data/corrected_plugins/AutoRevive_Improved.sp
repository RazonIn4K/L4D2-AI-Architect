#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

#define REVIVE_DELAY 3.0
#define REVIVE_HEALTH 50

public Plugin myinfo =
{
    name = "Auto Revive",
    author = "Developer",
    description = "Automatically revives players after they become incapacitated",
    version = "1.1",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("player_incapacitated", Event_PlayerIncapacitated);
    HookEvent("revive_success", Event_ReviveSuccess);
    HookEvent("player_death", Event_PlayerDeath);
}

public void Event_PlayerIncapacitated(Event event, const char[] name, bool dontBroadcast)
{
    int survivor = GetClientOfUserId(event.GetInt("userid"));
    
    if (survivor > 0 && survivor <= MaxClients && IsClientInGame(survivor) && GetClientTeam(survivor) == 2)
    {
        if (IsClientIncapacitated(survivor))
        {
            PrintToChat(survivor, "\x04[Auto Revive] \x01You will be revived in %d seconds!", RoundToFloor(REVIVE_DELAY));
            CreateTimer(REVIVE_DELAY, Timer_RevivePlayer, GetClientUserId(survivor));
        }
    }
}

public void Event_ReviveSuccess(Event event, const char[] name, bool dontBroadcast)
{
    // Cancel pending revive if player was revived by someone else
    int subject = GetClientOfUserId(event.GetInt("subject"));
    if (subject > 0 && subject <= MaxClients)
    {
        // Note: We can't easily cancel specific timers, but the revive check will fail
    }
}

public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast)
{
    // Cancel pending revive if player died
    int userid = event.GetInt("userid");
    // Note: Timer will check if player is still incapacitated
}

public Action Timer_RevivePlayer(Handle timer, int userid)
{
    int survivor = GetClientOfUserId(userid);
    if (survivor > 0 && IsClientInGame(survivor) && IsClientIncapacitated(survivor))
    {
        ReviveClient(survivor);
        PrintToChat(survivor, "\x04[Auto Revive] \x01You have been revived!");
    }
    return Plugin_Continue;
}

bool IsClientIncapacitated(int client)
{
    return GetEntProp(client, Prop_Send, "m_isIncapacitated") == 1;
}

void ReviveClient(int client)
{
    // Method 1: Try using Left4DHooks if available (most reliable)
    if (GetFeatureStatus(FeatureType_Native, "L4D_ReviveSurvivor") == FeatureStatus_Available)
    {
        L4D_ReviveSurvivor(client);
        return;
    }
    
    // Method 2: Try using command flags (requires admin access)
    int flags = GetCommandFlags("give");
    if (flags != -1)
    {
        SetCommandFlags("give", flags & ~FCVAR_CHEAT);
        FakeClientCommand(client, "give health");
        SetCommandFlags("give", flags);
        return;
    }
    
    // Method 3: Direct prop manipulation (less reliable but works as fallback)
    SetEntProp(client, Prop_Send, "m_isIncapacitated", 0);
    SetEntProp(client, Prop_Send, "m_iHealth", REVIVE_HEALTH);
    
    // Also reset bleedout state
    SetEntProp(client, Prop_Send, "m_bIsOnThirdStrike", 0);
    SetEntDataFloat(client, FindSendPropInfo("CTerrorPlayer", "m_flHealthBuffer"), 0.0);
    
    // Force a weapons check to ensure player has their weapons back
    GiveDefaultWeapons(client);
}

void GiveDefaultWeapons(int client)
{
    // Ensure player has at least a pistol
    if (GetPlayerWeaponSlot(client, 0) == -1)
    {
        int flags = GetCommandFlags("give");
        if (flags != -1)
        {
            SetCommandFlags("give", flags & ~FCVAR_CHEAT);
            FakeClientCommand(client, "give pistol");
            SetCommandFlags("give", flags);
        }
    }
}

// Native declaration for Left4DHooks (if available)
native void L4D_ReviveSurvivor(int client);
