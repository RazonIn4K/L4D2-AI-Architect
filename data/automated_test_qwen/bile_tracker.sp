#pragma semicolon 1
#pragma newdecls required

#include <sdktools>
#include <dhooks>

#define BILE_TIMER 5.0

bool g_bHasBile[MAXPLAYERS + 1];
Handle g_hBileTimer[MAXPLAYERS + 1];

public Plugin myinfo =
{
    name = "Boomer Bile Tracker",
    author = "Developer",
    description = "Tracks when players get covered in Boomer bile",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("player_bot_replace", Event_PlayerBotReplace);
}

public void OnClientPutInServer(int client)
{
    g_bHasBile[client] = false;
    g_hBileTimer[client] = null;
}

public void Event_PlayerBotReplace(Event event, const char[] name, bool dontBroadcast)
{
    int bot = GetClientOfUserId(event.GetInt("bot"));
    
    if (bot > 0 && IsPlayerAlive(bot))
    {
        // Reset bile state on bot replacement
        g_bHasBile[bot] = false;
        if (g_hBileTimer[bot] != null)
        {
            KillTimer(g_hBileTimer[bot]);
            g_hBileTimer[bot] = null;
        }
        
        // Notify of bile removal
        PrintToChatAll("%N's bile has worn off.", bot);
    }
}

// Function to check and announce bile status
void CheckAndAnnounceBile(int client)
{
    if (!IsPlayerAlive(client) || !IsPlayerInGame(client))
        return;
    
    // Check if player is getting bile (logic simplified)
    bool bIsGettingBile = IsPlayerGettingBile(client);
    
    if (bIsGettingBile && !g_bHasBile[client])
    {
        g_bHasBile[client] = true;
        g_hBileTimer[client] = CreateTimer(BILE_TIMER, Timer_BileExpired, client);
        PrintToChat(client, "\x04[Boomer Bile]\x01 You have been covered in bile!");
    }
    else if (!bIsGettingBile && g_bHasBile[client])
    {
        g_bHasBile[client] = false;
        KillTimer(g_hBileTimer[client]);
        g_hBileTimer[client] = null;
        PrintToChat(client, "\x04[Boomer Bile]\x01 Your bile has worn off.");
    }
}

// Timer callback when bile wears off
public Action Timer_BileExpired(Handle timer, int client)
{
    g_bHasBile[client] = false;
    g_hBileTimer[client] = null;
    PrintToChat(client, "\x04[Boomer Bile]\x01 Your bile has worn off.");
    return Plugin_Handled;
}

// Example function to simulate checking for bile
bool IsPlayerGettingBile(int client)
{
    // Implement logic to detect bile on player
    // This could involve checking damage over time, specific hitboxes, etc.
    // For demonstration, we'll use a simple counter
    static float fBileCounter[MAXPLAYERS + 1];
    fBileCounter[client] += 0.1;  // Increment counter
    
    // Simulate bile effect after 3 seconds of being "wet"
    if (fBileCounter[client] >= 3.0)
        return true;
    
    return false;
}