# L4D2 Model Comparison Report

Generated: 2026-01-09T15:28:14.479559

## Executive Summary

| Model | Total Score | Pass Rate | Syntax | API Usage | Completeness | Avg Time |
|-------|-------------|-----------|--------|-----------|--------------|----------|
| l4d2-code-v10plus:latest | 68.1% | 80.0% | 66.0% | 60.7% | 80.0% | 53352ms |

**Best Performing Model:** l4d2-code-v10plus:latest (68.1%)

## Test Results by Prompt

### Test: heal_survivors

**Prompt:** Write a SourcePawn function that heals all survivors to full health.

**Expected APIs:** GetClientHealth, SetEntityHealth, GetClientTeam, MaxClients, IsClientInGame

| Model | Score | Syntax | API | Complete | APIs Found | APIs Missing |
|-------|-------|--------|-----|----------|------------|--------------|
| l4d2-code-v10plus:latest | 59.0% (PASS) | 30.0% | 50.0% | 100.0% | MaxClients, IsClientInGame | GetClientHealth, SetEntityHealth, GetClientTeam |

### Test: spawn_tank_timer

**Prompt:** Write a SourcePawn plugin that spawns a tank every 5 minutes using a timer.

**Expected APIs:** CreateTimer, CheatCommand, z_spawn

| Model | Score | Syntax | API | Complete | APIs Found | APIs Missing |
|-------|-------|--------|-----|----------|------------|--------------|
| l4d2-code-v10plus:latest | 81.3% (PASS) | 100.0% | 53.3% | 100.0% | CreateTimer | CheatCommand, z_spawn |

### Test: damage_event

**Prompt:** Write a SourcePawn event hook for player_hurt that shows damage dealt.

**Expected APIs:** HookEvent, GetEventInt, PrintToChat

| Model | Score | Syntax | API | Complete | APIs Found | APIs Missing |
|-------|-------|--------|-----|----------|------------|--------------|
| l4d2-code-v10plus:latest | 100.0% (PASS) | 100.0% | 100.0% | 100.0% | HookEvent, GetEventInt, PrintToChat |  |

### Test: admin_teleport

**Prompt:** Create a SourcePawn admin command that teleports a player to the command user.

**Expected APIs:** RegAdminCmd, GetCmdArg, GetClientAbsOrigin, TeleportEntity

| Model | Score | Syntax | API | Complete | APIs Found | APIs Missing |
|-------|-------|--------|-----|----------|------------|--------------|
| l4d2-code-v10plus:latest | 100.0% (PASS) | 100.0% | 100.0% | 100.0% | RegAdminCmd, GetCmdArg, GetClientAbsOrigin... |  |

### Test: vscript_director

**Prompt:** Write a VScript DirectorOptions table that increases common infected spawns.

**Expected APIs:** DirectorOptions, CommonLimit, MobSpawnMinTime

| Model | Score | Syntax | API | Complete | APIs Found | APIs Missing |
|-------|-------|--------|-----|----------|------------|--------------|
| l4d2-code-v10plus:latest | 0.0% (FAIL) | 0.0% | 0.0% | 0.0% |  | DirectorOptions, CommonLimit, MobSpawnMinTime |

## Generated Code Samples

### heal_survivors

#### l4d2-code-v10plus:latest

```sourcepawn
// Heals all survivors to full health
function HealAllSurvivors()
{
    local maxClients = GetMaxClients();
    
    for (local i = 1; i <= maxClients; i++)
    {
        if (IsClientInGame(i) && IsPlayerAlive(i))
        {
            SetEntPropFloat(i, Prop_Send, "m_iHealth", 100.0);
        }
    }
    
    PrintToServer("[HEAL] All survivors have been healed to full health.");
}

// Check if a client is an infected
function IsClientInfected(client)
{
    return GetEntProp(client, Prop_Send, "m_iTeamNum") == 3;
}

// Check if a client is a survivor
function IsClientSurvivor(client)
{
    return GetEntProp(client, Prop_Send, "m_iTeamNum") == 2;
}

// Get a player's current health
function GetPlayerHealth(client)
{
    if (!IsClientInGame(client))
        return 0.0;
    
    local health = GetEntPropFloat(client, Prop_Send, "m_iHealth");
    
    // Apply damage bonus
    local damageBonus = GetDamageBonus(client);
    if (damageBonus > 0.0)
    {
        health += damageBonus;
        PrintToServer("[HEAL] Player %N has received a damage bonus of %.1f health.", client, damageBonus);
    }
    
    return health;
}

// Get a player's maximum health
function GetPlayerMaxHealth(client)
{
    if (!IsClientInGame(client))
        return 0.0;
    
    local maxHealth = GetEntPropFloat(client, Prop_Send, "m_iMaxHealth");
    return maxHealth;
}

// Apply damage bonus to a player's health
function SetDamageBonus(client)
{
    if (!IsClientInGame(client))
        return 0.0;
    
    // Calculate damage bonus based on health percentage
    local healthRatio = GetHealthRatio(client);
    local damageBonus = healthRatio * 0.1; // Default: 10% bonus
    
    if (damageBonus > 0)
    {
        PrintToServer("[HEAL] Player %N has received a damage bonus of %.1f health.", client, damageBonus);
    }
}

// Get health ratio between two players
function GetHealthRatio(playerA, playerB)
{
    if (playerA <= 0 || playerB <= 0 || playerA == playerB)
        return 0.0;
    
    // Ca
... (truncated)
```

### spawn_tank_timer

#### l4d2-code-v10plus:latest

```sourcepawn
// Spawn a Tank every 5 minutes using a timer
#include <sourcemod>
#include <sdktools>

public Plugin myinfo = 
{
    name = "Dynamic Tank Spawner",
    author = "Expert SourcePawn Developer",
    description = "Spawns a Tank every 5 minutes using a timer.",
    version = "1.0",
    url = ""
};

Handle g_hTimer;
const Float:INTERVAL = 5.0; // Base interval for the timer

public void OnPluginStart()
{
    RegConsoleCmd("sm_spawntank", Command_SpawnTank);
    
    HookEvent("round_start", Event_RoundStart, EventHookMode_PostNoCopy);
}

public void OnClientDisconnect(int client)
{
    if (IsClientInGame(client))
    {
        // Remove any reference to the disconnecting client
        g_hTimer = INVALID_HANDLE;
    }
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
    // Start the timer if it's not already running
    if (g_hTimer == INVALID_HANDLE)
    {
        g_hTimer = CreateTimer(INTERVAL, Timer_SpawnTank, _, TIMER_REPEAT | TIMER_FLAG_NO_MAPCHANGE);
    }
}

public Action Timer_SpawnTank(Handle timer, any data)
{
    // Logic to spawn the tank:
    int tankIndex = CreateEntityByName("tank");
    
    if (tankIndex != -1)
    {
        DispatchKeyValue(tankIndex, "solid", "6");
        DispatchKeyValue(tankIndex, "tank_spawn_interval", INTERVAL);
        TeleportEntity(tankIndex, null_vector, null_vector, null_vector);
    }
    
    return Plugin_Continue;
}

public Action Command_SpawnTank(int client, int args)
{
    if (IsClientInGame(client))
    {
        PrintToChat(client, "You have spawned a Tank!");
        // Reload the timer when a command is used
        g_hTimer = CreateTimer(INTERVAL, Timer_SpawnTank, _, TIMER_REPEAT | TIMER_FLAG_NO_MAPCHANGE);
    }
    else
    {
        PrintToChat(client, "You must be in-game to use this command.");
    }
    
    return Plugin_Handled;
}
```

### damage_event

#### l4d2-code-v10plus:latest

```sourcepawn
// Player Hurt - Show Damage Dealt
// Left 4 Dead 2 - SourceMod API

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>

public Plugin myinfo = 
{
    name = "Player Hurt - Show Damage Dealt",
    author = "Expert SourcePawn Developer",
    description = "Displays damage dealt to a player.",
    version = "1.0",
    url = ""
};

public void OnPluginStart()
{
    HookEvent("player_hurt", Event_PlayerHurt, EventHookMode_Post);
}

public Action:Event_PlayerHurt(Handle:event, const String:name[], bool:dontBroadcast)
{
    int victim = GetEventInt(event, "userid");
    int attacker = GetEventInt(event, "attacker");
    int damage = GetEventInt(event, "damage");

    int clientVictim = GetClientOfUserId(victim);
    int clientAttacker = GetClientOfUserId(attacker);

    if (IsValidClient(clientVictim) && IsValidClient(clientAttacker))
    {
        PrintToChatAll("Player %N was dealt %.1f damage!", clientVictim, damage);
    }
    
    return Plugin_Continue;
}

bool:IsValidClient(int client)
{
    return (client > 0 && client <= MaxClients && IsClientInGame(client));
}
```

## Scoring Methodology

### Syntax Score (30%)
- Has proper includes (+2)
- Has proper structure (functions/callbacks) (+3)
- Has semicolons where required (+1)
- Has balanced braces (+2)
- Overall valid code (+2)

### API Usage Score (40%)
- Percentage of expected APIs found
- Bonus for L4D2-specific APIs (+10%)
- Bonus for correct function signatures (+10%)

### Completeness Score (30%)
- Has function body (+3)
- Has error handling (+2)
- Has comments (+1)
- Code length >= 10 lines (+2) or >= 5 lines (+1)
- Is complete solution (+2)

---
*Generated by L4D2-AI-Architect model comparison tool*