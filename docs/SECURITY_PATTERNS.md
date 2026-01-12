# L4D2 SourcePawn Security Patterns

**Date**: January 7, 2026
**Purpose**: Comprehensive RED/BLUE team security guide for L4D2 SourcePawn plugin development

---

## Overview

This document covers security vulnerabilities and defenses specific to L4D2/Source engine SourcePawn plugins. Understanding both attack vectors (RED TEAM) and defensive patterns (BLUE TEAM) builds stronger security awareness.

### Threat Model

| Layer | Attack Surface | Impact |
|-------|---------------|--------|
| Engine | Network messages, entity system | RCE, server crash |
| Plugin | SQL, commands, file I/O | Data theft, privilege escalation |
| Web | SourceBans, forums | Lateral movement, credential theft |

---

## 1. SQL Injection

### RED TEAM: Attack Vector

```sourcepawn
// VULNERABLE: Direct string interpolation
public void SavePlayerData(int client, const char[] nickname)
{
    char query[512];
    Format(query, sizeof(query),
        "INSERT INTO players (steamid, name) VALUES ('%s', '%s')",
        steamid, nickname);  // nickname = "'; DROP TABLE players;--"
    SQL_TQuery(g_hDatabase, SQLCallback, query);
}
```

**Exploitation**: Player sets nickname to `'; DROP TABLE players;--` and deletes the database.

### BLUE TEAM: Defense

```sourcepawn
// SECURE: Use SQL_EscapeString
public void SavePlayerData(int client, const char[] nickname)
{
    char escapedName[128];
    SQL_EscapeString(g_hDatabase, nickname, escapedName, sizeof(escapedName));

    char query[512];
    Format(query, sizeof(query),
        "INSERT INTO players (steamid, name) VALUES ('%s', '%s')",
        steamid, escapedName);
    SQL_TQuery(g_hDatabase, SQLCallback, query);
}
```

**Best Practices**:
- Always use `SQL_EscapeString()` for user input
- Use parameterized queries when available
- Validate input format before database operations

---

## 2. Command Injection

### RED TEAM: Attack Vector

```sourcepawn
// VULNERABLE: User input in ServerCommand
public Action Cmd_Map(int client, int args)
{
    char mapname[64];
    GetCmdArg(1, mapname, sizeof(mapname));

    char cmd[128];
    Format(cmd, sizeof(cmd), "changelevel %s", mapname);
    ServerCommand(cmd);  // mapname = "c1m1; rcon_password hacked"
}
```

**Exploitation**: Player enters `c1m1; rcon_password hacked` to change RCON password.

### BLUE TEAM: Defense

```sourcepawn
// SECURE: Whitelist validation
static char g_sAllowedMaps[][] = {"c1m1_hotel", "c1m2_streets", "c2m1_highway"};

public Action Cmd_Map(int client, int args)
{
    char mapname[64];
    GetCmdArg(1, mapname, sizeof(mapname));

    // Whitelist check
    bool valid = false;
    for (int i = 0; i < sizeof(g_sAllowedMaps); i++)
    {
        if (StrEqual(mapname, g_sAllowedMaps[i], false))
        {
            valid = true;
            break;
        }
    }

    if (!valid)
    {
        ReplyToCommand(client, "[SM] Invalid map name.");
        return Plugin_Handled;
    }

    ServerCommand("changelevel %s", mapname);
    return Plugin_Handled;
}
```

**Best Practices**:
- Never pass user input directly to `ServerCommand()`
- Use whitelist validation for allowed values
- Strip dangerous characters: `;`, `|`, `&`, `$`

---

## 3. Entity Exhaustion DoS

### RED TEAM: Attack Vector

```sourcepawn
// VULNERABLE: Unbounded entity spawning
public Action Cmd_SpawnZombies(int client, int args)
{
    int count = GetCmdArgInt(1);  // count = 9999

    for (int i = 0; i < count; i++)
    {
        int zombie = CreateEntityByName("infected");
        DispatchSpawn(zombie);  // Server crashes at 2048 edicts
    }
}
```

**Exploitation**: Run `sm_spawnzombies 9999` to crash server by exceeding 2048 edict limit.

### BLUE TEAM: Defense

```sourcepawn
// SECURE: Edict limit checking
#define MAX_SAFE_EDICTS 1900
#define MAX_SPAWN_PER_CMD 20

public Action Cmd_SpawnZombies(int client, int args)
{
    // Check edict headroom
    int currentEdicts = GetEdictCount();
    if (currentEdicts > MAX_SAFE_EDICTS)
    {
        ReplyToCommand(client, "[SM] Entity limit reached (%d/2048)", currentEdicts);
        return Plugin_Handled;
    }

    // Cap spawn count
    int count = GetCmdArgInt(1);
    if (count > MAX_SPAWN_PER_CMD)
        count = MAX_SPAWN_PER_CMD;

    // Calculate safe spawn
    int available = MAX_SAFE_EDICTS - currentEdicts;
    if (count > available)
        count = available;

    for (int i = 0; i < count; i++)
    {
        int zombie = CreateEntityByName("infected");
        if (IsValidEntity(zombie))
            DispatchSpawn(zombie);
    }

    return Plugin_Handled;
}
```

**Best Practices**:
- Always check `GetEdictCount() < 1900` before spawning
- Cap maximum entities per command
- Implement entity cleanup timers

---

## 4. Path Traversal

### RED TEAM: Attack Vector

```sourcepawn
// VULNERABLE: User-controlled path
public void SavePlayerConfig(int client, const char[] filename)
{
    char path[PLATFORM_MAX_PATH];
    BuildPath(Path_SM, path, sizeof(path), "configs/%s.cfg", filename);
    // filename = "../../cfg/server" overwrites server.cfg

    File file = OpenFile(path, "w");
    // Writes to unintended location
}
```

**Exploitation**: Use `../..` to escape intended directory and overwrite critical files.

### BLUE TEAM: Defense

```sourcepawn
// SECURE: Path validation
public void SavePlayerConfig(int client, const char[] filename)
{
    // Strip path traversal attempts
    char safeName[64];
    strcopy(safeName, sizeof(safeName), filename);
    ReplaceString(safeName, sizeof(safeName), "..", "");
    ReplaceString(safeName, sizeof(safeName), "/", "");
    ReplaceString(safeName, sizeof(safeName), "\\", "");

    // Validate only alphanumeric
    for (int i = 0; safeName[i] != '\0'; i++)
    {
        if (!IsCharAlphaNum(safeName[i]) && safeName[i] != '_')
        {
            LogError("Invalid filename character in: %s", filename);
            return;
        }
    }

    char path[PLATFORM_MAX_PATH];
    BuildPath(Path_SM, path, sizeof(path), "configs/%s.cfg", safeName);

    File file = OpenFile(path, "w");
    // Safe file operation
}
```

**Best Practices**:
- Strip `..`, `/`, `\` from filenames
- Validate only alphanumeric characters
- Use absolute path checks with `StrContains()`

---

## 5. Buffer Overflow

### RED TEAM: Attack Vector

```sourcepawn
// VULNERABLE: Hardcoded buffer size mismatch
public void ProcessName(const char[] input)
{
    char buffer[32];
    strcopy(buffer, 256, input);  // Wrong size, can overflow!
}
```

**Impact**: Memory corruption, potential code execution.

### BLUE TEAM: Defense

```sourcepawn
// SECURE: Always use sizeof()
public void ProcessName(const char[] input)
{
    char buffer[32];
    strcopy(buffer, sizeof(buffer), input);  // Safe
}
```

**Best Practices**:
- Always use `sizeof(buffer)` never hardcoded sizes
- Validate input length before copying
- Use `FormatEx()` for bounded formatting

---

## 6. Admin Permission Bypass

### RED TEAM: Attack Vector

```sourcepawn
// VULNERABLE: Missing permission check in callback
public void OnAdminMenuReady(Handle topmenu)
{
    AddMenuItem(g_hAdminMenu, "kick_all", AdminMenu_KickAll);
}

public void AdminMenu_KickAll(int client, int item)
{
    // No permission check - any player with menu access can kick!
    KickAllPlayers();
}
```

### BLUE TEAM: Defense

```sourcepawn
// SECURE: Explicit permission verification
public void AdminMenu_KickAll(int client, int item)
{
    if (!CheckCommandAccess(client, "sm_kickall", ADMFLAG_ROOT))
    {
        PrintToChat(client, "[SM] You don't have permission.");
        return;
    }

    KickAllPlayers();
}
```

**Best Practices**:
- Always use `CheckCommandAccess()` in callbacks
- Don't rely solely on menu visibility for security
- Log admin actions with `LogAction()`

---

## 7. Race Conditions

### RED TEAM: Attack Vector

```sourcepawn
// VULNERABLE: Stale entity reference across callback
public void OnEntityCreated(int entity)
{
    CreateTimer(5.0, Timer_ProcessEntity, entity);
}

public Action Timer_ProcessEntity(Handle timer, int entity)
{
    // Entity may be invalid/reused after 5 seconds!
    SetEntityHealth(entity, 100);  // Wrong entity or crash
}
```

### BLUE TEAM: Defense

```sourcepawn
// SECURE: Use EntIndexToEntRef for persistence
public void OnEntityCreated(int entity)
{
    int ref = EntIndexToEntRef(entity);
    CreateTimer(5.0, Timer_ProcessEntity, ref);
}

public Action Timer_ProcessEntity(Handle timer, int ref)
{
    int entity = EntRefToEntIndex(ref);
    if (entity == INVALID_ENT_REFERENCE)
        return Plugin_Stop;  // Entity no longer exists

    if (!IsValidEntity(entity))
        return Plugin_Stop;

    SetEntityHealth(entity, 100);  // Safe
    return Plugin_Continue;
}
```

**Best Practices**:
- Use `EntIndexToEntRef()` for delayed entity operations
- Use `GetClientUserId()` for client persistence across callbacks
- Always validate before accessing

---

## 8. Information Disclosure

### RED TEAM: Attack Vector

```sourcepawn
// VULNERABLE: Verbose error logging
public void ConnectDatabase()
{
    char error[256];
    g_hDatabase = SQL_Connect("storage", true, error, sizeof(error));

    if (g_hDatabase == null)
    {
        // Exposes database path and error to all players
        PrintToChatAll("[ERROR] Database failed: %s", error);
        LogError("DB path: /home/gameserver/db.sq3 - Error: %s", error);
    }
}
```

### BLUE TEAM: Defense

```sourcepawn
// SECURE: Sanitized logging
public void ConnectDatabase()
{
    char error[256];
    g_hDatabase = SQL_Connect("storage", true, error, sizeof(error));

    if (g_hDatabase == null)
    {
        // Generic user message
        PrintToChatAll("[SM] Database temporarily unavailable.");

        // Detailed error to logs only (not chat)
        LogError("Database connection failed - contact admin");
    }
}
```

**Best Practices**:
- Never expose paths, passwords, or tokens in chat
- Use generic error messages for players
- Log sensitive details server-side only

---

## Automated Security Validation

The test suite (`scripts/evaluation/automated_test.py`) includes security pattern detection:

```python
SECURITY_ANTIPATTERNS = [
    (r'Format\s*\([^;]*(?:SELECT|INSERT|UPDATE|DELETE)[^;]*%s',
     "SQL injection risk"),
    (r'(?:ServerCommand|ServerExecute)\s*\([^)]*%s',
     "Command injection risk"),
    (r'CreateEntityByName\s*\([^)]+\)(?!.*GetEdictCount)',
     "Entity exhaustion risk"),
    # ... more patterns
]
```

Run security analysis:

```bash
python scripts/evaluation/automated_test.py --model openai
```

Output includes:
- Security Score: 0-10 (higher = safer)
- Total Issues: Count of vulnerabilities detected
- Critical: Count of severe issues (injection, overflow)

---

## Training Data

Security training examples are in `data/anti_patterns/l4d2_security_patterns.jsonl`:

| Pattern | Type | Purpose |
|---------|------|---------|
| SQL Injection | RED/BLUE | Teach escape patterns |
| Command Injection | RED/BLUE | Teach whitelist validation |
| Entity Exhaustion | RED/BLUE | Teach edict limit checking |
| Path Traversal | RED/BLUE | Teach path validation |
| Buffer Overflow | RED/BLUE | Teach sizeof() usage |

To include in training:
```bash
# Combine with main dataset
cat data/anti_patterns/l4d2_security_patterns.jsonl >> data/openai_finetune/train_v6.jsonl
```

---

## Quick Reference

| Vulnerability | Detection Pattern | Defense Function |
|--------------|-------------------|------------------|
| SQL Injection | `Format(*SELECT*%s` | `SQL_EscapeString()` |
| Command Injection | `ServerCommand(*%s` | Whitelist validation |
| Entity Exhaustion | `CreateEntityByName` | `GetEdictCount() < 1900` |
| Path Traversal | `BuildPath(*%s` | Strip `..`, validate chars |
| Buffer Overflow | `strcopy(*, 256` | `sizeof(buffer)` |
| Admin Bypass | `RegAdminCmd` | `CheckCommandAccess()` |
| Race Condition | `CreateTimer(*entity` | `EntIndexToEntRef()` |
| Info Disclosure | `PrintToChatAll(*error` | Server-side logging only |

---

---

## RED TEAM Attack Training Data

Comprehensive attack chain examples are available in `data/anti_patterns/`:

### Attack Chain Files

| File | Examples | Focus |
|------|----------|-------|
| `l4d2_redteam_attacks.jsonl` | 8 | Multi-stage attack chains |
| `l4d2_game_exploits.jsonl` | 3 | L4D2-specific exploits |
| `l4d2_security_patterns.jsonl` | 10 | General security patterns |
| `contrastive_pairs.jsonl` | 87 | Contrastive learning pairs |
| `l4d2_anti_patterns.jsonl` | 22 | API anti-patterns |

### Attack Categories Covered

**Stage 1 - Reconnaissance**
- Plugin enumeration (`sm_plugins list`)
- SQL-enabled feature identification
- Injection point probing

**Stage 2 - Initial Access**
- SQL injection (UNION, stacked queries)
- Command injection (ServerCommand, RCON)
- Path traversal (BuildPath exploitation)

**Stage 3 - Privilege Escalation**
- Database admin insertion
- RCON password extraction
- Admin flag manipulation

**Stage 4 - L4D2 Game Exploits**
- Director manipulation (infinite intensity, spawn hijacking)
- Survivor state manipulation (godmode, speed hacks)
- Entity exhaustion (2048 edict DoS)
- Special infected army spawning
- Witch minefield creation
- Item deprivation attacks
- Teleport traps

**Stage 5 - Persistence**
- Database backdoors
- Config file persistence
- Plugin-based backdoors
- Cron/scheduled tasks

**Stage 6 - Lateral Movement**
- SourceBans exploitation
- Shared database access
- Credential reuse attacks
- Network pivoting

### Attack Chain Examples

```
ATTACK TIMELINE (Full Compromise):
T+0:00  - Join server, reconnaissance
T+0:10  - Identify SQL injection point
T+0:20  - Extract admin credentials + RCON
T+0:30  - RCON into game server
T+0:40  - Install persistence backdoor
T+1:00  - Pivot to other infrastructure
```

### Including in Training

```bash
# Combine all attack patterns for training
cat data/anti_patterns/l4d2_redteam_attacks.jsonl \
    data/anti_patterns/l4d2_game_exploits.jsonl \
    data/anti_patterns/l4d2_security_patterns.jsonl \
    >> data/openai_finetune/train_v6_security.jsonl
```

---

## References

- [SourceMod Security Best Practices](https://wiki.alliedmods.net/Security)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Source Engine Entity System](https://developer.valvesoftware.com/wiki/Entity)
- [HackerOne #807772](https://hackerone.com/reports/807772) - Source Engine RCE
