# L4D2 Server Security Threat Model

**Last Updated**: January 7, 2026  
**Scope**: Left 4 Dead 2 dedicated servers running SourceMod plugins, Points Reloaded, and web integrations

---

## Executive Summary

This document maps the attack surface of a production L4D2 server from the perspective of a skilled attacker who starts as a **normal client** and attempts to escalate through multiple layers: engine bugs → plugin exploitation → web/database compromise → host takeover.

**Key Principle**: Assume old Source 1 engine vulnerabilities persist. Defense must be layered across all boundaries.

---

## Layer 1: Source Engine Attack Surface

### 1.1 Historical Context

**Known RCE Chains (2018-2021)**:
- Server→Client RCE via network message bugs (`CL_CopyExistingEntity`, BSP/ZIP parsers)
- HackerOne reports (e.g., #807772) documented OOB reads in netmessage handlers
- Kill animation exploit, spray exploit, model/ragdoll buffer overflows

**References**:
- [Secret Club: Source Engine RCE](https://secret.club/2021/04/20/source-engine-rce-invite.html)
- [CTF.re: Source Engine Exploitation Part 2](https://ctf.re/source-engine/exploitation/2021/05/01/source-engine-2/)
- [HackerOne #807772](https://hackerone.com/reports/807772)

### 1.2 Current State (2024-2025)

**Patching Status**: Valve has patched *some* critical RCE chains, but:
- Old flood/crash vectors keep reappearing
- Community reports ongoing DDoS attacks against L4D2 servers
- No guarantee all netmessage handlers are hardened

**Practical Attacker Capabilities**:
- **DoS/Crash**: Crafted packets to exhaust server CPU or trigger crashes
- **Entity Pressure**: Abuse entity limits (~2048 networked entities) via spawn floods
- **Lag Exploitation**: Force server into degenerate states (lag, packet storms)

**Mitigation**:
- Keep SRCDS binaries current (monitor Valve updates)
- Rate-limit client connections and packet rates
- Monitor entity counts and implement caps in plugins
- **Accept**: Remote crash/DoS is always on the table with Source 1

---

## Layer 2: Plugin/SourceMod Attack Surface

### 2.1 Plugin Injection Vectors

**Threat**: If an attacker gains arbitrary command execution in SRCDS context (via engine bug or plugin vulnerability), they can:
- Load malicious `.smx` plugins
- Execute console commands with server privileges
- Access SQL connections, file I/O, and network sockets

**Attack Paths**:
1. **SQL Injection in Custom Plugins**
   - Points Reloaded, stats trackers, or custom economy plugins often use SQL
   - Unsanitized player names, chat messages, or item names → SQL injection
   - Example: `SELECT * FROM points WHERE name = '{playername}'` with playername = `'; DROP TABLE points; --`

2. **Command Injection**
   - Plugins that shell out or use `ServerCommand()` with unsanitized input
   - Example: Admin plugin that runs `!exec <filename>` without path validation

3. **File System Access**
   - Plugins with file write capabilities (logs, configs, data dumps)
   - Path traversal: `../../../../etc/passwd` if not using `safe_path` validation

4. **Memory Corruption**
   - SourcePawn is memory-safe, but native extensions (`.so`/`.dll`) are not
   - SDKHooks, Left4DHooks, or custom natives could have buffer overflows

### 2.2 Points Reloaded Specific Risks

**High-Value Target**: PR manages virtual economy, items, quests, and player progression.

**Attack Scenarios**:
- **Economy Manipulation**: Exploit race conditions in point transactions
- **Item Duplication**: Abuse quest completion or vending machine logic
- **Privilege Escalation**: Manipulate admin flags or class restrictions via DB tampering

**Mitigation**:
- Audit all PR SQL queries for injection vulnerabilities
- Use parameterized queries exclusively
- Implement transaction logging and anomaly detection
- Rate-limit vending machine purchases and quest completions

---

## Layer 3: Web/Admin Panel Attack Surface

### 3.1 Common Web Surfaces

**Typical Setup**:
- **SourceBans**: Ban management (PHP + MySQL)
- **Forums**: Community discussion (phpBB, MyBB, etc.)
- **Stats/Leaderboards**: Player statistics (PHP/Python + MySQL)
- **Admin Consoles**: RCON, server control panels

**Standard Web Vulnerabilities**:
- SQL injection in ban lookups, player searches
- XSS in forum posts, ban reasons, player names
- CSRF in admin actions (ban, unban, config changes)
- Authentication bypass (weak passwords, session fixation)
- File upload vulnerabilities (avatar uploads, attachment handling)

### 3.2 Pivot Opportunities

**Once Web Access is Gained**:
1. **Database Credentials**: Web config files often contain DB passwords
2. **RCON Passwords**: Admin panels may store or expose RCON credentials
3. **Plugin Configs**: Access to SourceMod configs, API keys, database connection strings
4. **Server File System**: If web server runs on same host, potential for file read/write

**Mitigation**:
- Separate web services from game server (different hosts/VLANs)
- Use read-only DB users for web stats/leaderboards
- Never store RCON passwords in web-accessible locations
- Implement strict CSP, input validation, and output encoding
- Regular security audits of web applications

---

## Layer 4: Infrastructure & Host Takeover

### 4.1 Database Layer

**MySQL/MariaDB Risks**:
- Weak root passwords or default credentials
- Exposed to internet (port 3306 open)
- Shared credentials across services
- No network segmentation (web + game + DB on same subnet)

**Attacker Goals**:
- Dump player data (emails, IPs, Steam IDs)
- Modify admin flags, points, items
- Plant backdoors in stored procedures
- Pivot to other databases on same host

### 4.2 Host/Container Escape

**If Attacker Reaches Host**:
- **Docker/LXC Escape**: Misconfigured containers with privileged mode
- **Kernel Exploits**: Old kernel versions with known CVEs
- **Lateral Movement**: SSH keys, credential reuse, other services on host

**Mitigation**:
- Run game server in unprivileged container/VM
- Minimal host services (no unnecessary daemons)
- Network segmentation (game, web, DB on separate VLANs)
- Regular patching of host OS and container runtime
- Monitoring and alerting on suspicious activity

---

## Layer 5: DDoS & Network Attacks

### 5.1 Game Port DDoS

**Active Threat**: L4D2 servers are regularly targeted by DDoS attacks.

**Attack Types**:
- UDP flood on game port (27015)
- Query floods (A2S_INFO, A2S_PLAYER)
- Crafted packets to trigger expensive operations

**Mitigation**:
- DDoS protection service (Cloudflare Spectrum, OVH Game DDoS protection)
- Rate limiting at firewall level
- Query rate limiting in SourceMod
- Geographic IP filtering if player base is regional

### 5.2 Amplification Attacks

**Risk**: L4D2 servers can be abused as amplification vectors for DDoS attacks against third parties.

**Mitigation**:
- Disable or rate-limit server query responses
- Implement source IP validation
- Monitor outbound traffic for anomalies

---

## Threat Scenarios & Response

### Scenario 1: Malicious Client → Engine RCE

**Attack Chain**:
1. Client connects with crafted packets
2. Exploits netmessage handler bug (e.g., OOB read)
3. Achieves arbitrary code execution in SRCDS process
4. Loads malicious plugin or exfiltrates data

**Detection**:
- Server crashes or unexpected restarts
- Unknown plugins loaded
- Unusual network traffic patterns
- Memory corruption errors in logs

**Response**:
- Isolate server immediately
- Review recent connections and ban suspicious IPs
- Update SRCDS to latest version
- Audit loaded plugins and remove unknowns
- Check for data exfiltration (DB dumps, config files)

### Scenario 2: Plugin Vulnerability → Database Compromise

**Attack Chain**:
1. Attacker finds SQL injection in custom plugin
2. Dumps database credentials and player data
3. Modifies admin flags or economy data
4. Plants backdoor for persistent access

**Detection**:
- Unusual SQL queries in logs
- Unexpected admin flag changes
- Economy anomalies (point inflation, item duplication)
- New admin accounts or suspicious logins

**Response**:
- Rotate all database passwords immediately
- Audit SQL query logs for injection attempts
- Restore database from clean backup
- Patch vulnerable plugin or remove it
- Implement parameterized queries

### Scenario 3: Web Panel Compromise → Server Takeover

**Attack Chain**:
1. Attacker exploits web vulnerability (SQLi, XSS, file upload)
2. Gains access to admin panel or database
3. Retrieves RCON password or server credentials
4. Takes control of game server via RCON
5. Loads malicious plugins or shuts down server

**Detection**:
- Unauthorized admin panel logins
- RCON commands from unexpected IPs
- Plugin changes or server config modifications
- Web server logs showing exploitation attempts

**Response**:
- Change all RCON and admin passwords
- Audit web application for vulnerabilities
- Review server logs for malicious commands
- Restore server configs from known-good state
- Consider moving web services to separate host

---

## Security Checklist

### Engine/Server Level
- [ ] SRCDS binaries up to date
- [ ] Rate limiting on client connections
- [ ] Entity count monitoring and caps
- [ ] DDoS protection service configured
- [ ] Regular server restarts to clear state

### Plugin Level
- [ ] All SQL queries use parameterized statements
- [ ] Input validation on all player-provided data
- [ ] File operations use `safe_path` validation
- [ ] No shell command execution with user input
- [ ] Regular plugin security audits

### Web Level
- [ ] Web services on separate host/VLAN
- [ ] Strong authentication (2FA for admins)
- [ ] Regular security updates for web apps
- [ ] Read-only DB users for stats/leaderboards
- [ ] No RCON passwords in web-accessible locations

### Infrastructure Level
- [ ] Database not exposed to internet
- [ ] Network segmentation (game/web/DB)
- [ ] Unprivileged containers/VMs
- [ ] Host OS and kernel up to date
- [ ] Monitoring and alerting configured

### Operational
- [ ] Regular backups (DB, configs, plugins)
- [ ] Incident response plan documented
- [ ] Log retention and analysis
- [ ] Security contact information published
- [ ] Regular security training for admins

---

## References & Further Reading

### Source Engine Security
- [Secret Club: Source Engine RCE](https://secret.club/2021/04/20/source-engine-rce-invite.html)
- [CTF.re: Source Engine Exploitation](https://ctf.re/source-engine/exploitation/2021/05/01/source-engine-2/)
- [Synacktiv: CS:GO Attack Surface](https://synacktiv.com/publications/exploring-counter-strike-global-offensive-attack-surface)

### L4D2 Specific
- [AlliedModders: L4D2 Events](https://wiki.alliedmods.net/Left_4_dead_2_events)
- [Valve Developer Wiki: L4D2 Scripting](https://developer.valvesoftware.com/wiki/Left_4_Dead_2)
- [L4D2 Community Update](https://github.com/Tsuey/L4D2-Community-Update)

### General Server Security
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CIS Benchmarks](https://www.cisecurity.org/cis-benchmarks/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)

---

## Maintenance

This document should be reviewed and updated:
- After any security incident
- When new vulnerabilities are disclosed
- Quarterly as part of security review process
- When infrastructure or architecture changes

**Document Owner**: Server Operations Team  
**Last Security Audit**: [To be scheduled]  
**Next Review Date**: April 7, 2026
