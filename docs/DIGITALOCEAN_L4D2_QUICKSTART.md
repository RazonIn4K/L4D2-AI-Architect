# DigitalOcean L4D2 Server Quickstart

A tested, production-ready guide for running a Left 4 Dead 2 dedicated server on Digital Ocean with enhanced AI spawning.

## Server Details

| Item | Value |
|------|-------|
| **Droplet Name** | l4d2-ai-server |
| **IP Address** | 104.248.183.166 |
| **Provider** | Digital Ocean (NYC3) |
| **SSH Key** | `~/.ssh/l4d2_do` |
| **Docker Image** | `laoyutang/l4d2:latest` |
| **SourceMod Version** | 1.11.0.6968 |
| **Game Port** | 27015 (UDP) |
| **RCON Port** | 27015 (TCP) |
| **RCON Password** | ai2026 |
| **Monthly Cost** | ~$12 (Basic, 2GB RAM) |

## Quick Start

### 1. Power On Droplet

Via Digital Ocean Console or API:
```bash
# Using doctl CLI (if authenticated)
doctl compute droplet-action power-on <droplet-id>
```

### 2. SSH into Server

```bash
ssh -i ~/.ssh/l4d2_do root@104.248.183.166
```

### 3. Start L4D2 Server

```bash
# Start Docker container
docker run -d --name l4d2-server --network host \
    -e L4D2_RCON_PASSWORD=ai2026 \
    -e L4D2_TICK=30 \
    laoyutang/l4d2:latest

# Wait for server to start
sleep 30

# Check status
docker logs l4d2-server 2>&1 | tail -20
```

### 4. Connect to Server

From L4D2 game console:
```
connect 104.248.183.166:27015
```

## Stable Enhanced Spawn Settings

These settings have been tested and work without crashing:

```bash
# Apply via RCON (using Python or other RCON client)
sm_cvar z_hunter_limit 2
sm_cvar z_smoker_limit 2
sm_cvar z_boomer_limit 2
sm_cvar z_charger_limit 2
sm_cvar z_spitter_limit 2
sm_cvar z_jockey_limit 2
sm_cvar z_special_spawn_interval 25
sm_cvar z_common_limit 25
sm_cvar sv_cheats 1
sm_cvar sv_hibernate_when_empty 0
```

### Settings That Cause Crashes (AVOID)

| Setting | Problem |
|---------|---------|
| `sv_gravity 150` (moon gravity) | Physics glitches, crashes |
| `z_speed 500` (turbo zombies) | Too many calculations |
| Spawn limits > 4 per type | Entity overload |
| Spawning > 5 infected at once | Server crash |
| Multiple tanks simultaneously | Frequent crashes |

## RCON Commands

### Python RCON Client

```python
import socket
import struct
import time

RCON_HOST = "127.0.1.1"  # Note: 127.0.1.1 for host networking
RCON_PORT = 27015
RCON_PASSWORD = "ai2026"

def rcon_cmd(host, port, password, cmd):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    try:
        sock.connect((host, port))
        # Auth (type 3)
        body = password.encode("utf-8") + b"\x00\x00"
        pkt = struct.pack("<iii", 4+4+len(body), 1, 3) + body
        sock.send(pkt)
        sock.recv(4096)
        sock.recv(4096)
        # Command (type 2)
        body = cmd.encode("utf-8") + b"\x00\x00"
        pkt = struct.pack("<iii", 4+4+len(body), 2, 2) + body
        sock.send(pkt)
        time.sleep(0.2)
        return sock.recv(4096).decode("utf-8", errors="replace")
    finally:
        sock.close()
```

### Useful Commands

| Command | Description |
|---------|-------------|
| `status` | Show server status and players |
| `sm plugins list` | List loaded SourceMod plugins |
| `changelevel c1m1_hotel` | Change to Dead Center |
| `sm_cvar <cvar> <value>` | Set a cvar via SourceMod |
| `z_spawn hunter` | Spawn a hunter |
| `z_spawn tank` | Spawn a tank |
| `z_spawn mob` | Spawn a horde |
| `z_spawn witch` | Spawn a witch |
| `god 1` / `god 0` | Toggle god mode |
| `give weapon_rifle_m60` | Give M60 |
| `give first_aid_kit` | Give health kit |

### Campaign Maps

| Campaign | Map Codes |
|----------|-----------|
| Dead Center | c1m1_hotel - c1m4_atrium |
| Dark Carnival | c2m1_highway - c2m5_concert |
| Swamp Fever | c3m1_plankcountry - c3m4_plantation |
| Hard Rain | c4m1_milltown_a - c4m5_milltown_escape |
| The Parish | c5m1_waterfront - c5m5_bridge |

## Docker Image Details

### laoyutang/l4d2

Pre-configured Chinese L4D2 Docker image with:

- **MetaMod** - Plugin framework
- **SourceMod 1.11.0.6968** - Server-side scripting
- **83+ Plugins** including:
  - Special Spawner (1.3.7) - Enhanced infected spawning
  - Bots plugin - Bot management
  - Admin tools - Server administration
- **l4dtoolz** - Server capacity expansion (up to 32 players)

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `L4D2_RCON_PASSWORD` | laoyutangnb! | RCON authentication password |
| `L4D2_PORT` | 27015 | Game server port |
| `L4D2_TICK` | 30 | Server tick rate (30, 60, or 100) |

### Container Paths

| Path | Description |
|------|-------------|
| `/l4d2/` | Game installation root |
| `/l4d2/left4dead2/addons/sourcemod/` | SourceMod installation |
| `/l4d2/left4dead2/cfg/server.cfg` | Main server config |
| `/start.sh` | Container startup script |

## Troubleshooting

### Server Won't Start

```bash
# Check Docker logs
docker logs l4d2-server 2>&1 | tail -50

# Restart container
docker stop l4d2-server && docker rm l4d2-server
docker run -d --name l4d2-server --network host \
    -e L4D2_RCON_PASSWORD=ai2026 laoyutang/l4d2:latest
```

### RCON Connection Refused

The server binds to `127.0.1.1` when using host networking:
```python
# Wrong
RCON_HOST = '127.0.0.1'

# Correct
RCON_HOST = '127.0.1.1'
```

### Players Can't Connect

```bash
# Check firewall
ufw allow 27015/udp
ufw allow 27015/tcp

# Verify server is running
docker ps | grep l4d2

# Check listening
netstat -ulnp | grep 27015
```

### Server Crashed

```bash
# Check for crash logs
docker logs l4d2-server 2>&1 | grep -i "crash\|fault\|dump"

# Server auto-restarts, but you may need to restart container
docker restart l4d2-server
```

### High Packet Loss

If players experience packet loss:
```bash
# Network optimization
sm_cvar net_maxcleartime 0.001
sm_cvar sv_maxrate 0
sm_cvar sv_minrate 100000
```

## Stopping the Server

### Stop Container Only

```bash
ssh -i ~/.ssh/l4d2_do root@104.248.183.166 "docker stop l4d2-server"
```

### Power Off Droplet (Save Costs)

```bash
ssh -i ~/.ssh/l4d2_do root@104.248.183.166 "shutdown -h now"
```

Or via Digital Ocean Console.

## Key Learnings

1. **Use `laoyutang/l4d2`** - Has SourceMod pre-installed (other images crash)
2. **Host networking required** - For proper port binding
3. **RCON on 127.0.1.1** - Special binding for host network mode
4. **sm_cvar bypasses sv_cheats** - Use for spawn commands
5. **Spawn gradually** - Never spawn more than 3-4 infected at once
6. **Keep limits reasonable** - 2-3 per infected type is stable
