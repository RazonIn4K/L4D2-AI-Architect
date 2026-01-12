# Digital Ocean L4D2 AI Server Setup Guide

This guide walks you through setting up a complete L4D2 AI server on Digital Ocean with your $200 credits.

## Cost Breakdown

| Droplet Type | Monthly Cost | RAM | CPU | Best For |
|--------------|-------------|-----|-----|----------|
| Basic $24 | $24/mo | 4GB | 2 vCPU | Minimum viable |
| Basic $48 | $48/mo | 8GB | 4 vCPU | **Recommended** |
| Premium $48 | $48/mo | 4GB | 2 vCPU | Low latency |

**With $200 credits**: 4-8 months of server time!

## Quick Start (15 minutes)

### Step 1: Create Droplet

1. Go to [Digital Ocean](https://cloud.digitalocean.com/)
2. Click **Create** → **Droplets**
3. Choose:
   - **Region**: Closest to you (e.g., NYC, SFO)
   - **Image**: Ubuntu 22.04 LTS
   - **Size**: Basic → $48/month (8GB, 4 vCPU) **Recommended**
   - **Authentication**: SSH Key (more secure) or Password
   - **Hostname**: `l4d2-ai-server`

4. Click **Create Droplet**
5. Note your Droplet's IP address

### Step 2: Connect to Server

```bash
# SSH into your droplet
ssh root@YOUR_DROPLET_IP

# Example:
ssh root@167.99.123.45
```

### Step 3: Run Installation Script

```bash
# Clone the project
git clone https://github.com/YOUR_USERNAME/L4D2-AI-Architect.git
cd L4D2-AI-Architect

# Make script executable and run
chmod +x scripts/deploy/deploy_digitalocean.sh
sudo ./scripts/deploy/deploy_digitalocean.sh
```

The script will:
- Install L4D2 dedicated server (~15GB, takes 10-30 min)
- Install SourceMod and MetaMod
- Set up Python environment
- Install AI bridge plugin
- Configure everything

### Step 4: Configure Firewall

```bash
# Allow game traffic
ufw allow 27015/udp   # Game server
ufw allow 27020/tcp   # AI bridge
ufw allow 22/tcp      # SSH
ufw enable
```

### Step 5: Start the Server

```bash
# Start L4D2 server
sudo systemctl start l4d2-server

# Check status
sudo systemctl status l4d2-server

# View logs
journalctl -u l4d2-server -f
```

### Step 6: Start AI Controller

```bash
# Open a tmux session (keeps running after disconnect)
tmux new -s ai

# Start AI with a personality
cd /opt/l4d2_ai
./start_ai.sh aggressive

# Detach from tmux: Ctrl+B, then D
# Reattach later: tmux attach -t ai
```

### Step 7: Connect Your Game

1. Open **Left 4 Dead 2** on your computer
2. Open console (`~` key)
3. Type: `connect YOUR_DROPLET_IP:27015`
4. Once in-game, type in console: `sm_ai_connect 27020`
5. Watch the AI play!

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Digital Ocean Droplet                     │
│                                                              │
│  ┌─────────────────┐         ┌─────────────────────────┐    │
│  │  L4D2 Server    │◄───────►│  AI Controller (Python) │    │
│  │  (Port 27015)   │  TCP    │  (Port 27020)           │    │
│  │                 │ 27020   │                         │    │
│  │  SourceMod      │         │  PPO Models:            │    │
│  │  AI Bridge      │         │  - aggressive           │    │
│  │  Plugin         │         │  - balanced             │    │
│  └────────┬────────┘         │  - medic                │    │
│           │                  │  - defender             │    │
│           │ UDP 27015        │  - speedrunner          │    │
│           │                  └─────────────────────────┘    │
└───────────┼─────────────────────────────────────────────────┘
            │
            ▼
    ┌───────────────┐
    │  Your L4D2    │
    │  Game Client  │
    └───────────────┘
```

## Personality Options

| Personality | Behavior | Best For |
|-------------|----------|----------|
| `aggressive` | High kill focus, charges forward | Action gameplay |
| `balanced` | Moderate all-around | General play |
| `medic` | Prioritizes healing team | Support role |
| `defender` | Protects teammates | Team survival |
| `speedrunner` | Rushes to objectives | Speed runs |

Change personality:
```bash
# Stop current AI (Ctrl+C in tmux)
./start_ai.sh medic
```

## AI Director Modes

You can also run the AI Director instead of bots:

| Mode | Difficulty | Spawns |
|------|------------|--------|
| `relaxed` | Easy | Fewer zombies, more items |
| `standard` | Normal | Balanced |
| `intense` | Hard | More hordes, special infected |
| `nightmare` | Extreme | Maximum chaos |

```bash
# Run director instead of bot AI
cd /opt/l4d2_ai/L4D2-AI-Architect
python scripts/director/test_director.py --mode nightmare
```

## Troubleshooting

### Server won't start
```bash
# Check logs
journalctl -u l4d2-server -n 50

# Manual start for debugging
cd /opt/l4d2_server
./srcds_run -game left4dead2 -console
```

### Can't connect to server
```bash
# Check firewall
ufw status

# Check if server is listening
netstat -tulpn | grep 27015
```

### AI not controlling bots
```bash
# In-game console:
sm_ai_status  # Check connection
sm_ai_debug 1 # Enable debug mode

# Check AI controller logs in tmux
tmux attach -t ai
```

### Low FPS/Performance
```bash
# Upgrade droplet (can do live resize)
# Or reduce bot count in server.cfg:
echo "sm_cvar sb_max_team_melee_weapons 1" >> /opt/l4d2_server/left4dead2/cfg/server.cfg
```

## Monitoring & Management

### View Server Status
```bash
# Server status
sudo systemctl status l4d2-server

# Resource usage
htop

# Disk usage
df -h
```

### Server Console Access
```bash
# Attach to server console
screen -r l4d2
# Detach: Ctrl+A, then D
```

### Useful Server Commands (in-game console)
```
sm_ai_status          # AI connection status
sm_ai_connect 27020   # Connect to AI
sm_ai_disconnect      # Disconnect AI
changelevel c2m1_highway  # Change map
kick Bot              # Kick a bot
```

## Cost Optimization

### Pause When Not Using
```bash
# Stop server to save money (can power off droplet)
sudo systemctl stop l4d2-server

# Power off from DO console to stop billing
# (Disk storage still charged at ~$0.10/GB/month)
```

### Snapshot for Backup
1. Power off droplet
2. Create snapshot in DO console (~$0.05/GB/month storage)
3. Restore anytime

### Destroy When Done
If you're done experimenting:
1. Download any trained models you want to keep
2. Destroy droplet (stops all charges)
3. Can recreate from snapshot later

## Advanced: Custom Training

To train new models on the server:

```bash
# This requires GPU - not recommended on DO basic droplets
# Instead, train locally and upload models

# Upload trained models
scp -r model_adapters/rl_agents/* root@YOUR_IP:/opt/l4d2_ai/L4D2-AI-Architect/model_adapters/rl_agents/
```

## Files Reference

| File | Purpose |
|------|---------|
| `/opt/l4d2_server/` | L4D2 dedicated server |
| `/opt/l4d2_server/start_server.sh` | Server start script |
| `/opt/l4d2_server/left4dead2/cfg/server.cfg` | Server config |
| `/opt/l4d2_ai/` | AI controller environment |
| `/opt/l4d2_ai/start_ai.sh` | AI start script |
| `/opt/l4d2_ai/L4D2-AI-Architect/` | Full project code |

## Support

If you run into issues:
1. Check the troubleshooting section above
2. Review server logs: `journalctl -u l4d2-server`
3. Check AI logs in tmux session
4. Verify firewall rules with `ufw status`
