# L4D2 AI Integration Setup Guide

This guide explains how to set up a Left 4 Dead 2 dedicated server with AI bot control and Director integration.

## Prerequisites

- Left 4 Dead 2 (Steam)
- SteamCMD (for dedicated server)
- Python 3.10+ with trained models
- SourceMod 1.11+

## Quick Start

### Option 1: Local Testing (Single Player + Listen Server)

1. **Start L4D2 with console enabled:**
   ```
   # Add to Steam launch options:
   -console -dev
   ```

2. **Start a local server:**
   ```
   map c13m4_cutthroat_creek
   ```

3. **The SourceMod plugin will auto-connect to Python on localhost:27050**

### Option 2: Dedicated Server Setup

#### Step 1: Install SteamCMD

```bash
# macOS
mkdir ~/steamcmd && cd ~/steamcmd
curl -sqL "https://steamcdn-a.akamaihd.net/client/installer/steamcmd_osx.tar.gz" | tar zxvf -

# Linux
mkdir ~/steamcmd && cd ~/steamcmd
curl -sqL "https://steamcdn-a.akamaihd.net/client/installer/steamcmd_linux.tar.gz" | tar zxvf -

# Windows: Download from https://steamcdn-a.akamaihd.net/client/installer/steamcmd.zip
```

#### Step 2: Install L4D2 Dedicated Server

```bash
cd ~/steamcmd
./steamcmd.sh +force_install_dir ~/l4d2_server +login anonymous +app_update 222860 validate +quit
```

#### Step 3: Install SourceMod

```bash
# Download latest SourceMod
cd ~/l4d2_server/left4dead2
wget https://sm.alliedmods.net/smdrop/1.11/sourcemod-1.11.0-git6968-linux.tar.gz
tar -xzf sourcemod-*.tar.gz

# Download MetaMod:Source
wget https://mms.alliedmods.net/mmsdrop/1.11/mmsource-1.11.0-git1155-linux.tar.gz
tar -xzf mmsource-*.tar.gz
```

#### Step 4: Install Socket Extension

The AI bridge requires the Socket extension for TCP communication.

```bash
# Download from: https://forums.alliedmods.net/showthread.php?t=67640
# Place socket.ext.so in:
~/l4d2_server/left4dead2/addons/sourcemod/extensions/
```

#### Step 5: Compile and Install the AI Bridge Plugin

```bash
# Copy the plugin source
cp /path/to/L4D2-AI-Architect/data/l4d2_server/addons/sourcemod/scripting/l4d2_ai_bridge.sp \
   ~/l4d2_server/left4dead2/addons/sourcemod/scripting/

# Compile
cd ~/l4d2_server/left4dead2/addons/sourcemod/scripting
./spcomp l4d2_ai_bridge.sp -o ../plugins/l4d2_ai_bridge.smx

# Verify
ls -la ../plugins/l4d2_ai_bridge.smx
```

#### Step 6: Configure the Server

Create `~/l4d2_server/left4dead2/cfg/server.cfg`:

```cfg
// Server name
hostname "L4D2 AI Testing Server"

// Network settings
sv_lan 0
sv_region 0

// Bot settings
sb_all_bot_game 1
sb_stop 0

// AI Director settings
director_force_panic_event 0
director_panic_forever 0

// Allow cheats for testing
sv_cheats 1
```

#### Step 7: Start the Server

```bash
cd ~/l4d2_server
./srcds_run -game left4dead2 +map c13m4_cutthroat_creek +maxplayers 8
```

## Connecting Python to the Game

### Start the Python AI Controller

```bash
cd /path/to/L4D2-AI-Architect
source activate.sh

# Connect to local server
python scripts/rl_training/train_ppo.py --mode demo \
    --model model_adapters/rl_agents/all_personalities_20260110_161633/aggressive/final_model \
    --personality aggressive \
    --env mnemosyne \
    --host 127.0.0.1 \
    --port 27050
```

### In-Game Commands

Once connected, use these console commands:

```
// Connect to Python (if not auto-connected)
sm_ai_connect 127.0.0.1 27050

// Disconnect from Python
sm_ai_disconnect

// Toggle AI Director control
sm_ai_director

// Check status
sm_ai_status
```

## Available AI Models

### Bot Personalities

| Personality | Best For | Command |
|-------------|----------|---------|
| `aggressive` | Combat-focused play | `--personality aggressive` |
| `balanced` | All-around play | `--personality balanced` |
| `medic` | Team support | `--personality medic` |
| `speedrunner` | Rushing objectives | `--personality speedrunner` |
| `defender` | Protecting teammates | `--personality defender` |

### Director Personalities

| Personality | Behavior | Command |
|-------------|----------|---------|
| `standard` | Balanced difficulty | `--personality standard` |
| `relaxed` | Easier gameplay | `--personality relaxed` |
| `intense` | Harder gameplay | `--personality intense` |
| `nightmare` | Maximum difficulty | `--personality nightmare` |

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        Your Computer                            │
│                                                                  │
│  ┌─────────────────────┐         ┌──────────────────────────┐  │
│  │   Python Controller │  TCP    │  L4D2 Dedicated Server   │  │
│  │                     │◄───────►│                          │  │
│  │  ┌───────────────┐  │  27050  │  ┌────────────────────┐  │  │
│  │  │ PPO Bot Agent │  │         │  │ l4d2_ai_bridge.smx │  │  │
│  │  └───────────────┘  │         │  └────────────────────┘  │  │
│  │  ┌───────────────┐  │         │           │              │  │
│  │  │  AI Director  │  │         │           ▼              │  │
│  │  └───────────────┘  │         │  ┌────────────────────┐  │  │
│  └─────────────────────┘         │  │  Survivor Bots     │  │  │
│                                   │  │  + Infected AI     │  │  │
│                                   │  └────────────────────┘  │  │
│                                   └──────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **Game State Collection** (10 Hz):
   - Survivor positions, health, weapons
   - Infected counts and positions
   - Game events (deaths, items, objectives)

2. **Python Processing**:
   - Neural network inference
   - Action selection based on observations
   - Director decisions for spawning

3. **Action Execution**:
   - Bot movement and combat
   - Item usage and healing
   - Director spawns and events

## Troubleshooting

### Plugin Not Loading

```bash
# Check SourceMod is installed
ls ~/l4d2_server/left4dead2/addons/sourcemod/

# Check plugin compiled
ls ~/l4d2_server/left4dead2/addons/sourcemod/plugins/*.smx

# Check server logs
tail -f ~/l4d2_server/left4dead2/addons/sourcemod/logs/errors_*.log
```

### Connection Issues

```bash
# Test port is open
nc -zv 127.0.0.1 27050

# Check firewall
sudo ufw allow 27050/tcp  # Linux
```

### Python Errors

```bash
# Ensure virtual environment is active
source activate.sh

# Check dependencies
pip list | grep stable-baselines3

# Verify model exists
ls model_adapters/rl_agents/*/final_model*
```

## Performance Tips

1. **Use `--n-envs 1`** for real game (no parallelization needed)
2. **Set `--render` flag** to disable if running headless
3. **Increase `--eval-episodes`** for more stable metrics
4. **Use `aggressive` personality** for best mock-trained performance

## Next Steps

1. **Fine-tune on Real Data**: Train with `--env mnemosyne` for better transfer
2. **Custom Personalities**: Modify reward configs in `train_ppo.py`
3. **Director Tuning**: Adjust spawn rates in `train_director_rl.py`
4. **Multi-Bot Control**: Extend plugin to control multiple survivors

## Files Reference

| File | Purpose |
|------|---------|
| `scripts/rl_training/train_ppo.py` | Bot training and inference |
| `scripts/rl_training/mnemosyne_env.py` | Game environment wrapper |
| `scripts/director/train_director_rl.py` | Director training |
| `scripts/director/bridge.py` | Game communication |
| `data/l4d2_server/addons/sourcemod/scripting/l4d2_ai_bridge.sp` | SourceMod plugin |

## Support

- Check logs in `data/training_logs/`
- Run `python scripts/director/test_director.py --demo` for offline testing
- Use TensorBoard: `tensorboard --logdir data/training_logs --port 6006`
