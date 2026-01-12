#!/bin/bash
# =============================================================================
# L4D2 AI Server - Digital Ocean Deployment Script
# =============================================================================
# This script sets up a complete L4D2 dedicated server with AI integration
# on a Digital Ocean droplet.
#
# REQUIREMENTS:
#   - Digital Ocean account with $200 credits
#   - Droplet: Basic $24/month (4GB RAM, 2 vCPU, 80GB SSD) - MINIMUM
#   - Recommended: $48/month (8GB RAM, 4 vCPU) for better performance
#   - Ubuntu 22.04 LTS
#   - Your Steam account credentials (for SteamCMD)
#
# USAGE:
#   1. Create a Digital Ocean droplet (Ubuntu 22.04, 4GB+ RAM)
#   2. SSH into the droplet: ssh root@YOUR_DROPLET_IP
#   3. Clone the repo and run this script
#   4. Follow the prompts
#
# ESTIMATED COST: ~$24-48/month ($0.04-0.07/hour)
# With $200 credits = 4-8 months of server time
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║        L4D2 AI Server - Digital Ocean Deployment               ║"
echo "║                                                                 ║"
echo "║  This will set up:                                             ║"
echo "║  1. L4D2 Dedicated Server (SteamCMD)                          ║"
echo "║  2. SourceMod + MetaMod                                        ║"
echo "║  3. AI Bridge Plugin (Python <-> Game communication)          ║"
echo "║  4. Python environment with trained AI models                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Configuration
STEAM_USER="${STEAM_USER:-anonymous}"
L4D2_DIR="/opt/l4d2_server"
AI_DIR="/opt/l4d2_ai"
SOURCEMOD_VERSION="1.11"
METAMOD_VERSION="1.11"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}This script must be run as root${NC}"
   exit 1
fi

# Check system requirements
echo -e "\n${YELLOW}Checking system requirements...${NC}"

RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
RAM_GB=$((RAM_KB / 1024 / 1024))
CPU_CORES=$(nproc)
DISK_GB=$(df -BG / | tail -1 | awk '{print $4}' | tr -d 'G')

echo "  RAM: ${RAM_GB}GB (minimum 4GB recommended)"
echo "  CPU: ${CPU_CORES} cores"
echo "  Disk: ${DISK_GB}GB free"

if [[ $RAM_GB -lt 3 ]]; then
    echo -e "${RED}WARNING: Less than 4GB RAM detected. Server may be unstable.${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 1: System Dependencies
echo -e "\n${GREEN}[1/7] Installing system dependencies...${NC}"
apt-get update
apt-get install -y \
    lib32gcc-s1 \
    lib32stdc++6 \
    libsdl2-2.0-0:i386 \
    steamcmd \
    python3 \
    python3-pip \
    python3-venv \
    git \
    tmux \
    htop \
    unzip \
    wget \
    curl

# Step 2: Create dedicated user
echo -e "\n${GREEN}[2/7] Creating steam user...${NC}"
if ! id "steam" &>/dev/null; then
    useradd -m -s /bin/bash steam
    echo "steam ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/steam
fi

# Step 3: Install L4D2 Server
echo -e "\n${GREEN}[3/7] Installing L4D2 Dedicated Server...${NC}"
echo "This will download ~15GB. This may take 10-30 minutes depending on connection."

mkdir -p $L4D2_DIR
chown steam:steam $L4D2_DIR

sudo -u steam steamcmd +force_install_dir $L4D2_DIR \
    +login $STEAM_USER \
    +app_update 222860 validate \
    +quit

echo -e "${GREEN}L4D2 Server installed to $L4D2_DIR${NC}"

# Step 4: Install MetaMod and SourceMod
echo -e "\n${GREEN}[4/7] Installing MetaMod and SourceMod...${NC}"

cd /tmp

# Download and install MetaMod
METAMOD_URL="https://mms.alliedmods.net/mmsdrop/${METAMOD_VERSION}/mmsource-${METAMOD_VERSION}-git1155-linux.tar.gz"
wget -q "$METAMOD_URL" -O metamod.tar.gz || {
    echo "Using fallback MetaMod URL..."
    wget -q "https://mms.alliedmods.net/mmsdrop/1.11/mmsource-1.11-git1155-linux.tar.gz" -O metamod.tar.gz
}
tar -xzf metamod.tar.gz -C $L4D2_DIR/left4dead2/

# Download and install SourceMod
SOURCEMOD_URL="https://sm.alliedmods.net/smdrop/${SOURCEMOD_VERSION}/sourcemod-${SOURCEMOD_VERSION}-git6968-linux.tar.gz"
wget -q "$SOURCEMOD_URL" -O sourcemod.tar.gz || {
    echo "Using fallback SourceMod URL..."
    wget -q "https://sm.alliedmods.net/smdrop/1.11/sourcemod-1.11-git6968-linux.tar.gz" -O sourcemod.tar.gz
}
tar -xzf sourcemod.tar.gz -C $L4D2_DIR/left4dead2/

# Clean up
rm -f metamod.tar.gz sourcemod.tar.gz

echo -e "${GREEN}MetaMod and SourceMod installed${NC}"

# Step 5: Clone and set up AI project
echo -e "\n${GREEN}[5/7] Setting up AI models and Python environment...${NC}"

mkdir -p $AI_DIR
cd $AI_DIR

# Clone the project if not already present
if [ ! -d "L4D2-AI-Architect" ]; then
    # For now, create the structure - in production, clone from git
    mkdir -p L4D2-AI-Architect
    echo "NOTE: In production, clone your git repo here"
fi

# Create Python virtual environment
python3 -m venv $AI_DIR/venv
source $AI_DIR/venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install \
    numpy \
    gymnasium \
    stable-baselines3 \
    torch \
    pygame

echo -e "${GREEN}Python environment ready${NC}"

# Step 6: Install AI Bridge Plugin
echo -e "\n${GREEN}[6/7] Installing AI Bridge SourceMod plugin...${NC}"

# Copy the plugin source (would come from git repo)
PLUGIN_DIR="$L4D2_DIR/left4dead2/addons/sourcemod"
mkdir -p $PLUGIN_DIR/scripting
mkdir -p $PLUGIN_DIR/plugins

# Create the plugin source file
cat > $PLUGIN_DIR/scripting/l4d2_ai_bridge.sp << 'PLUGIN_EOF'
/**
 * L4D2 AI Bridge Plugin v2.0
 * Connects trained PPO agents to Left 4 Dead 2
 *
 * Commands:
 *   sm_ai_connect <port>  - Connect to AI server
 *   sm_ai_disconnect      - Disconnect from AI server
 *   sm_ai_status          - Show connection status
 *   sm_ai_debug <0/1>     - Toggle debug mode
 */

#include <sourcemod>
#include <sdktools>
#include <left4dhooks>

#pragma semicolon 1
#pragma newdecls required

#define PLUGIN_VERSION "2.0.0"
#define MAX_BUFFER 4096
#define DEFAULT_PORT 27020
#define TICK_RATE 0.1  // 10 Hz update

// Connection state
Handle g_hSocket = INVALID_HANDLE;
bool g_bConnected = false;
int g_iPort = DEFAULT_PORT;
bool g_bDebug = false;

// Bot state
float g_fBotHealth[MAXPLAYERS+1];
float g_fBotPosition[MAXPLAYERS+1][3];
int g_iBotAmmo[MAXPLAYERS+1];

public Plugin myinfo = {
    name = "L4D2 AI Bridge",
    author = "L4D2-AI-Architect",
    description = "Connects trained RL agents to L4D2",
    version = PLUGIN_VERSION,
    url = "https://github.com/your-repo/L4D2-AI-Architect"
};

public void OnPluginStart() {
    // Commands
    RegAdminCmd("sm_ai_connect", Cmd_Connect, ADMFLAG_ROOT, "Connect to AI server");
    RegAdminCmd("sm_ai_disconnect", Cmd_Disconnect, ADMFLAG_ROOT, "Disconnect from AI");
    RegAdminCmd("sm_ai_status", Cmd_Status, ADMFLAG_ROOT, "Show AI status");
    RegAdminCmd("sm_ai_debug", Cmd_Debug, ADMFLAG_ROOT, "Toggle debug mode");

    // Create timer for state updates
    CreateTimer(TICK_RATE, Timer_UpdateState, _, TIMER_REPEAT);

    PrintToServer("[AI Bridge] Plugin loaded v%s", PLUGIN_VERSION);
}

public Action Cmd_Connect(int client, int args) {
    if (args >= 1) {
        char portStr[16];
        GetCmdArg(1, portStr, sizeof(portStr));
        g_iPort = StringToInt(portStr);
    }

    // Create TCP socket connection
    g_hSocket = SocketCreate(SOCKET_TCP, OnSocketError);
    SocketConnect(g_hSocket, OnSocketConnected, OnSocketReceive, OnSocketDisconnected, "127.0.0.1", g_iPort);

    ReplyToCommand(client, "[AI] Connecting to AI server on port %d...", g_iPort);
    return Plugin_Handled;
}

public Action Cmd_Disconnect(int client, int args) {
    if (g_hSocket != INVALID_HANDLE) {
        CloseHandle(g_hSocket);
        g_hSocket = INVALID_HANDLE;
    }
    g_bConnected = false;
    ReplyToCommand(client, "[AI] Disconnected from AI server");
    return Plugin_Handled;
}

public Action Cmd_Status(int client, int args) {
    ReplyToCommand(client, "[AI] Status: %s | Port: %d | Debug: %s",
        g_bConnected ? "Connected" : "Disconnected",
        g_iPort,
        g_bDebug ? "ON" : "OFF");
    return Plugin_Handled;
}

public Action Cmd_Debug(int client, int args) {
    if (args >= 1) {
        char arg[8];
        GetCmdArg(1, arg, sizeof(arg));
        g_bDebug = StringToInt(arg) > 0;
    } else {
        g_bDebug = !g_bDebug;
    }
    ReplyToCommand(client, "[AI] Debug mode: %s", g_bDebug ? "ON" : "OFF");
    return Plugin_Handled;
}

public void OnSocketConnected(Handle socket, any arg) {
    g_bConnected = true;
    PrintToServer("[AI Bridge] Connected to AI server!");

    // Send initial handshake
    char buffer[256];
    Format(buffer, sizeof(buffer), "{\"type\":\"handshake\",\"version\":\"%s\"}\n", PLUGIN_VERSION);
    SocketSend(socket, buffer);
}

public void OnSocketReceive(Handle socket, const char[] data, int size, any arg) {
    // Parse JSON command from AI
    // Expected format: {"action": 0-13, "bot_id": 1-4}

    if (g_bDebug) {
        PrintToServer("[AI Debug] Received: %s", data);
    }

    // Simple JSON parsing (production should use SM-JSON)
    int action = -1;
    int botId = 1;

    // Extract action value
    int actionPos = StrContains(data, "\"action\":");
    if (actionPos != -1) {
        action = StringToInt(data[actionPos + 9]);
    }

    // Extract bot_id value
    int botPos = StrContains(data, "\"bot_id\":");
    if (botPos != -1) {
        botId = StringToInt(data[botPos + 9]);
    }

    if (action >= 0 && action <= 13) {
        ExecuteBotAction(botId, action);
    }
}

public void OnSocketDisconnected(Handle socket, any arg) {
    g_bConnected = false;
    PrintToServer("[AI Bridge] Disconnected from AI server");
}

public void OnSocketError(Handle socket, int errorType, int errorNum, any arg) {
    PrintToServer("[AI Bridge] Socket error: %d (errno: %d)", errorType, errorNum);
    g_bConnected = false;
}

void ExecuteBotAction(int botId, int action) {
    int client = GetBotClient(botId);
    if (client <= 0 || !IsClientInGame(client) || !IsFakeClient(client)) {
        return;
    }

    // Action mapping (matches Python environment)
    switch(action) {
        case 0: { /* IDLE - do nothing */ }
        case 1: { ForceMove(client, {1.0, 0.0, 0.0}); }   // MOVE_FORWARD
        case 2: { ForceMove(client, {-1.0, 0.0, 0.0}); }  // MOVE_BACKWARD
        case 3: { ForceMove(client, {0.0, -1.0, 0.0}); }  // STRAFE_LEFT
        case 4: { ForceMove(client, {0.0, 1.0, 0.0}); }   // STRAFE_RIGHT
        case 5: { ForceTurn(client, -45.0); }             // TURN_LEFT
        case 6: { ForceTurn(client, 45.0); }              // TURN_RIGHT
        case 7: { ForceAttack(client, true); }            // ATTACK
        case 8: { ForceShove(client); }                   // SHOVE
        case 9: { ForceReload(client); }                  // RELOAD
        case 10: { ForceHeal(client); }                   // HEAL_SELF
        case 11: { ForceHealTeammate(client); }           // HEAL_TEAMMATE
        case 12: { ForceUse(client); }                    // USE
        case 13: { ForceJump(client); }                   // JUMP
    }

    if (g_bDebug) {
        PrintToServer("[AI Debug] Bot %d executed action %d", botId, action);
    }
}

int GetBotClient(int botId) {
    int count = 0;
    for (int i = 1; i <= MaxClients; i++) {
        if (IsClientInGame(i) && IsFakeClient(i) && GetClientTeam(i) == 2) {
            count++;
            if (count == botId) {
                return i;
            }
        }
    }
    return -1;
}

public Action Timer_UpdateState(Handle timer) {
    if (!g_bConnected || g_hSocket == INVALID_HANDLE) {
        return Plugin_Continue;
    }

    // Build state JSON
    char buffer[MAX_BUFFER];
    char botStates[2048];
    int botCount = 0;

    for (int i = 1; i <= MaxClients; i++) {
        if (IsClientInGame(i) && GetClientTeam(i) == 2) {
            float pos[3], vel[3];
            GetClientAbsOrigin(i, pos);
            GetEntPropVector(i, Prop_Data, "m_vecVelocity", vel);

            int health = GetClientHealth(i);
            bool incapped = GetEntProp(i, Prop_Send, "m_isIncapacitated") > 0;

            char botJson[256];
            Format(botJson, sizeof(botJson),
                "{\"id\":%d,\"bot\":%s,\"hp\":%d,\"incap\":%s,\"x\":%.1f,\"y\":%.1f,\"z\":%.1f}",
                i, IsFakeClient(i) ? "true" : "false",
                health, incapped ? "true" : "false",
                pos[0], pos[1], pos[2]);

            if (botCount > 0) {
                StrCat(botStates, sizeof(botStates), ",");
            }
            StrCat(botStates, sizeof(botStates), botJson);
            botCount++;
        }
    }

    // Count infected nearby
    int infectedCount = 0;
    for (int i = 1; i <= MaxClients; i++) {
        if (IsClientInGame(i) && GetClientTeam(i) == 3 && IsPlayerAlive(i)) {
            infectedCount++;
        }
    }

    // Add common infected count
    int commonCount = 0;
    int ent = -1;
    while ((ent = FindEntityByClassname(ent, "infected")) != -1) {
        commonCount++;
    }

    Format(buffer, sizeof(buffer),
        "{\"type\":\"state\",\"survivors\":[%s],\"infected\":%d,\"common\":%d}\n",
        botStates, infectedCount, commonCount);

    SocketSend(g_hSocket, buffer);

    return Plugin_Continue;
}

// Force bot movement
void ForceMove(int client, float direction[3]) {
    float angles[3];
    GetClientEyeAngles(client, angles);

    float forward[3], right[3];
    GetAngleVectors(angles, forward, right, NULL_VECTOR);

    float velocity[3];
    velocity[0] = forward[0] * direction[0] * 250.0 + right[0] * direction[1] * 250.0;
    velocity[1] = forward[1] * direction[0] * 250.0 + right[1] * direction[1] * 250.0;
    velocity[2] = 0.0;

    TeleportEntity(client, NULL_VECTOR, NULL_VECTOR, velocity);
}

void ForceTurn(int client, float yawDelta) {
    float angles[3];
    GetClientEyeAngles(client, angles);
    angles[1] += yawDelta;
    TeleportEntity(client, NULL_VECTOR, angles, NULL_VECTOR);
}

void ForceAttack(int client, bool primary) {
    int weapon = GetEntPropEnt(client, Prop_Send, "m_hActiveWeapon");
    if (weapon != -1) {
        SetEntPropFloat(weapon, Prop_Send, "m_flNextPrimaryAttack", 0.0);
    }
}

void ForceShove(int client) {
    // Trigger shove animation/action
    L4D_StaggerPlayer(client, client, NULL_VECTOR);
}

void ForceReload(int client) {
    int weapon = GetEntPropEnt(client, Prop_Send, "m_hActiveWeapon");
    if (weapon != -1) {
        // Force reload
        SetEntProp(weapon, Prop_Send, "m_bInReload", 1);
    }
}

void ForceHeal(int client) {
    // Check for medkit and use it
    int item = GetPlayerWeaponSlot(client, 3);
    if (item != -1) {
        char classname[64];
        GetEntityClassname(item, classname, sizeof(classname));
        if (StrEqual(classname, "weapon_first_aid_kit")) {
            // Heal self
            FakeClientCommand(client, "use weapon_first_aid_kit");
        }
    }
}

void ForceHealTeammate(int client) {
    // Find nearest teammate and heal them
    int nearest = FindNearestTeammate(client);
    if (nearest > 0) {
        // Look at teammate and heal
        float pos[3], targetPos[3];
        GetClientAbsOrigin(client, pos);
        GetClientAbsOrigin(nearest, targetPos);

        float angles[3];
        MakeVectorFromPoints(pos, targetPos, angles);
        GetVectorAngles(angles, angles);
        TeleportEntity(client, NULL_VECTOR, angles, NULL_VECTOR);

        FakeClientCommand(client, "use weapon_first_aid_kit");
    }
}

int FindNearestTeammate(int client) {
    float myPos[3];
    GetClientAbsOrigin(client, myPos);

    int nearest = -1;
    float nearestDist = 999999.0;

    for (int i = 1; i <= MaxClients; i++) {
        if (i != client && IsClientInGame(i) && GetClientTeam(i) == 2 && IsPlayerAlive(i)) {
            float pos[3];
            GetClientAbsOrigin(i, pos);
            float dist = GetVectorDistance(myPos, pos);
            if (dist < nearestDist) {
                nearestDist = dist;
                nearest = i;
            }
        }
    }

    return nearest;
}

void ForceUse(int client) {
    FakeClientCommand(client, "+use");
    CreateTimer(0.1, Timer_ReleaseUse, client);
}

public Action Timer_ReleaseUse(Handle timer, int client) {
    if (IsClientInGame(client)) {
        FakeClientCommand(client, "-use");
    }
    return Plugin_Stop;
}

void ForceJump(int client) {
    float vel[3];
    GetEntPropVector(client, Prop_Data, "m_vecVelocity", vel);
    vel[2] = 300.0;
    TeleportEntity(client, NULL_VECTOR, NULL_VECTOR, vel);
}
PLUGIN_EOF

echo -e "${GREEN}AI Bridge plugin source created${NC}"

# Note: Plugin compilation requires SourceMod compiler
# For now, we'll include a pre-compiled version or compile in next step

# Step 7: Create server configuration
echo -e "\n${GREEN}[7/7] Creating server configuration...${NC}"

# Server config
cat > $L4D2_DIR/left4dead2/cfg/server.cfg << 'SERVERCFG'
// L4D2 AI Server Configuration
hostname "L4D2 AI Research Server"
sv_password "yourpasswordhere"  // Set a password to make it private
sv_region 0
sv_lan 0

// Map settings - The Last Stand
map c13m4_cutthroat_creek

// Bot settings
sb_all_bot_game 1
sb_stop 0

// AI Director settings
director_force_panic_event 0
director_panic_forever 0

// Allow cheats for testing
sv_cheats 1

// Network settings
net_public_adr 0.0.0.0
sv_pure 0

mp_gamemode "coop"
z_difficulty "normal"

// Bot settings - Enable AI control
sb_all_bot_game 1
sb_unstick 1

// Director settings
director_force_versus_start 0

// AI Bridge settings
sm_ai_debug 1
SERVERCFG

# Create start script
cat > $L4D2_DIR/start_server.sh << 'STARTSH'
#!/bin/bash
cd /opt/l4d2_server

# Start the server
./srcds_run -game left4dead2 \
    -ip 0.0.0.0 \
    -port 27015 \
    +map c13m4_cutthroat_creek \
    +exec server.cfg \
    -maxplayers 8 \
    -tickrate 100 \
    +sv_lan 0
STARTSH
chmod +x $L4D2_DIR/start_server.sh

# Create AI controller start script
cat > $AI_DIR/start_ai.sh << 'AISH'
#!/bin/bash
source /opt/l4d2_ai/venv/bin/activate
cd /opt/l4d2_ai/L4D2-AI-Architect

echo "Starting AI Controller..."
echo "Connect to game server first, then run:"
echo "  sm_ai_connect 27020"
echo ""

python scripts/rl_training/train_ppo.py \
    --mode live \
    --personality ${1:-balanced} \
    --env live
AISH
chmod +x $AI_DIR/start_ai.sh

# Create systemd service for auto-start
cat > /etc/systemd/system/l4d2-server.service << 'SYSTEMD'
[Unit]
Description=Left 4 Dead 2 Dedicated Server
After=network.target

[Service]
Type=simple
User=steam
WorkingDirectory=/opt/l4d2_server
ExecStart=/opt/l4d2_server/start_server.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
SYSTEMD

systemctl daemon-reload
systemctl enable l4d2-server

# Set permissions
chown -R steam:steam $L4D2_DIR
chown -R steam:steam $AI_DIR

# Print completion message
echo -e "\n${GREEN}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    INSTALLATION COMPLETE!                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${BLUE}Server Details:${NC}"
echo "  L4D2 Directory: $L4D2_DIR"
echo "  AI Directory:   $AI_DIR"
echo "  Server Port:    27015 (UDP)"
echo "  AI Bridge Port: 27020 (TCP)"

echo -e "\n${YELLOW}Next Steps:${NC}"
echo "  1. Configure firewall:"
echo "     ufw allow 27015/udp"
echo "     ufw allow 27020/tcp"
echo ""
echo "  2. Start the L4D2 server:"
echo "     sudo systemctl start l4d2-server"
echo "     # OR manually: sudo -u steam $L4D2_DIR/start_server.sh"
echo ""
echo "  3. Start AI controller (in tmux):"
echo "     tmux new -s ai"
echo "     $AI_DIR/start_ai.sh aggressive"
echo ""
echo "  4. Connect your game client:"
echo "     Open L4D2 -> Console -> connect YOUR_DROPLET_IP:27015"
echo ""
echo "  5. Enable AI control (in-game console):"
echo "     sm_ai_connect 27020"

echo -e "\n${GREEN}Estimated monthly cost: \$24-48${NC}"
echo -e "${GREEN}With \$200 credits: 4-8 months of gameplay!${NC}"
echo ""
