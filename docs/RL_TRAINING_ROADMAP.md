# RL Training Pipeline Roadmap

**Last Updated**: January 8, 2026
**Status**: Development (Not Production Ready)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current Implementation State](#current-implementation-state)
3. [Requirements for Live Training](#requirements-for-live-training)
4. [Mock/Simulation Training Alternative](#mocksimulation-training-alternative)
5. [GPU Requirements](#gpu-requirements)
6. [Bot Personality System](#bot-personality-system)
7. [Step-by-Step Live Server Guide](#step-by-step-live-server-guide)
8. [Advancing Without a Game Server](#advancing-without-a-game-server)
9. [Blockers and Risks](#blockers-and-risks)
10. [Recommended Next Steps](#recommended-next-steps)

---

## Executive Summary

The RL training component of L4D2-AI-Architect trains Proximal Policy Optimization (PPO) agents to control survivor bots in Left 4 Dead 2. The implementation is **functional but incomplete** - Python code is ready, but the critical **Mnemosyne SourceMod plugin** that bridges Python to the game does not exist in this repository.

### Current Capabilities

| Component | Status | Notes |
|-----------|--------|-------|
| Gymnasium Environment | Complete | `mnemosyne_env.py` - 20D observation, 14 actions |
| PPO Training Script | Complete | `train_ppo.py` - SB3 integration, checkpoints |
| Personality System | Complete | 5 presets with reward shaping |
| AI Director | Complete | Rule-based and RL modes |
| Game Bridge | Complete | TCP/UDP/Mock protocols |
| **Mnemosyne Plugin** | **MISSING** | SourceMod plugin not in repo |
| Simulation Mode | Basic | Random walk, limited fidelity |

### Bottom Line

- **With L4D2 Server**: Need to build the Mnemosyne SourceMod plugin first
- **Without Server**: Can train with mock environment but results will not transfer well to real gameplay

---

## Current Implementation State

### 1. Mnemosyne Environment (`scripts/rl_training/mnemosyne_env.py`)

A Gymnasium-compatible wrapper that defines:

**Observation Space (20D Continuous)**:
```python
[
    health / 100.0,              # 0: Normalized health
    is_alive,                    # 1: Binary alive status
    is_incapped,                 # 2: Binary incapped status
    position_x / 10000.0,        # 3-5: Normalized position
    position_y / 10000.0,
    position_z / 1000.0,
    velocity_x / 500.0,          # 6-8: Normalized velocity
    velocity_y / 500.0,
    velocity_z / 500.0,
    angle_pitch / 180.0,         # 9-10: Normalized view angles
    angle_yaw / 180.0,
    primary_weapon / 20.0,       # 11: Weapon ID
    ammo / 100.0,                # 12: Ammo count
    nearby_enemies / 10.0,       # 13: Enemy density
    nearest_enemy_dist / 2000.0, # 14: Distance to threat
    nearest_teammate_dist / 2000.0, # 15: Team cohesion
    teammates_alive / 3.0,       # 16: Team status
    teammates_incapped / 3.0,    # 17: Team distress
    in_safe_room,                # 18: Objective reached
    near_objective,              # 19: Objective proximity
]
```

**Action Space (14 Discrete Actions)**:
```python
class BotAction(IntEnum):
    IDLE = 0
    MOVE_FORWARD = 1
    MOVE_BACKWARD = 2
    MOVE_LEFT = 3
    MOVE_RIGHT = 4
    ATTACK = 5
    USE = 6
    RELOAD = 7
    CROUCH = 8
    JUMP = 9
    SHOVE = 10
    HEAL_SELF = 11
    HEAL_OTHER = 12
    THROW_ITEM = 13
```

**Communication Protocol**:
- UDP-based binary protocol
- Port: 27050 (configurable)
- Message types:
  - `0x01`: Connect/Acknowledge
  - `0x02`: Disconnect
  - `0x03`: Action command
  - `0x04`: State update
  - `0x05`: Reset episode

### 2. PPO Training Script (`scripts/rl_training/train_ppo.py`)

Stable-Baselines3 PPO with:
- Vectorized environments (parallel training)
- TensorBoard logging
- Checkpoint saving/resuming
- Personality-based reward shaping
- Evaluation callbacks

### 3. AI Director (`scripts/director/`)

Three-mode director system:
- **Rule-Based**: Configurable spawn rates, stress calculation
- **RL-Based**: Learn optimal difficulty curve (placeholder)
- **Hybrid**: Combine both approaches

---

## Requirements for Live Training

### Critical Missing Component: Mnemosyne SourceMod Plugin

The Python environment expects a SourceMod plugin that:

1. **Listens on UDP port 27050** for Python connections
2. **Takes control of bot inputs** via `OnPlayerRunCmd`
3. **Sends game state** in binary format (see protocol above)
4. **Handles episode resets** (teleport to start, reset health)

**What needs to be built**:

```cpp
// Conceptual SourceMod plugin structure
#include <sourcemod>
#include <sdktools>
#include <socket>  // or use UDP extension

Handle g_Socket;
int g_BotId;

public void OnPluginStart()
{
    // Create UDP socket listening on 27050
    // Register for bot control
}

public Action OnPlayerRunCmd(int client, int &buttons, ...)
{
    // If bot is controlled by Python agent:
    // 1. Read action from message queue
    // 2. Translate to buttons/angles
    // 3. Override buttons
}

// Timer to send state updates at 100Hz
public Action Timer_SendState(Handle timer)
{
    // Pack GameState into binary format
    // Send via UDP
}
```

**Effort Estimate**: 2-4 days for experienced SourceMod developer

### L4D2 Dedicated Server Requirements

| Requirement | Specification |
|-------------|---------------|
| OS | Windows or Linux |
| L4D2 Server | Dedicated server installation |
| SourceMod | Version 1.11+ |
| Metamod | Version 1.11+ |
| Extensions | Socket extension or UDP library |
| Network | Low latency connection to training machine |

### Server Setup Steps

```bash
# 1. Install SteamCMD and L4D2 Dedicated Server
steamcmd +login anonymous +force_install_dir ./l4d2_server \
    +app_update 222860 validate +quit

# 2. Install Metamod:Source
# Download from metamodsource.net, extract to addons/

# 3. Install SourceMod
# Download from sourcemod.net, extract to addons/

# 4. Install Mnemosyne plugin (DOES NOT EXIST YET)
# Copy to addons/sourcemod/plugins/

# 5. Configure server
echo "sv_cheats 1" >> cfg/server.cfg
echo "sm plugins load mnemosyne" >> cfg/server.cfg
```

---

## Mock/Simulation Training Alternative

### Current Simulation Mode

The environment includes basic simulation when no server is connected:

```python
def _simulate_state(self) -> GameState:
    """Generate simulated state for testing without game connection."""
    # Simple simulation: random walk with some enemy encounters
    # - 5% chance of taking damage per step
    # - Random enemy spawning
    # - Random position drift
```

### Limitations of Current Simulation

| Aspect | Simulation | Real Game |
|--------|------------|-----------|
| Movement physics | Random walk | Source engine physics |
| Combat | 5% damage chance | Actual enemy AI |
| Navigation | Open field | Complex map geometry |
| Team behavior | Static | AI teammate decisions |
| Episode length | Fixed steps | Map completion |
| Transfer learning | Poor | N/A |

### Improving the Simulation

Options for better offline training:

1. **Enhanced Mock Environment**
   - Record real gameplay data and replay trajectories
   - Model enemy behavior with simple state machines
   - Add map geometry constraints

2. **L4D2 Replay Analysis**
   - Parse demo files (`.dem`) to extract state transitions
   - Train on recorded human gameplay
   - Imitation learning baseline

3. **Unity/Godot Simulator**
   - Build simplified L4D2 mechanics in a game engine
   - Faster iteration than real game
   - Full control over physics and AI

---

## GPU Requirements

### PPO Training is CPU-Bound (Mostly)

PPO with MLP policy is lightweight compared to LLM training:

| Config | VRAM Required | Training Speed |
|--------|---------------|----------------|
| 4 parallel envs, MLP | < 1GB | ~1000 steps/sec |
| 8 parallel envs, MLP | < 2GB | ~2000 steps/sec |
| 16 parallel envs, MLP | < 4GB | ~4000 steps/sec |

### Vultr GPU Recommendations

| Instance | Best For | Cost |
|----------|----------|------|
| **CPU-only** | Mock training, prototyping | $0.05-0.10/hr |
| **A40 (48GB)** | Production training with CNN | $1.50/hr |
| **A100 (40GB)** | Fast large-batch training | $2.00/hr |

**Recommendation**: Start with CPU or small GPU. PPO with discrete actions and MLP policy does not require significant GPU resources.

### Training Time Estimates

For 1 million timesteps with 4 parallel environments:

| Environment | Time | Notes |
|-------------|------|-------|
| Mock (no game) | ~15-30 minutes | CPU-bound |
| Live game | ~3-6 hours | Limited by game tick rate |

---

## Bot Personality System

### Five Preset Personalities

Each personality uses different reward weights to shape behavior:

#### 1. Balanced (Default)
```python
{
    "kill": 1.0,
    "damage_dealt": 0.1,
    "damage_taken": -0.1,
    "heal_teammate": 5.0,
    "incapped": -10.0,
    "death": -50.0,
    "safe_room": 100.0,
    "survival": 0.01,
    "proximity_to_team": 0.001,
}
```
- Well-rounded behavior
- Balances combat, healing, and objectives
- Stays with team

#### 2. Aggressive
```python
{
    "kill": 3.0,            # 3x kill reward
    "damage_dealt": 0.3,    # Values damage output
    "damage_taken": -0.05,  # Less cautious
    "heal_teammate": 1.0,   # Low healing priority
    "incapped": -5.0,       # Risk-tolerant
    "death": -30.0,
    "safe_room": 50.0,      # Less objective-focused
    "survival": 0.005,
    "proximity_to_team": 0.0,  # Solo player
}
```
- Prioritizes kills over survival
- Ignores team coordination
- PPO config: Higher entropy (exploration)

#### 3. Medic
```python
{
    "kill": 0.5,
    "damage_dealt": 0.05,
    "damage_taken": -0.2,   # Very cautious
    "heal_teammate": 15.0,  # 3x healing reward
    "incapped": -15.0,
    "death": -100.0,        # Survival priority
    "safe_room": 100.0,
    "survival": 0.02,
    "proximity_to_team": 0.01,  # Team cohesion
}
```
- Avoids combat, focuses on support
- High team proximity reward
- PPO config: Lower entropy (exploitation)

#### 4. Speedrunner
```python
{
    "kill": 0.2,            # Minimal combat
    "damage_dealt": 0.0,
    "damage_taken": -0.05,
    "heal_teammate": 0.5,
    "incapped": -20.0,
    "death": -50.0,
    "safe_room": 200.0,     # 2x objective reward
    "survival": 0.0,
    "proximity_to_team": -0.001,  # Negative! Leaves team behind
}
```
- Rushes to objectives
- Ignores teammates and combat
- PPO config: Short gamma (immediate rewards)

#### 5. Defender
```python
{
    "kill": 2.0,
    "damage_dealt": 0.2,
    "damage_taken": -0.15,
    "heal_teammate": 8.0,
    "incapped": -15.0,
    "death": -80.0,
    "safe_room": 80.0,
    "survival": 0.02,
    "proximity_to_team": 0.02,  # High team cohesion
}
```
- Protects teammates
- Strong team coordination
- Balanced combat and support

### PPO Hyperparameter Adjustments by Personality

```python
if personality == "aggressive":
    config["ent_coef"] = 0.02  # More exploration
    config["gamma"] = 0.95     # Less long-term planning
elif personality == "medic":
    config["gamma"] = 0.995    # More long-term planning
    config["ent_coef"] = 0.005 # More exploitation
elif personality == "speedrunner":
    config["gamma"] = 0.9      # Short-term focused
    config["ent_coef"] = 0.03  # High exploration
```

---

## Step-by-Step Live Server Guide

### Prerequisites Checklist

- [ ] L4D2 Dedicated Server running
- [ ] SourceMod 1.11+ installed
- [ ] Mnemosyne plugin installed (needs development)
- [ ] Python environment with dependencies
- [ ] Network path between server and training machine

### Training Commands

```bash
# 1. Activate environment
cd /Users/davidortiz/left4dead-model/L4D2-AI-Architect
source venv/bin/activate  # or: source activate.sh

# 2. Install RL dependencies
pip install stable-baselines3[extra] gymnasium tensorboard

# 3. Start L4D2 server with Mnemosyne plugin
# (On game server machine)
./srcds_run -game left4dead2 +map c1m1_hotel +sv_lan 1

# 4. Start TensorBoard (optional, new terminal)
tensorboard --logdir data/training_logs/rl --port 6006

# 5. Start training
python scripts/rl_training/train_ppo.py \
    --mode train \
    --timesteps 1000000 \
    --n-envs 4 \
    --personality balanced \
    --host <GAME_SERVER_IP> \
    --port 27050

# 6. Monitor at http://localhost:6006
```

### Resuming Training

```bash
# Resume from checkpoint
python scripts/rl_training/train_ppo.py \
    --mode train \
    --resume model_adapters/rl_agents/ppo_balanced_TIMESTAMP/checkpoints/ppo_checkpoint_500000_steps \
    --timesteps 500000
```

### Evaluating a Model

```bash
# Evaluate trained model
python scripts/rl_training/train_ppo.py \
    --mode eval \
    --model model_adapters/rl_agents/ppo_balanced_TIMESTAMP/final_model \
    --eval-episodes 100 \
    --personality balanced

# Demo mode (visual)
python scripts/rl_training/train_ppo.py \
    --mode demo \
    --model model_adapters/rl_agents/ppo_balanced_TIMESTAMP/final_model \
    --personality balanced
```

---

## Advancing Without a Game Server

### Option 1: Improve the Mock Environment

**Current simulation is too simple**. To make it useful:

```python
# Enhanced simulation ideas:

class EnhancedMockEnv(MnemosyneEnv):
    def __init__(self):
        super().__init__()
        # Load map geometry (simplified nav mesh)
        self.nav_mesh = load_nav_mesh("c1m1_hotel")
        # Load enemy behavior model
        self.enemy_ai = SimpleZombieAI()

    def _simulate_state(self):
        # Physics-based movement
        new_pos = self._apply_physics(self.current_state.position, action)

        # Collision with geometry
        if not self.nav_mesh.is_valid_position(new_pos):
            new_pos = self._find_valid_position(new_pos)

        # Enemy AI
        for enemy in self.enemies:
            enemy.update(self.current_state.position)
            if enemy.can_attack():
                damage = enemy.attack()
                self.current_state.health -= damage
```

**Effort**: 1-2 weeks

### Option 2: Build a Simple Game Simulator

Create a lightweight L4D2-like environment:

```
/simulated_l4d2/
├── maps/
│   └── simple_corridor.json  # 2D grid map
├── entities/
│   ├── survivor.py           # Player mechanics
│   └── zombie.py             # Enemy AI
└── simulator.py              # Main loop
```

**Effort**: 2-4 weeks

### Option 3: Imitation Learning from Demos

1. Collect L4D2 demo files from skilled players
2. Parse demos to extract state-action pairs
3. Train with behavioral cloning
4. Fine-tune with RL

**Effort**: 1-2 weeks (if demo parsing works)

### Option 4: Transfer from Other Games

Train on similar games with better simulation support:
- Use DeepMind's OpenSpiel for abstract games
- Use PettingZoo for multi-agent scenarios
- Transfer policy architecture to L4D2

**Effort**: Variable

### Recommendation

**Prioritize Option 1 (Enhanced Mock)**:
1. Does not require game server
2. Allows rapid iteration
3. Can be improved incrementally
4. Eventually test on real game

---

## Blockers and Risks

### Critical Blockers

| Blocker | Impact | Mitigation |
|---------|--------|------------|
| **Missing Mnemosyne Plugin** | Cannot train on real game | Build plugin or use simulation |
| **No demo parsing** | Cannot do imitation learning | Implement demo parser |
| **Simulation-reality gap** | Mock-trained agents may fail | Validate on real game |

### Risks

| Risk | Likelihood | Impact | Notes |
|------|------------|--------|-------|
| SourceMod API changes | Low | Medium | Plugin may need updates |
| Training instability | Medium | Medium | PPO is generally stable |
| Reward hacking | High | High | Agents may exploit reward design |
| Performance overhead | Medium | Medium | Game may lag during training |

### Technical Debt

- Environment registration uses bare `except:` clause
- UDP protocol has no authentication
- No versioning on protocol format
- Limited error recovery in bridge

---

## Recommended Next Steps

### Immediate (This Week)

1. **Decide on training approach**:
   - If building Mnemosyne plugin: Allocate SourceMod developer time
   - If simulation-only: Enhance mock environment

2. **Test current simulation**:
   ```bash
   python scripts/rl_training/mnemosyne_env.py
   # Runs 100 random steps in simulation mode
   ```

3. **Run a mock training session**:
   ```bash
   python scripts/rl_training/train_ppo.py \
       --timesteps 100000 \
       --personality balanced
   # Will run in simulation mode (no game server)
   ```

### Short Term (2-4 Weeks)

If building the Mnemosyne plugin:
1. Design protocol specification document
2. Implement basic bot control (movement only)
3. Add observation streaming
4. Test single-bot training
5. Add multi-bot support

If enhancing simulation:
1. Add simple 2D map geometry
2. Implement basic zombie AI
3. Add weapon mechanics
4. Test training convergence

### Long Term (1-3 Months)

1. Multi-agent training (all 4 survivors)
2. Curriculum learning (easy to hard maps)
3. Self-play for infected control
4. Integration with AI Director

---

## Appendix: File Reference

| File | Purpose |
|------|---------|
| `scripts/rl_training/mnemosyne_env.py` | Gymnasium environment wrapper |
| `scripts/rl_training/train_ppo.py` | PPO training with SB3 |
| `scripts/director/director.py` | AI Director main logic |
| `scripts/director/bridge.py` | Game communication layer |
| `scripts/director/policy.py` | Director decision policies |
| `configs/director_config.yaml` | Director configuration |

---

## Appendix: Protocol Specification

### UDP Message Format

```
Byte 0: Message Type
Bytes 1+: Payload

Connect (0x01):     [type, bot_id]
Disconnect (0x02):  [type, bot_id]
Action (0x03):      [type, bot_id, action_id]
State (0x04):       [type, <binary GameState>]
Reset (0x05):       [type, bot_id]
```

### GameState Binary Format

```c
struct GameState {
    uint8_t  bot_id;              // 1 byte
    uint16_t health;              // 2 bytes
    uint8_t  is_alive;            // 1 byte
    uint8_t  is_incapped;         // 1 byte
    float    position[3];         // 12 bytes
    float    velocity[3];         // 12 bytes
    float    angle[2];            // 8 bytes
    uint8_t  primary_weapon;      // 1 byte
    uint8_t  secondary_weapon;    // 1 byte
    uint8_t  throwable;           // 1 byte
    uint8_t  health_item;         // 1 byte
    uint16_t ammo;                // 2 bytes
    uint8_t  nearby_enemies;      // 1 byte
    float    nearest_enemy_dist;  // 4 bytes
    uint8_t  teammates_alive;     // 1 byte
    float    nearest_teammate_dist; // 4 bytes
    uint8_t  teammates_incapped;  // 1 byte
    uint8_t  in_safe_room;        // 1 byte
    uint8_t  near_objective;      // 1 byte
    // Total: ~56 bytes
};
```

---

*Document generated January 8, 2026*
