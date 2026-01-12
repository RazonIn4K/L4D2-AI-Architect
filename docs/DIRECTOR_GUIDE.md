# L4D2 AI Director Guide

This guide covers the AI Director system for Left 4 Dead 2, which dynamically manages gameplay intensity by controlling special infected spawns, hordes, and events.

## Director Modes

The AI Director supports three operational modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| **RCON** | Direct RCON commands to game server | Simple setup, no plugins needed |
| **Rule-Based** | Configurable rules in YAML | Predictable behavior, easy to tune |
| **RL (Reinforcement Learning)** | PPO-trained neural network | Adaptive, learns optimal difficulty |
| **Hybrid** | Rule-based with RL suggestions | Best of both worlds |

## Quick Start

### Option 1: RCON Director (Simplest)

Works with any SourceMod-enabled server without additional plugins.

```bash
# On the game server
cd scripts/director
python rcon_director.py

# Or remotely
python rcon_director.py --host 104.248.183.166 --password ai2026
```

### Option 2: Full Director System

Requires the SourceMod bridge plugin installed.

```bash
# Start the director
./run_ai_director.sh

# Or with specific personality
./run_ai_director.sh --personality intense
```

## RCON Director

The `rcon_director.py` script is a standalone director that works via RCON:

### Features

- Monitors player count automatically
- Activates when humans join (1+ human)
- Intense mode when lobby is full (4+ players)
- Spawns special infected at configurable intervals
- Spawns Tanks periodically in intense mode
- Triggers hordes randomly

### Configuration

```bash
python rcon_director.py \
    --host 127.0.1.1 \
    --port 27015 \
    --password ai2026 \
    --spawn-min 25 \
    --spawn-max 45 \
    --tank-interval 180
```

### Stable Settings

The RCON director uses conservative settings to prevent server crashes:

```python
STABLE_LIMITS = {
    "z_hunter_limit": 2,
    "z_smoker_limit": 2,
    "z_boomer_limit": 2,
    "z_charger_limit": 2,
    "z_spitter_limit": 2,
    "z_jockey_limit": 2,
    "z_special_spawn_interval": 25,
    "z_common_limit": 25,
}
```

## Full Director System

### Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Director.py    │────▶│    Bridge.py    │────▶│  Game Server    │
│  (Decision)     │     │  (Communication)│     │  (L4D2)         │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │
        ▼
┌─────────────────┐
│   Policy.py     │
│  (Rule/RL/Hybrid)│
└─────────────────┘
```

### Components

| File | Purpose |
|------|---------|
| `director.py` | Main director logic, state management |
| `policy.py` | Decision-making (rule-based, RL, hybrid) |
| `bridge.py` | Game server communication |
| `simulation.py` | Testing without live server |
| `train_director_rl.py` | Train RL-based director |
| `rcon_director.py` | Standalone RCON-based director |

### Configuration

Edit `configs/director_config.yaml`:

```yaml
director:
  mode: "hybrid"  # rule_based, rl, or hybrid
  personality: "standard"  # standard, relaxed, intense, nightmare

spawn_rates:
  base_special_interval: 30
  min_special_interval: 10
  max_special_interval: 60

stress_factors:
  low_health_threshold: 30
  panic_threshold: 0.7
  relaxation_threshold: 0.3
```

## Director Personalities

| Personality | Special Interval | Tank Frequency | Horde Rate | Description |
|-------------|------------------|----------------|------------|-------------|
| **Standard** | 25-45s | Every 3 min | 5%/min | Balanced experience |
| **Relaxed** | 45-60s | Every 5 min | 2%/min | Casual gameplay |
| **Intense** | 15-30s | Every 2 min | 10%/min | Challenging |
| **Nightmare** | 10-20s | Every 1 min | 15%/min | Maximum difficulty |

## Training RL Director

### Prerequisites

```bash
pip install stable-baselines3 gymnasium
```

### Training

```bash
cd scripts/director
python train_director_rl.py \
    --timesteps 500000 \
    --personality standard \
    --save-path ../../model_adapters/director_agents/
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--timesteps` | 500000 | Total training steps |
| `--personality` | standard | Director personality to train |
| `--learning-rate` | 3e-4 | PPO learning rate |
| `--n-envs` | 4 | Parallel environments |
| `--checkpoint-freq` | 50000 | Save frequency |

### Evaluating Trained Models

```bash
python train_director_rl.py \
    --mode eval \
    --model ../../model_adapters/director_agents/standard/final_model \
    --eval-episodes 100
```

## Using the Director

### Starting with Rule-Based Mode

```python
from scripts.director import Director, DirectorConfig

config = DirectorConfig.from_yaml("configs/director_config.yaml")
config.mode = "rule_based"
config.personality = "standard"

director = Director(config)
director.run()
```

### Starting with RL Mode

```python
from scripts.director import Director, DirectorConfig

config = DirectorConfig.from_yaml("configs/director_config.yaml")
config.mode = "rl"
config.model_path = "model_adapters/director_agents/standard/final_model"

director = Director(config)
director.run()
```

### Hybrid Mode

Combines rule-based decisions with RL suggestions:

```python
config.mode = "hybrid"
config.rl_influence = 0.5  # 50% RL, 50% rules
```

## Simulation Mode

Test the director without a live game server:

```bash
cd scripts/director
python simulation.py --personality intense --episodes 10
```

This runs the director against a simulated game environment to verify behavior.

## Spawn Commands Reference

| Command | Effect |
|---------|--------|
| `z_spawn hunter` | Spawn Hunter |
| `z_spawn smoker` | Spawn Smoker |
| `z_spawn boomer` | Spawn Boomer |
| `z_spawn charger` | Spawn Charger |
| `z_spawn spitter` | Spawn Spitter |
| `z_spawn jockey` | Spawn Jockey |
| `z_spawn tank` | Spawn Tank |
| `z_spawn witch` | Spawn Witch |
| `z_spawn mob` | Spawn Horde |
| `director_force_panic_event` | Force Panic Event |

## Troubleshooting

### Director Won't Connect

1. Verify RCON is enabled on server
2. Check RCON password matches
3. For host networking: use `127.0.1.1` not `127.0.0.1`

### Spawns Not Working

1. Ensure `sv_cheats 1` is set (director does this automatically)
2. Verify SourceMod is loaded: `sm version`
3. Check spawn limits aren't at 0

### Server Crashes

1. Reduce spawn limits (max 2-3 per type)
2. Increase spawn intervals (min 20 seconds)
3. Avoid moon gravity and turbo zombies
4. Never spawn more than 3-4 infected at once

### High Latency

1. Run director on the same machine as server
2. Use UDP mode in bridge if available
3. Reduce status check frequency

## Integration with Bot Agents

The director can coordinate with RL bot agents:

```python
# Director informs bot controller of difficulty
director.set_difficulty_callback(bot_controller.on_difficulty_change)

# Bot controller can request spawns
bot_controller.set_spawn_callback(director.request_spawn)
```

## API Reference

### Director Class

```python
class Director:
    def __init__(self, config: DirectorConfig)
    def run(self) -> None
    def stop(self) -> None
    def spawn_infected(self, type: str) -> bool
    def spawn_horde(self) -> bool
    def get_game_state(self) -> GameState
    def set_personality(self, name: str) -> None
```

### DirectorConfig

```python
@dataclass
class DirectorConfig:
    mode: str = "hybrid"
    personality: str = "standard"
    host: str = "127.0.1.1"
    port: int = 27015
    password: str = "ai2026"
    model_path: Optional[str] = None
    rl_influence: float = 0.5
```

## Files

| File | Location |
|------|----------|
| Director code | `scripts/director/` |
| Configuration | `configs/director_config.yaml` |
| RCON Director | `scripts/director/rcon_director.py` |
| RL Training | `scripts/director/train_director_rl.py` |
| Simulation | `scripts/director/simulation.py` |
| Launcher | `run_ai_director.sh` |
