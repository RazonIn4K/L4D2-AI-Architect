#!/usr/bin/env python3
"""
AI Director RL Training Script

Trains a PPO-based AI Director for L4D2 using Stable-Baselines3.
The director learns to create optimal gameplay pacing by controlling
spawn rates, events, and item distribution.

Features:
- Gymnasium environment wrapping the Director class
- PPO training with customizable personalities
- Engagement-based reward shaping
- TensorBoard logging and evaluation metrics
- Integration with bridge.py for live game use

Usage:
    # Train with default (standard) personality
    python train_director_rl.py --timesteps 500000

    # Train with specific personality
    python train_director_rl.py --personality intense --timesteps 1000000

    # Resume training from checkpoint
    python train_director_rl.py --resume model_adapters/director_agents/latest

    # Evaluate trained director
    python train_director_rl.py --mode eval --model model_adapters/director_agents/best_model
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

# Add parent to path for security utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_path, safe_write_json, safe_read_yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "model_adapters" / "director_agents"
LOGS_DIR = PROJECT_ROOT / "data" / "training_logs" / "director"
CONFIGS_DIR = PROJECT_ROOT / "configs"


def _resolve_path_within_root(path: Path, root: Path) -> Path:
    """Resolve path and ensure it stays within project root."""
    root_resolved = root.resolve()
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = root_resolved / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root_resolved):
        raise ValueError(f"Refusing path outside project root: {resolved}")
    return resolved


def install_dependencies():
    """Install required RL dependencies."""
    import subprocess
    deps = [
        "stable-baselines3[extra]",
        "gymnasium",
        "tensorboard",
    ]
    for dep in deps:
        # Use subprocess.run with list arguments to avoid command injection
        subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)


try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        EvalCallback,
        CheckpointCallback,
        CallbackList,
    )
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor
except ImportError:
    logger.info("Installing dependencies...")
    install_dependencies()
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        EvalCallback,
        CheckpointCallback,
        CallbackList,
    )
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor

# Import director components
try:
    from .director import L4D2Director, DirectorMode, GameState
    from .bridge import GameBridge, MockBridge
except ImportError:
    from director import L4D2Director, DirectorMode, GameState
    from bridge import GameBridge, MockBridge


class DirectorAction(IntEnum):
    """Discrete action space for the AI Director."""
    IDLE = 0                    # Do nothing this tick
    SPAWN_COMMONS_LOW = 1       # Spawn 1-3 common infected
    SPAWN_COMMONS_MED = 2       # Spawn 4-8 common infected
    SPAWN_COMMONS_HIGH = 3      # Spawn 9-15 common infected
    SPAWN_SMOKER = 4            # Spawn a smoker
    SPAWN_BOOMER = 5            # Spawn a boomer
    SPAWN_HUNTER = 6            # Spawn a hunter
    SPAWN_SPITTER = 7           # Spawn a spitter
    SPAWN_JOCKEY = 8            # Spawn a jockey
    SPAWN_WITCH = 9             # Spawn a witch
    SPAWN_TANK = 10             # Spawn a tank
    TRIGGER_PANIC = 11          # Trigger panic event (horde)
    DROP_HEALTH = 12            # Spawn health item
    DROP_THROWABLE = 13         # Spawn throwable item
    DROP_AMMO = 14              # Spawn ammo pile


@dataclass
class DirectorObservation:
    """Observation state for the director."""
    # Player stress and health
    avg_stress: float = 0.0
    avg_health: float = 100.0
    min_health: float = 100.0
    players_incapped: int = 0
    players_dead: int = 0

    # Current threats
    common_count: int = 0
    special_count: int = 0
    witch_count: int = 0
    tank_active: bool = False
    panic_active: bool = False

    # Inventory state
    items_available: int = 0
    health_packs_used: int = 0

    # Progress
    flow_progress: float = 0.0
    time_since_start: float = 0.0

    # Recent events
    recent_kills: int = 0
    recent_damage_taken: int = 0

    def to_observation(self) -> np.ndarray:
        """Convert to normalized numpy array for the policy."""
        return np.array([
            self.avg_stress,                          # 0-1
            self.avg_health / 100.0,                  # 0-1
            self.min_health / 100.0,                  # 0-1
            min(self.players_incapped, 4) / 4.0,      # 0-1
            min(self.players_dead, 4) / 4.0,          # 0-1
            min(self.common_count, 50) / 50.0,        # 0-1
            min(self.special_count, 8) / 8.0,         # 0-1
            min(self.witch_count, 3) / 3.0,           # 0-1
            float(self.tank_active),                  # 0-1
            float(self.panic_active),                 # 0-1
            min(self.items_available, 20) / 20.0,     # 0-1
            min(self.health_packs_used, 10) / 10.0,   # 0-1
            self.flow_progress,                        # 0-1
            min(self.time_since_start, 1800) / 1800,  # 0-1 (30 min max)
            min(self.recent_kills, 20) / 20.0,        # 0-1
            min(self.recent_damage_taken, 200) / 200, # 0-1
        ], dtype=np.float32)


@dataclass
class EpisodeMetrics:
    """Metrics tracked during an episode for evaluation."""
    survival_time: float = 0.0
    total_damage_taken: int = 0
    items_used: int = 0
    kills: int = 0
    deaths: int = 0
    panic_events: int = 0
    tanks_spawned: int = 0
    witches_spawned: int = 0
    engagement_scores: List[float] = field(default_factory=list)
    stress_history: List[float] = field(default_factory=list)

    def calculate_engagement_score(self) -> float:
        """
        Calculate overall engagement score.
        Good engagement = sustained moderate stress with variation.
        Bad engagement = too easy (low stress) or too hard (high stress/deaths).
        """
        if not self.stress_history:
            return 0.0

        avg_stress = np.mean(self.stress_history)
        stress_variance = np.var(self.stress_history)

        # Optimal stress is around 0.4-0.6
        stress_optimality = 1.0 - abs(avg_stress - 0.5) * 2

        # Some variance is good (dynamic gameplay), but not too much
        variance_score = min(stress_variance * 10, 1.0)

        # Penalty for deaths
        death_penalty = max(0, 1.0 - self.deaths * 0.25)

        # Bonus for longer survival
        time_bonus = min(self.survival_time / 600, 1.0)  # Max at 10 minutes

        engagement = (
            0.4 * stress_optimality +
            0.2 * variance_score +
            0.2 * death_penalty +
            0.2 * time_bonus
        )

        return np.clip(engagement, 0, 1)


# Director personality presets
DIRECTOR_PERSONALITIES = {
    "relaxed": {
        "description": "Lower spawn rates, more items, gentle pacing",
        "target_stress": 0.3,
        "spawn_multiplier": 0.6,
        "item_multiplier": 1.5,
        "tank_frequency": 0.3,
        "witch_frequency": 0.5,
        "panic_frequency": 0.5,
        "reward_weights": {
            "engagement": 1.0,
            "survival_bonus": 2.0,
            "stress_penalty_high": -0.5,
            "stress_penalty_low": -0.1,
            "death_penalty": -1.0,
            "action_penalty": -0.01,
        }
    },
    "standard": {
        "description": "Balanced gameplay, moderate challenge",
        "target_stress": 0.5,
        "spawn_multiplier": 1.0,
        "item_multiplier": 1.0,
        "tank_frequency": 1.0,
        "witch_frequency": 1.0,
        "panic_frequency": 1.0,
        "reward_weights": {
            "engagement": 1.5,
            "survival_bonus": 1.0,
            "stress_penalty_high": -0.3,
            "stress_penalty_low": -0.3,
            "death_penalty": -2.0,
            "action_penalty": -0.005,
        }
    },
    "intense": {
        "description": "Higher spawn rates, fewer items, faster pacing",
        "target_stress": 0.7,
        "spawn_multiplier": 1.5,
        "item_multiplier": 0.7,
        "tank_frequency": 1.5,
        "witch_frequency": 1.3,
        "panic_frequency": 1.5,
        "reward_weights": {
            "engagement": 2.0,
            "survival_bonus": 0.5,
            "stress_penalty_high": -0.1,
            "stress_penalty_low": -0.5,
            "death_penalty": -1.5,
            "action_penalty": -0.003,
        }
    },
    "nightmare": {
        "description": "Maximum pressure, minimal resources",
        "target_stress": 0.85,
        "spawn_multiplier": 2.0,
        "item_multiplier": 0.4,
        "tank_frequency": 2.0,
        "witch_frequency": 2.0,
        "panic_frequency": 2.0,
        "reward_weights": {
            "engagement": 2.5,
            "survival_bonus": 0.2,
            "stress_penalty_high": 0.0,
            "stress_penalty_low": -1.0,
            "death_penalty": -0.5,  # Deaths are expected
            "action_penalty": -0.001,
        }
    },
}


class DirectorEnv(gym.Env):
    """
    Gymnasium environment for training the AI Director.

    The director observes the game state and decides what to spawn
    or what events to trigger to maintain optimal player engagement.

    Observation Space (16D continuous):
        - Average player stress level
        - Average/min player health
        - Players incapped/dead
        - Current threat counts (commons, specials, witch, tank)
        - Panic/event status
        - Item availability
        - Flow progress and time
        - Recent combat stats

    Action Space (15 discrete actions):
        - Idle, spawn various enemies, trigger events, drop items

    Reward:
        Based on maintaining target engagement level without
        overwhelming or underwhelming the players.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        personality: str = "standard",
        use_mock: bool = True,
        host: str = "localhost",
        port: int = 27050,
        max_episode_steps: int = 3000,  # ~5 minutes at 10Hz
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.personality = personality
        self.personality_config = DIRECTOR_PERSONALITIES.get(personality, DIRECTOR_PERSONALITIES["standard"])
        self.use_mock = use_mock
        self.host = host
        self.port = port
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        # Create bridge
        if use_mock:
            self.bridge = MockBridge()
        else:
            self.bridge = GameBridge(host, port)

        # Define spaces
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(16,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(len(DirectorAction))

        # Episode state
        self.current_step = 0
        self.episode_start_time = 0.0
        self.prev_observation: Optional[DirectorObservation] = None
        self.current_observation: Optional[DirectorObservation] = None
        self.metrics = EpisodeMetrics()

        # Action cooldowns (prevent spam)
        self.action_cooldowns = {
            DirectorAction.SPAWN_TANK: 0,
            DirectorAction.SPAWN_WITCH: 0,
            DirectorAction.TRIGGER_PANIC: 0,
        }
        self.cooldown_times = {
            DirectorAction.SPAWN_TANK: 300,    # 5 minutes
            DirectorAction.SPAWN_WITCH: 180,   # 3 minutes
            DirectorAction.TRIGGER_PANIC: 120, # 2 minutes
        }

        # Simulation state (for mock mode)
        self.sim_state = self._init_simulation_state()

    def _init_simulation_state(self) -> Dict[str, Any]:
        """Initialize simulation state for mock mode."""
        return {
            "survivors": [
                {"id": i, "health": 100, "tempHealth": 0, "isIncapped": False, "isDead": False,
                 "position": [i * 100, 0, 0]}
                for i in range(4)
            ],
            "commonInfected": 0,
            "specialInfected": [0, 0, 0, 0, 0],
            "witchCount": 0,
            "tankCount": 0,
            "tankActive": False,
            "panicActive": False,
            "itemsAvailable": 8,
            "healthPacksUsed": 0,
            "flowProgress": 0.0,
            "gameTime": 0.0,
            "recentKills": 0,
            "recentDamageTaken": 0,
        }

    def _get_observation_from_state(self, state: Dict[str, Any]) -> DirectorObservation:
        """Convert raw game state to DirectorObservation."""
        survivors = state.get("survivors", [])
        alive_survivors = [s for s in survivors if not s.get("isDead", False)]

        if alive_survivors:
            healths = [s.get("health", 0) + s.get("tempHealth", 0) for s in alive_survivors]
            avg_health = np.mean(healths)
            min_health = min(healths)
        else:
            avg_health = 0
            min_health = 0

        # Calculate stress based on current threats
        stress = self._calculate_stress(state)

        return DirectorObservation(
            avg_stress=stress,
            avg_health=avg_health,
            min_health=min_health,
            players_incapped=sum(1 for s in survivors if s.get("isIncapped", False)),
            players_dead=sum(1 for s in survivors if s.get("isDead", False)),
            common_count=state.get("commonInfected", 0),
            special_count=sum(state.get("specialInfected", [0, 0, 0, 0, 0])),
            witch_count=state.get("witchCount", 0),
            tank_active=state.get("tankActive", False),
            panic_active=state.get("panicActive", False),
            items_available=state.get("itemsAvailable", 0),
            health_packs_used=state.get("healthPacksUsed", 0),
            flow_progress=state.get("flowProgress", 0.0),
            time_since_start=state.get("gameTime", 0.0) - self.episode_start_time,
            recent_kills=state.get("recentKills", 0),
            recent_damage_taken=state.get("recentDamageTaken", 0),
        )

    def _calculate_stress(self, state: Dict[str, Any]) -> float:
        """Calculate current player stress level."""
        stress = 0.0

        survivors = state.get("survivors", [])
        alive_survivors = [s for s in survivors if not s.get("isDead", False)]

        if not alive_survivors:
            return 1.0

        # Low health stress
        avg_health = np.mean([s.get("health", 0) + s.get("tempHealth", 0) for s in alive_survivors])
        if avg_health < 50:
            stress += 0.3 * (1.0 - avg_health / 50.0)

        # Threat stress
        common = state.get("commonInfected", 0)
        if common > 10:
            stress += 0.2 * min(1.0, common / 30.0)

        specials = sum(state.get("specialInfected", [0, 0, 0, 0, 0]))
        if specials > 0:
            stress += 0.15 * min(1.0, specials / 4.0)

        if state.get("tankActive", False):
            stress += 0.3

        if state.get("panicActive", False):
            stress += 0.2

        if state.get("witchCount", 0) > 0:
            stress += 0.1

        # Incapped teammates
        incapped = sum(1 for s in survivors if s.get("isIncapped", False))
        stress += 0.15 * incapped

        return min(1.0, stress)

    def _apply_action(self, action: DirectorAction) -> bool:
        """
        Apply the director action to the game/simulation.
        Returns True if action was executed, False if on cooldown.
        """
        current_time = self.current_step

        # Check cooldowns
        if action in self.action_cooldowns:
            if self.action_cooldowns[action] > current_time:
                return False
            # Set cooldown
            cooldown = self.cooldown_times.get(action, 0)
            # Apply personality modifier
            if action == DirectorAction.SPAWN_TANK:
                cooldown /= self.personality_config["tank_frequency"]
            elif action == DirectorAction.SPAWN_WITCH:
                cooldown /= self.personality_config["witch_frequency"]
            elif action == DirectorAction.TRIGGER_PANIC:
                cooldown /= self.personality_config["panic_frequency"]
            self.action_cooldowns[action] = current_time + int(cooldown)

        if self.use_mock:
            return self._apply_action_simulation(action)
        else:
            return self._apply_action_bridge(action)

    def _apply_action_simulation(self, action: DirectorAction) -> bool:
        """Apply action in simulation mode."""
        spawn_mult = self.personality_config["spawn_multiplier"]
        item_mult = self.personality_config["item_multiplier"]

        if action == DirectorAction.IDLE:
            pass
        elif action == DirectorAction.SPAWN_COMMONS_LOW:
            self.sim_state["commonInfected"] += int(np.random.randint(1, 4) * spawn_mult)
        elif action == DirectorAction.SPAWN_COMMONS_MED:
            self.sim_state["commonInfected"] += int(np.random.randint(4, 9) * spawn_mult)
        elif action == DirectorAction.SPAWN_COMMONS_HIGH:
            self.sim_state["commonInfected"] += int(np.random.randint(9, 16) * spawn_mult)
        elif action == DirectorAction.SPAWN_SMOKER:
            self.sim_state["specialInfected"][0] += 1
        elif action == DirectorAction.SPAWN_BOOMER:
            self.sim_state["specialInfected"][1] += 1
        elif action == DirectorAction.SPAWN_HUNTER:
            self.sim_state["specialInfected"][2] += 1
        elif action == DirectorAction.SPAWN_SPITTER:
            self.sim_state["specialInfected"][3] += 1
        elif action == DirectorAction.SPAWN_JOCKEY:
            self.sim_state["specialInfected"][4] += 1
        elif action == DirectorAction.SPAWN_WITCH:
            self.sim_state["witchCount"] += 1
            self.metrics.witches_spawned += 1
        elif action == DirectorAction.SPAWN_TANK:
            self.sim_state["tankCount"] = 1
            self.sim_state["tankActive"] = True
            self.metrics.tanks_spawned += 1
        elif action == DirectorAction.TRIGGER_PANIC:
            self.sim_state["panicActive"] = True
            self.sim_state["commonInfected"] += int(20 * spawn_mult)
            self.metrics.panic_events += 1
        elif action == DirectorAction.DROP_HEALTH:
            self.sim_state["itemsAvailable"] += int(1 * item_mult)
        elif action == DirectorAction.DROP_THROWABLE:
            self.sim_state["itemsAvailable"] += int(1 * item_mult)
        elif action == DirectorAction.DROP_AMMO:
            pass  # Ammo doesn't affect itemsAvailable

        return True

    def _apply_action_bridge(self, action: DirectorAction) -> bool:
        """Apply action through the game bridge."""
        spawn_mult = self.personality_config["spawn_multiplier"]

        action_map = {
            DirectorAction.SPAWN_COMMONS_LOW: ("spawn_common", {"count": int(2 * spawn_mult)}),
            DirectorAction.SPAWN_COMMONS_MED: ("spawn_common", {"count": int(6 * spawn_mult)}),
            DirectorAction.SPAWN_COMMONS_HIGH: ("spawn_common", {"count": int(12 * spawn_mult)}),
            DirectorAction.SPAWN_SMOKER: ("spawn_special", {"type": "smoker"}),
            DirectorAction.SPAWN_BOOMER: ("spawn_special", {"type": "boomer"}),
            DirectorAction.SPAWN_HUNTER: ("spawn_special", {"type": "hunter"}),
            DirectorAction.SPAWN_SPITTER: ("spawn_special", {"type": "spitter"}),
            DirectorAction.SPAWN_JOCKEY: ("spawn_special", {"type": "jockey"}),
            DirectorAction.SPAWN_WITCH: ("spawn_witch", {}),
            DirectorAction.SPAWN_TANK: ("spawn_tank", {}),
            DirectorAction.TRIGGER_PANIC: ("trigger_panic", {}),
            DirectorAction.DROP_HEALTH: ("spawn_item", {"type": "medkit"}),
            DirectorAction.DROP_THROWABLE: ("spawn_item", {"type": "molotov"}),
            DirectorAction.DROP_AMMO: ("spawn_item", {"type": "ammo"}),
        }

        if action == DirectorAction.IDLE:
            return True

        if action in action_map:
            cmd_type, params = action_map[action]
            self.bridge.send_director_command(cmd_type, params)
            return True

        return False

    def _simulate_world_step(self):
        """Simulate one world step in mock mode."""
        # Simulate combat and attrition

        # Survivors fight commons
        alive_survivors = [s for s in self.sim_state["survivors"] if not s.get("isDead", False) and not s.get("isIncapped", False)]

        if alive_survivors and self.sim_state["commonInfected"] > 0:
            # Kill some commons
            kills = min(len(alive_survivors) * 2, self.sim_state["commonInfected"])
            self.sim_state["commonInfected"] = max(0, self.sim_state["commonInfected"] - kills)
            self.sim_state["recentKills"] = kills
            self.metrics.kills += kills

            # Take some damage
            damage = int(self.sim_state["commonInfected"] * 0.5)
            if damage > 0:
                for survivor in alive_survivors:
                    dmg = np.random.randint(0, min(5, damage // len(alive_survivors) + 1))
                    survivor["health"] = max(0, survivor["health"] - dmg)
                    self.sim_state["recentDamageTaken"] += dmg
                    self.metrics.total_damage_taken += dmg

        # Specials deal damage
        total_specials = sum(self.sim_state["specialInfected"])
        if total_specials > 0 and alive_survivors:
            for survivor in alive_survivors:
                if np.random.random() < 0.1 * total_specials:
                    dmg = np.random.randint(5, 15)
                    survivor["health"] = max(0, survivor["health"] - dmg)
                    self.metrics.total_damage_taken += dmg
            # Survivors kill specials
            kills = min(total_specials, len(alive_survivors))
            for i in range(5):
                if self.sim_state["specialInfected"][i] > 0 and kills > 0:
                    self.sim_state["specialInfected"][i] -= 1
                    kills -= 1
                    self.metrics.kills += 1

        # Tank damage
        if self.sim_state["tankActive"] and alive_survivors:
            for survivor in alive_survivors:
                if np.random.random() < 0.2:
                    dmg = np.random.randint(10, 30)
                    survivor["health"] = max(0, survivor["health"] - dmg)
                    self.metrics.total_damage_taken += dmg
            # Chance to kill tank
            if np.random.random() < 0.05 * len(alive_survivors):
                self.sim_state["tankActive"] = False
                self.sim_state["tankCount"] = 0
                self.metrics.kills += 1

        # Panic event decays
        if self.sim_state["panicActive"]:
            if np.random.random() < 0.1:
                self.sim_state["panicActive"] = False

        # Check for incap/death
        for survivor in self.sim_state["survivors"]:
            if survivor["health"] <= 0 and not survivor.get("isIncapped", False) and not survivor.get("isDead", False):
                survivor["isIncapped"] = True
            elif survivor.get("isIncapped", False):
                # Incapped survivors slowly die
                if np.random.random() < 0.02:
                    survivor["isDead"] = True
                    self.metrics.deaths += 1

        # Progress forward
        alive_count = sum(1 for s in self.sim_state["survivors"] if not s.get("isDead", False))
        if alive_count > 0:
            self.sim_state["flowProgress"] += 0.001 * alive_count

        # Update game time
        self.sim_state["gameTime"] += 0.1

        # Healing (if items available)
        incapped = [s for s in self.sim_state["survivors"] if s.get("isIncapped", False)]
        if incapped and self.sim_state["itemsAvailable"] > 0 and np.random.random() < 0.1:
            incapped[0]["isIncapped"] = False
            incapped[0]["health"] = 30
            self.sim_state["itemsAvailable"] -= 1
            self.sim_state["healthPacksUsed"] += 1
            self.metrics.items_used += 1

    def _calculate_reward(self, prev_obs: DirectorObservation, curr_obs: DirectorObservation, action: DirectorAction) -> float:
        """
        Calculate reward based on engagement and game state.

        The director is rewarded for:
        - Maintaining target stress level (engagement)
        - Keeping survivors alive
        - Not overwhelming or underwhelming players
        """
        weights = self.personality_config["reward_weights"]
        target_stress = self.personality_config["target_stress"]
        reward = 0.0

        # Engagement reward (being close to target stress)
        stress_diff = abs(curr_obs.avg_stress - target_stress)
        engagement = 1.0 - min(stress_diff * 2, 1.0)
        reward += weights["engagement"] * engagement
        self.metrics.engagement_scores.append(engagement)

        # Stress tracking
        self.metrics.stress_history.append(curr_obs.avg_stress)

        # Stress penalties (too high or too low)
        if curr_obs.avg_stress > target_stress + 0.2:
            reward += weights["stress_penalty_high"] * (curr_obs.avg_stress - target_stress - 0.2)
        elif curr_obs.avg_stress < target_stress - 0.2:
            reward += weights["stress_penalty_low"] * (target_stress - 0.2 - curr_obs.avg_stress)

        # Survival bonus
        alive_ratio = (4 - curr_obs.players_dead) / 4.0
        reward += weights["survival_bonus"] * alive_ratio * 0.01

        # Death penalty
        new_deaths = curr_obs.players_dead - prev_obs.players_dead
        if new_deaths > 0:
            reward += weights["death_penalty"] * new_deaths

        # Action penalty (prevent spam)
        if action != DirectorAction.IDLE:
            reward += weights["action_penalty"]

        return reward

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)

        # Reset episode state
        self.current_step = 0
        self.episode_start_time = time.time()
        self.metrics = EpisodeMetrics()

        # Reset cooldowns
        for action in self.action_cooldowns:
            self.action_cooldowns[action] = 0

        # Reset simulation state
        self.sim_state = self._init_simulation_state()

        # Connect bridge if not mock
        if not self.use_mock:
            self.bridge.connect()
            self.bridge.reset_episode()
            time.sleep(0.5)
            raw_state = self.bridge.get_game_state()
            if raw_state:
                self.current_observation = self._get_observation_from_state(raw_state)
            else:
                self.current_observation = DirectorObservation()
        else:
            self.current_observation = self._get_observation_from_state(self.sim_state)

        self.prev_observation = self.current_observation

        observation = self.current_observation.to_observation()
        info = {
            "personality": self.personality,
            "target_stress": self.personality_config["target_stress"],
        }

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        self.current_step += 1

        # Apply the action
        action_enum = DirectorAction(action)
        action_applied = self._apply_action(action_enum)

        # Simulate world step or get state from bridge
        if self.use_mock:
            self._simulate_world_step()
            self.current_observation = self._get_observation_from_state(self.sim_state)
        else:
            time.sleep(0.1)  # 10Hz tick rate
            raw_state = self.bridge.get_game_state()
            if raw_state:
                self.current_observation = self._get_observation_from_state(raw_state)
            else:
                self.current_observation = self.prev_observation

        # Calculate reward
        reward = self._calculate_reward(self.prev_observation, self.current_observation, action_enum)

        # Update survival time
        self.metrics.survival_time = self.current_step * 0.1

        # Check termination
        all_dead = self.current_observation.players_dead >= 4
        reached_end = self.current_observation.flow_progress >= 1.0
        terminated = all_dead or reached_end
        truncated = self.current_step >= self.max_episode_steps

        observation = self.current_observation.to_observation()

        info = {
            "step": self.current_step,
            "action": action_enum.name,
            "action_applied": action_applied,
            "stress": self.current_observation.avg_stress,
            "target_stress": self.personality_config["target_stress"],
            "engagement": self.metrics.engagement_scores[-1] if self.metrics.engagement_scores else 0,
            "deaths": self.current_observation.players_dead,
            "flow_progress": self.current_observation.flow_progress,
            "survival_time": self.metrics.survival_time,
            "all_dead": all_dead,
            "reached_end": reached_end,
        }

        if terminated or truncated:
            info["episode_metrics"] = {
                "survival_time": self.metrics.survival_time,
                "total_damage_taken": self.metrics.total_damage_taken,
                "items_used": self.metrics.items_used,
                "kills": self.metrics.kills,
                "deaths": self.metrics.deaths,
                "panic_events": self.metrics.panic_events,
                "tanks_spawned": self.metrics.tanks_spawned,
                "witches_spawned": self.metrics.witches_spawned,
                "engagement_score": self.metrics.calculate_engagement_score(),
                "avg_stress": np.mean(self.metrics.stress_history) if self.metrics.stress_history else 0,
            }

        self.prev_observation = self.current_observation

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment state."""
        if self.render_mode == "human":
            obs = self.current_observation
            print(f"\n--- Director Step {self.current_step} ({self.personality}) ---")
            print(f"Stress: {obs.avg_stress:.2f} (target: {self.personality_config['target_stress']:.2f})")
            print(f"Health: avg={obs.avg_health:.0f} min={obs.min_health:.0f}")
            print(f"Threats: {obs.common_count} commons, {obs.special_count} specials")
            print(f"Tank: {obs.tank_active} | Panic: {obs.panic_active}")
            print(f"Progress: {obs.flow_progress:.1%} | Deaths: {obs.players_dead}")

    def close(self):
        """Clean up resources."""
        if not self.use_mock:
            self.bridge.disconnect()


class DirectorMetricsCallback(BaseCallback):
    """Custom callback to log director-specific metrics."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_metrics = []

    def _on_step(self) -> bool:
        # Check for episode end
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals.get("infos", [{}])[i]
                if "episode_metrics" in info:
                    metrics = info["episode_metrics"]
                    self.episode_metrics.append(metrics)

                    # Log to TensorBoard
                    if self.logger:
                        for key, value in metrics.items():
                            self.logger.record(f"director/{key}", value)

        return True

    def _on_training_end(self):
        if self.episode_metrics:
            avg_engagement = np.mean([m["engagement_score"] for m in self.episode_metrics])
            avg_survival = np.mean([m["survival_time"] for m in self.episode_metrics])
            logger.info(f"Training complete - Avg Engagement: {avg_engagement:.3f}, Avg Survival: {avg_survival:.1f}s")


def make_director_env(
    personality: str = "standard",
    use_mock: bool = True,
    host: str = "localhost",
    port: int = 27050,
    rank: int = 0,
) -> Callable[[], gym.Env]:
    """Factory function to create director environments."""
    def _init() -> gym.Env:
        env = DirectorEnv(
            personality=personality,
            use_mock=use_mock,
            host=host,
            port=port + rank,
        )
        return Monitor(env)
    return _init


def create_vectorized_env(
    n_envs: int = 4,
    personality: str = "standard",
    use_mock: bool = True,
    host: str = "localhost",
    base_port: int = 27050,
) -> VecMonitor:
    """Create vectorized environment for parallel training."""
    env_fns = [
        make_director_env(personality, use_mock, host, base_port, i)
        for i in range(n_envs)
    ]

    vec_env = DummyVecEnv(env_fns)
    return VecMonitor(vec_env)


def get_ppo_config(personality: str = "standard") -> Dict[str, Any]:
    """Get PPO hyperparameters tuned for director training."""
    config = {
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "normalize_advantage": True,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "verbose": 1,
    }

    # Personality-specific adjustments
    if personality == "relaxed":
        config["gamma"] = 0.995  # More long-term planning
        config["ent_coef"] = 0.005
    elif personality == "intense":
        config["ent_coef"] = 0.02  # More exploration
        config["gamma"] = 0.95
    elif personality == "nightmare":
        config["ent_coef"] = 0.03  # High exploration
        config["gamma"] = 0.9

    return config


def train(
    total_timesteps: int = 500_000,
    n_envs: int = 4,
    personality: str = "standard",
    save_path: Optional[Path] = None,
    resume_from: Optional[Path] = None,
    use_mock: bool = True,
    host: str = "localhost",
    base_port: int = 27050,
    eval_freq: int = 10000,
    save_freq: int = 50000,
    log_dir: Optional[Path] = None,
):
    """Main training function for the AI Director."""

    # Setup directories
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = MODELS_DIR / f"director_{personality}_{timestamp}"
    save_path = _resolve_path_within_root(Path(save_path), PROJECT_ROOT)
    save_path.mkdir(parents=True, exist_ok=True)

    if log_dir is None:
        log_dir = LOGS_DIR / save_path.name
    log_dir = _resolve_path_within_root(Path(log_dir), PROJECT_ROOT)
    log_dir.mkdir(parents=True, exist_ok=True)

    if resume_from is not None:
        resume_from = _resolve_path_within_root(Path(resume_from), PROJECT_ROOT)

    logger.info(f"Training AI Director with '{personality}' personality")
    logger.info(f"Model save path: {save_path}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Target stress: {DIRECTOR_PERSONALITIES[personality]['target_stress']}")

    # Create environments
    logger.info(f"Creating {n_envs} parallel environments...")
    train_env = create_vectorized_env(n_envs, personality, use_mock, host, base_port)
    eval_env = create_vectorized_env(1, personality, use_mock, host, base_port + 100)

    # Get PPO config
    ppo_config = get_ppo_config(personality)
    ppo_config["tensorboard_log"] = str(log_dir)

    # Create or load model
    if resume_from and resume_from.exists():
        logger.info(f"Resuming from {resume_from}")
        model = PPO.load(resume_from, env=train_env)
        model.tensorboard_log = str(log_dir)
    else:
        logger.info("Creating new PPO model...")
        model = PPO(env=train_env, **ppo_config)

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,
        save_path=str(save_path / "checkpoints"),
        name_prefix="director_checkpoint",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path / "best_model"),
        log_path=str(log_dir / "eval"),
        eval_freq=eval_freq // n_envs,
        n_eval_episodes=10,
        deterministic=True,
    )

    metrics_callback = DirectorMetricsCallback(verbose=1)

    callbacks = CallbackList([checkpoint_callback, eval_callback, metrics_callback])

    # Train
    logger.info(f"Starting training for {total_timesteps:,} timesteps...")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    # Save final model
    final_path = save_path / "final_model"
    model.save(final_path)
    logger.info(f"Final model saved to {final_path}")

    # Evaluate final model
    logger.info("Evaluating final model...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=20, deterministic=True
    )
    logger.info(f"Final evaluation: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Save training info
    info = {
        "personality": personality,
        "personality_config": DIRECTOR_PERSONALITIES[personality],
        "total_timesteps": total_timesteps,
        "n_envs": n_envs,
        "final_mean_reward": float(mean_reward),
        "final_std_reward": float(std_reward),
        "ppo_config": {k: str(v) for k, v in ppo_config.items()},
        "completed_at": datetime.now().isoformat(),
    }

    safe_write_json(
        str(save_path / "training_info.json"),
        info,
        PROJECT_ROOT
    )

    # Cleanup
    train_env.close()
    eval_env.close()

    return model, save_path


def evaluate_director(
    model_path: Path,
    n_episodes: int = 50,
    personality: str = "standard",
    use_mock: bool = True,
    render: bool = False,
):
    """Evaluate a trained director model."""
    model_path = _resolve_path_within_root(model_path, PROJECT_ROOT)
    logger.info(f"Loading model from {model_path}")

    env = DirectorEnv(
        personality=personality,
        use_mock=use_mock,
        render_mode="human" if render else None,
    )
    env = Monitor(env)

    model = PPO.load(model_path, env=env)

    logger.info(f"Evaluating for {n_episodes} episodes...")

    all_metrics = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if render:
                env.render()

        if "episode_metrics" in info:
            all_metrics.append(info["episode_metrics"])
            logger.info(f"Episode {ep + 1}: Engagement={info['episode_metrics']['engagement_score']:.3f}, "
                       f"Survival={info['episode_metrics']['survival_time']:.1f}s")

    # Summary statistics
    if all_metrics:
        print("\n" + "=" * 60)
        print(f"EVALUATION SUMMARY ({n_episodes} episodes)")
        print("=" * 60)

        metrics_summary = {
            "engagement_score": np.mean([m["engagement_score"] for m in all_metrics]),
            "survival_time": np.mean([m["survival_time"] for m in all_metrics]),
            "total_damage_taken": np.mean([m["total_damage_taken"] for m in all_metrics]),
            "kills": np.mean([m["kills"] for m in all_metrics]),
            "deaths": np.mean([m["deaths"] for m in all_metrics]),
            "avg_stress": np.mean([m["avg_stress"] for m in all_metrics]),
        }

        for key, value in metrics_summary.items():
            print(f"  {key}: {value:.3f}")

        print("=" * 60)

        return metrics_summary

    env.close()
    return None


def demo(
    model_path: Path,
    personality: str = "standard",
    use_mock: bool = True,
):
    """Run a demo of the trained director."""
    model_path = _resolve_path_within_root(model_path, PROJECT_ROOT)
    logger.info(f"Loading model from {model_path}")

    env = DirectorEnv(
        personality=personality,
        use_mock=use_mock,
        render_mode="human",
    )

    model = PPO.load(model_path, env=env)

    obs, info = env.reset()

    print("\n" + "=" * 60)
    print("DIRECTOR DEMO MODE - Press Ctrl+C to stop")
    print(f"Personality: {personality}")
    print(f"Target stress: {DIRECTOR_PERSONALITIES[personality]['target_stress']}")
    print("=" * 60)

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            env.render()
            print(f"Action: {DirectorAction(action).name} | Reward: {reward:.3f}")

            time.sleep(0.2)  # Slow down for viewing

            if terminated or truncated:
                print("\n--- Episode ended ---")
                if "episode_metrics" in info:
                    print(f"Engagement Score: {info['episode_metrics']['engagement_score']:.3f}")
                    print(f"Survival Time: {info['episode_metrics']['survival_time']:.1f}s")
                print("-" * 40)

                obs, info = env.reset()

    except KeyboardInterrupt:
        print("\nDemo stopped")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train AI Director for L4D2")

    # Mode
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "eval", "demo"],
                        help="Operation mode")

    # Training parameters
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--personality", type=str, default="standard",
                        choices=list(DIRECTOR_PERSONALITIES.keys()),
                        help="Director personality preset")

    # Paths
    parser.add_argument("--save-path", type=str,
                        help="Path to save model")
    parser.add_argument("--resume", type=str,
                        help="Resume from checkpoint")
    parser.add_argument("--model", type=str,
                        help="Model path for eval/demo")

    # Environment options
    parser.add_argument("--use-mock", action="store_true", default=True,
                        help="Use mock environment (no game server)")
    parser.add_argument("--live", action="store_true",
                        help="Use live game server")
    parser.add_argument("--host", type=str, default="localhost",
                        help="Game server host")
    parser.add_argument("--port", type=int, default=27050,
                        help="Base port for connections")

    # Evaluation
    parser.add_argument("--eval-episodes", type=int, default=50,
                        help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true",
                        help="Render during evaluation/demo")

    args = parser.parse_args()

    # Determine mock mode
    use_mock = not args.live

    try:
        if args.mode == "train":
            train(
                total_timesteps=args.timesteps,
                n_envs=args.n_envs,
                personality=args.personality,
                save_path=Path(args.save_path) if args.save_path else None,
                resume_from=Path(args.resume) if args.resume else None,
                use_mock=use_mock,
                host=args.host,
                base_port=args.port,
            )

        elif args.mode == "eval":
            if not args.model:
                logger.error("--model required for evaluation")
                sys.exit(1)
            evaluate_director(
                model_path=Path(args.model),
                n_episodes=args.eval_episodes,
                personality=args.personality,
                use_mock=use_mock,
                render=args.render,
            )

        elif args.mode == "demo":
            if not args.model:
                logger.error("--model required for demo")
                sys.exit(1)
            demo(
                model_path=Path(args.model),
                personality=args.personality,
                use_mock=use_mock,
            )

    except ValueError as e:
        logger.error(str(e))
        sys.exit(2)


if __name__ == "__main__":
    main()
