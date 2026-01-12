#!/usr/bin/env python3
"""
Director Policy Module

Implements decision-making logic for the AI Director.
Supports rule-based, RL-based, and hybrid modes.
"""

import sys
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

# Add parent to path for security utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_read_json

# Project root for path validation
PROJECT_ROOT = Path(__file__).parent.parent.parent

logger = logging.getLogger(__name__)


class DirectorMode(IntEnum):
    RULE_BASED = 0
    RL_BASED = 1
    HYBRID = 2


@dataclass
class DirectorAction:
    """Action the director can take"""
    action_type: str
    parameters: Dict[str, Any]
    priority: int
    reason: str


class RuleBasedPolicy:
    """Rule-based director policy"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.last_panic_time = 0
        self.last_tank_time = 0
        self.last_witch_time = 0
        self.min_panic_interval = 120  # seconds
        self.min_tank_interval = 300   # seconds
        self.min_witch_interval = 180  # seconds
        
    def decide(self, state: Dict[str, Any], metrics: Dict[str, Any]) -> List[DirectorAction]:
        """Make decisions based on rules"""
        actions = []
        current_time = state.get("game_time", 0)
        
        # Basic spawn decisions
        actions.extend(self._decide_common_spawns(state))
        actions.extend(self._decide_special_spawns(state))
        actions.extend(self._decide_events(state, current_time))
        actions.extend(self._decide_items(state))
        
        return actions
    
    def _decide_common_spawns(self, state: Dict[str, Any]) -> List[DirectorAction]:
        """Decide common infected spawns"""
        actions = []
        
        # Base spawn rate
        base_rate = self.config["spawn_rates"]["common_base"]
        multiplier = self.config["spawn_rates"]["common_multiplier"]
        
        # Adjust based on stress
        stress = state.get("stress_level", 0)
        adjusted_rate = base_rate + (stress * multiplier * 5)
        
        # Adjust based on flow progress
        flow = state.get("flow_progress", 0)
        if flow > 0.8:  # Near end
            adjusted_rate *= 1.5
        
        # Don't spawn if too many
        current_count = state.get("common_infected", 0)
        if current_count < 30 and np.random.random() < adjusted_rate:
            count = np.random.randint(3, 8)
            actions.append(DirectorAction(
                action_type="spawn_common",
                parameters={"count": count},
                priority=1,
                reason=f"Base spawn (stress: {stress:.2f})"
            ))
        
        return actions
    
    def _decide_special_spawns(self, state: Dict[str, Any]) -> List[DirectorAction]:
        """Decide special infected spawns"""
        actions = []
        
        base_rate = self.config["spawn_rates"]["special_base"]
        multiplier = self.config["spawn_rates"]["special_multiplier"]
        
        # Count current specials
        specials = state.get("special_infected", [0, 0, 0, 0, 0])
        current_specials = sum(specials)
        
        # Spawn based on stress and flow
        stress = state.get("stress_level", 0)
        flow = state.get("flow_progress", 0)
        
        if current_specials < 4 and np.random.random() < (base_rate + stress * multiplier):
            # Choose which special to spawn
            special_types = ["smoker", "boomer", "hunter", "spitter", "jockey"]
            
            # Prefer certain specials based on situation
            if flow > 0.5 and specials[2] < 1:  # Hunters in open areas
                chosen = "hunter"
            elif stress > 0.7 and specials[1] < 1:  # Boomers when stressed
                chosen = "boomer"
            else:
                chosen = np.random.choice(special_types)
            
            actions.append(DirectorAction(
                action_type="spawn_special",
                parameters={"type": chosen},
                priority=3,
                reason=f"Special spawn (stress: {stress:.2f})"
            ))
        
        return actions
    
    def _decide_events(self, state: Dict[str, Any], current_time: float) -> List[DirectorAction]:
        """Decide on major events (panics, tanks, witches)"""
        actions = []
        
        # Panic events
        if current_time - self.last_panic_time > self.min_panic_interval:
            stress = state.get("stress_level", 0)
            flow = state.get("flow_progress", 0)
            
            # Trigger panic based on conditions
            panic_chance = 0.01
            if stress > 0.6:
                panic_chance += 0.02
            if 0.3 < flow < 0.7:  # Mid-map
                panic_chance += 0.01
            
            if np.random.random() < panic_chance:
                actions.append(DirectorAction(
                    action_type="trigger_panic",
                    parameters={},
                    priority=5,
                    reason="Scheduled panic event"
                ))
                self.last_panic_time = current_time
        
        # Tank spawns
        if current_time - self.last_tank_time > self.min_tank_interval:
            flow = state.get("flow_progress", 0)
            
            # Tank chance increases with progress
            tank_chance = self.config["spawn_rates"]["tank_chance"]
            if flow > 0.5:
                tank_chance *= 2
            if flow > 0.8:
                tank_chance *= 3
            
            if state.get("tank_count", 0) == 0 and np.random.random() < tank_chance:
                actions.append(DirectorAction(
                    action_type="spawn_tank",
                    parameters={},
                    priority=6,
                    reason="Scheduled tank spawn"
                ))
                self.last_tank_time = current_time
        
        # Witch spawns
        if current_time - self.last_witch_time > self.min_witch_interval:
            witch_chance = self.config["spawn_rates"]["witch_chance"]
            flow = state.get("flow_progress", 0)
            
            # More witches in dark areas (simplified)
            if 0.2 < flow < 0.8:
                witch_chance *= 1.5
            
            if state.get("witch_count", 0) == 0 and np.random.random() < witch_chance:
                actions.append(DirectorAction(
                    action_type="spawn_witch",
                    parameters={},
                    priority=4,
                    reason="Scheduled witch spawn"
                ))
                self.last_witch_time = current_time
        
        return actions
    
    def _decide_items(self, state: Dict[str, Any]) -> List[DirectorAction]:
        """Decide item spawns"""
        actions = []
        
        survivors = state.get("survivors", [])
        if not survivors:
            return actions
        
        # Check team health
        avg_health = np.mean([s.get("health", 0) + s.get("tempHealth", 0) for s in survivors])
        items_available = state.get("items_available", 0)
        
        flow_control = self.config["flow_control"]
        items_per_survivor = items_available / len(survivors)
        
        # Spawn health items if team is struggling
        if avg_health < 40 and items_per_survivor < flow_control["max_items_per_checkpoint"]:
            if np.random.random() < 0.3:
                item_type = "medkit" if avg_health < 20 else "pills"
                actions.append(DirectorAction(
                    action_type="spawn_item",
                    parameters={"type": item_type},
                    priority=2,
                    reason=f"Low team health: {avg_health:.0f}"
                ))
        
        # Spawn throwables for special management
        specials = sum(state.get("special_infected", [0, 0, 0, 0, 0]))
        if specials > 2 and items_per_survivor < 2:
            if np.random.random() < 0.2:
                actions.append(DirectorAction(
                    action_type="spawn_item",
                    parameters={"type": "molotov"},
                    priority=2,
                    reason="Special infected management"
                ))
        
        return actions


class RLBasedPolicy:
    """RL-based director policy using trained PPO model from Stable-Baselines3"""

    # Action mapping from model output to director commands
    ACTION_MAP = {
        0: None,  # IDLE
        1: ("spawn_common", {"count": 2}),      # SPAWN_COMMONS_LOW
        2: ("spawn_common", {"count": 6}),      # SPAWN_COMMONS_MED
        3: ("spawn_common", {"count": 12}),     # SPAWN_COMMONS_HIGH
        4: ("spawn_special", {"type": "smoker"}),
        5: ("spawn_special", {"type": "boomer"}),
        6: ("spawn_special", {"type": "hunter"}),
        7: ("spawn_special", {"type": "spitter"}),
        8: ("spawn_special", {"type": "jockey"}),
        9: ("spawn_witch", {}),
        10: ("spawn_tank", {}),
        11: ("trigger_panic", {}),
        12: ("spawn_item", {"type": "medkit"}),
        13: ("spawn_item", {"type": "molotov"}),
        14: ("spawn_item", {"type": "ammo"}),
    }

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.difficulty_multiplier = 1.0
        self._load_model(model_path)

        # Fallback policy when model unavailable
        self.rule_policy = RuleBasedPolicy(self._get_default_config())

    def _load_model(self, model_path: Optional[str]) -> None:
        """Load trained PPO model from Stable-Baselines3 checkpoint."""
        if not model_path:
            logger.info("No RL model path provided, will use rule-based fallback")
            return

        model_file = Path(model_path)
        if not model_file.exists():
            # Try common extensions
            for ext in [".zip", ""]:
                candidate = Path(str(model_path) + ext)
                if candidate.exists():
                    model_file = candidate
                    break

        if not model_file.exists():
            logger.warning(f"RL model not found at {model_path}, using rule-based fallback")
            return

        try:
            from stable_baselines3 import PPO
            self.model = PPO.load(str(model_file))
            logger.info(f"Loaded RL director model from {model_file}")
        except ImportError:
            logger.warning("stable_baselines3 not installed, using rule-based fallback")
        except Exception as e:
            logger.error(f"Failed to load RL model: {e}, using rule-based fallback")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config for fallback"""
        return {
            "spawn_rates": {
                "common_base": 1.0,
                "common_multiplier": 0.1,
                "special_base": 0.02,
                "special_multiplier": 0.05,
                "witch_chance": 0.001,
                "tank_chance": 0.0005
            },
            "stress_factors": {
                "low_health": 0.3,
                "horde_active": 0.4,
                "special_active": 0.2,
                "item_shortage": 0.1
            },
            "flow_control": {
                "min_items_per_checkpoint": 4,
                "max_items_per_checkpoint": 8,
                "health_pack_ratio": 0.3,
                "throwable_ratio": 0.2
            }
        }
    
    def _state_to_observation(self, state: Dict[str, Any]) -> np.ndarray:
        """Convert game state dict to normalized observation vector for the model."""
        survivors = state.get("survivors", [])
        if survivors:
            healths = [s.get("health", 0) + s.get("tempHealth", 0) for s in survivors]
            avg_health = np.mean(healths)
            min_health = min(healths)
            incapped = sum(1 for s in survivors if s.get("incapped", False))
            dead = sum(1 for s in survivors if s.get("dead", False))
        else:
            avg_health, min_health, incapped, dead = 100, 100, 0, 0

        specials = state.get("special_infected", [0, 0, 0, 0, 0])

        return np.array([
            state.get("stress_level", 0.0),                           # avg_stress
            avg_health / 100.0,                                       # avg_health
            min_health / 100.0,                                       # min_health
            min(incapped, 4) / 4.0,                                   # players_incapped
            min(dead, 4) / 4.0,                                       # players_dead
            min(state.get("common_infected", 0), 50) / 50.0,          # common_count
            min(sum(specials), 8) / 8.0,                              # special_count
            min(state.get("witch_count", 0), 3) / 3.0,                # witch_count
            float(state.get("tank_count", 0) > 0),                    # tank_active
            float(state.get("panic_active", False)),                  # panic_active
            min(state.get("items_available", 0), 20) / 20.0,          # items_available
            min(state.get("health_packs_used", 0), 10) / 10.0,        # health_packs_used
            state.get("flow_progress", 0.0),                          # flow_progress
            min(state.get("game_time", 0), 1800) / 1800.0,            # time_since_start
            min(state.get("recent_kills", 0), 20) / 20.0,             # recent_kills
            min(state.get("recent_damage_taken", 0), 200) / 200.0,    # recent_damage_taken
        ], dtype=np.float32)

    def decide(self, state: Dict[str, Any], metrics: Dict[str, Any]) -> List[DirectorAction]:
        """Make decisions using RL model with fallback to rule-based policy."""
        if self.model is None:
            return self.rule_policy.decide(state, metrics)

        try:
            obs = self._state_to_observation(state)
            action, _ = self.model.predict(obs, deterministic=False)
            action_idx = int(action)

            action_def = self.ACTION_MAP.get(action_idx)
            if action_def is None:
                return []  # IDLE action

            action_type, params = action_def

            # Apply difficulty multiplier to spawn counts
            if action_type == "spawn_common" and "count" in params:
                params = dict(params)
                params["count"] = max(1, int(params["count"] * self.difficulty_multiplier))

            return [DirectorAction(
                action_type=action_type,
                parameters=params,
                priority=5 if action_type in ("spawn_tank", "trigger_panic") else 3,
                reason=f"RL policy action {action_idx}"
            )]

        except Exception as e:
            logger.warning(f"RL inference failed: {e}, using rule-based fallback")
            return self.rule_policy.decide(state, metrics)

    def update_difficulty(self, difficulty: float):
        """
        Adjust RL policy behavior via difficulty multiplier.
        Higher difficulty increases spawn counts and reduces item drops.
        """
        self.difficulty_multiplier = max(0.5, min(2.0, difficulty))
        logger.debug(f"RL policy difficulty multiplier set to {self.difficulty_multiplier}")


class HybridPolicy:
    """Hybrid policy combining rules and RL"""
    
    def __init__(self, config: Dict[str, Any], model_path: Optional[str] = None):
        self.rule_policy = RuleBasedPolicy(config)
        self.rl_policy = RLBasedPolicy(model_path)
        self.rl_weight = 0.5  # How much to trust RL vs rules
        
    def decide(self, state: Dict[str, Any], metrics: Dict[str, Any]) -> List[DirectorAction]:
        """Combine decisions from both policies"""
        rule_actions = self.rule_policy.decide(state, metrics)
        rl_actions = self.rl_policy.decide(state, metrics)
        
        # Simple combination: prefer RL for major events, rules for spawns
        combined = []
        
        # Add spawn actions from rules
        for action in rule_actions:
            if action.action_type in ["spawn_common", "spawn_special"]:
                combined.append(action)
        
        # Add event actions from RL (or rules if RL doesn't have them)
        event_actions = [a for a in rl_actions if a.action_type in ["trigger_panic", "spawn_tank", "spawn_witch"]]
        if not event_actions:
            event_actions = [a for a in rule_actions if a.action_type in ["trigger_panic", "spawn_tank", "spawn_witch"]]
        combined.extend(event_actions)
        
        # Add item actions from rules (more reliable)
        combined.extend([a for a in rule_actions if a.action_type == "spawn_item"])
        
        return combined
    
    def update_difficulty(self, difficulty: float):
        """Update policy difficulty"""
        self.rule_policy.config["base_difficulty"] = difficulty
        self.rl_policy.update_difficulty(difficulty)


class DirectorPolicy:
    """Main policy interface"""
    
    def __init__(self, mode: DirectorMode, config_path: Optional[str] = None):
        self.mode = mode
        
        # Load configuration using secure file reading
        if config_path and Path(config_path).exists():
            self.config = safe_read_json(config_path, PROJECT_ROOT)
        else:
            self.config = self._get_default_config()
        
        # Initialize appropriate policy
        if mode == DirectorMode.RULE_BASED:
            self.policy = RuleBasedPolicy(self.config)
        elif mode == DirectorMode.RL_BASED:
            self.policy = RLBasedPolicy()
        else:  # HYBRID
            self.policy = HybridPolicy(self.config)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "spawn_rates": {
                "common_base": 1.0,
                "common_multiplier": 0.1,
                "special_base": 0.02,
                "special_multiplier": 0.05,
                "witch_chance": 0.001,
                "tank_chance": 0.0005
            },
            "stress_factors": {
                "low_health": 0.3,
                "horde_active": 0.4,
                "special_active": 0.2,
                "item_shortage": 0.1
            },
            "flow_control": {
                "min_items_per_checkpoint": 4,
                "max_items_per_checkpoint": 8,
                "health_pack_ratio": 0.3,
                "throwable_ratio": 0.2
            },
            "difficulty": {
                "base_difficulty": 1.0,
                "adaptive_scaling": True
            }
        }
    
    def decide(self, state: Dict[str, Any], metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Make director decisions"""
        actions = self.policy.decide(state, metrics)
        
        # Convert to command format
        commands = []
        for action in actions:
            commands.append({
                "command_type": action.action_type,
                "parameters": action.parameters,
                "priority": action.priority,
                "delay": 0.0  # Immediate execution
            })
        
        return commands
    
    def update_difficulty(self, difficulty: float):
        """Update policy difficulty"""
        if hasattr(self.policy, 'update_difficulty'):
            self.policy.update_difficulty(difficulty)
        
        # Update config
        self.config["difficulty"]["base_difficulty"] = difficulty
        
        # Adjust spawn rates
        rate_multiplier = difficulty
        self.config["spawn_rates"]["common_base"] *= rate_multiplier
        self.config["spawn_rates"]["special_base"] *= rate_multiplier
