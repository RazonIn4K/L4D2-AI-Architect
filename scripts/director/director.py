#!/usr/bin/env python3
"""
L4D2 AI Director

Manages game difficulty, spawn rates, and events to create
dynamic gameplay experiences. Can operate in rule-based mode
or learn from RL training.
"""

import sys
import time
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import IntEnum
import threading
from pathlib import Path

# Add parent to path for security utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_read_json

try:
    from .bridge import GameBridge
    from .policy import DirectorPolicy
except ImportError:
    # For standalone testing
    from bridge import GameBridge
    from policy import DirectorPolicy

PROJECT_ROOT = Path(__file__).parent.parent.parent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DirectorMode(IntEnum):
    RULE_BASED = 0
    RL_BASED = 1
    HYBRID = 2


@dataclass
class GameState:
    """Current game state snapshot"""
    game_time: float
    round_time: float
    survivors: List[Dict[str, Any]]
    common_infected: int
    special_infected: List[int]  # [smoker, boomer, hunter, spitter, jockey]
    witch_count: int
    tank_count: int
    flow_progress: float  # 0-1, how far through map
    stress_level: float  # 0-1, calculated from various factors
    items_available: int
    health_packs_used: int
    recent_deaths: int
    panic_active: bool
    tank_active: bool


@dataclass
class DirectorCommand:
    """Command to send to the game"""
    command_type: str
    parameters: Dict[str, Any]
    priority: int  # Higher = more urgent
    delay: float  # Seconds before executing


class L4D2Director:
    """Main AI Director class"""
    
    def __init__(self, 
                 mode: DirectorMode = DirectorMode.RULE_BASED,
                 config_path: Optional[str] = None,
                 bridge_host: str = "localhost",
                 bridge_port: int = 27050):
        self.mode = mode
        self.bridge = GameBridge(bridge_host, bridge_port)
        self.policy = DirectorPolicy(mode, config_path)
        
        # Director state
        self.is_running = False
        self.last_update = time.time()
        self.update_rate = 10.0  # Hz
        self.command_queue: List[DirectorCommand] = []
        
        # Metrics tracking
        self.metrics = {
            "spawned_common": 0,
            "spawned_special": 0,
            "spawned_witches": 0,
            "spawned_tanks": 0,
            "triggered_panics": 0,
            "items_spawned": 0,
            "avg_stress": 0.0,
            "survivor_deaths": 0
        }
        
        # Load configuration
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load director configuration"""
        default_config = {
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
                "adaptive_scaling": True,
                "player_performance_window": 300,  # seconds
                "death_penalty": 0.1,
                "speed_bonus": 0.05
            }
        }
        
        if config_path:
            try:
                # Use safe_read_json which validates path and reads in one operation
                user_config = safe_read_json(config_path, PROJECT_ROOT)
                # Merge with defaults
                default_config.update(user_config)
            except FileNotFoundError:
                logger.info(f"Config file not found: {config_path}, using defaults")
            except ValueError as e:
                logger.warning(f"Invalid config path: {e}")

        return default_config
    
    def start(self):
        """Start the director loop"""
        if self.is_running:
            logger.warning("Director is already running")
            return
        
        self.is_running = True
        self.bridge.connect()
        
        # Start director thread
        self.director_thread = threading.Thread(target=self._director_loop, daemon=True)
        self.director_thread.start()
        
        logger.info("AI Director started")
    
    def stop(self):
        """Stop the director"""
        self.is_running = False
        self.bridge.disconnect()
        logger.info("AI Director stopped")
    
    def _director_loop(self):
        """Main director update loop"""
        while self.is_running:
            try:
                # Get current game state
                game_state = self.bridge.get_game_state()
                if not game_state:
                    time.sleep(0.1)
                    continue
                
                # Parse and analyze state
                state = self._parse_game_state(game_state)
                
                # Update metrics
                self._update_metrics(state)
                
                # Make decisions
                commands = self.policy.decide(state, self.metrics)
                
                # Queue commands
                for cmd in commands:
                    self._queue_command(cmd)
                
                # Execute queued commands
                self._execute_commands()
                
                # Sleep to maintain update rate
                elapsed = time.time() - self.last_update
                sleep_time = max(0, (1.0 / self.update_rate) - elapsed)
                time.sleep(sleep_time)
                self.last_update = time.time()
                
            except Exception as e:
                logger.error(f"Error in director loop: {e}")
                time.sleep(1.0)
    
    def _parse_game_state(self, raw_state: Dict) -> GameState:
        """Parse raw game state from bridge"""
        survivors = raw_state.get("survivors", [])
        
        # Calculate stress level
        stress = self._calculate_stress(survivors, raw_state)
        
        # Estimate flow progress (simplified)
        flow = self._estimate_flow_progress(survivors)
        
        return GameState(
            game_time=raw_state.get("gameTime", 0.0),
            round_time=raw_state.get("roundTime", 0.0),
            survivors=survivors,
            common_infected=raw_state.get("commonInfected", 0),
            special_infected=raw_state.get("specialInfected", [0, 0, 0, 0, 0]),
            witch_count=raw_state.get("witchCount", 0),
            tank_count=raw_state.get("tankCount", 0),
            flow_progress=flow,
            stress_level=stress,
            items_available=raw_state.get("itemsAvailable", 0),
            health_packs_used=raw_state.get("healthPacksUsed", 0),
            recent_deaths=raw_state.get("recentDeaths", 0),
            panic_active=raw_state.get("panicActive", False),
            tank_active=raw_state.get("tankActive", False)
        )
    
    def _calculate_stress(self, survivors: List[Dict], state: Dict) -> float:
        """Calculate team stress level (0-1)"""
        if not survivors:
            return 1.0
        
        stress = 0.0
        factors = self.config["stress_factors"]
        
        # Low health stress
        avg_health = np.mean([s.get("health", 0) + s.get("tempHealth", 0) for s in survivors])
        if avg_health < 50:
            stress += factors["low_health"] * (1.0 - avg_health / 50.0)
        
        # Horde stress
        if state.get("panicActive", False):
            stress += factors["horde_active"]
        elif state.get("commonInfected", 0) > 20:
            stress += factors["horde_active"] * 0.5
        
        # Special infected stress
        special_count = sum(state.get("specialInfected", [0, 0, 0, 0, 0]))
        if special_count > 0:
            stress += factors["special_active"] * min(1.0, special_count / 5.0)
        
        # Tank stress
        if state.get("tankActive", False):
            stress += 0.3  # Fixed high stress for tank
        
        # Item shortage stress
        items_per_survivor = state.get("itemsAvailable", 0) / len(survivors)
        if items_per_survivor < 1:
            stress += factors["item_shortage"] * (1.0 - items_per_survivor)
        
        return min(1.0, stress)
    
    def _estimate_flow_progress(self, survivors: List[Dict]) -> float:
        """Estimate how far through the map survivors are"""
        # This is a simplified version
        # In practice, you'd use nav mesh or trigger positions
        if not survivors:
            return 0.0
        
        # Use average distance from start (simplified)
        avg_x = np.mean([s.get("position", [0, 0, 0])[0] for s in survivors])
        # Map this to 0-1 range based on typical map sizes
        progress = min(1.0, max(0.0, avg_x / 10000.0))
        return progress
    
    def _update_metrics(self, state: GameState):
        """Update internal metrics"""
        # Update running averages
        self.metrics["avg_stress"] = (
            self.metrics["avg_stress"] * 0.9 + state.stress_level * 0.1
        )
        
        # Track deaths
        self.metrics["survivor_deaths"] = state.recent_deaths
    
    def _queue_command(self, command: DirectorCommand):
        """Add a command to the execution queue"""
        self.command_queue.append(command)
        # Sort by priority
        self.command_queue.sort(key=lambda x: x.priority, reverse=True)
    
    def _execute_commands(self):
        """Execute queued commands"""
        current_time = time.time()
        commands_to_execute = []
        
        # Check which commands are ready
        remaining_commands = []
        for cmd in self.command_queue:
            if current_time >= cmd.delay:
                commands_to_execute.append(cmd)
            else:
                remaining_commands.append(cmd)
        
        self.command_queue = remaining_commands
        
        # Execute commands
        for cmd in commands_to_execute:
            try:
                self.bridge.send_director_command(cmd.command_type, cmd.parameters)
                
                # Update metrics
                if cmd.command_type == "spawn_common":
                    self.metrics["spawned_common"] += cmd.parameters.get("count", 1)
                elif cmd.command_type == "spawn_special":
                    self.metrics["spawned_special"] += 1
                elif cmd.command_type == "spawn_witch":
                    self.metrics["spawned_witches"] += 1
                elif cmd.command_type == "spawn_tank":
                    self.metrics["spawned_tanks"] += 1
                elif cmd.command_type == "trigger_panic":
                    self.metrics["triggered_panics"] += 1
                elif cmd.command_type == "spawn_item":
                    self.metrics["items_spawned"] += 1
                    
            except Exception as e:
                logger.error(f"Failed to execute command {cmd.command_type}: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current director metrics"""
        return self.metrics.copy()
    
    def set_difficulty(self, difficulty: float):
        """Adjust overall difficulty (0.5 to 2.0)"""
        difficulty = max(0.5, min(2.0, difficulty))
        self.config["base_difficulty"] = difficulty
        self.policy.update_difficulty(difficulty)
        logger.info(f"Difficulty set to {difficulty}")


def main():
    """Standalone director server"""
    import argparse
    import logging
    
    parser = argparse.ArgumentParser(description="L4D2 AI Director")
    parser.add_argument("--mode", choices=["rule", "rl", "hybrid"], 
                       default="rule", help="Director mode")
    parser.add_argument("--config", type=str, 
                       help="Path to config file")
    parser.add_argument("--host", type=str, default="localhost",
                       help="Bridge host")
    parser.add_argument("--port", type=int, default=27050,
                       help="Bridge port")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create director
    mode_map = {"rule": DirectorMode.RULE_BASED, 
                "rl": DirectorMode.RL_BASED,
                "hybrid": DirectorMode.HYBRID}
    
    director = L4D2Director(
        mode=mode_map[args.mode],
        config_path=args.config,
        bridge_host=args.host,
        bridge_port=args.port
    )
    
    try:
        director.start()
        
        # Keep running
        while True:
            time.sleep(1)
            
            # Print metrics every 30 seconds
            if int(time.time()) % 30 == 0:
                metrics = director.get_metrics()
                logger.info(f"Metrics: {metrics}")
                
    except KeyboardInterrupt:
        logger.info("Shutting down director...")
        director.stop()


if __name__ == "__main__":
    main()
