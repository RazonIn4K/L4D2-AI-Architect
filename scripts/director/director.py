#!/usr/bin/env python3
"""
L4D2 AI Director

Manages game difficulty, spawn rates, and events to create
dynamic gameplay experiences. Can operate in rule-based mode
or learn from RL training.

Features:
- Rule-based, RL-based, and hybrid decision modes
- Simulation mode for testing without a live game server
- Comprehensive decision logging and replay support
- Detailed statistics tracking with per-minute metrics
"""

import sys
import time
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import IntEnum
import threading
from pathlib import Path
from datetime import datetime
from collections import deque

# Add parent to path for security utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_read_json, safe_write_json, safe_write_jsonl, safe_read_yaml

try:
    from .bridge import GameBridge
    from .policy import DirectorPolicy
    from .simulation import SimulationBridge, SimulationConfig, Scenario, DecisionReplay
except ImportError:
    # For standalone testing
    from bridge import GameBridge
    from policy import DirectorPolicy
    from simulation import SimulationBridge, SimulationConfig, Scenario, DecisionReplay

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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


@dataclass
class DecisionLogEntry:
    """Detailed log entry for a director decision"""
    timestamp: float
    game_time: float
    command_type: str
    parameters: Dict[str, Any]
    reason: str
    state_snapshot: Dict[str, Any]
    metrics_snapshot: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "game_time": self.game_time,
            "command_type": self.command_type,
            "parameters": self.parameters,
            "reason": self.reason,
            "state": self.state_snapshot,
            "metrics": self.metrics_snapshot
        }


@dataclass
class DirectorStatistics:
    """Comprehensive statistics tracking"""
    # Spawn counts
    spawned_common: int = 0
    spawned_special: int = 0
    spawned_witches: int = 0
    spawned_tanks: int = 0
    spawned_items: int = 0
    triggered_panics: int = 0

    # Time-based tracking
    session_start: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    total_decisions: int = 0

    # Per-minute tracking (rolling window)
    spawns_per_minute_window: deque = field(default_factory=lambda: deque(maxlen=60))
    events_per_minute_window: deque = field(default_factory=lambda: deque(maxlen=60))

    # Stress tracking
    stress_samples: deque = field(default_factory=lambda: deque(maxlen=100))
    avg_stress: float = 0.0
    max_stress: float = 0.0
    min_stress: float = 1.0

    # Survivor tracking
    survivor_deaths: int = 0
    avg_survivor_health: float = 100.0
    flow_progress_samples: deque = field(default_factory=lambda: deque(maxlen=100))

    # Command type breakdown
    command_counts: Dict[str, int] = field(default_factory=dict)

    def record_spawn(self, spawn_type: str, count: int = 1):
        """Record a spawn event"""
        self.spawns_per_minute_window.append((time.time(), spawn_type, count))

        if spawn_type == "common":
            self.spawned_common += count
        elif spawn_type == "special":
            self.spawned_special += count
        elif spawn_type == "witch":
            self.spawned_witches += count
        elif spawn_type == "tank":
            self.spawned_tanks += count
        elif spawn_type == "item":
            self.spawned_items += count

    def record_event(self, event_type: str):
        """Record an event"""
        self.events_per_minute_window.append((time.time(), event_type))
        if event_type == "panic":
            self.triggered_panics += 1

    def record_decision(self, command_type: str):
        """Record a decision"""
        self.total_decisions += 1
        self.command_counts[command_type] = self.command_counts.get(command_type, 0) + 1

    def record_stress(self, stress: float):
        """Record stress level"""
        self.stress_samples.append(stress)
        self.avg_stress = np.mean(list(self.stress_samples))
        self.max_stress = max(self.max_stress, stress)
        self.min_stress = min(self.min_stress, stress)

    def record_flow_progress(self, progress: float):
        """Record flow progress"""
        self.flow_progress_samples.append(progress)

    def get_spawns_per_minute(self) -> Dict[str, float]:
        """Calculate spawns per minute by type"""
        now = time.time()
        one_minute_ago = now - 60

        counts = {"common": 0, "special": 0, "witch": 0, "tank": 0, "item": 0, "total": 0}
        for timestamp, spawn_type, count in self.spawns_per_minute_window:
            if timestamp >= one_minute_ago:
                counts[spawn_type] = counts.get(spawn_type, 0) + count
                counts["total"] += count

        return counts

    def get_events_per_minute(self) -> int:
        """Calculate events per minute"""
        now = time.time()
        one_minute_ago = now - 60
        return sum(1 for ts, _ in self.events_per_minute_window if ts >= one_minute_ago)

    def get_session_duration(self) -> float:
        """Get session duration in seconds"""
        return time.time() - self.session_start

    def get_decisions_per_minute(self) -> float:
        """Calculate decisions per minute"""
        duration_minutes = self.get_session_duration() / 60.0
        if duration_minutes > 0:
            return self.total_decisions / duration_minutes
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "session_duration": self.get_session_duration(),
            "total_decisions": self.total_decisions,
            "decisions_per_minute": self.get_decisions_per_minute(),
            "spawns": {
                "common": self.spawned_common,
                "special": self.spawned_special,
                "witches": self.spawned_witches,
                "tanks": self.spawned_tanks,
                "items": self.spawned_items
            },
            "spawns_per_minute": self.get_spawns_per_minute(),
            "events": {
                "panics": self.triggered_panics,
                "per_minute": self.get_events_per_minute()
            },
            "stress": {
                "current": list(self.stress_samples)[-1] if self.stress_samples else 0,
                "average": self.avg_stress,
                "max": self.max_stress,
                "min": self.min_stress
            },
            "survivor_deaths": self.survivor_deaths,
            "avg_survivor_health": self.avg_survivor_health,
            "command_breakdown": self.command_counts.copy()
        }


class DecisionLogger:
    """
    Comprehensive logging for director decisions.

    Supports multiple output formats and detail levels.
    """

    def __init__(self, log_dir: Optional[str] = None, detail_level: str = "full"):
        self.log_dir = Path(log_dir) if log_dir else PROJECT_ROOT / "data" / "director_logs"
        self.detail_level = detail_level  # "minimal", "standard", "full"
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Decision log
        self.decisions: List[DecisionLogEntry] = []

        # File handles
        self._decision_log_path = self.log_dir / f"decisions_{self.session_id}.jsonl"
        self._summary_log_path = self.log_dir / f"summary_{self.session_id}.json"

        logger.info(f"Decision logger initialized: {self._decision_log_path}")

    def log_decision(self, command: DirectorCommand, state: GameState,
                    metrics: Dict[str, Any], reason: str = ""):
        """Log a director decision with full context"""
        entry = DecisionLogEntry(
            timestamp=time.time(),
            game_time=state.game_time,
            command_type=command.command_type,
            parameters=command.parameters.copy(),
            reason=reason,
            state_snapshot=self._create_state_snapshot(state),
            metrics_snapshot=metrics.copy()
        )

        self.decisions.append(entry)

        # Write to file immediately (append mode)
        with open(self._decision_log_path, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

        # Log based on detail level
        if self.detail_level == "full":
            logger.info(
                f"[DECISION] {command.command_type} | "
                f"Time: {state.game_time:.1f}s | "
                f"Stress: {state.stress_level:.2f} | "
                f"Flow: {state.flow_progress:.2f} | "
                f"Reason: {reason}"
            )
        elif self.detail_level == "standard":
            logger.info(f"[DECISION] {command.command_type} @ {state.game_time:.1f}s")
        # minimal: no per-decision logging

    def _create_state_snapshot(self, state: GameState) -> Dict[str, Any]:
        """Create a state snapshot for logging"""
        if self.detail_level == "minimal":
            return {
                "game_time": state.game_time,
                "stress_level": state.stress_level,
                "flow_progress": state.flow_progress
            }
        elif self.detail_level == "standard":
            return {
                "game_time": state.game_time,
                "stress_level": state.stress_level,
                "flow_progress": state.flow_progress,
                "common_infected": state.common_infected,
                "special_infected": sum(state.special_infected),
                "panic_active": state.panic_active,
                "tank_active": state.tank_active
            }
        else:  # full
            survivor_summary = []
            for s in state.survivors:
                survivor_summary.append({
                    "id": s.get("id"),
                    "health": s.get("health", 0) + s.get("tempHealth", 0),
                    "incapped": s.get("isIncapped", False),
                    "dead": s.get("isDead", False)
                })
            return {
                "game_time": state.game_time,
                "round_time": state.round_time,
                "stress_level": state.stress_level,
                "flow_progress": state.flow_progress,
                "survivors": survivor_summary,
                "common_infected": state.common_infected,
                "special_infected": state.special_infected,
                "witch_count": state.witch_count,
                "tank_count": state.tank_count,
                "items_available": state.items_available,
                "panic_active": state.panic_active,
                "tank_active": state.tank_active
            }

    def save_summary(self, statistics: DirectorStatistics):
        """Save session summary"""
        summary = {
            "session_id": self.session_id,
            "total_decisions": len(self.decisions),
            "statistics": statistics.to_dict(),
            "log_file": str(self._decision_log_path)
        }
        safe_write_json(str(self._summary_log_path), summary, PROJECT_ROOT)
        logger.info(f"Saved session summary to {self._summary_log_path}")

    def get_decisions_by_type(self, command_type: str) -> List[DecisionLogEntry]:
        """Get all decisions of a specific type"""
        return [d for d in self.decisions if d.command_type == command_type]

    def get_decisions_in_range(self, start_time: float, end_time: float) -> List[DecisionLogEntry]:
        """Get decisions within a time range"""
        return [d for d in self.decisions if start_time <= d.game_time <= end_time]


class L4D2Director:
    """Main AI Director class with simulation and logging support"""

    def __init__(self,
                 mode: DirectorMode = DirectorMode.RULE_BASED,
                 config_path: Optional[str] = None,
                 bridge_host: str = "localhost",
                 bridge_port: int = 27050,
                 simulation_mode: bool = False,
                 simulation_config: Optional[SimulationConfig] = None,
                 log_decisions: bool = True,
                 log_detail_level: str = "standard"):
        self.mode = mode
        self.simulation_mode = simulation_mode

        # Create appropriate bridge
        if simulation_mode:
            self.bridge: Union[GameBridge, SimulationBridge] = SimulationBridge(
                simulation_config or SimulationConfig()
            )
            logger.info("Director initialized in SIMULATION mode")
        else:
            self.bridge = GameBridge(bridge_host, bridge_port)
            logger.info(f"Director initialized for game server at {bridge_host}:{bridge_port}")

        self.policy = DirectorPolicy(mode, config_path)

        # Director state
        self.is_running = False
        self.last_update = time.time()
        self.update_rate = 10.0  # Hz
        self.command_queue: List[DirectorCommand] = []

        # Statistics tracking
        self.statistics = DirectorStatistics()

        # Decision logging
        self.decision_logger: Optional[DecisionLogger] = None
        if log_decisions:
            self.decision_logger = DecisionLogger(detail_level=log_detail_level)

        # Legacy metrics (for backward compatibility)
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

        # Replay support
        self._replay_mode = False
        self._replay: Optional[DecisionReplay] = None

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
                config_file = Path(config_path)
                if config_file.suffix in [".yaml", ".yml"]:
                    user_config = safe_read_yaml(config_path, PROJECT_ROOT)
                else:
                    user_config = safe_read_json(config_path, PROJECT_ROOT)
                # Merge with defaults
                default_config.update(user_config)
                logger.info(f"Loaded config from {config_path}")
            except FileNotFoundError:
                logger.info(f"Config file not found: {config_path}, using defaults")
            except ValueError as e:
                logger.warning(f"Invalid config path: {e}")

        return default_config

    def load_scenario(self, scenario: Scenario):
        """Load a scenario for simulation mode"""
        if not self.simulation_mode:
            logger.warning("Scenarios only work in simulation mode")
            return

        if isinstance(self.bridge, SimulationBridge):
            self.bridge.load_scenario(scenario)
            logger.info(f"Loaded scenario: {scenario.name}")

    def start(self):
        """Start the director loop"""
        if self.is_running:
            logger.warning("Director is already running")
            return

        self.is_running = True
        self.bridge.connect()

        # Reset statistics
        self.statistics = DirectorStatistics()

        # Start director thread
        self.director_thread = threading.Thread(target=self._director_loop, daemon=True)
        self.director_thread.start()

        logger.info("AI Director started")

    def stop(self):
        """Stop the director"""
        self.is_running = False
        self.bridge.disconnect()

        # Save logs
        if self.decision_logger:
            self.decision_logger.save_summary(self.statistics)

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

                # Update metrics and statistics
                self._update_metrics(state)
                self._update_statistics(state)

                # Make decisions (or replay)
                if self._replay_mode and self._replay:
                    commands = self._get_replay_commands(state)
                else:
                    # Convert GameState to dict for policy (policy expects dict format)
                    state_dict = self._state_to_dict(state)
                    commands = self.policy.decide(state_dict, self.metrics)

                # Queue commands
                for cmd in commands:
                    self._queue_command(cmd)

                # Execute queued commands
                self._execute_commands(state)

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

    def _state_to_dict(self, state: GameState) -> Dict[str, Any]:
        """Convert GameState dataclass to dict for policy compatibility"""
        return {
            "game_time": state.game_time,
            "round_time": state.round_time,
            "survivors": state.survivors,
            "common_infected": state.common_infected,
            "special_infected": state.special_infected,
            "witch_count": state.witch_count,
            "tank_count": state.tank_count,
            "flow_progress": state.flow_progress,
            "stress_level": state.stress_level,
            "items_available": state.items_available,
            "health_packs_used": state.health_packs_used,
            "recent_deaths": state.recent_deaths,
            "panic_active": state.panic_active,
            "tank_active": state.tank_active
        }

    def _update_metrics(self, state: GameState):
        """Update internal metrics (legacy)"""
        # Update running averages
        self.metrics["avg_stress"] = (
            self.metrics["avg_stress"] * 0.9 + state.stress_level * 0.1
        )

        # Track deaths
        self.metrics["survivor_deaths"] = state.recent_deaths

    def _update_statistics(self, state: GameState):
        """Update comprehensive statistics"""
        self.statistics.record_stress(state.stress_level)
        self.statistics.record_flow_progress(state.flow_progress)
        self.statistics.survivor_deaths = state.recent_deaths

        # Calculate average survivor health
        if state.survivors:
            self.statistics.avg_survivor_health = np.mean([
                s.get("health", 0) + s.get("tempHealth", 0)
                for s in state.survivors
            ])

    def _queue_command(self, command: Dict[str, Any]):
        """Add a command to the execution queue"""
        cmd = DirectorCommand(
            command_type=command.get("command_type", ""),
            parameters=command.get("parameters", {}),
            priority=command.get("priority", 1),
            delay=command.get("delay", 0.0)
        )
        self.command_queue.append(cmd)
        # Sort by priority
        self.command_queue.sort(key=lambda x: x.priority, reverse=True)

    def _execute_commands(self, state: GameState):
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

                # Update legacy metrics
                if cmd.command_type == "spawn_common":
                    count = cmd.parameters.get("count", 1)
                    self.metrics["spawned_common"] += count
                    self.statistics.record_spawn("common", count)
                elif cmd.command_type == "spawn_special":
                    self.metrics["spawned_special"] += 1
                    self.statistics.record_spawn("special")
                elif cmd.command_type == "spawn_witch":
                    self.metrics["spawned_witches"] += 1
                    self.statistics.record_spawn("witch")
                elif cmd.command_type == "spawn_tank":
                    self.metrics["spawned_tanks"] += 1
                    self.statistics.record_spawn("tank")
                elif cmd.command_type == "trigger_panic":
                    self.metrics["triggered_panics"] += 1
                    self.statistics.record_event("panic")
                elif cmd.command_type == "spawn_item":
                    self.metrics["items_spawned"] += 1
                    self.statistics.record_spawn("item")

                # Record decision
                self.statistics.record_decision(cmd.command_type)

                # Log decision
                if self.decision_logger:
                    reason = cmd.parameters.get("_reason", "policy decision")
                    self.decision_logger.log_decision(cmd, state, self.metrics, reason)

            except Exception as e:
                logger.error(f"Failed to execute command {cmd.command_type}: {e}")

    def _get_replay_commands(self, state: GameState) -> List[Dict[str, Any]]:
        """Get commands from replay"""
        if not self._replay:
            return []

        # Get decisions near current game time
        decisions = self._replay.get_decisions_at_time(state.game_time, tolerance=0.5)

        commands = []
        for decision in decisions:
            commands.append({
                "command_type": decision.get("command_type"),
                "parameters": decision.get("parameters", {}),
                "priority": 5,
                "delay": 0.0
            })

        return commands

    def enable_replay(self, log_path: str):
        """Enable replay mode from a log file"""
        self._replay = DecisionReplay(log_path)
        self._replay_mode = True
        logger.info(f"Replay mode enabled from {log_path}")

    def disable_replay(self):
        """Disable replay mode"""
        self._replay_mode = False
        self._replay = None
        logger.info("Replay mode disabled")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current director metrics (legacy)"""
        return self.metrics.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return self.statistics.to_dict()

    def get_spawns_per_minute(self) -> Dict[str, float]:
        """Get spawns per minute breakdown"""
        return self.statistics.get_spawns_per_minute()

    def get_events_per_minute(self) -> int:
        """Get events per minute"""
        return self.statistics.get_events_per_minute()

    def get_decisions_per_minute(self) -> float:
        """Get decisions per minute"""
        return self.statistics.get_decisions_per_minute()

    def set_difficulty(self, difficulty: float):
        """Adjust overall difficulty (0.5 to 2.0)"""
        difficulty = max(0.5, min(2.0, difficulty))
        self.config["difficulty"]["base_difficulty"] = difficulty
        self.policy.update_difficulty(difficulty)
        logger.info(f"Difficulty set to {difficulty}")

    def print_statistics_report(self):
        """Print a formatted statistics report"""
        stats = self.get_statistics()

        print("\n" + "=" * 50)
        print("AI DIRECTOR STATISTICS REPORT")
        print("=" * 50)

        print(f"\nSession Duration: {stats['session_duration']:.1f}s "
              f"({stats['session_duration']/60:.1f} minutes)")
        print(f"Total Decisions: {stats['total_decisions']}")
        print(f"Decisions/Minute: {stats['decisions_per_minute']:.2f}")

        print("\n--- Spawns ---")
        spawns = stats["spawns"]
        print(f"  Common Infected: {spawns['common']}")
        print(f"  Special Infected: {spawns['special']}")
        print(f"  Witches: {spawns['witches']}")
        print(f"  Tanks: {spawns['tanks']}")
        print(f"  Items: {spawns['items']}")

        print("\n--- Spawns/Minute (Last 60s) ---")
        spm = stats["spawns_per_minute"]
        print(f"  Common: {spm.get('common', 0)}")
        print(f"  Special: {spm.get('special', 0)}")
        print(f"  Total: {spm.get('total', 0)}")

        print("\n--- Events ---")
        print(f"  Panic Events: {stats['events']['panics']}")
        print(f"  Events/Minute: {stats['events']['per_minute']}")

        print("\n--- Stress Analysis ---")
        stress = stats["stress"]
        print(f"  Current: {stress['current']:.2f}")
        print(f"  Average: {stress['average']:.2f}")
        print(f"  Range: {stress['min']:.2f} - {stress['max']:.2f}")

        print("\n--- Survivors ---")
        print(f"  Deaths: {stats['survivor_deaths']}")
        print(f"  Avg Health: {stats['avg_survivor_health']:.1f}")

        print("\n--- Command Breakdown ---")
        for cmd_type, count in sorted(stats["command_breakdown"].items()):
            print(f"  {cmd_type}: {count}")

        print("=" * 50 + "\n")


def main():
    """Standalone director server with simulation support"""
    import argparse

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

    # Simulation options
    parser.add_argument("--simulate", action="store_true",
                       help="Run in simulation mode (no game server required)")
    parser.add_argument("--scenario", choices=["default", "stress", "easy", "finale"],
                       default="default", help="Simulation scenario")
    parser.add_argument("--duration", type=float, default=300.0,
                       help="Simulation duration in seconds")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")

    # Logging options
    parser.add_argument("--log-decisions", action="store_true", default=True,
                       help="Enable decision logging")
    parser.add_argument("--log-detail", choices=["minimal", "standard", "full"],
                       default="standard", help="Decision log detail level")

    # Replay options
    parser.add_argument("--replay", type=str,
                       help="Replay decisions from log file")

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Create director
    mode_map = {"rule": DirectorMode.RULE_BASED,
                "rl": DirectorMode.RL_BASED,
                "hybrid": DirectorMode.HYBRID}

    # Simulation config
    sim_config = None
    if args.simulate:
        sim_config = SimulationConfig(
            max_duration=args.duration,
            seed=args.seed
        )

    director = L4D2Director(
        mode=mode_map[args.mode],
        config_path=args.config,
        bridge_host=args.host,
        bridge_port=args.port,
        simulation_mode=args.simulate,
        simulation_config=sim_config,
        log_decisions=args.log_decisions,
        log_detail_level=args.log_detail
    )

    # Load scenario if in simulation mode
    if args.simulate:
        scenario_map = {
            "default": Scenario.create_default,
            "stress": Scenario.create_stress_test,
            "easy": Scenario.create_easy,
            "finale": Scenario.create_finale
        }
        scenario = scenario_map[args.scenario]()
        director.load_scenario(scenario)

    # Enable replay if specified
    if args.replay:
        director.enable_replay(args.replay)

    try:
        director.start()

        # Keep running
        last_report_time = time.time()
        report_interval = 30.0  # Print stats every 30 seconds

        while True:
            time.sleep(1)

            # Check if simulation ended
            if args.simulate:
                if isinstance(director.bridge, SimulationBridge) and not director.bridge.running:
                    logger.info("Simulation ended")
                    break

            # Periodic statistics report
            if time.time() - last_report_time >= report_interval:
                stats = director.get_statistics()
                spm = director.get_spawns_per_minute()
                logger.info(
                    f"Stats | Decisions: {stats['total_decisions']} | "
                    f"Spawns/min: {spm.get('total', 0)} | "
                    f"Stress: {stats['stress']['average']:.2f}"
                )
                last_report_time = time.time()

    except KeyboardInterrupt:
        logger.info("Shutting down director...")

    finally:
        director.stop()
        director.print_statistics_report()


if __name__ == "__main__":
    main()
