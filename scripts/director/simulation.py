#!/usr/bin/env python3
"""
Simulation Module for AI Director Testing

Provides a simulation environment that mimics game behavior without
requiring a live L4D2 server. Useful for:
- Testing director logic
- Replaying decisions from logs
- Running automated tests
- Benchmarking policy performance
"""

import sys
import time
import json
import logging
import random
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from copy import deepcopy

# Add parent to path for security utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_read_json, safe_write_json, safe_write_jsonl

PROJECT_ROOT = Path(__file__).parent.parent.parent

logger = logging.getLogger(__name__)


class SimulationEvent(Enum):
    """Events that can occur in simulation"""
    SURVIVOR_DAMAGE = "survivor_damage"
    SURVIVOR_HEAL = "survivor_heal"
    SURVIVOR_DEATH = "survivor_death"
    SURVIVOR_REVIVE = "survivor_revive"
    SURVIVOR_MOVE = "survivor_move"
    COMMON_KILLED = "common_killed"
    SPECIAL_KILLED = "special_killed"
    WITCH_KILLED = "witch_killed"
    TANK_KILLED = "tank_killed"
    ITEM_USED = "item_used"
    PANIC_END = "panic_end"
    ROUND_END = "round_end"


@dataclass
class SimulationConfig:
    """Configuration for simulation behavior"""
    # Time settings
    tick_rate: float = 10.0  # Updates per second
    time_scale: float = 1.0  # Speed multiplier (2.0 = 2x speed)
    max_duration: float = 1800.0  # Max simulation time (30 minutes)

    # Survivor settings
    num_survivors: int = 4
    survivor_move_speed: float = 220.0  # units per second
    survivor_max_health: int = 100

    # Map settings
    map_length: float = 10000.0  # Total map length in units
    checkpoint_positions: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])

    # Combat simulation
    common_damage_rate: float = 0.1  # Damage per common per second
    special_damage_rate: float = 2.0  # Damage per special per second
    tank_damage_rate: float = 10.0  # Tank damage per second
    survivor_kill_rate: float = 5.0  # Commons killed per second per survivor

    # Random events
    random_damage_chance: float = 0.02  # Chance per tick of random damage
    item_find_chance: float = 0.01  # Chance per tick of finding items

    # Difficulty response
    death_morale_penalty: float = 0.2  # Speed reduction per death

    # Seed for reproducibility (None = random)
    seed: Optional[int] = None


@dataclass
class SurvivorState:
    """State of a single survivor"""
    id: int
    name: str
    health: int = 100
    temp_health: int = 0
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    angle: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    weapon: str = "pistol"
    is_incapped: bool = False
    is_dead: bool = False
    items: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "health": self.health,
            "tempHealth": self.temp_health,
            "position": self.position.copy(),
            "angle": self.angle.copy(),
            "weapon": self.weapon,
            "isIncapped": self.is_incapped,
            "isDead": self.is_dead,
            "items": self.items.copy()
        }


@dataclass
class SimulationState:
    """Complete simulation state"""
    # Time
    game_time: float = 0.0
    round_time: float = 0.0
    tick_count: int = 0

    # Survivors
    survivors: List[SurvivorState] = field(default_factory=list)

    # Enemies
    common_infected: int = 0
    special_infected: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 0])  # smoker, boomer, hunter, spitter, jockey
    witch_count: int = 0
    tank_count: int = 0

    # Items and environment
    items_available: int = 0
    health_packs_used: int = 0

    # Events
    panic_active: bool = False
    panic_end_time: float = 0.0
    tank_active: bool = False

    # Statistics
    recent_deaths: int = 0
    total_kills: int = 0
    total_damage_dealt: int = 0
    total_damage_taken: int = 0

    def to_bridge_format(self) -> Dict[str, Any]:
        """Convert to format expected by GameBridge"""
        return {
            "type": "game_state",
            "gameTime": self.game_time,
            "roundTime": self.round_time,
            "survivors": [s.to_dict() for s in self.survivors],
            "commonInfected": self.common_infected,
            "specialInfected": self.special_infected.copy(),
            "witchCount": self.witch_count,
            "tankCount": self.tank_count,
            "itemsAvailable": self.items_available,
            "healthPacksUsed": self.health_packs_used,
            "recentDeaths": self.recent_deaths,
            "panicActive": self.panic_active,
            "tankActive": self.tank_active
        }


class Scenario:
    """Predefined scenario for testing"""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.initial_state: Optional[SimulationState] = None
        self.events: List[Dict[str, Any]] = []  # Timed events
        self.success_conditions: List[Callable[[SimulationState], bool]] = []
        self.failure_conditions: List[Callable[[SimulationState], bool]] = []

    @classmethod
    def create_default(cls) -> "Scenario":
        """Create default scenario with standard settings"""
        scenario = cls("default", "Standard playthrough")
        scenario.initial_state = SimulationState()
        return scenario

    @classmethod
    def create_stress_test(cls) -> "Scenario":
        """Create high-stress scenario for testing director response"""
        scenario = cls("stress_test", "High stress scenario with low health team")

        # Start with damaged survivors
        state = SimulationState()
        survivors = [
            SurvivorState(id=1, name="Coach", health=30),
            SurvivorState(id=2, name="Ellis", health=45),
            SurvivorState(id=3, name="Nick", health=20, is_incapped=True),
            SurvivorState(id=4, name="Rochelle", health=50)
        ]
        state.survivors = survivors
        state.common_infected = 20
        state.special_infected = [1, 1, 1, 0, 1]  # 4 specials
        state.items_available = 1
        state.recent_deaths = 1

        scenario.initial_state = state

        # Add timed events (more pressure over time)
        scenario.events = [
            {"time": 30.0, "type": "spawn_special", "params": {"type": "hunter"}},
            {"time": 60.0, "type": "trigger_panic", "params": {}},
            {"time": 120.0, "type": "spawn_tank", "params": {}},
        ]

        return scenario

    @classmethod
    def create_easy(cls) -> "Scenario":
        """Create easy scenario for baseline testing"""
        scenario = cls("easy", "Easy scenario with healthy team")

        state = SimulationState()
        survivors = [
            SurvivorState(id=1, name="Coach", health=100, items=["medkit"]),
            SurvivorState(id=2, name="Ellis", health=100, items=["pills"]),
            SurvivorState(id=3, name="Nick", health=100, items=["molotov"]),
            SurvivorState(id=4, name="Rochelle", health=100, items=["pipe_bomb"])
        ]
        state.survivors = survivors
        state.items_available = 6

        scenario.initial_state = state
        return scenario

    @classmethod
    def create_finale(cls) -> "Scenario":
        """Create finale scenario with escalating difficulty"""
        scenario = cls("finale", "Finale scenario with escalating waves")

        state = SimulationState()
        survivors = [
            SurvivorState(id=i+1, name=name, health=80, position=[8000.0, 0.0, 0.0])
            for i, name in enumerate(["Coach", "Ellis", "Nick", "Rochelle"])
        ]
        state.survivors = survivors
        state.common_infected = 10
        state.items_available = 4

        scenario.initial_state = state

        # Finale wave events
        scenario.events = [
            {"time": 0.0, "type": "trigger_panic", "params": {}},
            {"time": 30.0, "type": "spawn_common", "params": {"count": 20}},
            {"time": 60.0, "type": "spawn_tank", "params": {}},
            {"time": 120.0, "type": "trigger_panic", "params": {}},
            {"time": 150.0, "type": "spawn_common", "params": {"count": 30}},
            {"time": 180.0, "type": "spawn_tank", "params": {}},
        ]

        return scenario

    @classmethod
    def load_from_file(cls, path: str) -> "Scenario":
        """Load scenario from JSON file"""
        data = safe_read_json(path, PROJECT_ROOT)
        scenario = cls(data.get("name", "custom"), data.get("description", ""))

        # Parse initial state
        if "initial_state" in data:
            state_data = data["initial_state"]
            state = SimulationState()

            # Parse survivors
            if "survivors" in state_data:
                state.survivors = [
                    SurvivorState(
                        id=s.get("id", i+1),
                        name=s.get("name", f"Survivor{i+1}"),
                        health=s.get("health", 100),
                        temp_health=s.get("tempHealth", 0),
                        position=s.get("position", [0.0, 0.0, 0.0]),
                        items=s.get("items", [])
                    )
                    for i, s in enumerate(state_data["survivors"])
                ]

            # Parse other state
            state.common_infected = state_data.get("commonInfected", 0)
            state.special_infected = state_data.get("specialInfected", [0, 0, 0, 0, 0])
            state.witch_count = state_data.get("witchCount", 0)
            state.tank_count = state_data.get("tankCount", 0)
            state.items_available = state_data.get("itemsAvailable", 0)
            state.panic_active = state_data.get("panicActive", False)

            scenario.initial_state = state

        # Parse events
        scenario.events = data.get("events", [])

        return scenario


class SimulationBridge:
    """
    Bridge implementation that simulates game behavior.

    Drop-in replacement for GameBridge that doesn't require a server.
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.state = SimulationState()
        self.is_connected = False
        self.running = False

        # Callbacks
        self.state_callbacks: List[Callable] = []
        self.event_callbacks: List[Callable] = []

        # Command history for replay analysis
        self.command_history: List[Dict[str, Any]] = []

        # Scenario support
        self.scenario: Optional[Scenario] = None
        self.scenario_event_index = 0

        # Threading
        self.simulation_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Random state for reproducibility
        if self.config.seed is not None:
            random.seed(self.config.seed)

    def connect(self) -> bool:
        """Start the simulation"""
        if self.is_connected:
            return True

        self._initialize_state()
        self.is_connected = True
        self.running = True

        # Start simulation thread
        self.simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.simulation_thread.start()

        logger.info("Simulation started")
        return True

    def disconnect(self):
        """Stop the simulation"""
        self.running = False
        self.is_connected = False

        if self.simulation_thread:
            self.simulation_thread.join(timeout=2.0)

        logger.info("Simulation stopped")

    def load_scenario(self, scenario: Scenario):
        """Load a predefined scenario"""
        self.scenario = scenario
        self.scenario_event_index = 0

        if scenario.initial_state:
            with self._lock:
                self.state = deepcopy(scenario.initial_state)

        logger.info(f"Loaded scenario: {scenario.name}")

    def _initialize_state(self):
        """Initialize simulation state"""
        with self._lock:
            if self.scenario and self.scenario.initial_state:
                self.state = deepcopy(self.scenario.initial_state)
            else:
                self.state = SimulationState()
                # Create default survivors
                names = ["Coach", "Ellis", "Nick", "Rochelle"]
                self.state.survivors = [
                    SurvivorState(
                        id=i+1,
                        name=names[i],
                        health=100,
                        position=[0.0, i * 50.0, 0.0]
                    )
                    for i in range(self.config.num_survivors)
                ]

            self.state.game_time = 0.0
            self.state.round_time = 0.0
            self.state.tick_count = 0

    def _simulation_loop(self):
        """Main simulation loop"""
        last_tick = time.time()
        tick_interval = 1.0 / self.config.tick_rate

        while self.running:
            current_time = time.time()
            elapsed = current_time - last_tick

            if elapsed >= tick_interval:
                scaled_dt = elapsed * self.config.time_scale

                with self._lock:
                    self._update_simulation(scaled_dt)
                    self._process_scenario_events()
                    self.state.tick_count += 1

                # Notify callbacks
                state_dict = self.get_game_state()
                for callback in self.state_callbacks:
                    try:
                        callback(state_dict)
                    except Exception as e:
                        logger.error(f"Error in state callback: {e}")

                last_tick = current_time

                # Check max duration
                if self.state.game_time >= self.config.max_duration:
                    logger.info("Simulation reached max duration")
                    self._emit_event(SimulationEvent.ROUND_END)
                    self.running = False
            else:
                time.sleep(tick_interval - elapsed)

    def _update_simulation(self, dt: float):
        """Update simulation state"""
        self.state.game_time += dt
        self.state.round_time += dt

        # Update survivors
        self._update_survivors(dt)

        # Process combat
        self._process_combat(dt)

        # Update panic state
        if self.state.panic_active and self.state.game_time >= self.state.panic_end_time:
            self.state.panic_active = False
            self._emit_event(SimulationEvent.PANIC_END)

        # Random events
        self._process_random_events(dt)

        # Natural enemy decay (survivors killing enemies)
        self._process_enemy_decay(dt)

    def _update_survivors(self, dt: float):
        """Update survivor positions and states"""
        alive_survivors = [s for s in self.state.survivors if not s.is_dead]
        if not alive_survivors:
            return

        # Calculate movement speed (reduced by morale)
        morale_penalty = self.state.recent_deaths * self.config.death_morale_penalty
        speed = self.config.survivor_move_speed * (1.0 - min(0.5, morale_penalty))

        # Move survivors forward
        for survivor in alive_survivors:
            if not survivor.is_incapped:
                survivor.position[0] += speed * dt
                survivor.position[0] = min(survivor.position[0], self.config.map_length)

    def _process_combat(self, dt: float):
        """Process combat damage"""
        alive_survivors = [s for s in self.state.survivors if not s.is_dead and not s.is_incapped]
        if not alive_survivors:
            return

        # Common infected damage
        if self.state.common_infected > 0:
            damage = int(self.state.common_infected * self.config.common_damage_rate * dt)
            if damage > 0:
                victim = random.choice(alive_survivors)
                self._damage_survivor(victim, damage)

        # Special infected damage
        special_count = sum(self.state.special_infected)
        if special_count > 0:
            damage = int(special_count * self.config.special_damage_rate * dt)
            if damage > 0:
                victim = random.choice(alive_survivors)
                self._damage_survivor(victim, damage)

        # Tank damage
        if self.state.tank_count > 0:
            damage = int(self.state.tank_count * self.config.tank_damage_rate * dt)
            if damage > 0:
                victim = random.choice(alive_survivors)
                self._damage_survivor(victim, damage)

    def _damage_survivor(self, survivor: SurvivorState, damage: int):
        """Apply damage to a survivor"""
        self.state.total_damage_taken += damage

        # Apply to temp health first
        if survivor.temp_health > 0:
            temp_damage = min(damage, survivor.temp_health)
            survivor.temp_health -= temp_damage
            damage -= temp_damage

        # Then regular health
        survivor.health -= damage

        if survivor.health <= 0:
            survivor.health = 0
            if not survivor.is_incapped:
                survivor.is_incapped = True
                logger.debug(f"{survivor.name} incapacitated")
            else:
                survivor.is_dead = True
                self.state.recent_deaths += 1
                logger.debug(f"{survivor.name} died")
                self._emit_event(SimulationEvent.SURVIVOR_DEATH, {"survivor_id": survivor.id})

    def _process_enemy_decay(self, dt: float):
        """Survivors killing enemies"""
        alive_survivors = [s for s in self.state.survivors if not s.is_dead and not s.is_incapped]
        if not alive_survivors:
            return

        # Common infected kills
        kill_rate = len(alive_survivors) * self.config.survivor_kill_rate * dt
        killed = min(int(kill_rate), self.state.common_infected)
        if killed > 0:
            self.state.common_infected -= killed
            self.state.total_kills += killed

        # Special infected kills (slower)
        if random.random() < 0.1 * dt * len(alive_survivors):
            for i in range(len(self.state.special_infected)):
                if self.state.special_infected[i] > 0:
                    self.state.special_infected[i] -= 1
                    self.state.total_kills += 1
                    self._emit_event(SimulationEvent.SPECIAL_KILLED, {"type": i})
                    break

        # Tank damage (very slow)
        if self.state.tank_count > 0 and random.random() < 0.02 * dt * len(alive_survivors):
            self.state.tank_count -= 1
            self.state.tank_active = self.state.tank_count > 0
            self._emit_event(SimulationEvent.TANK_KILLED)

    def _process_random_events(self, dt: float):
        """Process random events"""
        if random.random() < self.config.random_damage_chance * dt:
            alive = [s for s in self.state.survivors if not s.is_dead]
            if alive:
                victim = random.choice(alive)
                damage = random.randint(5, 15)
                self._damage_survivor(victim, damage)

        if random.random() < self.config.item_find_chance * dt:
            self.state.items_available += 1

    def _process_scenario_events(self):
        """Process scenario-defined events"""
        if not self.scenario or not self.scenario.events:
            return

        while self.scenario_event_index < len(self.scenario.events):
            event = self.scenario.events[self.scenario_event_index]
            if event["time"] <= self.state.game_time:
                self._execute_scenario_event(event)
                self.scenario_event_index += 1
            else:
                break

    def _execute_scenario_event(self, event: Dict[str, Any]):
        """Execute a scenario event"""
        event_type = event.get("type", "")
        params = event.get("params", {})

        logger.debug(f"Executing scenario event: {event_type}")

        if event_type == "spawn_common":
            self.state.common_infected += params.get("count", 5)
        elif event_type == "spawn_special":
            special_type = params.get("type", "hunter")
            idx = ["smoker", "boomer", "hunter", "spitter", "jockey"].index(special_type)
            self.state.special_infected[idx] += 1
        elif event_type == "spawn_witch":
            self.state.witch_count += 1
        elif event_type == "spawn_tank":
            self.state.tank_count += 1
            self.state.tank_active = True
        elif event_type == "trigger_panic":
            self.state.panic_active = True
            self.state.panic_end_time = self.state.game_time + 30.0
            self.state.common_infected += 15

    def _emit_event(self, event: SimulationEvent, data: Optional[Dict] = None):
        """Emit an event to callbacks"""
        event_data = {
            "type": "event",
            "event_type": event.value,
            "time": self.state.game_time,
            "data": data or {}
        }
        for callback in self.event_callbacks:
            try:
                callback(event_data)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")

    def get_game_state(self) -> Optional[Dict[str, Any]]:
        """Get current game state in bridge format"""
        with self._lock:
            return self.state.to_bridge_format()

    def send_director_command(self, command_type: str, parameters: Dict[str, Any]):
        """Process a director command"""
        timestamp = time.time()

        # Record command
        command_record = {
            "timestamp": timestamp,
            "game_time": self.state.game_time,
            "command_type": command_type,
            "parameters": parameters
        }
        self.command_history.append(command_record)

        # Execute command in simulation
        with self._lock:
            self._execute_command(command_type, parameters)

        logger.debug(f"Executed command: {command_type} with {parameters}")

    def _execute_command(self, command_type: str, parameters: Dict[str, Any]):
        """Execute a director command in the simulation"""
        if command_type == "spawn_common":
            count = parameters.get("count", 5)
            self.state.common_infected += count

        elif command_type == "spawn_special":
            special_type = parameters.get("type", "hunter")
            type_map = {"smoker": 0, "boomer": 1, "hunter": 2, "spitter": 3, "jockey": 4}
            idx = type_map.get(special_type, 2)
            self.state.special_infected[idx] += 1

        elif command_type == "spawn_witch":
            self.state.witch_count += 1

        elif command_type == "spawn_tank":
            self.state.tank_count += 1
            self.state.tank_active = True

        elif command_type == "trigger_panic":
            self.state.panic_active = True
            self.state.panic_end_time = self.state.game_time + 30.0
            self.state.common_infected += 15

        elif command_type == "spawn_item":
            self.state.items_available += 1

    def send_bot_action(self, bot_id: int, action: str, **kwargs):
        """Process bot action (for compatibility)"""
        logger.debug(f"Bot action: {bot_id} -> {action}")

    def reset_episode(self):
        """Reset the simulation"""
        self._initialize_state()
        self.command_history.clear()
        self.scenario_event_index = 0
        logger.info("Simulation reset")

    def add_state_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for state updates"""
        self.state_callbacks.append(callback)

    def remove_state_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Remove state callback"""
        if callback in self.state_callbacks:
            self.state_callbacks.remove(callback)

    def add_event_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for events"""
        self.event_callbacks.append(callback)

    def get_command_history(self) -> List[Dict[str, Any]]:
        """Get history of all commands sent"""
        return self.command_history.copy()

    def export_command_history(self, path: str):
        """Export command history to file"""
        safe_write_jsonl(path, self.command_history, PROJECT_ROOT)
        logger.info(f"Exported {len(self.command_history)} commands to {path}")

    def get_simulation_stats(self) -> Dict[str, Any]:
        """Get simulation statistics"""
        with self._lock:
            avg_health = 0.0
            alive_count = 0
            for s in self.state.survivors:
                if not s.is_dead:
                    avg_health += s.health + s.temp_health
                    alive_count += 1
            if alive_count > 0:
                avg_health /= alive_count

            # Calculate flow progress
            if self.state.survivors:
                avg_x = sum(s.position[0] for s in self.state.survivors) / len(self.state.survivors)
                flow_progress = min(1.0, avg_x / self.config.map_length)
            else:
                flow_progress = 0.0

            return {
                "game_time": self.state.game_time,
                "tick_count": self.state.tick_count,
                "alive_survivors": alive_count,
                "avg_health": avg_health,
                "flow_progress": flow_progress,
                "total_kills": self.state.total_kills,
                "total_damage_taken": self.state.total_damage_taken,
                "commands_issued": len(self.command_history),
                "common_infected": self.state.common_infected,
                "special_infected": sum(self.state.special_infected),
                "panic_active": self.state.panic_active,
                "tank_active": self.state.tank_active
            }


class DecisionReplay:
    """
    Replay director decisions from a log file.

    Useful for analyzing past director behavior and debugging.
    """

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.decisions: List[Dict[str, Any]] = []
        self.current_index = 0
        self._load_log()

    def _load_log(self):
        """Load decisions from log file"""
        path = Path(log_path) if isinstance(log_path := self.log_path, str) else log_path

        if path.suffix == ".jsonl":
            with open(path, "r") as f:
                self.decisions = [json.loads(line) for line in f if line.strip()]
        else:
            data = safe_read_json(str(path), PROJECT_ROOT)
            self.decisions = data if isinstance(data, list) else data.get("decisions", [])

        # Sort by timestamp/game_time
        self.decisions.sort(key=lambda x: x.get("game_time", x.get("timestamp", 0)))

        logger.info(f"Loaded {len(self.decisions)} decisions from {self.log_path}")

    def reset(self):
        """Reset replay to beginning"""
        self.current_index = 0

    def get_decisions_at_time(self, game_time: float, tolerance: float = 0.5) -> List[Dict[str, Any]]:
        """Get all decisions at a specific game time"""
        return [
            d for d in self.decisions
            if abs(d.get("game_time", 0) - game_time) <= tolerance
        ]

    def get_next_decision(self) -> Optional[Dict[str, Any]]:
        """Get next decision in sequence"""
        if self.current_index >= len(self.decisions):
            return None

        decision = self.decisions[self.current_index]
        self.current_index += 1
        return decision

    def get_decisions_in_range(self, start_time: float, end_time: float) -> List[Dict[str, Any]]:
        """Get all decisions within a time range"""
        return [
            d for d in self.decisions
            if start_time <= d.get("game_time", 0) <= end_time
        ]

    def get_decision_summary(self) -> Dict[str, Any]:
        """Get summary of all decisions"""
        summary = {
            "total_decisions": len(self.decisions),
            "by_type": {},
            "time_range": (0, 0),
            "decisions_per_minute": 0
        }

        if not self.decisions:
            return summary

        # Count by type
        for decision in self.decisions:
            cmd_type = decision.get("command_type", "unknown")
            summary["by_type"][cmd_type] = summary["by_type"].get(cmd_type, 0) + 1

        # Time range
        times = [d.get("game_time", 0) for d in self.decisions]
        summary["time_range"] = (min(times), max(times))

        # Decisions per minute
        duration_minutes = (summary["time_range"][1] - summary["time_range"][0]) / 60.0
        if duration_minutes > 0:
            summary["decisions_per_minute"] = len(self.decisions) / duration_minutes

        return summary

    def replay_to_bridge(self, bridge: SimulationBridge, speed: float = 1.0):
        """Replay all decisions to a simulation bridge"""
        if not self.decisions:
            logger.warning("No decisions to replay")
            return

        self.reset()
        start_time = time.time()
        first_decision_time = self.decisions[0].get("game_time", 0)

        for decision in self.decisions:
            decision_time = decision.get("game_time", 0)

            # Wait for appropriate time
            if speed > 0:
                target_elapsed = (decision_time - first_decision_time) / speed
                actual_elapsed = time.time() - start_time
                if target_elapsed > actual_elapsed:
                    time.sleep(target_elapsed - actual_elapsed)

            # Send command
            cmd_type = decision.get("command_type")
            params = decision.get("parameters", {})
            if cmd_type:
                bridge.send_director_command(cmd_type, params)

        logger.info(f"Replayed {len(self.decisions)} decisions")


def main():
    """Test simulation"""
    import argparse

    parser = argparse.ArgumentParser(description="AI Director Simulation")
    parser.add_argument("--scenario", choices=["default", "stress", "easy", "finale"],
                       default="default", help="Scenario to run")
    parser.add_argument("--duration", type=float, default=60.0, help="Simulation duration")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--replay", type=str, help="Replay log file")

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    # Create configuration
    config = SimulationConfig(
        max_duration=args.duration,
        seed=args.seed
    )

    # Create bridge
    bridge = SimulationBridge(config)

    # Load scenario
    scenario_map = {
        "default": Scenario.create_default,
        "stress": Scenario.create_stress_test,
        "easy": Scenario.create_easy,
        "finale": Scenario.create_finale
    }
    scenario = scenario_map[args.scenario]()
    bridge.load_scenario(scenario)

    # State callback
    def on_state(state):
        if state.get("gameTime", 0) % 5 < 0.1:  # Every ~5 seconds
            survivors = state.get("survivors", [])
            alive = sum(1 for s in survivors if not s.get("isDead", False))
            print(f"Time: {state.get('gameTime', 0):.1f}s | "
                  f"Survivors: {alive}/4 | "
                  f"Common: {state.get('commonInfected', 0)} | "
                  f"Specials: {sum(state.get('specialInfected', []))}")

    bridge.add_state_callback(on_state)

    # Connect and run
    if bridge.connect():
        print(f"Running {args.scenario} scenario for {args.duration}s...")

        # Replay if specified
        if args.replay:
            replay = DecisionReplay(args.replay)
            print(f"Replaying decisions from {args.replay}")
            replay.replay_to_bridge(bridge, speed=1.0)

        try:
            while bridge.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping simulation...")

        bridge.disconnect()

        # Print stats
        stats = bridge.get_simulation_stats()
        print("\n--- Simulation Stats ---")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Export command history
        history = bridge.get_command_history()
        if history:
            print(f"\nRecorded {len(history)} commands")


if __name__ == "__main__":
    main()
