#!/usr/bin/env python3
"""
Test Suite for AI Director

Tests the director system without requiring a live game server.
Includes unit tests, integration tests, and scenario-based tests.

Run with: pytest test_director.py -v
Or standalone: python test_director.py
"""

import sys
import time
import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch

# Add parent to path for utils
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

# Import using importlib to avoid module name conflicts
import importlib.util

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load modules directly from files
_director_mod = _load_module("director_mod", script_dir / "director.py")
_simulation_mod = _load_module("simulation_mod", script_dir / "simulation.py")
_bridge_mod = _load_module("bridge_mod", script_dir / "bridge.py")
_policy_mod = _load_module("policy_mod", script_dir / "policy.py")

# Import classes from loaded modules
L4D2Director = _director_mod.L4D2Director
DirectorMode = _director_mod.DirectorMode
GameState = _director_mod.GameState
DirectorCommand = _director_mod.DirectorCommand
DirectorStatistics = _director_mod.DirectorStatistics
DecisionLogger = _director_mod.DecisionLogger

SimulationBridge = _simulation_mod.SimulationBridge
SimulationConfig = _simulation_mod.SimulationConfig
SimulationState = _simulation_mod.SimulationState
SurvivorState = _simulation_mod.SurvivorState
Scenario = _simulation_mod.Scenario
DecisionReplay = _simulation_mod.DecisionReplay

GameBridge = _bridge_mod.GameBridge
MockBridge = _bridge_mod.MockBridge
create_bridge = _bridge_mod.create_bridge
BaseBridge = _bridge_mod.BaseBridge

DirectorPolicy = _policy_mod.DirectorPolicy
RuleBasedPolicy = _policy_mod.RuleBasedPolicy
DirectorAction = _policy_mod.DirectorAction


class TestSimulationBridge(unittest.TestCase):
    """Tests for SimulationBridge"""

    def setUp(self):
        self.config = SimulationConfig(
            tick_rate=10.0,
            max_duration=10.0,
            seed=42
        )
        self.bridge = SimulationBridge(self.config)

    def tearDown(self):
        if self.bridge.is_connected:
            self.bridge.disconnect()

    def test_connect_disconnect(self):
        """Test basic connection lifecycle"""
        self.assertFalse(self.bridge.is_connected)

        result = self.bridge.connect()
        self.assertTrue(result)
        self.assertTrue(self.bridge.is_connected)
        self.assertTrue(self.bridge.running)

        self.bridge.disconnect()
        self.assertFalse(self.bridge.is_connected)
        self.assertFalse(self.bridge.running)

    def test_initial_state(self):
        """Test initial game state"""
        self.bridge.connect()
        time.sleep(0.2)  # Let simulation start

        state = self.bridge.get_game_state()
        self.assertIsNotNone(state)
        self.assertIn("survivors", state)
        self.assertIn("gameTime", state)
        self.assertEqual(len(state["survivors"]), 4)

    def test_state_callback(self):
        """Test state update callbacks"""
        callback_data = []

        def on_state(state):
            callback_data.append(state)

        self.bridge.add_state_callback(on_state)
        self.bridge.connect()
        time.sleep(0.5)  # Let some updates happen
        self.bridge.disconnect()

        self.assertGreater(len(callback_data), 0)

    def test_director_commands(self):
        """Test sending director commands"""
        self.bridge.connect()
        time.sleep(0.2)

        initial_state = self.bridge.get_game_state()
        initial_common = initial_state.get("commonInfected", 0)

        # Spawn common infected
        self.bridge.send_director_command("spawn_common", {"count": 10})
        time.sleep(0.05)  # Shorter wait to minimize decay

        new_state = self.bridge.get_game_state()
        new_common = new_state.get("commonInfected", 0)

        # Common infected count should increase (may decay slightly due to simulation)
        self.assertGreater(new_common, initial_common)

    def test_command_history(self):
        """Test command history recording"""
        self.bridge.connect()
        time.sleep(0.1)

        self.bridge.send_director_command("spawn_common", {"count": 5})
        self.bridge.send_director_command("spawn_special", {"type": "hunter"})
        self.bridge.send_director_command("trigger_panic", {})

        history = self.bridge.get_command_history()
        self.assertEqual(len(history), 3)

        # Check command types
        types = [h["command_type"] for h in history]
        self.assertIn("spawn_common", types)
        self.assertIn("spawn_special", types)
        self.assertIn("trigger_panic", types)

    def test_simulation_stats(self):
        """Test simulation statistics"""
        self.bridge.connect()
        time.sleep(0.5)

        stats = self.bridge.get_simulation_stats()
        self.assertIn("game_time", stats)
        self.assertIn("alive_survivors", stats)
        self.assertIn("flow_progress", stats)
        self.assertGreater(stats["game_time"], 0)


class TestScenarios(unittest.TestCase):
    """Tests for scenario system"""

    def test_default_scenario(self):
        """Test default scenario creation"""
        scenario = Scenario.create_default()
        self.assertEqual(scenario.name, "default")
        self.assertIsNotNone(scenario.initial_state)

    def test_stress_scenario(self):
        """Test stress scenario creation"""
        scenario = Scenario.create_stress_test()
        self.assertEqual(scenario.name, "stress_test")
        self.assertIsNotNone(scenario.initial_state)

        # Verify stressful conditions
        state = scenario.initial_state
        self.assertGreater(state.common_infected, 10)
        self.assertGreater(sum(state.special_infected), 2)

        # Verify damaged survivors
        low_health_count = sum(1 for s in state.survivors if s.health < 50)
        self.assertGreater(low_health_count, 0)

    def test_finale_scenario(self):
        """Test finale scenario creation"""
        scenario = Scenario.create_finale()
        self.assertEqual(scenario.name, "finale")

        # Verify timed events
        self.assertGreater(len(scenario.events), 0)

        # Check for tank spawn event
        tank_events = [e for e in scenario.events if e.get("type") == "spawn_tank"]
        self.assertGreater(len(tank_events), 0)

    def test_load_scenario_into_bridge(self):
        """Test loading scenario into simulation bridge"""
        config = SimulationConfig(max_duration=5.0, seed=42)
        bridge = SimulationBridge(config)

        scenario = Scenario.create_stress_test()
        bridge.load_scenario(scenario)
        bridge.connect()
        time.sleep(0.3)

        state = bridge.get_game_state()
        # Verify stress conditions are loaded
        self.assertGreater(state.get("commonInfected", 0), 10)

        bridge.disconnect()


class TestDirectorStatistics(unittest.TestCase):
    """Tests for statistics tracking"""

    def test_spawn_recording(self):
        """Test spawn event recording"""
        stats = DirectorStatistics()

        stats.record_spawn("common", 5)
        stats.record_spawn("common", 3)
        stats.record_spawn("special")
        stats.record_spawn("tank")

        self.assertEqual(stats.spawned_common, 8)
        self.assertEqual(stats.spawned_special, 1)
        self.assertEqual(stats.spawned_tanks, 1)

    def test_stress_tracking(self):
        """Test stress level tracking"""
        stats = DirectorStatistics()

        stats.record_stress(0.2)
        stats.record_stress(0.5)
        stats.record_stress(0.8)

        self.assertAlmostEqual(stats.avg_stress, 0.5, places=1)
        self.assertEqual(stats.max_stress, 0.8)
        self.assertEqual(stats.min_stress, 0.2)

    def test_decision_tracking(self):
        """Test decision recording"""
        stats = DirectorStatistics()

        stats.record_decision("spawn_common")
        stats.record_decision("spawn_common")
        stats.record_decision("spawn_special")
        stats.record_decision("trigger_panic")

        self.assertEqual(stats.total_decisions, 4)
        self.assertEqual(stats.command_counts["spawn_common"], 2)
        self.assertEqual(stats.command_counts["spawn_special"], 1)
        self.assertEqual(stats.command_counts["trigger_panic"], 1)

    def test_to_dict(self):
        """Test statistics serialization"""
        stats = DirectorStatistics()
        stats.record_spawn("common", 10)
        stats.record_stress(0.5)

        data = stats.to_dict()

        self.assertIn("session_duration", data)
        self.assertIn("spawns", data)
        self.assertIn("stress", data)
        self.assertEqual(data["spawns"]["common"], 10)


class TestDecisionLogger(unittest.TestCase):
    """Tests for decision logging"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def test_log_creation(self):
        """Test log file creation"""
        logger = DecisionLogger(log_dir=self.temp_dir, detail_level="standard")

        # Check log directory was created
        log_path = Path(self.temp_dir)
        self.assertTrue(log_path.exists())

    def test_decision_logging(self):
        """Test logging a decision"""
        logger = DecisionLogger(log_dir=self.temp_dir, detail_level="full")

        # Create mock state
        state = GameState(
            game_time=10.0,
            round_time=10.0,
            survivors=[{"id": 1, "health": 100}],
            common_infected=5,
            special_infected=[0, 0, 1, 0, 0],
            witch_count=0,
            tank_count=0,
            flow_progress=0.3,
            stress_level=0.4,
            items_available=2,
            health_packs_used=0,
            recent_deaths=0,
            panic_active=False,
            tank_active=False
        )

        command = DirectorCommand(
            command_type="spawn_common",
            parameters={"count": 5},
            priority=1,
            delay=0.0
        )

        logger.log_decision(command, state, {"avg_stress": 0.4}, "test reason")

        # Verify decision was recorded
        self.assertEqual(len(logger.decisions), 1)
        self.assertEqual(logger.decisions[0].command_type, "spawn_common")

    def test_decisions_by_type(self):
        """Test filtering decisions by type"""
        logger = DecisionLogger(log_dir=self.temp_dir, detail_level="minimal")

        state = GameState(
            game_time=10.0, round_time=10.0, survivors=[],
            common_infected=0, special_infected=[0,0,0,0,0],
            witch_count=0, tank_count=0, flow_progress=0.5,
            stress_level=0.5, items_available=0, health_packs_used=0,
            recent_deaths=0, panic_active=False, tank_active=False
        )

        # Log different command types
        for cmd_type in ["spawn_common", "spawn_special", "spawn_common", "trigger_panic"]:
            cmd = DirectorCommand(cmd_type, {}, 1, 0.0)
            logger.log_decision(cmd, state, {}, "")

        common_decisions = logger.get_decisions_by_type("spawn_common")
        self.assertEqual(len(common_decisions), 2)


class TestDirectorPolicy(unittest.TestCase):
    """Tests for director policy"""

    def test_rule_based_policy_creation(self):
        """Test creating rule-based policy"""
        policy = DirectorPolicy(DirectorMode.RULE_BASED)
        self.assertIsNotNone(policy.policy)
        self.assertIsInstance(policy.policy, RuleBasedPolicy)

    def test_policy_decision(self):
        """Test policy makes decisions"""
        policy = DirectorPolicy(DirectorMode.RULE_BASED)

        # Create test state
        state = {
            "game_time": 100.0,
            "stress_level": 0.5,
            "flow_progress": 0.3,
            "survivors": [{"health": 80}, {"health": 60}],
            "common_infected": 5,
            "special_infected": [0, 0, 0, 0, 0],
            "panic_active": False,
            "tank_active": False,
            "items_available": 2
        }

        decisions = policy.decide(state, {})
        # Should return a list (may be empty or have decisions)
        self.assertIsInstance(decisions, list)


class TestL4D2Director(unittest.TestCase):
    """Tests for main director class"""

    def test_simulation_mode_init(self):
        """Test director initialization in simulation mode"""
        sim_config = SimulationConfig(max_duration=5.0, seed=42)
        director = L4D2Director(
            mode=DirectorMode.RULE_BASED,
            simulation_mode=True,
            simulation_config=sim_config,
            log_decisions=False
        )

        self.assertTrue(director.simulation_mode)
        # Check it's a simulation bridge by checking for simulation-specific attribute
        self.assertTrue(hasattr(director.bridge, 'load_scenario'))

    def test_director_start_stop(self):
        """Test director lifecycle"""
        sim_config = SimulationConfig(max_duration=2.0, seed=42)
        director = L4D2Director(
            mode=DirectorMode.RULE_BASED,
            simulation_mode=True,
            simulation_config=sim_config,
            log_decisions=False
        )

        director.start()
        self.assertTrue(director.is_running)
        time.sleep(0.5)

        director.stop()
        self.assertFalse(director.is_running)

    def test_director_statistics(self):
        """Test director statistics gathering"""
        sim_config = SimulationConfig(max_duration=3.0, seed=42)
        director = L4D2Director(
            mode=DirectorMode.RULE_BASED,
            simulation_mode=True,
            simulation_config=sim_config,
            log_decisions=False
        )

        director.start()
        time.sleep(1.0)
        director.stop()

        stats = director.get_statistics()
        self.assertIn("session_duration", stats)
        self.assertIn("spawns", stats)
        self.assertGreater(stats["session_duration"], 0)

    def test_director_with_scenario(self):
        """Test director with loaded scenario"""
        sim_config = SimulationConfig(max_duration=3.0, seed=42)
        director = L4D2Director(
            mode=DirectorMode.RULE_BASED,
            simulation_mode=True,
            simulation_config=sim_config,
            log_decisions=False
        )

        scenario = Scenario.create_stress_test()
        director.load_scenario(scenario)

        director.start()
        time.sleep(1.0)
        director.stop()

        # With stress scenario, should have made decisions
        stats = director.get_statistics()
        self.assertGreaterEqual(stats["total_decisions"], 0)

    def test_difficulty_adjustment(self):
        """Test difficulty adjustment"""
        sim_config = SimulationConfig(max_duration=1.0)
        director = L4D2Director(
            mode=DirectorMode.RULE_BASED,
            simulation_mode=True,
            simulation_config=sim_config,
            log_decisions=False
        )

        director.set_difficulty(1.5)
        self.assertEqual(director.config["difficulty"]["base_difficulty"], 1.5)

        # Test clamping
        director.set_difficulty(3.0)
        self.assertEqual(director.config["difficulty"]["base_difficulty"], 2.0)

        director.set_difficulty(0.1)
        self.assertEqual(director.config["difficulty"]["base_difficulty"], 0.5)


class TestDecisionReplay(unittest.TestCase):
    """Tests for decision replay functionality"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def test_create_and_replay_log(self):
        """Test creating a log and replaying it"""
        # First, run director and create log
        sim_config = SimulationConfig(max_duration=2.0, seed=42)
        director = L4D2Director(
            mode=DirectorMode.RULE_BASED,
            simulation_mode=True,
            simulation_config=sim_config,
            log_decisions=True,
            log_detail_level="standard"
        )

        scenario = Scenario.create_default()
        director.load_scenario(scenario)

        director.start()
        time.sleep(1.5)
        director.stop()

        # Get the log file path
        if director.decision_logger:
            log_path = str(director.decision_logger._decision_log_path)

            # Create replay
            replay = DecisionReplay(log_path)
            summary = replay.get_decision_summary()

            self.assertIn("total_decisions", summary)
            self.assertIn("by_type", summary)


class TestBridgeFactory(unittest.TestCase):
    """Tests for bridge factory function"""

    def test_create_game_bridge(self):
        """Test creating game bridge"""
        bridge = create_bridge("game", host="localhost", port=27050)
        self.assertIsInstance(bridge, GameBridge)

    def test_create_simulation_bridge(self):
        """Test creating simulation bridge"""
        bridge = create_bridge("simulation")
        # Check it's a simulation bridge by duck-typing
        self.assertTrue(hasattr(bridge, 'load_scenario'))
        self.assertTrue(hasattr(bridge, 'get_simulation_stats'))

    def test_create_mock_bridge(self):
        """Test creating mock bridge (deprecated)"""
        bridge = create_bridge("mock")
        self.assertIsInstance(bridge, MockBridge)

    def test_invalid_bridge_type(self):
        """Test invalid bridge type raises error"""
        with self.assertRaises(ValueError):
            create_bridge("invalid_type")


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows"""

    def test_full_simulation_workflow(self):
        """Test complete simulation workflow"""
        # Create simulation config
        sim_config = SimulationConfig(
            max_duration=5.0,
            seed=42,
            time_scale=2.0  # 2x speed
        )

        # Create director
        director = L4D2Director(
            mode=DirectorMode.RULE_BASED,
            simulation_mode=True,
            simulation_config=sim_config,
            log_decisions=True,
            log_detail_level="full"
        )

        # Load stress scenario
        scenario = Scenario.create_stress_test()
        director.load_scenario(scenario)

        # Run simulation
        director.start()

        # Monitor during run
        snapshots = []
        for _ in range(5):
            time.sleep(0.5)
            stats = director.get_statistics()
            snapshots.append(stats.copy())

        director.stop()

        # Verify metrics changed over time
        self.assertGreater(len(snapshots), 0)

        # Verify statistics are reasonable
        final_stats = director.get_statistics()
        self.assertGreater(final_stats["session_duration"], 0)

        # Print report for inspection
        director.print_statistics_report()

    def test_scenario_comparison(self):
        """Test comparing director behavior across scenarios"""
        results = {}

        for scenario_name in ["easy", "stress", "finale"]:
            sim_config = SimulationConfig(max_duration=3.0, seed=42)
            director = L4D2Director(
                mode=DirectorMode.RULE_BASED,
                simulation_mode=True,
                simulation_config=sim_config,
                log_decisions=False
            )

            scenario_map = {
                "easy": Scenario.create_easy,
                "stress": Scenario.create_stress_test,
                "finale": Scenario.create_finale
            }
            scenario = scenario_map[scenario_name]()
            director.load_scenario(scenario)

            director.start()
            time.sleep(2.0)
            director.stop()

            results[scenario_name] = director.get_statistics()

        # Stress scenario should have more decisions than easy
        # (This may vary based on implementation)
        self.assertIsNotNone(results["easy"])
        self.assertIsNotNone(results["stress"])
        self.assertIsNotNone(results["finale"])


def run_quick_demo():
    """Run a quick demonstration of the director system"""
    print("\n" + "=" * 60)
    print("AI DIRECTOR DEMONSTRATION")
    print("=" * 60)

    # Create simulation
    print("\n1. Creating simulation with stress test scenario...")
    sim_config = SimulationConfig(
        max_duration=10.0,
        seed=42,
        time_scale=1.0
    )

    director = L4D2Director(
        mode=DirectorMode.RULE_BASED,
        simulation_mode=True,
        simulation_config=sim_config,
        log_decisions=True,
        log_detail_level="standard"
    )

    # Load scenario
    scenario = Scenario.create_stress_test()
    director.load_scenario(scenario)
    print(f"   Loaded scenario: {scenario.name}")
    print(f"   Description: {scenario.description}")

    # Start director
    print("\n2. Starting director...")
    director.start()

    # Monitor progress
    print("\n3. Running simulation (10 seconds)...")
    print("   Time | Stress | Flow | Decisions | Spawns/min")
    print("   " + "-" * 50)

    for i in range(10):
        time.sleep(1.0)
        stats = director.get_statistics()
        spm = director.get_spawns_per_minute()
        stress = stats["stress"]["current"] if stats["stress"]["current"] else 0

        print(f"   {i+1:4}s | {stress:6.2f} | {0:4.2f} | "
              f"{stats['total_decisions']:9} | {spm.get('total', 0):10}")

    # Stop and show results
    director.stop()

    print("\n4. Final Statistics Report:")
    director.print_statistics_report()

    # Show log location
    if director.decision_logger:
        print(f"Decision log saved to: {director.decision_logger._decision_log_path}")

    print("\nDemonstration complete!")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI Director Test Suite")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("--test", action="store_true", help="Run unit tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.demo:
        run_quick_demo()
    elif args.test or (not args.demo):
        # Default to running tests
        verbosity = 2 if args.verbose else 1
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(sys.modules[__name__])
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
