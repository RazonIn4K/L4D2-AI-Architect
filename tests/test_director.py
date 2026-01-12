#!/usr/bin/env python3
"""
Tests for AI Director Policy Module

Tests the RuleBasedPolicy, RLBasedPolicy, and HybridPolicy implementations
including the newly implemented RL model loading and inference.
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import numpy as np
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from director.policy import (
    DirectorAction,
    DirectorMode,
    DirectorPolicy,
    HybridPolicy,
    RLBasedPolicy,
    RuleBasedPolicy,
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Return a sample director configuration."""
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


@pytest.fixture
def sample_game_state() -> Dict[str, Any]:
    """Return a sample game state for testing."""
    return {
        "game_time": 120.0,
        "stress_level": 0.5,
        "flow_progress": 0.4,
        "common_infected": 15,
        "special_infected": [1, 0, 1, 0, 0],  # smoker, boomer, hunter, spitter, jockey
        "witch_count": 0,
        "tank_count": 0,
        "panic_active": False,
        "items_available": 4,
        "health_packs_used": 2,
        "recent_kills": 10,
        "recent_damage_taken": 30,
        "survivors": [
            {"health": 80, "tempHealth": 10, "incapped": False, "dead": False},
            {"health": 60, "tempHealth": 20, "incapped": False, "dead": False},
            {"health": 100, "tempHealth": 0, "incapped": False, "dead": False},
            {"health": 40, "tempHealth": 30, "incapped": False, "dead": False},
        ]
    }


@pytest.fixture
def low_health_state(sample_game_state: Dict[str, Any]) -> Dict[str, Any]:
    """Return a game state with low survivor health."""
    state = sample_game_state.copy()
    state["survivors"] = [
        {"health": 20, "tempHealth": 5, "incapped": False, "dead": False},
        {"health": 15, "tempHealth": 10, "incapped": False, "dead": False},
        {"health": 30, "tempHealth": 0, "incapped": False, "dead": False},
        {"health": 10, "tempHealth": 0, "incapped": True, "dead": False},
    ]
    return state


@pytest.fixture
def empty_metrics() -> Dict[str, Any]:
    """Return empty metrics dictionary."""
    return {}


# ==============================================================================
# RULE-BASED POLICY TESTS
# ==============================================================================

class TestRuleBasedPolicy:
    """Tests for the rule-based director policy."""

    def test_initialization(self, sample_config):
        """Test that RuleBasedPolicy initializes correctly."""
        policy = RuleBasedPolicy(sample_config)

        assert policy.config == sample_config
        assert policy.last_panic_time == 0
        assert policy.last_tank_time == 0
        assert policy.last_witch_time == 0

    def test_decide_returns_list(self, sample_config, sample_game_state, empty_metrics):
        """Test that decide() returns a list of DirectorAction objects."""
        policy = RuleBasedPolicy(sample_config)
        actions = policy.decide(sample_game_state, empty_metrics)

        assert isinstance(actions, list)
        for action in actions:
            assert isinstance(action, DirectorAction)

    def test_action_structure(self, sample_config, sample_game_state, empty_metrics):
        """Test that actions have correct structure."""
        policy = RuleBasedPolicy(sample_config)

        # Run multiple times to get some actions
        all_actions = []
        for _ in range(100):
            all_actions.extend(policy.decide(sample_game_state, empty_metrics))

        if all_actions:
            action = all_actions[0]
            assert hasattr(action, 'action_type')
            assert hasattr(action, 'parameters')
            assert hasattr(action, 'priority')
            assert hasattr(action, 'reason')
            assert isinstance(action.parameters, dict)
            assert isinstance(action.priority, int)

    def test_low_health_spawns_items(self, sample_config, low_health_state, empty_metrics):
        """Test that low health triggers item spawns."""
        policy = RuleBasedPolicy(sample_config)

        # Run multiple times to get item spawn
        item_spawns = []
        for _ in range(100):
            actions = policy.decide(low_health_state, empty_metrics)
            item_spawns.extend([a for a in actions if a.action_type == "spawn_item"])

        # Should spawn some health items when team health is low
        assert len(item_spawns) > 0, "Should spawn items when team health is low"

    def test_valid_action_types(self, sample_config, sample_game_state, empty_metrics):
        """Test that all action types are valid."""
        valid_types = {
            "spawn_common", "spawn_special", "spawn_witch", "spawn_tank",
            "trigger_panic", "spawn_item"
        }

        policy = RuleBasedPolicy(sample_config)

        all_actions = []
        for _ in range(200):
            all_actions.extend(policy.decide(sample_game_state, empty_metrics))

        for action in all_actions:
            assert action.action_type in valid_types, \
                f"Invalid action type: {action.action_type}"


# ==============================================================================
# RL-BASED POLICY TESTS
# ==============================================================================

class TestRLBasedPolicy:
    """Tests for the RL-based director policy."""

    def test_initialization_no_model(self):
        """Test RLBasedPolicy initializes without model path."""
        policy = RLBasedPolicy(model_path=None)

        assert policy.model is None
        assert policy.difficulty_multiplier == 1.0
        assert policy.rule_policy is not None

    def test_initialization_missing_model(self, tmp_path):
        """Test RLBasedPolicy handles missing model file gracefully."""
        fake_path = tmp_path / "nonexistent_model.zip"
        policy = RLBasedPolicy(model_path=str(fake_path))

        assert policy.model is None  # Should fallback gracefully

    def test_action_map_defined(self):
        """Test that ACTION_MAP is properly defined."""
        assert hasattr(RLBasedPolicy, 'ACTION_MAP')
        assert len(RLBasedPolicy.ACTION_MAP) == 15  # 15 discrete actions

        # Check specific mappings
        assert RLBasedPolicy.ACTION_MAP[0] is None  # IDLE
        assert RLBasedPolicy.ACTION_MAP[10] == ("spawn_tank", {})
        assert RLBasedPolicy.ACTION_MAP[11] == ("trigger_panic", {})

    def test_state_to_observation(self, sample_game_state):
        """Test that state converts to correct observation format."""
        policy = RLBasedPolicy(model_path=None)
        obs = policy._state_to_observation(sample_game_state)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (16,)
        assert obs.dtype == np.float32

        # All values should be normalized (roughly 0-1)
        assert np.all(obs >= 0) and np.all(obs <= 1.5), \
            f"Observation values should be normalized, got range [{obs.min()}, {obs.max()}]"

    def test_state_to_observation_empty_survivors(self):
        """Test observation generation with no survivors."""
        policy = RLBasedPolicy(model_path=None)
        state = {"survivors": [], "stress_level": 0.5}
        obs = policy._state_to_observation(state)

        assert obs.shape == (16,)
        assert obs[1] == 1.0  # avg_health should default to 100/100

    def test_decide_without_model(self, sample_game_state, empty_metrics):
        """Test decide() falls back to rule-based when no model."""
        policy = RLBasedPolicy(model_path=None)
        actions = policy.decide(sample_game_state, empty_metrics)

        assert isinstance(actions, list)
        # Should get rule-based actions
        for action in actions:
            assert isinstance(action, DirectorAction)

    def test_update_difficulty(self):
        """Test difficulty update method."""
        policy = RLBasedPolicy(model_path=None)

        # Test normal range
        policy.update_difficulty(1.5)
        assert policy.difficulty_multiplier == 1.5

        # Test clamping to max
        policy.update_difficulty(3.0)
        assert policy.difficulty_multiplier == 2.0

        # Test clamping to min
        policy.update_difficulty(0.1)
        assert policy.difficulty_multiplier == 0.5

    def test_decide_with_mock_model(self, sample_game_state, empty_metrics):
        """Test decide() with mocked PPO model."""
        # Setup mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array(4), None)  # SPAWN_SMOKER

        policy = RLBasedPolicy(model_path=None)
        policy.model = mock_model

        actions = policy.decide(sample_game_state, empty_metrics)

        assert len(actions) == 1
        assert actions[0].action_type == "spawn_special"
        assert actions[0].parameters == {"type": "smoker"}

    def test_decide_idle_action(self, sample_game_state, empty_metrics):
        """Test that IDLE action returns empty list."""
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array(0), None)  # IDLE

        policy = RLBasedPolicy(model_path=None)
        policy.model = mock_model

        actions = policy.decide(sample_game_state, empty_metrics)

        assert len(actions) == 0  # IDLE returns no actions

    def test_difficulty_affects_spawn_counts(self, sample_game_state, empty_metrics):
        """Test that difficulty multiplier affects spawn counts."""
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array(2), None)  # SPAWN_COMMONS_MED (count: 6)

        policy = RLBasedPolicy(model_path=None)
        policy.model = mock_model
        policy.update_difficulty(2.0)

        actions = policy.decide(sample_game_state, empty_metrics)

        assert len(actions) == 1
        assert actions[0].action_type == "spawn_common"
        assert actions[0].parameters["count"] == 12  # 6 * 2.0 difficulty


# ==============================================================================
# HYBRID POLICY TESTS
# ==============================================================================

class TestHybridPolicy:
    """Tests for the hybrid policy combining rules and RL."""

    def test_initialization(self, sample_config):
        """Test HybridPolicy initializes both sub-policies."""
        policy = HybridPolicy(sample_config)

        assert isinstance(policy.rule_policy, RuleBasedPolicy)
        assert isinstance(policy.rl_policy, RLBasedPolicy)
        assert policy.rl_weight == 0.5

    def test_decide_combines_policies(self, sample_config, sample_game_state, empty_metrics):
        """Test that decide() returns combined actions."""
        policy = HybridPolicy(sample_config)

        all_actions = []
        for _ in range(100):
            actions = policy.decide(sample_game_state, empty_metrics)
            all_actions.extend(actions)

        assert isinstance(all_actions, list)

        # Should have spawn actions from rules
        spawn_actions = [a for a in all_actions if "spawn" in a.action_type]
        assert len(spawn_actions) > 0

    def test_update_difficulty(self, sample_config):
        """Test that difficulty updates both policies."""
        policy = HybridPolicy(sample_config)

        policy.update_difficulty(1.8)

        assert policy.rule_policy.config.get("base_difficulty", 1.0) == 1.8
        assert policy.rl_policy.difficulty_multiplier == 1.8


# ==============================================================================
# DIRECTOR POLICY (MAIN INTERFACE) TESTS
# ==============================================================================

class TestDirectorPolicy:
    """Tests for the main DirectorPolicy interface."""

    def test_rule_mode_initialization(self):
        """Test initialization in RULE_BASED mode."""
        policy = DirectorPolicy(DirectorMode.RULE_BASED)

        assert policy.mode == DirectorMode.RULE_BASED
        assert isinstance(policy.policy, RuleBasedPolicy)

    def test_rl_mode_initialization(self):
        """Test initialization in RL_BASED mode."""
        policy = DirectorPolicy(DirectorMode.RL_BASED)

        assert policy.mode == DirectorMode.RL_BASED
        assert isinstance(policy.policy, RLBasedPolicy)

    def test_hybrid_mode_initialization(self):
        """Test initialization in HYBRID mode."""
        policy = DirectorPolicy(DirectorMode.HYBRID)

        assert policy.mode == DirectorMode.HYBRID
        assert isinstance(policy.policy, HybridPolicy)

    def test_decide_returns_commands(self, sample_game_state, empty_metrics):
        """Test that decide() returns command dictionaries."""
        policy = DirectorPolicy(DirectorMode.RULE_BASED)

        all_commands = []
        for _ in range(100):
            commands = policy.decide(sample_game_state, empty_metrics)
            all_commands.extend(commands)

        assert isinstance(all_commands, list)

        if all_commands:
            cmd = all_commands[0]
            assert "command_type" in cmd
            assert "parameters" in cmd
            assert "priority" in cmd
            assert "delay" in cmd

    def test_update_difficulty(self):
        """Test difficulty update on main policy."""
        policy = DirectorPolicy(DirectorMode.RULE_BASED)

        policy.update_difficulty(1.5)

        assert policy.config["difficulty"]["base_difficulty"] == 1.5

    def test_default_config_values(self):
        """Test that default config has expected structure."""
        policy = DirectorPolicy(DirectorMode.RULE_BASED)

        assert "spawn_rates" in policy.config
        assert "stress_factors" in policy.config
        assert "flow_control" in policy.config
        assert "difficulty" in policy.config

        assert policy.config["spawn_rates"]["common_base"] > 0
        assert policy.config["difficulty"]["adaptive_scaling"] is True


# ==============================================================================
# EDGE CASE TESTS
# ==============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_state(self, sample_config, empty_metrics):
        """Test handling of empty state dictionary."""
        policy = RuleBasedPolicy(sample_config)
        actions = policy.decide({}, empty_metrics)

        assert isinstance(actions, list)  # Should not crash

    def test_negative_game_time(self, sample_config, sample_game_state, empty_metrics):
        """Test handling of negative game time."""
        sample_game_state["game_time"] = -100
        policy = RuleBasedPolicy(sample_config)

        actions = policy.decide(sample_game_state, empty_metrics)
        assert isinstance(actions, list)

    def test_extreme_stress_values(self, sample_config, sample_game_state, empty_metrics):
        """Test handling of extreme stress values."""
        policy = RuleBasedPolicy(sample_config)

        # Very high stress
        sample_game_state["stress_level"] = 10.0
        actions_high = policy.decide(sample_game_state, empty_metrics)
        assert isinstance(actions_high, list)

        # Negative stress
        sample_game_state["stress_level"] = -1.0
        actions_neg = policy.decide(sample_game_state, empty_metrics)
        assert isinstance(actions_neg, list)

    def test_all_survivors_dead(self, sample_config, empty_metrics):
        """Test handling when all survivors are dead."""
        state = {
            "survivors": [
                {"health": 0, "tempHealth": 0, "incapped": False, "dead": True},
                {"health": 0, "tempHealth": 0, "incapped": False, "dead": True},
                {"health": 0, "tempHealth": 0, "incapped": False, "dead": True},
                {"health": 0, "tempHealth": 0, "incapped": False, "dead": True},
            ],
            "game_time": 100,
            "stress_level": 1.0,
            "flow_progress": 0.5,
        }

        policy = RuleBasedPolicy(sample_config)
        actions = policy.decide(state, empty_metrics)
        assert isinstance(actions, list)


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
