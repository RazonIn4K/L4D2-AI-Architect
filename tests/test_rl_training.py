#!/usr/bin/env python3
"""
Tests for the RL Training Pipeline

This module provides comprehensive tests for the EnhancedL4D2Env
reinforcement learning environment used for training L4D2 bot agents.

Tests cover:
1. Environment initialization and reset
2. Step function behavior
3. Observation and action space validation
4. PPO training smoke test
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project paths for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "rl_training"))

from rl_training.enhanced_mock_env import (
    EnhancedL4D2Env,
    BotAction,
    ZombieType,
    ItemType,
    Vector3,
    PlayerState,
    MapState,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def env():
    """Create a fresh EnhancedL4D2Env instance."""
    environment = EnhancedL4D2Env(max_episode_steps=1000, seed=42)
    yield environment
    environment.close()


@pytest.fixture
def seeded_env():
    """Create an environment with fixed seed for reproducibility."""
    environment = EnhancedL4D2Env(max_episode_steps=500, seed=12345)
    yield environment
    environment.close()


@pytest.fixture
def easy_env():
    """Create an environment on easy difficulty."""
    environment = EnhancedL4D2Env(
        max_episode_steps=500,
        difficulty="easy",
        seed=42
    )
    yield environment
    environment.close()


@pytest.fixture
def expert_env():
    """Create an environment on expert difficulty."""
    environment = EnhancedL4D2Env(
        max_episode_steps=500,
        difficulty="expert",
        seed=42
    )
    yield environment
    environment.close()


@pytest.fixture
def reset_env(env):
    """Return an environment that has been reset."""
    env.reset(seed=42)
    return env


# ==============================================================================
# Test: Environment Initialization
# ==============================================================================

class TestEnvInitialization:
    """Tests for EnhancedL4D2Env initialization."""

    def test_env_creation(self, env):
        """Test that environment can be created."""
        assert env is not None
        assert isinstance(env, EnhancedL4D2Env)

    def test_observation_space_defined(self, env):
        """Test that observation space is properly defined."""
        assert env.observation_space is not None
        assert env.observation_space.shape == (20,)
        assert env.observation_space.dtype == np.float32

    def test_action_space_defined(self, env):
        """Test that action space is properly defined."""
        assert env.action_space is not None
        assert env.action_space.n == len(BotAction)
        assert env.action_space.n == 14

    def test_max_episode_steps(self, env):
        """Test max episode steps configuration."""
        assert env.max_episode_steps == 1000

    def test_difficulty_settings(self, easy_env, expert_env):
        """Test difficulty settings are applied."""
        assert easy_env.diff["damage_mult"] == 0.5
        assert easy_env.diff["spawn_mult"] == 0.5

        assert expert_env.diff["damage_mult"] == 2.0
        assert expert_env.diff["spawn_mult"] == 2.0

    def test_reward_config_exists(self, env):
        """Test that reward configuration is defined."""
        assert env.reward_config is not None
        assert "kill" in env.reward_config
        assert "death" in env.reward_config
        assert "safe_room" in env.reward_config

    def test_seed_reproducibility(self):
        """Test that seeding produces reproducible environments."""
        env1 = EnhancedL4D2Env(seed=42)
        env2 = EnhancedL4D2Env(seed=42)

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        np.testing.assert_array_equal(obs1, obs2)

        env1.close()
        env2.close()


# ==============================================================================
# Test: Environment Reset
# ==============================================================================

class TestEnvReset:
    """Tests for EnhancedL4D2Env reset functionality."""

    def test_reset_returns_tuple(self, env):
        """Test that reset returns observation and info."""
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reset_observation_shape(self, env):
        """Test that reset returns correctly shaped observation."""
        obs, info = env.reset()
        assert obs.shape == (20,)
        assert obs.dtype == np.float32

    def test_reset_observation_bounds(self, env):
        """Test that observation values are within bounds."""
        obs, _ = env.reset()
        assert np.all(obs >= -1.0)
        assert np.all(obs <= 1.0)

    def test_reset_info_dict(self, env):
        """Test that reset returns expected info dictionary."""
        _, info = env.reset()
        assert isinstance(info, dict)
        assert "health" in info
        assert "progress" in info
        assert "stats" in info

    def test_reset_initializes_player(self, env):
        """Test that reset initializes player state."""
        env.reset()
        assert env.player is not None
        assert env.player.health == 100
        assert env.player.is_alive is True
        assert env.player.ammo == 100

    def test_reset_initializes_teammates(self, env):
        """Test that reset initializes teammates."""
        env.reset()
        assert len(env.teammates) == 3
        for tm in env.teammates:
            assert tm.is_alive is True
            assert tm.health == 100

    def test_reset_initializes_map(self, env):
        """Test that reset initializes map state."""
        env.reset()
        assert env.map_state is not None
        assert env.map_state.total_distance == 5000.0
        assert env.map_state.current_progress == 0.0

    def test_reset_clears_entities(self, env):
        """Test that reset clears zombies and items."""
        env.reset()
        # Run a few steps to spawn zombies
        for _ in range(50):
            env.step(BotAction.MOVE_FORWARD)

        # Reset should clear them
        env.reset()
        assert len(env.zombies) == 0
        assert len(env.items) == 0

    def test_reset_clears_statistics(self, env):
        """Test that reset clears statistics."""
        env.reset()
        assert env.stats["kills"] == 0
        assert env.stats["damage_dealt"] == 0
        assert env.stats["distance_traveled"] == 0.0

    def test_reset_with_seed(self, env):
        """Test reset with explicit seed."""
        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)


# ==============================================================================
# Test: Environment Step
# ==============================================================================

class TestEnvStep:
    """Tests for EnhancedL4D2Env step functionality."""

    def test_step_returns_tuple(self, reset_env):
        """Test that step returns 5-element tuple."""
        result = reset_env.step(BotAction.IDLE)
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_step_observation_shape(self, reset_env):
        """Test that step returns correctly shaped observation."""
        obs, _, _, _, _ = reset_env.step(BotAction.IDLE)
        assert obs.shape == (20,)
        assert obs.dtype == np.float32

    def test_step_observation_bounds(self, reset_env):
        """Test that step observation stays within bounds."""
        for _ in range(20):
            action = reset_env.action_space.sample()
            obs, _, _, _, _ = reset_env.step(action)
            assert np.all(obs >= -1.0), f"Obs below -1: {obs}"
            assert np.all(obs <= 1.0), f"Obs above 1: {obs}"

    def test_step_reward_is_float(self, reset_env):
        """Test that step reward is a float."""
        _, reward, _, _, _ = reset_env.step(BotAction.IDLE)
        assert isinstance(reward, (int, float))

    def test_step_terminated_is_bool(self, reset_env):
        """Test that terminated is boolean."""
        _, _, terminated, _, _ = reset_env.step(BotAction.IDLE)
        assert isinstance(terminated, bool)

    def test_step_truncated_is_bool(self, reset_env):
        """Test that truncated is boolean."""
        _, _, _, truncated, _ = reset_env.step(BotAction.IDLE)
        assert isinstance(truncated, bool)

    def test_step_info_dict(self, reset_env):
        """Test that step returns expected info keys."""
        _, _, _, _, info = reset_env.step(BotAction.IDLE)
        assert isinstance(info, dict)
        assert "health" in info
        assert "is_alive" in info
        assert "step" in info
        assert "episode_reward" in info
        assert "progress" in info

    def test_step_increments_counter(self, reset_env):
        """Test that step counter increments."""
        assert reset_env.current_step == 0
        reset_env.step(BotAction.IDLE)
        assert reset_env.current_step == 1
        reset_env.step(BotAction.IDLE)
        assert reset_env.current_step == 2

    def test_move_forward_updates_position(self, reset_env):
        """Test that MOVE_FORWARD updates player position."""
        initial_x = reset_env.player.position.x
        reset_env.step(BotAction.MOVE_FORWARD)
        assert reset_env.player.position.x > initial_x

    def test_move_backward_updates_position(self, reset_env):
        """Test that MOVE_BACKWARD updates player position."""
        # First move forward a bit
        for _ in range(5):
            reset_env.step(BotAction.MOVE_FORWARD)

        pos_before = reset_env.player.position.x
        reset_env.step(BotAction.MOVE_BACKWARD)
        assert reset_env.player.position.x < pos_before

    def test_attack_uses_ammo(self, reset_env):
        """Test that ATTACK action uses ammo."""
        initial_ammo = reset_env.player.ammo
        reset_env.step(BotAction.ATTACK)
        assert reset_env.player.ammo == initial_ammo - 1

    def test_reload_restores_ammo(self, reset_env):
        """Test that RELOAD action restores ammo."""
        # Use some ammo
        for _ in range(10):
            reset_env.step(BotAction.ATTACK)

        ammo_before = reset_env.player.ammo
        reset_env.step(BotAction.RELOAD)
        assert reset_env.player.ammo > ammo_before

    def test_truncation_at_max_steps(self, env):
        """Test that episode truncates at max steps."""
        env.max_episode_steps = 10
        env.reset()

        for i in range(15):
            _, _, terminated, truncated, _ = env.step(BotAction.IDLE)
            if terminated or truncated:
                assert i >= 9  # Should truncate at step 10
                break

    def test_all_actions_valid(self, reset_env):
        """Test that all actions can be executed without error."""
        for action in BotAction:
            obs, reward, terminated, truncated, info = reset_env.step(action)
            assert obs is not None
            if terminated:
                reset_env.reset()


# ==============================================================================
# Test: Observation and Action Space Validation
# ==============================================================================

class TestSpaceValidation:
    """Tests for observation and action space validation."""

    def test_observation_contains_valid(self, env):
        """Test observation space contains method."""
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

    def test_action_space_sample(self, env):
        """Test action space sampling."""
        for _ in range(100):
            action = env.action_space.sample()
            assert env.action_space.contains(action)
            assert 0 <= action < len(BotAction)

    def test_action_space_bounds(self, env):
        """Test action space bounds."""
        assert env.action_space.start == 0
        assert env.action_space.n == 14

    def test_observation_indices(self, reset_env):
        """Test that observation indices map to expected values."""
        obs, _ = reset_env.reset()

        # Index 0: health / 100
        assert 0 <= obs[0] <= 1.0  # Health normalized

        # Index 1: is_alive (should be 1.0)
        assert obs[1] == 1.0

        # Index 2: is_incapped (should be 0.0)
        assert obs[2] == 0.0

        # Index 12: ammo / 100 (should be close to 1.0)
        assert 0.9 <= obs[12] <= 1.0

    def test_random_actions_produce_valid_obs(self, env):
        """Test that random actions always produce valid observations."""
        env.reset(seed=42)

        for _ in range(50):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)

            assert env.observation_space.contains(obs), \
                f"Invalid observation: {obs}"

            if terminated or truncated:
                env.reset()


# ==============================================================================
# Test: Vector3 Helper Class
# ==============================================================================

class TestVector3:
    """Tests for the Vector3 helper class."""

    def test_vector3_creation(self):
        """Test Vector3 creation."""
        v = Vector3(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_vector3_addition(self):
        """Test Vector3 addition."""
        v1 = Vector3(1.0, 2.0, 3.0)
        v2 = Vector3(4.0, 5.0, 6.0)
        result = v1 + v2
        assert result.x == 5.0
        assert result.y == 7.0
        assert result.z == 9.0

    def test_vector3_subtraction(self):
        """Test Vector3 subtraction."""
        v1 = Vector3(5.0, 7.0, 9.0)
        v2 = Vector3(1.0, 2.0, 3.0)
        result = v1 - v2
        assert result.x == 4.0
        assert result.y == 5.0
        assert result.z == 6.0

    def test_vector3_scalar_multiply(self):
        """Test Vector3 scalar multiplication."""
        v = Vector3(2.0, 3.0, 4.0)
        result = v * 2.0
        assert result.x == 4.0
        assert result.y == 6.0
        assert result.z == 8.0

    def test_vector3_magnitude(self):
        """Test Vector3 magnitude calculation."""
        v = Vector3(3.0, 4.0, 0.0)
        assert v.magnitude() == 5.0

    def test_vector3_normalized(self):
        """Test Vector3 normalization."""
        v = Vector3(3.0, 4.0, 0.0)
        n = v.normalized()
        assert abs(n.magnitude() - 1.0) < 0.001

    def test_vector3_distance(self):
        """Test Vector3 distance calculation."""
        v1 = Vector3(0.0, 0.0, 0.0)
        v2 = Vector3(3.0, 4.0, 0.0)
        assert v1.distance_to(v2) == 5.0


# ==============================================================================
# Test: Game Mechanics
# ==============================================================================

class TestGameMechanics:
    """Tests for game mechanics simulation."""

    def test_progress_increases_with_movement(self, reset_env):
        """Test that map progress increases when moving forward."""
        initial_progress = reset_env.map_state.current_progress

        for _ in range(10):
            reset_env.step(BotAction.MOVE_FORWARD)

        assert reset_env.map_state.current_progress > initial_progress

    def test_health_item_usage(self, reset_env):
        """Test health item usage mechanics."""
        # Damage the player
        reset_env.player.health = 50
        reset_env.player.health_item = 1  # Medkit

        reset_env.step(BotAction.HEAL_SELF)

        assert reset_env.player.health > 50
        assert reset_env.player.health_item == 0

    def test_shove_has_cooldown(self, reset_env):
        """Test that shove action has cooldown."""
        reset_env.step(BotAction.SHOVE)
        assert reset_env.shove_cooldown > 0

    def test_map_progress_percent(self, reset_env):
        """Test map progress percentage calculation."""
        assert reset_env.map_state.progress_percent == 0.0

        # Move forward significantly
        for _ in range(100):
            reset_env.step(BotAction.MOVE_FORWARD)

        assert reset_env.map_state.progress_percent > 0.0


# ==============================================================================
# Test: PPO Training Smoke Test
# ==============================================================================

class TestPPOSmoke:
    """Quick smoke test for PPO training compatibility."""

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_ppo_training_100_steps(self, env):
        """Test that PPO can train for 100 steps without errors."""
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.env_checker import check_env
        except ImportError:
            pytest.skip("stable-baselines3 not installed")

        # Check environment is compatible
        check_env(env, warn=True)

        # Create PPO model with minimal config for speed
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=32,
            batch_size=32,
            n_epochs=1,
            learning_rate=3e-4,
            verbose=0,
            seed=42,
        )

        # Train for exactly 100 steps (will round up to n_steps multiple)
        model.learn(total_timesteps=100)

        # Verify model can predict
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)

        assert env.action_space.contains(action)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_env_checker_passes(self, env):
        """Test that environment passes SB3 env checker."""
        try:
            from stable_baselines3.common.env_checker import check_env
        except ImportError:
            pytest.skip("stable-baselines3 not installed")

        # This will raise if there are issues
        check_env(env, warn=True)

    def test_multiple_episodes(self, env):
        """Test running multiple episodes."""
        for episode in range(3):
            obs, info = env.reset(seed=episode)
            total_reward = 0

            for step in range(50):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                if terminated or truncated:
                    break

            # Episode should have run
            assert env.current_step > 0


# ==============================================================================
# Test: Edge Cases
# ==============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_ammo_attack(self, reset_env):
        """Test attack with no ammo."""
        reset_env.player.ammo = 0
        initial_ammo = reset_env.player.ammo

        reset_env.step(BotAction.ATTACK)

        assert reset_env.player.ammo == initial_ammo  # No change

    def test_heal_at_full_health(self, reset_env):
        """Test healing when already at full health."""
        reset_env.player.health = 100
        reset_env.player.health_item = 1

        reset_env.step(BotAction.HEAL_SELF)

        # Should not use item when at full health
        assert reset_env.player.health_item == 1

    def test_throw_without_throwable(self, reset_env):
        """Test throw action with no throwable item."""
        reset_env.player.throwable = 0

        # Should not crash
        obs, reward, _, _, _ = reset_env.step(BotAction.THROW_ITEM)
        assert obs is not None

    def test_heal_other_no_target(self, reset_env):
        """Test heal other with no valid target."""
        reset_env.player.health_item = 1

        # Move teammates far away
        for tm in reset_env.teammates:
            tm.position = Vector3(10000, 10000, 0)

        reset_env.step(BotAction.HEAL_OTHER)

        # Item should not be used
        assert reset_env.player.health_item == 1

    def test_incapped_player_cannot_move(self, reset_env):
        """Test that incapped player cannot move."""
        reset_env.player.is_incapped = True
        initial_pos = reset_env.player.position.x

        reset_env.step(BotAction.MOVE_FORWARD)

        assert reset_env.player.position.x == initial_pos

    def test_grabbed_player_cannot_move(self, reset_env):
        """Test that grabbed player cannot move."""
        reset_env.player.is_grabbed = True
        initial_pos = reset_env.player.position.x

        reset_env.step(BotAction.MOVE_FORWARD)

        assert reset_env.player.position.x == initial_pos
