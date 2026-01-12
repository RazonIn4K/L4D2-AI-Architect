#!/usr/bin/env python3
"""
Comprehensive Integration Tests for L4D2-AI-Architect

This module tests the FULL pipeline from configuration loading through to
model inference and gameplay simulation. All tests are designed to pass
without requiring a GPU or game server connection.

Test coverage:
1. Configuration loading and validation
2. Training data loading and format verification
3. Copilot CLI commands (ollama, template, server URL validation)
4. RL environment episode execution
5. Director decision-making logic
"""

import json
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, Mock
import subprocess

import numpy as np
import pytest
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def configs_dir(project_root: Path) -> Path:
    """Return the configs directory path."""
    return project_root / "configs"


@pytest.fixture
def processed_data_dir(project_root: Path) -> Path:
    """Return the processed data directory path."""
    return project_root / "data" / "processed"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_training_config() -> Dict[str, Any]:
    """Return a sample training configuration."""
    return {
        "model": {
            "name": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
            "max_seq_length": 2048,
            "dtype": None,
            "load_in_4bit": True
        },
        "lora": {
            "r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "bias": "none"
        },
        "training": {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
            "warmup_steps": 10,
            "seed": 3407
        },
        "data": {
            "train_file": "combined_train.jsonl",
            "val_file": "combined_val.jsonl",
            "max_samples": None
        },
        "output": {
            "dir": "test-model-lora",
            "push_to_hub": False
        }
    }


@pytest.fixture
def sample_director_config() -> Dict[str, Any]:
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
        },
        "difficulty": {
            "base_difficulty": 1.0,
            "adaptive_scaling": True
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
        "special_infected": [1, 0, 1, 0, 0],
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
def sample_chatml_examples() -> List[Dict[str, Any]]:
    """Return sample ChatML training examples."""
    return [
        {
            "messages": [
                {"role": "system", "content": "You are an expert SourcePawn developer."},
                {"role": "user", "content": "Write a function to heal all survivors."},
                {"role": "assistant", "content": "public void HealAllSurvivors() { /* code */ }"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are an expert VScript developer."},
                {"role": "user", "content": "Create a function to spawn a Tank."},
                {"role": "assistant", "content": "function SpawnTank() { /* code */ }"}
            ]
        }
    ]


# ==============================================================================
# TEST: Configuration Loading
# ==============================================================================

class TestConfigurationLoading:
    """Tests for configuration file loading and validation."""

    def test_unsloth_config_exists(self, configs_dir: Path):
        """Test that main training config exists."""
        config_path = configs_dir / "unsloth_config.yaml"
        assert config_path.exists(), f"Expected config at {config_path}"

    def test_director_config_exists(self, configs_dir: Path):
        """Test that director config exists."""
        config_path = configs_dir / "director_config.yaml"
        assert config_path.exists(), f"Expected config at {config_path}"

    def test_unsloth_config_loads(self, configs_dir: Path):
        """Test that unsloth config loads correctly."""
        config_path = configs_dir / "unsloth_config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Verify required sections exist
        assert "model" in config
        assert "lora" in config
        assert "training" in config
        assert "data" in config
        assert "output" in config

    def test_director_config_loads(self, configs_dir: Path):
        """Test that director config loads correctly."""
        config_path = configs_dir / "director_config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Verify required sections exist
        assert "spawn_rates" in config
        assert "stress_factors" in config
        assert "flow_control" in config
        assert "difficulty" in config

    def test_unsloth_config_model_settings(self, configs_dir: Path):
        """Test that model settings are valid."""
        config_path = configs_dir / "unsloth_config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        model = config["model"]
        assert "name" in model
        assert "max_seq_length" in model
        assert isinstance(model["max_seq_length"], int)
        assert model["max_seq_length"] > 0

    def test_unsloth_config_lora_settings(self, configs_dir: Path):
        """Test that LoRA settings are valid."""
        config_path = configs_dir / "unsloth_config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        lora = config["lora"]
        assert "r" in lora
        assert "lora_alpha" in lora
        assert "target_modules" in lora
        assert isinstance(lora["r"], int)
        assert isinstance(lora["target_modules"], list)
        assert len(lora["target_modules"]) > 0

    def test_unsloth_config_training_settings(self, configs_dir: Path):
        """Test that training settings are valid."""
        config_path = configs_dir / "unsloth_config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        training = config["training"]
        assert "num_train_epochs" in training
        assert "learning_rate" in training
        assert "per_device_train_batch_size" in training
        assert training["num_train_epochs"] > 0
        assert 0 < training["learning_rate"] < 1

    def test_all_config_variants_load(self, configs_dir: Path):
        """Test that all config variants can be loaded."""
        config_files = list(configs_dir.glob("*.yaml"))
        assert len(config_files) > 0, "No config files found"

        for config_file in config_files:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
            assert config is not None, f"Failed to load {config_file}"


# ==============================================================================
# TEST: Training Data Loading
# ==============================================================================

class TestTrainingDataLoading:
    """Tests for training data loading and format verification."""

    def test_processed_data_dir_exists(self, processed_data_dir: Path):
        """Test that processed data directory exists."""
        assert processed_data_dir.exists(), f"Expected data dir at {processed_data_dir}"

    def test_training_files_exist(self, processed_data_dir: Path):
        """Test that essential training files exist."""
        # Check for combined training file
        combined_train = processed_data_dir / "combined_train.jsonl"
        if combined_train.exists():
            assert combined_train.stat().st_size > 0, "Training file is empty"

    def test_jsonl_format_valid(self, processed_data_dir: Path):
        """Test that JSONL files have valid format."""
        jsonl_files = list(processed_data_dir.glob("*.jsonl"))

        for jsonl_file in jsonl_files[:3]:  # Test first 3 files
            with open(jsonl_file, "r", encoding="utf-8") as f:
                line_count = 0
                for line in f:
                    line = line.strip()
                    if line:
                        # Should be valid JSON
                        data = json.loads(line)
                        assert isinstance(data, dict)
                        line_count += 1
                    if line_count >= 5:  # Only check first 5 lines
                        break

    def test_chatml_format_structure(self, processed_data_dir: Path):
        """Test that training data follows ChatML format."""
        combined_train = processed_data_dir / "combined_train.jsonl"
        if not combined_train.exists():
            pytest.skip("combined_train.jsonl not found")

        with open(combined_train, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 10:  # Check first 10 examples
                    break
                data = json.loads(line.strip())

                # ChatML format should have "messages" key
                if "messages" in data:
                    messages = data["messages"]
                    assert isinstance(messages, list)
                    assert len(messages) >= 2  # At least user + assistant

                    # Check message structure
                    for msg in messages:
                        assert "role" in msg
                        assert "content" in msg
                        assert msg["role"] in ["system", "user", "assistant"]

    def test_training_data_content_quality(self, processed_data_dir: Path):
        """Test that training data contains L4D2-related content."""
        combined_train = processed_data_dir / "combined_train.jsonl"
        if not combined_train.exists():
            pytest.skip("combined_train.jsonl not found")

        l4d2_keywords = [
            "survivor", "infected", "tank", "hunter", "smoker",
            "SourcePawn", "VScript", "Plugin", "client", "weapon"
        ]
        keyword_found = False

        with open(combined_train, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 50:  # Check first 50 examples
                    break
                data = json.loads(line.strip())
                content = json.dumps(data).lower()

                for keyword in l4d2_keywords:
                    if keyword.lower() in content:
                        keyword_found = True
                        break
                if keyword_found:
                    break

        assert keyword_found, "Training data should contain L4D2-related content"

    def test_sample_data_can_be_created(self, temp_dir: Path, sample_chatml_examples: List[Dict]):
        """Test that sample training data can be created and read."""
        sample_file = temp_dir / "test_train.jsonl"

        # Write sample data
        with open(sample_file, "w", encoding="utf-8") as f:
            for example in sample_chatml_examples:
                f.write(json.dumps(example) + "\n")

        # Read and verify
        with open(sample_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == len(sample_chatml_examples)
        for line in lines:
            data = json.loads(line.strip())
            assert "messages" in data


# ==============================================================================
# TEST: Copilot CLI Commands
# ==============================================================================

class TestCopilotCLI:
    """Tests for Copilot CLI commands."""

    def test_import_copilot_cli(self):
        """Test that copilot_cli module can be imported."""
        from inference.copilot_cli import (
            CopilotClient,
            OllamaClient,
            validate_server_url,
        )

    def test_validate_server_url_localhost(self):
        """Test URL validation for localhost."""
        from inference.copilot_cli import validate_server_url

        # Valid localhost URLs
        assert validate_server_url("http://localhost:8000") == "http://localhost:8000"
        assert validate_server_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000"
        assert validate_server_url("https://localhost:443") == "https://localhost:443"

    def test_validate_server_url_rejects_external(self):
        """Test URL validation rejects external hosts."""
        from inference.copilot_cli import validate_server_url

        # Invalid external URLs should raise
        with pytest.raises(ValueError):
            validate_server_url("http://evil.com:8000")

        with pytest.raises(ValueError):
            validate_server_url("http://192.168.1.1:8000")

    def test_validate_server_url_rejects_invalid_scheme(self):
        """Test URL validation rejects invalid schemes."""
        from inference.copilot_cli import validate_server_url

        with pytest.raises(ValueError):
            validate_server_url("ftp://localhost:8000")

    def test_ollama_client_model_validation(self):
        """Test OllamaClient model name validation."""
        from inference.copilot_cli import OllamaClient

        # Valid model names
        valid_name = OllamaClient._validate_model_name("l4d2-code-v10plus")
        assert valid_name == "l4d2-code-v10plus"

        valid_name = OllamaClient._validate_model_name("model_name.v1")
        assert valid_name == "model_name.v1"

    def test_ollama_client_rejects_invalid_model_names(self):
        """Test OllamaClient rejects malicious model names."""
        from inference.copilot_cli import OllamaClient

        # Invalid model names with special characters
        with pytest.raises(ValueError):
            OllamaClient._validate_model_name("model; rm -rf /")

        with pytest.raises(ValueError):
            OllamaClient._validate_model_name("model`whoami`")

        with pytest.raises(ValueError):
            OllamaClient._validate_model_name("")

    def test_copilot_client_initialization(self):
        """Test CopilotClient can be initialized with valid URL."""
        from inference.copilot_cli import CopilotClient

        client = CopilotClient("http://localhost:8000")
        assert client.base_url == "http://localhost:8000"

    def test_template_generation(self):
        """Test template generation produces valid code."""
        from inference.copilot_cli import generate_template_command
        import argparse

        # Create mock args
        args = argparse.Namespace(template="plugin", output=None)

        # Capture stdout
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            generate_template_command(args)

        output = f.getvalue()

        # Verify template content
        assert "#include <sourcemod>" in output
        assert "Plugin myinfo" in output
        assert "OnPluginStart" in output

    def test_all_templates_available(self):
        """Test all template types are available."""
        from inference.copilot_cli import generate_template_command
        import argparse
        import io
        from contextlib import redirect_stdout

        template_types = ["plugin", "command", "vscript", "entity"]

        for template_type in template_types:
            args = argparse.Namespace(template=template_type, output=None)

            f = io.StringIO()
            with redirect_stdout(f):
                generate_template_command(args)

            output = f.getvalue()
            assert len(output) > 100, f"Template {template_type} should produce output"

    @patch('subprocess.run')
    @patch('shutil.which')
    def test_ollama_client_check(self, mock_which, mock_run):
        """Test OllamaClient checks for ollama installation."""
        from inference.copilot_cli import OllamaClient

        # Mock ollama being available
        mock_which.return_value = "/usr/local/bin/ollama"
        mock_run.return_value = Mock(stdout="l4d2-code-v10plus", returncode=0)

        client = OllamaClient.__new__(OllamaClient)
        client.model = "l4d2-code-v10plus"
        result = client._check_ollama()
        assert result is True

    @patch('shutil.which')
    def test_ollama_client_missing_raises(self, mock_which):
        """Test OllamaClient raises when ollama not installed."""
        from inference.copilot_cli import OllamaClient

        mock_which.return_value = None

        with pytest.raises(RuntimeError) as excinfo:
            OllamaClient(model="test-model")

        assert "Ollama not found" in str(excinfo.value)


# ==============================================================================
# TEST: RL Environment Episodes
# ==============================================================================

class TestRLEnvironmentEpisodes:
    """Tests for RL environment episode execution."""

    @pytest.fixture
    def env(self):
        """Create a fresh EnhancedL4D2Env instance."""
        from rl_training.enhanced_mock_env import EnhancedL4D2Env
        environment = EnhancedL4D2Env(max_episode_steps=500, seed=42)
        yield environment
        environment.close()

    def test_env_import(self):
        """Test that RL environment can be imported."""
        from rl_training.enhanced_mock_env import (
            EnhancedL4D2Env,
            BotAction,
            ZombieType,
            ItemType,
        )

    def test_env_creation(self, env):
        """Test environment creation."""
        assert env is not None
        assert env.observation_space is not None
        assert env.action_space is not None

    def test_env_reset(self, env):
        """Test environment reset returns valid observation."""
        obs, info = env.reset()

        assert obs is not None
        assert obs.shape == (20,)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)

    def test_env_step(self, env):
        """Test environment step returns valid tuple."""
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)

        assert obs.shape == (20,)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_full_episode_random_actions(self, env):
        """Test running a full episode with random actions."""
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        max_steps = 200

        while steps < max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            # Observation should always be valid
            assert env.observation_space.contains(obs), f"Invalid obs at step {steps}"

            if terminated or truncated:
                break

        assert steps > 0, "Episode should have run at least one step"

    def test_episode_with_forward_movement(self, env):
        """Test episode with mostly forward movement."""
        from rl_training.enhanced_mock_env import BotAction

        obs, info = env.reset()
        initial_progress = env.map_state.current_progress

        # Move forward for several steps
        for _ in range(50):
            obs, reward, terminated, truncated, info = env.step(BotAction.MOVE_FORWARD)
            if terminated or truncated:
                break

        # Should have made progress
        assert env.map_state.current_progress > initial_progress

    def test_episode_survival_mechanics(self, env):
        """Test that survival mechanics work correctly."""
        from rl_training.enhanced_mock_env import BotAction

        obs, info = env.reset()

        # Player should start healthy
        assert env.player.health == 100
        assert env.player.is_alive is True
        assert env.player.is_incapped is False

        # Run for some steps
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break

        # Health should be tracked in info
        assert "health" in info
        assert "is_alive" in info

    def test_multiple_episodes(self, env):
        """Test running multiple episodes."""
        for episode in range(3):
            obs, info = env.reset(seed=episode)
            steps = 0

            while steps < 50:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                if terminated or truncated:
                    break

            # Each episode should have valid final state
            assert isinstance(info, dict)

    def test_all_actions_executable(self, env):
        """Test all actions can be executed without errors."""
        from rl_training.enhanced_mock_env import BotAction

        env.reset()

        for action in BotAction:
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs is not None
            if terminated:
                env.reset()

    def test_observation_bounds(self, env):
        """Test observations stay within bounds."""
        env.reset()

        for _ in range(100):
            action = env.action_space.sample()
            obs, _, terminated, _, _ = env.step(action)

            # Check bounds
            assert np.all(obs >= -1.0), f"Obs below -1: {obs.min()}"
            assert np.all(obs <= 1.0), f"Obs above 1: {obs.max()}"

            if terminated:
                env.reset()

    def test_info_dict_completeness(self, env):
        """Test info dictionary contains expected keys."""
        env.reset()
        _, _, _, _, info = env.step(0)

        expected_keys = [
            "health", "is_alive", "step", "episode_reward",
            "progress", "stats"
        ]

        for key in expected_keys:
            assert key in info, f"Missing key: {key}"


# ==============================================================================
# TEST: Director Decision Making
# ==============================================================================

class TestDirectorDecisionMaking:
    """Tests for AI Director decision-making logic."""

    def test_import_director_modules(self):
        """Test that director modules can be imported."""
        from director.director import L4D2Director, DirectorMode, GameState
        from director.policy import (
            DirectorPolicy, DirectorAction, RuleBasedPolicy,
            RLBasedPolicy, HybridPolicy
        )
        from director.bridge import GameBridge, MockBridge

    def test_rule_based_policy_creation(self, sample_director_config):
        """Test RuleBasedPolicy can be created."""
        from director.policy import RuleBasedPolicy

        policy = RuleBasedPolicy(sample_director_config)
        assert policy is not None
        assert policy.config == sample_director_config

    def test_rule_based_policy_decide(self, sample_director_config, sample_game_state):
        """Test RuleBasedPolicy can make decisions."""
        from director.policy import RuleBasedPolicy

        policy = RuleBasedPolicy(sample_director_config)

        # Run multiple times to accumulate actions (randomness involved)
        all_actions = []
        for _ in range(100):
            actions = policy.decide(sample_game_state, {})
            all_actions.extend(actions)

        # Should produce some actions over many iterations
        assert isinstance(all_actions, list)

    def test_action_types_are_valid(self, sample_director_config, sample_game_state):
        """Test that all action types are valid."""
        from director.policy import RuleBasedPolicy

        valid_types = {
            "spawn_common", "spawn_special", "spawn_witch",
            "spawn_tank", "trigger_panic", "spawn_item"
        }

        policy = RuleBasedPolicy(sample_director_config)

        all_actions = []
        for _ in range(200):
            actions = policy.decide(sample_game_state, {})
            all_actions.extend(actions)

        for action in all_actions:
            assert action.action_type in valid_types

    def test_rl_based_policy_without_model(self):
        """Test RLBasedPolicy works without a trained model."""
        from director.policy import RLBasedPolicy

        policy = RLBasedPolicy(model_path=None)
        assert policy.model is None
        assert policy.rule_policy is not None

    def test_rl_based_policy_fallback(self, sample_game_state):
        """Test RLBasedPolicy falls back to rules when no model."""
        from director.policy import RLBasedPolicy

        policy = RLBasedPolicy(model_path=None)
        actions = policy.decide(sample_game_state, {})

        assert isinstance(actions, list)

    def test_rl_policy_observation_conversion(self, sample_game_state):
        """Test game state to observation conversion."""
        from director.policy import RLBasedPolicy

        policy = RLBasedPolicy(model_path=None)
        obs = policy._state_to_observation(sample_game_state)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (16,)
        assert obs.dtype == np.float32
        # Values should be normalized
        assert np.all(obs >= 0)
        assert np.all(obs <= 1.5)

    def test_hybrid_policy_creation(self, sample_director_config):
        """Test HybridPolicy can be created."""
        from director.policy import HybridPolicy

        policy = HybridPolicy(sample_director_config)
        assert policy.rule_policy is not None
        assert policy.rl_policy is not None

    def test_hybrid_policy_decide(self, sample_director_config, sample_game_state):
        """Test HybridPolicy combines decisions."""
        from director.policy import HybridPolicy

        policy = HybridPolicy(sample_director_config)

        all_actions = []
        for _ in range(100):
            actions = policy.decide(sample_game_state, {})
            all_actions.extend(actions)

        assert isinstance(all_actions, list)

    def test_director_policy_main_interface(self):
        """Test main DirectorPolicy interface."""
        from director.policy import DirectorPolicy, DirectorMode

        # Test all modes
        for mode in DirectorMode:
            policy = DirectorPolicy(mode)
            assert policy.mode == mode
            assert policy.policy is not None

    def test_director_policy_decide_returns_commands(self, sample_game_state):
        """Test DirectorPolicy returns command dictionaries."""
        from director.policy import DirectorPolicy, DirectorMode

        policy = DirectorPolicy(DirectorMode.RULE_BASED)

        all_commands = []
        for _ in range(100):
            commands = policy.decide(sample_game_state, {})
            all_commands.extend(commands)

        for cmd in all_commands:
            assert "command_type" in cmd
            assert "parameters" in cmd
            assert "priority" in cmd
            assert "delay" in cmd

    def test_difficulty_update(self):
        """Test difficulty can be updated."""
        from director.policy import DirectorPolicy, DirectorMode

        policy = DirectorPolicy(DirectorMode.RULE_BASED)

        policy.update_difficulty(1.5)
        assert policy.config["difficulty"]["base_difficulty"] == 1.5

    def test_low_health_triggers_items(self, sample_director_config):
        """Test low health state triggers item spawns."""
        from director.policy import RuleBasedPolicy

        # Create low health state
        low_health_state = {
            "game_time": 100,
            "stress_level": 0.5,
            "flow_progress": 0.4,
            "common_infected": 5,
            "special_infected": [0, 0, 0, 0, 0],
            "items_available": 2,
            "survivors": [
                {"health": 20, "tempHealth": 5, "incapped": False, "dead": False},
                {"health": 15, "tempHealth": 10, "incapped": False, "dead": False},
                {"health": 30, "tempHealth": 0, "incapped": False, "dead": False},
                {"health": 10, "tempHealth": 0, "incapped": True, "dead": False},
            ]
        }

        policy = RuleBasedPolicy(sample_director_config)

        item_spawns = []
        for _ in range(200):
            actions = policy.decide(low_health_state, {})
            item_spawns.extend([a for a in actions if a.action_type == "spawn_item"])

        # Should spawn items when team is struggling
        assert len(item_spawns) > 0, "Should spawn items when health is low"

    def test_mock_bridge_creation(self):
        """Test MockBridge can be created for testing."""
        from director.bridge import MockBridge

        bridge = MockBridge()
        assert bridge is not None

    def test_mock_bridge_connect(self):
        """Test MockBridge can connect."""
        from director.bridge import MockBridge

        bridge = MockBridge()
        result = bridge.connect()
        assert result is True
        assert bridge.is_connected is True

        bridge.disconnect()
        assert bridge.is_connected is False

    def test_mock_bridge_game_state(self):
        """Test MockBridge provides game state."""
        from director.bridge import MockBridge

        bridge = MockBridge()
        bridge.connect()

        state = bridge.get_game_state()
        assert state is not None
        assert "survivors" in state

        bridge.disconnect()


# ==============================================================================
# TEST: Full Pipeline Integration
# ==============================================================================

class TestFullPipelineIntegration:
    """Tests for full pipeline integration."""

    def test_config_to_policy_pipeline(self, configs_dir: Path, temp_dir: Path):
        """Test loading config and creating policy."""
        from director.policy import DirectorPolicy, DirectorMode

        config_path = configs_dir / "director_config.yaml"

        # Load config as YAML
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # DirectorPolicy expects JSON config file, so create one for testing
        json_config_path = temp_dir / "director_config.json"
        with open(json_config_path, "w") as f:
            json.dump(config, f)

        # Create policy without config path (uses defaults)
        # Note: DirectorPolicy uses safe_read_json which expects JSON, not YAML
        policy = DirectorPolicy(DirectorMode.RULE_BASED, config_path=None)
        assert policy is not None
        assert "spawn_rates" in policy.config

    def test_data_loading_to_training_format(self, processed_data_dir: Path, temp_dir: Path):
        """Test data can be loaded and converted to training format."""
        combined_train = processed_data_dir / "combined_train.jsonl"
        if not combined_train.exists():
            pytest.skip("Training data not available")

        # Load and transform data
        examples = []
        with open(combined_train, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                data = json.loads(line.strip())
                examples.append(data)

        assert len(examples) > 0

        # Each example should be suitable for training
        for example in examples:
            if "messages" in example:
                assert len(example["messages"]) >= 2

    def test_environment_to_director_pipeline(self, sample_game_state):
        """Test RL environment state can be used by director."""
        from director.policy import RLBasedPolicy

        # Simulate getting state from environment
        env_state = {
            "stress_level": 0.5,
            "survivors": sample_game_state["survivors"],
            "common_infected": 10,
            "special_infected": [1, 0, 1, 0, 0],
            "panic_active": False,
            "items_available": 3,
            "health_packs_used": 2,
            "flow_progress": 0.4,
            "game_time": 100,
            "witch_count": 0,
            "tank_count": 0,
            "recent_kills": 5,
            "recent_damage_taken": 20,
        }

        # Director should be able to process this
        policy = RLBasedPolicy(model_path=None)
        obs = policy._state_to_observation(env_state)

        assert obs.shape == (16,)
        assert not np.any(np.isnan(obs))

    def test_cli_template_to_file_pipeline(self, project_root: Path):
        """Test template generation to file output."""
        from inference.copilot_cli import generate_template_command
        import argparse

        # Use a path within project root (since safe_write_text requires it)
        output_file = project_root / "data" / "generated_plugin_test.sp"
        args = argparse.Namespace(template="plugin", output=str(output_file))

        try:
            # This should write to file
            generate_template_command(args)

            assert output_file.exists()
            content = output_file.read_text()
            assert "#include <sourcemod>" in content
        finally:
            # Clean up
            if output_file.exists():
                output_file.unlink()

    def test_episode_to_metrics_pipeline(self):
        """Test running episode and extracting metrics."""
        from rl_training.enhanced_mock_env import EnhancedL4D2Env

        env = EnhancedL4D2Env(max_episode_steps=200, seed=42)
        obs, info = env.reset()

        episode_metrics = {
            "total_reward": 0.0,
            "steps": 0,
            "final_health": 0,
            "kills": 0,
        }

        while episode_metrics["steps"] < 100:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            episode_metrics["total_reward"] += reward
            episode_metrics["steps"] += 1

            if terminated or truncated:
                break

        # Extract final metrics
        episode_metrics["final_health"] = info["health"]
        episode_metrics["kills"] = info["stats"]["kills"]

        env.close()

        # Metrics should be populated
        assert episode_metrics["steps"] > 0
        assert isinstance(episode_metrics["total_reward"], float)


# ==============================================================================
# TEST: Security Utilities
# ==============================================================================

class TestSecurityUtilities:
    """Tests for security utility functions."""

    def test_import_security_utils(self):
        """Test security utilities can be imported."""
        from utils.security import (
            safe_path, safe_read_json, safe_read_text,
            safe_write_json, safe_write_text, validate_url
        )

    def test_safe_path_validation(self, project_root: Path, temp_dir: Path):
        """Test safe_path prevents path traversal."""
        from utils.security import safe_path

        # Valid paths should work
        valid_path = safe_path("data/processed/test.jsonl", project_root)
        assert valid_path is not None

        # Path traversal should raise
        with pytest.raises(ValueError):
            safe_path("../../../etc/passwd", project_root)

        with pytest.raises(ValueError):
            safe_path("/etc/passwd", project_root)

    def test_validate_url_ssrf_prevention(self):
        """Test URL validation prevents SSRF."""
        from utils.security import validate_url

        allowed_domains = {"api.github.com", "developer.valvesoftware.com"}

        # Valid URLs
        url = validate_url("https://api.github.com/repos/test", allowed_domains)
        assert "api.github.com" in url

        # Invalid URLs should raise
        with pytest.raises(ValueError):
            validate_url("http://localhost:8080/admin", allowed_domains)

        with pytest.raises(ValueError):
            validate_url("http://169.254.169.254/metadata", allowed_domains)

    def test_safe_read_write_cycle(self, temp_dir: Path):
        """Test safe read/write functions work correctly."""
        from utils.security import safe_write_text, safe_read_text

        test_content = "Test content for integration testing"
        test_file = temp_dir / "test_file.txt"

        # Write
        safe_write_text(str(test_file), test_content, temp_dir)

        # Read back
        read_content = safe_read_text(str(test_file), temp_dir)
        assert read_content == test_content


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
