#!/usr/bin/env python3
"""
Comprehensive Test Suite for L4D2-AI-Architect

This module tests all major components of the L4D2 AI training system:
- Data pipeline (loading, format validation, deduplication)
- Embeddings (generation, FAISS search, dimensions)
- RL Environment (reset, step, observation/action spaces, rewards)
- Security utilities (path traversal, URL validation, safe writes)
- Configuration validation (YAML validity, required fields)

Run with: pytest tests/ -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


# ==============================================================================
# DATA PIPELINE TESTS
# ==============================================================================

class TestDataPipeline:
    """Tests for the data loading and preparation pipeline."""

    def test_load_training_data(self, processed_data_dir: Path):
        """Test that training data can be loaded from JSONL files."""
        # Find any available training file
        training_files = list(processed_data_dir.glob("*.jsonl"))

        if not training_files:
            pytest.skip("No training data files found in processed directory")

        # Load the first available file
        train_file = training_files[0]
        data = []
        with open(train_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        assert len(data) > 0, f"Training file {train_file.name} is empty"
        assert isinstance(data[0], dict), "Training examples should be dictionaries"

    def test_dataset_format_valid(self, processed_data_dir: Path):
        """Test that training data follows the ChatML format."""
        training_files = list(processed_data_dir.glob("*.jsonl"))

        if not training_files:
            pytest.skip("No training data files found")

        # Check format of first file
        train_file = training_files[0]
        with open(train_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue

                example = json.loads(line)

                # Check for messages key
                assert "messages" in example, f"Example {i} missing 'messages' key"

                messages = example["messages"]
                assert isinstance(messages, list), f"Example {i}: 'messages' should be a list"
                assert len(messages) >= 2, f"Example {i}: should have at least 2 messages"

                # Check message structure
                for j, msg in enumerate(messages):
                    assert "role" in msg, f"Example {i}, message {j}: missing 'role'"
                    assert "content" in msg, f"Example {i}, message {j}: missing 'content'"
                    assert msg["role"] in ["system", "user", "assistant"], \
                        f"Example {i}, message {j}: invalid role '{msg['role']}'"

                # Only check first 10 examples for speed
                if i >= 10:
                    break

    def test_no_duplicate_examples(self, processed_data_dir: Path):
        """Test that training data has no exact duplicates."""
        training_files = list(processed_data_dir.glob("*train*.jsonl"))

        if not training_files:
            pytest.skip("No training data files found")

        train_file = training_files[0]
        seen_hashes = set()
        duplicates = 0

        with open(train_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                # Use line hash for deduplication check
                line_hash = hash(line.strip())
                if line_hash in seen_hashes:
                    duplicates += 1
                else:
                    seen_hashes.add(line_hash)

        total_examples = len(seen_hashes) + duplicates
        duplicate_rate = duplicates / max(total_examples, 1)

        # Allow up to 5% duplicates
        assert duplicate_rate < 0.05, \
            f"Too many duplicates: {duplicates}/{total_examples} ({duplicate_rate:.1%})"

    def test_sample_data_fixture(self, sample_jsonl_file: Path):
        """Test that sample JSONL fixture creates valid data."""
        assert sample_jsonl_file.exists(), "Sample JSONL file should exist"

        data = []
        with open(sample_jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))

        assert len(data) == 3, "Should have 3 sample examples"

        for example in data:
            assert "messages" in example
            assert len(example["messages"]) >= 2


# ==============================================================================
# EMBEDDING TESTS
# ==============================================================================

class TestEmbeddings:
    """Tests for embedding generation and FAISS search functionality."""

    def test_embedding_generation(self, sample_embeddings: np.ndarray):
        """Test that embedding generation produces valid vectors."""
        assert sample_embeddings.ndim == 2, "Embeddings should be 2D"
        assert sample_embeddings.shape[0] == 10, "Should have 10 samples"
        assert sample_embeddings.shape[1] == 384, "Should have 384 dimensions"
        assert sample_embeddings.dtype == np.float32, "Should be float32"

        # Check that embeddings are not all zeros
        assert np.abs(sample_embeddings).sum() > 0, "Embeddings should not be all zeros"

    def test_embedding_dimensions(self, embeddings_dir: Path):
        """Test that stored embeddings have correct dimensions."""
        if not embeddings_dir.exists():
            pytest.skip("Embeddings directory does not exist")

        embedding_files = list(embeddings_dir.glob("*.npy"))
        if not embedding_files:
            pytest.skip("No embedding files found")

        for emb_file in embedding_files:
            embeddings = np.load(emb_file)
            assert embeddings.ndim == 2, f"{emb_file.name} should be 2D"
            # sentence-transformers typically produces 384 or 768 dim embeddings
            assert embeddings.shape[1] in [384, 768, 1024], \
                f"{emb_file.name} has unexpected dimension {embeddings.shape[1]}"

    def test_faiss_search(self, embeddings_dir: Path):
        """Test that FAISS index can perform similarity search."""
        try:
            import faiss
        except ImportError:
            pytest.skip("FAISS not installed")

        index_path = embeddings_dir / "faiss_index.bin"
        if not index_path.exists():
            pytest.skip("FAISS index file not found")

        # Load the index
        index = faiss.read_index(str(index_path))

        assert index.ntotal > 0, "FAISS index should contain vectors"

        # Get dimension from index
        dim = index.d

        # Create a random query vector
        query = np.random.randn(1, dim).astype(np.float32)

        # Perform search
        k = min(5, index.ntotal)
        distances, indices = index.search(query, k)

        assert distances.shape == (1, k), "Should return correct number of distances"
        assert indices.shape == (1, k), "Should return correct number of indices"
        assert all(idx >= 0 and idx < index.ntotal for idx in indices[0]), \
            "Indices should be valid"

    def test_embedding_metadata(self, embeddings_dir: Path):
        """Test that embedding metadata is valid JSON."""
        metadata_path = embeddings_dir / "metadata.json"

        if not metadata_path.exists():
            pytest.skip("Metadata file not found")

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        assert isinstance(metadata, (list, dict)), "Metadata should be list or dict"

        if isinstance(metadata, list):
            assert len(metadata) > 0, "Metadata should not be empty"


# ==============================================================================
# RL ENVIRONMENT TESTS
# ==============================================================================

class TestRLEnvironment:
    """Tests for the Mnemosyne RL environment."""

    @pytest.fixture
    def env(self):
        """Create a Mnemosyne environment instance."""
        from rl_training.mnemosyne_env import MnemosyneEnv
        env = MnemosyneEnv(host="localhost", port=27050, timeout=0.1)
        yield env
        env.close()

    def test_env_reset(self, env):
        """Test that environment reset returns valid observation and info."""
        obs, info = env.reset()

        assert isinstance(obs, np.ndarray), "Observation should be numpy array"
        assert obs.shape == (20,), f"Observation shape should be (20,), got {obs.shape}"
        assert obs.dtype == np.float32, "Observation should be float32"

        assert isinstance(info, dict), "Info should be a dictionary"
        assert "connected" in info, "Info should contain 'connected' key"

    def test_env_step(self, env):
        """Test that environment step returns valid outputs."""
        env.reset()

        # Take a random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(obs, np.ndarray), "Observation should be numpy array"
        assert obs.shape == (20,), "Observation shape should be (20,)"
        assert isinstance(reward, (int, float)), "Reward should be numeric"
        assert isinstance(terminated, bool), "Terminated should be boolean"
        assert isinstance(truncated, bool), "Truncated should be boolean"
        assert isinstance(info, dict), "Info should be a dictionary"

    def test_observation_space(self, env):
        """Test that observation space is correctly defined."""
        from gymnasium import spaces

        assert isinstance(env.observation_space, spaces.Box), \
            "Observation space should be Box"
        assert env.observation_space.shape == (20,), \
            f"Observation space shape should be (20,), got {env.observation_space.shape}"
        assert env.observation_space.dtype == np.float32, \
            "Observation space dtype should be float32"

        # Check bounds
        assert env.observation_space.low.min() == -1.0, "Lower bound should be -1.0"
        assert env.observation_space.high.max() == 1.0, "Upper bound should be 1.0"

    def test_action_space(self, env):
        """Test that action space is correctly defined."""
        from gymnasium import spaces
        from rl_training.mnemosyne_env import BotAction

        assert isinstance(env.action_space, spaces.Discrete), \
            "Action space should be Discrete"
        assert env.action_space.n == len(BotAction), \
            f"Action space should have {len(BotAction)} actions"

    def test_reward_range(self, env):
        """Test that rewards are within reasonable bounds."""
        env.reset()

        total_reward = 0
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Individual step rewards should be bounded
            assert -100 <= reward <= 150, f"Reward {reward} out of expected range"

            if terminated or truncated:
                break

    def test_game_state_default(self):
        """Test that default game state is valid."""
        from rl_training.mnemosyne_env import GameState

        state = GameState.default()

        assert state.health == 100, "Default health should be 100"
        assert state.is_alive is True, "Default should be alive"
        assert state.is_incapped is False, "Default should not be incapped"
        assert state.teammates_alive == 3, "Default should have 3 teammates"

    def test_game_state_to_observation(self):
        """Test that game state converts to valid observation."""
        from rl_training.mnemosyne_env import GameState

        state = GameState.default()
        obs = state.to_observation()

        assert isinstance(obs, np.ndarray), "Observation should be numpy array"
        assert obs.shape == (20,), "Observation should have 20 dimensions"
        assert obs.dtype == np.float32, "Observation should be float32"

        # Check normalization - most values should be in [-1, 1]
        assert np.all(obs >= -2) and np.all(obs <= 2), \
            "Observation values should be roughly normalized"


# ==============================================================================
# SECURITY TESTS
# ==============================================================================

class TestSecurity:
    """Tests for security utilities (path traversal, SSRF prevention)."""

    def test_safe_path_blocks_traversal(self, temp_dir: Path):
        """Test that safe_path blocks path traversal attempts."""
        from utils.security import safe_path
        import platform

        # These paths should trigger traversal detection on all platforms
        definitely_malicious = [
            "../../../etc/passwd",
            "/etc/passwd",
            "data/../../../secret.txt",
            "valid/../../../etc/shadow",
            "foo/bar/../../../../../../etc/hosts",
        ]

        # Windows-style paths only matter on Windows
        # (on Unix, backslashes are valid filename characters)
        if platform.system() == "Windows":
            definitely_malicious.append("..\\..\\..\\Windows\\System32\\config\\SAM")

        for malicious in definitely_malicious:
            with pytest.raises(ValueError, match="Path traversal|Invalid path"):
                safe_path(malicious, temp_dir)

    def test_safe_path_allows_valid(self, temp_dir: Path):
        """Test that safe_path allows valid paths."""
        from utils.security import safe_path

        # Create subdirectory
        subdir = temp_dir / "subdir"
        subdir.mkdir()

        valid_paths = [
            "file.txt",
            "subdir/file.txt",
            "data/processed/train.jsonl",
        ]

        for valid in valid_paths:
            result = safe_path(valid, temp_dir, create_parents=True)
            assert str(temp_dir) in str(result), \
                f"Path {result} should be within {temp_dir}"

    def test_url_validation(self, malicious_urls: List[str]):
        """Test that validate_url blocks malicious URLs."""
        from utils.security import validate_url

        for url in malicious_urls:
            with pytest.raises(ValueError):
                validate_url(url)

    def test_url_validation_allows_valid(self, valid_urls: List[str]):
        """Test that validate_url allows valid URLs."""
        from utils.security import validate_url

        for url in valid_urls:
            result = validate_url(url)
            assert result == url or result.startswith("https://"), \
                f"Valid URL {url} should be allowed"

    def test_safe_write(self, temp_dir: Path):
        """Test that safe_write functions work correctly."""
        from utils.security import safe_write_json, safe_write_text, safe_write_jsonl

        # Test JSON write
        json_data = {"key": "value", "number": 42}
        json_path = safe_write_json("test.json", json_data, temp_dir)
        assert json_path.exists(), "JSON file should be created"
        with open(json_path) as f:
            loaded = json.load(f)
        assert loaded == json_data, "JSON content should match"

        # Test text write
        text_content = "Hello, World!\nLine 2"
        text_path = safe_write_text("test.txt", text_content, temp_dir)
        assert text_path.exists(), "Text file should be created"
        with open(text_path) as f:
            loaded = f.read()
        assert loaded == text_content, "Text content should match"

        # Test JSONL write
        jsonl_data = [{"a": 1}, {"b": 2}, {"c": 3}]
        jsonl_path = safe_write_jsonl("test.jsonl", jsonl_data, temp_dir)
        assert jsonl_path.exists(), "JSONL file should be created"
        with open(jsonl_path) as f:
            loaded = [json.loads(line) for line in f]
        assert loaded == jsonl_data, "JSONL content should match"

    def test_safe_write_blocks_traversal(self, temp_dir: Path):
        """Test that safe write functions block path traversal."""
        from utils.security import safe_write_json, safe_write_text

        malicious_paths = [
            "../../../etc/evil.txt",
            "/etc/passwd",
            "data/../../secret.json",
        ]

        for malicious in malicious_paths:
            with pytest.raises(ValueError):
                safe_write_json(malicious, {"evil": True}, temp_dir)

            with pytest.raises(ValueError):
                safe_write_text(malicious, "evil content", temp_dir)

    def test_sanitize_filename(self):
        """Test filename sanitization."""
        from utils.security import sanitize_filename

        test_cases = [
            ("normal_file.txt", "normal_file.txt"),
            ("../../../etc/passwd", "_.._.._.._etc_passwd"),
            ("file\x00name.txt", "filename.txt"),
            ("   ...file...   ", "file"),
            ("", "unnamed"),
            ("a" * 300, "a" * 255),
        ]

        for input_name, expected_prefix in test_cases:
            result = sanitize_filename(input_name)
            assert "/" not in result, f"Result should not contain '/': {result}"
            assert "\\" not in result, f"Result should not contain '\\': {result}"
            assert "\x00" not in result, f"Result should not contain null byte: {result}"
            assert len(result) <= 255, f"Result should be <= 255 chars: {len(result)}"


# ==============================================================================
# CONFIG TESTS
# ==============================================================================

class TestConfigs:
    """Tests for configuration file validation."""

    def test_configs_valid_yaml(self, configs_dir: Path):
        """Test that all config files are valid YAML."""
        if not configs_dir.exists():
            pytest.skip("Configs directory does not exist")

        yaml_files = list(configs_dir.glob("*.yaml")) + list(configs_dir.glob("*.yml"))

        if not yaml_files:
            pytest.skip("No YAML config files found")

        for yaml_file in yaml_files:
            with open(yaml_file, "r", encoding="utf-8") as f:
                try:
                    config = yaml.safe_load(f)
                    assert config is not None, f"{yaml_file.name} loaded as None"
                    assert isinstance(config, dict), f"{yaml_file.name} should be a dict"
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML in {yaml_file.name}: {e}")

    def test_config_required_fields(self, configs_dir: Path):
        """Test that training configs have required fields."""
        if not configs_dir.exists():
            pytest.skip("Configs directory does not exist")

        # Find training configs
        training_configs = [
            f for f in configs_dir.glob("unsloth_config*.yaml")
        ]

        if not training_configs:
            pytest.skip("No unsloth config files found")

        required_sections = ["model", "lora", "training", "data", "output"]
        required_model_fields = ["name", "max_seq_length"]
        required_training_fields = ["num_train_epochs", "learning_rate"]
        required_data_fields = ["train_file"]

        for config_file in training_configs:
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # Check top-level sections
            for section in required_sections:
                assert section in config, \
                    f"{config_file.name} missing section: {section}"

            # Check model fields
            for field in required_model_fields:
                assert field in config["model"], \
                    f"{config_file.name} missing model.{field}"

            # Check training fields
            for field in required_training_fields:
                assert field in config["training"], \
                    f"{config_file.name} missing training.{field}"

            # Check data fields
            for field in required_data_fields:
                assert field in config["data"], \
                    f"{config_file.name} missing data.{field}"

    def test_config_values_valid(self, configs_dir: Path):
        """Test that config values are within valid ranges."""
        if not configs_dir.exists():
            pytest.skip("Configs directory does not exist")

        training_configs = list(configs_dir.glob("unsloth_config*.yaml"))

        if not training_configs:
            pytest.skip("No unsloth config files found")

        for config_file in training_configs:
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # Validate model settings
            assert config["model"]["max_seq_length"] > 0, \
                f"{config_file.name}: max_seq_length must be positive"
            assert config["model"]["max_seq_length"] <= 32768, \
                f"{config_file.name}: max_seq_length too large"

            # Validate LoRA settings
            assert config["lora"]["r"] > 0, \
                f"{config_file.name}: LoRA rank must be positive"
            assert config["lora"]["lora_alpha"] >= config["lora"]["r"], \
                f"{config_file.name}: lora_alpha should be >= r"
            assert 0 <= config["lora"]["lora_dropout"] < 1, \
                f"{config_file.name}: lora_dropout must be in [0, 1)"

            # Validate training settings
            assert config["training"]["num_train_epochs"] > 0, \
                f"{config_file.name}: epochs must be positive"
            assert config["training"]["learning_rate"] > 0, \
                f"{config_file.name}: learning_rate must be positive"
            assert config["training"]["learning_rate"] < 1, \
                f"{config_file.name}: learning_rate too large"

    def test_director_config_valid(self, configs_dir: Path):
        """Test that director config is valid."""
        director_config = configs_dir / "director_config.yaml"

        if not director_config.exists():
            pytest.skip("Director config not found")

        with open(director_config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        assert isinstance(config, dict), "Director config should be a dict"
        # Add more specific director config validations as needed


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.slow
    def test_full_data_pipeline(self, sample_jsonl_file: Path, temp_dir: Path):
        """Test loading, validating, and processing training data."""
        # Load data
        data = []
        with open(sample_jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))

        # Validate format
        for example in data:
            assert "messages" in example
            for msg in example["messages"]:
                assert "role" in msg
                assert "content" in msg

        # Write processed data
        from utils.security import safe_write_jsonl
        output_path = safe_write_jsonl("processed.jsonl", data, temp_dir)
        assert output_path.exists()

        # Verify round-trip
        with open(output_path, "r", encoding="utf-8") as f:
            reloaded = [json.loads(line) for line in f]

        assert len(reloaded) == len(data)
        assert reloaded[0] == data[0]

    @pytest.mark.slow
    def test_env_episode_simulation(self):
        """Test running a full episode in simulation mode."""
        from rl_training.mnemosyne_env import MnemosyneEnv

        env = MnemosyneEnv(timeout=0.01, max_episode_steps=100)
        obs, info = env.reset()

        total_reward = 0
        steps = 0

        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        env.close()

        assert steps > 0, "Should complete at least one step"
        assert "episode_reward" in info, "Info should contain episode_reward"


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
