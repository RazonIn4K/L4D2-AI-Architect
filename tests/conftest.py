#!/usr/bin/env python3
"""
Pytest Fixtures and Configuration for L4D2-AI-Architect Test Suite

This module provides shared fixtures for testing all major components
of the L4D2 AI training system.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List

import pytest
import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Import project modules
from utils.security import safe_path, validate_url, safe_write_json, safe_write_text


# ==============================================================================
# Path Fixtures
# ==============================================================================

@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def data_dir(project_root: Path) -> Path:
    """Return the data directory path."""
    return project_root / "data"


@pytest.fixture
def processed_data_dir(data_dir: Path) -> Path:
    """Return the processed data directory path."""
    return data_dir / "processed"


@pytest.fixture
def configs_dir(project_root: Path) -> Path:
    """Return the configs directory path."""
    return project_root / "configs"


@pytest.fixture
def embeddings_dir(data_dir: Path) -> Path:
    """Return the embeddings directory path."""
    return data_dir / "embeddings"


# ==============================================================================
# Temporary Directory Fixtures
# ==============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_data_dir(temp_dir: Path) -> Path:
    """Create a temporary data directory structure."""
    data_path = temp_dir / "data" / "processed"
    data_path.mkdir(parents=True, exist_ok=True)
    return data_path


# ==============================================================================
# Sample Data Fixtures
# ==============================================================================

@pytest.fixture
def sample_training_example() -> Dict[str, Any]:
    """Return a sample ChatML training example."""
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are an expert SourcePawn developer specializing in Left 4 Dead 2 plugins."
            },
            {
                "role": "user",
                "content": "Write a function to heal all survivors to full health."
            },
            {
                "role": "assistant",
                "content": '''public void HealAllSurvivors()
{
    for (int i = 1; i <= MaxClients; i++)
    {
        if (IsClientInGame(i) && GetClientTeam(i) == 2 && IsPlayerAlive(i))
        {
            SetEntProp(i, Prop_Send, "m_iHealth", 100);
            SetEntProp(i, Prop_Send, "m_currentReviveCount", 0);
            SetEntProp(i, Prop_Send, "m_bIsOnThirdStrike", false);
        }
    }
}'''
            }
        ]
    }


@pytest.fixture
def sample_training_data(sample_training_example: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a list of sample training examples."""
    examples = [sample_training_example]

    # Add more diverse examples
    examples.append({
        "messages": [
            {
                "role": "system",
                "content": "You are an expert VScript developer for Left 4 Dead 2."
            },
            {
                "role": "user",
                "content": "Create a function to spawn a Tank at a specific location."
            },
            {
                "role": "assistant",
                "content": '''function SpawnTankAtLocation(pos)
{
    local spawn_flags = 0;
    local tank = SpawnEntityFromTable("infected", {
        classname = "infected",
        targetname = "spawned_tank"
    });

    if (tank)
    {
        tank.SetOrigin(pos);
        Entities.FindByName(null, "director").SetSpawnReason(tank, 0);
    }
    return tank;
}'''
            }
        ]
    })

    examples.append({
        "messages": [
            {
                "role": "system",
                "content": "You are an expert SourcePawn developer."
            },
            {
                "role": "user",
                "content": "Write a timer callback to check player positions."
            },
            {
                "role": "assistant",
                "content": '''public Action Timer_CheckPositions(Handle timer)
{
    for (int i = 1; i <= MaxClients; i++)
    {
        if (IsClientInGame(i) && IsPlayerAlive(i))
        {
            float pos[3];
            GetClientAbsOrigin(i, pos);
            PrintToServer("Player %d at %.1f, %.1f, %.1f", i, pos[0], pos[1], pos[2]);
        }
    }
    return Plugin_Continue;
}'''
            }
        ]
    })

    return examples


@pytest.fixture
def sample_jsonl_file(temp_data_dir: Path, sample_training_data: List[Dict[str, Any]]) -> Path:
    """Create a temporary JSONL file with sample training data."""
    file_path = temp_data_dir / "test_train.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for example in sample_training_data:
            f.write(json.dumps(example) + "\n")
    return file_path


# ==============================================================================
# Config Fixtures
# ==============================================================================

@pytest.fixture
def sample_config() -> Dict[str, Any]:
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
def sample_config_file(temp_dir: Path, sample_config: Dict[str, Any]) -> Path:
    """Create a temporary YAML config file."""
    import yaml

    config_path = temp_dir / "test_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(sample_config, f)
    return config_path


# ==============================================================================
# Embedding Fixtures
# ==============================================================================

@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Return sample embedding vectors."""
    # 10 samples with 384-dimensional embeddings (sentence-transformers default)
    np.random.seed(42)
    return np.random.randn(10, 384).astype(np.float32)


@pytest.fixture
def sample_embedding_metadata() -> List[Dict[str, str]]:
    """Return sample metadata for embeddings."""
    return [
        {"id": f"sample_{i}", "prompt": f"Test prompt {i}", "language": "sourcepawn"}
        for i in range(10)
    ]


# ==============================================================================
# RL Environment Fixtures
# ==============================================================================

@pytest.fixture
def mock_game_state() -> Dict[str, Any]:
    """Return a mock game state dictionary."""
    return {
        "bot_id": 0,
        "health": 100,
        "is_alive": True,
        "is_incapped": False,
        "position": (0.0, 0.0, 0.0),
        "velocity": (0.0, 0.0, 0.0),
        "angle": (0.0, 0.0),
        "primary_weapon": 5,
        "secondary_weapon": 1,
        "throwable": 0,
        "health_item": 1,
        "ammo": 50,
        "nearby_enemies": 2,
        "nearest_enemy_dist": 500.0,
        "nearest_teammate_dist": 100.0,
        "teammates_alive": 3,
        "teammates_incapped": 0,
        "in_safe_room": False,
        "near_objective": False
    }


# ==============================================================================
# Security Test Fixtures
# ==============================================================================

@pytest.fixture
def malicious_paths() -> List[str]:
    """Return a list of malicious path traversal attempts."""
    return [
        "../../../etc/passwd",
        "..\\..\\..\\Windows\\System32\\config\\SAM",
        "/etc/passwd",
        "data/../../../secret.txt",
        "valid/../../../etc/shadow",
        "foo/bar/../../../../../../etc/hosts",
        "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "....//....//....//etc/passwd",
    ]


@pytest.fixture
def malicious_urls() -> List[str]:
    """Return a list of malicious SSRF URLs."""
    return [
        "http://localhost:8080/admin",
        "http://127.0.0.1:22/",
        "http://192.168.1.1/",
        "http://10.0.0.1/internal",
        "http://169.254.169.254/latest/meta-data/",
        "file:///etc/passwd",
        "ftp://evil.com/malware",
        "http://evil.com/attack",
    ]


@pytest.fixture
def valid_urls() -> List[str]:
    """Return a list of valid allowed URLs."""
    return [
        "https://api.github.com/repos/owner/repo",
        "https://raw.githubusercontent.com/owner/repo/main/file.txt",
        "https://developer.valvesoftware.com/wiki/L4D2_SDK",
        "https://wiki.alliedmods.net/SourceMod",
    ]


# ==============================================================================
# Pytest Configuration
# ==============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
