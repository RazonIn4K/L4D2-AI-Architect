#!/usr/bin/env python3
"""
Continuous Training System for L4D2-AI-Architect

Monitors for new training data and automatically triggers retraining,
manages model versions with semantic versioning, and deploys new models.

Features:
- File watcher for new training data
- Configurable retraining triggers (threshold, schedule, manual)
- Semantic versioning with changelog generation
- Rollback capability
- GGUF export and Ollama deployment
- API hot-reload support

Usage:
    python continuous_training.py watch --data-dir data/processed
    python continuous_training.py train-if-needed --min-new-examples 100
    python continuous_training.py status
    python continuous_training.py rollback --version 1.2.0
    python continuous_training.py deploy --version 1.3.0
"""

import os
import sys
import json
import argparse
import logging
import hashlib
import shutil
import subprocess
import time
import signal
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum

# Add parent to path for security utils and training scripts
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import (
    safe_path, safe_read_json, safe_write_json, safe_read_yaml, safe_write_text
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "model_adapters"
EXPORTS_DIR = PROJECT_ROOT / "exports"
CONFIG_DIR = PROJECT_ROOT / "configs"
STATE_DIR = PROJECT_ROOT / "data" / "continuous_training"


class TrainingTrigger(Enum):
    """Reasons for triggering training."""
    NEW_DATA = "new_data"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    DATA_CHANGE = "data_change"


@dataclass
class DatasetState:
    """Tracks the state of training data."""
    file_path: str
    line_count: int
    file_hash: str
    last_modified: str

    @classmethod
    def from_file(cls, file_path: Path) -> "DatasetState":
        """Create state from file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")

        # Count lines
        line_count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in f:
                line_count += 1

        # Compute hash of first 100 lines + last 100 lines for efficiency
        hasher = hashlib.sha256()
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            sample = lines[:100] + lines[-100:] if len(lines) > 200 else lines
            for line in sample:
                hasher.update(line.encode('utf-8'))

        return cls(
            file_path=str(file_path),
            line_count=line_count,
            file_hash=hasher.hexdigest()[:16],
            last_modified=datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        )


@dataclass
class ModelVersion:
    """Represents a model version."""
    major: int = 1
    minor: int = 0
    patch: int = 0

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    @classmethod
    def parse(cls, version_str: str) -> "ModelVersion":
        """Parse version string."""
        parts = version_str.replace("v", "").split(".")
        return cls(
            major=int(parts[0]) if len(parts) > 0 else 1,
            minor=int(parts[1]) if len(parts) > 1 else 0,
            patch=int(parts[2]) if len(parts) > 2 else 0
        )

    def bump_major(self) -> "ModelVersion":
        return ModelVersion(self.major + 1, 0, 0)

    def bump_minor(self) -> "ModelVersion":
        return ModelVersion(self.major, self.minor + 1, 0)

    def bump_patch(self) -> "ModelVersion":
        return ModelVersion(self.major, self.minor, self.patch + 1)


@dataclass
class TrainingRecord:
    """Record of a training run."""
    version: str
    trigger: str
    started_at: str
    completed_at: Optional[str] = None
    status: str = "pending"
    dataset_state: Optional[Dict[str, Any]] = None
    training_config: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    model_path: Optional[str] = None
    export_path: Optional[str] = None
    changelog: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ContinuousTrainingState:
    """Persistent state for continuous training."""
    current_version: str = "1.0.0"
    last_dataset_state: Optional[Dict[str, Any]] = None
    last_training_time: Optional[str] = None
    training_history: List[Dict[str, Any]] = field(default_factory=list)
    active_model: Optional[str] = None
    deployed_ollama_model: Optional[str] = None

    @classmethod
    def load(cls, state_file: Path) -> "ContinuousTrainingState":
        """Load state from file."""
        if state_file.exists():
            try:
                data = safe_read_json(str(state_file), PROJECT_ROOT)
                return cls(**data)
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
        return cls()

    def save(self, state_file: Path) -> None:
        """Save state to file."""
        state_file.parent.mkdir(parents=True, exist_ok=True)
        safe_write_json(str(state_file), asdict(self), PROJECT_ROOT)


class ContinuousTrainingSystem:
    """Main system for continuous training management."""

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        model_dir: Path = MODEL_DIR,
        config_path: Optional[Path] = None,
        min_new_examples: int = 100,
        check_interval: int = 300,  # 5 minutes
        schedule_hours: int = 24,   # Daily training
    ):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.config_path = config_path or CONFIG_DIR / "unsloth_config.yaml"
        self.min_new_examples = min_new_examples
        self.check_interval = check_interval
        self.schedule_hours = schedule_hours

        # State management
        self.state_file = STATE_DIR / "state.json"
        self.state = ContinuousTrainingState.load(self.state_file)

        # Watch control
        self._stop_watching = threading.Event()
        self._watch_thread: Optional[threading.Thread] = None

    def get_dataset_files(self) -> List[Path]:
        """Get monitored dataset files."""
        patterns = ["combined_train.jsonl", "*_train.jsonl", "l4d2_train_*.jsonl"]
        files = []
        for pattern in patterns:
            files.extend(self.data_dir.glob(pattern))
        return sorted(set(files), key=lambda p: p.stat().st_mtime, reverse=True)

    def get_current_dataset_state(self) -> Dict[str, DatasetState]:
        """Get current state of all monitored datasets."""
        states = {}
        for file_path in self.get_dataset_files():
            try:
                states[file_path.name] = DatasetState.from_file(file_path)
            except Exception as e:
                logger.warning(f"Failed to get state for {file_path}: {e}")
        return states

    def check_data_changes(self) -> Tuple[bool, int, str]:
        """
        Check for data changes since last training.

        Returns:
            Tuple of (has_significant_changes, new_example_count, change_description)
        """
        current_states = self.get_current_dataset_state()

        if not self.state.last_dataset_state:
            # First run - consider all data as new
            total_lines = sum(s.line_count for s in current_states.values())
            return True, total_lines, "Initial training data"

        last_states = {
            k: DatasetState(**v) if isinstance(v, dict) else v
            for k, v in self.state.last_dataset_state.items()
        }

        new_examples = 0
        changes = []

        for name, current in current_states.items():
            if name not in last_states:
                new_examples += current.line_count
                changes.append(f"New dataset: {name} ({current.line_count} examples)")
            else:
                last = last_states[name]
                if isinstance(last, dict):
                    last = DatasetState(**last)

                if current.file_hash != last.file_hash:
                    diff = current.line_count - last.line_count
                    new_examples += max(0, diff)
                    if diff > 0:
                        changes.append(f"{name}: +{diff} examples")
                    elif diff < 0:
                        changes.append(f"{name}: {diff} examples (reduced)")
                    else:
                        changes.append(f"{name}: content changed (same size)")

        has_significant = new_examples >= self.min_new_examples
        description = "; ".join(changes) if changes else "No changes"

        return has_significant, new_examples, description

    def should_train(self) -> Tuple[bool, TrainingTrigger, str]:
        """
        Determine if training should be triggered.

        Returns:
            Tuple of (should_train, trigger_reason, description)
        """
        # Check for data changes
        has_changes, new_count, change_desc = self.check_data_changes()
        if has_changes:
            return True, TrainingTrigger.NEW_DATA, f"{new_count} new examples: {change_desc}"

        # Check scheduled training
        if self.state.last_training_time:
            last_training = datetime.fromisoformat(self.state.last_training_time)
            hours_since = (datetime.now() - last_training).total_seconds() / 3600
            if hours_since >= self.schedule_hours:
                return True, TrainingTrigger.SCHEDULED, f"{hours_since:.1f} hours since last training"

        return False, TrainingTrigger.MANUAL, "No trigger condition met"

    def generate_changelog(
        self,
        version: ModelVersion,
        trigger: TrainingTrigger,
        dataset_state: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate changelog entry for this version."""
        lines = [
            f"## Version {version}",
            f"",
            f"**Release Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Trigger:** {trigger.value}",
            f"",
            "### Training Data",
        ]

        for name, state in dataset_state.items():
            if isinstance(state, dict):
                lines.append(f"- {name}: {state.get('line_count', '?')} examples")
            else:
                lines.append(f"- {name}: {state.line_count} examples")

        if metrics:
            lines.extend([
                "",
                "### Metrics",
            ])
            for key, value in metrics.items():
                lines.append(f"- {key}: {value}")

        lines.extend([
            "",
            "---",
            ""
        ])

        return "\n".join(lines)

    def run_training(
        self,
        trigger: TrainingTrigger,
        bump_type: str = "patch",
        dataset_file: Optional[str] = None,
    ) -> TrainingRecord:
        """
        Execute a training run.

        Args:
            trigger: Reason for training
            bump_type: Version bump type (major, minor, patch)
            dataset_file: Specific dataset file to use

        Returns:
            TrainingRecord with results
        """
        # Determine new version
        current = ModelVersion.parse(self.state.current_version)
        if bump_type == "major":
            new_version = current.bump_major()
        elif bump_type == "minor":
            new_version = current.bump_minor()
        else:
            new_version = current.bump_patch()

        # Create training record
        record = TrainingRecord(
            version=str(new_version),
            trigger=trigger.value,
            started_at=datetime.now().isoformat(),
            dataset_state={
                k: asdict(v) for k, v in self.get_current_dataset_state().items()
            }
        )

        logger.info(f"Starting training v{new_version} (trigger: {trigger.value})")

        try:
            # Load and modify config
            if self.config_path.exists():
                config = safe_read_yaml(str(self.config_path), PROJECT_ROOT)
            else:
                config = self._get_default_config()

            # Set output directory with version
            version_dir = f"l4d2-code-v{new_version}"
            config["output"]["dir"] = version_dir

            # Override dataset if specified (validate path to prevent traversal)
            if dataset_file:
                validated_dataset = safe_path(dataset_file, PROJECT_ROOT)
                config["data"]["train_file"] = str(validated_dataset)

            record.training_config = config

            # Save temporary config
            temp_config = STATE_DIR / f"config_v{new_version}.yaml"
            temp_config.parent.mkdir(parents=True, exist_ok=True)

            import yaml
            with open(temp_config, 'w') as f:
                yaml.dump(config, f)

            # Run training script
            train_script = Path(__file__).parent / "train_unsloth.py"

            cmd = [
                sys.executable,
                str(train_script),
                "--config", str(temp_config)
            ]

            logger.info(f"Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT)
            )

            if result.returncode != 0:
                record.status = "failed"
                record.error = result.stderr[-2000:] if result.stderr else "Unknown error"
                logger.error(f"Training failed: {record.error}")
            else:
                record.status = "completed"
                record.model_path = str(self.model_dir / version_dir / "final")

                # Extract metrics from training output
                record.metrics = self._parse_training_metrics(result.stdout)

                # Generate changelog
                record.changelog = self.generate_changelog(
                    new_version, trigger, record.dataset_state, record.metrics
                )

                # Update state
                self.state.current_version = str(new_version)
                self.state.last_training_time = datetime.now().isoformat()
                self.state.last_dataset_state = record.dataset_state
                self.state.active_model = record.model_path

                logger.info(f"Training completed: v{new_version}")

        except Exception as e:
            record.status = "failed"
            record.error = str(e)
            logger.error(f"Training error: {e}")

        record.completed_at = datetime.now().isoformat()

        # Save record to history
        self.state.training_history.append(asdict(record))
        self.state.save(self.state_file)

        # Save changelog
        if record.changelog:
            changelog_file = STATE_DIR / "CHANGELOG.md"
            existing = ""
            if changelog_file.exists():
                existing = changelog_file.read_text()
            safe_write_text(
                str(changelog_file),
                record.changelog + existing,
                PROJECT_ROOT
            )

        return record

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default training configuration."""
        return {
            "model": {
                "name": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
                "max_seq_length": 2048,
                "dtype": None,
                "load_in_4bit": True,
            },
            "lora": {
                "r": 32,
                "lora_alpha": 64,
                "lora_dropout": 0,
                "target_modules": [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                "bias": "none",
                "use_gradient_checkpointing": "unsloth",
                "use_rslora": False,
            },
            "training": {
                "num_train_epochs": 3,
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "learning_rate": 2e-4,
                "weight_decay": 0.01,
                "warmup_steps": 10,
                "lr_scheduler_type": "linear",
                "optim": "adamw_8bit",
                "fp16": False,
                "bf16": True,
                "logging_steps": 10,
                "save_steps": 100,
                "save_total_limit": 3,
                "seed": 3407,
            },
            "data": {
                "train_file": "combined_train.jsonl",
                "val_file": "combined_val.jsonl",
                "max_samples": None,
            },
            "output": {
                "dir": "l4d2-code-lora",
                "push_to_hub": False,
                "hub_model_id": None,
            }
        }

    def _parse_training_metrics(self, output: str) -> Dict[str, Any]:
        """Parse training metrics from output."""
        metrics = {}

        # Look for common metric patterns
        import re

        # Training time
        time_match = re.search(r"Training completed in (\S+)", output)
        if time_match:
            metrics["training_time"] = time_match.group(1)

        # Loss values
        loss_matches = re.findall(r"'loss':\s*([\d.]+)", output)
        if loss_matches:
            metrics["final_loss"] = float(loss_matches[-1])
            metrics["initial_loss"] = float(loss_matches[0])

        # Sample counts
        sample_match = re.search(r"Loaded (\d+) examples", output)
        if sample_match:
            metrics["train_samples"] = int(sample_match.group(1))

        return metrics

    def export_model(
        self,
        version: Optional[str] = None,
        quantization: str = "q4_k_m"
    ) -> Optional[Path]:
        """
        Export model to GGUF format.

        Args:
            version: Specific version to export (default: current)
            quantization: GGUF quantization method

        Returns:
            Path to exported GGUF or None if failed
        """
        version = version or self.state.current_version

        # Find model path
        model_path = None
        for record in reversed(self.state.training_history):
            if record.get("version") == version and record.get("model_path"):
                model_path = Path(record["model_path"])
                break

        if not model_path or not model_path.exists():
            # Try to find by version pattern
            pattern = f"l4d2-code-v{version}"
            candidates = list(self.model_dir.glob(f"{pattern}*/final"))
            if candidates:
                model_path = candidates[0]

        if not model_path or not model_path.exists():
            logger.error(f"Model not found for version {version}")
            return None

        logger.info(f"Exporting model v{version} to GGUF ({quantization})")

        export_script = Path(__file__).parent / "export_gguf_cpu.py"
        output_dir = EXPORTS_DIR / f"l4d2-v{version}"

        cmd = [
            sys.executable,
            str(export_script),
            "--adapter", str(model_path),
            "--output", str(output_dir),
            "--quantize", quantization,
            "--create-modelfile"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT)
            )

            if result.returncode == 0:
                logger.info(f"Export completed: {output_dir}")

                # Update record
                for record in self.state.training_history:
                    if record.get("version") == version:
                        record["export_path"] = str(output_dir)
                        break
                self.state.save(self.state_file)

                return output_dir
            else:
                logger.error(f"Export failed: {result.stderr}")
                return None

        except Exception as e:
            logger.error(f"Export error: {e}")
            return None

    def deploy_to_ollama(
        self,
        version: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> bool:
        """
        Deploy model to Ollama.

        Args:
            version: Version to deploy (default: current)
            model_name: Ollama model name (default: l4d2-code-v{version})

        Returns:
            True if deployment succeeded
        """
        version = version or self.state.current_version
        model_name = model_name or f"l4d2-code-v{version}"

        # Find export path
        export_path = None
        for record in reversed(self.state.training_history):
            if record.get("version") == version and record.get("export_path"):
                export_path = Path(record["export_path"])
                break

        if not export_path:
            export_path = EXPORTS_DIR / f"l4d2-v{version}"

        # Check for GGUF
        gguf_dir = export_path / "gguf" if export_path.exists() else None
        if not gguf_dir or not gguf_dir.exists():
            logger.info("GGUF not found, exporting first...")
            export_path = self.export_model(version)
            if not export_path:
                return False
            gguf_dir = export_path / "gguf"

        modelfile = gguf_dir / "Modelfile"
        if not modelfile.exists():
            logger.error(f"Modelfile not found at {modelfile}")
            return False

        # Check Ollama availability
        if not shutil.which("ollama"):
            logger.error("Ollama not installed")
            return False

        logger.info(f"Deploying to Ollama as '{model_name}'")

        try:
            result = subprocess.run(
                ["ollama", "create", model_name, "-f", str(modelfile)],
                cwd=str(gguf_dir),
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                logger.info(f"Deployed: ollama run {model_name}")
                self.state.deployed_ollama_model = model_name
                self.state.save(self.state_file)
                return True
            else:
                logger.error(f"Ollama create failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Deployment error: {e}")
            return False

    def rollback(self, target_version: str) -> bool:
        """
        Rollback to a previous version.

        Args:
            target_version: Version to rollback to

        Returns:
            True if rollback succeeded
        """
        # Find the version in history
        target_record = None
        for record in self.state.training_history:
            if record.get("version") == target_version:
                target_record = record
                break

        if not target_record:
            logger.error(f"Version {target_version} not found in history")
            return False

        model_path = target_record.get("model_path")
        if not model_path or not Path(model_path).exists():
            logger.error(f"Model files for v{target_version} not found")
            return False

        logger.info(f"Rolling back to v{target_version}")

        # Update state
        self.state.current_version = target_version
        self.state.active_model = model_path
        self.state.save(self.state_file)

        # Redeploy if we had an Ollama model
        if self.state.deployed_ollama_model:
            self.deploy_to_ollama(target_version)

        logger.info(f"Rollback complete: now running v{target_version}")
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        has_changes, new_count, change_desc = self.check_data_changes()
        should, trigger, trigger_desc = self.should_train()

        return {
            "current_version": self.state.current_version,
            "active_model": self.state.active_model,
            "deployed_ollama_model": self.state.deployed_ollama_model,
            "last_training": self.state.last_training_time,
            "training_count": len(self.state.training_history),
            "data_changes": {
                "has_significant_changes": has_changes,
                "new_example_count": new_count,
                "description": change_desc,
            },
            "next_training": {
                "should_train": should,
                "trigger": trigger.value,
                "description": trigger_desc,
            },
            "monitored_datasets": list(self.get_current_dataset_state().keys()),
            "min_new_examples_threshold": self.min_new_examples,
            "schedule_hours": self.schedule_hours,
        }

    def get_version_history(self) -> List[Dict[str, Any]]:
        """Get version history with details."""
        history = []
        for record in self.state.training_history:
            history.append({
                "version": record.get("version"),
                "status": record.get("status"),
                "trigger": record.get("trigger"),
                "started_at": record.get("started_at"),
                "completed_at": record.get("completed_at"),
                "model_path": record.get("model_path"),
                "export_path": record.get("export_path"),
                "metrics": record.get("metrics"),
            })
        return history

    def watch(self, daemon: bool = False) -> None:
        """
        Start watching for changes and trigger training.

        Args:
            daemon: Run as background daemon
        """
        logger.info(f"Starting continuous training watch")
        logger.info(f"  Data dir: {self.data_dir}")
        logger.info(f"  Check interval: {self.check_interval}s")
        logger.info(f"  Min new examples: {self.min_new_examples}")
        logger.info(f"  Schedule: every {self.schedule_hours}h")

        def watch_loop():
            while not self._stop_watching.is_set():
                try:
                    should, trigger, desc = self.should_train()

                    if should:
                        logger.info(f"Training triggered: {desc}")
                        self.run_training(trigger)

                        # Auto-export and deploy
                        if self.export_model():
                            self.deploy_to_ollama()

                except Exception as e:
                    logger.error(f"Watch loop error: {e}")

                # Wait for next check
                self._stop_watching.wait(self.check_interval)

        # Signal handling for graceful shutdown
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal")
            self._stop_watching.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        if daemon:
            self._watch_thread = threading.Thread(target=watch_loop, daemon=True)
            self._watch_thread.start()
            logger.info("Watch thread started in background")
        else:
            watch_loop()

    def stop_watching(self) -> None:
        """Stop the watch loop."""
        self._stop_watching.set()
        if self._watch_thread:
            self._watch_thread.join(timeout=5)

    def trigger_api_reload(self, api_url: str = "http://localhost:8000") -> bool:
        """
        Signal the API server to reload the model.

        Args:
            api_url: Base URL of the copilot API

        Returns:
            True if reload succeeded
        """
        try:
            import requests
            response = requests.post(
                f"{api_url}/v1/reload",
                json={"model_path": self.state.active_model},
                timeout=30
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"API reload failed: {e}")
            return False


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Continuous Training System for L4D2-AI-Architect",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Watch for new data and auto-train
  python continuous_training.py watch --data-dir data/processed

  # Check if training is needed
  python continuous_training.py train-if-needed --min-new-examples 100

  # Manual training with version bump
  python continuous_training.py train --bump minor

  # Show system status
  python continuous_training.py status

  # Export and deploy latest model
  python continuous_training.py deploy

  # Rollback to previous version
  python continuous_training.py rollback --version 1.2.0

  # Show version history
  python continuous_training.py history
        """
    )

    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR),
                       help="Training data directory")
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR),
                       help="Model output directory")
    parser.add_argument("--config", type=str,
                       help="Training config YAML file")
    parser.add_argument("--min-new-examples", type=int, default=100,
                       help="Minimum new examples to trigger training")
    parser.add_argument("--check-interval", type=int, default=300,
                       help="Check interval in seconds (for watch mode)")
    parser.add_argument("--schedule-hours", type=int, default=24,
                       help="Hours between scheduled training")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Log level")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Watch command
    watch_parser = subparsers.add_parser("watch", help="Watch for changes and auto-train")
    watch_parser.add_argument("--daemon", action="store_true",
                             help="Run in background")

    # Train-if-needed command
    train_if_parser = subparsers.add_parser("train-if-needed",
                                            help="Train if conditions are met")
    train_if_parser.add_argument("--force", action="store_true",
                                help="Force training even if no changes")
    train_if_parser.add_argument("--bump", choices=["major", "minor", "patch"],
                                default="patch", help="Version bump type")

    # Train command
    train_parser = subparsers.add_parser("train", help="Trigger manual training")
    train_parser.add_argument("--bump", choices=["major", "minor", "patch"],
                             default="patch", help="Version bump type")
    train_parser.add_argument("--dataset", type=str,
                             help="Specific dataset file to use")

    # Status command
    subparsers.add_parser("status", help="Show system status")

    # History command
    subparsers.add_parser("history", help="Show version history")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export model to GGUF")
    export_parser.add_argument("--version", type=str, help="Version to export")
    export_parser.add_argument("--quantize", default="q4_k_m",
                              choices=["f16", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q4_k_s", "q3_k_m", "q2_k"],
                              help="Quantization method")

    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy to Ollama")
    deploy_parser.add_argument("--version", type=str, help="Version to deploy")
    deploy_parser.add_argument("--name", type=str, help="Ollama model name")

    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback to previous version")
    rollback_parser.add_argument("--version", type=str, required=True,
                                help="Version to rollback to")

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Initialize system
    config_path = Path(args.config) if args.config else None
    system = ContinuousTrainingSystem(
        data_dir=Path(args.data_dir),
        model_dir=Path(args.model_dir),
        config_path=config_path,
        min_new_examples=args.min_new_examples,
        check_interval=args.check_interval,
        schedule_hours=args.schedule_hours,
    )

    # Execute command
    if args.command == "watch":
        system.watch(daemon=args.daemon)

    elif args.command == "train-if-needed":
        should, trigger, desc = system.should_train()
        if should or args.force:
            if args.force:
                trigger = TrainingTrigger.MANUAL
                desc = "Forced manual training"
            print(f"Training triggered: {desc}")
            record = system.run_training(trigger, bump_type=args.bump)
            print(f"Training {record.status}: v{record.version}")
            if record.status == "completed":
                system.export_model()
                system.deploy_to_ollama()
        else:
            print(f"No training needed: {desc}")

    elif args.command == "train":
        record = system.run_training(
            TrainingTrigger.MANUAL,
            bump_type=args.bump,
            dataset_file=args.dataset
        )
        print(f"Training {record.status}: v{record.version}")
        if record.error:
            print(f"Error: {record.error}")

    elif args.command == "status":
        status = system.get_status()
        print("\n=== Continuous Training Status ===\n")
        print(f"Current Version: v{status['current_version']}")
        print(f"Active Model: {status['active_model'] or 'None'}")
        print(f"Ollama Model: {status['deployed_ollama_model'] or 'None'}")
        print(f"Last Training: {status['last_training'] or 'Never'}")
        print(f"Training Runs: {status['training_count']}")
        print()
        print("Data Changes:")
        dc = status['data_changes']
        print(f"  New examples: {dc['new_example_count']}")
        print(f"  Significant: {'Yes' if dc['has_significant_changes'] else 'No'}")
        print(f"  Details: {dc['description']}")
        print()
        print("Next Training:")
        nt = status['next_training']
        print(f"  Should train: {'Yes' if nt['should_train'] else 'No'}")
        print(f"  Trigger: {nt['trigger']}")
        print(f"  Reason: {nt['description']}")
        print()
        print("Configuration:")
        print(f"  Min new examples: {status['min_new_examples_threshold']}")
        print(f"  Schedule: every {status['schedule_hours']} hours")
        print(f"  Monitored datasets: {', '.join(status['monitored_datasets'])}")

    elif args.command == "history":
        history = system.get_version_history()
        if not history:
            print("No training history found")
        else:
            print("\n=== Version History ===\n")
            for entry in reversed(history):
                status_icon = "[ok]" if entry['status'] == "completed" else "[!!]"
                print(f"{status_icon} v{entry['version']} - {entry['trigger']} - {entry['started_at']}")
                if entry['metrics']:
                    metrics_str = ", ".join(f"{k}={v}" for k, v in entry['metrics'].items())
                    print(f"     Metrics: {metrics_str}")
                if entry['model_path']:
                    print(f"     Model: {entry['model_path']}")
                if entry['export_path']:
                    print(f"     Export: {entry['export_path']}")
                print()

    elif args.command == "export":
        result = system.export_model(
            version=args.version,
            quantization=args.quantize
        )
        if result:
            print(f"Exported to: {result}")
        else:
            print("Export failed")
            sys.exit(1)

    elif args.command == "deploy":
        success = system.deploy_to_ollama(
            version=args.version,
            model_name=args.name
        )
        if success:
            print("Deployment successful")
        else:
            print("Deployment failed")
            sys.exit(1)

    elif args.command == "rollback":
        success = system.rollback(args.version)
        if success:
            print(f"Rolled back to v{args.version}")
        else:
            print("Rollback failed")
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
