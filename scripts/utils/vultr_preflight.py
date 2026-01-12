#!/usr/bin/env python3
"""
Vultr Pre-Training Validation Script

Run this ON THE VULTR INSTANCE before starting training to verify:
1. CUDA/GPU availability and memory
2. Training data files exist and are valid
3. Config file is valid YAML
4. Sufficient disk space
5. PyTorch can use GPU
6. Estimated training time based on config

Designed to be FAST (< 5 seconds) with clear PASS/FAIL output.

Usage:
    python scripts/utils/vultr_preflight.py
    python scripts/utils/vultr_preflight.py --config configs/unsloth_config_a100.yaml
    python scripts/utils/vultr_preflight.py --verbose
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Terminal colors
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

# Check result constants
PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
SKIP = "SKIP"

def status_icon(status: str) -> str:
    """Return colored status icon."""
    icons = {
        PASS: f"{Colors.GREEN}[PASS]{Colors.RESET}",
        FAIL: f"{Colors.RED}[FAIL]{Colors.RESET}",
        WARN: f"{Colors.YELLOW}[WARN]{Colors.RESET}",
        SKIP: f"{Colors.BLUE}[SKIP]{Colors.RESET}",
    }
    return icons.get(status, status)


class PreflightCheck:
    """Individual preflight check result."""

    def __init__(self, name: str, status: str, message: str, details: Optional[str] = None):
        self.name = name
        self.status = status
        self.message = message
        self.details = details

    def __str__(self) -> str:
        line = f"{status_icon(self.status)} {self.name}: {self.message}"
        if self.details:
            line += f"\n       {self.details}"
        return line


def check_cuda_available() -> PreflightCheck:
    """Check if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            return PreflightCheck(
                "CUDA Available",
                PASS,
                f"CUDA {torch.version.cuda} detected"
            )
        else:
            return PreflightCheck(
                "CUDA Available",
                FAIL,
                "CUDA not available - training will be extremely slow",
                "Install CUDA toolkit or check nvidia-smi"
            )
    except ImportError:
        return PreflightCheck(
            "CUDA Available",
            FAIL,
            "PyTorch not installed",
            "Run: pip install torch"
        )


def check_gpu_info() -> PreflightCheck:
    """Check GPU details and memory."""
    try:
        import torch
        if not torch.cuda.is_available():
            return PreflightCheck("GPU Info", SKIP, "No GPU available")

        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        # Determine if GPU is adequate
        if gpu_memory_gb >= 40:
            status = PASS
            msg = f"{gpu_name} ({gpu_memory_gb:.0f}GB) - Excellent"
        elif gpu_memory_gb >= 24:
            status = PASS
            msg = f"{gpu_name} ({gpu_memory_gb:.0f}GB) - Good"
        elif gpu_memory_gb >= 16:
            status = WARN
            msg = f"{gpu_name} ({gpu_memory_gb:.0f}GB) - May need smaller batch size"
        else:
            status = WARN
            msg = f"{gpu_name} ({gpu_memory_gb:.0f}GB) - Limited, use batch_size=1"

        return PreflightCheck("GPU Info", status, msg)

    except Exception as e:
        return PreflightCheck("GPU Info", FAIL, f"Error: {e}")


def check_pytorch_cuda() -> PreflightCheck:
    """Test that PyTorch can actually use CUDA."""
    try:
        import torch
        if not torch.cuda.is_available():
            return PreflightCheck("PyTorch CUDA", SKIP, "No CUDA available")

        # Quick tensor test
        start = time.time()
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        del x, y
        torch.cuda.empty_cache()

        return PreflightCheck(
            "PyTorch CUDA",
            PASS,
            f"GPU compute verified ({elapsed*1000:.1f}ms matmul test)"
        )

    except Exception as e:
        return PreflightCheck(
            "PyTorch CUDA",
            FAIL,
            f"GPU compute failed: {e}",
            "Check CUDA installation and driver compatibility"
        )


def check_training_data(project_root: Path, config: Optional[Dict] = None) -> PreflightCheck:
    """Check that training data files exist."""
    data_dir = project_root / "data" / "processed"

    # Get train file from config or use default
    train_file = "combined_train.jsonl"
    if config and "data" in config:
        train_file = config["data"].get("train_file", train_file)

    train_path = data_dir / train_file

    if not train_path.exists():
        return PreflightCheck(
            "Training Data",
            FAIL,
            f"Training file not found: {train_file}",
            f"Expected at: {train_path}"
        )

    # Count samples and check format
    try:
        sample_count = 0
        valid_format = True
        with open(train_path, 'r') as f:
            for i, line in enumerate(f):
                sample_count += 1
                if i == 0:  # Only check first line for speed
                    data = json.loads(line)
                    if 'messages' not in data:
                        valid_format = False

        size_mb = train_path.stat().st_size / (1024 * 1024)

        if not valid_format:
            return PreflightCheck(
                "Training Data",
                WARN,
                f"{train_file}: {sample_count} samples ({size_mb:.1f}MB)",
                "Warning: Expected ChatML format with 'messages' field"
            )

        return PreflightCheck(
            "Training Data",
            PASS,
            f"{train_file}: {sample_count} samples ({size_mb:.1f}MB)"
        )

    except json.JSONDecodeError as e:
        return PreflightCheck(
            "Training Data",
            FAIL,
            f"Invalid JSON in {train_file}",
            str(e)
        )
    except Exception as e:
        return PreflightCheck(
            "Training Data",
            FAIL,
            f"Error reading {train_file}: {e}"
        )


def check_config_file(config_path: Path) -> Tuple[PreflightCheck, Optional[Dict]]:
    """Check config file exists and is valid YAML."""
    if not config_path.exists():
        return PreflightCheck(
            "Config File",
            FAIL,
            f"Config not found: {config_path.name}",
            f"Expected at: {config_path}"
        ), None

    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Validate required sections
        required = ['model', 'lora', 'training', 'data', 'output']
        missing = [k for k in required if k not in config]

        if missing:
            return PreflightCheck(
                "Config File",
                WARN,
                f"{config_path.name}: Missing sections: {missing}"
            ), config

        return PreflightCheck(
            "Config File",
            PASS,
            f"{config_path.name}: Valid YAML with all required sections"
        ), config

    except yaml.YAMLError as e:
        return PreflightCheck(
            "Config File",
            FAIL,
            f"Invalid YAML in {config_path.name}",
            str(e)[:100]
        ), None
    except ImportError:
        return PreflightCheck(
            "Config File",
            FAIL,
            "PyYAML not installed",
            "Run: pip install pyyaml"
        ), None
    except Exception as e:
        return PreflightCheck(
            "Config File",
            FAIL,
            f"Error reading config: {e}"
        ), None


def check_disk_space(project_root: Path) -> PreflightCheck:
    """Check available disk space."""
    try:
        # Check disk space where models will be saved
        model_dir = project_root / "model_adapters"
        model_dir.mkdir(exist_ok=True)

        total, used, free = shutil.disk_usage(model_dir)
        free_gb = free / (1024**3)

        # Need at least 10GB for training (checkpoints, logs, exports)
        if free_gb >= 50:
            status = PASS
            msg = f"{free_gb:.0f}GB free - Plenty of space"
        elif free_gb >= 20:
            status = PASS
            msg = f"{free_gb:.0f}GB free - Sufficient"
        elif free_gb >= 10:
            status = WARN
            msg = f"{free_gb:.0f}GB free - May run low during training"
        else:
            status = FAIL
            msg = f"{free_gb:.0f}GB free - Insufficient (need 10GB+)"

        return PreflightCheck("Disk Space", status, msg)

    except Exception as e:
        return PreflightCheck("Disk Space", WARN, f"Could not check: {e}")


def check_output_dirs(project_root: Path, config: Optional[Dict] = None) -> PreflightCheck:
    """Check output directories can be created."""
    try:
        # Get output dir from config
        output_dir_name = "l4d2-mistral-v10plus-lora"
        if config and "output" in config:
            output_dir_name = config["output"].get("dir", output_dir_name)

        model_dir = project_root / "model_adapters" / output_dir_name
        logs_dir = project_root / "data" / "training_logs"

        # Try to create directories
        model_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        return PreflightCheck(
            "Output Dirs",
            PASS,
            f"model_adapters/{output_dir_name}/ and training_logs/ ready"
        )

    except PermissionError:
        return PreflightCheck(
            "Output Dirs",
            FAIL,
            "Permission denied creating output directories",
            "Check file permissions or run with appropriate user"
        )
    except Exception as e:
        return PreflightCheck(
            "Output Dirs",
            FAIL,
            f"Error creating output dirs: {e}"
        )


def estimate_training_time(config: Optional[Dict], sample_count: int) -> PreflightCheck:
    """Estimate training time based on config and hardware."""
    if not config:
        return PreflightCheck(
            "Training Estimate",
            SKIP,
            "Cannot estimate without valid config"
        )

    try:
        import torch

        # Get training params
        training_cfg = config.get("training", {})
        epochs = training_cfg.get("num_train_epochs", 3)
        batch_size = training_cfg.get("per_device_train_batch_size", 4)
        grad_accum = training_cfg.get("gradient_accumulation_steps", 4)

        effective_batch = batch_size * grad_accum
        steps_per_epoch = sample_count // effective_batch
        total_steps = steps_per_epoch * epochs

        # Estimate time per step based on GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            # Time per step estimates (seconds) - rough approximations
            if "a100" in gpu_name or "h100" in gpu_name or gpu_memory >= 40:
                sec_per_step = 2.5
                gpu_tier = "High-end"
            elif "a40" in gpu_name or "l40" in gpu_name or gpu_memory >= 40:
                sec_per_step = 3.5
                gpu_tier = "Professional"
            elif "4090" in gpu_name or "3090" in gpu_name or gpu_memory >= 20:
                sec_per_step = 4.0
                gpu_tier = "Consumer"
            else:
                sec_per_step = 6.0
                gpu_tier = "Entry"
        else:
            sec_per_step = 60.0  # CPU is very slow
            gpu_tier = "CPU"

        total_seconds = total_steps * sec_per_step
        hours = total_seconds / 3600

        details = (
            f"Samples: {sample_count} | Epochs: {epochs} | "
            f"Batch: {batch_size}x{grad_accum}={effective_batch} | "
            f"Steps: ~{total_steps}"
        )

        if hours < 1:
            time_str = f"{total_seconds/60:.0f} minutes"
        else:
            time_str = f"{hours:.1f} hours"

        return PreflightCheck(
            "Training Estimate",
            PASS,
            f"~{time_str} ({gpu_tier} GPU)",
            details
        )

    except Exception as e:
        return PreflightCheck(
            "Training Estimate",
            WARN,
            f"Could not estimate: {e}"
        )


def check_unsloth_available() -> PreflightCheck:
    """Check if Unsloth is installed."""
    try:
        from unsloth import FastLanguageModel
        return PreflightCheck(
            "Unsloth",
            PASS,
            "Unsloth library available"
        )
    except ImportError:
        return PreflightCheck(
            "Unsloth",
            WARN,
            "Unsloth not installed - using standard training",
            "pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'"
        )
    except Exception as e:
        return PreflightCheck(
            "Unsloth",
            WARN,
            f"Unsloth import issue: {e}"
        )


def run_preflight(config_path: Optional[Path] = None, verbose: bool = False) -> bool:
    """
    Run all preflight checks.

    Returns True if all critical checks pass.
    """
    start_time = time.time()

    # Determine project root
    project_root = Path(__file__).parent.parent.parent.resolve()

    # Default config path
    if config_path is None:
        config_path = project_root / "configs" / "unsloth_config.yaml"
        # Try A100 config if default doesn't exist
        if not config_path.exists():
            a100_config = project_root / "configs" / "unsloth_config_a100.yaml"
            if a100_config.exists():
                config_path = a100_config

    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}  Vultr Pre-Training Validation{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")

    checks: List[PreflightCheck] = []
    config: Optional[Dict] = None
    sample_count = 0

    # 1. CUDA availability
    checks.append(check_cuda_available())

    # 2. GPU info
    checks.append(check_gpu_info())

    # 3. PyTorch CUDA test
    checks.append(check_pytorch_cuda())

    # 4. Config file
    config_check, config = check_config_file(config_path)
    checks.append(config_check)

    # 5. Training data
    data_check = check_training_data(project_root, config)
    checks.append(data_check)

    # Extract sample count for time estimate
    if data_check.status == PASS or data_check.status == WARN:
        try:
            # Parse sample count from message (e.g., "file: 1234 samples")
            parts = data_check.message.split()
            for i, p in enumerate(parts):
                if p == "samples":
                    sample_count = int(parts[i-1])
                    break
        except (ValueError, IndexError):
            sample_count = 1000  # Default estimate

    # 6. Disk space
    checks.append(check_disk_space(project_root))

    # 7. Output directories
    checks.append(check_output_dirs(project_root, config))

    # 8. Unsloth (optional)
    checks.append(check_unsloth_available())

    # 9. Training time estimate
    checks.append(estimate_training_time(config, sample_count))

    # Print results
    for check in checks:
        print(check)
        if verbose and check.details:
            print(f"       {Colors.BLUE}Detail:{Colors.RESET} {check.details}")

    elapsed = time.time() - start_time

    # Summary
    passed = sum(1 for c in checks if c.status == PASS)
    failed = sum(1 for c in checks if c.status == FAIL)
    warned = sum(1 for c in checks if c.status == WARN)
    skipped = sum(1 for c in checks if c.status == SKIP)

    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}  Summary{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")

    print(f"\n  Checks: {passed} passed, {warned} warnings, {failed} failed, {skipped} skipped")
    print(f"  Time: {elapsed:.2f}s\n")

    if failed > 0:
        print(f"{Colors.RED}{Colors.BOLD}PREFLIGHT FAILED{Colors.RESET}")
        print(f"\nFix the issues above before starting training.")
        print(f"This prevents wasting GPU credits on a failed run.\n")
        return False
    elif warned > 0:
        print(f"{Colors.YELLOW}{Colors.BOLD}PREFLIGHT PASSED WITH WARNINGS{Colors.RESET}")
        print(f"\nTraining can proceed, but review warnings above.\n")
        return True
    else:
        print(f"{Colors.GREEN}{Colors.BOLD}PREFLIGHT PASSED{Colors.RESET}")
        print(f"\nSystem is ready for training. Start with:\n")
        print(f"  python scripts/training/train_unsloth.py --config {config_path.name}\n")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Vultr pre-training validation - run before starting GPU training"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to training config YAML (default: configs/unsloth_config.yaml)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output for each check"
    )

    args = parser.parse_args()

    success = run_preflight(args.config, args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
