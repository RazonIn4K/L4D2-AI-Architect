#!/usr/bin/env python3
"""
Multi-Model Training Orchestrator for Vultr Credit Burn

Trains multiple base models (Mistral, CodeLlama, Qwen) sequentially
to maximize value from limited GPU credits.

Usage:
    python train_all_models.py                    # Train all models
    python train_all_models.py --models mistral   # Train only Mistral
    python train_all_models.py --models mistral codellama  # Train two
    python train_all_models.py --dry-run          # Show plan without training
    python train_all_models.py --export-only      # Just export existing models
    python train_all_models.py --fast             # Use fast configs (20-30% faster)
    python train_all_models.py --turbo            # Use turbo configs (35-45% faster, A100 optimized)

Estimated times on A100 40GB:
    Standard mode:
    - Mistral-7B: ~4.0 hours ($10.00)
    - CodeLlama-7B: ~4.0 hours ($10.00)
    - Qwen2.5-7B: ~4.0 hours ($10.00)
    - All three: ~12.0 hours ($30.00)

    Turbo mode (--turbo):
    - Mistral-7B: ~1.5 hours ($3.75)
    - CodeLlama-7B: ~1.5 hours ($3.75)
    - Qwen2.5-7B: ~1.8 hours ($4.50)
    - All three: ~4.8 hours ($12.00)
"""

import os
import sys
import argparse
import logging
import subprocess
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Add parent to path for security utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_path, safe_write_json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "model_adapters"
EXPORTS_DIR = PROJECT_ROOT / "exports"

# Model configurations with estimated training times
# Standard configs (balanced speed/quality)
MODEL_CONFIGS = {
    "mistral": {
        "name": "Mistral-7B-Instruct",
        "config": "unsloth_config_v15.yaml",
        "config_fast": "unsloth_config_v15_fast.yaml",
        "config_turbo": "unsloth_config_a100_turbo.yaml",
        "base_model": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        "output_name": "l4d2-mistral-v15-lora",
        "output_name_fast": "l4d2-mistral-v15-fast-lora",
        "output_name_turbo": "l4d2-mistral-v15-turbo-lora",
        "estimated_hours": 4.0,
        "estimated_hours_fast": 2.5,
        "estimated_hours_turbo": 1.5,
        "priority": 1,  # Train first
    },
    "codellama": {
        "name": "CodeLlama-7B",
        "config": "unsloth_config_codellama.yaml",
        "config_fast": "unsloth_config_codellama_fast.yaml",
        "config_turbo": "unsloth_config_codellama_turbo.yaml",
        "base_model": "unsloth/codellama-7b-instruct-bnb-4bit",
        "output_name": "l4d2-codellama-v15-lora",
        "output_name_fast": "l4d2-codellama-v15-fast-lora",
        "output_name_turbo": "l4d2-codellama-v15-turbo-lora",
        "estimated_hours": 4.0,
        "estimated_hours_fast": 2.5,
        "estimated_hours_turbo": 1.5,
        "priority": 2,
    },
    "qwen": {
        "name": "Qwen2.5-Coder-7B",
        "config": "unsloth_config_qwen.yaml",
        "config_fast": "unsloth_config_qwen_fast.yaml",
        "config_turbo": "unsloth_config_qwen_turbo.yaml",
        "base_model": "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
        "output_name": "l4d2-qwen-v15-lora",
        "output_name_fast": "l4d2-qwen-v15-fast-lora",
        "output_name_turbo": "l4d2-qwen-v15-turbo-lora",
        "estimated_hours": 4.0,
        "estimated_hours_fast": 3.0,
        "estimated_hours_turbo": 1.8,
        "priority": 3,
    },
    "llama3": {
        "name": "Llama-3-8B",
        "config": "unsloth_config_llama3.yaml",
        "config_fast": None,  # No fast config yet
        "config_turbo": None,  # No turbo config yet
        "base_model": "unsloth/llama-3-8b-instruct-bnb-4bit",
        "output_name": "l4d2-llama3-v15-lora",
        "output_name_fast": None,
        "output_name_turbo": None,
        "estimated_hours": 4.5,
        "estimated_hours_fast": None,
        "estimated_hours_turbo": None,
        "priority": 4,
    },
}


def check_gpu() -> Dict[str, Any]:
    """Check GPU availability and specs."""
    import torch

    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        return {"available": False}

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

    return {
        "available": True,
        "name": gpu_name,
        "memory_gb": gpu_memory,
        "is_a100": "A100" in gpu_name,
        "is_h100": "H100" in gpu_name or "GH200" in gpu_name,
    }


def check_training_data() -> Optional[Path]:
    """Find best available training data."""
    for version in ["v15", "v14", "v13", "v12", "v11", "v10"]:
        data_path = DATA_DIR / f"l4d2_train_{version}.jsonl"
        if data_path.exists():
            size_mb = data_path.stat().st_size / 1024 / 1024
            with open(data_path, 'r') as f:
                line_count = sum(1 for _ in f)
            logger.info(f"Found training data: {data_path.name} ({line_count} samples, {size_mb:.1f} MB)")
            return data_path
    return None


def check_config_exists(config_name: str) -> bool:
    """Check if config file exists."""
    config_path = CONFIGS_DIR / config_name
    return config_path.exists()


def train_model(model_key: str, config: Dict[str, Any], dry_run: bool = False, fast: bool = False, turbo: bool = False) -> bool:
    """Train a single model."""
    logger.info(f"\n{'='*60}")
    if turbo:
        mode_str = " (TURBO MODE)"
    elif fast:
        mode_str = " (FAST MODE)"
    else:
        mode_str = ""
    logger.info(f"TRAINING: {config['name']}{mode_str}")
    logger.info(f"{'='*60}")

    # Use turbo config if available and requested (highest priority)
    if turbo and config.get("config_turbo"):
        config_file = config["config_turbo"]
        output_name = config.get("output_name_turbo", config["output_name"])
        est_hours = config.get("estimated_hours_turbo", config["estimated_hours"])
    # Use fast config if available and requested
    elif fast and config.get("config_fast"):
        config_file = config["config_fast"]
        output_name = config.get("output_name_fast", config["output_name"])
        est_hours = config.get("estimated_hours_fast", config["estimated_hours"])
    else:
        config_file = config["config"]
        output_name = config["output_name"]
        est_hours = config["estimated_hours"]

    config_path = CONFIGS_DIR / config_file

    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return False

    if dry_run:
        logger.info(f"[DRY RUN] Would train with config: {config_path}")
        logger.info(f"[DRY RUN] Output would be: {OUTPUT_DIR / output_name}")
        logger.info(f"[DRY RUN] Estimated time: {est_hours} hours")
        return True

    # Run training
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "training" / "train_unsloth.py"),
        "--config", str(config_path),
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    start_time = datetime.now()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            check=True,
        )
        elapsed = datetime.now() - start_time
        logger.info(f"Training completed in {elapsed}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return False


def export_model(model_key: str, config: Dict[str, Any], dry_run: bool = False) -> bool:
    """Export a trained model to GGUF."""
    adapter_path = OUTPUT_DIR / config["output_name"] / "final"

    if not adapter_path.exists():
        logger.warning(f"No trained model found at {adapter_path}")
        return False

    export_path = EXPORTS_DIR / f"{config['output_name']}"

    if dry_run:
        logger.info(f"[DRY RUN] Would export: {adapter_path} -> {export_path}")
        return True

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "training" / "export_gguf_cpu.py"),
        "--adapter", str(adapter_path),
        "--output", str(export_path),
        "--quantize", "q4_k_m",
        "--create-modelfile",
    ]

    logger.info(f"Exporting: {config['name']} -> {export_path}")

    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
        logger.info(f"Export completed: {export_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Export failed: {e}")
        return False


def print_plan(models: List[str], gpu_info: Dict[str, Any], fast: bool = False, turbo: bool = False):
    """Print training plan with time and cost estimates."""
    print("\n" + "=" * 70)
    if turbo:
        mode_str = " (TURBO MODE)"
    elif fast:
        mode_str = " (FAST MODE)"
    else:
        mode_str = ""
    print(f"VULTR CREDIT BURN - TRAINING PLAN{mode_str}")
    print("=" * 70)
    print(f"\nGPU: {gpu_info.get('name', 'Unknown')} ({gpu_info.get('memory_gb', 0):.0f} GB)")
    print(f"Hourly Rate: $2.50 (A100) / $1.50 (A40)")
    print()

    total_hours = 0
    total_cost = 0
    hourly_rate = 2.50 if gpu_info.get("is_a100") else 1.50

    print(f"{'Model':<25} {'Est. Time':<15} {'Est. Cost':<12} {'Priority':<10}")
    print("-" * 62)

    for model_key in models:
        if model_key in MODEL_CONFIGS:
            config = MODEL_CONFIGS[model_key]
            # Use turbo estimates if available and requested (highest priority)
            if turbo and config.get("estimated_hours_turbo"):
                hours = config["estimated_hours_turbo"]
            # Use fast estimates if available and requested
            elif fast and config.get("estimated_hours_fast"):
                hours = config["estimated_hours_fast"]
            else:
                hours = config["estimated_hours"]
            cost = hours * hourly_rate
            total_hours += hours
            total_cost += cost
            print(f"{config['name']:<25} {hours:.1f} hours{'':<7} ${cost:.2f}{'':<8} {config['priority']}")

    print("-" * 62)
    print(f"{'TOTAL':<25} {total_hours:.1f} hours{'':<7} ${total_cost:.2f}")
    print()

    # Time estimates
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=total_hours)
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"Est. End:   {end_time.strftime('%Y-%m-%d %H:%M')}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Train multiple L4D2 code models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train_all_models.py                    # Train all models
    python train_all_models.py --models mistral   # Train only Mistral
    python train_all_models.py --dry-run          # Show plan only
    python train_all_models.py --export-only      # Export without training
    python train_all_models.py --fast             # Use fast configs (20-30% faster)
    python train_all_models.py --turbo            # Use turbo configs (35-45% faster)
        """
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_CONFIGS.keys()),
        default=["mistral", "codellama", "qwen"],
        help="Models to train (default: mistral codellama qwen)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without actually training"
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export existing models, skip training"
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip GGUF export after training"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use speed-optimized configs (20-30%% faster, slightly smaller batch)"
    )
    parser.add_argument(
        "--turbo",
        action="store_true",
        help="Use turbo configs for maximum speed (35-45%% faster, optimized for A100)"
    )

    args = parser.parse_args()

    # Turbo takes precedence over fast
    if args.turbo and args.fast:
        logger.warning("Both --turbo and --fast specified. Using --turbo (faster).")

    print("\n" + "=" * 70)
    print("L4D2-AI-ARCHITECT: MULTI-MODEL TRAINING ORCHESTRATOR")
    print("=" * 70)

    # Check GPU
    gpu_info = check_gpu()
    if not gpu_info["available"]:
        logger.error("No GPU available. Exiting.")
        sys.exit(1)

    logger.info(f"GPU: {gpu_info['name']} ({gpu_info['memory_gb']:.0f} GB)")

    # Check training data
    training_data = check_training_data()
    if not training_data and not args.export_only:
        logger.error("No training data found!")
        sys.exit(1)

    # Sort models by priority
    models = sorted(args.models, key=lambda x: MODEL_CONFIGS[x]["priority"])

    # Print plan
    print_plan(models, gpu_info, fast=args.fast, turbo=args.turbo)

    if args.dry_run:
        print("[DRY RUN MODE - No actual training will occur]")
        print()

    # Confirm before starting
    if not args.dry_run and not args.export_only:
        print("Press Enter to start training (Ctrl+C to cancel)...")
        try:
            input()
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)

    # Track results
    results = {
        "start_time": datetime.now().isoformat(),
        "gpu": gpu_info,
        "models": {},
    }

    # Train each model
    for model_key in models:
        config = MODEL_CONFIGS[model_key]

        if not args.export_only:
            # Train
            success = train_model(model_key, config, dry_run=args.dry_run, fast=args.fast, turbo=args.turbo)
            results["models"][model_key] = {
                "training": "success" if success else "failed",
                "fast_mode": args.fast,
                "turbo_mode": args.turbo,
            }

            if not success and not args.dry_run:
                logger.warning(f"Training failed for {config['name']}, continuing with next model...")
                continue

        # Export
        if not args.skip_export:
            export_success = export_model(model_key, config, dry_run=args.dry_run)
            if model_key not in results["models"]:
                results["models"][model_key] = {}
            results["models"][model_key]["export"] = "success" if export_success else "failed"

    # Save results
    results["end_time"] = datetime.now().isoformat()
    results_path = PROJECT_ROOT / "data" / "training_logs" / f"multi_model_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        safe_write_json(str(results_path), results, PROJECT_ROOT)
        logger.info(f"Results saved to {results_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)

    for model_key, status in results.get("models", {}).items():
        config = MODEL_CONFIGS[model_key]
        train_status = status.get("training", "skipped")
        export_status = status.get("export", "skipped")
        print(f"{config['name']:<25} Train: {train_status:<10} Export: {export_status}")

    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
