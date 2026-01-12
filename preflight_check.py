#!/usr/bin/env python3
"""
Pre-flight Check for Vultr Deployment

Run this locally before deploying to Vultr to verify everything is ready.

Usage:
    python preflight_check.py
"""

import sys
import json
from pathlib import Path

# Colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
BOLD = '\033[1m'
NC = '\033[0m'

def ok(msg):
    print(f"{GREEN}[OK]{NC} {msg}")

def fail(msg):
    print(f"{RED}[FAIL]{NC} {msg}")

def warn(msg):
    print(f"{YELLOW}[WARN]{NC} {msg}")

def info(msg):
    print(f"{BLUE}[INFO]{NC} {msg}")

def main():
    print()
    print(f"{BOLD}{'='*60}{NC}")
    print(f"{BOLD}L4D2-AI-ARCHITECT: PRE-FLIGHT CHECK{NC}")
    print(f"{BOLD}{'='*60}{NC}")
    print()

    project_root = Path(__file__).parent
    all_ok = True

    # 1. Check training data
    print(f"{BOLD}1. Training Data{NC}")
    data_dir = project_root / "data" / "processed"

    datasets = []
    for version in ["v15", "v14", "v13", "v12", "v11", "v10"]:
        path = data_dir / f"l4d2_train_{version}.jsonl"
        if path.exists():
            with open(path, 'r') as f:
                count = sum(1 for _ in f)
            size_mb = path.stat().st_size / 1024 / 1024
            datasets.append((version, count, size_mb))
            ok(f"l4d2_train_{version}.jsonl: {count} samples ({size_mb:.1f} MB)")

    if not datasets:
        fail("No training data found!")
        all_ok = False
    else:
        best = datasets[0]
        info(f"Best dataset: V{best[0].upper()} with {best[1]} samples")

    # Check validation data
    val_path = data_dir / "combined_val.jsonl"
    if val_path.exists():
        with open(val_path, 'r') as f:
            val_count = sum(1 for _ in f)
        ok(f"Validation data: {val_count} samples")
    else:
        warn("No validation data found (training will work without it)")

    print()

    # 2. Check configs
    print(f"{BOLD}2. Training Configs{NC}")
    configs_dir = project_root / "configs"

    required_configs = [
        ("unsloth_config_v15.yaml", "Mistral V15"),
        ("unsloth_config_codellama.yaml", "CodeLlama V15"),
        ("unsloth_config_qwen.yaml", "Qwen V15"),
    ]

    # Optional fast configs
    fast_configs = [
        ("unsloth_config_v15_fast.yaml", "Mistral V15 FAST"),
        ("unsloth_config_codellama_fast.yaml", "CodeLlama V15 FAST"),
        ("unsloth_config_qwen_fast.yaml", "Qwen V15 FAST"),
    ]

    for config_file, name in required_configs:
        path = configs_dir / config_file
        if path.exists():
            # Check if it uses V15 dataset
            content = path.read_text()
            if "l4d2_train_v15" in content:
                ok(f"{config_file}: {name} (uses V15 dataset)")
            else:
                warn(f"{config_file}: {name} (NOT using V15 dataset)")
        else:
            fail(f"{config_file}: NOT FOUND")
            all_ok = False

    # Check fast configs (optional but good to have)
    for config_file, name in fast_configs:
        path = configs_dir / config_file
        if path.exists():
            content = path.read_text()
            if "packing: true" in content:
                ok(f"{config_file}: {name} (packing enabled)")
            else:
                ok(f"{config_file}: {name}")
        else:
            info(f"{config_file}: Not found (optional)")

    print()

    # 3. Check training scripts
    print(f"{BOLD}3. Training Scripts{NC}")
    scripts = [
        "scripts/training/train_unsloth.py",
        "scripts/training/train_all_models.py",
        "scripts/training/export_gguf_cpu.py",
    ]

    for script in scripts:
        path = project_root / script
        if path.exists():
            # Check for known issues
            content = path.read_text()
            if "save_method=\"lora\"" in content:
                warn(f"{script}: Contains potentially buggy save_method='lora'")
            else:
                ok(f"{script}")
        else:
            fail(f"{script}: NOT FOUND")
            all_ok = False

    print()

    # 4. Check deployment scripts
    print(f"{BOLD}4. Deployment Scripts{NC}")
    deploy_scripts = [
        "deploy_to_vultr.sh",
        "scripts/utils/vultr_credit_burn.sh",
        "scripts/utils/vultr_quickstart.sh",
    ]

    for script in deploy_scripts:
        path = project_root / script
        if path.exists():
            ok(script)
        else:
            warn(f"{script}: NOT FOUND (may not be critical)")

    print()

    # 5. Check requirements
    print(f"{BOLD}5. Requirements{NC}")
    req_path = project_root / "requirements.txt"
    if req_path.exists():
        ok("requirements.txt exists")
    else:
        fail("requirements.txt NOT FOUND")
        all_ok = False

    print()

    # 6. Estimate training
    print(f"{BOLD}6. Training Estimates (A100 @ $2.50/hr){NC}")
    if datasets:
        best_samples = datasets[0][1]
        # Unsloth on A100 achieves ~2500-3000 samples/hour for 7B models
        # 2773 samples * 3 epochs = 8319 effective samples
        # At 2500 samples/hour = ~3.3 hours + 20% buffer = ~4 hours
        hours_per_model = (best_samples * 3) / 2500 * 1.2  # 20% buffer
        models = ["Mistral", "CodeLlama", "Qwen"]

        total_hours = 0
        for model in models:
            info(f"{model}: ~{hours_per_model:.1f} hours (${hours_per_model * 2.50:.2f})")
            total_hours += hours_per_model

        print()
        info(f"Total for all 3 models: ~{total_hours:.1f} hours (${total_hours * 2.50:.2f})")
        info(f"Single model (Mistral only): ~{hours_per_model:.1f} hours (${hours_per_model * 2.50:.2f})")

    print()

    # Summary
    print(f"{BOLD}{'='*60}{NC}")
    if all_ok:
        print(f"{GREEN}{BOLD}PRE-FLIGHT CHECK PASSED!{NC}")
        print()
        print("Ready to deploy. Run:")
        print()
        print(f"  {BLUE}./deploy_to_vultr.sh YOUR_VULTR_IP{NC}")
        print()
    else:
        print(f"{RED}{BOLD}PRE-FLIGHT CHECK FAILED!{NC}")
        print()
        print("Fix the issues above before deploying.")
        sys.exit(1)

    print(f"{'='*60}")
    print()


if __name__ == "__main__":
    main()
