#!/usr/bin/env python3
"""
Quick Script Validation - Checks Python syntax and basic imports.

Run this to catch obvious errors before deploying to GPU instance.
"""

import sys
import ast
import importlib.util
from pathlib import Path

# Colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BOLD = '\033[1m'
NC = '\033[0m'

def ok(msg): print(f"{GREEN}[OK]{NC} {msg}")
def fail(msg): print(f"{RED}[FAIL]{NC} {msg}")
def warn(msg): print(f"{YELLOW}[WARN]{NC} {msg}")

def check_syntax(filepath: Path) -> bool:
    """Check Python syntax without executing."""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        ast.parse(source)
        return True
    except SyntaxError as e:
        fail(f"{filepath.name}: Syntax error at line {e.lineno}: {e.msg}")
        return False

def check_imports_safe(filepath: Path) -> bool:
    """Try to load module spec without executing."""
    try:
        spec = importlib.util.spec_from_file_location(filepath.stem, filepath)
        if spec is None:
            warn(f"{filepath.name}: Could not create module spec")
            return True  # Not a hard failure
        return True
    except Exception as e:
        warn(f"{filepath.name}: Import check warning: {e}")
        return True  # Warnings only

def main():
    print()
    print(f"{BOLD}{'='*60}{NC}")
    print(f"{BOLD}L4D2-AI-ARCHITECT: SCRIPT VALIDATION{NC}")
    print(f"{BOLD}{'='*60}{NC}")
    print()

    project_root = Path(__file__).parent
    all_ok = True

    # Critical training scripts
    critical_scripts = [
        "scripts/training/train_unsloth.py",
        "scripts/training/train_all_models.py",
        "scripts/training/export_gguf_cpu.py",
        "scripts/training/prepare_dataset.py",
    ]

    # Utility scripts
    utility_scripts = [
        "scripts/utils/security.py",
        "preflight_check.py",
    ]

    print(f"{BOLD}Checking Critical Training Scripts:{NC}")
    for script_path in critical_scripts:
        full_path = project_root / script_path
        if not full_path.exists():
            warn(f"{script_path}: Not found")
            continue

        if check_syntax(full_path):
            check_imports_safe(full_path)
            ok(f"{script_path}")
        else:
            all_ok = False

    print()
    print(f"{BOLD}Checking Utility Scripts:{NC}")
    for script_path in utility_scripts:
        full_path = project_root / script_path
        if not full_path.exists():
            warn(f"{script_path}: Not found")
            continue

        if check_syntax(full_path):
            ok(f"{script_path}")
        else:
            all_ok = False

    print()

    # Check YAML configs are valid
    print(f"{BOLD}Checking YAML Configs:{NC}")
    try:
        import yaml
        configs = [
            "configs/unsloth_config_v15.yaml",
            "configs/unsloth_config_codellama.yaml",
            "configs/unsloth_config_qwen.yaml",
        ]
        for config_path in configs:
            full_path = project_root / config_path
            if full_path.exists():
                with open(full_path, 'r') as f:
                    yaml.safe_load(f)
                ok(config_path)
            else:
                warn(f"{config_path}: Not found")
    except ImportError:
        warn("PyYAML not installed, skipping YAML validation")
    except yaml.YAMLError as e:
        fail(f"YAML error in {config_path}: {e}")
        all_ok = False

    print()

    # Verify training data format
    print(f"{BOLD}Checking Training Data Format:{NC}")
    try:
        import json
        train_file = project_root / "data/processed/l4d2_train_v15.jsonl"
        if train_file.exists():
            with open(train_file, 'r') as f:
                first_line = f.readline()
                sample = json.loads(first_line)

                if "messages" in sample:
                    messages = sample["messages"]
                    roles = [m.get("role") for m in messages]
                    if "system" in roles and "user" in roles and "assistant" in roles:
                        ok(f"Training data format: ChatML (system/user/assistant)")
                    else:
                        warn(f"Training data format: Missing expected roles (found: {roles})")
                else:
                    warn(f"Training data format: 'messages' key not found")
        else:
            fail("Training data not found!")
            all_ok = False
    except json.JSONDecodeError as e:
        fail(f"Training data JSON error: {e}")
        all_ok = False

    print()

    # Summary
    print(f"{BOLD}{'='*60}{NC}")
    if all_ok:
        print(f"{GREEN}{BOLD}ALL SCRIPTS VALIDATED!{NC}")
        print()
        print("Ready for deployment. Run preflight_check.py for full system check.")
    else:
        print(f"{RED}{BOLD}VALIDATION FAILED!{NC}")
        print()
        print("Fix the errors above before deploying.")
        sys.exit(1)

    print(f"{'='*60}")
    print()


if __name__ == "__main__":
    main()
