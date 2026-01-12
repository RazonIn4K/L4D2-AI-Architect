#!/usr/bin/env python3
"""
Pre-flight Checklist for L4D2 Training Deployment

Run this locally before deploying to Vultr A100 to ensure everything is ready.

Usage:
    python scripts/utils/preflight_check.py
"""

import json
import os
import sys
from pathlib import Path

# Add parent to path for security imports
sys.path.insert(0, str(Path(__file__).parent))
from security import safe_path, safe_read_text

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def check_mark(passed: bool) -> str:
    return f"{Colors.GREEN}✓{Colors.RESET}" if passed else f"{Colors.RED}✗{Colors.RESET}"

def warn_mark() -> str:
    return f"{Colors.YELLOW}⚠{Colors.RESET}"

def main():
    print(f"\n{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}L4D2-AI-Architect Pre-flight Checklist{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}\n")

    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)

    all_passed = True
    warnings = []

    # 1. Check training data
    print(f"{Colors.BLUE}[1/6] Training Data{Colors.RESET}")
    train_file = Path("data/processed/combined_train.jsonl")
    val_file = Path("data/processed/combined_val.jsonl")

    if train_file.exists():
        # Use safe_path to validate file is within project
        validated_train = safe_path(str(train_file), project_root)
        with open(validated_train) as f:
            train_count = sum(1 for _ in f)
        size_mb = validated_train.stat().st_size / (1024 * 1024)
        print(f"  {check_mark(True)} Training data: {train_count} samples ({size_mb:.1f} MB)")

        # Validate format
        with open(validated_train) as f:
            try:
                sample = json.loads(f.readline())
                if 'messages' in sample:
                    print(f"  {check_mark(True)} Format: ChatML (messages array)")
                else:
                    print(f"  {check_mark(False)} Format: Unknown (expected 'messages' field)")
                    all_passed = False
            except json.JSONDecodeError:
                print(f"  {check_mark(False)} Format: Invalid JSON")
                all_passed = False
    else:
        print(f"  {check_mark(False)} Training data: NOT FOUND")
        all_passed = False

    if val_file.exists():
        validated_val = safe_path(str(val_file), project_root)
        with open(validated_val) as f:
            val_count = sum(1 for _ in f)
        print(f"  {check_mark(True)} Validation data: {val_count} samples")
    else:
        print(f"  {warn_mark()} Validation data: Not found (optional)")
        warnings.append("No validation data - training will work but no eval metrics")

    # 2. Check config file
    print(f"\n{Colors.BLUE}[2/6] Configuration{Colors.RESET}")
    config_file = Path("configs/unsloth_config_a100.yaml")
    if config_file.exists():
        print(f"  {check_mark(True)} A100 config: {config_file}")
    else:
        print(f"  {warn_mark()} A100 config not found, will use defaults")
        warnings.append("A100 config missing - script will auto-detect settings")

    # 3. Check training script
    print(f"\n{Colors.BLUE}[3/6] Training Script{Colors.RESET}")
    train_script = Path("scripts/training/train_runpod.py")
    if train_script.exists():
        print(f"  {check_mark(True)} Training script: {train_script}")

        # Check for key imports (use safe_path for validation)
        validated_script = safe_path(str(train_script), project_root)
        with open(validated_script) as f:
            content = f.read()
            if "detect_gpu_type" in content:
                print(f"  {check_mark(True)} GPU auto-detection: Enabled")
            else:
                print(f"  {warn_mark()} GPU auto-detection: Not found")
                warnings.append("Update train_runpod.py for A100 auto-detection")
    else:
        print(f"  {check_mark(False)} Training script: NOT FOUND")
        all_passed = False

    # 4. Check inference script
    print(f"\n{Colors.BLUE}[4/6] Inference Script{Colors.RESET}")
    inference_script = Path("scripts/inference/test_lora.py")
    if inference_script.exists():
        print(f"  {check_mark(True)} Inference script: {inference_script}")
    else:
        print(f"  {warn_mark()} Inference script: Not found")
        warnings.append("No inference script for testing trained model")

    # 5. Check output directories
    print(f"\n{Colors.BLUE}[5/6] Output Directories{Colors.RESET}")
    model_dir = Path("model_adapters")
    logs_dir = Path("data/training_logs")

    model_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"  {check_mark(True)} Model output: {model_dir}")
    print(f"  {check_mark(True)} Logs output: {logs_dir}")

    # 6. Estimate training requirements
    print(f"\n{Colors.BLUE}[6/6] Resource Estimates{Colors.RESET}")
    if train_file.exists():
        # Rough estimates based on sample count
        samples = train_count
        epochs = 3
        batch_size = 8
        steps_per_epoch = samples // batch_size
        total_steps = steps_per_epoch * epochs

        print(f"  Samples: {samples}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size} (A100 recommended)")
        print(f"  Estimated steps: ~{total_steps}")
        print(f"  Estimated time: ~1.5-2 hours (A100)")
        print(f"  Estimated cost: ~$1.50-2.00 (at $1/hr)")

    # Summary
    print(f"\n{Colors.BLUE}{'='*60}{Colors.RESET}")
    if all_passed:
        print(f"{Colors.GREEN}All checks passed! Ready for deployment.{Colors.RESET}")
    else:
        print(f"{Colors.RED}Some checks failed. Please fix before deploying.{Colors.RESET}")

    if warnings:
        print(f"\n{Colors.YELLOW}Warnings:{Colors.RESET}")
        for w in warnings:
            print(f"  {warn_mark()} {w}")

    # Deployment commands
    print(f"\n{Colors.BLUE}Deployment Commands:{Colors.RESET}")
    print("""
# 1. Upload to Vultr instance:
scp -r data/processed/*.jsonl root@<VULTR_IP>:/root/L4D2-AI-Architect/data/processed/
scp scripts/training/train_runpod.py root@<VULTR_IP>:/root/L4D2-AI-Architect/scripts/training/

# 2. SSH and run setup:
ssh root@<VULTR_IP>
cd /root/L4D2-AI-Architect
./scripts/utils/vultr_setup_a100.sh

# 3. Quick validation (20 steps):
tmux new -s training
source venv/bin/activate
python scripts/training/train_runpod.py --model mistral --max-steps 20

# 4. Full training (if validation passes):
python scripts/training/train_runpod.py --model mistral --batch-size 8 --epochs 3

# 5. Download trained model:
tar -czvf l4d2-lora.tar.gz model_adapters/
# From local: scp root@<VULTR_IP>:/root/L4D2-AI-Architect/l4d2-lora.tar.gz ./
""")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
