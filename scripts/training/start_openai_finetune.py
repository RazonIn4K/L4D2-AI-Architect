#!/usr/bin/env python3
"""
Start OpenAI Fine-Tuning Job for L4D2 SourcePawn Model

Usage:
    python scripts/training/start_openai_finetune.py [--version v5]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def upload_file(client: OpenAI, file_path: Path, purpose: str = "fine-tune") -> str:
    """Upload a file to OpenAI and return file ID."""
    print(f"Uploading {file_path.name}...")

    with open(file_path, "rb") as f:
        response = client.files.create(file=f, purpose=purpose)

    print(f"  -> File ID: {response.id}")
    return response.id


def start_finetune(
    client: OpenAI,
    training_file_id: str,
    validation_file_id: str,
    model: str = "gpt-4o-mini-2024-07-18",
    suffix: str = "l4d2-sourcemod-v5",
    n_epochs: int = 3
) -> dict:
    """Start a fine-tuning job."""
    print(f"\nStarting fine-tuning job...")
    print(f"  Base model: {model}")
    print(f"  Suffix: {suffix}")
    print(f"  Epochs: {n_epochs}")

    response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model=model,
        suffix=suffix,
        hyperparameters={
            "n_epochs": n_epochs
        }
    )

    return {
        "job_id": response.id,
        "status": response.status,
        "model": response.model,
        "created_at": response.created_at,
        "fine_tuned_model": response.fine_tuned_model
    }


def main():
    parser = argparse.ArgumentParser(description="Start OpenAI fine-tuning job")
    parser.add_argument("--version", default="v5", help="Dataset version (v4, v5)")
    parser.add_argument("--model", default="gpt-4o-mini-2024-07-18", help="Base model")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--dry-run", action="store_true", help="Just validate files, don't start job")
    args = parser.parse_args()

    print("=" * 60)
    print(f"L4D2 SourcePawn OpenAI Fine-Tuning - {args.version.upper()}")
    print("=" * 60)

    # File paths
    train_path = PROJECT_ROOT / f"data/openai_finetune/train_{args.version}.jsonl"
    eval_path = PROJECT_ROOT / f"data/openai_finetune/eval_{args.version}.jsonl"

    # Validate files exist
    if not train_path.exists():
        print(f"ERROR: Training file not found: {train_path}")
        sys.exit(1)
    if not eval_path.exists():
        print(f"ERROR: Eval file not found: {eval_path}")
        sys.exit(1)

    # Count examples
    with open(train_path) as f:
        train_count = sum(1 for _ in f)
    with open(eval_path) as f:
        eval_count = sum(1 for _ in f)

    print(f"\nDataset: {args.version}")
    print(f"  Training examples: {train_count}")
    print(f"  Validation examples: {eval_count}")

    # Validate JSON format
    print("\nValidating JSONL format...")
    errors = []
    for path, name in [(train_path, "train"), (eval_path, "eval")]:
        with open(path) as f:
            for i, line in enumerate(f, 1):
                try:
                    obj = json.loads(line)
                    if "messages" not in obj:
                        errors.append(f"{name}:{i} - Missing 'messages' key")
                except json.JSONDecodeError as e:
                    errors.append(f"{name}:{i} - JSON error: {e}")

    if errors:
        print("ERRORS found:")
        for err in errors[:10]:
            print(f"  {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        sys.exit(1)

    print("  -> Format valid")

    if args.dry_run:
        print("\n[DRY RUN] Would start fine-tuning with above settings")
        return

    # Initialize client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\nERROR: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key'")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Upload files
    print("\n" + "-" * 40)
    train_file_id = upload_file(client, train_path)
    eval_file_id = upload_file(client, eval_path)

    # Start fine-tuning
    print("-" * 40)
    suffix = f"l4d2-sourcemod-{args.version}"
    result = start_finetune(
        client,
        train_file_id,
        eval_file_id,
        model=args.model,
        suffix=suffix,
        n_epochs=args.epochs
    )

    print("\n" + "=" * 60)
    print("FINE-TUNING JOB STARTED")
    print("=" * 60)
    print(f"Job ID: {result['job_id']}")
    print(f"Status: {result['status']}")
    print(f"Base Model: {result['model']}")

    # Save job info
    job_info_path = PROJECT_ROOT / f"data/finetune_job_{args.version}.json"
    with open(job_info_path, "w") as f:
        json.dump({
            "job_id": result["job_id"],
            "version": args.version,
            "base_model": args.model,
            "training_file_id": train_file_id,
            "validation_file_id": eval_file_id,
            "train_examples": train_count,
            "eval_examples": eval_count,
            "epochs": args.epochs,
            "started_at": datetime.now().isoformat()
        }, f, indent=2)

    print(f"\nJob info saved to: {job_info_path}")
    print(f"\nMonitor progress:")
    print(f"  openai api fine_tuning.jobs.retrieve -i {result['job_id']}")
    print(f"\nOr check: https://platform.openai.com/finetune")


if __name__ == "__main__":
    main()
