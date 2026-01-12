#!/usr/bin/env python3
"""
Start V9 Fine-tuning Job on OpenAI

Usage:
    # With Doppler for API key management
    doppler run --project local-mac-work --config dev_personal -- python scripts/training/start_v9_finetune.py

    # Or with environment variable
    OPENAI_API_KEY=sk-xxx python scripts/training/start_v9_finetune.py
"""

import os
import sys
import json
from pathlib import Path
from openai import OpenAI

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

def main():
    # Initialize OpenAI client
    client = OpenAI()

    training_file = PROCESSED_DIR / "l4d2_train_v9.jsonl"

    if not training_file.exists():
        print(f"ERROR: Training file not found: {training_file}")
        print("Run prepare_v9_dataset.py first!")
        return

    # Count examples
    with open(training_file) as f:
        num_examples = sum(1 for _ in f)

    print("=" * 60)
    print("L4D2 V9 Fine-tuning")
    print("=" * 60)
    print(f"Training file: {training_file.name}")
    print(f"Examples: {num_examples}")
    print()

    # 1. Upload file
    print(">>> Uploading training file...")
    with open(training_file, "rb") as f:
        upload_response = client.files.create(
            file=f,
            purpose="fine-tune"
        )

    file_id = upload_response.id
    print(f"    File ID: {file_id}")

    # Save file ID
    file_id_path = PROCESSED_DIR / "v9_file_id.txt"
    with open(file_id_path, 'w') as f:
        f.write(file_id)

    # 2. Create fine-tuning job
    print("\n>>> Creating fine-tuning job...")
    # Use 4 epochs for larger dataset (1100+ examples)
    # Analysis shows additional epoch improves convergence without overfitting
    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model="gpt-4o-mini-2024-07-18",
        suffix="l4d2-sourcemod-v9",
        hyperparameters={
            "n_epochs": 4,  # Up from 3 for larger dataset
            "batch_size": "auto",
            "learning_rate_multiplier": "auto"
        }
    )

    print(f"    Job ID: {job.id}")
    print(f"    Status: {job.status}")

    # Save job ID
    job_id_path = PROCESSED_DIR / "v9_job_id.txt"
    with open(job_id_path, 'w') as f:
        f.write(job.id)

    print("\n>>> Fine-tuning started!")
    print(f"    Monitor with: python scripts/training/monitor_finetune.py --job {job.id}")
    print()

    # Estimate time
    est_minutes = (num_examples * 3) / 50  # ~50 examples/minute per epoch
    print(f"    Estimated time: {est_minutes:.0f}-{est_minutes*1.5:.0f} minutes")

    return job.id

if __name__ == "__main__":
    main()
