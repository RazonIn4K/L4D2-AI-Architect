#!/usr/bin/env python3
"""
Check OpenAI Fine-Tuning Job Status

Usage:
    python scripts/training/check_finetune_status.py [--job-id JOB_ID]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_api_key() -> str:
    """Get OpenAI API key from environment or Doppler."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key

    # Try Doppler
    try:
        result = subprocess.run(
            ["doppler", "secrets", "get", "OPENAI_API_KEY",
             "--project", "local-mac-work", "--config", "dev_personal", "--plain"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass

    print("ERROR: Could not get OpenAI API key")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Check fine-tuning job status")
    parser.add_argument("--job-id", help="Job ID (reads from finetune_job_v5.json if not provided)")
    parser.add_argument("--version", default="v5", help="Dataset version to look up job for")
    args = parser.parse_args()

    # Get job ID
    job_id = args.job_id
    if not job_id:
        job_file = PROJECT_ROOT / f"data/finetune_job_{args.version}.json"
        if job_file.exists():
            with open(job_file) as f:
                job_data = json.load(f)
                job_id = job_data.get("job_id")
        else:
            print(f"ERROR: No job ID provided and {job_file} not found")
            sys.exit(1)

    print(f"Checking job: {job_id}")
    print("-" * 50)

    client = OpenAI(api_key=get_api_key())

    # Get job status
    job = client.fine_tuning.jobs.retrieve(job_id)

    status_emoji = {
        "validating_files": "🔍",
        "queued": "⏳",
        "running": "🏃",
        "succeeded": "✅",
        "failed": "❌",
        "cancelled": "🚫"
    }

    emoji = status_emoji.get(job.status, "❓")
    print(f"Status: {emoji} {job.status}")
    print(f"Model: {job.model}")

    if job.fine_tuned_model:
        print(f"\n✅ FINE-TUNED MODEL READY:")
        print(f"   {job.fine_tuned_model}")

    if job.trained_tokens:
        print(f"Trained tokens: {job.trained_tokens:,}")

    if job.error:
        print(f"\n❌ Error: {job.error.message}")

    # Get recent events
    print("\nRecent events:")
    events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=10)
    for event in reversed(list(events.data)):
        timestamp = datetime.fromtimestamp(event.created_at).strftime("%H:%M:%S")
        print(f"  [{timestamp}] {event.message}")

    # Update job file with final model if succeeded
    if job.status == "succeeded" and job.fine_tuned_model:
        job_file = PROJECT_ROOT / f"data/finetune_job_{args.version}.json"
        if job_file.exists():
            with open(job_file) as f:
                job_data = json.load(f)
            job_data["fine_tuned_model"] = job.fine_tuned_model
            job_data["status"] = "succeeded"
            job_data["completed_at"] = datetime.now().isoformat()
            with open(job_file, "w") as f:
                json.dump(job_data, f, indent=2)
            print(f"\nJob info updated: {job_file}")


if __name__ == "__main__":
    main()
