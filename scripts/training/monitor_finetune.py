#!/usr/bin/env python3
"""
Monitor OpenAI fine-tuning job progress.

Usage:
    python scripts/training/monitor_finetune.py                    # Use saved job ID
    python scripts/training/monitor_finetune.py ftjob-xxxxx        # Specific job ID
    python scripts/training/monitor_finetune.py --watch            # Continuous monitoring
"""

import argparse
import os
import sys
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)
    return OpenAI(api_key=api_key)


def get_job_status(client, job_id):
    """Get current job status."""
    job = client.fine_tuning.jobs.retrieve(job_id)
    return job


def print_status(job):
    """Print formatted job status."""
    print("=" * 60)
    print(f"Job ID:     {job.id}")
    print(f"Status:     {job.status}")
    print(f"Model:      {job.model}")

    if job.fine_tuned_model:
        print(f"Output:     {job.fine_tuned_model}")
    if job.trained_tokens:
        print(f"Tokens:     {job.trained_tokens:,}")
    if job.error and job.error.message:
        print(f"Error:      {job.error.message}")
    print("=" * 60)


def print_events(client, job_id, limit=10):
    """Print recent events."""
    events = client.fine_tuning.jobs.list_events(job_id, limit=limit)
    print("\nRecent Events:")
    for e in reversed(list(events)):
        level = e.level.upper() if e.level else "INFO"
        print(f"  [{level}] {e.message}")


def watch_job(client, job_id, interval=30):
    """Watch job until completion."""
    print(f"Watching job {job_id} (checking every {interval}s)...")
    print("Press Ctrl+C to stop.\n")

    while True:
        try:
            job = get_job_status(client, job_id)
            print_status(job)

            if job.status in ["succeeded", "failed", "cancelled"]:
                print_events(client, job_id)
                if job.status == "succeeded":
                    print(f"\n✓ Fine-tuned model ready: {job.fine_tuned_model}")
                    save_model_id(job.fine_tuned_model)
                return job

            print_events(client, job_id, limit=3)
            print(f"\nNext check in {interval}s...")
            time.sleep(interval)
            print("\033[H\033[J", end="")  # Clear screen

        except KeyboardInterrupt:
            print("\nStopped watching.")
            return None


def save_model_id(model_id):
    """Save model ID for use by other scripts."""
    model_file = PROJECT_ROOT / "data" / "processed" / "v8_model_id.txt"
    model_file.write_text(model_id)
    print(f"Model ID saved to: {model_file}")


def main():
    parser = argparse.ArgumentParser(description="Monitor OpenAI fine-tuning job")
    parser.add_argument("job_id", nargs="?", help="Job ID (optional, uses saved ID if not provided)")
    parser.add_argument("--watch", action="store_true", help="Continuously monitor until completion")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds (default: 30)")
    args = parser.parse_args()

    # Get job ID
    job_id = args.job_id
    if not job_id:
        job_file = PROJECT_ROOT / "data" / "processed" / "v8_job_id.txt"
        if job_file.exists():
            job_id = job_file.read_text().strip()
            print(f"Using saved job ID: {job_id}")
        else:
            print("ERROR: No job ID provided and no saved job ID found")
            sys.exit(1)

    client = get_client()

    if args.watch:
        watch_job(client, job_id, args.interval)
    else:
        job = get_job_status(client, job_id)
        print_status(job)
        print_events(client, job_id)

        if job.status == "succeeded":
            print(f"\n✓ Fine-tuned model: {job.fine_tuned_model}")
            save_model_id(job.fine_tuned_model)


if __name__ == "__main__":
    main()
