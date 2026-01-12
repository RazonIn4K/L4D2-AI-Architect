#!/usr/bin/env python3
"""
V9 Training Completion Monitor
Automatically captures model ID when training completes and saves to config.

Usage:
    doppler run --project local-mac-work --config dev_personal -- python scripts/training/monitor_v9_completion.py
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

JOB_ID = "ftjob-SHRoV0rl5ttvxJH1khIiKm2q"
CHECK_INTERVAL = 60  # Check every 60 seconds


def main():
    client = OpenAI()

    print("=" * 60)
    print("V9 Training Completion Monitor")
    print("=" * 60)
    print(f"Job ID: {JOB_ID}")
    print(f"Check interval: {CHECK_INTERVAL}s")
    print()

    last_step = 0

    while True:
        try:
            job = client.fine_tuning.jobs.retrieve(JOB_ID)

            # Get latest events
            events = client.fine_tuning.jobs.list_events(
                fine_tuning_job_id=JOB_ID, limit=3
            )

            timestamp = datetime.now().strftime("%H:%M:%S")

            if job.status == "succeeded":
                model_id = job.fine_tuned_model
                print(f"\n[{timestamp}] TRAINING COMPLETE!")
                print(f"Model ID: {model_id}")

                # Save model ID
                model_id_path = PROCESSED_DIR / "v9_model_id.txt"
                with open(model_id_path, "w") as f:
                    f.write(model_id)
                print(f"Saved to: {model_id_path}")

                # Save training stats
                stats_path = PROCESSED_DIR / "v9_training_stats.json"
                stats = {
                    "job_id": JOB_ID,
                    "model_id": model_id,
                    "status": "succeeded",
                    "completed_at": datetime.now().isoformat(),
                    "trained_tokens": job.trained_tokens if hasattr(job, 'trained_tokens') else None
                }
                with open(stats_path, "w") as f:
                    json.dump(stats, f, indent=2)
                print(f"Stats saved to: {stats_path}")

                print("\nV9 is ready for evaluation!")
                print(f"Run: python scripts/evaluation/openai_evals.py quick --model v9")
                break

            elif job.status == "failed":
                print(f"\n[{timestamp}] TRAINING FAILED!")
                if hasattr(job, 'error') and job.error:
                    print(f"Error: {job.error}")
                break

            elif job.status == "cancelled":
                print(f"\n[{timestamp}] Training was cancelled")
                break

            else:
                # Still running - show progress
                import re
                for event in events.data:
                    match = re.search(r'Step (\d+)/(\d+)', event.message)
                    if match:
                        step = int(match.group(1))
                        total = int(match.group(2))
                        if step > last_step:
                            last_step = step
                            progress = (step / total) * 100
                            loss_match = re.search(r'loss=([0-9.]+)', event.message)
                            loss = loss_match.group(1) if loss_match else "?"
                            print(f"[{timestamp}] Step {step}/{total} ({progress:.1f}%) - loss: {loss}")
                        break

        except Exception as e:
            print(f"[{timestamp}] Error checking status: {e}")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
