#!/usr/bin/env python3
"""
Script to upload fixed training files to OpenAI for fine-tuning
"""
import os
from openai import OpenAI

# Initialize OpenAI client
# Try to get API key from Doppler first
api_key = None

# Method 1: Try Doppler CLI
try:
    import subprocess
    result = subprocess.run(['doppler', 'run', '--project', 'local-mac', '--plain', 'print', 'OPENAI_API_KEY'], 
                          capture_output=True, text=True, check=True)
    api_key = result.stdout.strip()
    print("✓ Retrieved API key from Doppler project 'local-mac'")
except (subprocess.CalledProcessError, FileNotFoundError):
    print("Could not retrieve API key from Doppler")

# Method 2: Try environment variable
if not api_key:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("✓ Retrieved API key from environment variable")

# Method 3: Try .env file
if not api_key:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("✓ Retrieved API key from .env file")

if not api_key:
    print("ERROR: OPENAI_API_KEY not found")
    print("Please ensure Doppler is configured with the 'local-mac' project or set the OPENAI_API_KEY environment variable")
    exit(1)

client = OpenAI(api_key=api_key)

# Upload the fixed training file
print("Uploading fixed training file...")
training_file = client.files.create(
    file=open("data/processed/l4d2_combined_train_fixed.jsonl", "rb"),
    purpose="fine-tune"
)
print(f"Training file uploaded successfully with ID: {training_file.id}")

# Reuse the existing validation file (no changes needed)
validation_file_id = "file-EbxziztK8bDzxJeCRWTr3M"  # From the error message

# Create the fine-tuning job
print("\nCreating fine-tuning job...")
job = client.fine_tuning.jobs.create(
    training_file=training_file.id,
    validation_file=validation_file_id,
    model="gpt-4o-mini-2024-07-18",
    suffix="l4d2-v3-antipatterns-fixed",
    hyperparameters={
        "epoch_count": 3,
        "batch_size": "auto",
        "learning_rate_multiplier": 1.8
    },
    seed=974678267
)

print(f"\nFine-tuning job created: {job.id}")
print(f"Status: {job.status}")
print(f"\nYou can monitor the job at: https://platform.openai.com/finetune")
