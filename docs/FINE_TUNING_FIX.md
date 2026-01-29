# Fix for OpenAI Fine-Tuning File Format Error

## Problem
The training file `l4d2_combined_train.jsonl` contains extra fields ("type", "error_category", etc.) that are not part of the OpenAI fine-tuning format.

## Solution
I've created a fixed version: `l4d2_combined_train_fixed.jsonl`

## Upload Methods

### Method 1: Using the Python Script (Recommended - Auto-detects Doppler)

The script automatically tries to fetch your API key from:
1. Doppler project 'local-mac' (primary method)
2. Environment variable OPENAI_API_KEY
3. .env file

```bash
python scripts/fix_training_upload.py
```

### Method 2: Manual Upload with Doppler

1. **Get API key from Doppler:**
```bash
export OPENAI_API_KEY=$(doppler run --project local-mac --plain print OPENAI_API_KEY)
```

2. **Upload the fixed training file:**
```bash
curl https://api.openai.com/v1/files \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F "purpose=fine-tune" \
  -F "file=@L4D2-AI-Architect/data/processed/l4d2_combined_train_fixed.jsonl"
```

3. **Note the file ID returned (e.g., file-xxxxxxxxx)**

4. **Create the fine-tuning job:**
```bash
curl https://api.openai.com/v1/fine_tuning/jobs \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "training_file": "FILE_ID_FROM_STEP_3",
    "validation_file": "file-EbxziztK8bDzxJeCRWTr3M",
    "model": "gpt-4o-mini-2024-07-18",
    "suffix": "l4d2-v3-antipatterns-fixed",
    "hyperparameters": {
      "epoch_count": 3,
      "batch_size": "auto",
      "learning_rate_multiplier": 1.8
    },
    "seed": 974678267
  }'
```

## Python Alternative

If you prefer Python, install the requirements:
```bash
pip install openai python-dotenv
```

Then run:
```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Upload fixed training file
training_file = client.files.create(
    file=open("L4D2-AI-Architect/data/processed/l4d2_combined_train_fixed.jsonl", "rb"),
    purpose="fine-tune"
)

# Create fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=training_file.id,
    validation_file="file-EbxziztK8bDzxJeCRWTr3M",
    model="gpt-4o-mini-2024-07-18",
    suffix="l4d2-v3-antipatterns-fixed",
    hyperparameters={
        "epoch_count": 3,
        "batch_size": "auto", 
        "learning_rate_multiplier": 1.8
    },
    seed=974678267
)

print(f"Job created: {job.id}")
```

## Verification

The fixed file has been validated to contain only the required "messages" field for each example. All 661 training examples have been preserved.
