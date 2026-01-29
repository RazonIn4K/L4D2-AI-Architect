#!/bin/bash

# Upload fixed training file to OpenAI using Doppler for authentication

echo "=== OpenAI Fine-Tuning Upload Script ==="
echo "Using Doppler project 'local-mac' for API key"
echo

# Get API key from Doppler
echo "Fetching API key from Doppler..."
OPENAI_API_KEY=$(doppler run --project local-mac --plain print OPENAI_API_KEY)

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: Could not retrieve OPENAI_API_KEY from Doppler"
    echo "Please ensure:"
    echo "1. Doppler CLI is installed and authenticated"
    echo "2. The 'local-mac' project exists"
    echo "3. The OPENAI_API_KEY secret is configured in the project"
    exit 1
fi

echo "✓ API key retrieved successfully"
echo

# Upload the fixed training file
echo "Uploading fixed training file..."
TRAINING_FILE_RESPONSE=$(curl -s https://api.openai.com/v1/files \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -F "purpose=fine-tune" \
    -F "file=@L4D2-AI-Architect/data/processed/l4d2_combined_train_fixed.jsonl")

# Extract file ID from response
TRAINING_FILE_ID=$(echo $TRAINING_FILE_RESPONSE | jq -r '.id')

if [ "$TRAINING_FILE_ID" = "null" ]; then
    echo "ERROR: Failed to upload training file"
    echo "Response: $TRAINING_FILE_RESPONSE"
    exit 1
fi

echo "✓ Training file uploaded successfully"
echo "File ID: $TRAINING_FILE_ID"
echo

# Create the fine-tuning job
echo "Creating fine-tuning job..."
JOB_RESPONSE=$(curl -s https://api.openai.com/v1/fine_tuning/jobs \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -d "{
        \"training_file\": \"$TRAINING_FILE_ID\",
        \"validation_file\": \"file-EbxziztK8bDzxJeCRWTr3M\",
        \"model\": \"gpt-4o-mini-2024-07-18\",
        \"suffix\": \"l4d2-v3-antipatterns-fixed\",
        \"hyperparameters\": {
            \"epoch_count\": 3,
            \"batch_size\": \"auto\",
            \"learning_rate_multiplier\": 1.8
        },
        \"seed\": 974678267
    }")

# Extract job ID from response
JOB_ID=$(echo $JOB_RESPONSE | jq -r '.id')

if [ "$JOB_ID" = "null" ]; then
    echo "ERROR: Failed to create fine-tuning job"
    echo "Response: $JOB_RESPONSE"
    exit 1
fi

echo "✓ Fine-tuning job created successfully!"
echo
echo "Job Details:"
echo "  Job ID: $JOB_ID"
echo "  Status: Check at https://platform.openai.com/finetune"
echo
echo "You can monitor the job status with:"
echo "  doppler run --project local-mac -- openai api fine_tuning.jobs.retrieve $JOB_ID"
