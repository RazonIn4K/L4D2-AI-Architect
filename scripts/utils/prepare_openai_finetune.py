#!/usr/bin/env python3
"""
Prepare L4D2 Training Data for OpenAI Fine-Tuning

This script:
1. Validates the ChatML format is compatible with OpenAI
2. Counts tokens using tiktoken (cl100k_base encoding)
3. Splits into train/eval sets
4. Estimates fine-tuning costs
5. Outputs OpenAI-compatible JSONL files

Usage:
    pip install tiktoken
    python scripts/utils/prepare_openai_finetune.py
"""

import json
import random
from pathlib import Path

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not installed. Install with: pip install tiktoken")
    print("Using approximate token counting (words * 1.3)")


def count_tokens(text: str, encoding=None) -> int:
    """Count tokens in text using tiktoken or approximate."""
    if TIKTOKEN_AVAILABLE and encoding:
        return len(encoding.encode(text))
    else:
        # Approximate: ~1.3 tokens per word for code
        return int(len(text.split()) * 1.3)


def validate_message(msg: dict) -> tuple[bool, str]:
    """Validate a single message object."""
    if "role" not in msg:
        return False, "Missing 'role' field"
    if "content" not in msg:
        return False, "Missing 'content' field"
    if msg["role"] not in ["system", "user", "assistant"]:
        return False, f"Invalid role: {msg['role']}"
    if not isinstance(msg["content"], str):
        return False, "Content must be a string"
    return True, ""


def validate_example(example: dict, line_num: int) -> tuple[bool, str]:
    """Validate a single training example."""
    if "messages" not in example:
        return False, f"Line {line_num}: Missing 'messages' field"

    messages = example["messages"]
    if not isinstance(messages, list):
        return False, f"Line {line_num}: 'messages' must be a list"

    if len(messages) < 2:
        return False, f"Line {line_num}: Need at least 2 messages (user + assistant)"

    # Check for required assistant message
    has_assistant = any(m.get("role") == "assistant" for m in messages)
    if not has_assistant:
        return False, f"Line {line_num}: Missing assistant message"

    # Validate each message
    for i, msg in enumerate(messages):
        valid, error = validate_message(msg)
        if not valid:
            return False, f"Line {line_num}, message {i}: {error}"

    return True, ""


def main():
    print("=" * 60)
    print("OpenAI Fine-Tuning Data Preparation")
    print("=" * 60)

    # Paths
    project_root = Path(__file__).parent.parent.parent
    input_file = project_root / "data" / "processed" / "combined_train.jsonl"
    output_dir = project_root / "data" / "openai_finetune"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = output_dir / "train.jsonl"
    eval_file = output_dir / "eval.jsonl"

    # Initialize tiktoken if available
    encoding = None
    if TIKTOKEN_AVAILABLE:
        encoding = tiktoken.get_encoding("cl100k_base")
        print("✓ Using tiktoken for accurate token counting")

    # Load and validate data
    print(f"\nLoading data from: {input_file}")

    examples = []
    errors = []
    total_tokens = 0

    with open(input_file) as f:
        for line_num, line in enumerate(f, 1):
            try:
                example = json.loads(line.strip())
                valid, error = validate_example(example, line_num)

                if valid:
                    # Count tokens in this example
                    example_tokens = 0
                    for msg in example["messages"]:
                        example_tokens += count_tokens(msg["content"], encoding)
                        example_tokens += 4  # Role tokens overhead

                    example["_tokens"] = example_tokens
                    total_tokens += example_tokens
                    examples.append(example)
                else:
                    errors.append(error)

            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {e}")

    print(f"\n✓ Loaded {len(examples)} valid examples")
    if errors:
        print(f"✗ {len(errors)} examples had errors:")
        for err in errors[:5]:
            print(f"  - {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    # Token statistics
    print(f"\nToken Statistics:")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Avg tokens/example: {total_tokens // len(examples) if examples else 0}")

    # Cost estimation
    print(f"\nCost Estimation (3 epochs):")
    training_tokens = total_tokens * 3  # 3 epochs default

    # GPT-4o-mini pricing
    gpt4o_mini_cost = (training_tokens / 1_000_000) * 3.00
    print(f"  GPT-4o-mini: ~${gpt4o_mini_cost:.2f}")

    # GPT-3.5-turbo pricing
    gpt35_cost = (training_tokens / 1_000_000) * 8.00
    print(f"  GPT-3.5-turbo: ~${gpt35_cost:.2f}")

    # Split into train/eval (90/10)
    random.seed(42)
    random.shuffle(examples)

    eval_size = min(50, len(examples) // 10)  # Max 50 or 10%
    eval_examples = examples[:eval_size]
    train_examples = examples[eval_size:]

    print(f"\nDataset Split:")
    print(f"  Training: {len(train_examples)} examples")
    print(f"  Evaluation: {len(eval_examples)} examples")

    # Write output files (remove internal _tokens field)
    def clean_example(ex):
        return {"messages": ex["messages"]}

    with open(train_file, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(clean_example(ex)) + "\n")

    with open(eval_file, "w") as f:
        for ex in eval_examples:
            f.write(json.dumps(clean_example(ex)) + "\n")

    print(f"\n✓ Output files created:")
    print(f"  {train_file}")
    print(f"  {eval_file}")

    # OpenAI CLI commands
    print(f"\n{'=' * 60}")
    print("OpenAI Fine-Tuning Commands")
    print("=" * 60)
    print("""
# 1. Validate the data format:
openai api fine_tuning.jobs.create \\
  --training-file train.jsonl \\
  --model gpt-4o-mini-2024-07-18 \\
  --suffix "l4d2-sourcemod" \\
  --dry-run

# 2. Upload training file:
openai api files.create -f data/openai_finetune/train.jsonl -p fine-tune

# 3. Create fine-tuning job:
openai api fine_tuning.jobs.create \\
  -t <FILE_ID> \\
  -m gpt-4o-mini-2024-07-18 \\
  --suffix "l4d2-sourcemod"

# 4. Monitor progress:
openai api fine_tuning.jobs.list
openai api fine_tuning.jobs.retrieve -i <JOB_ID>

# Or use the web interface:
# https://platform.openai.com/finetune
""")

    print("\nDone! Your data is ready for OpenAI fine-tuning.")
    print("Visit: https://platform.openai.com/finetune to upload and start training.")


if __name__ == "__main__":
    main()
