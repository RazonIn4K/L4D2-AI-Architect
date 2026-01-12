#!/usr/bin/env python3
"""
Create V11 Training Dataset

Merges all V10 data sources:
1. data/processed/l4d2_train_v10.jsonl (771 examples)
2. All synthetic_v10_batch*.jsonl files
3. synthetic_v10_advanced.jsonl
4. synthetic_v10_combined.jsonl

Removes duplicates based on the 'messages' content.
"""

import json
import hashlib
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

sys.path.insert(0, str(SCRIPT_DIR.parent))
from utils.security import safe_read_text, safe_write_jsonl, safe_path


def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of examples."""
    examples = []
    try:
        content = safe_read_text(str(filepath), PROJECT_ROOT)
        for line_num, line in enumerate(content.strip().split('\n'), 1):
            if line.strip():
                try:
                    obj = json.loads(line)
                    # Only keep messages field for training format
                    if 'messages' in obj:
                        examples.append({'messages': obj['messages']})
                except json.JSONDecodeError as e:
                    print(f"  Warning: Invalid JSON on line {line_num} in {filepath.name}: {e}")
    except FileNotFoundError:
        print(f"  Warning: File not found: {filepath.name}")
    except Exception as e:
        print(f"  Warning: Could not load {filepath.name}: {e}")
    return examples


def get_message_hash(messages: List[Dict[str, str]]) -> str:
    """Create a hash of the messages content for deduplication."""
    # Create a normalized string from all messages
    content_parts = []
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        content_parts.append(f"{role}:{content}")

    full_content = '|'.join(content_parts)
    return hashlib.md5(full_content.encode('utf-8')).hexdigest()


def get_user_message(example: Dict[str, Any]) -> str:
    """Extract the user message from an example."""
    for msg in example.get('messages', []):
        if msg.get('role') == 'user':
            return msg.get('content', '')
    return ''


def main():
    print("=" * 70)
    print("L4D2 V11 Training Dataset Creation")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Track all examples and sources
    all_examples = []
    sources = {}

    # 1. Load V10 base dataset
    print("Step 1: Loading V10 base dataset...")
    v10_file = PROCESSED_DIR / "l4d2_train_v10.jsonl"
    if v10_file.exists():
        v10_examples = load_jsonl(v10_file)
        print(f"  [+] l4d2_train_v10.jsonl: {len(v10_examples)} examples")
        sources["l4d2_train_v10.jsonl"] = len(v10_examples)
        all_examples.extend(v10_examples)
    else:
        print("  [!] l4d2_train_v10.jsonl not found!")
        return

    # 2. Load all synthetic_v10_batch*.jsonl files
    print("\nStep 2: Loading synthetic batch files...")
    batch_files = sorted(PROCESSED_DIR.glob("synthetic_v10_batch*.jsonl"))

    batch_total = 0
    for batch_file in batch_files:
        examples = load_jsonl(batch_file)
        if examples:
            print(f"  [+] {batch_file.name}: {len(examples)} examples")
            sources[batch_file.name] = len(examples)
            all_examples.extend(examples)
            batch_total += len(examples)

    print(f"  Batch files subtotal: {batch_total} examples")

    # 3. Load synthetic_v10_advanced.jsonl
    print("\nStep 3: Loading synthetic_v10_advanced.jsonl...")
    advanced_file = PROCESSED_DIR / "synthetic_v10_advanced.jsonl"
    if advanced_file.exists():
        advanced_examples = load_jsonl(advanced_file)
        print(f"  [+] synthetic_v10_advanced.jsonl: {len(advanced_examples)} examples")
        sources["synthetic_v10_advanced.jsonl"] = len(advanced_examples)
        all_examples.extend(advanced_examples)
    else:
        print("  [!] synthetic_v10_advanced.jsonl not found")

    # 4. Load synthetic_v10_combined.jsonl
    print("\nStep 4: Loading synthetic_v10_combined.jsonl...")
    combined_file = PROCESSED_DIR / "synthetic_v10_combined.jsonl"
    if combined_file.exists():
        combined_examples = load_jsonl(combined_file)
        print(f"  [+] synthetic_v10_combined.jsonl: {len(combined_examples)} examples")
        sources["synthetic_v10_combined.jsonl"] = len(combined_examples)
        all_examples.extend(combined_examples)
    else:
        print("  [!] synthetic_v10_combined.jsonl not found")

    # Summary before deduplication
    print("\n" + "-" * 70)
    print(f"Total examples before deduplication: {len(all_examples)}")
    print("-" * 70)

    # 5. Deduplicate based on full message content hash
    print("\nStep 5: Removing exact duplicates (full message hash)...")
    seen_hashes = set()
    unique_examples = []
    exact_duplicates = 0

    for example in all_examples:
        msg_hash = get_message_hash(example.get('messages', []))
        if msg_hash not in seen_hashes:
            seen_hashes.add(msg_hash)
            unique_examples.append(example)
        else:
            exact_duplicates += 1

    print(f"  Removed {exact_duplicates} exact duplicates")
    print(f"  Remaining: {len(unique_examples)} examples")

    # 6. Also remove duplicates based on user message only (catches near-duplicates)
    print("\nStep 6: Removing duplicates with same user message...")
    seen_user_msgs = set()
    final_examples = []
    user_duplicates = 0

    for example in unique_examples:
        user_msg = get_user_message(example)
        # Normalize for comparison
        normalized_user = user_msg.strip().lower()

        if normalized_user and normalized_user not in seen_user_msgs:
            seen_user_msgs.add(normalized_user)
            final_examples.append(example)
        elif not normalized_user:
            # Keep examples without user messages (edge case)
            final_examples.append(example)
        else:
            user_duplicates += 1

    print(f"  Removed {user_duplicates} user message duplicates")
    print(f"  Final unique examples: {len(final_examples)}")

    # 7. Save V11 dataset
    print("\nStep 7: Saving V11 dataset...")
    output_file = PROCESSED_DIR / "l4d2_train_v11.jsonl"
    safe_write_jsonl(str(output_file), final_examples, PROJECT_ROOT)
    print(f"  Saved to: {output_file}")

    # 8. Save statistics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats = {
        "version": "v11",
        "created": timestamp,
        "total_unique_examples": len(final_examples),
        "original_total": len(all_examples),
        "exact_duplicates_removed": exact_duplicates,
        "user_msg_duplicates_removed": user_duplicates,
        "sources": sources,
        "source_files_count": len(sources)
    }

    stats_file = PROCESSED_DIR / "v11_dataset_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved to: {stats_file}")

    # Final Summary
    print("\n" + "=" * 70)
    print("V11 Dataset Summary")
    print("=" * 70)
    print("\nSource files:")
    for source, count in sources.items():
        print(f"  {source}: {count} examples")

    print("\nDeduplication:")
    print(f"  Original total: {len(all_examples)}")
    print(f"  Exact duplicates removed: {exact_duplicates}")
    print(f"  User message duplicates removed: {user_duplicates}")
    print(f"  Total duplicates removed: {exact_duplicates + user_duplicates}")

    print("\nFinal Result:")
    print(f"  V11 unique examples: {len(final_examples)}")
    print("=" * 70)

    return output_file, len(final_examples)


if __name__ == "__main__":
    result = main()
    if result:
        output_file, count = result
        print(f"\nSuccess! Created {output_file} with {count} unique examples.")
