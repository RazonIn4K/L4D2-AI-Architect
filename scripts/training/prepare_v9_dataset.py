#!/usr/bin/env python3
"""
Prepare V9 Training Dataset
Combines V8 base + all new synthetic data for maximum coverage.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"

sys.path.insert(0, str(SCRIPT_DIR.parent))
from utils.security import safe_read_text, safe_write_jsonl, safe_path

def load_jsonl(filepath: Path) -> list:
    """Load JSONL file and return list of examples."""
    examples = []
    try:
        content = safe_read_text(filepath, PROJECT_ROOT)
        for line in content.strip().split('\n'):
            if line.strip():
                obj = json.loads(line)
                # Only keep messages field for OpenAI format
                if 'messages' in obj:
                    examples.append({'messages': obj['messages']})
    except Exception as e:
        print(f"  Warning: Could not load {filepath.name}: {e}")
    return examples

def main():
    print("=" * 60)
    print("L4D2 V9 Training Dataset Preparation")
    print("=" * 60)

    # Track all examples
    all_examples = []
    sources = {}

    # 1. Load V8 base (already cleaned, 779 examples)
    v8_file = PROCESSED_DIR / "l4d2_train_v8.jsonl"
    if v8_file.exists():
        v8_examples = load_jsonl(v8_file)
        print(f"\n✓ V8 Base: {len(v8_examples)} examples")
        sources["v8_base"] = len(v8_examples)
        all_examples.extend(v8_examples)
    else:
        print("\n✗ V8 Base not found!")
        return

    # 2. Load all new synthetic data from data/synthetic/
    print("\nLoading new synthetic data...")
    synthetic_files = list(SYNTHETIC_DIR.glob("*.jsonl"))

    # Skip already-incorporated files
    skip_files = {"weighted_cleaned.jsonl", "antipatterns_cleaned.jsonl"}  # Already in V8

    new_synthetic = 0
    for sf in sorted(synthetic_files):
        if sf.name in skip_files:
            print(f"  - Skipping {sf.name} (already in V8)")
            continue

        examples = load_jsonl(sf)
        if examples:
            print(f"  + {sf.name}: {len(examples)} examples")
            sources[sf.name] = len(examples)
            all_examples.extend(examples)
            new_synthetic += len(examples)

    print(f"\n✓ New Synthetic: {new_synthetic} examples")

    # 3. Deduplicate based on user message content
    print("\nDeduplicating (exact matches)...")
    seen = set()
    unique_examples = []
    exact_duplicates = 0

    for ex in all_examples:
        # Create hash from user message
        user_msg = ""
        for msg in ex.get('messages', []):
            if msg.get('role') == 'user':
                user_msg = msg.get('content', '')
                break

        if user_msg and user_msg not in seen:
            seen.add(user_msg)
            unique_examples.append(ex)
        else:
            exact_duplicates += 1

    print(f"  Removed {exact_duplicates} exact duplicates")

    # 3b. Remove near-duplicates (fuzzy matching)
    print("Removing near-duplicates...")

    def normalize(text):
        """Normalize text for fuzzy matching."""
        import re
        # Remove specific variable names and numbers
        text = re.sub(r'\b(Tank|Witch|Hunter|Smoker|Boomer|Charger|Spitter|Jockey)\b', 'INFECTED', text)
        text = re.sub(r'\d+', 'N', text)
        text = text.lower().strip()
        return text

    normalized_seen = {}
    final_examples = []
    near_duplicates = 0

    for ex in unique_examples:
        user_msg = ""
        for msg in ex.get('messages', []):
            if msg.get('role') == 'user':
                user_msg = msg.get('content', '')
                break

        norm = normalize(user_msg)
        # Check if we've seen something very similar
        if norm not in normalized_seen:
            normalized_seen[norm] = user_msg
            final_examples.append(ex)
        else:
            near_duplicates += 1

    print(f"  Removed {near_duplicates} near-duplicates")
    print(f"  Unique examples: {len(final_examples)}")

    unique_examples = final_examples

    # 4. Save V9 training file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = PROCESSED_DIR / f"l4d2_train_v9.jsonl"

    safe_write_jsonl(output_file, unique_examples, PROJECT_ROOT)

    print(f"\n✓ Saved: {output_file}")
    print(f"  Total examples: {len(unique_examples)}")

    # 5. Save stats
    stats = {
        "version": "v9",
        "timestamp": timestamp,
        "total_examples": len(unique_examples),
        "exact_duplicates_removed": exact_duplicates,
        "near_duplicates_removed": near_duplicates,
        "sources": sources
    }

    stats_file = PROCESSED_DIR / "v9_dataset_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"  Stats saved to: {stats_file}")

    # Summary
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    for source, count in sources.items():
        print(f"  {source}: {count}")
    print(f"\n  TOTAL: {len(unique_examples)} examples")
    print("=" * 60)

    return output_file

if __name__ == "__main__":
    main()
