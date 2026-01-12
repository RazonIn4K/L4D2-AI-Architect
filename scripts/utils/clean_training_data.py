#!/usr/bin/env python3
"""
Clean Training Data - Remove examples with incorrect L4D2 patterns and merge anti-patterns.

This script:
1. Filters out training examples containing known-wrong L4D2 patterns
2. Merges in high-quality anti-pattern examples
3. Outputs a cleaned v4 training dataset

Usage:
    python scripts/utils/clean_training_data.py
"""

import json
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Patterns that indicate incorrect L4D2 knowledge
# If ANY of these appear in an example, the example is excluded
FORBIDDEN_PATTERNS = [
    # Wrong events (hallucinated)
    "smoker_tongue_grab",
    "smoker_tongue_release",
    "boomer_vomit",
    "boomer_bile",
    "player_biled",
    "charger_grab",
    "jockey_grab",
    "panic_start",
    "panic_end",
    "horde_start",
    "horde_end",
    "tank_health_changed",
    "spitter_death",

    # Wrong functions
    "RandomFloat(",    # Should be GetRandomFloat
    "RandomInt(",      # Should be GetRandomInt
    "GetEntityModel(", # Should use GetEntPropString
    "TakeDamage(",     # Should be SDKHooks_TakeDamage
    "RoundFloat(",     # Should be RoundToNearest

    # Wrong properties
    '"m_flSpeed"',     # Should be m_flLaggedMovementValue
    '"m_flMaxSpeed"',  # Should be m_flLaggedMovementValue in L4D2
    '"m_isInSafeRoom"', # Should be m_isInMissionStartArea
    '"m_iHealthBuffer"', # Should be m_healthBuffer (float)

    # Wrong includes
    "<l4d2_boomer>",
    "<l4d2_infected_boost>",
    "<l4d2_survivor_boost>",
    "<l4d2_infected>",
]

# More lenient patterns - only remove if they're clearly teaching wrong patterns
# (not if they appear in comments explaining what NOT to do)
CONTEXT_FORBIDDEN = [
    # "pounce" alone is tricky - "lunge_pounce" contains it
    # Only forbid if it's HookEvent("pounce")
    'HookEvent("pounce"',
    'HookEvent("pounce",',
]


def should_exclude(text: str) -> tuple[bool, str]:
    """Check if a training example should be excluded."""
    text_lower = text.lower()

    # Check absolute forbidden patterns
    for pattern in FORBIDDEN_PATTERNS:
        if pattern.lower() in text_lower:
            return True, pattern

    # Check context-sensitive patterns
    for pattern in CONTEXT_FORBIDDEN:
        if pattern.lower() in text_lower:
            return True, pattern

    return False, ""


def clean_dataset(input_path: Path, output_path: Path) -> dict:
    """Clean a JSONL dataset by removing examples with forbidden patterns."""
    stats = {
        "input_count": 0,
        "output_count": 0,
        "removed_count": 0,
        "removed_patterns": Counter(),
    }

    cleaned_examples = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            stats["input_count"] += 1

            try:
                example = json.loads(line)
                # Convert to string for pattern matching
                example_text = json.dumps(example)

                exclude, pattern = should_exclude(example_text)

                if exclude:
                    stats["removed_count"] += 1
                    stats["removed_patterns"][pattern] += 1
                else:
                    cleaned_examples.append(example)
                    stats["output_count"] += 1

            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed line")
                continue

    # Write cleaned output
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in cleaned_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    return stats


def merge_anti_patterns(cleaned_path: Path, anti_patterns_path: Path, output_path: Path) -> dict:
    """Merge anti-pattern examples into the cleaned dataset."""
    stats = {
        "base_count": 0,
        "anti_patterns_added": 0,
        "final_count": 0,
    }

    # Read cleaned base data
    examples = []
    with open(cleaned_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
                stats["base_count"] += 1

    # Read and add anti-patterns
    with open(anti_patterns_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                example = json.loads(line)
                # Remove metadata fields not needed for training
                if "type" in example:
                    del example["type"]
                if "error_category" in example:
                    del example["error_category"]
                if "wrong_pattern" in example:
                    del example["wrong_pattern"]
                if "correct_pattern" in example:
                    del example["correct_pattern"]

                examples.append(example)
                stats["anti_patterns_added"] += 1

    # Write merged output
    stats["final_count"] = len(examples)
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    return stats


def main():
    print("=" * 60)
    print("L4D2 Training Data Cleaner & Anti-Pattern Merger")
    print("=" * 60)

    # Paths
    input_path = PROJECT_ROOT / "data/processed/l4d2_combined_train_fixed.jsonl"
    cleaned_path = PROJECT_ROOT / "data/processed/l4d2_train_v4_cleaned.jsonl"
    anti_patterns_path = PROJECT_ROOT / "data/anti_patterns/l4d2_anti_patterns.jsonl"
    final_path = PROJECT_ROOT / "data/processed/l4d2_train_v4_final.jsonl"

    # Also create OpenAI format
    openai_train_path = PROJECT_ROOT / "data/openai_finetune/train_v4.jsonl"
    openai_eval_path = PROJECT_ROOT / "data/openai_finetune/eval_v4.jsonl"

    # Step 1: Clean the base dataset
    print(f"\nStep 1: Cleaning base dataset...")
    print(f"  Input: {input_path}")

    if not input_path.exists():
        print(f"  ERROR: Input file not found: {input_path}")
        sys.exit(1)

    clean_stats = clean_dataset(input_path, cleaned_path)

    print(f"  Input examples:   {clean_stats['input_count']}")
    print(f"  Removed:          {clean_stats['removed_count']}")
    print(f"  Output examples:  {clean_stats['output_count']}")

    if clean_stats['removed_patterns']:
        print(f"\n  Removed patterns breakdown:")
        for pattern, count in clean_stats['removed_patterns'].most_common():
            print(f"    [{count}x] {pattern}")

    # Step 2: Merge anti-patterns
    print(f"\nStep 2: Merging anti-patterns...")
    print(f"  Anti-patterns: {anti_patterns_path}")

    if not anti_patterns_path.exists():
        print(f"  WARNING: Anti-patterns file not found, skipping merge")
        # Just copy cleaned to final
        import shutil
        shutil.copy(cleaned_path, final_path)
        merge_stats = {"base_count": clean_stats['output_count'], "anti_patterns_added": 0, "final_count": clean_stats['output_count']}
    else:
        merge_stats = merge_anti_patterns(cleaned_path, anti_patterns_path, final_path)

    print(f"  Base examples:      {merge_stats['base_count']}")
    print(f"  Anti-patterns:      {merge_stats['anti_patterns_added']}")
    print(f"  Final examples:     {merge_stats['final_count']}")

    # Step 3: Create OpenAI format (split 90/10 for train/eval)
    print(f"\nStep 3: Creating OpenAI format...")

    with open(final_path, 'r') as f:
        all_examples = [json.loads(line) for line in f if line.strip()]

    # Shuffle for split (use fixed seed for reproducibility)
    import random
    random.seed(42)
    random.shuffle(all_examples)

    split_idx = int(len(all_examples) * 0.9)
    train_examples = all_examples[:split_idx]
    eval_examples = all_examples[split_idx:]

    openai_train_path.parent.mkdir(parents=True, exist_ok=True)

    with open(openai_train_path, 'w') as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    with open(openai_eval_path, 'w') as f:
        for ex in eval_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    print(f"  Training set:  {len(train_examples)} examples -> {openai_train_path}")
    print(f"  Eval set:      {len(eval_examples)} examples -> {openai_eval_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Original examples:     {clean_stats['input_count']}")
    print(f"Removed (bad):         {clean_stats['removed_count']}")
    print(f"Anti-patterns added:   {merge_stats['anti_patterns_added']}")
    print(f"Final training set:    {len(train_examples)}")
    print(f"Final eval set:        {len(eval_examples)}")
    print(f"\nFiles created:")
    print(f"  {cleaned_path}")
    print(f"  {final_path}")
    print(f"  {openai_train_path}")
    print(f"  {openai_eval_path}")


if __name__ == "__main__":
    main()
