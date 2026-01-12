#!/usr/bin/env python3
"""
Fix API Contamination in Training Data

This script performs surgical corrections to training data to fix
wrong L4D2 API usage that contaminates model outputs.

Based on benchmark analysis showing:
- RandomInt used instead of GetRandomInt (155 vs 89 occurrences)
- RandomFloat used instead of GetRandomFloat (916 vs 847 occurrences)

Unlike clean_training_data.py which removes entire examples,
this script REPLACES wrong patterns with correct ones to preserve
valuable training data.

Usage:
    python scripts/utils/fix_api_contamination.py
    python scripts/utils/fix_api_contamination.py --input data/processed/l4d2_train_v15.jsonl --output data/processed/l4d2_train_v16_fixed.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent

# ============================================================================
# API REPLACEMENTS - Pattern -> Correct Pattern
# ============================================================================

# Simple string replacements (order matters - more specific first)
# NOTE: These are applied AFTER regex replacements
SIMPLE_REPLACEMENTS = [
    # Entity health (Tank health issue)
    ("GetClientHealth(client)", "GetEntProp(client, Prop_Send, \"m_iHealth\")"),

    # Movement speed
    ('"m_flSpeed"', '"m_flLaggedMovementValue"'),
    ('"m_flMaxSpeed"', '"m_flLaggedMovementValue"'),

    # Saferoom detection
    ('"m_isInSafeRoom"', '"m_isInMissionStartArea"'),

    # Health buffer (use correct type)
    ('"m_iHealthBuffer"', '"m_healthBuffer"'),
]

# Regex replacements for more complex patterns
# Use negative lookbehind to avoid matching GetRandomInt/GetRandomFloat
REGEX_REPLACEMENTS = [
    # Fix RandomInt - only match if NOT preceded by "Get"
    (r'(?<!Get)RandomInt\s*\(', 'GetRandomInt('),
    (r'(?<!Get)RandomFloat\s*\(', 'GetRandomFloat('),

    # Fix wrong events
    (r'HookEvent\s*\(\s*"pounce"\s*,', 'HookEvent("lunge_pounce",'),
    (r'HookEvent\s*\(\s*"smoker_tongue_grab"\s*,', 'HookEvent("tongue_grab",'),
    (r'HookEvent\s*\(\s*"charger_grab"\s*,', 'HookEvent("charger_carry_start",'),
    (r'HookEvent\s*\(\s*"boomer_vomit"\s*,', 'HookEvent("player_now_it",'),
]

# Patterns to completely remove (still bad even after fixes)
REMOVE_IF_CONTAINS = [
    "l4d2_boomer.inc",
    "l4d2_infected_boost.inc",
    "prop_tank.inc",
    "GetEntityModel(",  # Hallucinated function
    "TakeDamage(",      # Should be SDKHooks_TakeDamage
]


def apply_replacements(text: str) -> Tuple[str, List[str]]:
    """Apply all replacements to text. Returns (fixed_text, list_of_fixes)."""
    fixes_applied = []

    # Apply simple replacements
    for old, new in SIMPLE_REPLACEMENTS:
        if old in text:
            count = text.count(old)
            text = text.replace(old, new)
            fixes_applied.append(f"{old} -> {new} ({count}x)")

    # Apply regex replacements
    for pattern, replacement in REGEX_REPLACEMENTS:
        matches = re.findall(pattern, text)
        if matches:
            text = re.sub(pattern, replacement, text)
            fixes_applied.append(f"regex:{pattern[:30]}... ({len(matches)}x)")

    return text, fixes_applied


def should_remove(text: str) -> Tuple[bool, str]:
    """Check if example should be removed entirely."""
    for pattern in REMOVE_IF_CONTAINS:
        if pattern in text:
            return True, pattern
    return False, ""


def fix_dataset(input_path: Path, output_path: Path, verbose: bool = False) -> Dict:
    """Fix a JSONL dataset by correcting wrong API patterns."""
    stats = {
        "input_count": 0,
        "output_count": 0,
        "fixed_count": 0,
        "removed_count": 0,
        "unchanged_count": 0,
        "fixes_applied": Counter(),
        "removed_patterns": Counter(),
    }

    fixed_examples = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            stats["input_count"] += 1

            try:
                example = json.loads(line)
                example_text = json.dumps(example, ensure_ascii=False)

                # Check if should be removed entirely
                should_rm, rm_pattern = should_remove(example_text)
                if should_rm:
                    stats["removed_count"] += 1
                    stats["removed_patterns"][rm_pattern] += 1
                    if verbose:
                        print(f"  [REMOVE] Line {line_num}: {rm_pattern}")
                    continue

                # Apply fixes
                fixed_text, fixes = apply_replacements(example_text)

                if fixes:
                    stats["fixed_count"] += 1
                    for fix in fixes:
                        stats["fixes_applied"][fix] += 1
                    if verbose:
                        print(f"  [FIX] Line {line_num}: {', '.join(fixes)}")

                    # Parse back to dict
                    fixed_example = json.loads(fixed_text)
                    fixed_examples.append(fixed_example)
                else:
                    stats["unchanged_count"] += 1
                    fixed_examples.append(example)

                stats["output_count"] += 1

            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed line {line_num}: {e}")
                continue

    # Write fixed output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in fixed_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    return stats


def main():
    parser = argparse.ArgumentParser(description="Fix API contamination in L4D2 training data")
    parser.add_argument("--input", type=str, default="data/processed/l4d2_train_v15.jsonl",
                       help="Input JSONL file")
    parser.add_argument("--output", type=str, default="data/processed/l4d2_train_v16_fixed.jsonl",
                       help="Output JSONL file")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show each fix as it happens")
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output

    print("=" * 70)
    print("L4D2 API Contamination Fixer")
    print("=" * 70)
    print(f"\nInput:  {input_path}")
    print(f"Output: {output_path}")

    if not input_path.exists():
        print(f"\nERROR: Input file not found: {input_path}")
        sys.exit(1)

    print("\nProcessing...")
    stats = fix_dataset(input_path, output_path, args.verbose)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nInput examples:     {stats['input_count']}")
    print(f"Output examples:    {stats['output_count']}")
    print(f"Fixed examples:     {stats['fixed_count']}")
    print(f"Removed examples:   {stats['removed_count']}")
    print(f"Unchanged:          {stats['unchanged_count']}")

    if stats['fixes_applied']:
        print(f"\nFixes applied:")
        for fix, count in stats['fixes_applied'].most_common(20):
            print(f"  [{count:4d}x] {fix}")

    if stats['removed_patterns']:
        print(f"\nRemoved patterns:")
        for pattern, count in stats['removed_patterns'].most_common():
            print(f"  [{count:4d}x] {pattern}")

    print(f"\n✓ Fixed dataset saved to: {output_path}")

    # Verify the fix worked (use regex to avoid counting GetRandomInt as RandomInt)
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    wrong_ri = 0
    right_ri = 0
    wrong_rf = 0
    right_rf = 0

    with open(output_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                # Parse and re-serialize to get decoded content
                obj = json.loads(line)
                text = json.dumps(obj)
                # Count using negative lookbehind
                wrong_ri += len(re.findall(r'(?<!Get)RandomInt\s*\(', text))
                right_ri += len(re.findall(r'GetRandomInt\s*\(', text))
                wrong_rf += len(re.findall(r'(?<!Get)RandomFloat\s*\(', text))
                right_rf += len(re.findall(r'GetRandomFloat\s*\(', text))
            except:
                pass

    print(f"\nRandom API usage in output:")
    print(f"  GetRandomInt (correct):   {right_ri}")
    print(f"  RandomInt (WRONG):        {wrong_ri}")
    print(f"  GetRandomFloat (correct): {right_rf}")
    print(f"  RandomFloat (WRONG):      {wrong_rf}")

    if wrong_ri == 0 and wrong_rf == 0:
        print("\n✓ All wrong API patterns fixed!")
    else:
        print(f"\n⚠ Some wrong patterns remain - may need manual review")


if __name__ == "__main__":
    main()
