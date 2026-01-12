#!/usr/bin/env python3
"""
Sanitize Security Pattern Files for V7 Training

Root Cause Analysis (V6 Regression):
- V6 added 35 security patterns that contained WRONG APIs (RandomFloat, m_flSpeed)
- This created conflicting training signals - "good" examples using forbidden patterns
- Result: Wrong API avoidance dropped from 100% (V5) to 86.7% (V6)

Solution:
- Replace all wrong APIs with correct L4D2 equivalents
- Maintain security teaching while preserving API correctness signal

Usage:
    python scripts/utils/sanitize_security_patterns.py
    python scripts/utils/sanitize_security_patterns.py --dry-run
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

# API corrections - wrong -> correct
API_CORRECTIONS = {
    # Random functions
    r'\bRandomFloat\s*\(': 'GetRandomFloat(',
    r'\bRandomInt\s*\(': 'GetRandomInt(',

    # Speed property - L4D2 uses m_flLaggedMovementValue
    r'\bm_flSpeed\b': 'm_flLaggedMovementValue',
    r'\bm_flMaxSpeed\b': 'm_flLaggedMovementValue',

    # Event names
    r'"pounce"': '"lunge_pounce"',
    r'"smoker_tongue_grab"': '"tongue_grab"',
    r'"boomer_vomit"': '"player_now_it"',
    r'"charger_grab"': '"charger_carry_start"',
}

# Files to sanitize
SECURITY_PATTERN_FILES = [
    "data/anti_patterns/l4d2_security_patterns.jsonl",
    "data/anti_patterns/l4d2_redteam_attacks.jsonl",
    "data/anti_patterns/l4d2_game_exploits.jsonl",
    "data/anti_patterns/l4d2_anti_patterns.jsonl",
    "data/anti_patterns/contrastive_pairs.jsonl",
]


def count_violations(text: str) -> dict:
    """Count API violations in text."""
    counts = {}
    for pattern in API_CORRECTIONS.keys():
        matches = re.findall(pattern, text)
        if matches:
            counts[pattern] = len(matches)
    return counts


def sanitize_text(text: str) -> tuple[str, int]:
    """Apply all API corrections to text. Returns (sanitized_text, fix_count)."""
    fix_count = 0
    for wrong_pattern, correct in API_CORRECTIONS.items():
        matches = re.findall(wrong_pattern, text)
        if matches:
            fix_count += len(matches)
            text = re.sub(wrong_pattern, correct, text)
    return text, fix_count


def sanitize_example(example: dict) -> tuple[dict, int]:
    """Sanitize a single training example."""
    total_fixes = 0

    if "messages" in example:
        for msg in example["messages"]:
            if "content" in msg:
                msg["content"], fixes = sanitize_text(msg["content"])
                total_fixes += fixes

    return example, total_fixes


def process_file(filepath: Path, dry_run: bool = False) -> dict:
    """Process a single JSONL file."""
    if not filepath.exists():
        return {"file": str(filepath), "status": "not_found"}

    examples = []
    total_fixes = 0
    violations_before = {}

    # Read and analyze
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)

                # Count violations before
                for msg in ex.get("messages", []):
                    content = msg.get("content", "")
                    for pattern, count in count_violations(content).items():
                        violations_before[pattern] = violations_before.get(pattern, 0) + count

                # Sanitize
                sanitized, fixes = sanitize_example(ex)
                examples.append(sanitized)
                total_fixes += fixes

    # Write if not dry run
    if not dry_run and total_fixes > 0:
        with open(filepath, 'w') as f:
            for ex in examples:
                f.write(json.dumps(ex) + '\n')

    return {
        "file": str(filepath.name),
        "examples": len(examples),
        "fixes_applied": total_fixes,
        "violations_before": violations_before,
        "status": "dry_run" if dry_run else ("modified" if total_fixes > 0 else "clean")
    }


def main():
    dry_run = "--dry-run" in sys.argv

    print("=" * 60)
    print("L4D2 Security Pattern Sanitizer")
    print("=" * 60)
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print()

    print("API Corrections to apply:")
    for wrong, correct in API_CORRECTIONS.items():
        print(f"  {wrong} -> {correct}")
    print()

    total_fixes = 0
    results = []

    for relpath in SECURITY_PATTERN_FILES:
        filepath = PROJECT_ROOT / relpath
        print(f"Processing: {relpath}...")
        result = process_file(filepath, dry_run)
        results.append(result)

        if result["status"] == "not_found":
            print(f"  [SKIP] File not found")
        else:
            print(f"  Examples: {result['examples']}")
            print(f"  Fixes: {result['fixes_applied']}")
            if result['violations_before']:
                print(f"  Violations found:")
                for pattern, count in result['violations_before'].items():
                    print(f"    {pattern}: {count}")
            total_fixes += result.get('fixes_applied', 0)
        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files processed: {len(results)}")
    print(f"Total fixes applied: {total_fixes}")
    print()

    if dry_run:
        print("This was a DRY RUN. No files were modified.")
        print("Run without --dry-run to apply fixes.")
    else:
        print("All files have been sanitized.")
        print("Security patterns now use correct L4D2 APIs.")

    return total_fixes


if __name__ == "__main__":
    main()
