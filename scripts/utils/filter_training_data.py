#!/usr/bin/env python3
"""
Filter and improve L4D2 training data quality.

This script:
1. Removes low-quality examples (too short, docs only, gibberish)
2. Filters for examples with clear task-to-code mapping
3. Outputs cleaned dataset for re-fine-tuning

Usage:
    python scripts/utils/filter_training_data.py
"""

import json
import re
from pathlib import Path
from typing import Optional


def is_valid_sourcepawn(code: str) -> bool:
    """Check if code looks like valid SourcePawn."""
    # Must have some SourcePawn-specific patterns
    sp_patterns = [
        r'\bvoid\b', r'\bint\b', r'\bfloat\b', r'\bbool\b',
        r'\bchar\b', r'\bAction\b', r'\bPlugin_', r'\bFCVAR_',
        r'public\s+void', r'public\s+Action', r'public\s+Plugin',
        r'CreateConVar', r'RegConsoleCmd', r'HookEvent',
        r'GetClientTeam', r'IsClientInGame', r'PrintToChat',
    ]

    has_sp_pattern = any(re.search(p, code) for p in sp_patterns)

    # Must have balanced braces (within tolerance)
    open_braces = code.count('{')
    close_braces = code.count('}')
    balanced = abs(open_braces - close_braces) <= 2

    # Must have reasonable length
    reasonable_length = 50 < len(code) < 10000

    return has_sp_pattern and balanced and reasonable_length


def is_clear_prompt(prompt: str) -> bool:
    """Check if prompt clearly describes a task."""
    # Reject vague "Implement:" style prompts
    if prompt.startswith(("Implement:", "//", "/*", "/**", "*")):
        return False

    # Reject prompts that are mostly code
    code_indicators = ['{', '}', '()', ';', 'void ', 'int ', 'float ']
    code_count = sum(prompt.count(ind) for ind in code_indicators)
    if code_count > 5:
        return False

    # Should have action words
    action_words = [
        'write', 'create', 'implement', 'add', 'make', 'build',
        'function', 'plugin', 'hook', 'command', 'timer',
        'when', 'that', 'which', 'for', 'to'
    ]
    prompt_lower = prompt.lower()
    has_action = any(word in prompt_lower for word in action_words)

    # Reasonable length
    reasonable_length = 20 < len(prompt) < 500

    return has_action and reasonable_length


def is_response_relevant(prompt: str, response: str) -> bool:
    """Check if response is relevant to prompt."""
    # Extract key terms from prompt
    prompt_words = set(re.findall(r'\b[a-z]{4,}\b', prompt.lower()))
    response_lower = response.lower()

    # At least some prompt keywords should appear in response
    matching_words = sum(1 for w in prompt_words if w in response_lower)

    return matching_words >= 2 or len(prompt_words) < 3


def score_example(messages: list) -> tuple[int, list[str]]:
    """
    Score a training example from 0-100.
    Returns (score, list of issues).
    """
    if len(messages) < 3:
        return 0, ["missing_messages"]

    system = messages[0].get("content", "")
    user = messages[1].get("content", "")
    assistant = messages[2].get("content", "")

    issues = []
    score = 100

    # Check prompt quality
    if not is_clear_prompt(user):
        issues.append("unclear_prompt")
        score -= 40

    # Check response is valid SourcePawn
    if not is_valid_sourcepawn(assistant):
        issues.append("invalid_sourcepawn")
        score -= 30

    # Check response length
    if len(assistant) < 100:
        issues.append("too_short")
        score -= 20
    elif len(assistant) > 5000:
        issues.append("too_long")
        score -= 10

    # Check for documentation-only response
    if assistant.strip().startswith(("###", "/**", "* @")):
        issues.append("documentation_only")
        score -= 30

    # Check for relevance
    if not is_response_relevant(user, assistant):
        issues.append("irrelevant_response")
        score -= 20

    # Check for complete functions
    if "void " not in assistant and "public " not in assistant and "Action " not in assistant:
        issues.append("no_function")
        score -= 15

    return max(0, score), issues


def filter_dataset(input_path: Path, output_path: Path, min_score: int = 60):
    """Filter dataset to keep only high-quality examples."""

    with open(input_path) as f:
        data = [json.loads(line) for line in f]

    print(f"Input examples: {len(data)}")

    # Score all examples
    scored = []
    for ex in data:
        score, issues = score_example(ex["messages"])
        scored.append((ex, score, issues))

    # Filter by minimum score
    filtered = [(ex, score, issues) for ex, score, issues in scored if score >= min_score]

    print(f"Filtered examples (score >= {min_score}): {len(filtered)}")

    # Show score distribution
    scores = [s for _, s, _ in scored]
    print(f"\nScore distribution:")
    for threshold in [80, 60, 40, 20, 0]:
        count = sum(1 for s in scores if s >= threshold)
        print(f"  >= {threshold}: {count} ({100*count/len(scores):.1f}%)")

    # Save filtered data
    with open(output_path, 'w') as f:
        for ex, score, issues in filtered:
            f.write(json.dumps(ex) + '\n')

    print(f"\nSaved to: {output_path}")

    # Show issue breakdown
    all_issues = {}
    for _, _, issues in scored:
        for issue in issues:
            all_issues[issue] = all_issues.get(issue, 0) + 1

    print("\nIssue breakdown (all data):")
    for issue, count in sorted(all_issues.items(), key=lambda x: -x[1]):
        print(f"  {issue}: {count} ({100*count/len(data):.1f}%)")

    return filtered


def main():
    project_root = Path(__file__).parent.parent.parent
    input_path = project_root / "data" / "processed" / "combined_train.jsonl"
    output_path = project_root / "data" / "processed" / "filtered_train.jsonl"

    print("=" * 60)
    print("L4D2 Training Data Quality Filter")
    print("=" * 60)
    print()

    filtered = filter_dataset(input_path, output_path, min_score=60)

    # Show some examples of kept vs removed
    print("\n" + "=" * 60)
    print("Sample KEPT examples (score >= 60):")
    print("=" * 60)
    for ex, score, issues in filtered[:3]:
        user = ex["messages"][1]["content"][:100]
        print(f"\nScore: {score}, Issues: {issues}")
        print(f"USER: {user}...")


if __name__ == "__main__":
    main()
