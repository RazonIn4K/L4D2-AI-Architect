#!/usr/bin/env python3
"""
Comprehensive evaluation script for L4D2 SourcePawn models.
Evaluates both OpenAI fine-tuned models and local TinyLlama LoRA.

Usage:
    # Evaluate OpenAI model
    python scripts/evaluation/evaluate_models.py --model openai --model-id ft:gpt-4o-mini-2024-07-18:...

    # Evaluate local TinyLlama
    python scripts/evaluation/evaluate_models.py --model local --model-path model_adapters/l4d2-lora

    # Evaluate both and compare
    python scripts/evaluation/evaluate_models.py --compare
"""

import json
import re
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_path

PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class EvalResult:
    """Single evaluation result."""
    prompt: str
    response: str
    has_code: bool = False
    has_sourcepawn_syntax: bool = False
    is_relevant: bool = False
    has_l4d2_apis: bool = False
    code_lines: int = 0
    quality_score: float = 0.0
    issues: list = field(default_factory=list)


class SourcePawnEvaluator:
    """Automatic evaluator for SourcePawn code quality."""

    # SourcePawn syntax patterns
    SOURCEPAWN_PATTERNS = [
        r'#include\s*<\w+>',           # Include statements
        r'#pragma\s+\w+',               # Pragma directives
        r'public\s+(void|Action|bool|int|float)',  # Public functions
        r'stock\s+(void|bool|int|float)',  # Stock functions
        r'native\s+\w+',                # Native declarations
        r'forward\s+\w+',               # Forward declarations
        r'methodmap\s+\w+',             # Method maps
        r'enum\s+\w+',                  # Enums
        r'CreateTimer\s*\(',            # Timer creation
        r'RegConsoleCmd\s*\(',          # Console commands
        r'RegAdminCmd\s*\(',            # Admin commands
        r'HookEvent\s*\(',              # Event hooks
        r'SDKHook\s*\(',                # SDK hooks
    ]

    # L4D2-specific API patterns
    L4D2_API_PATTERNS = [
        r'GetClientTeam\s*\(',          # Team checks
        r'L4D_GetPlayerClass',          # L4D player class
        r'IsClientInGame\s*\(',         # Client validation
        r'IsPlayerAlive\s*\(',          # Alive check
        r'SetEntityHealth\s*\(',        # Health modification
        r'CheatCommand\s*\(',           # Cheat commands
        r'L4D2Direct_',                 # L4D2Direct API
        r'L4D_',                        # L4D API prefix
        r'survivor|infected|tank|witch|boomer|hunter|smoker|spitter|charger|jockey',  # L4D2 entities
        r'saferoom|safehouse',          # Saferoom references
        r'player_death|player_spawn|player_hurt',  # L4D2 events
        r'tank_spawn|witch_spawn',      # Special infected events
    ]

    # Code quality patterns
    QUALITY_PATTERNS = {
        'has_comments': r'//.*|/\*[\s\S]*?\*/',
        'has_error_handling': r'if\s*\([^)]*==\s*(INVALID_HANDLE|null|false|-1)\)',
        'has_client_validation': r'IsValidClient|IsClientInGame|IsClientConnected',
        'has_proper_includes': r'#include\s*<(sourcemod|sdktools|left4dead|left4dhooks)>',
    }

    def evaluate_response(self, prompt: str, response: str) -> EvalResult:
        """Evaluate a single model response."""
        result = EvalResult(prompt=prompt, response=response)

        # Check if response contains code
        code_blocks = self._extract_code(response)
        result.has_code = len(code_blocks) > 0

        if not result.has_code:
            # Check for inline code without blocks
            result.has_code = any(re.search(p, response, re.IGNORECASE)
                                 for p in self.SOURCEPAWN_PATTERNS[:5])

        # Check for SourcePawn syntax
        sp_matches = sum(1 for p in self.SOURCEPAWN_PATTERNS
                        if re.search(p, response, re.IGNORECASE))
        result.has_sourcepawn_syntax = sp_matches >= 2

        # Check for L4D2-specific APIs
        l4d2_matches = sum(1 for p in self.L4D2_API_PATTERNS
                          if re.search(p, response, re.IGNORECASE))
        result.has_l4d2_apis = l4d2_matches >= 1

        # Count code lines
        result.code_lines = self._count_code_lines(response)

        # Check relevance to prompt
        result.is_relevant = self._check_relevance(prompt, response)

        # Calculate quality score
        result.quality_score = self._calculate_score(result, response)

        # Identify issues
        result.issues = self._identify_issues(result, response)

        return result

    def _extract_code(self, text: str) -> list:
        """Extract code blocks from response."""
        # Match ```sourcepawn, ```cpp, ```c, or plain ``` blocks
        pattern = r'```(?:sourcepawn|sp|cpp|c)?\n?([\s\S]*?)```'
        return re.findall(pattern, text)

    def _count_code_lines(self, text: str) -> int:
        """Count lines that look like code."""
        code_indicators = ['{', '}', ';', '#include', 'public ', 'void ', 'int ', 'float ']
        lines = text.split('\n')
        code_lines = sum(1 for line in lines
                        if any(ind in line for ind in code_indicators))
        return code_lines

    def _check_relevance(self, prompt: str, response: str) -> bool:
        """Check if response is relevant to the prompt."""
        # Extract key terms from prompt
        prompt_lower = prompt.lower()
        response_lower = response.lower()

        # Key task indicators
        task_indicators = {
            'heal': ['heal', 'health', 'SetEntityHealth'],
            'tank': ['tank', 'Tank'],
            'spawn': ['spawn', 'Create', 'Spawn'],
            'timer': ['timer', 'CreateTimer', 'Timer'],
            'damage': ['damage', 'hurt', 'OnTakeDamage'],
            'teleport': ['teleport', 'Teleport', 'SetAbsOrigin'],
            'speed': ['speed', 'velocity', 'SetEntPropFloat'],
            'kill': ['kill', 'death', 'ForcePlayerSuicide'],
            'weapon': ['weapon', 'GivePlayerItem', 'GetPlayerWeaponSlot'],
            'revive': ['revive', 'incap', 'Revive'],
            'witch': ['witch', 'Witch'],
            'horde': ['horde', 'panic', 'CheatCommand'],
            'friendly fire': ['friendly', 'teamkill', 'team damage'],
            'leaderboard': ['leaderboard', 'stats', 'score'],
            'saferoom': ['saferoom', 'safehouse', 'safe room'],
        }

        for task, indicators in task_indicators.items():
            if task in prompt_lower:
                if any(ind.lower() in response_lower for ind in indicators):
                    return True

        # Generic relevance - response mentions something from prompt
        prompt_words = set(re.findall(r'\b\w{4,}\b', prompt_lower))
        response_words = set(re.findall(r'\b\w{4,}\b', response_lower))
        overlap = prompt_words & response_words

        return len(overlap) >= 2

    def _calculate_score(self, result: EvalResult, response: str) -> float:
        """Calculate overall quality score (0-100)."""
        score = 0.0

        # Has code (30 points)
        if result.has_code:
            score += 30

        # Has SourcePawn syntax (25 points)
        if result.has_sourcepawn_syntax:
            score += 25

        # Has L4D2 APIs (20 points)
        if result.has_l4d2_apis:
            score += 20

        # Is relevant to prompt (15 points)
        if result.is_relevant:
            score += 15

        # Code length bonus (up to 10 points)
        if result.code_lines >= 5:
            score += min(10, result.code_lines / 5)

        # Quality bonuses
        for name, pattern in self.QUALITY_PATTERNS.items():
            if re.search(pattern, response, re.IGNORECASE):
                score += 2.5

        return min(100, score)

    def _identify_issues(self, result: EvalResult, response: str) -> list:
        """Identify issues with the response."""
        issues = []

        if not result.has_code:
            issues.append("No code found in response")

        if not result.has_sourcepawn_syntax:
            issues.append("Missing SourcePawn syntax patterns")

        if not result.has_l4d2_apis:
            issues.append("No L4D2-specific APIs detected")

        if not result.is_relevant:
            issues.append("Response may not be relevant to prompt")

        if result.code_lines < 5:
            issues.append("Very short code response")

        # Check for hallucinated APIs
        hallucinated = [
            'GiveHealth', 'SetUserHealth', 'TrackedUser_',
            'SDKForm.', 'ADM_INSURGENCE', 'SDKTools.Create'
        ]
        for api in hallucinated:
            if api in response:
                issues.append(f"Possibly hallucinated API: {api}")

        # Check for JSON errors
        if '{"success":false' in response or '{"error":' in response:
            issues.append("Response contains JSON error")

        return issues


def load_test_cases(path: Path) -> list:
    """Load test cases from JSONL file."""
    cases = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                cases.append(data.get('item', {}).get('input', ''))
    return [c for c in cases if c]


def evaluate_openai_model(model_id: str, test_cases: list, evaluator: SourcePawnEvaluator) -> list:
    """Evaluate OpenAI fine-tuned model."""
    try:
        from openai import OpenAI
        client = OpenAI()
    except Exception as e:
        print(f"Error: Could not initialize OpenAI client: {e}")
        print("Make sure OPENAI_API_KEY is set")
        return []

    results = []
    system_prompt = (
        "You are an expert SourcePawn and VScript developer for Left 4 Dead 2 SourceMod plugins. "
        "Write clean, well-documented code with proper error handling."
    )

    for i, prompt in enumerate(test_cases, 1):
        print(f"  Evaluating prompt {i}/{len(test_cases)}...", end=" ", flush=True)
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.3
            )
            output = response.choices[0].message.content
            result = evaluator.evaluate_response(prompt, output)
            results.append(result)
            print(f"Score: {result.quality_score:.1f}")
        except Exception as e:
            print(f"Error: {e}")
            results.append(EvalResult(prompt=prompt, response=str(e), issues=[str(e)]))

    return results


def generate_report(results: list, model_name: str, output_path: Path) -> None:
    """Generate evaluation report."""
    if not results:
        print("No results to report")
        return

    # Calculate statistics
    total = len(results)
    has_code = sum(1 for r in results if r.has_code)
    has_syntax = sum(1 for r in results if r.has_sourcepawn_syntax)
    has_l4d2 = sum(1 for r in results if r.has_l4d2_apis)
    is_relevant = sum(1 for r in results if r.is_relevant)
    avg_score = sum(r.quality_score for r in results) / total if total > 0 else 0

    # Passing threshold
    passing = sum(1 for r in results if r.quality_score >= 50)

    report = f"""# L4D2 SourcePawn Model Evaluation Report

**Model**: {model_name}
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Test Cases**: {total}

## Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| Has Code | {has_code} | {has_code/total*100:.1f}% |
| Has SourcePawn Syntax | {has_syntax} | {has_syntax/total*100:.1f}% |
| Has L4D2 APIs | {has_l4d2} | {has_l4d2/total*100:.1f}% |
| Relevant to Prompt | {is_relevant} | {is_relevant/total*100:.1f}% |
| **Passing (≥50)** | **{passing}** | **{passing/total*100:.1f}%** |

**Average Quality Score**: {avg_score:.1f}/100

## Detailed Results

"""

    for i, result in enumerate(results, 1):
        status = "✅ PASS" if result.quality_score >= 50 else "❌ FAIL"
        report += f"""### Test Case {i}: {status} (Score: {result.quality_score:.1f})

**Prompt**: {result.prompt[:100]}...

**Metrics**:
- Has Code: {'✓' if result.has_code else '✗'}
- SourcePawn Syntax: {'✓' if result.has_sourcepawn_syntax else '✗'}
- L4D2 APIs: {'✓' if result.has_l4d2_apis else '✗'}
- Relevant: {'✓' if result.is_relevant else '✗'}
- Code Lines: {result.code_lines}

"""
        if result.issues:
            report += "**Issues**:\n"
            for issue in result.issues:
                report += f"- {issue}\n"
        report += "\n---\n\n"

    # Write report (with path validation)
    validated_path = safe_path(output_path, PROJECT_ROOT, create_parents=True)
    with open(validated_path, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {output_path}")
    print(f"\n{'='*50}")
    print(f"RESULTS SUMMARY: {model_name}")
    print(f"{'='*50}")
    print(f"Passing Rate: {passing}/{total} ({passing/total*100:.1f}%)")
    print(f"Average Score: {avg_score:.1f}/100")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate L4D2 SourcePawn models')
    parser.add_argument('--model', choices=['openai', 'local'], default='openai',
                       help='Model type to evaluate')
    parser.add_argument('--model-id', type=str,
                       help='OpenAI model ID (for --model openai)')
    parser.add_argument('--model-path', type=str, default='model_adapters/l4d2-mistral-v10plus-lora/final',
                       help='Path to local model (for --model local)')
    parser.add_argument('--test-cases', type=str, default='data/eval_test_cases.jsonl',
                       help='Path to test cases JSONL')
    parser.add_argument('--output', type=str, default='docs/model_evaluation.md',
                       help='Output report path')
    args = parser.parse_args()

    # Load test cases (with path validation)
    try:
        test_cases_path = safe_path(args.test_cases, PROJECT_ROOT)
    except ValueError as e:
        print(f"Error: Invalid test cases path: {e}")
        sys.exit(1)

    if not test_cases_path.exists():
        print(f"Error: Test cases not found at {test_cases_path}")
        sys.exit(1)

    test_cases = load_test_cases(test_cases_path)
    print(f"Loaded {len(test_cases)} test cases")

    # Initialize evaluator
    evaluator = SourcePawnEvaluator()

    if args.model == 'openai':
        if not args.model_id:
            print("Error: --model-id required for OpenAI evaluation")
            sys.exit(1)

        print(f"\nEvaluating OpenAI model: {args.model_id}")
        results = evaluate_openai_model(args.model_id, test_cases, evaluator)
        model_name = args.model_id
    else:
        print("Local model evaluation not yet implemented")
        print("Use the Vultr instance to run inference with TinyLlama LoRA")
        sys.exit(0)

    # Generate report
    output_path = PROJECT_ROOT / args.output
    generate_report(results, model_name, output_path)


if __name__ == '__main__':
    main()
