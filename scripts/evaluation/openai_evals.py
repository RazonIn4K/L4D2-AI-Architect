#!/usr/bin/env python3
"""
OpenAI Evals Integration for L4D2 SourcePawn Model

Uses OpenAI's Evals API to systematically evaluate the fine-tuned model
and identify areas where the training dataset needs improvement.

Usage:
    # Create an eval
    python scripts/evaluation/openai_evals.py create --name l4d2-sourcepawn-v7

    # Run eval on model
    python scripts/evaluation/openai_evals.py run --eval-id eval_xxx

    # Analyze results to identify dataset gaps
    python scripts/evaluation/openai_evals.py analyze --run-id run_xxx
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent.parent

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

# Model Configuration
AVAILABLE_MODELS = {
    "v7": "ft:gpt-4o-mini-2024-07-18:highencodelearning:l4d2-sourcemod-v7:CvTBCVPi",
    "v8": None,  # Will be loaded from file if available
    "v9": None   # Will be loaded from file if available
}

# Load V8 model ID if available
V8_MODEL_FILE = PROJECT_ROOT / "data" / "processed" / "v8_model_id.txt"
if V8_MODEL_FILE.exists():
    AVAILABLE_MODELS["v8"] = V8_MODEL_FILE.read_text().strip()

# Load V9 model ID if available
V9_MODEL_FILE = PROJECT_ROOT / "data" / "processed" / "v9_model_id.txt"
if V9_MODEL_FILE.exists():
    AVAILABLE_MODELS["v9"] = V9_MODEL_FILE.read_text().strip()

# Default to latest available model
if AVAILABLE_MODELS.get("v9"):
    MODEL_ID = AVAILABLE_MODELS["v9"]
elif AVAILABLE_MODELS.get("v8"):
    MODEL_ID = AVAILABLE_MODELS["v8"]
else:
    MODEL_ID = AVAILABLE_MODELS["v7"]

# L4D2-specific test cases for evaluation
L4D2_TEST_CASES = [
    {
        "id": "tank_spawn",
        "prompt": "Write a SourcePawn plugin that announces when a Tank spawns and shows its health",
        "expected_patterns": ["tank_spawn", "GetEntProp", "m_iHealth", "PrintToChatAll"],
        "forbidden_patterns": ["RandomFloat", "pounce"],
        "category": "events"
    },
    {
        "id": "hunter_pounce",
        "prompt": "Write a SourcePawn plugin that tracks Hunter pounce damage",
        "expected_patterns": ["lunge_pounce", "GetEventInt", "damage"],
        "forbidden_patterns": ["pounce", "smoker_tongue_grab"],
        "category": "special_infected"
    },
    {
        "id": "speed_boost",
        "prompt": "Write a SourcePawn plugin that gives survivors a speed boost when they kill special infected",
        "expected_patterns": ["player_death", "m_flLaggedMovementValue", "SetEntPropFloat", "CreateTimer"],
        "forbidden_patterns": ["m_flSpeed", "m_flMaxSpeed", "RandomFloat"],
        "category": "survivor_mechanics"
    },
    {
        "id": "random_timer",
        "prompt": "Write a SourcePawn plugin with a random timer between 10 and 30 seconds",
        "expected_patterns": ["GetRandomFloat", "CreateTimer"],
        "forbidden_patterns": ["RandomFloat", "RandomInt"],
        "category": "api_correctness"
    },
    {
        "id": "smoker_grab",
        "prompt": "Write a SourcePawn plugin that detects when a Smoker grabs a survivor",
        "expected_patterns": ["tongue_grab", "GetClientOfUserId"],
        "forbidden_patterns": ["smoker_tongue_grab", "smoker_grab"],
        "category": "special_infected"
    },
    {
        "id": "charger_carry",
        "prompt": "Write a SourcePawn plugin that tracks Charger carries",
        "expected_patterns": ["charger_carry_start", "GetEventInt"],
        "forbidden_patterns": ["charger_grab", "charger_carry"],
        "category": "special_infected"
    },
    {
        "id": "bile_throw",
        "prompt": "Write a SourcePawn plugin that detects when a survivor gets biled",
        "expected_patterns": ["player_now_it", "GetClientOfUserId"],
        "forbidden_patterns": ["boomer_vomit", "boomer_bile"],
        "category": "special_infected"
    },
    {
        "id": "friendly_fire",
        "prompt": "Write a SourcePawn plugin that prevents friendly fire damage",
        "expected_patterns": ["SDKHooks_TakeDamage", "OnTakeDamage", "GetClientTeam"],
        "forbidden_patterns": ["TakeDamage("],
        "category": "survivor_mechanics"
    },
    {
        "id": "saferoom_heal",
        "prompt": "Write a SourcePawn plugin that heals survivors when they enter a saferoom",
        "expected_patterns": ["player_entered", "SetEntityHealth", "IsClientInGame"],
        "forbidden_patterns": [],
        "category": "map_events"
    },
    {
        "id": "witch_proximity",
        "prompt": "Write a SourcePawn plugin that warns players when they get close to a Witch",
        "expected_patterns": ["witch", "GetEntPropVector", "GetVectorDistance", "PrintToChat"],
        "forbidden_patterns": [],
        "category": "special_infected"
    },
    # V9 Additional Test Cases - Map Events
    {
        "id": "finale_start",
        "prompt": "Write a SourcePawn plugin that detects when a finale starts",
        "expected_patterns": ["finale_start", "HookEvent", "GetEventString"],
        "forbidden_patterns": [],
        "category": "map_events"
    },
    {
        "id": "rescue_vehicle",
        "prompt": "Write a SourcePawn plugin that announces when the rescue vehicle is arriving",
        "expected_patterns": ["finale_vehicle", "PrintToChatAll"],
        "forbidden_patterns": [],
        "category": "map_events"
    },
    {
        "id": "gauntlet_run",
        "prompt": "Write a SourcePawn plugin that triggers an event at the start of a gauntlet run",
        "expected_patterns": ["gauntlet_finale_start", "HookEvent"],
        "forbidden_patterns": [],
        "category": "map_events"
    },
    # V9 Additional Test Cases - Special Infected Advanced
    {
        "id": "jockey_ride",
        "prompt": "Write a SourcePawn plugin that tracks how long a Jockey rides a survivor",
        "expected_patterns": ["jockey_ride", "jockey_ride_end", "GetEventFloat", "GetGameTime"],
        "forbidden_patterns": [],
        "category": "special_infected_advanced"
    },
    {
        "id": "spitter_spit",
        "prompt": "Write a SourcePawn plugin that detects when a Spitter creates an acid pool",
        "expected_patterns": ["ability_use", "GetEventString", "Spitter"],
        "forbidden_patterns": [],
        "category": "special_infected_advanced"
    },
    {
        "id": "tank_rock",
        "prompt": "Write a SourcePawn plugin that tracks Tank rock throws",
        "expected_patterns": ["tank_rock", "GetClientOfUserId"],
        "forbidden_patterns": [],
        "category": "special_infected_advanced"
    },
    # V9 Additional Test Cases - Error Handling
    {
        "id": "null_client_check",
        "prompt": "Write a SourcePawn plugin with proper client validation before operations",
        "expected_patterns": ["IsClientInGame", "IsClientConnected", "IsPlayerAlive"],
        "forbidden_patterns": [],
        "category": "error_handling"
    },
    {
        "id": "entity_validation",
        "prompt": "Write a SourcePawn plugin that safely handles entity operations",
        "expected_patterns": ["IsValidEntity", "IsValidEdict"],
        "forbidden_patterns": [],
        "category": "error_handling"
    }
]

# Grader prompt for evaluating L4D2 code quality
GRADER_PROMPT = """You are evaluating SourcePawn code for Left 4 Dead 2 plugins.

Score the code on these criteria (each 0-10):

1. **Syntax Correctness** (0-10): Valid SourcePawn syntax, proper includes, correct function signatures
2. **L4D2 API Correctness** (0-10): Uses correct L4D2-specific APIs and events:
   - GetRandomFloat() NOT RandomFloat()
   - lunge_pounce NOT pounce
   - tongue_grab NOT smoker_tongue_grab
   - player_now_it NOT boomer_vomit
   - charger_carry_start NOT charger_grab
   - m_flLaggedMovementValue for speed NOT m_flSpeed
3. **Task Completion** (0-10): Does the code accomplish the requested task?
4. **Code Quality** (0-10): Clean structure, proper error handling, comments

Expected patterns that SHOULD appear: {expected}
Forbidden patterns that should NOT appear: {forbidden}

Respond with JSON:
{{
    "syntax_score": <0-10>,
    "api_score": <0-10>,
    "task_score": <0-10>,
    "quality_score": <0-10>,
    "total_score": <0-40>,
    "pass": <true if total >= 28>,
    "expected_found": [<list of expected patterns found>],
    "forbidden_found": [<list of forbidden patterns found - should be empty>],
    "issues": [<list of specific issues>],
    "suggestions": [<suggestions for improvement>]
}}
"""


def get_client() -> OpenAI:
    """Get OpenAI client."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)
    return OpenAI(api_key=api_key)


def generate_code(client: OpenAI, prompt: str, temperature: float = 0.1, model_id: str = None) -> str:
    """Generate code using the fine-tuned model."""
    response = client.chat.completions.create(
        model=model_id or MODEL_ID,
        messages=[
            {"role": "system", "content": """You are an expert SourcePawn developer for L4D2.
CRITICAL: Use GetRandomFloat() NOT RandomFloat(). Use lunge_pounce NOT pounce.
Use tongue_grab NOT smoker_tongue_grab. Use player_now_it NOT boomer_vomit."""},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2048,
        temperature=temperature
    )
    return response.choices[0].message.content


def grade_code(client: OpenAI, code: str, test_case: Dict) -> Dict:
    """Grade generated code using GPT-4."""
    grader_prompt = GRADER_PROMPT.format(
        expected=", ".join(test_case["expected_patterns"]),
        forbidden=", ".join(test_case["forbidden_patterns"]) or "none"
    )
    
    response = client.chat.completions.create(
        model="gpt-4o",  # Use GPT-4 for grading
        messages=[
            {"role": "system", "content": grader_prompt},
            {"role": "user", "content": f"Grade this code:\n\n```sourcepawn\n{code}\n```"}
        ],
        max_tokens=1000,
        temperature=0
    )
    
    try:
        # Parse JSON response
        content = response.choices[0].message.content
        # Extract JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        return json.loads(content.strip())
    except (json.JSONDecodeError, IndexError):
        return {"error": "Failed to parse grader response", "raw": response.choices[0].message.content}


def run_evaluation(client: OpenAI, test_cases: List[Dict] = None, num_runs: int = 1, model_id: str = None) -> Dict:
    """Run full evaluation on all test cases."""
    if test_cases is None:
        test_cases = L4D2_TEST_CASES

    model_id = model_id or MODEL_ID
    results = {
        "model": model_id,
        "timestamp": datetime.now().isoformat(),
        "num_runs": num_runs,
        "test_results": [],
        "summary": {
            "total_tests": len(test_cases) * num_runs,
            "passed": 0,
            "failed": 0,
            "avg_scores": {
                "syntax": 0,
                "api": 0,
                "task": 0,
                "quality": 0,
                "total": 0
            },
            "by_category": {},
            "common_issues": [],
            "dataset_gaps": []
        }
    }
    
    all_issues = []
    category_scores = {}
    
    for run_num in range(num_runs):
        print(f"\n--- Run {run_num + 1}/{num_runs} ---")
        
        for i, test_case in enumerate(test_cases):
            print(f"[{i+1}/{len(test_cases)}] Testing: {test_case['id']}...", end=" ", flush=True)
            
            # Generate code
            code = generate_code(client, test_case["prompt"], model_id=model_id)
            
            # Grade code
            grade = grade_code(client, code, test_case)
            
            if "error" not in grade:
                test_result = {
                    "test_id": test_case["id"],
                    "run": run_num + 1,
                    "category": test_case["category"],
                    "prompt": test_case["prompt"],
                    "generated_code": code,
                    "grade": grade,
                    "passed": grade.get("pass", False)
                }
                
                # Update summary
                if grade.get("pass", False):
                    results["summary"]["passed"] += 1
                    print("PASS")
                else:
                    results["summary"]["failed"] += 1
                    print("FAIL")
                
                # Track scores
                results["summary"]["avg_scores"]["syntax"] += grade.get("syntax_score", 0)
                results["summary"]["avg_scores"]["api"] += grade.get("api_score", 0)
                results["summary"]["avg_scores"]["task"] += grade.get("task_score", 0)
                results["summary"]["avg_scores"]["quality"] += grade.get("quality_score", 0)
                results["summary"]["avg_scores"]["total"] += grade.get("total_score", 0)
                
                # Track by category
                cat = test_case["category"]
                if cat not in category_scores:
                    category_scores[cat] = {"passed": 0, "total": 0, "scores": []}
                category_scores[cat]["total"] += 1
                category_scores[cat]["scores"].append(grade.get("total_score", 0))
                if grade.get("pass", False):
                    category_scores[cat]["passed"] += 1
                
                # Collect issues
                if grade.get("issues"):
                    all_issues.extend(grade["issues"])
                if grade.get("forbidden_found"):
                    all_issues.append(f"Used forbidden pattern: {grade['forbidden_found']}")
                
            else:
                test_result = {
                    "test_id": test_case["id"],
                    "run": run_num + 1,
                    "error": grade["error"]
                }
                print("ERROR")
            
            results["test_results"].append(test_result)
    
    # Calculate averages
    total = results["summary"]["total_tests"]
    for key in results["summary"]["avg_scores"]:
        results["summary"]["avg_scores"][key] /= max(total, 1)
    
    # Calculate category stats
    for cat, stats in category_scores.items():
        results["summary"]["by_category"][cat] = {
            "pass_rate": stats["passed"] / max(stats["total"], 1) * 100,
            "avg_score": sum(stats["scores"]) / max(len(stats["scores"]), 1),
            "tests": stats["total"]
        }
    
    # Identify common issues
    issue_counts = {}
    for issue in all_issues:
        issue_lower = issue.lower()
        for key in issue_counts:
            if key in issue_lower or issue_lower in key:
                issue_counts[key] += 1
                break
        else:
            issue_counts[issue] = 1
    
    results["summary"]["common_issues"] = sorted(
        issue_counts.items(), key=lambda x: x[1], reverse=True
    )[:10]
    
    # Identify dataset gaps (categories with low scores)
    for cat, stats in results["summary"]["by_category"].items():
        if stats["pass_rate"] < 70 or stats["avg_score"] < 28:
            results["summary"]["dataset_gaps"].append({
                "category": cat,
                "pass_rate": stats["pass_rate"],
                "avg_score": stats["avg_score"],
                "recommendation": f"Add more training examples for {cat} scenarios"
            })
    
    return results


def print_results(results: Dict):
    """Print evaluation results in a readable format."""
    print("\n" + "=" * 60)
    print("L4D2 SOURCEPAWN MODEL EVALUATION RESULTS")
    print("=" * 60)
    
    summary = results["summary"]
    print(f"\nModel: {results['model']}")
    print(f"Date: {results['timestamp']}")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']} ({summary['passed']/summary['total_tests']*100:.1f}%)")
    print(f"Failed: {summary['failed']} ({summary['failed']/summary['total_tests']*100:.1f}%)")
    
    print("\n--- Average Scores (out of 10) ---")
    print(f"Syntax:  {summary['avg_scores']['syntax']:.1f}")
    print(f"API:     {summary['avg_scores']['api']:.1f}")
    print(f"Task:    {summary['avg_scores']['task']:.1f}")
    print(f"Quality: {summary['avg_scores']['quality']:.1f}")
    print(f"Total:   {summary['avg_scores']['total']:.1f}/40")
    
    print("\n--- Results by Category ---")
    for cat, stats in summary["by_category"].items():
        print(f"{cat}: {stats['pass_rate']:.0f}% pass rate, avg score {stats['avg_score']:.1f}")
    
    if summary["common_issues"]:
        print("\n--- Common Issues ---")
        for issue, count in summary["common_issues"][:5]:
            print(f"  [{count}x] {issue[:80]}")
    
    if summary["dataset_gaps"]:
        print("\n--- Dataset Improvement Recommendations ---")
        for gap in summary["dataset_gaps"]:
            print(f"  - {gap['category']}: {gap['recommendation']}")
            print(f"    (pass rate: {gap['pass_rate']:.0f}%, avg score: {gap['avg_score']:.1f})")


def save_results(results: Dict, output_path: Path):
    """Save results to JSON file."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="OpenAI Evals for L4D2 Model")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run evaluation")
    run_parser.add_argument("--runs", type=int, default=1, help="Number of runs per test")
    run_parser.add_argument("--output", default="data/eval_results.json", help="Output file")
    run_parser.add_argument("--model", choices=["v7", "v8", "v9"], default=None,
                           help="Model version to evaluate (v7, v8, or v9, default: latest available)")

    # Quick command
    quick_parser = subparsers.add_parser("quick", help="Quick 3-test evaluation")
    quick_parser.add_argument("--model", choices=["v7", "v8", "v9"], default=None,
                             help="Model version to evaluate (v7, v8, or v9, default: latest available)")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze previous results")
    analyze_parser.add_argument("--file", required=True, help="Results file to analyze")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    client = get_client()
    
    if args.command == "run":
        # Get model ID - default to latest available
        model_key = args.model
        if model_key is None:
            for v in ["v9", "v8", "v7"]:
                if AVAILABLE_MODELS.get(v):
                    model_key = v
                    break
        model_id = AVAILABLE_MODELS.get(model_key)
        if not model_id:
            print(f"ERROR: Model {model_key} not available")
            print(f"Available models: {[k for k, v in AVAILABLE_MODELS.items() if v]}")
            return

        print(f"Running full evaluation on {model_key}...")
        print(f"Model: {model_id}")
        results = run_evaluation(client, num_runs=args.runs, model_id=model_id)
        print_results(results)

        output_path = PROJECT_ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_results(results, output_path)

    elif args.command == "quick":
        # Get model ID - default to latest available
        model_key = args.model
        if model_key is None:
            for v in ["v9", "v8", "v7"]:
                if AVAILABLE_MODELS.get(v):
                    model_key = v
                    break
        model_id = AVAILABLE_MODELS.get(model_key)
        if not model_id:
            print(f"ERROR: Model {model_key} not available")
            print(f"Available models: {[k for k, v in AVAILABLE_MODELS.items() if v]}")
            return

        print(f"Running quick evaluation on {model_key} (3 tests)...")
        print(f"Model: {model_id}")
        quick_tests = L4D2_TEST_CASES[:3]
        results = run_evaluation(client, test_cases=quick_tests, num_runs=1, model_id=model_id)
        print_results(results)
        
    elif args.command == "analyze":
        with open(args.file) as f:
            results = json.load(f)
        print_results(results)


if __name__ == "__main__":
    main()
