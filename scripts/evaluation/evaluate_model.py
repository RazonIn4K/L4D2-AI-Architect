#!/usr/bin/env python3
"""
L4D2-AI-Architect: Model Evaluation Suite

Comprehensive evaluation of trained models on L4D2-specific tasks.

Usage:
    python evaluate_model.py --model model_adapters/l4d2-mistral-v15-lora/final
    python evaluate_model.py --model exports/l4d2-mistral-v15/gguf --backend ollama
    python evaluate_model.py --compare model1 model2 model3  # Compare multiple models
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_write_json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Evaluation test cases covering key L4D2 modding scenarios
EVAL_CASES = [
    # SourcePawn Basics
    {
        "id": "sp_basic_function",
        "category": "SourcePawn Basics",
        "prompt": "Write a SourcePawn function that heals all survivors to full health.",
        "expected_patterns": ["public", "void", "GetClientHealth", "SetEntityHealth", "for", "MaxClients"],
        "language": "sourcepawn",
    },
    {
        "id": "sp_event_hook",
        "category": "SourcePawn Events",
        "prompt": "Write a SourcePawn event hook for when a player takes damage from an infected.",
        "expected_patterns": ["HookEvent", "player_hurt", "GetEventInt", "attacker", "userid"],
        "language": "sourcepawn",
    },
    {
        "id": "sp_command",
        "category": "SourcePawn Commands",
        "prompt": "Create a SourcePawn admin command that teleports all survivors to the command user's position.",
        "expected_patterns": ["RegAdminCmd", "Command_", "GetClientAbsOrigin", "TeleportEntity", "ADMFLAG"],
        "language": "sourcepawn",
    },
    {
        "id": "sp_timer",
        "category": "SourcePawn Timers",
        "prompt": "Write a SourcePawn timer that spawns a tank every 5 minutes during gameplay.",
        "expected_patterns": ["CreateTimer", "Timer_", "CheatCommand", "z_spawn", "tank"],
        "language": "sourcepawn",
    },
    {
        "id": "sp_convars",
        "category": "SourcePawn ConVars",
        "prompt": "Create a SourcePawn plugin with ConVars for configuring zombie spawn rate and health multiplier.",
        "expected_patterns": ["CreateConVar", "ConVar", "GetFloat", "GetInt", "HookConVarChange"],
        "language": "sourcepawn",
    },
    # VScript Basics
    {
        "id": "vs_basic_function",
        "category": "VScript Basics",
        "prompt": "Write a VScript function that gives all survivors adrenaline.",
        "expected_patterns": ["function", "foreach", "survivor", "GiveItem", "adrenaline"],
        "language": "vscript",
    },
    {
        "id": "vs_director",
        "category": "VScript Director",
        "prompt": "Write a VScript director options table that increases zombie spawns and enables tank spawns.",
        "expected_patterns": ["DirectorOptions", "CommonLimit", "MobSpawnMinTime", "TankLimit", "PreferredMobDirection"],
        "language": "vscript",
    },
    {
        "id": "vs_mutation",
        "category": "VScript Mutations",
        "prompt": "Create a VScript mutation that makes all common infected run faster and deal more damage.",
        "expected_patterns": ["MutationOptions", "function", "ZombieDamage", "WanderSpeed", "RunSpeed"],
        "language": "vscript",
    },
    # Advanced Patterns
    {
        "id": "sp_native",
        "category": "SourcePawn Advanced",
        "prompt": "Write a SourcePawn native declaration for a function that gets a survivor's incap count.",
        "expected_patterns": ["native", "int", "GetSurvivorIncapCount", "client"],
        "language": "sourcepawn",
    },
    {
        "id": "sp_methodmap",
        "category": "SourcePawn Advanced",
        "prompt": "Create a SourcePawn methodmap for managing infected players with properties and methods.",
        "expected_patterns": ["methodmap", "property", "public", "get", "this"],
        "language": "sourcepawn",
    },
]


@dataclass
class EvalResult:
    """Result of a single evaluation test."""
    test_id: str
    category: str
    prompt: str
    response: str
    patterns_found: List[str]
    patterns_missing: List[str]
    score: float  # 0.0 to 1.0
    response_time_ms: float
    error: Optional[str] = None


@dataclass
class ModelEvaluation:
    """Complete evaluation results for a model."""
    model_path: str
    model_name: str
    backend: str
    timestamp: str
    total_tests: int
    passed_tests: int
    average_score: float
    average_response_time_ms: float
    category_scores: Dict[str, float]
    results: List[Dict[str, Any]]


def load_unsloth_model(model_path: str) -> Tuple[Any, Any]:
    """Load model using Unsloth for evaluation."""
    from unsloth import FastLanguageModel

    logger.info(f"Loading model from {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_unsloth(model, tokenizer, prompt: str, language: str = "sourcepawn") -> Tuple[str, float]:
    """Generate response using Unsloth model."""
    import torch

    # Format with system prompt
    if language == "vscript":
        system = "You are an expert VScript developer for Left 4 Dead 2. Generate clean, working Squirrel code."
    else:
        system = "You are an expert SourcePawn developer for Left 4 Dead 2. Generate clean, working SourceMod plugin code."

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    # Apply chat template
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    elapsed_ms = (time.time() - start_time) * 1000

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip(), elapsed_ms


def generate_ollama(model_name: str, prompt: str, language: str = "sourcepawn") -> Tuple[str, float]:
    """Generate response using Ollama."""
    import subprocess
    import json

    if language == "vscript":
        system = "You are an expert VScript developer for Left 4 Dead 2. Generate clean, working Squirrel code."
    else:
        system = "You are an expert SourcePawn developer for Left 4 Dead 2. Generate clean, working SourceMod plugin code."

    full_prompt = f"{system}\n\nUser: {prompt}\n\nAssistant:"

    start_time = time.time()

    try:
        result = subprocess.run(
            ["ollama", "run", model_name, full_prompt],
            capture_output=True,
            text=True,
            timeout=60,
        )
        elapsed_ms = (time.time() - start_time) * 1000
        return result.stdout.strip(), elapsed_ms
    except subprocess.TimeoutExpired:
        return "", 60000.0
    except Exception as e:
        return f"Error: {e}", 0.0


def evaluate_response(response: str, expected_patterns: List[str]) -> Tuple[List[str], List[str], float]:
    """Evaluate response against expected patterns."""
    found = []
    missing = []

    response_lower = response.lower()

    for pattern in expected_patterns:
        if pattern.lower() in response_lower:
            found.append(pattern)
        else:
            missing.append(pattern)

    score = len(found) / len(expected_patterns) if expected_patterns else 0.0
    return found, missing, score


def run_evaluation(
    model_path: str,
    backend: str = "unsloth",
    test_cases: Optional[List[Dict]] = None,
) -> ModelEvaluation:
    """Run full evaluation on a model."""

    test_cases = test_cases or EVAL_CASES
    results: List[EvalResult] = []

    # Load model based on backend
    model, tokenizer = None, None
    model_name = Path(model_path).name

    if backend == "unsloth":
        model, tokenizer = load_unsloth_model(model_path)
        generate_fn = lambda p, lang: generate_unsloth(model, tokenizer, p, lang)
    elif backend == "ollama":
        model_name = model_path  # For ollama, model_path is the model name
        generate_fn = lambda p, lang: generate_ollama(model_path, p, lang)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    logger.info(f"Running {len(test_cases)} evaluation tests...")

    for i, test in enumerate(test_cases):
        logger.info(f"[{i+1}/{len(test_cases)}] {test['id']}: {test['category']}")

        try:
            response, elapsed_ms = generate_fn(test["prompt"], test.get("language", "sourcepawn"))
            found, missing, score = evaluate_response(response, test["expected_patterns"])

            result = EvalResult(
                test_id=test["id"],
                category=test["category"],
                prompt=test["prompt"],
                response=response,
                patterns_found=found,
                patterns_missing=missing,
                score=score,
                response_time_ms=elapsed_ms,
            )
        except Exception as e:
            logger.error(f"Error on test {test['id']}: {e}")
            result = EvalResult(
                test_id=test["id"],
                category=test["category"],
                prompt=test["prompt"],
                response="",
                patterns_found=[],
                patterns_missing=test["expected_patterns"],
                score=0.0,
                response_time_ms=0.0,
                error=str(e),
            )

        results.append(result)

        # Log result
        status = "PASS" if score >= 0.5 else "FAIL"
        logger.info(f"  {status}: {score:.1%} ({len(found)}/{len(test['expected_patterns'])} patterns)")

    # Calculate aggregate metrics
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.score >= 0.5)
    avg_score = sum(r.score for r in results) / total_tests if total_tests else 0
    avg_time = sum(r.response_time_ms for r in results) / total_tests if total_tests else 0

    # Category breakdown
    category_scores: Dict[str, List[float]] = {}
    for r in results:
        if r.category not in category_scores:
            category_scores[r.category] = []
        category_scores[r.category].append(r.score)

    category_avgs = {cat: sum(scores)/len(scores) for cat, scores in category_scores.items()}

    return ModelEvaluation(
        model_path=model_path,
        model_name=model_name,
        backend=backend,
        timestamp=datetime.now().isoformat(),
        total_tests=total_tests,
        passed_tests=passed_tests,
        average_score=avg_score,
        average_response_time_ms=avg_time,
        category_scores=category_avgs,
        results=[asdict(r) for r in results],
    )


def print_evaluation_summary(eval_result: ModelEvaluation):
    """Print formatted evaluation summary."""
    print()
    print("=" * 70)
    print(f"MODEL EVALUATION: {eval_result.model_name}")
    print("=" * 70)
    print()
    print(f"Backend: {eval_result.backend}")
    print(f"Timestamp: {eval_result.timestamp}")
    print()
    print(f"Overall Score: {eval_result.average_score:.1%}")
    print(f"Tests Passed: {eval_result.passed_tests}/{eval_result.total_tests}")
    print(f"Avg Response Time: {eval_result.average_response_time_ms:.0f}ms")
    print()
    print("Category Breakdown:")
    print("-" * 40)
    for cat, score in sorted(eval_result.category_scores.items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {cat:<25} {bar} {score:.1%}")
    print()

    # Show failures
    failures = [r for r in eval_result.results if r["score"] < 0.5]
    if failures:
        print("Failed Tests:")
        print("-" * 40)
        for r in failures:
            print(f"  {r['test_id']}: {r['score']:.1%} (missing: {', '.join(r['patterns_missing'][:3])})")
    print()
    print("=" * 70)


def compare_models(eval_results: List[ModelEvaluation]):
    """Compare multiple model evaluations."""
    print()
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print()

    # Header
    models = [e.model_name[:20] for e in eval_results]
    print(f"{'Metric':<30} " + " ".join(f"{m:>15}" for m in models))
    print("-" * (30 + 16 * len(models)))

    # Overall score
    print(f"{'Overall Score':<30} " + " ".join(f"{e.average_score:>14.1%}" for e in eval_results))

    # Tests passed
    print(f"{'Tests Passed':<30} " + " ".join(f"{e.passed_tests:>11}/{e.total_tests}" for e in eval_results))

    # Response time
    print(f"{'Avg Response Time':<30} " + " ".join(f"{e.average_response_time_ms:>12.0f}ms" for e in eval_results))

    print()
    print("Category Scores:")
    print("-" * (30 + 16 * len(models)))

    # Get all categories
    all_cats = set()
    for e in eval_results:
        all_cats.update(e.category_scores.keys())

    for cat in sorted(all_cats):
        scores = []
        for e in eval_results:
            scores.append(e.category_scores.get(cat, 0.0))
        print(f"  {cat:<28} " + " ".join(f"{s:>14.1%}" for s in scores))

    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate L4D2 code models")
    parser.add_argument("--model", type=str, help="Path to model adapter or Ollama model name")
    parser.add_argument("--backend", type=str, default="unsloth", choices=["unsloth", "ollama"])
    parser.add_argument("--compare", nargs="+", help="Compare multiple models")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--quick", action="store_true", help="Run quick evaluation (first 5 tests)")

    args = parser.parse_args()

    if args.compare:
        # Compare multiple models
        eval_results = []
        for model_path in args.compare:
            backend = "ollama" if "/" not in model_path else "unsloth"
            result = run_evaluation(model_path, backend=backend)
            eval_results.append(result)
            print_evaluation_summary(result)

        compare_models(eval_results)
    elif args.model:
        # Evaluate single model
        test_cases = EVAL_CASES[:5] if args.quick else EVAL_CASES
        result = run_evaluation(args.model, backend=args.backend, test_cases=test_cases)
        print_evaluation_summary(result)

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(asdict(result), f, indent=2)
            logger.info(f"Results saved to {output_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
