#!/usr/bin/env python3
"""
Test the Qwen2.5-Coder-7B LoRA model against L4D2 test prompts.

This script:
1. Loads the Qwen base model + LoRA adapter
2. Generates code for each test prompt
3. Validates and scores the output
4. Reports pattern accuracy (correct vs forbidden APIs)
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Test prompts from automated_test.py
TEST_PROMPTS = [
    {
        "id": "speed_boost",
        "prompt": "Write a complete L4D2 SourcePawn plugin that gives survivors a 30% speed boost for 5 seconds when they kill a special infected. Use m_flLaggedMovementValue.",
        "expected_patterns": ["m_flLaggedMovementValue", "player_death", "GetClientOfUserId"],
        "forbidden_patterns": ["m_flSpeed", "m_flMaxSpeed"],
    },
    {
        "id": "no_ff_panic",
        "prompt": "Write a complete L4D2 SourcePawn plugin that disables friendly fire during panic events using SDKHook_OnTakeDamage.",
        "expected_patterns": ["SDKHook", "create_panic_event", "OnTakeDamage"],
        "forbidden_patterns": ["panic_start", "panic_end", "event.SetInt"],
    },
    {
        "id": "kill_tracker",
        "prompt": "Write a complete L4D2 SourcePawn plugin that tracks zombie kills per player using the infected_death event.",
        "expected_patterns": ["infected_death", "HookEvent"],
        "forbidden_patterns": [],
    },
    {
        "id": "saferoom_heal",
        "prompt": "Write a complete L4D2 SourcePawn plugin that heals survivors when they enter a saferoom checkpoint.",
        "expected_patterns": ["player_entered_checkpoint", "SetEntityHealth"],
        "forbidden_patterns": ["m_isInSafeRoom"],
    },
    {
        "id": "tank_announce",
        "prompt": "Write a complete L4D2 SourcePawn plugin that announces to all players when a Tank spawns with tank_spawn event.",
        "expected_patterns": ["tank_spawn", "PrintToChatAll"],
        "forbidden_patterns": ["tank_health_changed"],
    },
    {
        "id": "hunter_pounce",
        "prompt": "Write a complete L4D2 SourcePawn plugin that tracks Hunter pounce damage using the correct L4D2 event.",
        "expected_patterns": ["lunge_pounce"],
        "forbidden_patterns": ["pounce"],
    },
    {
        "id": "bile_tracker",
        "prompt": "Write a complete L4D2 SourcePawn plugin that tracks when players get covered in Boomer bile.",
        "expected_patterns": ["player_now_it"],
        "forbidden_patterns": ["boomer_vomit", "player_biled"],
    },
    {
        "id": "smoker_grab",
        "prompt": "Write a complete L4D2 SourcePawn plugin that announces when a Smoker grabs a survivor.",
        "expected_patterns": ["tongue_grab"],
        "forbidden_patterns": ["smoker_tongue_grab"],
    },
    {
        "id": "charger_carry",
        "prompt": "Write a complete L4D2 SourcePawn plugin that tracks Charger carries.",
        "expected_patterns": ["charger_carry_start"],
        "forbidden_patterns": ["charger_grab", "charger_impact"],
    },
    {
        "id": "random_timer",
        "prompt": "Write a complete L4D2 SourcePawn plugin that spawns a witch at a random time between 30 and 60 seconds using proper random functions.",
        "expected_patterns": ["GetRandomFloat", "CreateTimer"],
        "forbidden_patterns": ["RandomFloat", "RandomInt"],
    },
]


def check_patterns(code: str, expected: list, forbidden: list) -> dict:
    """Check if code contains expected patterns and avoids forbidden ones."""
    results = {
        "expected_found": [],
        "expected_missing": [],
        "forbidden_found": [],
    }

    code_lower = code.lower()

    for pattern in expected:
        if pattern.lower() in code_lower:
            results["expected_found"].append(pattern)
        else:
            results["expected_missing"].append(pattern)

    for pattern in forbidden:
        if pattern in code:  # Case-sensitive for forbidden
            results["forbidden_found"].append(pattern)

    return results


def calculate_score(code: str, pattern_check: dict) -> float:
    """Calculate a score from 0-10 based on code quality and patterns."""
    score = 0.0

    # Basic structure checks (4 points)
    if "#pragma semicolon" in code:
        score += 0.5
    if "#pragma newdecls required" in code:
        score += 0.5
    if "#include <sourcemod>" in code:
        score += 0.5
    if "public Plugin myinfo" in code:
        score += 0.5
    if "public void OnPluginStart" in code:
        score += 1.0
    if "HookEvent" in code:
        score += 1.0

    # Pattern accuracy (4 points)
    total_expected = len(pattern_check["expected_found"]) + len(pattern_check["expected_missing"])
    if total_expected > 0:
        expected_ratio = len(pattern_check["expected_found"]) / total_expected
        score += expected_ratio * 2.0

    # Penalty for forbidden patterns (up to -2 points)
    forbidden_penalty = len(pattern_check["forbidden_found"]) * 1.0
    score -= min(forbidden_penalty, 2.0)

    # Code completeness (2 points)
    if "return Plugin_" in code:
        score += 0.5
    if "CreateTimer" in code or "SetEntProp" in code:
        score += 0.5
    if len(code) > 500:  # Reasonable length
        score += 1.0

    return max(0.0, min(10.0, score))


def generate_with_qwen(prompt: str, model, tokenizer) -> str:
    """Generate code using the Qwen model."""
    import torch

    messages = [
        {"role": "system", "content": "You are an expert SourcePawn developer for Left 4 Dead 2. Generate complete, working plugin code."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=800,
            temperature=0.5,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract assistant response
    if "assistant\n" in response:
        response = response.split("assistant\n")[-1].strip()
    elif "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1].strip()

    return response


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    output_dir = PROJECT_ROOT / "data/automated_test_qwen"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("L4D2 Qwen Model Test Suite")
    print("=" * 60)

    # Load model
    print("\nLoading Qwen2.5-Coder-7B base model...")
    base_model = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model)

        print("Loading LoRA adapter...")
        adapter_path = PROJECT_ROOT / "model_adapters/l4d2-qwen-lora/final"
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model.eval()
        print("Model loaded successfully!\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nFalling back to inference via API or skipping...")
        return

    # Run tests
    results = {
        "model": "Qwen2.5-Coder-7B + L4D2 LoRA",
        "timestamp": datetime.now().isoformat(),
        "tests": [],
        "summary": {}
    }

    passed = 0
    total_score = 0.0
    expected_found_total = 0
    expected_total = 0
    forbidden_avoided_total = 0
    forbidden_total = 0

    for i, test in enumerate(TEST_PROMPTS, 1):
        print(f"[{i}/{len(TEST_PROMPTS)}] {test['id']}...")

        # Generate code
        code = generate_with_qwen(test["prompt"], model, tokenizer)

        # Save generated code
        output_path = output_dir / f"{test['id']}.sp"
        output_path.write_text(code)

        # Check patterns
        pattern_check = check_patterns(code, test["expected_patterns"], test["forbidden_patterns"])

        # Calculate score
        score = calculate_score(code, pattern_check)
        total_score += score

        # Track pattern accuracy
        expected_total += len(test["expected_patterns"])
        expected_found_total += len(pattern_check["expected_found"])
        forbidden_total += len(test["forbidden_patterns"])
        forbidden_avoided_total += len(test["forbidden_patterns"]) - len(pattern_check["forbidden_found"])

        # Determine pass/fail
        test_passed = score >= 6.0 and len(pattern_check["forbidden_found"]) == 0
        if test_passed:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"

        # Report
        pattern_info = ""
        if pattern_check["expected_missing"]:
            pattern_info += f" [Missing: {', '.join(pattern_check['expected_missing'])}]"
        if pattern_check["forbidden_found"]:
            pattern_info += f" [WRONG: {', '.join(pattern_check['forbidden_found'])}]"

        print(f"  {status} (score: {score:.1f}/10){pattern_info}")

        results["tests"].append({
            "id": test["id"],
            "passed": test_passed,
            "score": score,
            "pattern_check": pattern_check
        })

    # Summary
    pass_rate = (passed / len(TEST_PROMPTS)) * 100
    avg_score = total_score / len(TEST_PROMPTS)
    expected_accuracy = (expected_found_total / expected_total * 100) if expected_total > 0 else 100
    forbidden_avoidance = (forbidden_avoided_total / forbidden_total * 100) if forbidden_total > 0 else 100

    results["summary"] = {
        "pass_rate": pass_rate,
        "passed": passed,
        "total": len(TEST_PROMPTS),
        "average_score": avg_score,
        "expected_pattern_accuracy": expected_accuracy,
        "forbidden_pattern_avoidance": forbidden_avoidance
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Pass Rate:      {pass_rate:.1f}% ({passed}/{len(TEST_PROMPTS)})")
    print(f"Average Score:  {avg_score:.2f}/10")
    print(f"Expected APIs:  {expected_accuracy:.1f}% found")
    print(f"Wrong APIs:     {forbidden_avoidance:.1f}% avoided")

    # Save results
    results_path = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == "__main__":
    main()
