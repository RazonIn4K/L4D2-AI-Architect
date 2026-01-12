#!/usr/bin/env python3
"""
Automated Test Suite for L4D2 Code Generation Models

This script:
1. Generates multiple test plugins from prompts
2. Validates each generated plugin
3. Compares results across different models
4. Produces a summary report

Usage:
    python scripts/evaluation/automated_test.py --model local
    python scripts/evaluation/automated_test.py --model openai --prompts 10
    python scripts/evaluation/automated_test.py --compare
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_path, safe_write_json

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Test prompts covering various L4D2 scenarios
TEST_PROMPTS = [
    {
        "id": "speed_boost",
        "prompt": "Write a complete L4D2 SourcePawn plugin that gives survivors a 30% speed boost for 5 seconds when they kill a special infected. Use m_flLaggedMovementValue.",
        "expected_patterns": ["m_flLaggedMovementValue", "player_death", "GetClientUserId"],
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
        "lenient_validation": True,  # Accept valid code structure even if validator has strict issues
    },
    {
        "id": "saferoom_heal",
        "prompt": "Write a complete L4D2 SourcePawn plugin that heals survivors when they enter a saferoom checkpoint.",
        # Accept multiple valid L4D2 events and health-setting approaches
        "expected_patterns_any": [
            ["player_entered_checkpoint", "SetEntityHealth"],   # Option 1
            ["player_entered_checkpoint", "m_iHealth"],         # Option 2: SetEntProp int health
            ["player_entered_checkpoint", "m_flHealth"],        # Option 3: SetEntProp float health
            ["player_entered_safe_area", "SetEntityHealth"],    # Option 4: Alt event
            ["player_entered_safe_area", "m_iHealth"],          # Option 5: Alt event + SetEntProp int
            ["player_entered_safe_area", "m_flHealth"],         # Option 6: Alt event + SetEntProp float
            ["checkpoint_reached", "SetEntityHealth"],          # Option 7: checkpoint_reached event
            ["checkpoint_reached", "m_iHealth"],                # Option 8: checkpoint_reached + int
            ["checkpoint_reached", "m_flHealth"],               # Option 9: checkpoint_reached + float
        ],
        "expected_patterns": [],  # Use expected_patterns_any instead
        "forbidden_patterns": ["m_isInSafeRoom"],
        "lenient_validation": True,  # Accept valid code structure variations
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
        "forbidden_patterns": ["pounce"],  # just "pounce" alone is wrong
        "lenient_validation": True,  # Focus on correct L4D2 API usage
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
        "lenient_validation": True,  # Focus on correct random function usage
    },
]


def run_validation(file_path: Path) -> Dict:
    """Run the validation script on a file and return results."""
    try:
        result = subprocess.run(
            ["python", "scripts/evaluation/validate_generated_code.py", "validate", str(file_path)],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=30
        )

        # Parse output
        output = result.stdout + result.stderr
        passed = "PASS" in output and "FAIL" not in output.split("Overall:")[1] if "Overall:" in output else False

        # Extract score
        score = 0.0
        if "Score:" in output:
            try:
                score_line = [l for l in output.split("\n") if "Score:" in l][0]
                score = float(score_line.split(":")[1].split("/")[0].strip())
            except:
                pass

        return {
            "passed": passed,
            "score": score,
            "output": output[:1000]  # Truncate for storage
        }
    except Exception as e:
        return {
            "passed": False,
            "score": 0.0,
            "output": f"Error: {str(e)}"
        }


# Security anti-patterns to detect in generated code
# RED TEAM: These patterns indicate potential security vulnerabilities
SECURITY_ANTIPATTERNS = [
    # SQL Injection - Format() with SQL and %s without escaping
    (r'Format\s*\([^;]*(?:SELECT|INSERT|UPDATE|DELETE)[^;]*%s',
     "SQL injection risk: use SQL_EscapeString() before interpolating user input"),

    # Command Injection - ServerCommand/ServerExecute with user input
    (r'(?:ServerCommand|ServerExecute)\s*\([^)]*%s',
     "Command injection risk: whitelist allowed commands instead of interpolating"),

    # Entity Exhaustion - CreateEntityByName without edict check
    (r'CreateEntityByName\s*\([^)]+\)(?!.*GetEdictCount)',
     "Entity exhaustion risk: check GetEdictCount() < 1900 before spawning"),

    # Path Traversal - BuildPath with user input without validation
    (r'BuildPath\s*\([^)]*%s(?!.*(?:ReplaceString|StrContains\s*\([^)]*"\.\."))',
     "Path traversal risk: validate paths don't contain .. or absolute paths"),

    # Unbounded String Copy - strcopy without sizeof
    (r'strcopy\s*\([^,]+,\s*\d{3,}',
     "Buffer overflow risk: use sizeof(buffer) instead of hardcoded sizes"),

    # Missing Admin Check - sm_cmd without CheckCommandAccess
    (r'RegAdminCmd\s*\([^)]+\)(?!.*CheckCommandAccess)',
     "Admin bypass risk: verify permissions with CheckCommandAccess()"),

    # Direct Entity Use After Async - Using entity index after timer/callback
    (r'CreateTimer[^}]+\n[^}]*(?<![EntIndexTo|IsValid])Entity\s*\(',
     "Race condition risk: use EntIndexToEntRef() for entity persistence across callbacks"),

    # Verbose Error Logging - Exposing internal paths/details
    (r'(?:LogError|PrintToServer)\s*\([^)]*(?:path|password|key|token|secret)',
     "Information disclosure risk: don't log sensitive data like paths or credentials"),
]


def check_security_patterns(code: str) -> Dict:
    """
    BLUE TEAM: Check generated code for security vulnerabilities.
    Returns dict with security issues found and recommendations.
    """
    import re

    results = {
        "security_issues": [],
        "security_score": 10.0,  # Start at 10, deduct for issues
        "has_critical": False,
    }

    for pattern, description in SECURITY_ANTIPATTERNS:
        matches = re.findall(pattern, code, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if matches:
            severity = "CRITICAL" if any(x in description.lower() for x in ["injection", "rce", "overflow"]) else "WARNING"
            results["security_issues"].append({
                "pattern": pattern[:50] + "...",
                "description": description,
                "severity": severity,
                "matches": len(matches)
            })

            # Deduct points based on severity
            if severity == "CRITICAL":
                results["security_score"] -= 3.0
                results["has_critical"] = True
            else:
                results["security_score"] -= 1.0

    # Clamp score to 0
    results["security_score"] = max(0.0, results["security_score"])

    return results


def check_patterns(code: str, expected: List[str], forbidden: List[str], expected_any: List[List[str]] = None) -> Dict:
    """Check if code contains expected patterns and doesn't contain forbidden ones.

    Args:
        code: The generated code to check
        expected: List of patterns that must ALL be present
        forbidden: List of patterns that must NOT be present
        expected_any: List of pattern groups - at least ONE group must be fully satisfied
    """
    import re

    results = {
        "expected_found": [],
        "expected_missing": [],
        "forbidden_found": [],
        "alternative_match": None,  # Which alternative pattern group matched (if any)
    }

    # Handle expected_patterns_any - find best matching alternative
    if expected_any:
        best_match = None
        best_match_count = -1

        for i, alt_patterns in enumerate(expected_any):
            found_in_alt = []
            for pattern in alt_patterns:
                if pattern.lower() in code.lower():
                    found_in_alt.append(pattern)

            # Track best matching alternative
            if len(found_in_alt) > best_match_count:
                best_match_count = len(found_in_alt)
                best_match = {
                    "index": i,
                    "patterns": alt_patterns,
                    "found": found_in_alt,
                    "missing": [p for p in alt_patterns if p not in found_in_alt],
                    "complete": len(found_in_alt) == len(alt_patterns)
                }

        if best_match:
            results["alternative_match"] = best_match
            if best_match["complete"]:
                # Full match - all patterns in this alternative found
                results["expected_found"] = best_match["found"]
                results["expected_missing"] = []
            else:
                # Partial match - report what's missing from best alternative
                results["expected_found"] = best_match["found"]
                results["expected_missing"] = best_match["missing"]

    # Standard expected patterns (all must be present)
    for pattern in expected:
        if pattern.lower() in code.lower():
            results["expected_found"].append(pattern)
        else:
            results["expected_missing"].append(pattern)

    for pattern in forbidden:
        # Use smart pattern matching to avoid false positives
        if pattern == "pounce":
            # Only match HookEvent("pounce" not lunge_pounce
            if re.search(r'HookEvent\s*\(\s*["\']pounce["\']', code, re.IGNORECASE):
                results["forbidden_found"].append(pattern)
        elif pattern == "RandomFloat":
            # Only match RandomFloat( not GetRandomFloat(
            if re.search(r'(?<!Get)RandomFloat\s*\(', code):
                results["forbidden_found"].append(pattern)
        elif pattern == "RandomInt":
            # Only match RandomInt( not GetRandomInt(
            if re.search(r'(?<!Get)RandomInt\s*\(', code):
                results["forbidden_found"].append(pattern)
        elif pattern in ["smoker_tongue_grab", "boomer_vomit", "player_biled", "charger_grab", "charger_impact", "jockey_grab"]:
            # Only match if actually used in HookEvent, not in comments
            if re.search(rf'HookEvent\s*\(\s*["\']{ re.escape(pattern) }["\']', code, re.IGNORECASE):
                results["forbidden_found"].append(pattern)
        else:
            # Standard check for other patterns
            if pattern in code:
                results["forbidden_found"].append(pattern)

    return results


def generate_with_openai(prompt: str, output_path: Path) -> bool:
    """Generate code using the fine-tuned OpenAI model."""
    try:
        # Get API key from Doppler
        api_key_result = subprocess.run(
            ["doppler", "secrets", "get", "OPENAI_API_KEY",
             "--project", "local-mac-work", "--config", "dev_personal", "--plain"],
            capture_output=True,
            text=True,
            timeout=10
        )
        api_key = api_key_result.stdout.strip()

        if not api_key:
            print("  Warning: Could not get OpenAI API key from Doppler")
            return False

        # Run generation
        env = os.environ.copy()
        env["OPENAI_API_KEY"] = api_key

        result = subprocess.run(
            ["python", "scripts/inference/l4d2_codegen.py", "generate", prompt,
             "--output", str(output_path)],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env=env,
            timeout=120
        )

        return output_path.exists() and output_path.stat().st_size > 100

    except Exception as e:
        print(f"  Error generating with OpenAI: {e}")
        return False


def generate_with_local(prompt: str, output_path: Path) -> bool:
    """Generate code using the local LoRA model."""
    try:
        # Use the generate_test_plugins infrastructure
        result = subprocess.run(
            ["python", "-c", f"""
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "mistralai/Mistral-7B-Instruct-v0.3"
adapter_path = "model_adapters/l4d2-mistral-v10plus-lora/final"

model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)
if torch.backends.mps.is_available():
    model = model.to("mps")
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

messages = [
    {{"role": "system", "content": "You are an expert SourcePawn developer for Left 4 Dead 2. Generate complete, working plugin code."}},
    {{"role": "user", "content": '''{prompt}'''}}
]

chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(chat_text, return_tensors="pt")
if torch.backends.mps.is_available():
    inputs = {{k: v.to("mps") for k, v in inputs.items()}}

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=800, temperature=0.5, do_sample=True, pad_token_id=tokenizer.eos_token_id)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
if "<|assistant|>" in response:
    response = response.split("<|assistant|>")[-1].strip()

Path('{output_path}').write_text(response)
"""],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=180
        )

        return output_path.exists() and output_path.stat().st_size > 50

    except Exception as e:
        print(f"  Error generating with local model: {e}")
        return False


def run_test_suite(model_type: str = "openai", num_prompts: int = 10) -> Dict:
    """Run the full test suite."""
    results = {
        "model_type": model_type,
        "timestamp": datetime.now().isoformat(),
        "prompts_tested": 0,
        "passed": 0,
        "failed": 0,
        "total_score": 0.0,
        "pattern_accuracy": 0.0,
        "test_results": []
    }

    output_dir = PROJECT_ROOT / f"data/automated_test_{model_type}"
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts_to_test = TEST_PROMPTS[:num_prompts]

    print(f"\n{'='*60}")
    print(f"L4D2 Automated Test Suite - {model_type.upper()}")
    print(f"{'='*60}")
    print(f"Testing {len(prompts_to_test)} prompts\n")

    expected_total = 0
    expected_found = 0
    forbidden_total = 0
    forbidden_avoided = 0
    total_security_score = 0.0
    total_security_issues = 0
    critical_security_issues = 0

    for i, test in enumerate(prompts_to_test, 1):
        print(f"[{i}/{len(prompts_to_test)}] {test['id']}...")

        output_path = output_dir / f"{test['id']}.sp"

        # Generate code
        if model_type == "openai":
            success = generate_with_openai(test["prompt"], output_path)
        else:
            success = generate_with_local(test["prompt"], output_path)

        if not success:
            print(f"  SKIP - Generation failed")
            results["test_results"].append({
                "id": test["id"],
                "generated": False,
                "passed": False,
                "score": 0.0,
                "pattern_check": None
            })
            continue

        # Read generated code
        code = output_path.read_text()

        # Validate
        validation = run_validation(output_path)

        # Check patterns - support alternative valid patterns
        expected_any = test.get("expected_patterns_any", None)
        pattern_check = check_patterns(code, test["expected_patterns"], test["forbidden_patterns"], expected_any)

        # BLUE TEAM: Security vulnerability check
        security_check = check_security_patterns(code)

        # Update counters
        expected_total += len(test["expected_patterns"])
        expected_found += len(pattern_check["expected_found"])
        forbidden_total += len(test["forbidden_patterns"])
        forbidden_avoided += len(test["forbidden_patterns"]) - len(pattern_check["forbidden_found"])

        # Security counters
        total_security_score += security_check["security_score"]
        total_security_issues += len(security_check["security_issues"])
        critical_security_issues += sum(1 for i in security_check["security_issues"] if i["severity"] == "CRITICAL")

        results["prompts_tested"] += 1
        results["total_score"] += validation["score"]

        # CRITICAL FIX: Pass requires BOTH validator pass AND no forbidden patterns
        # This ensures L4D2-specific correctness (e.g., using lunge_pounce not pounce)
        has_forbidden = len(pattern_check["forbidden_found"]) > 0
        has_missing = len(pattern_check["expected_missing"]) > 0

        # Lenient validation: pass if patterns are correct even if validator is strict
        lenient_mode = test.get("lenient_validation", False)
        if lenient_mode:
            # For lenient tests: pass if expected patterns found, no forbidden, score >= 5
            test_passed = not has_missing and not has_forbidden and validation["score"] >= 5.0
        else:
            # Standard: require both validator pass and pattern correctness
            test_passed = validation["passed"] and not has_forbidden

        if test_passed:
            results["passed"] += 1
            status = "PASS"
        else:
            results["failed"] += 1
            if has_forbidden and validation["passed"]:
                status = "FAIL (wrong L4D2 APIs)"
            else:
                status = "FAIL"

        # Pattern status
        pattern_status = ""
        if pattern_check["expected_missing"]:
            pattern_status += f" [Missing: {', '.join(pattern_check['expected_missing'])}]"
        if pattern_check["forbidden_found"]:
            pattern_status += f" [WRONG: {', '.join(pattern_check['forbidden_found'])}]"

        # Security status
        security_status = ""
        if security_check["security_issues"]:
            critical_count = sum(1 for i in security_check["security_issues"] if i["severity"] == "CRITICAL")
            warning_count = len(security_check["security_issues"]) - critical_count
            if critical_count > 0:
                security_status += f" [SECURITY: {critical_count} critical"
            if warning_count > 0:
                security_status += f", {warning_count} warnings" if critical_count > 0 else f" [SECURITY: {warning_count} warnings"
            security_status += "]"

        print(f"  {status} (score: {validation['score']}/10){pattern_status}{security_status}")

        results["test_results"].append({
            "id": test["id"],
            "generated": True,
            "passed": test_passed,  # Use corrected pass status (includes pattern check)
            "validator_passed": validation["passed"],  # Raw validator result
            "score": validation["score"],
            "pattern_check": pattern_check,
            "security_check": security_check,  # BLUE TEAM security analysis
        })

    # Calculate final metrics
    if results["prompts_tested"] > 0:
        results["pass_rate"] = results["passed"] / results["prompts_tested"] * 100
        results["average_score"] = results["total_score"] / results["prompts_tested"]
        results["average_security_score"] = total_security_score / results["prompts_tested"]
    else:
        results["pass_rate"] = 0.0
        results["average_score"] = 0.0
        results["average_security_score"] = 10.0

    if expected_total > 0:
        results["expected_pattern_accuracy"] = expected_found / expected_total * 100
    else:
        results["expected_pattern_accuracy"] = 100.0

    if forbidden_total > 0:
        results["forbidden_pattern_avoidance"] = forbidden_avoided / forbidden_total * 100
    else:
        results["forbidden_pattern_avoidance"] = 100.0

    # Security metrics
    results["total_security_issues"] = total_security_issues
    results["critical_security_issues"] = critical_security_issues

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Pass Rate:      {results['pass_rate']:.1f}% ({results['passed']}/{results['prompts_tested']})")
    print(f"Average Score:  {results['average_score']:.2f}/10")
    print(f"Expected APIs:  {results['expected_pattern_accuracy']:.1f}% found")
    print(f"Wrong APIs:     {results['forbidden_pattern_avoidance']:.1f}% avoided")
    print(f"\n{'='*60}")
    print("SECURITY ANALYSIS (BLUE TEAM)")
    print(f"{'='*60}")
    print(f"Security Score: {results['average_security_score']:.1f}/10")
    print(f"Total Issues:   {total_security_issues}")
    print(f"Critical:       {critical_security_issues}")

    # Save results
    results_path = PROJECT_ROOT / f"data/test_results_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    safe_write_json(results_path, results, PROJECT_ROOT)
    print(f"\nResults saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Automated L4D2 Model Test Suite")
    parser.add_argument("--model", choices=["openai", "local", "both"], default="openai",
                       help="Model to test (default: openai)")
    parser.add_argument("--prompts", type=int, default=10,
                       help="Number of prompts to test (default: 10)")
    parser.add_argument("--compare", action="store_true",
                       help="Run comparison of both models")
    args = parser.parse_args()

    if args.compare or args.model == "both":
        print("Running comparison test...")
        openai_results = run_test_suite("openai", args.prompts)
        local_results = run_test_suite("local", args.prompts)

        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"{'Metric':<25} {'OpenAI':<15} {'Local':<15}")
        print("-" * 55)
        print(f"{'Pass Rate':<25} {openai_results['pass_rate']:.1f}%{'':<10} {local_results['pass_rate']:.1f}%")
        print(f"{'Average Score':<25} {openai_results['average_score']:.2f}/10{'':<8} {local_results['average_score']:.2f}/10")
        print(f"{'Expected APIs':<25} {openai_results['expected_pattern_accuracy']:.1f}%{'':<10} {local_results['expected_pattern_accuracy']:.1f}%")
    else:
        run_test_suite(args.model, args.prompts)


if __name__ == "__main__":
    main()
