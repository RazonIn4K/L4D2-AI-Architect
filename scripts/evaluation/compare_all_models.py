#!/usr/bin/env python3
"""
Model Comparison and Evaluation Script

Compares all available L4D2 code generation models by:
1. Loading model adapters from model_adapters/
2. Generating code from test prompts
3. Scoring outputs on syntax, API usage, and completeness
4. Creating a detailed markdown report

Supports Ollama models for local inference.

Usage:
    # Compare all available Ollama models
    python compare_all_models.py

    # Compare specific models
    python compare_all_models.py --models l4d2-code-v10plus,l4d2-code-v11

    # Use custom prompts
    python compare_all_models.py --prompts "Write a heal function" "Create a tank spawn timer"

    # Custom output path
    python compare_all_models.py --output results/comparison.md
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_write_text, safe_path

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_ADAPTERS_DIR = PROJECT_ROOT / "model_adapters"
RESULTS_DIR = PROJECT_ROOT / "results"


# =============================================================================
# TEST PROMPTS
# =============================================================================

DEFAULT_TEST_PROMPTS = [
    {
        "id": "heal_survivors",
        "prompt": "Write a SourcePawn function that heals all survivors to full health.",
        "language": "sourcepawn",
        "expected_apis": ["GetClientHealth", "SetEntityHealth", "GetClientTeam", "MaxClients", "IsClientInGame"],
        "expected_syntax": ["public", "void", "for", "if"],
    },
    {
        "id": "spawn_tank_timer",
        "prompt": "Write a SourcePawn plugin that spawns a tank every 5 minutes using a timer.",
        "language": "sourcepawn",
        "expected_apis": ["CreateTimer", "CheatCommand", "z_spawn"],
        "expected_syntax": ["#include", "public", "Timer_", "Action"],
    },
    {
        "id": "damage_event",
        "prompt": "Write a SourcePawn event hook for player_hurt that shows damage dealt.",
        "language": "sourcepawn",
        "expected_apis": ["HookEvent", "GetEventInt", "PrintToChat"],
        "expected_syntax": ["Event", "player_hurt", "userid", "attacker"],
    },
    {
        "id": "admin_teleport",
        "prompt": "Create a SourcePawn admin command that teleports a player to the command user.",
        "language": "sourcepawn",
        "expected_apis": ["RegAdminCmd", "GetCmdArg", "GetClientAbsOrigin", "TeleportEntity"],
        "expected_syntax": ["ADMFLAG", "Command_", "target", "float"],
    },
    {
        "id": "vscript_director",
        "prompt": "Write a VScript DirectorOptions table that increases common infected spawns.",
        "language": "vscript",
        "expected_apis": ["DirectorOptions", "CommonLimit", "MobSpawnMinTime"],
        "expected_syntax": ["<-", "function", "printl"],
    },
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SyntaxScore:
    """Syntax correctness scoring."""
    has_includes: bool = False
    has_proper_structure: bool = False
    has_semicolons: bool = False
    has_braces_balanced: bool = False
    is_valid_code: bool = False
    score: float = 0.0

    def calculate(self):
        """Calculate syntax score from components."""
        points = 0
        if self.has_includes:
            points += 2
        if self.has_proper_structure:
            points += 3
        if self.has_semicolons:
            points += 1
        if self.has_braces_balanced:
            points += 2
        if self.is_valid_code:
            points += 2
        self.score = points / 10.0


@dataclass
class APIScore:
    """API usage scoring."""
    apis_found: List[str] = field(default_factory=list)
    apis_missing: List[str] = field(default_factory=list)
    has_l4d2_specific: bool = False
    has_correct_signatures: bool = False
    score: float = 0.0

    def calculate(self, expected_apis: List[str]):
        """Calculate API score based on expected APIs."""
        if not expected_apis:
            self.score = 1.0
            return

        found_count = len(self.apis_found)
        total = len(expected_apis)
        base_score = found_count / total if total > 0 else 0

        # Bonus for L4D2-specific APIs
        bonus = 0.1 if self.has_l4d2_specific else 0
        # Bonus for correct signatures
        bonus += 0.1 if self.has_correct_signatures else 0

        self.score = min(1.0, base_score + bonus)


@dataclass
class CompletenessScore:
    """Completeness scoring."""
    has_function_body: bool = False
    has_error_handling: bool = False
    has_comments: bool = False
    code_length: int = 0
    is_complete_solution: bool = False
    score: float = 0.0

    def calculate(self):
        """Calculate completeness score."""
        points = 0
        if self.has_function_body:
            points += 3
        if self.has_error_handling:
            points += 2
        if self.has_comments:
            points += 1
        if self.code_length >= 10:
            points += 2
        elif self.code_length >= 5:
            points += 1
        if self.is_complete_solution:
            points += 2
        self.score = points / 10.0


@dataclass
class ModelOutput:
    """Output from a single model generation."""
    model_name: str
    prompt_id: str
    prompt_text: str
    response: str
    generation_time_ms: float
    syntax_score: SyntaxScore = field(default_factory=SyntaxScore)
    api_score: APIScore = field(default_factory=APIScore)
    completeness_score: CompletenessScore = field(default_factory=CompletenessScore)
    total_score: float = 0.0
    error: Optional[str] = None

    def calculate_total(self):
        """Calculate weighted total score."""
        self.total_score = (
            self.syntax_score.score * 0.3 +
            self.api_score.score * 0.4 +
            self.completeness_score.score * 0.3
        )


@dataclass
class ModelComparison:
    """Complete comparison results."""
    timestamp: str
    models_compared: List[str]
    test_prompts: List[Dict]
    results: Dict[str, List[ModelOutput]]  # model_name -> outputs
    summary: Dict[str, Dict[str, float]]  # model_name -> metrics


# =============================================================================
# OLLAMA INTERFACE
# =============================================================================

class OllamaInterface:
    """Interface for Ollama model inference."""

    SOURCEPAWN_SYSTEM = """You are an expert SourcePawn developer for Left 4 Dead 2 modding.
Generate clean, working SourceMod plugin code with proper includes, syntax, and L4D2-specific APIs.
Include #include <sourcemod> and other necessary includes. Use proper semicolons and braces."""

    VSCRIPT_SYSTEM = """You are an expert VScript/Squirrel developer for Left 4 Dead 2 modding.
Generate clean, working VScript code for Director scripts, mutations, and game logic."""

    def __init__(self):
        self._check_ollama()

    def _check_ollama(self) -> bool:
        """Verify Ollama is available."""
        if not shutil.which("ollama"):
            raise RuntimeError("Ollama not found. Install from https://ollama.ai")
        return True

    def list_models(self) -> List[str]:
        """Get list of available Ollama models."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            models = []
            for line in result.stdout.strip().split("\n")[1:]:  # Skip header
                if line.strip():
                    model_name = line.split()[0]
                    models.append(model_name)
            return models
        except Exception as e:
            print(f"Error listing models: {e}")
            return []

    def list_l4d2_models(self) -> List[str]:
        """Get list of L4D2-related models."""
        all_models = self.list_models()
        return [m for m in all_models if "l4d2" in m.lower()]

    def generate(self, model: str, prompt: str, language: str = "sourcepawn") -> Tuple[str, float]:
        """Generate code using Ollama model."""
        system = self.VSCRIPT_SYSTEM if language == "vscript" else self.SOURCEPAWN_SYSTEM
        full_prompt = f"{system}\n\nUser: {prompt}\n\nAssistant:"

        start_time = time.time()

        try:
            result = subprocess.run(
                ["ollama", "run", model, full_prompt],
                capture_output=True,
                text=True,
                timeout=120
            )
            elapsed_ms = (time.time() - start_time) * 1000
            return result.stdout.strip(), elapsed_ms
        except subprocess.TimeoutExpired:
            return "Error: Generation timed out", 120000.0
        except Exception as e:
            return f"Error: {e}", 0.0


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================

def analyze_syntax(code: str, language: str = "sourcepawn") -> SyntaxScore:
    """Analyze code syntax correctness."""
    score = SyntaxScore()

    if not code or "Error:" in code:
        return score

    code_lower = code.lower()

    # Check for includes (SourcePawn)
    if language == "sourcepawn":
        score.has_includes = "#include" in code
        score.has_semicolons = code.count(";") >= 3

        # Check braces balance
        open_braces = code.count("{")
        close_braces = code.count("}")
        score.has_braces_balanced = open_braces == close_braces and open_braces > 0

        # Check for proper structure (function definitions)
        has_function = bool(re.search(r"(public|stock|static|void)\s+\w+\s*\(", code))
        has_plugin_info = "public plugin myinfo" in code_lower or "myinfo" in code_lower
        score.has_proper_structure = has_function or has_plugin_info

        # Overall validity check
        score.is_valid_code = (
            score.has_includes and
            score.has_semicolons and
            score.has_braces_balanced
        )
    else:
        # VScript
        score.has_includes = True  # VScript doesn't use includes
        score.has_semicolons = True  # Squirrel doesn't require semicolons

        # Check braces balance
        open_braces = code.count("{")
        close_braces = code.count("}")
        score.has_braces_balanced = open_braces == close_braces

        # Check for proper structure
        has_function = "function" in code_lower
        has_table = "<-" in code or "DirectorOptions" in code
        score.has_proper_structure = has_function or has_table

        score.is_valid_code = score.has_proper_structure

    score.calculate()
    return score


def analyze_api_usage(code: str, expected_apis: List[str], language: str = "sourcepawn") -> APIScore:
    """Analyze API usage in generated code."""
    score = APIScore()

    if not code or "Error:" in code:
        score.apis_missing = expected_apis.copy()
        return score

    code_lower = code.lower()

    # Check for expected APIs
    for api in expected_apis:
        if api.lower() in code_lower:
            score.apis_found.append(api)
        else:
            score.apis_missing.append(api)

    # Check for L4D2-specific APIs
    l4d2_apis = [
        "l4d_", "left4dead", "survivor", "infected", "special",
        "tank", "witch", "boomer", "hunter", "smoker", "charger", "jockey", "spitter",
        "saferoom", "finale", "director"
    ]
    score.has_l4d2_specific = any(api in code_lower for api in l4d2_apis)

    # Check for correct function signatures (basic heuristic)
    if language == "sourcepawn":
        # Check for proper callback signatures
        has_proper_callback = bool(re.search(
            r"public\s+(Action|void)\s+\w+\s*\([^)]*\)", code
        ))
        score.has_correct_signatures = has_proper_callback
    else:
        score.has_correct_signatures = "function" in code_lower

    score.calculate(expected_apis)
    return score


def analyze_completeness(code: str, language: str = "sourcepawn") -> CompletenessScore:
    """Analyze code completeness."""
    score = CompletenessScore()

    if not code or "Error:" in code:
        return score

    # Count code lines (excluding empty lines and comments)
    lines = [l.strip() for l in code.split("\n") if l.strip() and not l.strip().startswith("//")]
    score.code_length = len(lines)

    # Check for function body
    has_function_with_body = bool(re.search(r"\{[^}]+\}", code))
    score.has_function_body = has_function_with_body

    # Check for error handling
    error_patterns = ["if", "IsClientInGame", "IsValidClient", "IsValidEntity", "try", "catch"]
    score.has_error_handling = any(p.lower() in code.lower() for p in error_patterns)

    # Check for comments
    score.has_comments = "//" in code or "/*" in code

    # Check if it's a complete solution (has both declaration and implementation)
    if language == "sourcepawn":
        has_plugin = "myinfo" in code.lower() or "OnPluginStart" in code
        has_implementation = "{" in code and "}" in code and score.code_length >= 10
        score.is_complete_solution = has_implementation
    else:
        score.is_complete_solution = "function" in code.lower() and score.code_length >= 5

    score.calculate()
    return score


# =============================================================================
# EVALUATION RUNNER
# =============================================================================

def evaluate_model(
    ollama: OllamaInterface,
    model_name: str,
    prompts: List[Dict],
    verbose: bool = True
) -> List[ModelOutput]:
    """Evaluate a single model on all prompts."""
    results = []

    for i, prompt_data in enumerate(prompts):
        if verbose:
            print(f"  [{i+1}/{len(prompts)}] {prompt_data['id']}...", end=" ", flush=True)

        try:
            response, gen_time = ollama.generate(
                model_name,
                prompt_data["prompt"],
                prompt_data.get("language", "sourcepawn")
            )

            output = ModelOutput(
                model_name=model_name,
                prompt_id=prompt_data["id"],
                prompt_text=prompt_data["prompt"],
                response=response,
                generation_time_ms=gen_time
            )

            # Score the output
            output.syntax_score = analyze_syntax(
                response,
                prompt_data.get("language", "sourcepawn")
            )
            output.api_score = analyze_api_usage(
                response,
                prompt_data.get("expected_apis", []),
                prompt_data.get("language", "sourcepawn")
            )
            output.completeness_score = analyze_completeness(
                response,
                prompt_data.get("language", "sourcepawn")
            )
            output.calculate_total()

            if verbose:
                status = "PASS" if output.total_score >= 0.5 else "FAIL"
                print(f"{status} ({output.total_score:.1%})")

        except Exception as e:
            output = ModelOutput(
                model_name=model_name,
                prompt_id=prompt_data["id"],
                prompt_text=prompt_data["prompt"],
                response="",
                generation_time_ms=0,
                error=str(e)
            )
            if verbose:
                print(f"ERROR: {e}")

        results.append(output)

    return results


def run_comparison(
    models: List[str],
    prompts: List[Dict],
    verbose: bool = True
) -> ModelComparison:
    """Run comparison across all models."""
    ollama = OllamaInterface()

    comparison = ModelComparison(
        timestamp=datetime.now().isoformat(),
        models_compared=models,
        test_prompts=prompts,
        results={},
        summary={}
    )

    for model in models:
        if verbose:
            print(f"\nEvaluating model: {model}")
            print("-" * 50)

        results = evaluate_model(ollama, model, prompts, verbose)
        comparison.results[model] = results

        # Calculate summary statistics
        if results:
            avg_total = sum(r.total_score for r in results) / len(results)
            avg_syntax = sum(r.syntax_score.score for r in results) / len(results)
            avg_api = sum(r.api_score.score for r in results) / len(results)
            avg_complete = sum(r.completeness_score.score for r in results) / len(results)
            avg_time = sum(r.generation_time_ms for r in results) / len(results)
            pass_rate = sum(1 for r in results if r.total_score >= 0.5) / len(results)

            comparison.summary[model] = {
                "total_score": avg_total,
                "syntax_score": avg_syntax,
                "api_score": avg_api,
                "completeness_score": avg_complete,
                "avg_generation_time_ms": avg_time,
                "pass_rate": pass_rate,
                "tests_passed": sum(1 for r in results if r.total_score >= 0.5),
                "tests_total": len(results)
            }

    return comparison


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_markdown_report(comparison: ModelComparison) -> str:
    """Generate a detailed markdown comparison report."""
    lines = [
        "# L4D2 Model Comparison Report",
        "",
        f"Generated: {comparison.timestamp}",
        "",
        "## Executive Summary",
        "",
    ]

    # Summary table
    lines.append("| Model | Total Score | Pass Rate | Syntax | API Usage | Completeness | Avg Time |")
    lines.append("|-------|-------------|-----------|--------|-----------|--------------|----------|")

    for model in comparison.models_compared:
        if model in comparison.summary:
            s = comparison.summary[model]
            lines.append(
                f"| {model} | {s['total_score']:.1%} | {s['pass_rate']:.1%} | "
                f"{s['syntax_score']:.1%} | {s['api_score']:.1%} | "
                f"{s['completeness_score']:.1%} | {s['avg_generation_time_ms']:.0f}ms |"
            )

    lines.append("")

    # Best model
    if comparison.summary:
        best_model = max(comparison.summary.keys(), key=lambda m: comparison.summary[m]["total_score"])
        best_score = comparison.summary[best_model]["total_score"]
        lines.append(f"**Best Performing Model:** {best_model} ({best_score:.1%})")
        lines.append("")

    # Detailed results per prompt
    lines.append("## Test Results by Prompt")
    lines.append("")

    for prompt_data in comparison.test_prompts:
        prompt_id = prompt_data["id"]
        lines.append(f"### Test: {prompt_id}")
        lines.append("")
        lines.append(f"**Prompt:** {prompt_data['prompt']}")
        lines.append("")
        lines.append(f"**Expected APIs:** {', '.join(prompt_data.get('expected_apis', []))}")
        lines.append("")

        # Results table for this prompt
        lines.append("| Model | Score | Syntax | API | Complete | APIs Found | APIs Missing |")
        lines.append("|-------|-------|--------|-----|----------|------------|--------------|")

        for model in comparison.models_compared:
            if model in comparison.results:
                for output in comparison.results[model]:
                    if output.prompt_id == prompt_id:
                        apis_found = ", ".join(output.api_score.apis_found[:3])
                        if len(output.api_score.apis_found) > 3:
                            apis_found += "..."
                        apis_missing = ", ".join(output.api_score.apis_missing[:3])
                        if len(output.api_score.apis_missing) > 3:
                            apis_missing += "..."

                        status = "PASS" if output.total_score >= 0.5 else "FAIL"
                        lines.append(
                            f"| {model} | {output.total_score:.1%} ({status}) | "
                            f"{output.syntax_score.score:.1%} | {output.api_score.score:.1%} | "
                            f"{output.completeness_score.score:.1%} | {apis_found} | {apis_missing} |"
                        )
                        break

        lines.append("")

    # Code samples
    lines.append("## Generated Code Samples")
    lines.append("")

    for prompt_data in comparison.test_prompts[:3]:  # First 3 prompts
        prompt_id = prompt_data["id"]
        lines.append(f"### {prompt_id}")
        lines.append("")

        for model in comparison.models_compared:
            if model in comparison.results:
                for output in comparison.results[model]:
                    if output.prompt_id == prompt_id:
                        lines.append(f"#### {model}")
                        lines.append("")
                        # Truncate long responses
                        code = output.response[:2000]
                        if len(output.response) > 2000:
                            code += "\n... (truncated)"
                        lang = prompt_data.get("language", "sourcepawn")
                        lines.append(f"```{lang}")
                        lines.append(code)
                        lines.append("```")
                        lines.append("")
                        break

    # Scoring methodology
    lines.append("## Scoring Methodology")
    lines.append("")
    lines.append("### Syntax Score (30%)")
    lines.append("- Has proper includes (+2)")
    lines.append("- Has proper structure (functions/callbacks) (+3)")
    lines.append("- Has semicolons where required (+1)")
    lines.append("- Has balanced braces (+2)")
    lines.append("- Overall valid code (+2)")
    lines.append("")
    lines.append("### API Usage Score (40%)")
    lines.append("- Percentage of expected APIs found")
    lines.append("- Bonus for L4D2-specific APIs (+10%)")
    lines.append("- Bonus for correct function signatures (+10%)")
    lines.append("")
    lines.append("### Completeness Score (30%)")
    lines.append("- Has function body (+3)")
    lines.append("- Has error handling (+2)")
    lines.append("- Has comments (+1)")
    lines.append("- Code length >= 10 lines (+2) or >= 5 lines (+1)")
    lines.append("- Is complete solution (+2)")
    lines.append("")
    lines.append("---")
    lines.append("*Generated by L4D2-AI-Architect model comparison tool*")

    return "\n".join(lines)


def generate_json_report(comparison: ModelComparison) -> Dict:
    """Generate JSON report data."""
    return {
        "timestamp": comparison.timestamp,
        "models_compared": comparison.models_compared,
        "test_prompts": comparison.test_prompts,
        "summary": comparison.summary,
        "results": {
            model: [
                {
                    "prompt_id": o.prompt_id,
                    "prompt_text": o.prompt_text,
                    "response": o.response[:1000],  # Truncate for JSON
                    "generation_time_ms": o.generation_time_ms,
                    "total_score": o.total_score,
                    "syntax_score": o.syntax_score.score,
                    "api_score": o.api_score.score,
                    "completeness_score": o.completeness_score.score,
                    "apis_found": o.api_score.apis_found,
                    "apis_missing": o.api_score.apis_missing,
                    "error": o.error
                }
                for o in outputs
            ]
            for model, outputs in comparison.results.items()
        }
    }


# =============================================================================
# CLI
# =============================================================================

def discover_models() -> List[str]:
    """Discover available L4D2 models."""
    models = []

    # Check Ollama models
    try:
        ollama = OllamaInterface()
        ollama_models = ollama.list_l4d2_models()
        models.extend(ollama_models)
    except RuntimeError:
        print("Warning: Ollama not available")

    # Check model adapters directory for reference
    if MODEL_ADAPTERS_DIR.exists():
        for adapter_dir in MODEL_ADAPTERS_DIR.iterdir():
            if adapter_dir.is_dir() and adapter_dir.name.startswith("l4d2"):
                # Note: These are LoRA adapters, not Ollama models
                # They would need to be exported first
                pass

    return models


def print_summary(comparison: ModelComparison):
    """Print summary to console."""
    print()
    print("=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print()

    # Summary table
    print(f"{'Model':<30} {'Score':>10} {'Pass Rate':>12} {'Avg Time':>12}")
    print("-" * 70)

    for model in comparison.models_compared:
        if model in comparison.summary:
            s = comparison.summary[model]
            print(
                f"{model:<30} {s['total_score']:>9.1%} "
                f"{s['pass_rate']:>11.1%} {s['avg_generation_time_ms']:>10.0f}ms"
            )

    print()

    # Best model
    if comparison.summary:
        best = max(comparison.summary.keys(), key=lambda m: comparison.summary[m]["total_score"])
        print(f"Best Model: {best} ({comparison.summary[best]['total_score']:.1%})")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Compare L4D2 code generation models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare all L4D2 Ollama models
    python compare_all_models.py

    # Compare specific models
    python compare_all_models.py --models l4d2-code-v10plus,l4d2-code-v11

    # Custom prompts
    python compare_all_models.py --prompts "Write a heal function"

    # Save report
    python compare_all_models.py --output results/comparison.md
"""
    )

    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of model names to compare"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        help="Custom test prompts (uses defaults if not specified)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for markdown report"
    )
    parser.add_argument(
        "--json",
        type=str,
        help="Output path for JSON report"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    # List models mode
    if args.list_models:
        print("Available L4D2 models:")
        models = discover_models()
        if models:
            for m in models:
                print(f"  - {m}")
        else:
            print("  No L4D2 models found in Ollama")
            print("\nTo install a model, export from training and run:")
            print("  ollama create l4d2-code-v10plus -f exports/l4d2-v10plus/gguf/Modelfile")
        return

    # Determine models to compare
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        models = discover_models()
        if not models:
            print("No L4D2 models found. Install models or specify with --models")
            print("Run with --list-models to see available options")
            sys.exit(1)

    print(f"Comparing {len(models)} model(s): {', '.join(models)}")

    # Determine test prompts
    if args.prompts:
        prompts = [
            {
                "id": f"custom_{i}",
                "prompt": p,
                "language": "sourcepawn",
                "expected_apis": [],
                "expected_syntax": []
            }
            for i, p in enumerate(args.prompts)
        ]
    else:
        prompts = DEFAULT_TEST_PROMPTS

    print(f"Running {len(prompts)} test(s)")

    # Run comparison
    verbose = not args.quiet
    comparison = run_comparison(models, prompts, verbose)

    # Print summary
    print_summary(comparison)

    # Generate reports
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Markdown report
    md_report = generate_markdown_report(comparison)

    if args.output:
        output_path = safe_path(args.output, PROJECT_ROOT, create_parents=True)
        safe_write_text(str(output_path), md_report, PROJECT_ROOT)
        print(f"\nMarkdown report saved to: {output_path}")
    else:
        default_output = RESULTS_DIR / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        safe_write_text(str(default_output), md_report, PROJECT_ROOT)
        print(f"\nMarkdown report saved to: {default_output}")

    # JSON report
    if args.json:
        json_data = generate_json_report(comparison)
        json_output = safe_path(args.json, PROJECT_ROOT, create_parents=True)
        with open(json_output, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"JSON report saved to: {json_output}")


if __name__ == "__main__":
    main()
