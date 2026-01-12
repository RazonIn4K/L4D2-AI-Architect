#!/usr/bin/env python3
"""
Synthetic Training Data Generator for L4D2 SourcePawn Model

Generates high-quality training examples using GPT-4 to augment the dataset.
Designed to run on Vultr GPU instances to maximize credit usage.

Usage:
    python scripts/training/generate_synthetic_data.py --num-examples 100
    python scripts/training/generate_synthetic_data.py --categories "special_infected,api_correctness"
    python scripts/training/generate_synthetic_data.py --anti-patterns --num-examples 50
"""

import argparse
import json
import os
import sys
import time
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_write_jsonl

PROJECT_ROOT = Path(__file__).parent.parent.parent

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

# Priority weights based on eval results (higher = generate more examples)
# V8 Eval Results:
# - survivor_mechanics: 100% pass, 36.2 avg (IMPROVED from V7's 83%)
# - api_correctness: 100% pass, 37.7 avg (IMPROVED from V7's 31.3)
# - special_infected: 93% pass, 35.2 avg (DROPPED from V7's 100%)
# - map_events: 100% pass, 29.3 avg (LOWEST SCORE)
# - events: 100% pass, 34.3 avg
CATEGORY_WEIGHTS = {
    "map_events": 3.0,              # Highest priority (lowest score in V8: 29.3)
    "special_infected_advanced": 2.5,  # High priority (dropped to 93% in V8)
    "special_infected": 2.0,        # Medium-high (reinforce correct patterns)
    "error_handling": 1.5,          # Important for robustness
    "survivor_mechanics": 1.0,      # Already improved in V8
    "api_correctness": 1.0,         # Already improved in V8
    "events": 1.0,
    "admin_commands": 0.5,
}

# Categories and their prompt templates
CATEGORIES = {
    "special_infected": {
        "description": "Plugins dealing with Hunter, Smoker, Boomer, Charger, Jockey, Spitter, Witch, Tank",
        "prompts": [
            "Write a SourcePawn plugin that tracks {infected} {action} and {result}",
            "Create a plugin that modifies {infected} behavior when {condition}",
            "Write a plugin that announces {infected} attacks with {detail}",
        ],
        "variables": {
            "infected": ["Hunter", "Smoker", "Boomer", "Charger", "Jockey", "Spitter", "Witch", "Tank"],
            "action": ["pounces", "grabs", "attacks", "spawns", "dies", "damages survivors"],
            "result": ["shows damage dealt", "plays a sound", "gives rewards", "announces to team"],
            "condition": ["survivors are close", "during panic events", "health is low", "timer expires"],
            "detail": ["damage numbers", "distance", "attacker name", "victim info"]
        }
    },
    "survivor_mechanics": {
        "description": "Plugins for survivor abilities, health, speed, items - HIGH PRIORITY based on eval gaps",
        "prompts": [
            "Write a plugin that {action} survivors when they {trigger}",
            "Create a plugin that tracks survivor {stat} and {display}",
            "Write a plugin that modifies survivor {attribute} based on {condition}",
            "Write a plugin that handles friendly fire damage with proper damage callback hooks",
            "Create a plugin that gives survivors temporary health after {trigger}",
            "Write a plugin that handles survivor incap states and revive mechanics",
        ],
        "variables": {
            "action": ["heals", "gives speed boost to", "protects", "revives", "teleports", "gives temporary health to"],
            "trigger": ["kill special infected", "enter saferoom", "take damage", "use medkit", "revive teammate", "are revived", "kill a Tank"],
            "stat": ["kills", "damage dealt", "healing done", "revives", "deaths", "friendly fire", "headshots"],
            "display": ["shows on HUD", "announces at round end", "saves to database", "displays leaderboard"],
            "attribute": ["speed using m_flLaggedMovementValue", "health", "damage resistance", "reload speed"],
            "condition": ["player count", "difficulty", "time elapsed", "kills achieved", "health is below 40"]
        }
    },
    "error_handling": {
        "description": "Plugins demonstrating proper error handling and edge cases - NEW based on eval gaps",
        "prompts": [
            "Write a plugin that safely handles player disconnect during {action}",
            "Create a plugin that validates client indices before {operation}",
            "Write a plugin that handles errors when {scenario} fails",
            "Create a plugin that properly checks IsClientInGame before {action}",
        ],
        "variables": {
            "action": ["healing", "teleporting", "damaging", "speed boost", "reviving"],
            "operation": ["applying damage", "setting speed", "healing", "spawning items"],
            "scenario": ["finding target player", "getting entity property", "SDKCall fails", "timer callback runs"]
        }
    },
    "events": {
        "description": "Plugins hooking game events like round start, panic, finale",
        "prompts": [
            "Write a plugin that {action} when {event} occurs",
            "Create a plugin that tracks {metric} during {event}",
            "Write a plugin that modifies game behavior during {event}",
        ],
        "variables": {
            "action": ["spawns zombies", "plays music", "gives items", "shows message", "starts timer"],
            "event": ["round starts", "panic event begins", "finale starts", "tank spawns", "witch is startled"],
            "metric": ["survivor health", "zombie kills", "time elapsed", "damage taken"]
        }
    },
    "api_correctness": {
        "description": "Examples specifically teaching correct L4D2 API usage - HIGH PRIORITY based on eval gaps",
        "prompts": [
            "Write a plugin using {api} to {purpose}",
            "Create a plugin that demonstrates correct usage of {api}",
            "Write a plugin that {action} using the proper L4D2 API",
            "Write a plugin that uses GetRandomFloat to generate a delay between {min} and {max} seconds",
            "Create a plugin that correctly uses {event} event for {infected} tracking",
            "Write a plugin that modifies player speed using m_flLaggedMovementValue property",
            "Create a plugin using SDKHooks_TakeDamage for custom damage with {damage_type}",
        ],
        "variables": {
            "api": ["GetRandomFloat", "GetRandomInt", "m_flLaggedMovementValue", "SDKHooks_TakeDamage", "SetEntPropFloat", "GetEntProp"],
            "purpose": ["generate random delays", "set player speed", "apply damage", "create timers", "modify entity properties"],
            "action": ["sets random spawn intervals", "modifies movement speed", "handles damage events", "tracks player stats"],
            "min": ["1.0", "5.0", "10.0", "0.5"],
            "max": ["5.0", "15.0", "30.0", "3.0"],
            "event": ["lunge_pounce", "tongue_grab", "player_now_it", "charger_carry_start", "jockey_ride"],
            "infected": ["Hunter", "Smoker", "Boomer", "Charger", "Jockey"],
            "damage_type": ["fire damage", "explosive damage", "fall damage", "melee damage"]
        }
    },
    "admin_commands": {
        "description": "Plugins with admin commands and ConVars",
        "prompts": [
            "Write a plugin with an admin command to {action}",
            "Create a plugin with ConVars to configure {setting}",
            "Write a plugin that lets admins {capability}",
        ],
        "variables": {
            "action": ["spawn tanks", "heal all survivors", "toggle god mode", "set difficulty"],
            "setting": ["spawn rates", "damage multipliers", "timer intervals", "team balance"],
            "capability": ["kick infected players", "restart rounds", "modify game settings"]
        }
    },
    "map_events": {
        "description": "Plugins for map transitions, saferooms, and level progression - LOW SCORE in evals",
        "prompts": [
            "Write a plugin that detects when survivors {map_action}",
            "Create a plugin that tracks {metric} across map transitions",
            "Write a plugin that modifies {behavior} when survivors reach the saferoom",
            "Create a plugin that handles {event} during map {phase}",
            "Write a plugin that saves {data} when the map changes",
        ],
        "variables": {
            "map_action": ["enter a saferoom", "leave the saferoom", "reach the rescue vehicle", "trigger the finale", "complete a chapter"],
            "metric": ["total kills", "damage taken", "time elapsed", "items used", "special infected killed"],
            "behavior": ["zombie spawns", "item availability", "door locks", "weapon spawns"],
            "event": ["round_end", "map_transition", "finale_start", "rescue_vehicle_ready"],
            "phase": ["start", "middle", "finale", "transition"],
            "data": ["player stats", "inventory", "scores", "achievements"]
        }
    },
    "special_infected_advanced": {
        "description": "Advanced special infected mechanics - DROPPED to 93% in V8 evals",
        "prompts": [
            "Write a plugin that tracks {infected} {stat} and announces {result}",
            "Create a plugin that modifies {infected} spawn behavior based on {condition}",
            "Write a plugin using the correct {event} event to track {infected} attacks",
            "Create a plugin that calculates {infected} damage using proper L4D2 events",
            "Write a plugin that prevents {infected} from {action} during {phase}",
        ],
        "variables": {
            "infected": ["Hunter", "Smoker", "Boomer", "Charger", "Jockey", "Spitter", "Witch", "Tank"],
            "stat": ["total damage", "successful attacks", "kills", "incaps caused", "death count"],
            "result": ["to all players", "with damage numbers", "in chat", "as a HUD element"],
            "condition": ["survivor count", "difficulty level", "time elapsed", "tank is alive"],
            "event": ["lunge_pounce", "tongue_grab", "player_now_it", "charger_carry_start", "jockey_ride", "spit_burst"],
            "action": ["spawning", "attacking", "using abilities", "targeting specific players"],
            "phase": ["panic events", "finales", "tank fights", "rescue sequences"]
        }
    }
}

# System prompt for generating training examples
GENERATOR_SYSTEM_PROMPT = """You are an expert SourcePawn developer creating training examples for an AI model.

Generate a complete, working SourcePawn plugin for Left 4 Dead 2 that accomplishes the requested task.

CRITICAL REQUIREMENTS:
1. Use CORRECT L4D2 APIs:
   - GetRandomFloat() NOT RandomFloat()
   - GetRandomInt() NOT RandomInt()
   - m_flLaggedMovementValue for speed NOT m_flSpeed or m_flMaxSpeed
   - SDKHooks_TakeDamage() NOT TakeDamage()

2. Use CORRECT L4D2 Events:
   - lunge_pounce NOT pounce (Hunter)
   - tongue_grab NOT smoker_tongue_grab (Smoker)
   - player_now_it NOT boomer_vomit (Boomer bile)
   - charger_carry_start NOT charger_grab (Charger)

3. Include proper structure:
   - #pragma semicolon 1
   - #pragma newdecls required
   - #include <sourcemod> and other necessary includes
   - Plugin myinfo block
   - OnPluginStart() function

4. Use proper coding practices:
   - Check IsClientInGame() before client operations
   - Validate client indices (> 0 && <= MaxClients)
   - Use GetClientTeam() (2=Survivor, 3=Infected)
   - Handle events properly with GetEventInt/GetEventFloat

Output ONLY the SourcePawn code, no explanations."""

# Anti-pattern teaching prompt
ANTIPATTERN_SYSTEM_PROMPT = """You are creating a training example that teaches the CORRECT way to do something in L4D2 SourcePawn.

The user will ask for a plugin. You must:
1. Write the CORRECT implementation
2. Add a comment block at the top explaining what NOT to do

Format:
```
/*
 * CORRECT IMPLEMENTATION
 * 
 * DO NOT USE:
 * - [wrong API] - Reason why it's wrong
 * 
 * INSTEAD USE:
 * - [correct API] - Reason why it's correct
 */

[rest of the plugin code]
```

CRITICAL - These are the WRONG APIs to warn about:
- RandomFloat() is WRONG → Use GetRandomFloat()
- RandomInt() is WRONG → Use GetRandomInt()  
- m_flSpeed is WRONG → Use m_flLaggedMovementValue
- "pounce" event is WRONG → Use "lunge_pounce"
- "smoker_tongue_grab" is WRONG → Use "tongue_grab"
- "boomer_vomit" is WRONG → Use "player_now_it"
- "charger_grab" is WRONG → Use "charger_carry_start"
- TakeDamage() is WRONG → Use SDKHooks_TakeDamage()

Output ONLY the code with the warning comments."""


def get_client() -> OpenAI:
    """Get OpenAI client."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)
    return OpenAI(api_key=api_key)


def generate_prompt(category: str) -> str:
    """Generate a random prompt for a category."""
    cat_data = CATEGORIES[category]
    template = random.choice(cat_data["prompts"])
    
    # Fill in variables
    for var_name, var_options in cat_data["variables"].items():
        placeholder = "{" + var_name + "}"
        if placeholder in template:
            template = template.replace(placeholder, random.choice(var_options))
    
    return template


def generate_example(client: OpenAI, prompt: str, is_antipattern: bool = False) -> Dict:
    """Generate a single training example."""
    system_prompt = ANTIPATTERN_SYSTEM_PROMPT if is_antipattern else GENERATOR_SYSTEM_PROMPT
    
    response = client.chat.completions.create(
        model="gpt-4o",  # Use GPT-4 for high-quality generation
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2048,
        temperature=0.7  # Some creativity for variety
    )
    
    code = response.choices[0].message.content
    
    # Clean up code (remove markdown if present)
    if "```sourcepawn" in code:
        code = code.split("```sourcepawn")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]
    
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are an expert SourcePawn developer for Left 4 Dead 2 SourceMod plugins. Write clean, well-documented code with proper error handling. Use correct L4D2 APIs and events."
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": code.strip()}
        ]
    }


def load_eval_results(eval_path: Path) -> Dict:
    """Load evaluation results to inform priority weighting."""
    if not eval_path.exists():
        return {}
    try:
        with open(eval_path) as f:
            return json.load(f)
    except Exception:
        return {}


def calculate_weighted_distribution(categories: List[str], num_examples: int, use_weights: bool) -> Dict[str, int]:
    """Calculate how many examples to generate per category based on weights."""
    if not use_weights:
        # Equal distribution
        per_cat = num_examples // len(categories)
        remainder = num_examples % len(categories)
        return {cat: per_cat + (1 if i < remainder else 0) for i, cat in enumerate(categories)}

    # Weighted distribution
    total_weight = sum(CATEGORY_WEIGHTS.get(cat, 1.0) for cat in categories)
    distribution = {}
    remaining = num_examples

    for i, cat in enumerate(categories):
        weight = CATEGORY_WEIGHTS.get(cat, 1.0)
        if i == len(categories) - 1:
            # Last category gets remainder
            distribution[cat] = remaining
        else:
            count = int(num_examples * (weight / total_weight))
            distribution[cat] = count
            remaining -= count

    return distribution


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--num-examples", type=int, default=50, help="Number of examples to generate")
    parser.add_argument("--categories", default="all", help="Comma-separated categories or 'all'")
    parser.add_argument("--anti-patterns", action="store_true", help="Generate anti-pattern teaching examples")
    parser.add_argument("--output", default="data/synthetic/generated.jsonl", help="Output file")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls (seconds)")
    parser.add_argument("--prioritize-gaps", action="store_true",
                        help="Weight categories based on eval gaps (survivor_mechanics, api_correctness get more)")
    parser.add_argument("--eval-results", default="data/eval_results.json",
                        help="Path to eval results for dynamic weighting")
    args = parser.parse_args()

    # Parse categories
    if args.categories == "all":
        categories = list(CATEGORIES.keys())
    else:
        categories = [c.strip() for c in args.categories.split(",")]
        for c in categories:
            if c not in CATEGORIES:
                print(f"ERROR: Unknown category '{c}'")
                print(f"Available: {', '.join(CATEGORIES.keys())}")
                sys.exit(1)
    
    # Calculate weighted distribution
    distribution = calculate_weighted_distribution(categories, args.num_examples, args.prioritize_gaps)

    print("=" * 60)
    print("L4D2 Synthetic Training Data Generator")
    print("=" * 60)
    print(f"Examples to generate: {args.num_examples}")
    print(f"Categories: {', '.join(categories)}")
    print(f"Prioritize gaps: {args.prioritize_gaps}")
    if args.prioritize_gaps:
        print(f"Distribution: {distribution}")
    print(f"Anti-pattern mode: {args.anti_patterns}")
    print(f"Output: {args.output}")
    print("-" * 60)

    client = get_client()
    examples = []
    errors = 0

    for category in categories:
        count = distribution[category]
        if count == 0:
            continue
        print(f"\nGenerating {count} examples for '{category}'...")
        
        for j in range(count):
            prompt = generate_prompt(category)
            print(f"  [{j+1}/{count}] {prompt[:60]}...", end=" ", flush=True)
            
            try:
                example = generate_example(client, prompt, args.anti_patterns)
                example["metadata"] = {
                    "category": category,
                    "anti_pattern": args.anti_patterns,
                    "generated_at": datetime.now().isoformat()
                }
                examples.append(example)
                print("OK")
            except Exception as e:
                print(f"ERROR: {e}")
                errors += 1
            
            time.sleep(args.delay)
    
    # Save output
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")
    
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Generated: {len(examples)} examples")
    print(f"Errors: {errors}")
    print(f"Output: {output_path}")
    
    # Calculate cost estimate
    est_tokens = len(examples) * 2000  # ~2000 tokens per example
    est_cost = est_tokens * 0.01 / 1000  # GPT-4 input cost
    print(f"Estimated cost: ${est_cost:.2f}")


if __name__ == "__main__":
    main()
