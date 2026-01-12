#!/usr/bin/env python3
"""
Create Contrastive Anti-Pattern Pairs for L4D2 Training

This script creates pairs of correct/incorrect examples to strongly reinforce
L4D2-specific API knowledge. Target: 8-10% of dataset as anti-patterns.

Each pattern gets:
- 1 detailed explanation (existing anti-pattern)
- 2-4 contrastive code examples (new)
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Contrastive pairs: Each entry creates multiple training examples
# Format: (prompt_variations, correct_code, incorrect_code, explanation)
CONTRASTIVE_PATTERNS = [
    # Hunter pounce - THE MOST FAILED PATTERN
    {
        "prompts": [
            "Write code to detect when a Hunter pounces a survivor",
            "Hook the Hunter pounce event in L4D2",
            "Track Hunter pounce damage",
            "Create a Hunter pounce announcer",
        ],
        "correct": '''HookEvent("lunge_pounce", Event_HunterPounce);

public void Event_HunterPounce(Event event, const char[] name, bool dontBroadcast)
{
    int hunter = GetClientOfUserId(event.GetInt("userid"));
    int victim = GetClientOfUserId(event.GetInt("victim"));
    int damage = event.GetInt("damage");

    PrintToChatAll("Hunter %N pounced %N for %d damage!", hunter, victim, damage);
}''',
        "wrong_event": "pounce",
        "correct_event": "lunge_pounce",
        "explanation": "In L4D2, the Hunter pounce event is 'lunge_pounce', NOT 'pounce'. The event 'pounce' does not exist.",
    },
    # Random float - SECOND MOST FAILED PATTERN
    {
        "prompts": [
            "Generate a random float between two values in SourcePawn",
            "Create a random delay timer",
            "Spawn something at a random time",
            "Use random values for spawn position offset",
        ],
        "correct": '''// Generate random float between min and max
float randomDelay = GetRandomFloat(30.0, 60.0);
CreateTimer(randomDelay, Timer_SpawnSomething);

// Random position offset
float offsetX = GetRandomFloat(-500.0, 500.0);
float offsetY = GetRandomFloat(-500.0, 500.0);

// Random chance (25%)
if (GetRandomFloat(0.0, 1.0) < 0.25)
{
    // Do something
}

// Random integer
int randomInt = GetRandomInt(1, 10);''',
        "wrong_funcs": ["RandomFloat", "RandomInt"],
        "correct_funcs": ["GetRandomFloat", "GetRandomInt"],
        "explanation": "In SourceMod, use GetRandomFloat() and GetRandomInt(), NOT RandomFloat() or RandomInt(). The latter do not exist.",
    },
    # Smoker tongue grab
    {
        "prompts": [
            "Detect when a Smoker grabs a survivor",
            "Hook the Smoker tongue event",
            "Track Smoker pulls",
        ],
        "correct": '''HookEvent("tongue_grab", Event_TongueGrab);
HookEvent("tongue_release", Event_TongueRelease);

public void Event_TongueGrab(Event event, const char[] name, bool dontBroadcast)
{
    int smoker = GetClientOfUserId(event.GetInt("userid"));
    int victim = GetClientOfUserId(event.GetInt("victim"));

    PrintToChatAll("Smoker %N grabbed %N!", smoker, victim);
}''',
        "wrong_event": "smoker_tongue_grab",
        "correct_event": "tongue_grab",
        "explanation": "The Smoker tongue event is 'tongue_grab', NOT 'smoker_tongue_grab'. Related events: tongue_release, tongue_pull_stopped.",
    },
    # Boomer bile
    {
        "prompts": [
            "Detect when a player gets covered in Boomer bile",
            "Track bile hits on survivors",
            "Hook the boomer vomit event",
        ],
        "correct": '''HookEvent("player_now_it", Event_PlayerBiled);
HookEvent("player_no_longer_it", Event_BileWoreOff);

public void Event_PlayerBiled(Event event, const char[] name, bool dontBroadcast)
{
    int victim = GetClientOfUserId(event.GetInt("userid"));
    int attacker = GetClientOfUserId(event.GetInt("attacker"));

    PrintToChatAll("%N got covered in bile!", victim);
}

// Check bile status via property
bool IsPlayerBiled(int client)
{
    return GetEntProp(client, Prop_Send, "m_bIsIT") == 1;
}''',
        "wrong_events": ["boomer_vomit", "boomer_bile", "player_biled"],
        "correct_event": "player_now_it",
        "explanation": "When players get biled, use 'player_now_it' event, NOT 'boomer_vomit' or 'player_biled'. The 'IT' refers to being 'it' (attracting the horde).",
    },
    # Charger carry
    {
        "prompts": [
            "Detect when a Charger grabs a survivor",
            "Track Charger carries",
            "Hook Charger attack events",
        ],
        "correct": '''HookEvent("charger_carry_start", Event_ChargerGrab);
HookEvent("charger_carry_end", Event_ChargerRelease);
HookEvent("charger_pummel_start", Event_PummelStart);

public void Event_ChargerGrab(Event event, const char[] name, bool dontBroadcast)
{
    int charger = GetClientOfUserId(event.GetInt("userid"));
    int victim = GetClientOfUserId(event.GetInt("victim"));

    PrintToChatAll("Charger %N grabbed %N!", charger, victim);
}''',
        "wrong_events": ["charger_grab", "charger_impact", "charger_hit"],
        "correct_event": "charger_carry_start",
        "explanation": "For Charger grabs, use 'charger_carry_start', NOT 'charger_grab' or 'charger_impact'. Related: charger_carry_end, charger_pummel_start/end.",
    },
    # Panic events
    {
        "prompts": [
            "Detect when a panic event (horde) starts",
            "Hook horde events in L4D2",
            "Track panic event timing",
        ],
        "correct": '''HookEvent("create_panic_event", Event_PanicStart);
HookEvent("panic_event_finished", Event_PanicEnd);

public void Event_PanicStart(Event event, const char[] name, bool dontBroadcast)
{
    PrintToChatAll("Horde incoming!");
}

public void Event_PanicEnd(Event event, const char[] name, bool dontBroadcast)
{
    PrintToChatAll("Horde defeated!");
}''',
        "wrong_events": ["panic_start", "panic_end", "horde_start", "horde_end"],
        "correct_events": ["create_panic_event", "panic_event_finished"],
        "explanation": "Panic events use 'create_panic_event' and 'panic_event_finished', NOT 'panic_start/panic_end' or 'horde_start/horde_end'.",
    },
    # Speed modification
    {
        "prompts": [
            "Change a player's movement speed in L4D2",
            "Give a survivor a speed boost",
            "Modify player speed multiplier",
        ],
        "correct": '''// CORRECT: m_flLaggedMovementValue is a MULTIPLIER
void SetPlayerSpeedMultiplier(int client, float multiplier)
{
    // 1.0 = normal speed, 1.5 = 50% faster, 0.5 = 50% slower
    SetEntPropFloat(client, Prop_Send, "m_flLaggedMovementValue", multiplier);
}

// Give 30% speed boost
SetPlayerSpeedMultiplier(client, 1.3);

// Reset to normal
SetPlayerSpeedMultiplier(client, 1.0);''',
        "wrong_props": ["m_flSpeed", "m_flMaxSpeed", "m_flMaxSpeedCrouched"],
        "correct_prop": "m_flLaggedMovementValue",
        "explanation": "In L4D2, use 'm_flLaggedMovementValue' (a multiplier), NOT 'm_flSpeed' or 'm_flMaxSpeed'. The value 1.0 = normal speed.",
    },
    # Jockey ride
    {
        "prompts": [
            "Detect when a Jockey rides a survivor",
            "Track Jockey attacks",
            "Hook Jockey events",
        ],
        "correct": '''HookEvent("jockey_ride", Event_JockeyRide);
HookEvent("jockey_ride_end", Event_JockeyRideEnd);

public void Event_JockeyRide(Event event, const char[] name, bool dontBroadcast)
{
    int jockey = GetClientOfUserId(event.GetInt("userid"));
    int victim = GetClientOfUserId(event.GetInt("victim"));

    PrintToChatAll("Jockey %N is riding %N!", jockey, victim);
}''',
        "wrong_events": ["jockey_grab", "jockey_pounce", "jockey_jump"],
        "correct_event": "jockey_ride",
        "explanation": "The Jockey attack event is 'jockey_ride', NOT 'jockey_grab'. Related: jockey_ride_end.",
    },
]

SYSTEM_PROMPT = "You are an expert SourcePawn developer specializing in SourceMod plugins for Left 4 Dead 2. You know the correct L4D2 APIs, events, and properties. Always use the canonical L4D2 event names and SourceMod functions."


def create_contrastive_examples():
    """Generate contrastive training examples."""
    examples = []

    for pattern in CONTRASTIVE_PATTERNS:
        prompts = pattern["prompts"]
        correct_code = pattern["correct"]
        explanation = pattern["explanation"]

        # Create multiple variations for each pattern
        for i, prompt in enumerate(prompts):
            # Variation 1: Direct question -> correct answer
            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": f"```sourcepawn\n{correct_code}\n```\n\n**Important**: {explanation}"}
                ]
            })

            # Variation 2: Ask about correct pattern explicitly
            if "correct_event" in pattern:
                examples.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"What is the correct event name for {prompt.lower()}?"},
                        {"role": "assistant", "content": f"The correct event is `{pattern['correct_event']}`. {explanation}\n\nExample:\n```sourcepawn\n{correct_code}\n```"}
                    ]
                })

            # Variation 3: Warn about wrong pattern
            wrong_key = None
            for key in ["wrong_event", "wrong_events", "wrong_funcs", "wrong_props"]:
                if key in pattern:
                    wrong_key = key
                    break

            if wrong_key:
                wrong_items = pattern[wrong_key]
                if isinstance(wrong_items, str):
                    wrong_items = [wrong_items]

                for wrong in wrong_items[:2]:  # Limit to 2 wrong examples per pattern
                    examples.append({
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": f"Can I use '{wrong}' in L4D2 SourceMod plugins?"},
                            {"role": "assistant", "content": f"**No, '{wrong}' does not exist in L4D2.** {explanation}\n\nUse this instead:\n```sourcepawn\n{correct_code}\n```"}
                        ]
                    })

    return examples


def main():
    print("=" * 60)
    print("Creating Contrastive Anti-Pattern Pairs")
    print("=" * 60)

    # Generate contrastive examples
    contrastive = create_contrastive_examples()
    print(f"\nGenerated {len(contrastive)} contrastive examples")

    # Write to file
    contrastive_path = PROJECT_ROOT / "data/anti_patterns/contrastive_pairs.jsonl"
    contrastive_path.parent.mkdir(parents=True, exist_ok=True)

    with open(contrastive_path, 'w', encoding='utf-8') as f:
        for ex in contrastive:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    print(f"Saved to: {contrastive_path}")

    # Now merge all anti-patterns into v5 training data
    print("\n" + "=" * 60)
    print("Creating V5 Training Data (8-10% anti-patterns)")
    print("=" * 60)

    # Read cleaned v4 base
    v4_path = PROJECT_ROOT / "data/processed/l4d2_train_v4_cleaned.jsonl"
    with open(v4_path, 'r') as f:
        base_examples = [json.loads(line) for line in f if line.strip()]

    print(f"Base examples (cleaned v4): {len(base_examples)}")

    # Read original anti-patterns
    anti_path = PROJECT_ROOT / "data/anti_patterns/l4d2_anti_patterns.jsonl"
    with open(anti_path, 'r') as f:
        original_anti = []
        for line in f:
            if line.strip():
                ex = json.loads(line)
                # Remove metadata
                for key in ["type", "error_category", "wrong_pattern", "correct_pattern"]:
                    ex.pop(key, None)
                original_anti.append(ex)

    print(f"Original anti-patterns: {len(original_anti)}")
    print(f"New contrastive pairs: {len(contrastive)}")

    # Combine all
    all_examples = base_examples + original_anti + contrastive
    total = len(all_examples)
    anti_count = len(original_anti) + len(contrastive)
    anti_percent = anti_count / total * 100

    print(f"\nTotal examples: {total}")
    print(f"Anti-pattern examples: {anti_count} ({anti_percent:.1f}%)")

    # Shuffle for training
    import random
    random.seed(42)
    random.shuffle(all_examples)

    # Split 90/10
    split_idx = int(len(all_examples) * 0.9)
    train = all_examples[:split_idx]
    eval_set = all_examples[split_idx:]

    # Save v5 files
    train_path = PROJECT_ROOT / "data/openai_finetune/train_v5.jsonl"
    eval_path = PROJECT_ROOT / "data/openai_finetune/eval_v5.jsonl"

    with open(train_path, 'w') as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    with open(eval_path, 'w') as f:
        for ex in eval_set:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    print(f"\nV5 Training: {len(train)} examples -> {train_path}")
    print(f"V5 Eval: {len(eval_set)} examples -> {eval_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Base (cleaned):      {len(base_examples)} examples")
    print(f"Original anti:       {len(original_anti)} examples")
    print(f"Contrastive pairs:   {len(contrastive)} examples")
    print(f"Total V5:            {total} examples")
    print(f"Anti-pattern ratio:  {anti_percent:.1f}% (target: 8-10%)")


if __name__ == "__main__":
    main()
