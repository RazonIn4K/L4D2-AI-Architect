#!/usr/bin/env python3
"""
Quick Dataset Augmentation Script

Creates variations of training data without external APIs by:
1. Paraphrasing user prompts using rule-based transformations
2. Adding/removing comments from code
3. Consistently renaming variables

Usage:
    python scripts/training/quick_augment.py --input data/processed/l4d2_train_v15.jsonl --output data/processed/l4d2_train_v15_augmented.jsonl
    python scripts/training/quick_augment.py --input data/processed/l4d2_train_v15.jsonl --multiplier 2
"""

import sys
import json
import re
import random
import argparse
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass

# Add parent to path for security utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_path, safe_write_jsonl, safe_read_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent


# ==============================================================================
# Prompt Paraphrasing Rules
# ==============================================================================

# Prefix variations for prompts
PROMPT_PREFIXES = [
    "Write",
    "Create",
    "Implement",
    "Build",
    "Develop",
    "Code",
    "Make",
    "Generate",
    "Design",
    "Construct",
]

# Verb variations for code tasks
VERB_REPLACEMENTS = {
    "write": ["create", "implement", "develop", "build", "code", "make"],
    "create": ["write", "implement", "develop", "build", "make", "generate"],
    "implement": ["write", "create", "develop", "build", "code"],
    "build": ["create", "implement", "develop", "write", "construct"],
    "make": ["create", "build", "develop", "write", "generate"],
    "add": ["include", "insert", "incorporate", "put in"],
    "remove": ["delete", "eliminate", "take out", "get rid of"],
    "fix": ["repair", "correct", "resolve", "patch", "debug"],
    "update": ["modify", "change", "revise", "alter"],
    "get": ["retrieve", "fetch", "obtain", "acquire"],
    "set": ["assign", "configure", "define", "establish"],
    "check": ["verify", "validate", "test", "examine"],
    "handle": ["manage", "process", "deal with", "take care of"],
}

# Phrase variations
PHRASE_REPLACEMENTS = {
    "a function": ["a method", "a routine", "code", "a procedure"],
    "a plugin": ["a SourceMod plugin", "an SM plugin", "a mod", "a script"],
    "for l4d2": ["for Left 4 Dead 2", "for L4D2", "for left4dead2"],
    "that hooks": ["that listens to", "that responds to", "to hook"],
    "that spawns": ["that creates", "that generates", "to spawn"],
    "that kills": ["that eliminates", "that removes", "to kill"],
    "that heals": ["that restores health to", "that cures", "to heal"],
    "all survivors": ["every survivor", "all players", "the survivors", "each survivor"],
    "all infected": ["every infected", "all zombies", "the infected", "all special infected"],
    "the player": ["a player", "players", "the client"],
    "when a": ["when the", "whenever a", "if a", "once a"],
}

# Structural variations
STRUCTURAL_PATTERNS = [
    # "Write X that Y" -> "Y using X"
    (r"^(Write|Create|Make|Build) (a \w+) that (.+)$", r"\3 using \2"),
    # "Write X for Y" -> "For Y, write X"
    (r"^(Write|Create|Make) (.+) for (.+)$", r"For \3, \1.lower() \2"),
    # "X plugin" -> "plugin for X"
    (r"^(.+) plugin$", r"plugin for \1"),
]


def paraphrase_prompt(prompt: str) -> List[str]:
    """
    Generate paraphrased versions of a prompt using rule-based transformations.

    Args:
        prompt: Original user prompt

    Returns:
        List of paraphrased prompts (may include original)
    """
    variations = [prompt]  # Always include original
    prompt_lower = prompt.lower()

    # Try verb replacements
    for verb, replacements in VERB_REPLACEMENTS.items():
        pattern = rf'\b{verb}\b'
        if re.search(pattern, prompt_lower):
            for replacement in random.sample(replacements, min(2, len(replacements))):
                # Case-preserving replacement
                def replace_preserving_case(match):
                    original = match.group(0)
                    if original.isupper():
                        return replacement.upper()
                    elif original[0].isupper():
                        return replacement.capitalize()
                    return replacement

                new_prompt = re.sub(pattern, replace_preserving_case, prompt, flags=re.IGNORECASE)
                if new_prompt != prompt:
                    variations.append(new_prompt)

    # Try phrase replacements
    for phrase, replacements in PHRASE_REPLACEMENTS.items():
        if phrase in prompt_lower:
            replacement = random.choice(replacements)
            # Case-insensitive replacement
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            new_prompt = pattern.sub(replacement, prompt)
            if new_prompt != prompt:
                variations.append(new_prompt)

    # Try prefix variations (for prompts starting with action words)
    for prefix in PROMPT_PREFIXES:
        if prompt.lower().startswith(prefix.lower() + " "):
            rest = prompt[len(prefix) + 1:]
            new_prefix = random.choice([p for p in PROMPT_PREFIXES if p.lower() != prefix.lower()])
            new_prompt = f"{new_prefix} {rest}"
            variations.append(new_prompt)
            break

    # Add "Please" or "Can you" variations occasionally
    if random.random() < 0.3 and not prompt.lower().startswith(("please", "can you")):
        variations.append(f"Please {prompt[0].lower()}{prompt[1:]}")

    if random.random() < 0.2 and not prompt.lower().startswith(("please", "can you")):
        # Remove "Implement:" prefix if present
        clean_prompt = re.sub(r'^Implement:\s*', '', prompt)
        variations.append(f"Can you {clean_prompt[0].lower()}{clean_prompt[1:]}?")

    # Deduplicate while preserving order
    seen = set()
    unique_variations = []
    for v in variations:
        v_normalized = v.strip()
        if v_normalized and v_normalized not in seen:
            seen.add(v_normalized)
            unique_variations.append(v_normalized)

    return unique_variations


# ==============================================================================
# Code Comment Manipulation
# ==============================================================================

# Common code comment patterns
COMMENT_TEMPLATES = [
    "// {description}",
    "/* {description} */",
    "// TODO: {description}",
    "// NOTE: {description}",
]

# Descriptions to add as comments
CODE_DESCRIPTIONS = {
    "function": [
        "Main function entry point",
        "Process the request",
        "Handle the operation",
        "Execute the logic",
    ],
    "loop": [
        "Iterate through all items",
        "Process each element",
        "Loop through the collection",
    ],
    "condition": [
        "Check the condition",
        "Validate the state",
        "Verify before proceeding",
    ],
    "return": [
        "Return the result",
        "Send back the response",
        "Complete and return",
    ],
}


def add_comments_to_code(code: str) -> str:
    """
    Add descriptive comments to code.

    Args:
        code: Original code string

    Returns:
        Code with added comments
    """
    lines = code.split('\n')
    new_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        indent_str = line[:indent] if indent > 0 else ""

        # Add comment before function definitions (30% chance)
        if re.match(r'^(public|stock|static|void|int|float|bool|Action|Handle)\s+\w+\s*\(', stripped):
            if random.random() < 0.3:
                comment = random.choice(CODE_DESCRIPTIONS["function"])
                new_lines.append(f"{indent_str}// {comment}")

        # Add comment before loops (20% chance)
        elif re.match(r'^(for|while)\s*\(', stripped):
            if random.random() < 0.2:
                comment = random.choice(CODE_DESCRIPTIONS["loop"])
                new_lines.append(f"{indent_str}// {comment}")

        # Add comment before conditionals (15% chance)
        elif re.match(r'^if\s*\(', stripped):
            if random.random() < 0.15:
                comment = random.choice(CODE_DESCRIPTIONS["condition"])
                new_lines.append(f"{indent_str}// {comment}")

        new_lines.append(line)

    return '\n'.join(new_lines)


def remove_comments_from_code(code: str) -> str:
    """
    Remove comments from code.

    Args:
        code: Original code string

    Returns:
        Code with comments removed
    """
    # Remove single-line comments (but keep URLs with //)
    code = re.sub(r'(?<!:)//(?!/)[^\n]*', '', code)

    # Remove multi-line comments
    code = re.sub(r'/\*[\s\S]*?\*/', '', code)

    # Clean up excessive blank lines
    code = re.sub(r'\n\s*\n\s*\n', '\n\n', code)

    return code.strip()


def toggle_comments(code: str) -> str:
    """
    Either add or remove comments from code based on current state.

    Args:
        code: Original code string

    Returns:
        Modified code
    """
    # Count existing comments
    comment_count = len(re.findall(r'//|/\*', code))
    line_count = code.count('\n') + 1
    comment_ratio = comment_count / max(line_count, 1)

    # If code has many comments, remove some
    if comment_ratio > 0.15:
        return remove_comments_from_code(code)
    # If code has few comments, add some
    else:
        return add_comments_to_code(code)


# ==============================================================================
# Variable Renaming
# ==============================================================================

# Common SourcePawn variable naming patterns
VARIABLE_PREFIXES = {
    "g_": ["g_cv", "g_h", "g_b", "g_i", "g_f", "g_s"],  # Globals
    "cv": ["cvar", "convar", "cv_"],
    "h": ["handle", "hndl", "h_"],
    "i": ["n", "num", "count", "idx"],
    "f": ["fl", "float", "val"],
    "b": ["is", "has", "can", "should"],
    "s": ["str", "sz", "name", "text"],
}

# Common variable name replacements
VARIABLE_REPLACEMENTS = {
    # Index variables
    "i": ["j", "k", "idx", "index", "n"],
    "j": ["i", "k", "idx", "n"],
    "k": ["i", "j", "idx", "n"],
    "idx": ["index", "i", "n"],
    "index": ["idx", "i", "n"],

    # Loop counters
    "count": ["num", "total", "cnt"],
    "num": ["count", "total", "amount"],
    "total": ["count", "num", "sum"],

    # Entity references
    "client": ["player", "userid", "ply"],
    "player": ["client", "ply", "user"],
    "entity": ["ent", "target", "obj"],
    "ent": ["entity", "target", "obj"],
    "target": ["entity", "ent", "victim"],
    "victim": ["target", "attacked", "hit"],
    "attacker": ["killer", "source", "aggressor"],

    # Data
    "data": ["info", "value", "param"],
    "info": ["data", "details", "params"],
    "value": ["val", "data", "result"],
    "result": ["ret", "output", "value"],
    "buffer": ["buf", "temp", "str"],
    "temp": ["tmp", "buffer", "hold"],

    # Time
    "time": ["timestamp", "tick", "moment"],
    "delay": ["wait", "interval", "timeout"],
    "interval": ["period", "delay", "frequency"],

    # Position
    "pos": ["position", "loc", "point"],
    "position": ["pos", "loc", "coords"],
    "origin": ["pos", "start", "source"],
    "vec": ["vector", "vPos", "point"],
}


def rename_variables(code: str) -> str:
    """
    Consistently rename variables in code.

    Args:
        code: Original code string

    Returns:
        Code with renamed variables
    """
    # Find all variable declarations and usages
    # This is a simplified approach - real parsing would be more complex

    # Build replacement map
    replacements = {}

    # Find local variable declarations
    var_pattern = r'\b(int|float|bool|char|Handle|Action|Menu|Panel|ConVar|StringMap|ArrayList)\s+(\w+)'

    for match in re.finditer(var_pattern, code):
        var_name = match.group(2)

        # Skip globals (g_) and constants (all caps)
        if var_name.startswith('g_') or var_name.isupper():
            continue

        # Skip if already in replacements
        if var_name in replacements:
            continue

        # Try to find a replacement
        var_lower = var_name.lower()
        for original, alternatives in VARIABLE_REPLACEMENTS.items():
            if var_lower == original or var_lower.endswith(original):
                # Pick a random alternative
                new_suffix = random.choice(alternatives)
                if var_lower == original:
                    new_name = new_suffix
                else:
                    prefix = var_name[:-len(original)]
                    new_name = prefix + new_suffix

                # Preserve original case style
                if var_name[0].isupper():
                    new_name = new_name[0].upper() + new_name[1:]

                replacements[var_name] = new_name
                break

    # Apply replacements (only if we have some)
    if not replacements:
        return code

    # Sort by length (longest first) to avoid partial replacements
    sorted_replacements = sorted(replacements.items(), key=lambda x: -len(x[0]))

    new_code = code
    for old_name, new_name in sorted_replacements:
        # Use word boundaries to avoid partial matches
        pattern = rf'\b{re.escape(old_name)}\b'
        new_code = re.sub(pattern, new_name, new_code)

    return new_code


# ==============================================================================
# Main Augmentation Logic
# ==============================================================================

@dataclass
class AugmentationStats:
    """Track augmentation statistics."""
    original_count: int = 0
    prompt_variations: int = 0
    comment_variations: int = 0
    variable_variations: int = 0
    total_generated: int = 0
    duplicates_removed: int = 0


def augment_example(example: Dict, stats: AugmentationStats) -> List[Dict]:
    """
    Generate augmented versions of a single training example.

    Args:
        example: Original training example in ChatML format
        stats: Statistics tracker

    Returns:
        List of augmented examples (including original)
    """
    augmented = [example]  # Always include original
    stats.original_count += 1

    messages = example.get("messages", [])
    if len(messages) < 3:
        return augmented

    system_msg = messages[0]
    user_msg = messages[1]
    assistant_msg = messages[2]

    user_content = user_msg.get("content", "")
    assistant_content = assistant_msg.get("content", "")

    # 1. Generate prompt variations
    prompt_variations = paraphrase_prompt(user_content)
    for new_prompt in prompt_variations[1:]:  # Skip first (original)
        new_example = {
            "messages": [
                system_msg.copy(),
                {"role": "user", "content": new_prompt},
                assistant_msg.copy(),
            ]
        }
        augmented.append(new_example)
        stats.prompt_variations += 1

    # 2. Generate comment variations (only for code responses)
    if len(assistant_content) > 100:  # Only for substantial code
        # Version with toggled comments
        modified_code = toggle_comments(assistant_content)
        if modified_code != assistant_content:
            new_example = {
                "messages": [
                    system_msg.copy(),
                    user_msg.copy(),
                    {"role": "assistant", "content": modified_code},
                ]
            }
            augmented.append(new_example)
            stats.comment_variations += 1

    # 3. Generate variable name variations (30% chance, only for code)
    if random.random() < 0.3 and len(assistant_content) > 100:
        modified_code = rename_variables(assistant_content)
        if modified_code != assistant_content:
            new_example = {
                "messages": [
                    system_msg.copy(),
                    user_msg.copy(),
                    {"role": "assistant", "content": modified_code},
                ]
            }
            augmented.append(new_example)
            stats.variable_variations += 1

    return augmented


def deduplicate_examples(examples: List[Dict]) -> Tuple[List[Dict], int]:
    """
    Remove duplicate examples based on content hash.

    Args:
        examples: List of training examples

    Returns:
        Tuple of (deduplicated examples, count of removed duplicates)
    """
    seen_hashes: Set[str] = set()
    unique_examples = []
    duplicates = 0

    for example in examples:
        # Create hash of user + assistant content
        messages = example.get("messages", [])
        if len(messages) >= 3:
            content_str = (
                messages[1].get("content", "") +
                messages[2].get("content", "")
            )
            content_hash = hashlib.md5(content_str.encode()).hexdigest()

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_examples.append(example)
            else:
                duplicates += 1
        else:
            unique_examples.append(example)

    return unique_examples, duplicates


def load_jsonl(input_path: Path) -> List[Dict]:
    """
    Load examples from a JSONL file.

    Args:
        input_path: Path to input JSONL file

    Returns:
        List of training examples
    """
    content = safe_read_text(str(input_path), PROJECT_ROOT)
    examples = []

    for line_num, line in enumerate(content.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            examples.append(json.loads(line))
        except json.JSONDecodeError as e:
            logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")

    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Quick dataset augmentation for L4D2 training data"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input JSONL file path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSONL file path (default: input_augmented.jsonl)"
    )
    parser.add_argument(
        "--multiplier", "-m",
        type=float,
        default=0,
        help="Target multiplier for dataset size (0 = no limit, generate all variations)"
    )
    parser.add_argument(
        "--no-prompt-aug",
        action="store_true",
        help="Disable prompt paraphrasing"
    )
    parser.add_argument(
        "--no-comment-aug",
        action="store_true",
        help="Disable comment toggling"
    )
    parser.add_argument(
        "--no-variable-aug",
        action="store_true",
        help="Disable variable renaming"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        default=True,
        help="Remove duplicate examples (default: True)"
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_false",
        dest="dedupe",
        help="Keep duplicate examples"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Validate input path
    input_path = safe_path(args.input, PROJECT_ROOT)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = safe_path(args.output, PROJECT_ROOT, create_parents=True)
    else:
        output_path = input_path.parent / f"{input_path.stem}_augmented.jsonl"

    logger.info("=" * 60)
    logger.info("QUICK DATASET AUGMENTATION")
    logger.info("=" * 60)
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Seed:   {args.seed}")

    # Load input data
    logger.info("Loading input data...")
    examples = load_jsonl(input_path)
    logger.info(f"Loaded {len(examples)} examples")

    # Initialize statistics
    stats = AugmentationStats()

    # Process each example
    logger.info("Generating augmentations...")
    augmented_examples = []

    for i, example in enumerate(examples):
        if (i + 1) % 500 == 0:
            logger.info(f"  Processed {i + 1}/{len(examples)} examples...")

        # Generate variations
        variations = augment_example(example, stats)

        # Apply augmentation toggles
        filtered_variations = [example]  # Always keep original
        for var in variations[1:]:
            # Check if this variation type is enabled
            is_prompt_var = var["messages"][1]["content"] != example["messages"][1]["content"]
            is_code_var = var["messages"][2]["content"] != example["messages"][2]["content"]

            if is_prompt_var and args.no_prompt_aug:
                continue
            if is_code_var and (args.no_comment_aug and args.no_variable_aug):
                continue

            filtered_variations.append(var)

        augmented_examples.extend(filtered_variations)

    stats.total_generated = len(augmented_examples)

    # Apply multiplier limit if specified
    if args.multiplier > 0:
        target_size = int(len(examples) * args.multiplier)
        if len(augmented_examples) > target_size:
            # Keep all originals, sample from variations
            originals = augmented_examples[:len(examples)]
            variations = augmented_examples[len(examples):]

            # Calculate how many variations to keep
            variations_needed = target_size - len(originals)
            if variations_needed > 0 and variations:
                sampled_variations = random.sample(
                    variations,
                    min(variations_needed, len(variations))
                )
                augmented_examples = originals + sampled_variations
            else:
                augmented_examples = originals[:target_size]

            logger.info(f"Applied {args.multiplier}x multiplier: {len(augmented_examples)} examples")

    # Deduplicate
    if args.dedupe:
        augmented_examples, duplicates = deduplicate_examples(augmented_examples)
        stats.duplicates_removed = duplicates
        logger.info(f"Removed {duplicates} duplicates")

    # Shuffle the dataset
    random.shuffle(augmented_examples)

    # Save output
    logger.info(f"Saving {len(augmented_examples)} examples...")
    safe_write_jsonl(str(output_path), augmented_examples, PROJECT_ROOT)

    # Print statistics
    logger.info("=" * 60)
    logger.info("AUGMENTATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Original examples:     {stats.original_count}")
    logger.info(f"Prompt variations:     {stats.prompt_variations}")
    logger.info(f"Comment variations:    {stats.comment_variations}")
    logger.info(f"Variable variations:   {stats.variable_variations}")
    logger.info(f"Total generated:       {stats.total_generated}")
    logger.info(f"Duplicates removed:    {stats.duplicates_removed}")
    logger.info(f"Final dataset size:    {len(augmented_examples)}")
    logger.info(f"Expansion ratio:       {len(augmented_examples) / stats.original_count:.2f}x")
    logger.info(f"Output saved to:       {output_path}")


if __name__ == "__main__":
    main()
