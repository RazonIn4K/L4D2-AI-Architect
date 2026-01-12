#!/usr/bin/env python3
"""
Dataset Preparation Script

Converts raw scraped data into JSONL format suitable for fine-tuning with Unsloth.
Supports both SourcePawn and VScript/Squirrel code generation tasks.

Usage:
    python prepare_dataset.py --input data/raw --output data/processed
"""

import sys
import json
import re
import random
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

# Add parent to path for security utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_path, safe_write_json, safe_write_jsonl, safe_read_text
from utils.game_detection import (
    detect_game, is_l4d2_code, filter_l4d2_training_data,
    GameType, ConfidenceLevel, GameDetectionResult
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ANTI_PATTERNS_DIR = PROJECT_ROOT / "data" / "anti_patterns"

# System prompts for different code types
SYSTEM_PROMPTS = {
    "sourcepawn": """You are an expert SourcePawn developer specializing in SourceMod plugins for Left 4 Dead 2. You write clean, efficient, and well-documented code following SourceMod best practices. You understand the Source Engine, game events, entity manipulation, and the SourceMod API.""",
    
    "vscript": """You are an expert L4D2 VScript developer. You write Squirrel scripts that manipulate the AI Director, spawn infected, control game flow, and create custom mutations. You understand DirectorOptions, game hooks, entity scripting, and the Expanded Mutation System.""",
    
    "gamedata": """You are an expert at writing SourceMod gamedata files for Left 4 Dead 2. You understand memory signatures, offsets, virtual function tables, and how to find and document game functions for use in plugins.""",
}

# Code quality patterns to filter
QUALITY_PATTERNS = {
    "good": [
        r"public\s+\w+\s+\w+\s*\(",  # Function definitions
        r"#include\s*<",             # Includes
        r"//.*\n",                   # Comments
        r"/\*[\s\S]*?\*/",           # Block comments
        r"if\s*\(",                  # Control flow
        r"for\s*\(",
        r"while\s*\(",
    ],
    "bad": [
        r"TODO",                     # Incomplete code
        r"FIXME",
        r"XXX",
        r"HACK",
        r"password",                 # Sensitive info
        r"api[_-]?key",
    ]
}


@dataclass
class TrainingExample:
    """A single training example."""
    instruction: str
    response: str
    language: str
    source: str
    quality_score: float
    detected_game: str = "unknown"
    game_confidence: str = "none"
    is_l4d2_compatible: bool = True


def load_github_data(raw_dir: Path) -> List[Dict]:
    """Load GitHub scraped data with path validation and symlink protection."""
    github_path = raw_dir / "github_plugins" / "github_plugins.jsonl"
    if not github_path.exists():
        logger.warning(f"GitHub data not found at {github_path}")
        return []

    # Validate path stays within PROJECT_ROOT and check for symlinks
    try:
        validated_path = safe_path(str(github_path), PROJECT_ROOT)
        # Reject symlinks to prevent symlink attacks
        if validated_path.is_symlink():
            logger.error(f"Security: Refusing to read symlink at {validated_path}")
            return []
    except ValueError as e:
        logger.error(f"Security: Path validation failed: {e}")
        return []

    data = []
    # Use safe_read_text to read the entire file content securely
    try:
        content = safe_read_text(str(validated_path), PROJECT_ROOT)
        for line in content.splitlines():
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Failed to load GitHub data: {e}")
        return []

    logger.info(f"Loaded {len(data)} GitHub code files")
    return data


def load_wiki_data(raw_dir: Path) -> List[Dict]:
    """Load Valve Wiki scraped data with path validation and symlink protection."""
    wiki_path = raw_dir / "valve_wiki" / "valve_wiki.jsonl"
    if not wiki_path.exists():
        logger.warning(f"Wiki data not found at {wiki_path}")
        return []

    # Validate path stays within PROJECT_ROOT and check for symlinks
    try:
        validated_path = safe_path(str(wiki_path), PROJECT_ROOT)
        # Reject symlinks to prevent symlink attacks
        if validated_path.is_symlink():
            logger.error(f"Security: Refusing to read symlink at {validated_path}")
            return []
    except ValueError as e:
        logger.error(f"Security: Path validation failed: {e}")
        return []

    data = []
    # Use safe_read_text to read the entire file content securely
    try:
        content = safe_read_text(str(validated_path), PROJECT_ROOT)
        for line in content.splitlines():
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Failed to load Wiki data: {e}")
        return []

    logger.info(f"Loaded {len(data)} Wiki pages")
    return data


def load_anti_patterns(anti_patterns_dir: Path) -> List[Dict]:
    """Load anti-pattern training examples with path validation and symlink protection."""
    anti_patterns = []

    # Look for all JSONL files in anti_patterns directory
    if not anti_patterns_dir.exists():
        logger.info(f"Anti-patterns directory not found at {anti_patterns_dir}")
        return []

    # Validate the directory path first
    try:
        validated_dir = safe_path(str(anti_patterns_dir), PROJECT_ROOT)
        # Reject symlinks to prevent symlink attacks
        if validated_dir.is_symlink():
            logger.error(f"Security: Refusing to read from symlinked directory at {validated_dir}")
            return []
    except ValueError as e:
        logger.error(f"Security: Path validation failed for anti-patterns directory: {e}")
        return []

    for jsonl_file in validated_dir.glob("*.jsonl"):
        # Validate each file path and check for symlinks
        try:
            validated_file = safe_path(str(jsonl_file), PROJECT_ROOT)
            # Reject symlinks to prevent symlink attacks
            if validated_file.is_symlink():
                logger.warning(f"Security: Skipping symlink at {validated_file}")
                continue
        except ValueError as e:
            logger.warning(f"Security: Skipping file with invalid path: {e}")
            continue

        # Use safe_read_text to read the file securely
        try:
            content = safe_read_text(str(validated_file), PROJECT_ROOT)
            for line in content.splitlines():
                if line.strip():
                    try:
                        example = json.loads(line)
                        # Anti-patterns are already in ChatML format
                        anti_patterns.append(example)
                    except json.JSONDecodeError:
                        continue
        except (ValueError, FileNotFoundError) as e:
            logger.warning(f"Failed to load anti-pattern file {jsonl_file}: {e}")
            continue

    logger.info(f"Loaded {len(anti_patterns)} anti-pattern examples")
    return anti_patterns


def detect_game_for_code(code: str, language: str) -> Tuple[str, str, bool]:
    """
    Detect which game the code is for using the game detection module.

    Returns:
        Tuple of (detected_game, confidence, is_l4d2_compatible)
    """
    if language != "sourcepawn":
        # VScript is typically L4D2-specific
        return "l4d2", "high", True

    result = detect_game(code)
    return (
        result.detected_game.value,
        result.confidence.value,
        result.is_l4d2_compatible
    )


def extract_functions(code: str, language: str) -> List[Tuple[str, str]]:
    """Extract functions with their docstrings/comments."""
    functions = []
    
    if language == "sourcepawn":
        # Match SourcePawn function definitions
        pattern = r'(/\*[\s\S]*?\*/\s*|//[^\n]*\n\s*)*(?:public|stock|static|native)?\s*(\w+)\s+(\w+)\s*\(([^)]*)\)\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}'
        
        for match in re.finditer(pattern, code):
            comment = match.group(1) or ""
            return_type = match.group(2)
            func_name = match.group(3)
            params = match.group(4)
            body = match.group(5)
            
            full_func = f"{return_type} {func_name}({params})\n{{\n{body}\n}}"
            functions.append((comment.strip(), full_func))
    
    elif language == "squirrel":
        # Match Squirrel function definitions
        pattern = r'(/\*[\s\S]*?\*/\s*|//[^\n]*\n\s*)*function\s+(\w+)\s*\(([^)]*)\)\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}'
        
        for match in re.finditer(pattern, code):
            comment = match.group(1) or ""
            func_name = match.group(2)
            params = match.group(3)
            body = match.group(4)
            
            full_func = f"function {func_name}({params})\n{{\n{body}\n}}"
            functions.append((comment.strip(), full_func))
    
    return functions


def generate_instruction_from_code(code: str, language: str) -> Optional[str]:
    """Generate a natural language instruction from code."""
    instructions = []
    
    # Extract key elements from the code
    if language == "sourcepawn":
        # Look for function names
        func_match = re.search(r'(?:public|stock)\s+\w+\s+(\w+)\s*\(', code)
        if func_match:
            func_name = func_match.group(1)
            # Convert CamelCase/snake_case to natural language
            words = re.sub(r'([A-Z])', r' \1', func_name)
            words = words.replace('_', ' ').lower().strip()
            instructions.append(f"Write a SourcePawn function that {words}")
        
        # Look for event hooks
        event_match = re.search(r'HookEvent\s*\(\s*"([^"]+)"', code)
        if event_match:
            event_name = event_match.group(1)
            instructions.append(f"Create a SourceMod plugin that hooks the '{event_name}' event")
        
        # Look for ConVars
        cvar_match = re.search(r'CreateConVar\s*\(\s*"([^"]+)"', code)
        if cvar_match:
            cvar_name = cvar_match.group(1)
            instructions.append(f"Write a plugin with a ConVar named '{cvar_name}'")
    
    elif language == "squirrel":
        # Look for DirectorOptions
        if "DirectorOptions" in code:
            instructions.append("Create a DirectorOptions table for L4D2")
        
        # Look for spawning code
        if re.search(r'SpawnZombie|ZSpawn', code):
            instructions.append("Write VScript code to spawn infected")
        
        # Look for entity manipulation
        if re.search(r'Entities\.|GetEntity', code):
            instructions.append("Write VScript code to manipulate entities")
    
    if instructions:
        return random.choice(instructions)
    return None


def calculate_quality_score(code: str) -> float:
    """Calculate a quality score for code (0.0 to 1.0)."""
    score = 0.5  # Start neutral
    
    # Positive signals
    for pattern in QUALITY_PATTERNS["good"]:
        if re.search(pattern, code):
            score += 0.1
    
    # Negative signals
    for pattern in QUALITY_PATTERNS["bad"]:
        if re.search(pattern, code, re.IGNORECASE):
            score -= 0.2
    
    # Length bonus (prefer medium-length code)
    lines = code.count('\n')
    if 10 < lines < 100:
        score += 0.1
    elif lines > 200:
        score -= 0.1
    
    # Has comments bonus
    comment_ratio = len(re.findall(r'//|/\*', code)) / max(lines, 1)
    if 0.05 < comment_ratio < 0.3:
        score += 0.1
    
    return max(0.0, min(1.0, score))


def create_training_examples_from_github(data: List[Dict], filter_l4d2: bool = True) -> List[TrainingExample]:
    """Create training examples from GitHub code files.

    Args:
        data: List of code file dictionaries
        filter_l4d2: If True, filter out non-L4D2 code (default: True)

    Returns:
        List of TrainingExample objects
    """
    examples = []
    filtered_count = 0

    for item in data:
        code = item.get("content", "")
        language = item.get("language", "sourcepawn")

        # Skip if too short or too long
        if len(code) < 100 or len(code) > 50000:
            continue

        # Calculate quality
        quality = calculate_quality_score(code)
        if quality < 0.3:
            continue

        # Detect game for the code
        detected_game, confidence, is_l4d2_compatible = detect_game_for_code(code, language)

        # Filter non-L4D2 code if requested
        if filter_l4d2 and not is_l4d2_compatible:
            filtered_count += 1
            continue

        # Extract functions and create examples
        functions = extract_functions(code, language)

        for comment, func_code in functions:
            # Skip tiny functions
            if len(func_code) < 50:
                continue

            # Re-detect game for this specific function
            func_game, func_conf, func_l4d2 = detect_game_for_code(func_code, language)
            if filter_l4d2 and not func_l4d2:
                continue

            # Generate instruction
            instruction = generate_instruction_from_code(func_code, language)
            if not instruction:
                # Use comment as instruction if available
                if comment and len(comment) > 20:
                    instruction = f"Implement: {comment[:200]}"
                else:
                    continue

            example = TrainingExample(
                instruction=instruction,
                response=func_code,
                language=language,
                source=item.get("repo_name", "unknown"),
                quality_score=quality,
                detected_game=func_game,
                game_confidence=func_conf,
                is_l4d2_compatible=func_l4d2,
            )
            examples.append(example)

        # Also create whole-file examples for smaller files
        if len(code) < 5000 and quality > 0.5:
            instruction = generate_instruction_from_code(code, language)
            if instruction:
                example = TrainingExample(
                    instruction=instruction,
                    response=code,
                    language=language,
                    source=item.get("repo_name", "unknown"),
                    quality_score=quality,
                    detected_game=detected_game,
                    game_confidence=confidence,
                    is_l4d2_compatible=is_l4d2_compatible,
                )
                examples.append(example)

    if filtered_count > 0:
        logger.info(f"Filtered {filtered_count} non-L4D2 code files")

    return examples


def create_training_examples_from_wiki(data: List[Dict]) -> List[TrainingExample]:
    """Create training examples from Wiki pages."""
    examples = []
    
    for page in data:
        code_blocks = page.get("code_blocks", [])
        title = page.get("title", "Unknown")
        
        for code in code_blocks:
            # Skip non-code content
            if len(code) < 50:
                continue
            
            # Detect language
            language = "squirrel" if "DirectorOptions" in code or "function " in code else "sourcepawn"
            
            # Quality check
            quality = calculate_quality_score(code)
            if quality < 0.3:
                continue
            
            # Generate instruction based on page title and code
            if "Director" in title:
                instruction = f"Write VScript code for: {title}"
            elif "Example" in title:
                instruction = "Implement the following L4D2 script example"
            else:
                instruction = generate_instruction_from_code(code, language)
                if not instruction:
                    instruction = f"Write L4D2 code related to: {title}"
            
            example = TrainingExample(
                instruction=instruction,
                response=code,
                language=language,
                source=f"wiki:{page.get('url', '')}",
                quality_score=quality,
            )
            examples.append(example)
    
    return examples


def format_for_unsloth(examples: List[TrainingExample], system_prompt_type: str) -> List[Dict]:
    """Format examples for Unsloth/ChatML format."""
    formatted = []
    system_prompt = SYSTEM_PROMPTS.get(system_prompt_type, SYSTEM_PROMPTS["sourcepawn"])
    
    for ex in examples:
        entry = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ex.instruction},
                {"role": "assistant", "content": ex.response},
            ]
        }
        formatted.append(entry)
    
    return formatted


def format_for_alpaca(examples: List[TrainingExample]) -> List[Dict]:
    """Format examples for Alpaca format."""
    formatted = []
    
    for ex in examples:
        entry = {
            "instruction": ex.instruction,
            "input": "",  # We don't use separate input
            "output": ex.response,
        }
        formatted.append(entry)
    
    return formatted


def split_dataset(data: List[Dict], train_ratio: float = 0.9) -> Tuple[List[Dict], List[Dict]]:
    """Split dataset into train and validation sets."""
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]


def save_dataset(data: List[Dict], output_path: Path, base_dir: Path) -> None:
    """Save dataset to JSONL file with path traversal protection."""
    # Use safe_write_jsonl which combines path validation and file writing
    safe_output = safe_write_jsonl(str(output_path), data, base_dir)
    logger.info(f"Saved {len(data)} examples to {safe_output}")


def main():
    parser = argparse.ArgumentParser(description="Prepare training dataset")
    parser.add_argument("--input", type=str, default=str(RAW_DIR),
                        help="Raw data directory")
    parser.add_argument("--output", type=str, default=str(PROCESSED_DIR),
                        help="Output directory")
    parser.add_argument("--format", type=str, default="unsloth",
                        choices=["unsloth", "alpaca", "both"],
                        help="Output format")
    parser.add_argument("--min-quality", type=float, default=0.4,
                        help="Minimum quality score (0.0-1.0)")
    parser.add_argument("--train-ratio", type=float, default=0.9,
                        help="Training set ratio")
    parser.add_argument("--filter-l4d2", action="store_true", default=True,
                        help="Filter to only include L4D2-compatible code (default: True)")
    parser.add_argument("--no-filter-l4d2", action="store_false", dest="filter_l4d2",
                        help="Disable L4D2 filtering (include all SourceMod code)")
    parser.add_argument("--include-anti-patterns", action="store_true", default=True,
                        help="Include anti-pattern examples in training (default: True)")
    parser.add_argument("--no-anti-patterns", action="store_false", dest="include_anti_patterns",
                        help="Exclude anti-pattern examples from training")
    parser.add_argument("--anti-patterns-dir", type=str, default=str(ANTI_PATTERNS_DIR),
                        help="Directory containing anti-pattern JSONL files")

    args = parser.parse_args()

    # Validate paths to prevent path traversal
    project_root = Path(__file__).parent.parent.parent
    raw_dir = safe_path(args.input, project_root)
    output_dir = safe_path(args.output, project_root, create_parents=True)
    anti_patterns_dir = safe_path(args.anti_patterns_dir, project_root)

    # Log configuration
    logger.info("=" * 50)
    logger.info("DATASET PREPARATION CONFIGURATION")
    logger.info("=" * 50)
    logger.info(f"L4D2 filtering: {'ENABLED' if args.filter_l4d2 else 'DISABLED'}")
    logger.info(f"Anti-patterns: {'INCLUDED' if args.include_anti_patterns else 'EXCLUDED'}")
    logger.info(f"Minimum quality: {args.min_quality}")
    logger.info("=" * 50)

    # Load raw data
    logger.info("Loading raw data...")
    github_data = load_github_data(raw_dir)
    wiki_data = load_wiki_data(raw_dir)

    if not github_data and not wiki_data:
        logger.error("No data found! Run scrapers first.")
        sys.exit(1)

    # Create training examples with game filtering
    logger.info("Creating training examples...")
    logger.info(f"Game filtering: {'enabled' if args.filter_l4d2 else 'disabled'}")
    examples = []
    examples.extend(create_training_examples_from_github(github_data, filter_l4d2=args.filter_l4d2))
    examples.extend(create_training_examples_from_wiki(wiki_data))
    
    # Filter by quality
    examples = [ex for ex in examples if ex.quality_score >= args.min_quality]
    logger.info(f"Created {len(examples)} examples after quality filtering")

    # Calculate game detection statistics
    game_stats = defaultdict(int)
    for ex in examples:
        game_stats[ex.detected_game] += 1
    logger.info("Game detection breakdown:")
    for game, count in sorted(game_stats.items(), key=lambda x: -x[1]):
        logger.info(f"  {game}: {count} examples")

    # Separate by language
    sourcepawn_examples = [ex for ex in examples if ex.language == "sourcepawn"]
    vscript_examples = [ex for ex in examples if ex.language == "squirrel"]

    logger.info(f"SourcePawn examples: {len(sourcepawn_examples)}")
    logger.info(f"VScript examples: {len(vscript_examples)}")

    # Load anti-pattern examples if requested
    anti_patterns = []
    if args.include_anti_patterns:
        anti_patterns = load_anti_patterns(anti_patterns_dir)

    # Format and save
    if args.format in ["unsloth", "both"]:
        # SourcePawn dataset
        if sourcepawn_examples:
            sp_formatted = format_for_unsloth(sourcepawn_examples, "sourcepawn")
            # Add anti-patterns to training set only
            if anti_patterns:
                sp_formatted.extend(anti_patterns)
                logger.info(f"Added {len(anti_patterns)} anti-patterns to SourcePawn training data")
            sp_train, sp_val = split_dataset(sp_formatted, args.train_ratio)
            save_dataset(sp_train, output_dir / "sourcepawn_train.jsonl", project_root)
            save_dataset(sp_val, output_dir / "sourcepawn_val.jsonl", project_root)

        # VScript dataset
        if vscript_examples:
            vs_formatted = format_for_unsloth(vscript_examples, "vscript")
            vs_train, vs_val = split_dataset(vs_formatted, args.train_ratio)
            save_dataset(vs_train, output_dir / "vscript_train.jsonl", project_root)
            save_dataset(vs_val, output_dir / "vscript_val.jsonl", project_root)

        # Combined dataset
        all_formatted = format_for_unsloth(examples, "sourcepawn")
        if anti_patterns:
            all_formatted.extend(anti_patterns)
        all_train, all_val = split_dataset(all_formatted, args.train_ratio)
        save_dataset(all_train, output_dir / "combined_train.jsonl", project_root)
        save_dataset(all_val, output_dir / "combined_val.jsonl", project_root)

    if args.format in ["alpaca", "both"]:
        alpaca_formatted = format_for_alpaca(examples)
        alpaca_train, alpaca_val = split_dataset(alpaca_formatted, args.train_ratio)
        save_dataset(alpaca_train, output_dir / "alpaca_train.jsonl", project_root)
        save_dataset(alpaca_val, output_dir / "alpaca_val.jsonl", project_root)

    # Save statistics
    stats = {
        "total_examples": len(examples),
        "sourcepawn_examples": len(sourcepawn_examples),
        "vscript_examples": len(vscript_examples),
        "anti_pattern_examples": len(anti_patterns),
        "l4d2_filtering_enabled": args.filter_l4d2,
        "avg_quality": sum(ex.quality_score for ex in examples) / len(examples) if examples else 0,
        "game_detection": dict(game_stats),
        "sources": dict(defaultdict(int)),
    }

    for ex in examples:
        source_type = ex.source.split(":")[0] if ":" in ex.source else "github"
        stats["sources"][source_type] = stats["sources"].get(source_type, 0) + 1

    # Save statistics using safe_write_json
    safe_write_json(
        str(output_dir / "dataset_stats.json"),
        stats,
        project_root
    )
    
    logger.info("=" * 50)
    logger.info("DATASET PREPARATION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Total examples: {len(examples)}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
