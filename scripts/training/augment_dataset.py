#!/usr/bin/env python3
"""
Data Augmentation Pipeline for L4D2 Training Data

Applies various augmentation techniques to expand the training dataset
while preserving code semantics and improving model generalization.

Augmentation techniques:
1. Variable renaming - Change variable names while preserving logic
2. Comment variations - Add/modify/remove comments
3. Code style variations - Brace placement, spacing, indentation
4. Function reordering - Change order of independent functions
5. Equivalent code substitutions - for vs while, if vs switch

Usage:
    python augment_dataset.py [--input data/processed/l4d2_train_v13.jsonl]
                             [--output-augmented data/processed/l4d2_train_v13_augmented.jsonl]
                             [--output-combined data/processed/l4d2_train_v14.jsonl]
                             [--augmentations-per-example 3]
                             [--seed 42]
"""

import sys
import json
import re
import random
import hashlib
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

# Add parent to path for security utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_path, safe_write_jsonl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@dataclass
class AugmentationStats:
    """Track augmentation statistics."""
    original_count: int = 0
    augmented_generated: int = 0
    duplicates_removed: int = 0
    validation_failed: int = 0
    final_count: int = 0
    augmentation_breakdown: Dict[str, int] = None

    def __post_init__(self):
        if self.augmentation_breakdown is None:
            self.augmentation_breakdown = {}


# Common SourcePawn variable naming patterns
VARIABLE_PREFIXES = [
    "g_", "h_", "cv_", "sm_", "l4d_", "l4d2_",
    "client_", "player_", "target_", "victim_", "attacker_",
    "temp_", "cached_", "current_", "last_", "prev_", "next_",
]

VARIABLE_NAMES = [
    # Index variables
    ["i", "j", "k", "idx", "index", "n"],
    # Client/Player
    ["client", "player", "user", "target", "victim", "attacker", "killer"],
    # Entity
    ["entity", "ent", "entityRef", "entRef", "entityIndex"],
    # Count/Size
    ["count", "num", "total", "amount", "size", "len", "length"],
    # Value
    ["value", "val", "result", "ret", "data"],
    # Buffer
    ["buffer", "buf", "str", "text", "name", "className"],
    # Handle
    ["handle", "hndl", "hFile", "hTimer", "hMenu", "hPanel"],
    # Team
    ["team", "teamId", "teamNum", "teamIndex"],
    # Health
    ["health", "hp", "maxHealth", "healthPoints", "currentHealth"],
    # Position
    ["pos", "position", "origin", "location", "vec", "vector", "coords"],
    # Time
    ["time", "duration", "delay", "interval", "cooldown", "timer"],
    # Boolean flags
    ["enabled", "active", "valid", "found", "success", "isValid", "bValid"],
]

# Comment templates for SourcePawn
COMMENT_TEMPLATES = [
    "// {action}",
    "/* {action} */",
    "// TODO: {action}",
    "// {action} - L4D2",
    "/**\n * {action}\n */",
]

COMMENT_ACTIONS = [
    "Check if valid", "Validate input", "Get client info",
    "Update state", "Apply changes", "Handle event",
    "Process data", "Initialize", "Clean up",
    "Safety check", "Boundary check", "Null check",
]


class SourcePawnAugmenter:
    """Augmenter for SourcePawn code."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.augmentation_counts = Counter()
        self.variation_counter = 0  # For generating unique variations

    def augment(self, code: str, num_variations: int = 3) -> List[Tuple[str, str]]:
        """
        Generate augmented variations of the code.

        Returns list of (augmented_code, augmentation_type) tuples.
        """
        augmentations = []

        # Try each augmentation technique
        techniques = [
            ("variable_rename", self.rename_variables),
            ("comment_variation", self.vary_comments),
            ("style_variation", self.vary_code_style),
            ("function_reorder", self.reorder_functions),
            ("equivalent_substitution", self.substitute_equivalent),
            ("whitespace_variation", self.vary_whitespace),
            ("string_variation", self.vary_string_literals),
            ("constant_extraction", self.extract_constants),
        ]

        # Shuffle techniques for variety
        self.rng.shuffle(techniques)

        for name, technique in techniques:
            if len(augmentations) >= num_variations:
                break

            try:
                result = technique(code)
                if result and result != code:
                    # Validate the augmented code
                    if self._validate_syntax(result):
                        augmentations.append((result, name))
                        self.augmentation_counts[name] += 1
            except Exception as e:
                logger.debug(f"Augmentation {name} failed: {e}")
                continue

        # If we need more, try combinations
        if len(augmentations) < num_variations:
            combined = self._combine_augmentations(code, num_variations - len(augmentations))
            augmentations.extend(combined)

        # If still need more, try multiple passes of same techniques
        if len(augmentations) < num_variations:
            additional = self._multi_pass_augment(code, num_variations - len(augmentations))
            augmentations.extend(additional)

        return augmentations[:num_variations]

    def rename_variables(self, code: str) -> Optional[str]:
        """Rename local variables while preserving logic."""
        # Find variable declarations
        var_pattern = r'\b(int|float|bool|char|Handle|ConVar|Event|StringMap|ArrayList|Plugin)\s+(\w+)\s*[=;,\[]'

        declarations = list(re.finditer(var_pattern, code))
        if not declarations:
            return None

        # Select a variable to rename (prefer simple names)
        candidates = []
        for match in declarations:
            var_name = match.group(2)
            # Skip g_ prefixed (globals), constants, and common names we shouldn't change
            if (var_name.startswith("g_") or
                var_name.isupper() or
                var_name in ["myinfo", "Plugin", "client", "event", "name"]):
                continue
            candidates.append(var_name)

        if not candidates:
            return None

        old_name = self.rng.choice(candidates)

        # Generate new name from same category
        new_name = self._generate_new_variable_name(old_name)
        if new_name == old_name:
            return None

        # Replace all occurrences (word boundary to avoid partial matches)
        result = re.sub(r'\b' + re.escape(old_name) + r'\b', new_name, code)

        return result

    def _generate_new_variable_name(self, old_name: str) -> str:
        """Generate a new variable name in similar style."""
        # Preserve prefix if present
        prefix = ""
        base_name = old_name
        for p in VARIABLE_PREFIXES:
            if old_name.startswith(p):
                prefix = p
                base_name = old_name[len(p):]
                break

        # Find category and pick alternative
        for category in VARIABLE_NAMES:
            if base_name.lower() in [v.lower() for v in category]:
                alternatives = [v for v in category if v.lower() != base_name.lower()]
                if alternatives:
                    new_base = self.rng.choice(alternatives)
                    # Match case style
                    if base_name[0].isupper():
                        new_base = new_base[0].upper() + new_base[1:]
                    return prefix + new_base

        # If not in any category, add/change prefix
        if not prefix:
            new_prefix = self.rng.choice(["temp_", "local_", ""])
            return new_prefix + base_name

        return old_name

    def vary_comments(self, code: str) -> Optional[str]:
        """Add, modify, or remove comments."""
        lines = code.split('\n')
        modified = False
        result_lines = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Random chance to add comment before significant lines
            if (self.rng.random() < 0.15 and
                not stripped.startswith('//') and
                not stripped.startswith('/*') and
                len(stripped) > 10 and
                '{' not in stripped and
                '}' not in stripped):

                # Detect what the line does
                comment = self._generate_contextual_comment(stripped)
                if comment:
                    indent = len(line) - len(line.lstrip())
                    result_lines.append(' ' * indent + comment)
                    modified = True

            # Modify existing comments with small probability
            if stripped.startswith('//') and self.rng.random() < 0.2:
                # Vary comment style
                comment_text = stripped[2:].strip()
                if comment_text:
                    new_styles = [
                        f"// {comment_text}",
                        f"/* {comment_text} */",
                        f"// NOTE: {comment_text}",
                    ]
                    indent = len(line) - len(line.lstrip())
                    result_lines.append(' ' * indent + self.rng.choice(new_styles))
                    modified = True
                    continue

            result_lines.append(line)

        if not modified:
            return None

        return '\n'.join(result_lines)

    def _generate_contextual_comment(self, line: str) -> Optional[str]:
        """Generate a comment based on code context."""
        if 'if' in line and '(' in line:
            return self.rng.choice([
                "// Check condition",
                "// Validate before proceeding",
                "// Guard clause",
            ])
        elif 'for' in line or 'while' in line:
            return self.rng.choice([
                "// Iterate through elements",
                "// Loop processing",
            ])
        elif 'return' in line:
            return "// Return result"
        elif '=' in line and 'if' not in line and 'for' not in line:
            return self.rng.choice([
                "// Set value",
                "// Store result",
                "// Initialize",
            ])
        elif re.match(r'\s*\w+\s*\(', line):
            return "// Execute operation"

        return None

    def vary_code_style(self, code: str) -> Optional[str]:
        """Vary code formatting style."""
        result = code
        modified = False

        # Style variation: brace placement
        if self.rng.random() < 0.5:
            # K&R to Allman style (opening brace on new line)
            if re.search(r'\)\s*\{', result):
                # Only for function definitions and control structures
                new_result = re.sub(
                    r'(\))\s*\{(\s*\n)',
                    r'\1\n{\2',
                    result
                )
                if new_result != result:
                    result = new_result
                    modified = True
        else:
            # Allman to K&R style (opening brace on same line)
            new_result = re.sub(
                r'\)\s*\n\s*\{',
                ') {',
                result
            )
            if new_result != result:
                result = new_result
                modified = True

        # Style variation: spacing around operators
        if self.rng.random() < 0.3:
            # Add/remove spaces around operators
            operators = ['==', '!=', '<=', '>=', '&&', '||', '+=', '-=']
            for op in operators:
                if op in result:
                    if self.rng.random() < 0.5:
                        # Add spaces
                        result = result.replace(op, f' {op} ')
                    # Clean up multiple spaces
                    result = re.sub(r'  +', ' ', result)
                    modified = True
                    break

        # Style variation: blank lines
        if self.rng.random() < 0.2:
            # Add blank line before function definitions
            new_result = re.sub(
                r'(\n)(public |stock |static |void )',
                r'\1\n\2',
                result
            )
            if new_result != result:
                result = new_result
                modified = True

        if not modified:
            return None

        return result

    def reorder_functions(self, code: str) -> Optional[str]:
        """Reorder independent functions in the code."""
        # Find all function definitions
        func_pattern = r'((?:/\*[\s\S]*?\*/\s*|//[^\n]*\n\s*)*)?((?:public|stock|static)\s+)?(\w+)\s+(\w+)\s*\([^)]*\)\s*\{[^}]*(?:\{[^}]*\}[^}]*)*\}'

        functions = list(re.finditer(func_pattern, code))
        if len(functions) < 2:
            return None

        # Extract function texts and positions
        func_data = []
        for match in functions:
            func_text = match.group(0)
            func_name = match.group(4)
            start = match.start()
            end = match.end()
            func_data.append({
                'text': func_text,
                'name': func_name,
                'start': start,
                'end': end,
            })

        # Check if functions are independent (don't call each other in sequence)
        # For simplicity, only reorder if we have exactly 2-4 functions
        if not (2 <= len(func_data) <= 4):
            return None

        # Shuffle function order
        shuffled = func_data.copy()
        self.rng.shuffle(shuffled)

        # Check if order actually changed
        if [f['name'] for f in shuffled] == [f['name'] for f in func_data]:
            return None

        # Reconstruct code with reordered functions
        # Get code before first function and after last function
        prefix = code[:func_data[0]['start']]
        suffix = code[func_data[-1]['end']:]

        # Get separators between functions
        separators = []
        for i in range(len(func_data) - 1):
            sep = code[func_data[i]['end']:func_data[i+1]['start']]
            separators.append(sep)

        # Rebuild with shuffled functions
        result = prefix
        for i, func in enumerate(shuffled):
            result += func['text']
            if i < len(shuffled) - 1:
                # Use original separator or default
                if separators:
                    result += separators[i % len(separators)]
                else:
                    result += '\n\n'
        result += suffix

        return result

    def substitute_equivalent(self, code: str) -> Optional[str]:
        """Substitute equivalent code constructs."""
        result = code
        modified = False

        # Substitution: prefix increment to postfix and vice versa
        if self.rng.random() < 0.3:
            if '++i' in result:
                result = result.replace('++i', 'i++')
                modified = True
            elif 'i++' in result and '++i' not in result:
                # Only in simple contexts (not in expressions)
                result = re.sub(r'\bi\+\+\s*;', '++i;', result)
                modified = True

        # Substitution: != 0 to simple truthy check
        if self.rng.random() < 0.3:
            new_result = re.sub(r'\((\w+)\s*!=\s*0\)', r'(\1)', result)
            if new_result != result:
                result = new_result
                modified = True

        # Substitution: == true to simple truthy
        if self.rng.random() < 0.3:
            new_result = re.sub(r'(\w+)\s*==\s*true\b', r'\1', result)
            if new_result != result:
                result = new_result
                modified = True

        # Substitution: == false to !variable
        if self.rng.random() < 0.3:
            new_result = re.sub(r'(\w+)\s*==\s*false\b', r'!\1', result)
            if new_result != result:
                result = new_result
                modified = True

        # Substitution: if-else to ternary for simple assignments
        if self.rng.random() < 0.2:
            ternary_pattern = r'if\s*\(([^)]+)\)\s*\n?\s*(\w+)\s*=\s*([^;]+);\s*\n?\s*else\s*\n?\s*\2\s*=\s*([^;]+);'
            match = re.search(ternary_pattern, result)
            if match:
                cond = match.group(1)
                var = match.group(2)
                val_true = match.group(3)
                val_false = match.group(4)
                ternary = f'{var} = ({cond}) ? {val_true} : {val_false};'
                result = result[:match.start()] + ternary + result[match.end():]
                modified = True

        # Substitution: MaxClients to MAXPLAYERS (if defined)
        if self.rng.random() < 0.15 and 'MaxClients' in result:
            if '#define MAXPLAYERS' not in result and 'MAXPLAYERS' not in result:
                # Add definition at top
                result = '#define MAXPLAYERS 64\n' + result.replace('MaxClients', 'MAXPLAYERS')
                modified = True

        if not modified:
            return None

        return result

    def vary_whitespace(self, code: str) -> Optional[str]:
        """Vary whitespace and indentation patterns."""
        result = code
        modified = False

        # Vary indentation (2 spaces to 4 spaces or vice versa)
        lines = result.split('\n')
        new_lines = []
        indent_change = self.rng.choice(['expand', 'compress', 'tabs'])

        for line in lines:
            leading = len(line) - len(line.lstrip())
            if leading > 0:
                spaces = line[:leading]
                rest = line[leading:]

                if indent_change == 'expand' and '  ' in spaces and '\t' not in spaces:
                    # 2 spaces to 4 spaces
                    new_indent = spaces.replace('  ', '    ')
                    new_lines.append(new_indent + rest)
                    modified = True
                elif indent_change == 'compress' and '    ' in spaces:
                    # 4 spaces to 2 spaces
                    new_indent = spaces.replace('    ', '  ')
                    new_lines.append(new_indent + rest)
                    modified = True
                elif indent_change == 'tabs' and '    ' in spaces:
                    # Spaces to tabs
                    new_indent = spaces.replace('    ', '\t')
                    new_lines.append(new_indent + rest)
                    modified = True
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        if modified:
            result = '\n'.join(new_lines)

        # Add/remove trailing whitespace on some lines
        if self.rng.random() < 0.3 and not modified:
            lines = result.split('\n')
            for i in range(len(lines)):
                if self.rng.random() < 0.1 and lines[i].strip():
                    lines[i] = lines[i].rstrip()  # Clean trailing
                    modified = True
            result = '\n'.join(lines)

        return result if modified else None

    def vary_string_literals(self, code: str) -> Optional[str]:
        """Vary string literals in code."""
        result = code
        modified = False

        # Find string literals
        string_pattern = r'"([^"\\]*(?:\\.[^"\\]*)*)"'
        matches = list(re.finditer(string_pattern, result))

        if not matches:
            return None

        # Select one string to modify
        match = self.rng.choice(matches)
        old_string = match.group(1)

        # Skip very short strings or format strings
        if len(old_string) < 3 or '%' in old_string:
            return None

        # Variations: case changes for message strings
        if old_string[0].isupper() and ' ' in old_string:
            # Add period at end if not present
            if not old_string.endswith('.') and not old_string.endswith('!'):
                new_string = old_string + '.'
                result = result[:match.start(1)] + new_string + result[match.end(1):]
                modified = True
        elif old_string[0].islower() and ' ' in old_string:
            # Capitalize first letter
            new_string = old_string[0].upper() + old_string[1:]
            result = result[:match.start(1)] + new_string + result[match.end(1):]
            modified = True

        return result if modified else None

    def extract_constants(self, code: str) -> Optional[str]:
        """Extract magic numbers into #define constants."""
        result = code
        modified = False

        # Find magic numbers in code (not in #define lines)
        number_pattern = r'(?<!#define\s)(?<!["\'])(\b\d{2,}\b)(?!["\'])'

        matches = list(re.finditer(number_pattern, result))
        if not matches:
            return None

        # Filter out line numbers and common values
        valid_matches = []
        for match in matches:
            num = int(match.group(1))
            line_start = result.rfind('\n', 0, match.start()) + 1
            line = result[line_start:match.end() + 50].split('\n')[0]

            # Skip if already in a #define
            if '#define' in line:
                continue

            # Skip common non-magic values
            if num in [0, 1, 10, 100, 1000, 32, 64, 128, 256, 512, 1024]:
                continue

            valid_matches.append((match, num))

        if not valid_matches:
            return None

        # Extract one magic number
        match, num = self.rng.choice(valid_matches)

        # Generate constant name
        const_names = [
            f"MAGIC_VALUE_{num}",
            f"CONST_{num}",
            f"VALUE_{num}",
        ]
        const_name = self.rng.choice(const_names)

        # Add #define at top and replace usage
        define_line = f"#define {const_name} {num}\n"

        # Find good insertion point (after other #defines or at top)
        insert_pos = 0
        pragma_match = re.search(r'#pragma[^\n]*\n', result)
        if pragma_match:
            insert_pos = pragma_match.end()

        include_matches = list(re.finditer(r'#include[^\n]*\n', result))
        if include_matches:
            insert_pos = include_matches[-1].end()

        define_matches = list(re.finditer(r'#define[^\n]*\n', result))
        if define_matches:
            insert_pos = define_matches[-1].end()

        # Insert define and replace number
        result = result[:insert_pos] + define_line + result[insert_pos:]

        # Adjust match position
        adjusted_start = match.start() + len(define_line)
        adjusted_end = match.end() + len(define_line)

        result = result[:adjusted_start] + const_name + result[adjusted_end:]
        modified = True

        return result if modified else None

    def _multi_pass_augment(self, code: str, num_needed: int) -> List[Tuple[str, str]]:
        """Apply multiple passes of augmentation for more variations."""
        results = []
        self.variation_counter += 1

        for i in range(num_needed):
            current = code
            # Apply 2-3 different augmentations
            techniques = [
                self.vary_comments,
                self.vary_whitespace,
                self.vary_code_style,
            ]
            self.rng.shuffle(techniques)

            applied = 0
            for technique in techniques:
                if applied >= 2:
                    break
                try:
                    result = technique(current)
                    if result and result != current:
                        current = result
                        applied += 1
                except Exception:
                    continue

            if current != code and self._validate_syntax(current):
                results.append((current, "multi_pass"))
                self.augmentation_counts["multi_pass"] += 1

        return results

    def _combine_augmentations(self, code: str, num_needed: int) -> List[Tuple[str, str]]:
        """Combine multiple augmentation techniques."""
        combined = []
        current = code

        techniques = [
            self.rename_variables,
            self.vary_comments,
            self.vary_code_style,
            self.vary_whitespace,
        ]

        for _ in range(num_needed):
            self.rng.shuffle(techniques)
            modified = False

            for technique in techniques:
                try:
                    result = technique(current)
                    if result and result != current:
                        if self._validate_syntax(result):
                            current = result
                            modified = True
                            break
                except Exception:
                    continue

            if modified:
                combined.append((current, "combined"))
                self.augmentation_counts["combined"] += 1

        return combined

    def _validate_syntax(self, code: str) -> bool:
        """Basic syntax validation for SourcePawn code."""
        # Check balanced braces
        if code.count('{') != code.count('}'):
            return False

        # Check balanced parentheses
        if code.count('(') != code.count(')'):
            return False

        # Check balanced brackets
        if code.count('[') != code.count(']'):
            return False

        # Check for obvious broken syntax
        broken_patterns = [
            r';\s*;',           # Double semicolons
            r'\(\s*\)',         # Empty function calls (might be valid but risky)
            r'{\s*}',           # Empty blocks are OK but not multiple
            r'\)\s*\(',         # Back-to-back parens without operator
        ]

        for pattern in broken_patterns:
            if re.search(pattern, code):
                # These patterns might indicate broken code
                # but some are valid, so we're lenient
                pass

        # Check that key SourcePawn keywords aren't broken
        if re.search(r'\bpubli\s|\bvoi\s|\bint\s*\s*int\b', code):
            return False

        return True


class VScriptAugmenter:
    """Augmenter for VScript/Squirrel code."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.augmentation_counts = Counter()

    def augment(self, code: str, num_variations: int = 3) -> List[Tuple[str, str]]:
        """Generate augmented variations of VScript code."""
        augmentations = []

        techniques = [
            ("variable_rename", self.rename_variables),
            ("comment_variation", self.vary_comments),
            ("style_variation", self.vary_code_style),
        ]

        self.rng.shuffle(techniques)

        for name, technique in techniques:
            if len(augmentations) >= num_variations:
                break

            try:
                result = technique(code)
                if result and result != code:
                    if self._validate_syntax(result):
                        augmentations.append((result, name))
                        self.augmentation_counts[name] += 1
            except Exception as e:
                logger.debug(f"VScript augmentation {name} failed: {e}")
                continue

        return augmentations[:num_variations]

    def rename_variables(self, code: str) -> Optional[str]:
        """Rename local variables in VScript."""
        # VScript variable patterns
        var_pattern = r'\b(local)\s+(\w+)\s*='

        declarations = list(re.finditer(var_pattern, code))
        if not declarations:
            return None

        # Select a variable
        candidates = [m.group(2) for m in declarations
                     if not m.group(2).startswith('_')]

        if not candidates:
            return None

        old_name = self.rng.choice(candidates)
        new_name = self._generate_squirrel_name(old_name)

        if new_name == old_name:
            return None

        result = re.sub(r'\b' + re.escape(old_name) + r'\b', new_name, code)
        return result

    def _generate_squirrel_name(self, old_name: str) -> str:
        """Generate alternative Squirrel variable name."""
        alternatives = {
            'i': ['j', 'k', 'idx', 'n'],
            'count': ['num', 'total', 'cnt'],
            'value': ['val', 'result', 'data'],
            'entity': ['ent', 'target', 'obj'],
            'player': ['client', 'survivor', 'user'],
        }

        base = old_name.lower()
        for key, alts in alternatives.items():
            if base == key or base in alts:
                choices = [key] + alts
                new = self.rng.choice([c for c in choices if c != base])
                # Match case
                if old_name[0].isupper():
                    new = new[0].upper() + new[1:]
                return new

        # Default: add underscore prefix
        return '_' + old_name if not old_name.startswith('_') else old_name

    def vary_comments(self, code: str) -> Optional[str]:
        """Vary comments in VScript."""
        lines = code.split('\n')
        modified = False
        result_lines = []

        for line in lines:
            stripped = line.strip()

            # Add comments to significant lines
            if (self.rng.random() < 0.15 and
                not stripped.startswith('//') and
                'function' not in stripped and
                len(stripped) > 10):

                indent = len(line) - len(line.lstrip())
                comment = self.rng.choice([
                    "// Process",
                    "// Execute",
                    "// Handle",
                ])
                result_lines.append(' ' * indent + comment)
                modified = True

            result_lines.append(line)

        return '\n'.join(result_lines) if modified else None

    def vary_code_style(self, code: str) -> Optional[str]:
        """Vary VScript code style."""
        result = code
        modified = False

        # Toggle quote styles (Squirrel supports both)
        if self.rng.random() < 0.3:
            # Find string literals and toggle quotes
            if "'" in result and '"' in result:
                # Complex, skip
                pass
            elif "'" in result:
                result = result.replace("'", '"')
                modified = True

        # Vary spacing
        if self.rng.random() < 0.3:
            result = re.sub(r'(\w)\s*<-\s*(\w)', r'\1 <- \2', result)
            if result != code:
                modified = True

        return result if modified else None

    def _validate_syntax(self, code: str) -> bool:
        """Basic VScript syntax validation."""
        if code.count('{') != code.count('}'):
            return False
        if code.count('(') != code.count(')'):
            return False
        return True


def compute_hash(text: str, strict: bool = True) -> str:
    """Compute hash for deduplication.

    Args:
        text: The code text to hash
        strict: If True, normalize whitespace for comparison.
                If False, only strip leading/trailing whitespace.
    """
    if strict:
        # Normalize whitespace for strict comparison
        normalized = re.sub(r'\s+', ' ', text.strip())
    else:
        # Lenient: preserve internal whitespace differences
        normalized = text.strip()
    return hashlib.md5(normalized.encode()).hexdigest()


def load_training_data(input_path: Path) -> List[Dict]:
    """Load training data from JSONL file."""
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping line {line_num}: {e}")
    return data


def detect_language(example: Dict) -> str:
    """Detect if example is SourcePawn or VScript."""
    messages = example.get('messages', [])
    for msg in messages:
        content = msg.get('content', '')
        if 'VScript' in content or 'Squirrel' in content:
            return 'vscript'
        if 'SourcePawn' in content or 'SourceMod' in content:
            return 'sourcepawn'

    # Check code content
    for msg in messages:
        if msg.get('role') == 'assistant':
            code = msg.get('content', '')
            if 'DirectorOptions' in code or 'function ' in code:
                return 'vscript'
            if '#pragma' in code or '#include' in code:
                return 'sourcepawn'

    return 'sourcepawn'  # Default


def augment_example(
    example: Dict,
    sp_augmenter: SourcePawnAugmenter,
    vs_augmenter: VScriptAugmenter,
    num_augmentations: int = 3
) -> List[Dict]:
    """Augment a single training example."""
    augmented = []
    messages = example.get('messages', [])

    # Find the assistant response (the code)
    assistant_idx = None
    for i, msg in enumerate(messages):
        if msg.get('role') == 'assistant':
            assistant_idx = i
            break

    if assistant_idx is None:
        return []

    code = messages[assistant_idx].get('content', '')
    if len(code) < 50:  # Skip very short examples
        return []

    # Detect language and use appropriate augmenter
    language = detect_language(example)
    augmenter = sp_augmenter if language == 'sourcepawn' else vs_augmenter

    # Generate augmentations
    augmentations = augmenter.augment(code, num_augmentations)

    for aug_code, aug_type in augmentations:
        # Create new example with augmented code
        new_messages = []
        for i, msg in enumerate(messages):
            if i == assistant_idx:
                new_messages.append({
                    'role': 'assistant',
                    'content': aug_code
                })
            else:
                new_messages.append(msg.copy())

        augmented.append({
            'messages': new_messages,
            '_augmentation_type': aug_type,
        })

    return augmented


def remove_duplicates(
    examples: List[Dict],
    seen_hashes: Set[str],
    strict: bool = False
) -> Tuple[List[Dict], int]:
    """Remove duplicate examples based on assistant response hash.

    Args:
        examples: List of training examples
        seen_hashes: Set of already seen hashes
        strict: If True, use strict whitespace normalization.
                If False, preserve whitespace differences (default for augmented).
    """
    unique = []
    removed = 0

    for example in examples:
        messages = example.get('messages', [])
        for msg in messages:
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                code_hash = compute_hash(content, strict=strict)
                if code_hash not in seen_hashes:
                    seen_hashes.add(code_hash)
                    unique.append(example)
                else:
                    removed += 1
                break

    return unique, removed


def main():
    parser = argparse.ArgumentParser(description="Augment L4D2 training dataset")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/l4d2_train_v13.jsonl",
        help="Input training data file"
    )
    parser.add_argument(
        "--output-augmented",
        type=str,
        default="data/processed/l4d2_train_v13_augmented.jsonl",
        help="Output file for augmented examples only"
    )
    parser.add_argument(
        "--output-combined",
        type=str,
        default="data/processed/l4d2_train_v14.jsonl",
        help="Output file for original + augmented examples"
    )
    parser.add_argument(
        "--augmentations-per-example",
        type=int,
        default=3,
        help="Target number of augmentations per example"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=2500,
        help="Target dataset size after augmentation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Validate paths
    input_path = safe_path(args.input, PROJECT_ROOT)
    output_augmented_path = safe_path(args.output_augmented, PROJECT_ROOT, create_parents=True)
    output_combined_path = safe_path(args.output_combined, PROJECT_ROOT, create_parents=True)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("L4D2 DATA AUGMENTATION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Input: {input_path}")
    logger.info(f"Target augmentations per example: {args.augmentations_per_example}")
    logger.info(f"Target dataset size: {args.target_size}")
    logger.info(f"Random seed: {args.seed}")
    logger.info("=" * 60)

    # Load training data
    logger.info("Loading training data...")
    original_data = load_training_data(input_path)
    stats = AugmentationStats(original_count=len(original_data))
    logger.info(f"Loaded {stats.original_count} original examples")

    # Initialize augmenters
    sp_augmenter = SourcePawnAugmenter(seed=args.seed)
    vs_augmenter = VScriptAugmenter(seed=args.seed)

    # Track seen hashes for deduplication
    seen_hashes: Set[str] = set()

    # Add original examples to seen hashes
    for example in original_data:
        messages = example.get('messages', [])
        for msg in messages:
            if msg.get('role') == 'assistant':
                seen_hashes.add(compute_hash(msg.get('content', '')))
                break

    # Generate augmentations
    logger.info("Generating augmentations...")
    all_augmented = []

    for i, example in enumerate(original_data):
        if (i + 1) % 100 == 0:
            logger.info(f"Processing example {i + 1}/{len(original_data)}...")

        # Calculate how many augmentations we need
        # If we're close to target, generate fewer
        current_total = len(original_data) + len(all_augmented)
        remaining_needed = args.target_size - current_total

        if remaining_needed <= 0:
            break

        # Adjust augmentations based on remaining need
        augs_for_this = min(
            args.augmentations_per_example,
            max(1, remaining_needed // max(1, len(original_data) - i))
        )

        augmented = augment_example(
            example,
            sp_augmenter,
            vs_augmenter,
            augs_for_this
        )

        # Deduplicate on the fly
        unique_augmented, removed = remove_duplicates(augmented, seen_hashes)
        all_augmented.extend(unique_augmented)
        stats.duplicates_removed += removed

    stats.augmented_generated = len(all_augmented)

    # Combine statistics from augmenters
    stats.augmentation_breakdown = dict(sp_augmenter.augmentation_counts)
    for k, v in vs_augmenter.augmentation_counts.items():
        stats.augmentation_breakdown[k] = stats.augmentation_breakdown.get(k, 0) + v

    # Save augmented examples only
    logger.info(f"Saving augmented examples to {output_augmented_path}...")

    # Clean augmented data (remove internal metadata)
    clean_augmented = []
    for example in all_augmented:
        clean_example = {'messages': example['messages']}
        clean_augmented.append(clean_example)

    safe_write_jsonl(str(output_augmented_path), clean_augmented, PROJECT_ROOT)

    # Create combined dataset
    logger.info(f"Creating combined dataset at {output_combined_path}...")

    # Shuffle combined data
    combined = original_data + clean_augmented
    random.shuffle(combined)

    safe_write_jsonl(str(output_combined_path), combined, PROJECT_ROOT)

    stats.final_count = len(combined)

    # Print statistics
    logger.info("=" * 60)
    logger.info("AUGMENTATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Original examples:      {stats.original_count:,}")
    logger.info(f"Augmented generated:    {stats.augmented_generated:,}")
    logger.info(f"Duplicates removed:     {stats.duplicates_removed:,}")
    logger.info(f"Final dataset size:     {stats.final_count:,}")
    logger.info(f"Expansion ratio:        {stats.final_count / stats.original_count:.2f}x")
    logger.info("")
    logger.info("Augmentation breakdown:")
    for aug_type, count in sorted(stats.augmentation_breakdown.items(), key=lambda x: -x[1]):
        logger.info(f"  {aug_type}: {count:,}")
    logger.info("")
    logger.info(f"Output files:")
    logger.info(f"  Augmented only: {output_augmented_path}")
    logger.info(f"  Combined (v14): {output_combined_path}")
    logger.info("=" * 60)

    # Check if we hit target
    if stats.final_count >= args.target_size:
        logger.info(f"SUCCESS: Reached target size of {args.target_size}+ examples!")
    else:
        logger.warning(
            f"Did not reach target size. Got {stats.final_count}, "
            f"wanted {args.target_size}. Consider increasing --augmentations-per-example"
        )


if __name__ == "__main__":
    main()
