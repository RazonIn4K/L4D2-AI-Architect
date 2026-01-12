#!/usr/bin/env python3
"""
L4D2 Code Explainer

A tool for generating comprehensive explanations of SourcePawn and VScript code
using the fine-tuned L4D2 modding LLM.

Features:
- Line-by-line breakdown
- Function summaries
- Variable purpose analysis
- Hook explanations
- Potential issue detection

Output formats: markdown, json, html
"""

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.security import safe_read_text, safe_write_text

PROJECT_ROOT = Path(__file__).parent.parent.parent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class LineExplanation:
    """Explanation for a single line of code."""
    line_number: int
    code: str
    explanation: str
    category: str = "code"  # code, comment, directive, empty


@dataclass
class FunctionSummary:
    """Summary of a function or method."""
    name: str
    start_line: int
    end_line: int
    signature: str
    purpose: str
    parameters: List[Dict[str, str]]
    return_type: str
    complexity: str  # low, medium, high


@dataclass
class VariableInfo:
    """Information about a variable."""
    name: str
    line_number: int
    var_type: str
    purpose: str
    scope: str  # global, local, parameter


@dataclass
class HookInfo:
    """Information about a game hook or event."""
    name: str
    line_number: int
    hook_type: str  # event, forward, callback
    triggered_by: str
    purpose: str


@dataclass
class PotentialIssue:
    """A potential issue or improvement."""
    line_number: int
    severity: str  # info, warning, error
    issue_type: str
    description: str
    suggestion: str


@dataclass
class CodeExplanation:
    """Complete code explanation result."""
    language: str
    file_name: Optional[str]
    summary: str
    lines: List[LineExplanation] = field(default_factory=list)
    functions: List[FunctionSummary] = field(default_factory=list)
    variables: List[VariableInfo] = field(default_factory=list)
    hooks: List[HookInfo] = field(default_factory=list)
    issues: List[PotentialIssue] = field(default_factory=list)
    includes: List[str] = field(default_factory=list)
    plugin_info: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "language": self.language,
            "file_name": self.file_name,
            "summary": self.summary,
            "lines": [asdict(line) for line in self.lines],
            "functions": [asdict(func) for func in self.functions],
            "variables": [asdict(var) for var in self.variables],
            "hooks": [asdict(hook) for hook in self.hooks],
            "issues": [asdict(issue) for issue in self.issues],
            "includes": self.includes,
            "plugin_info": self.plugin_info
        }
        return result


# =============================================================================
# Code Parser
# =============================================================================

class CodeParser:
    """Parse SourcePawn and VScript code to extract structural information."""

    # SourcePawn patterns
    SP_INCLUDE = re.compile(r'#include\s*<([^>]+)>')
    SP_DEFINE = re.compile(r'#define\s+(\w+)\s+(.*)')
    SP_PRAGMA = re.compile(r'#pragma\s+(\w+)\s*(.*)')
    SP_PLUGIN_INFO = re.compile(r'public\s+Plugin\s+myinfo\s*=\s*{([^}]+)}', re.DOTALL)
    SP_FUNCTION = re.compile(
        r'(?:public\s+|stock\s+|static\s+|native\s+)?((?:Action|void|int|float|bool|char|any|Handle)\s*(?:\[\])?\s*)?'
        r'(\w+)\s*\(([^)]*)\)\s*{'
    )
    SP_GLOBAL_VAR = re.compile(
        r'^(?:static\s+|new\s+)?(?:bool|int|float|char|Handle|ConVar)\s+(\w+)(?:\s*\[.*?\])?\s*(?:=|;)'
    )
    SP_EVENT_HOOK = re.compile(r'HookEvent\s*\(\s*"([^"]+)"\s*,\s*(\w+)')
    SP_FORWARD = re.compile(r'public\s+(?:void\s+)?On(\w+)\s*\(')

    # VScript patterns
    VS_FUNCTION = re.compile(r'function\s+(\w+)\s*\(([^)]*)\)')
    VS_VARIABLE = re.compile(r'(?:local\s+)?(\w+)\s*<-\s*')
    VS_EVENT = re.compile(r'function\s+OnGameEvent_(\w+)\s*\(')

    def __init__(self, code: str, language: str = "auto"):
        self.code = code
        self.lines = code.split('\n')
        self.language = self._detect_language(code) if language == "auto" else language

    def _detect_language(self, code: str) -> str:
        """Detect whether code is SourcePawn or VScript."""
        sp_indicators = ['#include', 'public Plugin', 'HookEvent', 'GetClientTeam', 'PrintToChat']
        vs_indicators = ['function ', ' <- ', 'DirectorOptions', 'printl(', 'local ']

        sp_score = sum(1 for ind in sp_indicators if ind in code)
        vs_score = sum(1 for ind in vs_indicators if ind in code)

        return "sourcepawn" if sp_score >= vs_score else "vscript"

    def parse_includes(self) -> List[str]:
        """Extract all include directives."""
        includes = []
        for line in self.lines:
            match = self.SP_INCLUDE.search(line)
            if match:
                includes.append(match.group(1))
        return includes

    def parse_plugin_info(self) -> Optional[Dict[str, str]]:
        """Extract plugin info block (SourcePawn)."""
        match = self.SP_PLUGIN_INFO.search(self.code)
        if not match:
            return None

        info_block = match.group(1)
        info = {}

        # Parse key-value pairs
        pairs = re.findall(r'(\w+)\s*=\s*"([^"]*)"', info_block)
        for key, value in pairs:
            info[key] = value

        return info

    def parse_functions(self) -> List[Dict[str, Any]]:
        """Extract all function definitions."""
        functions = []
        pattern = self.SP_FUNCTION if self.language == "sourcepawn" else self.VS_FUNCTION

        # Track brace depth to find function end
        for i, line in enumerate(self.lines, 1):
            match = pattern.search(line)
            if match:
                if self.language == "sourcepawn":
                    return_type = (match.group(1) or "void").strip()
                    name = match.group(2)
                    params = match.group(3)
                else:
                    return_type = "any"
                    name = match.group(1)
                    params = match.group(2)

                # Find function end
                end_line = self._find_function_end(i - 1)

                functions.append({
                    "name": name,
                    "start_line": i,
                    "end_line": end_line,
                    "return_type": return_type,
                    "parameters": params,
                    "signature": line.strip()
                })

        return functions

    def _find_function_end(self, start_index: int) -> int:
        """Find the closing brace of a function."""
        depth = 0
        for i in range(start_index, len(self.lines)):
            line = self.lines[i]
            depth += line.count('{') - line.count('}')
            if depth == 0 and '{' in self.lines[start_index]:
                return i + 1
        return len(self.lines)

    def parse_global_variables(self) -> List[Dict[str, Any]]:
        """Extract global variable declarations."""
        variables = []
        in_function = False
        brace_depth = 0

        for i, line in enumerate(self.lines, 1):
            # Track if we're inside a function
            brace_depth += line.count('{') - line.count('}')
            if 'public ' in line or 'void ' in line or 'Action ' in line:
                if '{' in line:
                    in_function = True
            if brace_depth == 0:
                in_function = False

            # Only look at top-level declarations
            if not in_function and brace_depth == 0:
                match = self.SP_GLOBAL_VAR.match(line.strip())
                if match:
                    variables.append({
                        "name": match.group(1),
                        "line_number": i,
                        "declaration": line.strip()
                    })

        return variables

    def parse_hooks(self) -> List[Dict[str, Any]]:
        """Extract event hooks and forwards."""
        hooks = []

        for i, line in enumerate(self.lines, 1):
            # HookEvent calls
            match = self.SP_EVENT_HOOK.search(line)
            if match:
                hooks.append({
                    "name": match.group(1),
                    "callback": match.group(2),
                    "line_number": i,
                    "hook_type": "event"
                })

            # Public forwards (OnPluginStart, OnClientPutInServer, etc.)
            match = self.SP_FORWARD.search(line)
            if match:
                forward_name = "On" + match.group(1)
                hooks.append({
                    "name": forward_name,
                    "callback": forward_name,
                    "line_number": i,
                    "hook_type": "forward"
                })

        return hooks


# =============================================================================
# LLM Backends
# =============================================================================

class OllamaExplainer:
    """Use Ollama for code explanation."""

    DEFAULT_MODEL = "l4d2-code-v10plus"

    def __init__(self, model: str = None):
        self.model = model or self.DEFAULT_MODEL
        self._check_ollama()

    def _check_ollama(self) -> None:
        """Verify Ollama is available."""
        if not shutil.which("ollama"):
            raise RuntimeError("Ollama not found. Install from https://ollama.ai")

    def is_model_available(self) -> bool:
        """Check if model is available."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True
            )
            return self.model in result.stdout
        except Exception:
            return False

    def generate(self, prompt: str, system: str = None) -> str:
        """Generate explanation using Ollama."""
        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        try:
            result = subprocess.run(
                ["ollama", "run", self.model, full_prompt],
                capture_output=True,
                text=True,
                timeout=180  # 3 minute timeout for longer explanations
            )
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            return "Error: Generation timed out"
        except Exception as e:
            return f"Error: {e}"


class MockExplainer:
    """Mock explainer for testing without LLM."""

    def __init__(self, model: str = None):
        self.model = model or "mock"

    def is_model_available(self) -> bool:
        return True

    def generate(self, prompt: str, system: str = None) -> str:
        """Generate mock explanations based on code analysis."""
        return "[Mock explanation - LLM not available]"


# =============================================================================
# Code Explainer
# =============================================================================

class CodeExplainer:
    """Main code explanation engine."""

    SYSTEM_PROMPT_SP = """You are an expert SourcePawn developer specializing in Left 4 Dead 2 modding.
Your task is to explain code clearly and thoroughly, identifying:
- The purpose and functionality of each code section
- Important L4D2-specific hooks, events, and APIs
- Best practices and potential issues
- How the code interacts with the game engine

Be precise, technical, and helpful. Reference specific L4D2 APIs and SourceMod functions."""

    SYSTEM_PROMPT_VS = """You are an expert VScript developer for Left 4 Dead 2.
Your task is to explain code clearly, covering:
- Director scripts and AI Director interactions
- VScript-specific patterns and Squirrel syntax
- Game events and entity manipulation
- L4D2-specific scripting APIs

Be concise but thorough. Reference specific L4D2 VScript APIs and Director options."""

    def __init__(self, backend: str = "ollama", model: str = None):
        self.backend_name = backend
        if backend == "ollama":
            try:
                self.backend = OllamaExplainer(model)
            except RuntimeError as e:
                logger.warning(f"Ollama not available: {e}. Using mock backend.")
                self.backend = MockExplainer(model)
        else:
            self.backend = MockExplainer(model)

    def explain(self, code: str, language: str = "auto", file_name: str = None) -> CodeExplanation:
        """Generate comprehensive code explanation."""
        parser = CodeParser(code, language)
        detected_lang = parser.language

        # Parse structural elements
        includes = parser.parse_includes()
        plugin_info = parser.parse_plugin_info()
        functions = parser.parse_functions()
        variables = parser.parse_global_variables()
        hooks = parser.parse_hooks()

        # Generate explanations using LLM
        system_prompt = self.SYSTEM_PROMPT_SP if detected_lang == "sourcepawn" else self.SYSTEM_PROMPT_VS

        # Get overall summary
        summary = self._generate_summary(code, detected_lang, system_prompt)

        # Build line explanations
        lines = self._explain_lines(code, detected_lang, system_prompt)

        # Build function summaries
        func_summaries = self._explain_functions(functions, code, system_prompt)

        # Build variable info
        var_info = self._explain_variables(variables, code, system_prompt)

        # Build hook info
        hook_info = self._explain_hooks(hooks, detected_lang, system_prompt)

        # Detect potential issues
        issues = self._detect_issues(code, detected_lang, functions)

        return CodeExplanation(
            language=detected_lang,
            file_name=file_name,
            summary=summary,
            lines=lines,
            functions=func_summaries,
            variables=var_info,
            hooks=hook_info,
            issues=issues,
            includes=includes,
            plugin_info=plugin_info
        )

    def _generate_summary(self, code: str, language: str, system_prompt: str) -> str:
        """Generate overall code summary."""
        prompt = f"""Provide a concise summary (2-3 sentences) of what this {language} code does:

```{language}
{code[:2000]}  # Truncate for summary
```

Summary:"""

        return self.backend.generate(prompt, system_prompt)

    def _explain_lines(self, code: str, language: str, system_prompt: str) -> List[LineExplanation]:
        """Generate line-by-line explanations."""
        lines = code.split('\n')
        explanations = []

        # Group lines into logical chunks for efficiency
        chunks = self._chunk_code(lines)

        for chunk in chunks:
            chunk_code = '\n'.join(line for _, line in chunk)
            start_line = chunk[0][0]

            prompt = f"""Explain each line of this {language} code briefly (one sentence per line):

```{language}
{chunk_code}
```

Format: Line N: explanation"""

            response = self.backend.generate(prompt, system_prompt)

            # Parse response and create LineExplanation objects
            for line_num, line_code in chunk:
                category = self._categorize_line(line_code)
                # Extract explanation from response or use default
                explanation = self._extract_line_explanation(response, line_num, line_code, category)
                explanations.append(LineExplanation(
                    line_number=line_num,
                    code=line_code,
                    explanation=explanation,
                    category=category
                ))

        return explanations

    def _chunk_code(self, lines: List[str], chunk_size: int = 20) -> List[List[tuple]]:
        """Split code into chunks for processing."""
        chunks = []
        current_chunk = []

        for i, line in enumerate(lines, 1):
            current_chunk.append((i, line))
            if len(current_chunk) >= chunk_size:
                chunks.append(current_chunk)
                current_chunk = []

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _categorize_line(self, line: str) -> str:
        """Categorize a line of code."""
        stripped = line.strip()
        if not stripped:
            return "empty"
        if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
            return "comment"
        if stripped.startswith('#'):
            return "directive"
        return "code"

    def _extract_line_explanation(self, response: str, line_num: int, line_code: str, category: str) -> str:
        """Extract explanation for a specific line from LLM response."""
        if category == "empty":
            return "Empty line"
        if category == "comment":
            return "Comment: " + line_code.strip().lstrip('/*/ ')

        # Try to find matching line explanation in response
        patterns = [
            rf'Line\s*{line_num}\s*:\s*(.+)',
            rf'{line_num}\s*:\s*(.+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1).strip()

        # Default explanations based on code patterns
        return self._generate_default_explanation(line_code)

    def _generate_default_explanation(self, line: str) -> str:
        """Generate default explanation based on code patterns."""
        stripped = line.strip()

        patterns = [
            (r'#include\s*<(\w+)>', lambda m: f"Include {m.group(1)} library"),
            (r'#pragma\s+semicolon', "Require semicolons"),
            (r'#pragma\s+newdecls', "Use new declaration syntax"),
            (r'#define\s+(\w+)', lambda m: f"Define constant {m.group(1)}"),
            (r'public\s+Plugin\s+myinfo', "Plugin metadata declaration"),
            (r'HookEvent\s*\(\s*"(\w+)"', lambda m: f"Hook game event '{m.group(1)}'"),
            (r'CreateTimer\s*\(', "Create a timer callback"),
            (r'PrintToChat\s*\(', "Print message to player's chat"),
            (r'PrintToChatAll\s*\(', "Print message to all players"),
            (r'GetClientOfUserId\s*\(', "Get client index from user ID"),
            (r'IsClientInGame\s*\(', "Check if client is in game"),
            (r'GetClientTeam\s*\(', "Get client's team number"),
            (r'return\s+Plugin_Handled', "Block event propagation"),
            (r'return\s+Plugin_Continue', "Allow event to continue"),
        ]

        for pattern, explanation in patterns:
            match = re.search(pattern, stripped)
            if match:
                if callable(explanation):
                    return explanation(match)
                return explanation

        return "Code statement"

    def _explain_functions(self, functions: List[Dict], code: str, system_prompt: str) -> List[FunctionSummary]:
        """Generate function summaries."""
        summaries = []

        for func in functions:
            # Extract function body
            lines = code.split('\n')[func['start_line']-1:func['end_line']]
            func_code = '\n'.join(lines)

            prompt = f"""Briefly explain this function (2-3 sentences):

```
{func_code[:1000]}
```

Include: purpose, key operations, and any L4D2-specific behavior."""

            response = self.backend.generate(prompt, system_prompt)

            # Parse parameters
            params = self._parse_parameters(func['parameters'])

            # Estimate complexity
            complexity = self._estimate_complexity(func_code)

            summaries.append(FunctionSummary(
                name=func['name'],
                start_line=func['start_line'],
                end_line=func['end_line'],
                signature=func['signature'],
                purpose=response,
                parameters=params,
                return_type=func['return_type'],
                complexity=complexity
            ))

        return summaries

    def _parse_parameters(self, param_string: str) -> List[Dict[str, str]]:
        """Parse function parameters."""
        if not param_string.strip():
            return []

        params = []
        for param in param_string.split(','):
            param = param.strip()
            if param:
                # Try to extract type and name
                parts = param.split()
                if len(parts) >= 2:
                    params.append({
                        "type": ' '.join(parts[:-1]),
                        "name": parts[-1].strip('&[]')
                    })
                else:
                    params.append({
                        "type": "unknown",
                        "name": param
                    })
        return params

    def _estimate_complexity(self, code: str) -> str:
        """Estimate function complexity."""
        # Simple heuristic based on code patterns
        complexity_indicators = [
            (r'\bif\b', 1),
            (r'\belse\b', 1),
            (r'\bfor\b', 2),
            (r'\bwhile\b', 2),
            (r'\bswitch\b', 2),
            (r'\bcase\b', 1),
            (r'&&|\|\|', 1),
        ]

        score = 0
        for pattern, weight in complexity_indicators:
            score += len(re.findall(pattern, code)) * weight

        if score <= 3:
            return "low"
        elif score <= 8:
            return "medium"
        return "high"

    def _explain_variables(self, variables: List[Dict], code: str, system_prompt: str) -> List[VariableInfo]:
        """Generate variable explanations."""
        var_info = []

        for var in variables:
            # Infer purpose from name and context
            name = var['name']
            purpose = self._infer_variable_purpose(name, var['declaration'])

            # Determine type from declaration
            var_type = self._extract_variable_type(var['declaration'])

            var_info.append(VariableInfo(
                name=name,
                line_number=var['line_number'],
                var_type=var_type,
                purpose=purpose,
                scope="global"
            ))

        return var_info

    def _infer_variable_purpose(self, name: str, declaration: str) -> str:
        """Infer variable purpose from naming conventions."""
        patterns = [
            (r'^g_', "Global variable: "),
            (r'^h', "Handle for: "),
            (r'^b', "Boolean flag for: "),
            (r'^i', "Integer counter/index for: "),
            (r'^fl', "Float value for: "),
            (r'Timer$', "Timer handle for: "),
            (r'Count$', "Count of: "),
            (r'Enabled$', "Toggle flag for: "),
        ]

        prefix = ""
        for pattern, desc in patterns:
            if re.search(pattern, name, re.IGNORECASE):
                prefix = desc
                break

        # Convert camelCase/PascalCase to readable text
        readable = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        readable = re.sub(r'^[bg]_?', '', readable)

        return prefix + readable.lower()

    def _extract_variable_type(self, declaration: str) -> str:
        """Extract variable type from declaration."""
        types = ['bool', 'int', 'float', 'char', 'Handle', 'ConVar', 'StringMap']
        for t in types:
            if t in declaration:
                return t
        return "unknown"

    def _explain_hooks(self, hooks: List[Dict], language: str, system_prompt: str) -> List[HookInfo]:
        """Generate hook explanations."""
        hook_info = []

        # Known L4D2 event descriptions
        event_descriptions = {
            "round_start": "Fired when a new round begins",
            "round_end": "Fired when the round ends",
            "player_spawn": "Fired when a player spawns",
            "player_death": "Fired when a player dies",
            "player_hurt": "Fired when a player takes damage",
            "tank_spawn": "Fired when a Tank spawns",
            "tank_killed": "Fired when a Tank is killed",
            "witch_spawn": "Fired when a Witch spawns",
            "witch_killed": "Fired when a Witch is killed",
            "infected_death": "Fired when common infected dies",
            "player_incapacitated": "Fired when survivor goes down",
            "revive_success": "Fired when a player is revived",
        }

        forward_descriptions = {
            "OnPluginStart": "Called when plugin loads - initialize hooks and commands",
            "OnPluginEnd": "Called when plugin unloads - cleanup resources",
            "OnMapStart": "Called when a new map loads",
            "OnMapEnd": "Called when map ends",
            "OnClientPutInServer": "Called when client fully connects",
            "OnClientDisconnect": "Called when client disconnects",
            "OnPlayerRunCmd": "Called every frame for player input processing",
        }

        for hook in hooks:
            name = hook['name']

            if hook['hook_type'] == 'event':
                triggered_by = "Game event system"
                purpose = event_descriptions.get(name, f"Game event: {name}")
            else:
                triggered_by = "SourceMod forward"
                purpose = forward_descriptions.get(name, f"Forward: {name}")

            hook_info.append(HookInfo(
                name=name,
                line_number=hook['line_number'],
                hook_type=hook['hook_type'],
                triggered_by=triggered_by,
                purpose=purpose
            ))

        return hook_info

    def _detect_issues(self, code: str, language: str, functions: List[Dict]) -> List[PotentialIssue]:
        """Detect potential issues in the code."""
        issues = []
        lines = code.split('\n')

        # Check for common issues
        issue_patterns = [
            # Memory leaks
            (r'CreateTimer\s*\([^)]*\)',
             "warning", "memory",
             "Timer created - ensure proper cleanup",
             "Consider using timer flags or tracking handles"),

            # Missing validation
            (r'GetClientOfUserId\s*\([^)]*\)\s*;(?!\s*if)',
             "warning", "validation",
             "Client index used without validation",
             "Check if client > 0 and IsClientInGame(client)"),

            # Hardcoded team numbers
            (r'GetClientTeam\s*\([^)]*\)\s*==\s*[23](?!\s*//)',
             "info", "magic_number",
             "Hardcoded team number",
             "Consider using named constants (TEAM_SURVIVOR=2, TEAM_INFECTED=3)"),

            # Potential division by zero
            (r'/\s*\w+(?!\s*//)',
             "info", "safety",
             "Division operation - check for zero",
             "Add zero-check before division"),
        ]

        for i, line in enumerate(lines, 1):
            for pattern, severity, issue_type, desc, suggestion in issue_patterns:
                if re.search(pattern, line):
                    issues.append(PotentialIssue(
                        line_number=i,
                        severity=severity,
                        issue_type=issue_type,
                        description=desc,
                        suggestion=suggestion
                    ))

        # Check for missing cleanup in OnPluginEnd
        if 'OnPluginStart' in code and 'OnPluginEnd' not in code:
            if 'CreateTimer' in code or 'HookEvent' in code:
                issues.append(PotentialIssue(
                    line_number=1,
                    severity="warning",
                    issue_type="cleanup",
                    description="No OnPluginEnd defined but resources are created",
                    suggestion="Add OnPluginEnd to clean up timers and unhook events"
                ))

        return issues


# =============================================================================
# Output Formatters
# =============================================================================

class MarkdownFormatter:
    """Format explanation as Markdown."""

    def format(self, explanation: CodeExplanation) -> str:
        """Format CodeExplanation as Markdown."""
        sections = []

        # Header
        title = explanation.file_name or "Code Explanation"
        sections.append(f"# {title}\n")
        sections.append(f"**Language:** {explanation.language.title()}\n")

        # Summary
        sections.append("## Summary\n")
        sections.append(f"{explanation.summary}\n")

        # Plugin Info (if available)
        if explanation.plugin_info:
            sections.append("## Plugin Information\n")
            sections.append("| Property | Value |")
            sections.append("|----------|-------|")
            for key, value in explanation.plugin_info.items():
                sections.append(f"| {key} | {value} |")
            sections.append("")

        # Includes
        if explanation.includes:
            sections.append("## Includes\n")
            for inc in explanation.includes:
                sections.append(f"- `{inc}`")
            sections.append("")

        # Functions
        if explanation.functions:
            sections.append("## Functions\n")
            for func in explanation.functions:
                sections.append(f"### `{func.name}` (lines {func.start_line}-{func.end_line})\n")
                sections.append(f"**Signature:** `{func.signature}`\n")
                sections.append(f"**Return Type:** `{func.return_type}`\n")
                sections.append(f"**Complexity:** {func.complexity}\n")
                if func.parameters:
                    sections.append("**Parameters:**")
                    for param in func.parameters:
                        sections.append(f"- `{param['name']}` ({param['type']})")
                    sections.append("")
                sections.append(f"**Purpose:** {func.purpose}\n")

        # Hooks
        if explanation.hooks:
            sections.append("## Hooks & Events\n")
            sections.append("| Hook | Type | Line | Purpose |")
            sections.append("|------|------|------|---------|")
            for hook in explanation.hooks:
                sections.append(f"| `{hook.name}` | {hook.hook_type} | {hook.line_number} | {hook.purpose} |")
            sections.append("")

        # Variables
        if explanation.variables:
            sections.append("## Global Variables\n")
            sections.append("| Variable | Type | Line | Purpose |")
            sections.append("|----------|------|------|---------|")
            for var in explanation.variables:
                sections.append(f"| `{var.name}` | {var.var_type} | {var.line_number} | {var.purpose} |")
            sections.append("")

        # Issues
        if explanation.issues:
            sections.append("## Potential Issues\n")
            for issue in explanation.issues:
                icon = {"error": "[!]", "warning": "[?]", "info": "[i]"}.get(issue.severity, "[?]")
                sections.append(f"### {icon} Line {issue.line_number}: {issue.issue_type}\n")
                sections.append(f"**Description:** {issue.description}\n")
                sections.append(f"**Suggestion:** {issue.suggestion}\n")

        # Line-by-line (abbreviated)
        if explanation.lines:
            sections.append("## Line-by-Line Breakdown\n")
            sections.append("```")
            for line in explanation.lines[:50]:  # Limit to first 50 lines
                if line.category != "empty":
                    sections.append(f"{line.line_number:4d} | {line.explanation}")
            if len(explanation.lines) > 50:
                sections.append(f"... ({len(explanation.lines) - 50} more lines)")
            sections.append("```\n")

        return '\n'.join(sections)


class JSONFormatter:
    """Format explanation as JSON."""

    def format(self, explanation: CodeExplanation) -> str:
        """Format CodeExplanation as JSON."""
        return json.dumps(explanation.to_dict(), indent=2)


class HTMLFormatter:
    """Format explanation as HTML."""

    def format(self, explanation: CodeExplanation) -> str:
        """Format CodeExplanation as HTML."""
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "  <meta charset='UTF-8'>",
            "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            f"  <title>Code Explanation: {explanation.file_name or 'Unknown'}</title>",
            "  <style>",
            self._get_styles(),
            "  </style>",
            "</head>",
            "<body>",
            "  <div class='container'>",
        ]

        # Header
        html_parts.append(f"    <h1>{explanation.file_name or 'Code Explanation'}</h1>")
        html_parts.append(f"    <p class='meta'>Language: <span class='lang'>{explanation.language.title()}</span></p>")

        # Summary
        html_parts.append("    <section class='summary'>")
        html_parts.append("      <h2>Summary</h2>")
        html_parts.append(f"      <p>{explanation.summary}</p>")
        html_parts.append("    </section>")

        # Plugin Info
        if explanation.plugin_info:
            html_parts.append("    <section class='plugin-info'>")
            html_parts.append("      <h2>Plugin Information</h2>")
            html_parts.append("      <table>")
            for key, value in explanation.plugin_info.items():
                html_parts.append(f"        <tr><th>{key}</th><td>{value}</td></tr>")
            html_parts.append("      </table>")
            html_parts.append("    </section>")

        # Functions
        if explanation.functions:
            html_parts.append("    <section class='functions'>")
            html_parts.append("      <h2>Functions</h2>")
            for func in explanation.functions:
                html_parts.append("      <div class='function'>")
                html_parts.append(f"        <h3><code>{func.name}</code></h3>")
                html_parts.append(f"        <p class='lines'>Lines {func.start_line}-{func.end_line}</p>")
                html_parts.append(f"        <p class='signature'><code>{func.signature}</code></p>")
                html_parts.append(f"        <p class='complexity'>Complexity: <span class='{func.complexity}'>{func.complexity}</span></p>")
                html_parts.append(f"        <p>{func.purpose}</p>")
                html_parts.append("      </div>")
            html_parts.append("    </section>")

        # Issues
        if explanation.issues:
            html_parts.append("    <section class='issues'>")
            html_parts.append("      <h2>Potential Issues</h2>")
            for issue in explanation.issues:
                html_parts.append(f"      <div class='issue {issue.severity}'>")
                html_parts.append(f"        <h4>Line {issue.line_number}: {issue.issue_type}</h4>")
                html_parts.append(f"        <p><strong>Description:</strong> {issue.description}</p>")
                html_parts.append(f"        <p><strong>Suggestion:</strong> {issue.suggestion}</p>")
                html_parts.append("      </div>")
            html_parts.append("    </section>")

        html_parts.extend([
            "  </div>",
            "</body>",
            "</html>"
        ])

        return '\n'.join(html_parts)

    def _get_styles(self) -> str:
        """Get CSS styles for HTML output."""
        return """
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      line-height: 1.6;
      color: #333;
      max-width: 1200px;
      margin: 0 auto;
      padding: 20px;
      background: #f5f5f5;
    }
    .container {
      background: white;
      padding: 30px;
      border-radius: 8px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
    h2 { color: #34495e; margin-top: 30px; }
    h3 { color: #2980b9; }
    code {
      background: #f8f8f8;
      padding: 2px 6px;
      border-radius: 3px;
      font-family: 'Monaco', 'Menlo', monospace;
    }
    .meta { color: #666; }
    .lang { background: #3498db; color: white; padding: 2px 8px; border-radius: 3px; }
    table { border-collapse: collapse; width: 100%; margin: 15px 0; }
    th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
    th { background: #f4f4f4; }
    .function { border-left: 4px solid #3498db; padding-left: 15px; margin: 15px 0; }
    .issue { padding: 15px; margin: 10px 0; border-radius: 5px; }
    .issue.error { background: #ffebee; border-left: 4px solid #e74c3c; }
    .issue.warning { background: #fff3e0; border-left: 4px solid #f39c12; }
    .issue.info { background: #e3f2fd; border-left: 4px solid #3498db; }
    .complexity .low { color: #27ae60; }
    .complexity .medium { color: #f39c12; }
    .complexity .high { color: #e74c3c; }
"""


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="L4D2 Code Explainer - Generate comprehensive code explanations"
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--file", "-f",
        help="Path to source code file (.sp or .nut)"
    )
    input_group.add_argument(
        "--code", "-c",
        help="Inline code to explain"
    )

    # Configuration options
    parser.add_argument(
        "--format", "-F",
        choices=["markdown", "json", "html"],
        default="markdown",
        help="Output format (default: markdown)"
    )
    parser.add_argument(
        "--language", "-l",
        choices=["auto", "sourcepawn", "vscript"],
        default="auto",
        help="Code language (default: auto-detect)"
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["ollama", "mock"],
        default="ollama",
        help="LLM backend (default: ollama)"
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Model name for LLM backend"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: stdout)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Read input code
    file_name = None
    if args.file:
        try:
            code = safe_read_text(args.file, PROJECT_ROOT)
            file_name = Path(args.file).name
            logger.info(f"Read {len(code)} characters from {args.file}")
        except FileNotFoundError:
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        except ValueError as e:
            print(f"Error: Invalid path - {e}", file=sys.stderr)
            sys.exit(1)
    else:
        code = args.code
        logger.info(f"Processing inline code ({len(code)} characters)")

    if not code.strip():
        print("Error: No code provided", file=sys.stderr)
        sys.exit(1)

    # Create explainer and generate explanation
    try:
        explainer = CodeExplainer(backend=args.backend, model=args.model)
        explanation = explainer.explain(code, language=args.language, file_name=file_name)
    except Exception as e:
        print(f"Error generating explanation: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Format output
    formatters = {
        "markdown": MarkdownFormatter(),
        "json": JSONFormatter(),
        "html": HTMLFormatter()
    }
    formatter = formatters[args.format]
    output = formatter.format(explanation)

    # Write output
    if args.output:
        try:
            safe_write_text(args.output, output, PROJECT_ROOT)
            print(f"Output written to {args.output}")
        except ValueError as e:
            print(f"Error: Invalid output path - {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(output)


if __name__ == "__main__":
    main()
