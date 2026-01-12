#!/usr/bin/env python3
"""
L4D2/SourcePawn Anti-Pattern Detection System

Scans generated SourcePawn code for common anti-patterns, security vulnerabilities,
and bad practices specific to L4D2 plugin development.

Detects 35+ anti-patterns including:
- Deprecated API usage
- Memory leaks (unclosed handles)
- Blocking operations in callbacks
- Incorrect client validation
- SQL injection vulnerabilities
- Race conditions
- Infinite loops
- Resource management issues
- L4D2-specific API misuse

Usage:
    python detect_antipatterns.py --code "public void OnPluginStart()..."
    python detect_antipatterns.py --file path/to/plugin.sp
    python detect_antipatterns.py --dir path/to/plugins/
    python detect_antipatterns.py --json --file plugin.sp  # JSON output

Integration with benchmark suite:
    from detect_antipatterns import AntiPatternDetector
    detector = AntiPatternDetector()
    result = detector.scan(code)
    print(f"Found {len(result.warnings)} anti-patterns")
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_path, safe_read_text, safe_write_json

PROJECT_ROOT = Path(__file__).parent.parent.parent


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class Severity(str, Enum):
    """Severity levels for anti-patterns."""
    CRITICAL = "critical"  # Security vulnerabilities, crashes
    HIGH = "high"          # Memory leaks, major bugs
    MEDIUM = "medium"      # Bad practices, performance issues
    LOW = "low"            # Style issues, minor improvements
    INFO = "info"          # Suggestions, best practices


class Category(str, Enum):
    """Categories of anti-patterns."""
    SECURITY = "security"
    MEMORY = "memory"
    PERFORMANCE = "performance"
    CORRECTNESS = "correctness"
    L4D2_API = "l4d2_api"
    DEPRECATED = "deprecated"
    STYLE = "style"


@dataclass
class AntiPattern:
    """Definition of an anti-pattern."""
    id: str
    name: str
    description: str
    severity: Severity
    category: Category
    pattern: str  # Regex pattern to match
    fix_suggestion: str
    examples: List[str] = field(default_factory=list)
    # Optional: custom checker function for complex patterns
    checker: Optional[Callable[[str], List[Tuple[int, str]]]] = None


@dataclass
class Warning:
    """A warning instance found in code."""
    antipattern_id: str
    name: str
    severity: Severity
    category: Category
    line_number: int
    line_content: str
    description: str
    fix_suggestion: str
    context: str = ""  # Surrounding code for context

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "antipattern_id": self.antipattern_id,
            "name": self.name,
            "severity": self.severity.value,
            "category": self.category.value,
            "line_number": self.line_number,
            "line_content": self.line_content,
            "description": self.description,
            "fix_suggestion": self.fix_suggestion,
            "context": self.context,
        }


@dataclass
class ScanResult:
    """Result of scanning code for anti-patterns."""
    file_path: str
    total_lines: int
    warnings: List[Warning]
    summary: Dict[str, int]
    score: float  # 0-100, higher is better

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_path": self.file_path,
            "total_lines": self.total_lines,
            "warnings": [w.to_dict() for w in self.warnings],
            "summary": self.summary,
            "score": self.score,
        }


# =============================================================================
# ANTI-PATTERN DEFINITIONS (35+ patterns)
# =============================================================================

def check_unclosed_handles(code: str) -> List[Tuple[int, str]]:
    """Check for handles that are opened but never closed."""
    findings = []
    lines = code.split('\n')

    # Track handle variables
    handle_vars = set()
    closed_vars = set()

    for i, line in enumerate(lines, 1):
        # Find handle creations
        create_patterns = [
            r'(\w+)\s*=\s*CreateMenu\s*\(',
            r'(\w+)\s*=\s*CreateTimer\s*\(',
            r'(\w+)\s*=\s*CreateKeyValues\s*\(',
            r'(\w+)\s*=\s*SQL_Connect\s*\(',
            r'(\w+)\s*=\s*OpenFile\s*\(',
            r'(\w+)\s*=\s*CreateDataPack\s*\(',
            r'(\w+)\s*=\s*CreateArray\s*\(',
            r'(\w+)\s*=\s*CreateTrie\s*\(',
        ]

        for pattern in create_patterns:
            match = re.search(pattern, line)
            if match:
                handle_vars.add((match.group(1), i, line.strip()))

        # Find handle closures
        close_patterns = [
            r'CloseHandle\s*\(\s*(\w+)\s*\)',
            r'delete\s+(\w+)',
            r'(\w+)\s*=\s*INVALID_HANDLE',
        ]

        for pattern in close_patterns:
            match = re.search(pattern, line)
            if match:
                closed_vars.add(match.group(1))

    # Report unclosed handles
    for var_name, line_num, line_content in handle_vars:
        if var_name not in closed_vars:
            findings.append((line_num, line_content))

    return findings


def check_blocking_in_callbacks(code: str) -> List[Tuple[int, str]]:
    """Check for blocking operations inside event callbacks."""
    findings = []
    lines = code.split('\n')

    in_callback = False
    callback_depth = 0
    callback_start = 0

    blocking_ops = [
        'SQL_Query', 'SQL_FastQuery', 'ReadFile', 'WriteFile',
        'FileExists', 'CreateDirectory', 'OpenFile', 'Sleep',
    ]

    for i, line in enumerate(lines, 1):
        # Detect callback functions
        if re.search(r'public\s+(?:Action\s+)?(?:void\s+)?(?:Event_|Timer_|Menu_|SDK_)', line):
            in_callback = True
            callback_start = i
            callback_depth = 0

        # Track brace depth
        callback_depth += line.count('{') - line.count('}')

        if in_callback and callback_depth <= 0 and '{' in lines[callback_start-1:i-1]:
            in_callback = False

        # Check for blocking operations
        if in_callback:
            for op in blocking_ops:
                if op in line and 'Threaded' not in line:
                    findings.append((i, line.strip()))

    return findings


def check_client_validation(code: str) -> List[Tuple[int, str]]:
    """Check for missing client validation."""
    findings = []
    lines = code.split('\n')

    for i, line in enumerate(lines, 1):
        # Check for client index usage without validation
        if re.search(r'GetClient(?:Name|Health|Team|Weapon)\s*\(\s*client\s*\)', line):
            # Look back for validation
            prev_lines = '\n'.join(lines[max(0, i-10):i])
            if 'IsClientInGame' not in prev_lines and 'IsValidClient' not in prev_lines:
                findings.append((i, line.strip()))

        # Check for GetClientOfUserId without validation
        if 'GetClientOfUserId' in line:
            next_lines = '\n'.join(lines[i:min(len(lines), i+5)])
            if 'IsClientInGame' not in next_lines and '> 0' not in next_lines:
                findings.append((i, line.strip()))

    return findings


def check_sql_injection(code: str) -> List[Tuple[int, str]]:
    """Check for potential SQL injection vulnerabilities."""
    findings = []
    lines = code.split('\n')

    for i, line in enumerate(lines, 1):
        # Format directly into SQL without escaping
        if re.search(r'Format\s*\([^)]*(?:SELECT|INSERT|UPDATE|DELETE)', line, re.IGNORECASE):
            if 'SQL_EscapeString' not in '\n'.join(lines[max(0, i-5):i]):
                findings.append((i, line.strip()))

        # Direct string concatenation in SQL
        if re.search(r'SQL_Query\s*\([^)]*\+\s*\w+', line):
            findings.append((i, line.strip()))

    return findings


def check_infinite_loops(code: str) -> List[Tuple[int, str]]:
    """Check for potential infinite loops."""
    findings = []
    lines = code.split('\n')

    for i, line in enumerate(lines, 1):
        # while(true) without break
        if re.search(r'while\s*\(\s*(?:true|1)\s*\)', line, re.IGNORECASE):
            # Check next 20 lines for break
            next_lines = '\n'.join(lines[i:min(len(lines), i+20)])
            if 'break' not in next_lines and 'return' not in next_lines:
                findings.append((i, line.strip()))

        # for loop with no increment
        if re.search(r'for\s*\([^;]*;\s*[^;]*;\s*\)', line):
            findings.append((i, line.strip()))

    return findings


def check_race_conditions(code: str) -> List[Tuple[int, str]]:
    """Check for potential race conditions."""
    findings = []
    lines = code.split('\n')

    # Global variables modified in multiple callbacks
    global_vars = set()
    callback_modifies = {}

    for i, line in enumerate(lines, 1):
        # Track global variable declarations
        if re.search(r'^(?:int|float|bool|char|Handle|ArrayList)\s+g_\w+', line):
            match = re.search(r'(g_\w+)', line)
            if match:
                global_vars.add(match.group(1))

        # Track modifications in callbacks
        if re.search(r'public\s+(?:Action\s+)?(?:void\s+)?(\w+)\s*\(', line):
            current_callback = re.search(r'public\s+(?:Action\s+)?(?:void\s+)?(Event_\w+|Timer_\w+|OnClient\w+)', line)
            if current_callback:
                callback_name = current_callback.group(1)
                for var in global_vars:
                    if re.search(rf'{var}\s*[+\-*/%]?=', line):
                        if var not in callback_modifies:
                            callback_modifies[var] = []
                        callback_modifies[var].append((callback_name, i, line.strip()))

    # Report variables modified in multiple callbacks without synchronization
    for var, modifications in callback_modifies.items():
        if len(set(m[0] for m in modifications)) > 1:
            for _, line_num, line_content in modifications:
                findings.append((line_num, f"Global variable {var} modified in multiple callbacks"))

    return findings


# All anti-patterns with their detection logic
ANTI_PATTERNS: List[AntiPattern] = [
    # ==========================================================================
    # SECURITY (7 patterns)
    # ==========================================================================
    AntiPattern(
        id="SEC001",
        name="SQL Injection",
        description="User input directly formatted into SQL query without escaping",
        severity=Severity.CRITICAL,
        category=Category.SECURITY,
        pattern=r'Format\s*\([^)]*(?:SELECT|INSERT|UPDATE|DELETE)',
        fix_suggestion="Use SQL_EscapeString() to sanitize user input before including in queries",
        checker=check_sql_injection,
    ),
    AntiPattern(
        id="SEC002",
        name="Command Injection",
        description="User input passed directly to ServerCommand or ClientCommand",
        severity=Severity.CRITICAL,
        category=Category.SECURITY,
        pattern=r'(?:Server|Client)Command\s*\([^)]*(?:GetCmdArg|args)',
        fix_suggestion="Validate and sanitize user input before passing to command execution",
    ),
    AntiPattern(
        id="SEC003",
        name="RCON Password Exposure",
        description="RCON password hardcoded or logged",
        severity=Severity.CRITICAL,
        category=Category.SECURITY,
        pattern=r'(?:rcon_password|sv_rcon_password)\s*["\']',
        fix_suggestion="Never hardcode RCON passwords; use ConVar with FCVAR_PROTECTED flag",
    ),
    AntiPattern(
        id="SEC004",
        name="Insufficient Admin Check",
        description="Admin command without proper permission validation",
        severity=Severity.HIGH,
        category=Category.SECURITY,
        pattern=r'RegAdminCmd\s*\([^)]*ADMFLAG_ROOT[^)]*\)',
        fix_suggestion="Use specific admin flags instead of ADMFLAG_ROOT when possible",
    ),
    AntiPattern(
        id="SEC005",
        name="Client Index Trust",
        description="Trusting client-provided index without validation",
        severity=Severity.HIGH,
        category=Category.SECURITY,
        pattern=r'GetCmdArg(?:Int)?\s*\([^)]*\)[^;]*(?:GetClient|TeleportEntity)',
        fix_suggestion="Always validate client indices with IsClientInGame() and bounds checking",
    ),
    AntiPattern(
        id="SEC006",
        name="Path Traversal",
        description="File path constructed from user input without sanitization",
        severity=Severity.HIGH,
        category=Category.SECURITY,
        pattern=r'(?:BuildPath|OpenFile)\s*\([^)]*GetCmdArg',
        fix_suggestion="Sanitize file paths and restrict to allowed directories",
    ),
    AntiPattern(
        id="SEC007",
        name="Unsafe Format String",
        description="User input used as format string",
        severity=Severity.MEDIUM,
        category=Category.SECURITY,
        pattern=r'(?:Print\w+|LogMessage|Format)\s*\(\s*[^,]*,\s*(?:buffer|input|arg)',
        fix_suggestion="Use %s format specifier for user input, never pass directly as format",
    ),

    # ==========================================================================
    # MEMORY (6 patterns)
    # ==========================================================================
    AntiPattern(
        id="MEM001",
        name="Unclosed Handle",
        description="Handle created but never closed, causing memory leak",
        severity=Severity.HIGH,
        category=Category.MEMORY,
        pattern=r'=\s*Create(?:Menu|Timer|KeyValues|DataPack|Array)\s*\(',
        fix_suggestion="Always close handles with CloseHandle() or delete when done",
        checker=check_unclosed_handles,
    ),
    AntiPattern(
        id="MEM002",
        name="Missing DataPack Reset",
        description="DataPack read without ResetPack(), reading garbage data",
        severity=Severity.MEDIUM,
        category=Category.MEMORY,
        pattern=r'ReadPackCell\s*\([^)]*\)(?:(?!ResetPack)[\s\S]){0,200}$',
        fix_suggestion="Call ResetPack() before reading from a DataPack passed to timer",
    ),
    AntiPattern(
        id="MEM003",
        name="Entity Reference Leak",
        description="EntIndexToEntRef stored without cleanup",
        severity=Severity.MEDIUM,
        category=Category.MEMORY,
        pattern=r'EntIndexToEntRef\s*\([^)]*\)(?:(?!EntRefToEntIndex)[\s\S])*$',
        fix_suggestion="Track entity references and clean up when entities are removed",
    ),
    AntiPattern(
        id="MEM004",
        name="StringMap/ArrayList Leak",
        description="Dynamic container created without cleanup",
        severity=Severity.HIGH,
        category=Category.MEMORY,
        pattern=r'new\s+(?:StringMap|ArrayList)\s*\(',
        fix_suggestion="Store in global variable and delete in OnPluginEnd or use local scope",
    ),
    AntiPattern(
        id="MEM005",
        name="Recursive Timer Without Limit",
        description="Timer recreates itself without termination condition",
        severity=Severity.MEDIUM,
        category=Category.MEMORY,
        pattern=r'CreateTimer\s*\([^)]*Timer_\w+[^)]*\)[\s\S]{0,500}CreateTimer\s*\([^)]*Timer_\w+',
        fix_suggestion="Add a termination condition or use TIMER_REPEAT flag instead",
    ),
    AntiPattern(
        id="MEM006",
        name="Menu Handle in Loop",
        description="Creating menus in a loop without closing previous",
        severity=Severity.MEDIUM,
        category=Category.MEMORY,
        pattern=r'for\s*\([^)]*\)[\s\S]{0,100}CreateMenu',
        fix_suggestion="Reuse menu handles or ensure proper cleanup in each iteration",
    ),

    # ==========================================================================
    # PERFORMANCE (6 patterns)
    # ==========================================================================
    AntiPattern(
        id="PERF001",
        name="Blocking Operation in Callback",
        description="Synchronous file/database operation in event callback",
        severity=Severity.HIGH,
        category=Category.PERFORMANCE,
        pattern=r'public\s+(?:Action\s+)?(?:Event_|Timer_)[\s\S]{0,500}(?:SQL_Query|ReadFile)',
        fix_suggestion="Use threaded queries (SQL_TQuery) and asynchronous file operations",
        checker=check_blocking_in_callbacks,
    ),
    AntiPattern(
        id="PERF002",
        name="FindEntityByClassname in Frame",
        description="Entity search in OnGameFrame or frequently called hook",
        severity=Severity.MEDIUM,
        category=Category.PERFORMANCE,
        pattern=r'OnGameFrame[\s\S]{0,300}FindEntityByClassname',
        fix_suggestion="Cache entity lists and update on spawn/death events instead",
    ),
    AntiPattern(
        id="PERF003",
        name="String Operation in Hot Path",
        description="Expensive string formatting in frequently called function",
        severity=Severity.LOW,
        category=Category.PERFORMANCE,
        pattern=r'OnPlayerRunCmd[\s\S]{0,200}(?:Format|StrCat|strcopy)',
        fix_suggestion="Pre-format strings or use direct printing where possible",
    ),
    AntiPattern(
        id="PERF004",
        name="Excessive Timer Creation",
        description="Creating new timer on every event instead of reusing",
        severity=Severity.MEDIUM,
        category=Category.PERFORMANCE,
        pattern=r'public\s+(?:void|Action)\s+Event_[\s\S]{0,200}CreateTimer',
        fix_suggestion="Consider using a single repeating timer or rate limiting",
    ),
    AntiPattern(
        id="PERF005",
        name="GetEntProp in Loop",
        description="Entity property access inside loop without caching",
        severity=Severity.LOW,
        category=Category.PERFORMANCE,
        pattern=r'for\s*\([^)]*\)[\s\S]{0,100}GetEntProp\s*\([^)]*,\s*[^,]*,\s*[^,)]*\)',
        fix_suggestion="Cache entity properties before loop when reading same property multiple times",
    ),
    AntiPattern(
        id="PERF006",
        name="TraceRay in OnGameFrame",
        description="Expensive trace operation every frame",
        severity=Severity.MEDIUM,
        category=Category.PERFORMANCE,
        pattern=r'OnGameFrame[\s\S]{0,300}TR_TraceRay',
        fix_suggestion="Rate-limit trace operations or use simpler distance checks first",
    ),

    # ==========================================================================
    # CORRECTNESS (8 patterns)
    # ==========================================================================
    AntiPattern(
        id="CORR001",
        name="Missing Client Validation",
        description="Client function called without IsClientInGame check",
        severity=Severity.HIGH,
        category=Category.CORRECTNESS,
        pattern=r'GetClient(?:Name|Health|Team)\s*\(\s*(?:client|i)\s*\)',
        fix_suggestion="Always check IsClientInGame(client) before accessing client properties",
        checker=check_client_validation,
    ),
    AntiPattern(
        id="CORR002",
        name="Infinite Loop Risk",
        description="Loop condition that may never terminate",
        severity=Severity.HIGH,
        category=Category.CORRECTNESS,
        pattern=r'while\s*\(\s*(?:true|1)\s*\)',
        fix_suggestion="Ensure loop has explicit break condition or use bounded iteration",
        checker=check_infinite_loops,
    ),
    AntiPattern(
        id="CORR003",
        name="Race Condition",
        description="Global state modified in multiple async callbacks",
        severity=Severity.MEDIUM,
        category=Category.CORRECTNESS,
        pattern=r'g_\w+\s*[+\-*/%]?=',
        fix_suggestion="Use atomic operations or protect shared state with proper synchronization",
        checker=check_race_conditions,
    ),
    AntiPattern(
        id="CORR004",
        name="Off-by-One MaxClients",
        description="Loop using <= MaxClients instead of < or starting at 0",
        severity=Severity.MEDIUM,
        category=Category.CORRECTNESS,
        pattern=r'for\s*\(\s*int\s+\w+\s*=\s*0\s*;\s*\w+\s*<=?\s*MaxClients',
        fix_suggestion="Client indices start at 1: for(int i = 1; i <= MaxClients; i++)",
    ),
    AntiPattern(
        id="CORR005",
        name="Entity Index Reuse",
        description="Storing entity index long-term without EntIndexToEntRef",
        severity=Severity.MEDIUM,
        category=Category.CORRECTNESS,
        pattern=r'g_\w+\s*=\s*(?:FindEntityByClassname|CreateEntityByName)',
        fix_suggestion="Use EntIndexToEntRef() for persistent entity references",
    ),
    AntiPattern(
        id="CORR006",
        name="Return Value Ignored",
        description="Important function return value not checked",
        severity=Severity.MEDIUM,
        category=Category.CORRECTNESS,
        pattern=r'^\s*(?:CreateEntityByName|SQL_Connect|OpenFile)\s*\(',
        fix_suggestion="Check return values for errors: if(entity == -1) or handle == null",
    ),
    AntiPattern(
        id="CORR007",
        name="Timer Handle Overwrite",
        description="Timer handle overwritten without killing previous timer",
        severity=Severity.MEDIUM,
        category=Category.CORRECTNESS,
        pattern=r'g_hTimer\s*=\s*CreateTimer(?:(?!KillTimer)[\s\S]){0,100}g_hTimer\s*=\s*CreateTimer',
        fix_suggestion="Kill existing timer before creating new one: if(g_hTimer != null) KillTimer(g_hTimer)",
    ),
    AntiPattern(
        id="CORR008",
        name="Event GetInt Without Userid Check",
        description="Getting userid from event without validating client",
        severity=Severity.MEDIUM,
        category=Category.CORRECTNESS,
        pattern=r'GetClientOfUserId\s*\(\s*event\.GetInt\s*\([^)]*\)\s*\)(?:(?!IsClientInGame)[\s\S]){0,50};',
        fix_suggestion="Always validate: int client = GetClientOfUserId(...); if(client > 0 && IsClientInGame(client))",
    ),

    # ==========================================================================
    # L4D2-SPECIFIC API (5 patterns)
    # ==========================================================================
    AntiPattern(
        id="L4D001",
        name="Wrong Random Function",
        description="Using RandomFloat/RandomInt instead of GetRandomFloat/GetRandomInt",
        severity=Severity.HIGH,
        category=Category.L4D2_API,
        pattern=r'\b(?<!Get)Random(?:Float|Int)\s*\(',
        fix_suggestion="Use GetRandomFloat() and GetRandomInt() in SourceMod",
    ),
    AntiPattern(
        id="L4D002",
        name="Wrong Speed Property",
        description="Using m_flMaxSpeed instead of m_flLaggedMovementValue for L4D2",
        severity=Severity.HIGH,
        category=Category.L4D2_API,
        pattern=r'["\']m_fl(?:Max)?Speed["\']',
        fix_suggestion="Use m_flLaggedMovementValue for L4D2 movement speed modification",
    ),
    AntiPattern(
        id="L4D003",
        name="Wrong Event Name",
        description="Using incorrect L4D2 event name",
        severity=Severity.HIGH,
        category=Category.L4D2_API,
        pattern=r'HookEvent\s*\(\s*["\'](?:pounce|smoker_tongue_grab|boomer_vomit|charger_grab|panic_start)["\']',
        fix_suggestion="Use correct L4D2 events: lunge_pounce, tongue_grab, player_now_it, charger_carry_start, create_panic_event",
    ),
    AntiPattern(
        id="L4D004",
        name="Deprecated L4D Function",
        description="Using deprecated Left4DHooks function",
        severity=Severity.MEDIUM,
        category=Category.L4D2_API,
        pattern=r'L4D_(?:GetSurvivorCount|GetInfectedCount)\s*\(',
        fix_suggestion="Check Left4DHooks documentation for current function names",
    ),
    AntiPattern(
        id="L4D005",
        name="Invalid Zombie Class Check",
        description="Hardcoded zombie class numbers instead of using defines",
        severity=Severity.LOW,
        category=Category.L4D2_API,
        pattern=r'(?:m_zombieClass|GetPlayerZombieClass)\s*[=<>!]+\s*[1-8](?!\d)',
        fix_suggestion="Use L4D2_ZOMBIECLASS_* defines for clarity and maintainability",
    ),

    # ==========================================================================
    # DEPRECATED (4 patterns)
    # ==========================================================================
    AntiPattern(
        id="DEP001",
        name="Old Style Declarations",
        description="Using old SourcePawn 1.6 declaration style",
        severity=Severity.LOW,
        category=Category.DEPRECATED,
        pattern=r'(?:new|decl)\s+(?:String|Float|Handle):',
        fix_suggestion="Use new syntax: char buffer[64], float value, Handle hndl",
    ),
    AntiPattern(
        id="DEP002",
        name="Deprecated CreateConVarEx",
        description="Using non-existent CreateConVarEx function",
        severity=Severity.MEDIUM,
        category=Category.DEPRECATED,
        pattern=r'CreateConVarEx\s*\(',
        fix_suggestion="Use standard CreateConVar() function",
    ),
    AntiPattern(
        id="DEP003",
        name="Old Timer Syntax",
        description="Using deprecated timer flag syntax",
        severity=Severity.LOW,
        category=Category.DEPRECATED,
        pattern=r'TIMER_FLAG_NO_MAPCHANGE',
        fix_suggestion="Use TIMER_HNDL_CLOSE or appropriate modern flags",
    ),
    AntiPattern(
        id="DEP004",
        name="Deprecated GetMaxClients",
        description="Using function instead of MaxClients constant",
        severity=Severity.INFO,
        category=Category.DEPRECATED,
        pattern=r'GetMaxClients\s*\(\s*\)',
        fix_suggestion="Use MaxClients constant directly for better performance",
    ),

    # ==========================================================================
    # STYLE (4 patterns)
    # ==========================================================================
    AntiPattern(
        id="STY001",
        name="Magic Numbers",
        description="Using hardcoded numbers without named constants",
        severity=Severity.LOW,
        category=Category.STYLE,
        pattern=r'(?:GetClientTeam|SetEntProp)\s*\([^)]*\)\s*==\s*[0-9]+',
        fix_suggestion="Define constants: #define TEAM_SURVIVOR 2, #define TEAM_INFECTED 3",
    ),
    AntiPattern(
        id="STY002",
        name="Missing Pragma",
        description="Missing recommended pragma directives",
        severity=Severity.INFO,
        category=Category.STYLE,
        pattern=r'^(?!.*#pragma\s+semicolon).*OnPluginStart',
        fix_suggestion="Add #pragma semicolon 1 and #pragma newdecls required at top of file",
    ),
    AntiPattern(
        id="STY003",
        name="Inconsistent Naming",
        description="Global variables not following g_ prefix convention",
        severity=Severity.INFO,
        category=Category.STYLE,
        pattern=r'^(?:int|float|bool|char|Handle)\s+(?!g_)\w+\s*[=;]',
        fix_suggestion="Prefix global variables with g_ for clarity: g_iCounter, g_hTimer",
    ),
    AntiPattern(
        id="STY004",
        name="Empty Catch Block",
        description="Exception handling that silently ignores errors",
        severity=Severity.LOW,
        category=Category.STYLE,
        pattern=r'catch\s*\([^)]*\)\s*\{\s*\}',
        fix_suggestion="Log errors or handle them appropriately, don't silently ignore",
    ),
]


# =============================================================================
# ANTI-PATTERN DETECTOR
# =============================================================================

class AntiPatternDetector:
    """Detects anti-patterns in SourcePawn code."""

    def __init__(self, patterns: List[AntiPattern] = None):
        self.patterns = patterns or ANTI_PATTERNS

    def scan(self, code: str, file_path: str = "code.sp") -> ScanResult:
        """Scan code for anti-patterns and return results."""
        warnings: List[Warning] = []
        lines = code.split('\n')
        total_lines = len(lines)

        for pattern in self.patterns:
            # Use custom checker if available
            if pattern.checker:
                findings = pattern.checker(code)
                for line_num, line_content in findings:
                    # Get context (3 lines before and after)
                    start = max(0, line_num - 4)
                    end = min(total_lines, line_num + 3)
                    context = '\n'.join(f"{i+start+1}: {lines[i+start]}" for i in range(end - start))

                    warnings.append(Warning(
                        antipattern_id=pattern.id,
                        name=pattern.name,
                        severity=pattern.severity,
                        category=pattern.category,
                        line_number=line_num,
                        line_content=line_content if isinstance(line_content, str) else lines[line_num-1].strip(),
                        description=pattern.description,
                        fix_suggestion=pattern.fix_suggestion,
                        context=context,
                    ))
            else:
                # Use regex pattern matching
                for i, line in enumerate(lines, 1):
                    if re.search(pattern.pattern, line, re.IGNORECASE):
                        # Get context
                        start = max(0, i - 4)
                        end = min(total_lines, i + 3)
                        context = '\n'.join(f"{j+start+1}: {lines[j+start]}" for j in range(end - start))

                        warnings.append(Warning(
                            antipattern_id=pattern.id,
                            name=pattern.name,
                            severity=pattern.severity,
                            category=pattern.category,
                            line_number=i,
                            line_content=line.strip(),
                            description=pattern.description,
                            fix_suggestion=pattern.fix_suggestion,
                            context=context,
                        ))

        # Calculate summary
        summary = {
            "total": len(warnings),
            "by_severity": {},
            "by_category": {},
        }

        for sev in Severity:
            count = sum(1 for w in warnings if w.severity == sev)
            if count > 0:
                summary["by_severity"][sev.value] = count

        for cat in Category:
            count = sum(1 for w in warnings if w.category == cat)
            if count > 0:
                summary["by_category"][cat.value] = count

        # Calculate score (100 = perfect, penalties for issues)
        score = 100.0
        for warning in warnings:
            if warning.severity == Severity.CRITICAL:
                score -= 15
            elif warning.severity == Severity.HIGH:
                score -= 10
            elif warning.severity == Severity.MEDIUM:
                score -= 5
            elif warning.severity == Severity.LOW:
                score -= 2
            elif warning.severity == Severity.INFO:
                score -= 1

        score = max(0, score)

        return ScanResult(
            file_path=file_path,
            total_lines=total_lines,
            warnings=warnings,
            summary=summary,
            score=score,
        )

    def scan_file(self, file_path: Path) -> ScanResult:
        """Scan a file for anti-patterns."""
        try:
            code = safe_read_text(str(file_path), PROJECT_ROOT)
            return self.scan(code, str(file_path))
        except Exception as e:
            return ScanResult(
                file_path=str(file_path),
                total_lines=0,
                warnings=[Warning(
                    antipattern_id="ERR001",
                    name="File Read Error",
                    severity=Severity.CRITICAL,
                    category=Category.CORRECTNESS,
                    line_number=0,
                    line_content="",
                    description=f"Could not read file: {e}",
                    fix_suggestion="Check file path and permissions",
                )],
                summary={"total": 1, "by_severity": {"critical": 1}},
                score=0,
            )

    def scan_directory(self, dir_path: Path) -> List[ScanResult]:
        """Scan all .sp files in a directory."""
        results = []
        for sp_file in dir_path.glob("**/*.sp"):
            result = self.scan_file(sp_file)
            results.append(result)
        return results


# =============================================================================
# CLI AND INTEGRATION
# =============================================================================

def print_warning(warning: Warning, verbose: bool = False) -> None:
    """Print a warning in human-readable format."""
    severity_colors = {
        Severity.CRITICAL: "\033[91m",  # Red
        Severity.HIGH: "\033[93m",      # Yellow
        Severity.MEDIUM: "\033[94m",    # Blue
        Severity.LOW: "\033[96m",       # Cyan
        Severity.INFO: "\033[90m",      # Gray
    }
    reset = "\033[0m"

    color = severity_colors.get(warning.severity, "")
    print(f"{color}[{warning.severity.value.upper()}]{reset} {warning.name} (Line {warning.line_number})")
    print(f"  ID: {warning.antipattern_id}")
    print(f"  {warning.description}")
    print(f"  Code: {warning.line_content[:80]}{'...' if len(warning.line_content) > 80 else ''}")
    print(f"  Fix: {warning.fix_suggestion}")

    if verbose and warning.context:
        print(f"  Context:")
        for line in warning.context.split('\n'):
            print(f"    {line}")
    print()


def print_summary(result: ScanResult) -> None:
    """Print scan summary."""
    print(f"\n{'='*60}")
    print(f"Anti-Pattern Scan Results: {result.file_path}")
    print(f"{'='*60}")
    print(f"Total Lines: {result.total_lines}")
    print(f"Warnings Found: {result.summary['total']}")
    print(f"Code Quality Score: {result.score:.1f}/100")

    if result.summary.get('by_severity'):
        print(f"\nBy Severity:")
        for sev, count in sorted(result.summary['by_severity'].items()):
            print(f"  {sev}: {count}")

    if result.summary.get('by_category'):
        print(f"\nBy Category:")
        for cat, count in sorted(result.summary['by_category'].items()):
            print(f"  {cat}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="L4D2/SourcePawn Anti-Pattern Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--code", type=str, help="SourcePawn code string to analyze")
    parser.add_argument("--file", type=str, help="Path to .sp file to analyze")
    parser.add_argument("--dir", type=str, help="Path to directory to analyze")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed context")
    parser.add_argument("--severity", type=str, choices=["critical", "high", "medium", "low", "info"],
                        help="Minimum severity to report")
    parser.add_argument("--category", type=str,
                        choices=["security", "memory", "performance", "correctness", "l4d2_api", "deprecated", "style"],
                        help="Filter by category")
    parser.add_argument("--list-patterns", action="store_true", help="List all anti-patterns")

    args = parser.parse_args()

    detector = AntiPatternDetector()

    # List patterns and exit
    if args.list_patterns:
        print(f"\n{'='*70}")
        print(f"L4D2/SourcePawn Anti-Patterns ({len(ANTI_PATTERNS)} patterns)")
        print(f"{'='*70}\n")

        for cat in Category:
            cat_patterns = [p for p in ANTI_PATTERNS if p.category == cat]
            if cat_patterns:
                print(f"\n{cat.value.upper()} ({len(cat_patterns)} patterns)")
                print("-" * 50)
                for p in cat_patterns:
                    print(f"  [{p.severity.value:8}] {p.id}: {p.name}")
                    print(f"             {p.description[:60]}...")
        return

    # Determine input
    if args.code:
        result = detector.scan(args.code, "inline_code")
        results = [result]
    elif args.file:
        file_path = safe_path(args.file, PROJECT_ROOT)
        result = detector.scan_file(file_path)
        results = [result]
    elif args.dir:
        dir_path = safe_path(args.dir, PROJECT_ROOT)
        results = detector.scan_directory(dir_path)
    else:
        parser.print_help()
        sys.exit(1)

    # Filter results
    if args.severity:
        min_severity = Severity(args.severity)
        severity_order = [Severity.INFO, Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        min_index = severity_order.index(min_severity)
        for result in results:
            result.warnings = [w for w in result.warnings
                               if severity_order.index(w.severity) >= min_index]

    if args.category:
        cat = Category(args.category)
        for result in results:
            result.warnings = [w for w in result.warnings if w.category == cat]

    # Output results
    if args.json or args.output:
        output_data = {
            "results": [r.to_dict() for r in results],
            "total_files": len(results),
            "total_warnings": sum(len(r.warnings) for r in results),
            "average_score": sum(r.score for r in results) / len(results) if results else 0,
        }

        if args.output:
            output_path = safe_path(args.output, PROJECT_ROOT, create_parents=True)
            safe_write_json(str(output_path), output_data, PROJECT_ROOT)
            print(f"Results saved to: {output_path}")
        else:
            print(json.dumps(output_data, indent=2))
    else:
        # Human-readable output
        for result in results:
            print_summary(result)

            if result.warnings:
                print(f"\n{'='*60}")
                print("Warnings:")
                print(f"{'='*60}\n")

                # Sort by severity
                sorted_warnings = sorted(
                    result.warnings,
                    key=lambda w: [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO].index(w.severity)
                )

                for warning in sorted_warnings:
                    print_warning(warning, args.verbose)


# =============================================================================
# BENCHMARK SUITE INTEGRATION
# =============================================================================

def evaluate_with_antipatterns(code: str, file_path: str = "code.sp") -> Dict[str, Any]:
    """
    Evaluate code for anti-patterns and return benchmark-compatible results.

    This function provides integration with the benchmark suite, returning
    a dictionary format compatible with benchmark test results.

    Args:
        code: SourcePawn code string to analyze
        file_path: Optional file path for reporting

    Returns:
        Dictionary with:
            - passed: bool - whether code passed anti-pattern check (score >= 70)
            - score: float - quality score (0-100)
            - antipattern_count: int - total number of warnings
            - critical_count: int - number of critical/high severity issues
            - warnings: list - list of warning dictionaries
            - summary: dict - summary statistics
    """
    detector = AntiPatternDetector()
    result = detector.scan(code, file_path)

    # Count critical/high severity issues
    critical_count = sum(
        1 for w in result.warnings
        if w.severity in (Severity.CRITICAL, Severity.HIGH)
    )

    # Determine pass/fail
    # Pass if score >= 70 AND no critical issues
    passed = result.score >= 70 and critical_count == 0

    return {
        "passed": passed,
        "score": result.score,
        "antipattern_count": len(result.warnings),
        "critical_count": critical_count,
        "warnings": [w.to_dict() for w in result.warnings],
        "summary": result.summary,
    }


def get_antipattern_score_penalty(code: str) -> float:
    """
    Get a score penalty based on anti-patterns found.

    For use in benchmark scoring where anti-patterns should reduce
    the overall score.

    Args:
        code: SourcePawn code string

    Returns:
        float: Penalty value (0.0 to 1.0) where 0 = no penalty, 1 = max penalty
    """
    detector = AntiPatternDetector()
    result = detector.scan(code)

    # Calculate penalty based on score
    # Score of 100 = 0 penalty, score of 0 = 1.0 penalty
    penalty = (100 - result.score) / 100

    return min(1.0, max(0.0, penalty))


def filter_antipatterns_by_category(
    warnings: List[Warning],
    categories: List[Category]
) -> List[Warning]:
    """
    Filter warnings to only include specified categories.

    Args:
        warnings: List of Warning objects
        categories: List of Category enums to include

    Returns:
        Filtered list of warnings
    """
    return [w for w in warnings if w.category in categories]


def get_l4d2_specific_issues(code: str) -> List[Warning]:
    """
    Get only L4D2-specific anti-pattern issues.

    Useful for evaluating model knowledge of L4D2 APIs specifically.

    Args:
        code: SourcePawn code string

    Returns:
        List of Warning objects for L4D2_API category only
    """
    detector = AntiPatternDetector()
    result = detector.scan(code)

    return [w for w in result.warnings if w.category == Category.L4D2_API]


def get_security_issues(code: str) -> List[Warning]:
    """
    Get only security-related anti-pattern issues.

    Args:
        code: SourcePawn code string

    Returns:
        List of Warning objects for SECURITY category only
    """
    detector = AntiPatternDetector()
    result = detector.scan(code)

    return [w for w in result.warnings if w.category == Category.SECURITY]


# Export key classes and functions for easy import
__all__ = [
    # Core classes
    "AntiPatternDetector",
    "AntiPattern",
    "Warning",
    "ScanResult",
    # Enums
    "Severity",
    "Category",
    # Data
    "ANTI_PATTERNS",
    # Integration functions
    "evaluate_with_antipatterns",
    "get_antipattern_score_penalty",
    "filter_antipatterns_by_category",
    "get_l4d2_specific_issues",
    "get_security_issues",
]


if __name__ == "__main__":
    main()
