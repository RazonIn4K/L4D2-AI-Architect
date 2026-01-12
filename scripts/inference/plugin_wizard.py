#!/usr/bin/env python3
"""
L4D2 Plugin Generator Wizard

Interactive CLI wizard for generating complete SourcePawn plugins for Left 4 Dead 2.
Supports both interactive and batch modes (from JSON spec).

Features:
- Guided plugin configuration
- Automatic code generation
- Code validation
- Template-based scaffolding
- L4D2-specific includes and patterns

Usage:
    # Interactive mode
    python plugin_wizard.py interactive

    # Batch mode from JSON spec
    python plugin_wizard.py batch --spec plugin_spec.json

    # Quick generate with minimal prompts
    python plugin_wizard.py quick --name "Tank Spawner" --author "Developer"
"""

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.security import safe_write_text, safe_read_json, safe_path

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ==============================================================================
# Enums and Constants
# ==============================================================================


class PluginCategory(Enum):
    """Plugin functionality categories."""
    TANK_SPAWNER = "tank_spawner"
    HEAL_SYSTEM = "heal_system"
    SPECIAL_INFECTED = "special_infected"
    WEAPONS = "weapons"
    ITEMS = "items"
    PLAYER_STATS = "player_stats"
    ADMIN_TOOLS = "admin_tools"
    GAMEPLAY = "gameplay"
    HUD = "hud"
    CUSTOM = "custom"


class IncludeType(Enum):
    """Available SourceMod includes for L4D2."""
    SOURCEMOD = "sourcemod"
    SDKTOOLS = "sdktools"
    SDKHOOKS = "sdkhooks"
    LEFT4DHOOKS = "left4dhooks"
    L4D2_DIRECT = "l4d2_direct"
    CLIENTPREFS = "clientprefs"
    MENUS = "menus"
    ADMINMENU = "adminmenu"
    COLORS = "colors"
    AUTOEXECCONFIG = "autoexecconfig"


# L4D2 Events commonly used
L4D2_EVENTS = {
    "player_spawn": "Player spawned (any team)",
    "player_death": "Player died",
    "player_hurt": "Player took damage",
    "player_incapacitated": "Survivor incapacitated",
    "revive_success": "Survivor revived",
    "heal_success": "Heal completed",
    "round_start": "Round started",
    "round_end": "Round ended",
    "infected_death": "Common infected killed",
    "tank_spawn": "Tank spawned",
    "tank_killed": "Tank killed",
    "witch_spawn": "Witch spawned",
    "witch_killed": "Witch killed",
    "charger_carry_start": "Charger grabbed survivor",
    "charger_carry_end": "Charger released survivor",
    "smoker_tongue_grab": "Smoker grabbed survivor",
    "hunter_pounced": "Hunter pounced survivor",
    "jockey_ride": "Jockey riding survivor",
    "player_now_it": "Player tagged (infected)",
    "weapon_fire": "Weapon fired",
    "weapon_reload": "Weapon reloaded",
    "item_pickup": "Item picked up",
    "door_open": "Door opened",
    "safe_room_entered": "Safe room entered",
}

# Common admin flags
ADMIN_FLAGS = {
    "ADMFLAG_GENERIC": "Generic admin (flag a)",
    "ADMFLAG_KICK": "Kick players (flag c)",
    "ADMFLAG_BAN": "Ban players (flag d)",
    "ADMFLAG_SLAY": "Slay players (flag e)",
    "ADMFLAG_CHEATS": "Cheats (flag n)",
    "ADMFLAG_ROOT": "Root access (flag z)",
    "ADMFLAG_CUSTOM1": "Custom flag 1 (flag o)",
}


# ==============================================================================
# Data Classes
# ==============================================================================


@dataclass
class ConVarSpec:
    """Specification for a ConVar."""
    name: str
    default: str
    description: str
    min_value: Optional[str] = None
    max_value: Optional[str] = None
    flags: str = "FCVAR_NOTIFY"

    def to_code(self) -> str:
        """Generate ConVar creation code."""
        bounds = ""
        if self.min_value is not None:
            bounds += f", true, {self.min_value}"
        if self.max_value is not None:
            if self.min_value is None:
                bounds += ", false, 0.0"
            bounds += f", true, {self.max_value}"

        return f'g_cv{self._var_name()} = CreateConVar("{self.name}", "{self.default}", "{self.description}", {self.flags}{bounds});'

    def _var_name(self) -> str:
        """Convert name to variable name."""
        return "".join(word.capitalize() for word in self.name.split("_"))


@dataclass
class CommandSpec:
    """Specification for a command."""
    name: str
    description: str
    callback: str
    admin_flag: Optional[str] = None
    is_admin: bool = False

    def to_code(self) -> str:
        """Generate command registration code."""
        if self.is_admin and self.admin_flag:
            return f'RegAdminCmd("{self.name}", {self.callback}, {self.admin_flag}, "{self.description}");'
        else:
            return f'RegConsoleCmd("{self.name}", {self.callback}, "{self.description}");'


@dataclass
class EventSpec:
    """Specification for an event hook."""
    event_name: str
    callback: str
    mode: str = "EventHookMode_Post"

    def to_code(self) -> str:
        """Generate event hook code."""
        return f'HookEvent("{self.event_name}", {self.callback}, {self.mode});'


@dataclass
class PluginSpec:
    """Complete specification for a plugin."""
    name: str
    author: str
    description: str
    version: str = "1.0.0"
    url: str = ""
    category: PluginCategory = PluginCategory.CUSTOM
    includes: List[str] = field(default_factory=lambda: ["sourcemod", "sdktools"])
    convars: List[ConVarSpec] = field(default_factory=list)
    commands: List[CommandSpec] = field(default_factory=list)
    events: List[EventSpec] = field(default_factory=list)
    custom_functions: List[str] = field(default_factory=list)
    team_checks: bool = True
    client_validation: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["category"] = self.category.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginSpec":
        """Create from dictionary."""
        data = data.copy()
        if "category" in data:
            data["category"] = PluginCategory(data["category"])
        if "convars" in data:
            data["convars"] = [ConVarSpec(**cv) for cv in data["convars"]]
        if "commands" in data:
            data["commands"] = [CommandSpec(**cmd) for cmd in data["commands"]]
        if "events" in data:
            data["events"] = [EventSpec(**evt) for evt in data["events"]]
        return cls(**data)


@dataclass
class ValidationResult:
    """Result of code validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


# ==============================================================================
# Code Generator
# ==============================================================================


class PluginGenerator:
    """Generates SourcePawn plugin code from specifications."""

    def __init__(self, spec: PluginSpec):
        self.spec = spec

    def generate(self) -> str:
        """Generate complete plugin code."""
        sections = [
            self._generate_header(),
            self._generate_includes(),
            self._generate_pragmas(),
            self._generate_defines(),
            self._generate_globals(),
            self._generate_plugin_info(),
            self._generate_on_plugin_start(),
            self._generate_on_map_start(),
            self._generate_command_callbacks(),
            self._generate_event_callbacks(),
            self._generate_helper_functions(),
        ]

        # Filter empty sections and join
        code = "\n\n".join(section for section in sections if section.strip())
        return code

    def _generate_header(self) -> str:
        """Generate file header comment."""
        timestamp = datetime.now().strftime("%Y-%m-%d")
        return f'''/**
 * {self.spec.name}
 *
 * {self.spec.description}
 *
 * @author {self.spec.author}
 * @version {self.spec.version}
 * @date {timestamp}
 */'''

    def _generate_includes(self) -> str:
        """Generate include statements."""
        lines = []
        for inc in self.spec.includes:
            lines.append(f"#include <{inc}>")
        return "\n".join(lines)

    def _generate_pragmas(self) -> str:
        """Generate pragma directives."""
        return """#pragma newdecls required
#pragma semicolon 1"""

    def _generate_defines(self) -> str:
        """Generate define statements."""
        lines = [f'#define PLUGIN_VERSION "{self.spec.version}"']

        if self.spec.team_checks:
            lines.extend([
                "",
                "#define TEAM_SPECTATOR 1",
                "#define TEAM_SURVIVOR 2",
                "#define TEAM_INFECTED 3",
            ])

        return "\n".join(lines)

    def _generate_globals(self) -> str:
        """Generate global variables."""
        lines = []

        # ConVar handles
        if self.spec.convars:
            lines.append("// ConVars")
            for cv in self.spec.convars:
                var_name = "".join(word.capitalize() for word in cv.name.split("_"))
                lines.append(f"ConVar g_cv{var_name};")

        return "\n".join(lines)

    def _generate_plugin_info(self) -> str:
        """Generate plugin info block."""
        return f'''public Plugin myinfo = {{
    name = "{self.spec.name}",
    author = "{self.spec.author}",
    description = "{self.spec.description}",
    version = PLUGIN_VERSION,
    url = "{self.spec.url}"
}};'''

    def _generate_on_plugin_start(self) -> str:
        """Generate OnPluginStart function."""
        lines = ["public void OnPluginStart() {"]

        # Version ConVar
        clean_name = re.sub(r"[^a-z0-9_]", "_", self.spec.name.lower())
        lines.append(f'    CreateConVar("sm_{clean_name}_version", PLUGIN_VERSION, "Plugin version", FCVAR_NOTIFY | FCVAR_DONTRECORD);')

        # Custom ConVars
        if self.spec.convars:
            lines.append("")
            lines.append("    // Configuration ConVars")
            for cv in self.spec.convars:
                lines.append(f"    {cv.to_code()}")

        # Commands
        if self.spec.commands:
            lines.append("")
            lines.append("    // Commands")
            for cmd in self.spec.commands:
                lines.append(f"    {cmd.to_code()}")

        # Event hooks
        if self.spec.events:
            lines.append("")
            lines.append("    // Event hooks")
            for evt in self.spec.events:
                lines.append(f"    {evt.to_code()}")

        # AutoExecConfig
        lines.append("")
        lines.append(f'    AutoExecConfig(true, "{clean_name}");')

        lines.append("}")
        return "\n".join(lines)

    def _generate_on_map_start(self) -> str:
        """Generate OnMapStart function."""
        return """public void OnMapStart() {
    // Map-specific initialization
}"""

    def _generate_command_callbacks(self) -> str:
        """Generate command callback functions."""
        if not self.spec.commands:
            return ""

        sections = ["// Command Callbacks"]
        for cmd in self.spec.commands:
            callback = self._generate_command_callback(cmd)
            sections.append(callback)

        return "\n\n".join(sections)

    def _generate_command_callback(self, cmd: CommandSpec) -> str:
        """Generate a single command callback."""
        lines = [f"public Action {cmd.callback}(int client, int args) {{"]

        if self.spec.client_validation:
            lines.append("    if (client == 0) {")
            lines.append('        ReplyToCommand(client, "[SM] This command can only be used in-game.");')
            lines.append("        return Plugin_Handled;")
            lines.append("    }")
            lines.append("")

        lines.append(f'    // TODO: Implement {cmd.name} logic')
        lines.append(f'    ReplyToCommand(client, "[SM] Command {cmd.name} executed.");')
        lines.append("    return Plugin_Handled;")
        lines.append("}")

        return "\n".join(lines)

    def _generate_event_callbacks(self) -> str:
        """Generate event callback functions."""
        if not self.spec.events:
            return ""

        sections = ["// Event Callbacks"]
        for evt in self.spec.events:
            callback = self._generate_event_callback(evt)
            sections.append(callback)

        return "\n\n".join(sections)

    def _generate_event_callback(self, evt: EventSpec) -> str:
        """Generate a single event callback."""
        lines = [f'public void {evt.callback}(Event event, const char[] name, bool dontBroadcast) {{']

        # Common event data extraction based on event type
        if "player" in evt.event_name or "survivor" in evt.event_name:
            lines.append('    int client = GetClientOfUserId(event.GetInt("userid"));')
            lines.append("")
            if self.spec.client_validation:
                lines.append("    if (!IsValidClient(client)) return;")
                lines.append("")

        if "death" in evt.event_name or "hurt" in evt.event_name:
            lines.append('    int attacker = GetClientOfUserId(event.GetInt("attacker"));')

        if "heal" in evt.event_name or "revive" in evt.event_name:
            lines.append('    int subject = GetClientOfUserId(event.GetInt("subject"));')

        lines.append(f"    // TODO: Implement {evt.event_name} handler")
        lines.append("}")

        return "\n".join(lines)

    def _generate_helper_functions(self) -> str:
        """Generate helper functions."""
        sections = ["// Helper Functions"]

        if self.spec.client_validation:
            sections.append("""bool IsValidClient(int client) {
    return client > 0 && client <= MaxClients && IsClientInGame(client);
}""")

        if self.spec.team_checks:
            sections.append("""bool IsValidSurvivor(int client) {
    return IsValidClient(client) && GetClientTeam(client) == TEAM_SURVIVOR && IsPlayerAlive(client);
}

bool IsValidInfected(int client) {
    return IsValidClient(client) && GetClientTeam(client) == TEAM_INFECTED && IsPlayerAlive(client);
}""")

        # Category-specific helpers
        if self.spec.category == PluginCategory.TANK_SPAWNER:
            sections.append(self._get_tank_helper())
        elif self.spec.category == PluginCategory.HEAL_SYSTEM:
            sections.append(self._get_heal_helper())
        elif self.spec.category == PluginCategory.SPECIAL_INFECTED:
            sections.append(self._get_special_infected_helper())

        # Custom functions
        for func in self.spec.custom_functions:
            sections.append(func)

        return "\n\n".join(sections)

    def _get_tank_helper(self) -> str:
        """Get tank-related helper functions."""
        return """bool IsTank(int client) {
    if (!IsValidInfected(client)) return false;

    char className[32];
    GetEntityClassname(client, className, sizeof(className));
    return StrEqual(className, "player") && GetEntProp(client, Prop_Send, "m_zombieClass") == 8;
}

void SpawnTank(float pos[3]) {
    int bot = CreateFakeClient("Tank");
    if (bot > 0) {
        ChangeClientTeam(bot, TEAM_INFECTED);
        // Use L4D2 natives if available
        // L4D_SetClass(bot, 8); // Tank class
        TeleportEntity(bot, pos, NULL_VECTOR, NULL_VECTOR);
    }
}"""

    def _get_heal_helper(self) -> str:
        """Get heal-related helper functions."""
        return """void HealPlayer(int client, int amount) {
    if (!IsValidSurvivor(client)) return;

    int health = GetClientHealth(client);
    int maxHealth = GetEntProp(client, Prop_Data, "m_iMaxHealth");
    int newHealth = health + amount;

    if (newHealth > maxHealth) {
        newHealth = maxHealth;
    }

    SetEntityHealth(client, newHealth);
}

void RevivePlayer(int client, int reviver) {
    if (!IsValidClient(client)) return;
    if (!IsPlayerAlive(client)) return;

    // Check if incapacitated
    if (GetEntProp(client, Prop_Send, "m_isIncapacitated")) {
        // Use SDKCall or game event to properly revive
        SetEntProp(client, Prop_Send, "m_isIncapacitated", 0);
        SetEntityHealth(client, 30);
    }
}"""

    def _get_special_infected_helper(self) -> str:
        """Get special infected helper functions."""
        return """// Zombie class IDs for L4D2
enum ZombieClass {
    ZC_Smoker = 1,
    ZC_Boomer = 2,
    ZC_Hunter = 3,
    ZC_Spitter = 4,
    ZC_Jockey = 5,
    ZC_Charger = 6,
    ZC_Witch = 7,
    ZC_Tank = 8
};

int GetZombieClass(int client) {
    if (!IsValidInfected(client)) return 0;
    return GetEntProp(client, Prop_Send, "m_zombieClass");
}

bool IsSpecialInfected(int client) {
    if (!IsValidInfected(client)) return false;
    int zombieClass = GetZombieClass(client);
    return zombieClass >= 1 && zombieClass <= 6;
}"""


# ==============================================================================
# Code Validator
# ==============================================================================


class CodeValidator:
    """Validates generated SourcePawn code."""

    # Required patterns
    REQUIRED_PATTERNS = {
        "include_sourcemod": (r'#include\s*<sourcemod>', "Missing #include <sourcemod>"),
        "pragma_newdecls": (r'#pragma\s+newdecls\s+required', "Missing #pragma newdecls required"),
        "pragma_semicolon": (r'#pragma\s+semicolon\s+1', "Missing #pragma semicolon 1"),
        "plugin_info": (r'public\s+Plugin\s+myinfo\s*=', "Missing plugin info block"),
        "on_plugin_start": (r'public\s+void\s+OnPluginStart\s*\(', "Missing OnPluginStart function"),
    }

    # Warning patterns
    WARNING_PATTERNS = {
        "hardcoded_paths": (r'[A-Za-z]:\\|/home/', "Hardcoded file paths detected"),
        "magic_numbers": (r'(?<!#define\s)\b\d{3,}\b(?!\s*[;,\)])', "Magic numbers without defines"),
        "empty_callbacks": (r'public\s+\w+\s+\w+\([^)]*\)\s*\{\s*\}', "Empty callback functions"),
    }

    # Suggestion patterns
    SUGGESTION_PATTERNS = {
        "missing_autoexec": (r'AutoExecConfig', "Consider using AutoExecConfig for ConVars", True),
        "client_validation": (r'IsValidClient|IsClientInGame', "Ensure client validation before operations", True),
    }

    @classmethod
    def validate(cls, code: str) -> ValidationResult:
        """Validate the generated code."""
        errors = []
        warnings = []
        suggestions = []

        # Check required patterns
        for name, (pattern, message) in cls.REQUIRED_PATTERNS.items():
            if not re.search(pattern, code):
                errors.append(message)

        # Check warning patterns
        for name, (pattern, message) in cls.WARNING_PATTERNS.items():
            if re.search(pattern, code):
                warnings.append(message)

        # Check suggestion patterns
        for name, (pattern, message, should_have) in cls.SUGGESTION_PATTERNS.items():
            has_pattern = bool(re.search(pattern, code))
            if should_have and not has_pattern:
                suggestions.append(message)

        # Check for balanced braces
        if code.count("{") != code.count("}"):
            errors.append("Unbalanced braces")

        # Check for balanced parentheses
        if code.count("(") != code.count(")"):
            errors.append("Unbalanced parentheses")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
        )


# ==============================================================================
# Interactive CLI Wizard
# ==============================================================================


class PluginWizard:
    """Interactive CLI wizard for plugin generation."""

    def __init__(self):
        self.spec: Optional[PluginSpec] = None

    def prompt(self, message: str, default: str = "", validator: Optional[Callable[[str], bool]] = None) -> str:
        """Prompt for input with optional default and validation."""
        while True:
            if default:
                prompt_text = f"{message} [{default}]: "
            else:
                prompt_text = f"{message}: "

            try:
                value = input(prompt_text).strip()
            except (KeyboardInterrupt, EOFError):
                print("\nWizard cancelled.")
                sys.exit(0)

            if not value and default:
                value = default

            if not value:
                print("This field is required.")
                continue

            if validator and not validator(value):
                continue

            return value

    def prompt_yes_no(self, message: str, default: bool = True) -> bool:
        """Prompt for yes/no answer."""
        default_str = "Y/n" if default else "y/N"
        response = self.prompt(f"{message} ({default_str})", "y" if default else "n")
        return response.lower() in ("y", "yes", "1", "true")

    def prompt_choice(self, message: str, choices: List[str], default: int = 0) -> str:
        """Prompt for a choice from a list."""
        print(f"\n{message}")
        for i, choice in enumerate(choices):
            marker = "*" if i == default else " "
            print(f"  {marker}[{i + 1}] {choice}")

        while True:
            response = self.prompt("Select option", str(default + 1))
            try:
                idx = int(response) - 1
                if 0 <= idx < len(choices):
                    return choices[idx]
            except ValueError:
                pass
            print(f"Please enter a number between 1 and {len(choices)}")

    def prompt_multi_choice(self, message: str, choices: Dict[str, str]) -> List[str]:
        """Prompt for multiple selections from a list."""
        print(f"\n{message}")
        choice_list = list(choices.keys())
        for i, (key, desc) in enumerate(choices.items()):
            print(f"  [{i + 1}] {key}: {desc}")

        print("\nEnter comma-separated numbers (e.g., 1,3,5) or 'all':")
        while True:
            response = self.prompt("Select options", "1")

            if response.lower() == "all":
                return choice_list

            try:
                indices = [int(x.strip()) - 1 for x in response.split(",")]
                selected = [choice_list[i] for i in indices if 0 <= i < len(choice_list)]
                if selected:
                    return selected
            except (ValueError, IndexError):
                pass

            print("Invalid selection. Please enter comma-separated numbers.")

    def run_interactive(self) -> PluginSpec:
        """Run the interactive wizard."""
        print("\n" + "=" * 60)
        print("       L4D2 Plugin Generator Wizard")
        print("=" * 60)
        print("\nThis wizard will guide you through creating a SourcePawn plugin.")
        print("Press Ctrl+C at any time to cancel.\n")

        # Basic information
        print("--- Basic Information ---")
        name = self.prompt("Plugin name", "My L4D2 Plugin")
        author = self.prompt("Author name")
        description = self.prompt("Plugin description", f"{name} for Left 4 Dead 2")
        version = self.prompt("Version", "1.0.0")
        url = self.prompt("URL (optional)", "")

        # Category selection
        category_choices = {cat.value: cat.name.replace("_", " ").title() for cat in PluginCategory}
        category_name = self.prompt_choice(
            "Select plugin category:",
            list(category_choices.values()),
            default=9  # Custom
        )
        category = next(cat for cat in PluginCategory if cat.name.replace("_", " ").title() == category_name)

        # Includes
        print("\n--- Required Includes ---")
        include_choices = {inc.value: inc.name for inc in IncludeType}
        selected_includes = self.prompt_multi_choice(
            "Select required includes:",
            include_choices
        )

        # ConVars
        print("\n--- ConVars (Configuration Variables) ---")
        convars = []
        if self.prompt_yes_no("Add ConVars for configuration?"):
            while True:
                cv_name = self.prompt("ConVar name (e.g., sm_plugin_enabled)", "")
                if not cv_name:
                    break
                cv_default = self.prompt("Default value", "1")
                cv_desc = self.prompt("Description", "Configuration option")
                cv_min = self.prompt("Minimum value (optional, press Enter to skip)", "")
                cv_max = self.prompt("Maximum value (optional, press Enter to skip)", "")

                convars.append(ConVarSpec(
                    name=cv_name,
                    default=cv_default,
                    description=cv_desc,
                    min_value=cv_min if cv_min else None,
                    max_value=cv_max if cv_max else None,
                ))

                if not self.prompt_yes_no("Add another ConVar?", default=False):
                    break

        # Commands
        print("\n--- Commands ---")
        commands = []
        if self.prompt_yes_no("Add commands?"):
            while True:
                cmd_name = self.prompt("Command name (e.g., sm_spawn_tank)", "")
                if not cmd_name:
                    break
                cmd_desc = self.prompt("Description", "Custom command")
                cmd_is_admin = self.prompt_yes_no("Require admin access?")
                cmd_flag = None
                if cmd_is_admin:
                    flag_choices = list(ADMIN_FLAGS.keys())
                    cmd_flag = self.prompt_choice("Select admin flag:", flag_choices)

                callback = "Command_" + "".join(word.capitalize() for word in cmd_name.replace("sm_", "").split("_"))
                commands.append(CommandSpec(
                    name=cmd_name,
                    description=cmd_desc,
                    callback=callback,
                    admin_flag=cmd_flag,
                    is_admin=cmd_is_admin,
                ))

                if not self.prompt_yes_no("Add another command?", default=False):
                    break

        # Events
        print("\n--- Event Hooks ---")
        events = []
        if self.prompt_yes_no("Hook game events?"):
            event_choices = self.prompt_multi_choice("Select events to hook:", L4D2_EVENTS)
            for event_name in event_choices:
                callback = "Event_" + "".join(word.capitalize() for word in event_name.split("_"))
                events.append(EventSpec(event_name=event_name, callback=callback))

        # Additional options
        print("\n--- Additional Options ---")
        team_checks = self.prompt_yes_no("Include team check helpers (survivor/infected)?")
        client_validation = self.prompt_yes_no("Include client validation helpers?")

        # Create specification
        self.spec = PluginSpec(
            name=name,
            author=author,
            description=description,
            version=version,
            url=url,
            category=category,
            includes=selected_includes,
            convars=convars,
            commands=commands,
            events=events,
            team_checks=team_checks,
            client_validation=client_validation,
        )

        return self.spec


# ==============================================================================
# Main Functions
# ==============================================================================


def generate_plugin(spec: PluginSpec) -> Tuple[str, ValidationResult]:
    """Generate plugin code from specification."""
    generator = PluginGenerator(spec)
    code = generator.generate()
    validation = CodeValidator.validate(code)
    return code, validation


def save_plugin(code: str, output_path: str) -> Path:
    """Save plugin code to file."""
    return safe_write_text(output_path, code, PROJECT_ROOT)


def run_interactive_mode(args: argparse.Namespace) -> int:
    """Run interactive wizard mode."""
    wizard = PluginWizard()

    try:
        spec = wizard.run_interactive()
    except (KeyboardInterrupt, EOFError):
        print("\nWizard cancelled.")
        return 1

    # Generate code
    print("\n" + "=" * 60)
    print("Generating plugin...")
    code, validation = generate_plugin(spec)

    # Show validation results
    print("\n--- Validation Results ---")
    if validation.is_valid:
        print("Status: VALID")
    else:
        print("Status: INVALID")
        for error in validation.errors:
            print(f"  ERROR: {error}")

    for warning in validation.warnings:
        print(f"  WARNING: {warning}")

    for suggestion in validation.suggestions:
        print(f"  SUGGESTION: {suggestion}")

    # Preview or save
    print("\n--- Generated Code Preview ---")
    preview_lines = code.split("\n")[:30]
    print("\n".join(preview_lines))
    if len(code.split("\n")) > 30:
        print(f"... ({len(code.split(chr(10))) - 30} more lines)")

    # Ask to save
    if wizard.prompt_yes_no("\nSave plugin to file?"):
        clean_name = re.sub(r"[^a-z0-9_]", "_", spec.name.lower())
        default_output = args.output or f"{clean_name}.sp"
        output_path = wizard.prompt("Output file path", default_output)

        try:
            saved_path = save_plugin(code, output_path)
            print(f"\nPlugin saved to: {saved_path}")

            # Optionally save spec
            if wizard.prompt_yes_no("Save specification for future use?", default=False):
                spec_path = str(saved_path).replace(".sp", "_spec.json")
                safe_write_text(spec_path, json.dumps(spec.to_dict(), indent=2), PROJECT_ROOT)
                print(f"Specification saved to: {spec_path}")

        except ValueError as e:
            print(f"Error saving file: {e}")
            return 1

    return 0


def run_batch_mode(args: argparse.Namespace) -> int:
    """Run batch mode from JSON specification."""
    if not args.spec:
        print("Error: --spec is required for batch mode")
        return 1

    try:
        spec_data = safe_read_json(args.spec, PROJECT_ROOT)
        spec = PluginSpec.from_dict(spec_data)
    except FileNotFoundError:
        print(f"Error: Specification file not found: {args.spec}")
        return 1
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Error: Invalid specification file: {e}")
        return 1

    # Generate code
    print(f"Generating plugin from: {args.spec}")
    code, validation = generate_plugin(spec)

    # Show validation
    if not validation.is_valid:
        print("Validation errors:")
        for error in validation.errors:
            print(f"  - {error}")

    for warning in validation.warnings:
        print(f"Warning: {warning}")

    # Save output
    if args.output:
        output_path = args.output
    else:
        clean_name = re.sub(r"[^a-z0-9_]", "_", spec.name.lower())
        output_path = f"{clean_name}.sp"

    try:
        saved_path = save_plugin(code, output_path)
        print(f"Plugin saved to: {saved_path}")
    except ValueError as e:
        print(f"Error saving file: {e}")
        return 1

    return 0 if validation.is_valid else 1


def run_quick_mode(args: argparse.Namespace) -> int:
    """Run quick generation mode with minimal input."""
    if not args.name:
        print("Error: --name is required for quick mode")
        return 1

    if not args.author:
        print("Error: --author is required for quick mode")
        return 1

    # Create minimal spec
    spec = PluginSpec(
        name=args.name,
        author=args.author,
        description=args.description or f"{args.name} for Left 4 Dead 2",
        version=args.version or "1.0.0",
        category=PluginCategory(args.category) if args.category else PluginCategory.CUSTOM,
        includes=["sourcemod", "sdktools"],
        team_checks=True,
        client_validation=True,
    )

    # Generate code
    print(f"Generating quick plugin: {spec.name}")
    code, validation = generate_plugin(spec)

    # Show validation
    if not validation.is_valid:
        for error in validation.errors:
            print(f"Error: {error}")

    # Save output
    if args.output:
        output_path = args.output
    else:
        clean_name = re.sub(r"[^a-z0-9_]", "_", spec.name.lower())
        output_path = f"{clean_name}.sp"

    try:
        saved_path = save_plugin(code, output_path)
        print(f"Plugin saved to: {saved_path}")
    except ValueError as e:
        print(f"Error saving file: {e}")
        return 1

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="L4D2 Plugin Generator Wizard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive wizard
  python plugin_wizard.py interactive

  # Generate from JSON specification
  python plugin_wizard.py batch --spec my_plugin_spec.json --output my_plugin.sp

  # Quick generate with minimal options
  python plugin_wizard.py quick --name "Tank Spawner" --author "Developer"
        """
    )

    subparsers = parser.add_subparsers(dest="mode", help="Generation mode")

    # Interactive mode
    interactive_parser = subparsers.add_parser("interactive", help="Interactive wizard mode")
    interactive_parser.add_argument("--output", "-o", help="Output file path")

    # Batch mode
    batch_parser = subparsers.add_parser("batch", help="Batch mode from JSON spec")
    batch_parser.add_argument("--spec", "-s", required=True, help="JSON specification file")
    batch_parser.add_argument("--output", "-o", help="Output file path")

    # Quick mode
    quick_parser = subparsers.add_parser("quick", help="Quick generation mode")
    quick_parser.add_argument("--name", "-n", required=True, help="Plugin name")
    quick_parser.add_argument("--author", "-a", required=True, help="Author name")
    quick_parser.add_argument("--description", "-d", help="Plugin description")
    quick_parser.add_argument("--version", "-v", default="1.0.0", help="Plugin version")
    quick_parser.add_argument("--category", "-c", choices=[c.value for c in PluginCategory],
                              default="custom", help="Plugin category")
    quick_parser.add_argument("--output", "-o", help="Output file path")

    args = parser.parse_args()

    if args.mode == "interactive":
        return run_interactive_mode(args)
    elif args.mode == "batch":
        return run_batch_mode(args)
    elif args.mode == "quick":
        return run_quick_mode(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
