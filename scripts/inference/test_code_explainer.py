#!/usr/bin/env python3
"""
Test suite for L4D2 Code Explainer

Tests parsing, explanation generation, and output formatting.
"""

import json
import sys
from pathlib import Path

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.code_explainer import (
    CodeParser,
    CodeExplainer,
    MarkdownFormatter,
    JSONFormatter,
    HTMLFormatter,
    MockExplainer
)

# =============================================================================
# Sample Code for Testing
# =============================================================================

SAMPLE_SOURCEPAWN = '''#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>

#define PLUGIN_VERSION "1.0"
#define BOOST_SPEED 1.4

public Plugin myinfo = {
    name = "Speed Boost Example",
    author = "Test Author",
    description = "Demonstrates speed boost on kill",
    version = PLUGIN_VERSION,
    url = ""
};

bool g_bHasBoost[MAXPLAYERS + 1];

public void OnPluginStart()
{
    HookEvent("player_death", Event_PlayerDeath);
    HookEvent("round_start", Event_RoundStart);
}

public void OnClientDisconnect(int client)
{
    g_bHasBoost[client] = false;
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
    for (int i = 1; i <= MaxClients; i++)
    {
        g_bHasBoost[i] = false;
    }
}

public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast)
{
    int victim = GetClientOfUserId(event.GetInt("userid"));
    int attacker = GetClientOfUserId(event.GetInt("attacker"));

    // Check if attacker is valid survivor
    if (attacker > 0 && IsClientInGame(attacker) && GetClientTeam(attacker) == 2)
    {
        // Check if victim is infected
        if (victim > 0 && IsClientInGame(victim) && GetClientTeam(victim) == 3)
        {
            GiveSpeedBoost(attacker);
        }
    }
}

void GiveSpeedBoost(int client)
{
    if (g_bHasBoost[client])
        return;

    g_bHasBoost[client] = true;
    SetEntPropFloat(client, Prop_Send, "m_flLaggedMovementValue", BOOST_SPEED);
    PrintToChat(client, "[Boost] Speed increased!");

    CreateTimer(5.0, Timer_ResetSpeed, GetClientUserId(client));
}

public Action Timer_ResetSpeed(Handle timer, int userid)
{
    int client = GetClientOfUserId(userid);

    if (client > 0 && IsClientInGame(client))
    {
        SetEntPropFloat(client, Prop_Send, "m_flLaggedMovementValue", 1.0);
        g_bHasBoost[client] = false;
        PrintToChat(client, "[Boost] Speed returned to normal.");
    }

    return Plugin_Continue;
}
'''

SAMPLE_VSCRIPT = '''// L4D2 Director Script Example
// Controls zombie spawning and difficulty

DirectorOptions <-
{
    CommonLimit = 30
    MobSpawnMinTime = 30
    MobSpawnMaxTime = 90
    MobMinSize = 15
    MobMaxSize = 30
    PreferredMobDirection = SPAWN_ANYWHERE
}

function OnGameEvent_round_start(params)
{
    printl("Round started - initializing director")

    local difficulty = Convars.GetFloat("z_difficulty")

    if (difficulty > 2.0)
    {
        DirectorOptions.CommonLimit <- 45
        DirectorOptions.MobMaxSize <- 40
    }
}

function OnGameEvent_player_death(params)
{
    local victim = params["userid"]
    local attacker = params["attacker"]

    printl("Player " + victim + " killed by " + attacker)
}

function Update()
{
    // Called every frame
    local time = Time()

    if (time > NextSpawnCheck)
    {
        CheckSpawnConditions()
        NextSpawnCheck = time + 5.0
    }
}

function CheckSpawnConditions()
{
    local survivors = GetSurvivorCount()

    if (survivors < 4)
    {
        DirectorOptions.CommonLimit <- 20
    }
}

NextSpawnCheck <- 0.0
'''


# =============================================================================
# Test Functions
# =============================================================================

def test_language_detection():
    """Test automatic language detection."""
    print("=" * 60)
    print("Testing Language Detection")
    print("=" * 60)

    sp_parser = CodeParser(SAMPLE_SOURCEPAWN, "auto")
    vs_parser = CodeParser(SAMPLE_VSCRIPT, "auto")

    print(f"SourcePawn sample detected as: {sp_parser.language}")
    print(f"VScript sample detected as: {vs_parser.language}")

    assert sp_parser.language == "sourcepawn", "Failed to detect SourcePawn"
    assert vs_parser.language == "vscript", "Failed to detect VScript"

    print("[PASS] Language detection works correctly\n")


def test_sourcepawn_parsing():
    """Test SourcePawn code parsing."""
    print("=" * 60)
    print("Testing SourcePawn Parsing")
    print("=" * 60)

    parser = CodeParser(SAMPLE_SOURCEPAWN, "sourcepawn")

    # Test includes
    includes = parser.parse_includes()
    print(f"Includes found: {includes}")
    assert "sourcemod" in includes
    assert "sdktools" in includes

    # Test plugin info
    plugin_info = parser.parse_plugin_info()
    print(f"Plugin info: {plugin_info}")
    assert plugin_info is not None
    assert plugin_info.get("name") == "Speed Boost Example"
    assert plugin_info.get("author") == "Test Author"

    # Test functions
    functions = parser.parse_functions()
    print(f"Functions found: {[f['name'] for f in functions]}")
    assert len(functions) >= 5
    func_names = [f['name'] for f in functions]
    assert "OnPluginStart" in func_names
    assert "GiveSpeedBoost" in func_names

    # Test variables
    variables = parser.parse_global_variables()
    print(f"Global variables found: {[v['name'] for v in variables]}")

    # Test hooks
    hooks = parser.parse_hooks()
    print(f"Hooks found: {[h['name'] for h in hooks]}")
    hook_names = [h['name'] for h in hooks]
    assert "player_death" in hook_names
    assert "round_start" in hook_names

    print("[PASS] SourcePawn parsing works correctly\n")


def test_vscript_parsing():
    """Test VScript code parsing."""
    print("=" * 60)
    print("Testing VScript Parsing")
    print("=" * 60)

    parser = CodeParser(SAMPLE_VSCRIPT, "vscript")

    # Test functions
    functions = parser.parse_functions()
    print(f"Functions found: {[f['name'] for f in functions]}")
    assert len(functions) >= 4
    func_names = [f['name'] for f in functions]
    assert "OnGameEvent_round_start" in func_names
    assert "Update" in func_names

    print("[PASS] VScript parsing works correctly\n")


def test_code_explainer():
    """Test the main CodeExplainer class."""
    print("=" * 60)
    print("Testing Code Explainer")
    print("=" * 60)

    # Use mock backend for testing
    explainer = CodeExplainer(backend="mock")

    # Test SourcePawn explanation
    explanation = explainer.explain(SAMPLE_SOURCEPAWN, file_name="test_plugin.sp")

    print(f"Language: {explanation.language}")
    print(f"File name: {explanation.file_name}")
    print(f"Summary: {explanation.summary[:100]}...")
    print(f"Functions: {len(explanation.functions)}")
    print(f"Variables: {len(explanation.variables)}")
    print(f"Hooks: {len(explanation.hooks)}")
    print(f"Issues: {len(explanation.issues)}")
    print(f"Lines: {len(explanation.lines)}")

    assert explanation.language == "sourcepawn"
    assert explanation.file_name == "test_plugin.sp"
    assert len(explanation.functions) > 0
    assert len(explanation.hooks) > 0

    print("[PASS] Code explainer works correctly\n")


def test_issue_detection():
    """Test potential issue detection."""
    print("=" * 60)
    print("Testing Issue Detection")
    print("=" * 60)

    code_with_issues = '''#include <sourcemod>

public void OnPluginStart()
{
    HookEvent("player_death", Event_Death);
    CreateTimer(5.0, Timer_Check, _, TIMER_REPEAT);
}

public void Event_Death(Event event, const char[] name, bool dontBroadcast)
{
    int client = GetClientOfUserId(event.GetInt("userid"));
    PrintToChat(client, "You died!");  // No validation!

    if (GetClientTeam(client) == 2)  // Magic number
    {
        // Do something
    }
}
'''

    explainer = CodeExplainer(backend="mock")
    explanation = explainer.explain(code_with_issues)

    print(f"Issues found: {len(explanation.issues)}")
    for issue in explanation.issues:
        print(f"  - Line {issue.line_number}: [{issue.severity}] {issue.issue_type}")
        print(f"    {issue.description}")

    # Should detect missing OnPluginEnd and magic numbers
    assert len(explanation.issues) > 0

    print("[PASS] Issue detection works correctly\n")


def test_markdown_formatter():
    """Test Markdown output formatting."""
    print("=" * 60)
    print("Testing Markdown Formatter")
    print("=" * 60)

    explainer = CodeExplainer(backend="mock")
    explanation = explainer.explain(SAMPLE_SOURCEPAWN, file_name="test_plugin.sp")

    formatter = MarkdownFormatter()
    output = formatter.format(explanation)

    print(f"Output length: {len(output)} characters")
    print("First 500 characters:")
    print(output[:500])
    print("...")

    # Check for expected sections
    assert "# test_plugin.sp" in output
    assert "## Summary" in output
    assert "## Functions" in output
    assert "sourcepawn" in output.lower()

    print("[PASS] Markdown formatter works correctly\n")


def test_json_formatter():
    """Test JSON output formatting."""
    print("=" * 60)
    print("Testing JSON Formatter")
    print("=" * 60)

    explainer = CodeExplainer(backend="mock")
    explanation = explainer.explain(SAMPLE_SOURCEPAWN, file_name="test_plugin.sp")

    formatter = JSONFormatter()
    output = formatter.format(explanation)

    print(f"Output length: {len(output)} characters")

    # Verify it's valid JSON
    data = json.loads(output)
    print(f"Top-level keys: {list(data.keys())}")

    assert "language" in data
    assert "functions" in data
    assert "hooks" in data
    assert data["language"] == "sourcepawn"

    print("[PASS] JSON formatter works correctly\n")


def test_html_formatter():
    """Test HTML output formatting."""
    print("=" * 60)
    print("Testing HTML Formatter")
    print("=" * 60)

    explainer = CodeExplainer(backend="mock")
    explanation = explainer.explain(SAMPLE_SOURCEPAWN, file_name="test_plugin.sp")

    formatter = HTMLFormatter()
    output = formatter.format(explanation)

    print(f"Output length: {len(output)} characters")

    # Check for expected HTML elements
    assert "<!DOCTYPE html>" in output
    assert "<title>" in output
    assert "test_plugin.sp" in output
    assert "</html>" in output

    print("[PASS] HTML formatter works correctly\n")


def test_vscript_explanation():
    """Test VScript code explanation."""
    print("=" * 60)
    print("Testing VScript Explanation")
    print("=" * 60)

    explainer = CodeExplainer(backend="mock")
    explanation = explainer.explain(SAMPLE_VSCRIPT, file_name="director.nut")

    print(f"Language: {explanation.language}")
    print(f"Functions: {len(explanation.functions)}")

    assert explanation.language == "vscript"
    assert len(explanation.functions) > 0

    formatter = MarkdownFormatter()
    output = formatter.format(explanation)
    print("\nMarkdown output preview:")
    print(output[:800])

    print("[PASS] VScript explanation works correctly\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("L4D2 CODE EXPLAINER TEST SUITE")
    print("=" * 60 + "\n")

    tests = [
        test_language_detection,
        test_sourcepawn_parsing,
        test_vscript_parsing,
        test_code_explainer,
        test_issue_detection,
        test_markdown_formatter,
        test_json_formatter,
        test_html_formatter,
        test_vscript_explanation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test.__name__}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
