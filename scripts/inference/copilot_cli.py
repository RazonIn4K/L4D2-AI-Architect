#!/usr/bin/env python3
"""
L4D2 Copilot CLI

Command-line interface for code completion and generation
using the fine-tuned L4D2 modding model.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Optional

try:
    import requests
except ImportError:
    print("requests not installed. Run: pip install requests")
    sys.exit(1)

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.copilot_server import CopilotServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CopilotClient:
    """Client for interacting with Copilot API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    def complete(self, 
                 prompt: str, 
                 max_tokens: int = 256,
                 temperature: float = 0.7,
                 language: str = "sourcepawn") -> str:
        """Get code completion"""
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/complete",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "language": language
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("completion", "")
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return ""
    
    def chat(self, messages: List[dict], max_tokens: int = 512) -> str:
        """Chat with the model"""
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat",
                json={
                    "messages": messages,
                    "max_tokens": max_tokens
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices", [])
                if choices:
                    return choices[0]["message"]["content"]
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Request failed: {e}")
        
        return ""
    
    def is_healthy(self) -> bool:
        """Check if server is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False


def complete_command(args):
    """Handle completion command"""
    client = CopilotClient(args.url)
    
    if not client.is_healthy():
        print(f"Error: Server at {args.url} is not responding")
        print("Start the server with: python scripts/inference/copilot_server.py")
        sys.exit(1)
    
    # Read prompt
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File {args.file} not found")
            sys.exit(1)
        
        with open(args.file, 'r') as f:
            prompt = f.read()
    elif args.prompt:
        prompt = args.prompt
    else:
        # Read from stdin
        print("Enter prompt (Ctrl-D to finish):")
        prompt = sys.stdin.read().strip()
    
    if not prompt:
        print("Error: No prompt provided")
        sys.exit(1)
    
    # Get completion
    print("\nGenerating completion...")
    completion = client.complete(
        prompt=prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        language=args.language
    )
    
    # Output
    if completion:
        if args.output:
            with open(args.output, 'w') as f:
                f.write(completion)
            print(f"Completion written to {args.output}")
        else:
            print("\n--- Completion ---")
            print(completion)
    else:
        print("No completion generated")


def chat_command(args):
    """Handle chat command"""
    client = CopilotClient(args.url)
    
    if not client.is_healthy():
        print(f"Error: Server at {args.url} is not responding")
        print("Start the server with: python scripts/inference/copilot_server.py")
        sys.exit(1)
    
    messages = []
    
    # Add system message
    if args.system:
        messages.append({"role": "system", "content": args.system})
    else:
        messages.append({
            "role": "system", 
            "content": "You are an expert Left 4 Dead 2 modding assistant. Help with SourcePawn and VScript code."
        })
    
    # Interactive chat
    print("L4D2 Copilot Chat (type 'quit' to exit)")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            # Add user message
            messages.append({"role": "user", "content": user_input})
            
            # Get response
            print("\nAssistant: ", end="", flush=True)
            response = client.chat(messages, max_tokens=args.max_tokens)
            
            if response:
                print(response)
                # Add assistant message
                messages.append({"role": "assistant", "content": response})
            else:
                print("Sorry, I couldn't generate a response.")
                
        except KeyboardInterrupt:
            break
        except EOFError:
            break
    
    print("\nGoodbye!")


def generate_template_command(args):
    """Generate code templates"""
    
    templates = {
        "plugin": '''#include <sourcemod>
#include <sdktools>

#pragma newdecls required
#pragma semicolon 1

#define PLUGIN_VERSION "1.0.0"

public Plugin myinfo = {
    name = "L4D2 Plugin",
    author = "Your Name",
    description = "Plugin description",
    version = PLUGIN_VERSION,
    url = "https://github.com/yourusername"
};

public void OnPluginStart() {
    // Plugin initialization
    PrintToServer("Plugin loaded");
}

public void OnMapStart() {
    // Map-specific initialization
}

public void OnMapEnd() {
    // Cleanup
}

public void OnClientPutInServer(int client) {
    // Player joined
}

public Action OnPlayerRunCmd(int client, int &buttons, int &impulse, 
                            float vel[3], float angles[3], int &weapon) {
    // Player movement hook
    return Plugin_Continue;
}''',
        
        "command": '''#include <sourcemod>

public Plugin myinfo = {
    name = "Custom Command",
    author = "Your Name",
    description = "Adds custom commands",
    version = "1.0.0"
};

public void OnPluginStart() {
    RegAdminCmd("sm_custom", Command_Custom, ADMFLAG_GENERIC, "Custom command");
    RegConsoleCmd("sm_info", Command_Info, "Show information");
}

public Action Command_Custom(int client, int args) {
    if (args < 1) {
        ReplyToCommand(client, "Usage: sm_custom <parameter>");
        return Plugin_Handled;
    }
    
    char param[256];
    GetCmdArg(1, param, sizeof(param));
    
    ReplyToCommand(client, "You entered: %s", param);
    return Plugin_Handled;
}

public Action Command_Info(int client, int args) {
    ReplyToCommand(client, "This is a custom command plugin");
    return Plugin_Handled;
}''',
        
        "vscript": """// L4D2 VScript Example
// Director script for custom events

function OnGameEvent_round_start(params)
{
    printl("Round started!")
    
    // Initialize custom variables
    DirectorOptions.cm_CommonLimit <- 20
    DirectorOptions.cm_MaxSpecials <- 8
    DirectorOptions.cm_DominatorLimit <- 4
}

function OnGameEvent_player_death(params)
{
    local victim = params["userid"]
    local attacker = params["attacker"]
    
    if (victim != null && attacker != null)
    {
        printl("Player " + victim + " was killed by " + attacker)
    }
}

function Update()
{
    // Called every frame
    // Add custom logic here
    
    // Example: Spawn common infected
    if (Time() > NextSpawnTime)
    {
        local spawnPos = Vector(0, 0, 0)
        SpawnZombie("common", spawnPos)
        NextSpawnTime = Time() + 5.0
    }
}

// Initialize
NextSpawnTime <- 0.0
""",
        
        "entity": '''#include <sourcemod>
#include <sdktools>

public Plugin myinfo = {
    name = "Entity Manager",
    author = "Your Name",
    description = "Manage game entities",
    version = "1.0.0"
};

public void OnPluginStart() {
    // Hook entity creation
    HookEvent("player_spawn", Event_PlayerSpawn);
    HookEvent("infected_death", Event_InfectedDeath);
}

public void Event_PlayerSpawn(Event event, const char[] name, bool dontBroadcast) {
    int client = GetClientOfUserId(event.GetInt("userid"));
    
    if (IsClientInGame(client) && GetClientTeam(client) == 2) {
        // Give survivor starting items
        GivePlayerItem(client, "pistol");
        GivePlayerItem(client, "first_aid_kit");
    }
}

public void Event_InfectedDeath(Event event, const char[] name, bool dontBroadcast) {
    // Handle infected death
    int killer = GetClientOfUserId(event.GetInt("attacker"));
    
    if (killer > 0) {
        // Award points or trigger effects
        PrintToChat(killer, "Infected killed!");
    }
}

// Custom entity spawning
public void SpawnCustomItem(float pos[3], const char[] classname) {
    int entity = CreateEntityByName(classname);
    
    if (entity != -1) {
        TeleportEntity(entity, pos, NULL_VECTOR, NULL_VECTOR);
        DispatchSpawn(entity);
        ActivateEntity(entity);
    }
}'''
    }
    
    if args.template not in templates:
        print(f"Error: Unknown template '{args.template}'")
        print(f"Available templates: {', '.join(templates.keys())}")
        sys.exit(1)
    
    template = templates[args.template]
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(template)
        print(f"Template written to {args.output}")
    else:
        print(template)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="L4D2 Copilot CLI")
    parser.add_argument("--url", default="http://localhost:8000",
                       help="Server URL")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Log level")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Complete command
    complete_parser = subparsers.add_parser("complete", help="Complete code")
    complete_parser.add_argument("--prompt", help="Code prompt")
    complete_parser.add_argument("--file", help="Read prompt from file")
    complete_parser.add_argument("--max-tokens", type=int, default=256,
                                help="Maximum tokens to generate")
    complete_parser.add_argument("--temperature", type=float, default=0.7,
                                help="Generation temperature")
    complete_parser.add_argument("--language", choices=["sourcepawn", "vscript", "auto"],
                                default="sourcepawn", help="Programming language")
    complete_parser.add_argument("--output", help="Output file")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat")
    chat_parser.add_argument("--system", help="System message")
    chat_parser.add_argument("--max-tokens", type=int, default=512,
                             help="Maximum tokens to generate")
    
    # Template command
    template_parser = subparsers.add_parser("template", help="Generate code templates")
    template_parser.add_argument("template", 
                                choices=["plugin", "command", "vscript", "entity"],
                                help="Template type")
    template_parser.add_argument("--output", help="Output file")
    
    # Server command
    server_parser = subparsers.add_parser("serve", help="Start inference server")
    server_parser.add_argument("--model-path", default="./model_adapters/l4d2-code-lora",
                               help="Path to fine-tuned model")
    server_parser.add_argument("--host", default="0.0.0.0", help="Server host")
    server_parser.add_argument("--port", type=int, default=8000, help="Server port")
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Execute command
    if args.command == "complete":
        complete_command(args)
    elif args.command == "chat":
        chat_command(args)
    elif args.command == "template":
        generate_template_command(args)
    elif args.command == "serve":
        # Start server
        server = CopilotServer(
            model_path=args.model_path,
            base_model="unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
        )
        server.run(host=args.host, port=args.port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
