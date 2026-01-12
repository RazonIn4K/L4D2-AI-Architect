#!/usr/bin/env python3
"""
L4D2 Copilot SDK - Example Usage Script

This script demonstrates all major features of the L4D2 Copilot SDK including:
- Basic code generation
- Multiple backends (Ollama, OpenAI, Server)
- Streaming generation
- Code validation
- Template expansion
- Batch generation
- Async operations
- Error handling

Run with: python scripts/inference/l4d2_sdk_examples.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.inference.l4d2_sdk import (
    L4D2Copilot,
    Language,
    TemplateType,
    ValidationError,
    ConfigurationError,
    create_copilot,
)


def print_header(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def example_basic_generation():
    """Basic code generation example."""
    print_header("Basic Code Generation")

    try:
        # Create copilot with Ollama backend (default)
        copilot = L4D2Copilot(backend="ollama")
        print(f"Backend: {copilot.backend_name}")
        print(f"Available: {copilot.is_available()}")

        if not copilot.is_available():
            print("Warning: Ollama backend not available. Skipping example.")
            return

        # Generate code
        prompt = "Write a SourcePawn function to heal all survivors to full health"
        print(f"\nPrompt: {prompt}\n")

        code = copilot.generate(prompt, max_tokens=256, temperature=0.7)
        print("Generated Code:")
        print("-" * 40)
        print(code)
        print("-" * 40)

    except ConfigurationError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"Error: {e}")


def example_generation_with_metadata():
    """Generation with full result metadata."""
    print_header("Generation with Metadata")

    try:
        copilot = L4D2Copilot(backend="ollama")

        if not copilot.is_available():
            print("Warning: Ollama backend not available. Skipping example.")
            return

        prompt = "Create a timer that prints 'Hello World' every 5 seconds"
        result = copilot.generate_result(prompt, max_tokens=256)

        print(f"Prompt: {prompt}\n")
        print(f"Model: {result.model}")
        print(f"Backend: {result.backend}")
        print(f"Tokens Generated: {result.tokens_generated}")
        print(f"Generation Time: {result.generation_time:.2f}s")
        print("\nGenerated Code:")
        print("-" * 40)
        print(result.code)
        print("-" * 40)

    except Exception as e:
        print(f"Error: {e}")


def example_code_validation():
    """Code validation example."""
    print_header("Code Validation")

    copilot = L4D2Copilot(backend="ollama")

    # Good SourcePawn code
    good_code = '''#include <sourcemod>
#include <sdktools>

#pragma newdecls required
#pragma semicolon 1

public Plugin myinfo = {
    name = "Test Plugin",
    author = "Developer",
    description = "A test plugin",
    version = "1.0.0"
};

public void OnPluginStart() {
    PrintToServer("Plugin loaded!");
}
'''

    print("Validating good code:")
    result = copilot.validate(good_code)
    print(f"  Valid: {result.is_valid}")
    print(f"  Language: {result.language.value if result.language else 'unknown'}")
    print(f"  Errors: {result.errors}")
    print(f"  Warnings: {result.warnings}")
    print(f"  Suggestions: {result.suggestions}")

    # Code with issues
    bad_code = '''public void DoSomething() {
    // TODO: implement this
    Format(buffer, 64, "%s", input);
}
'''

    print("\nValidating code with issues:")
    result = copilot.validate(bad_code)
    print(f"  Valid: {result.is_valid}")
    print(f"  Errors: {result.errors}")
    print(f"  Warnings: {result.warnings}")
    print(f"  Suggestions: {result.suggestions}")


def example_generation_with_validation():
    """Generate code with automatic validation."""
    print_header("Generation with Validation")

    try:
        copilot = L4D2Copilot(backend="ollama")

        if not copilot.is_available():
            print("Warning: Ollama backend not available. Skipping example.")
            return

        prompt = "Create a basic L4D2 plugin that tracks zombie kills"
        result = copilot.generate_with_validation(prompt, max_tokens=512)

        print(f"Prompt: {prompt}\n")
        print("Generated Code:")
        print("-" * 40)
        print(result.code)
        print("-" * 40)

        if result.validation:
            print(f"\nValidation:")
            print(f"  Valid: {result.validation.is_valid}")
            if result.validation.errors:
                print(f"  Errors: {result.validation.errors}")
            if result.validation.warnings:
                print(f"  Warnings: {result.validation.warnings}")
            if result.validation.suggestions:
                print(f"  Suggestions: {result.validation.suggestions}")

    except Exception as e:
        print(f"Error: {e}")


def example_templates():
    """Template expansion example."""
    print_header("Template Expansion")

    copilot = L4D2Copilot(backend="ollama")

    print("Available templates:", copilot.list_templates())
    print()

    # Expand plugin template
    plugin_code = copilot.expand_template(
        TemplateType.PLUGIN,
        name="Tank Spawner",
        author="L4D2 Modder",
        description="Spawns tanks on command",
        url="https://github.com/example/tank-spawner",
    )
    print("Plugin Template:")
    print("-" * 40)
    print(plugin_code[:500] + "...")
    print("-" * 40)

    # Expand command template
    command_code = copilot.expand_template(
        "command",
        name="Kill Counter",
        author="L4D2 Modder",
        command="killcount",
        command_cap="KillCount",
        description="Shows player kill count",
    )
    print("\nCommand Template:")
    print("-" * 40)
    print(command_code[:500] + "...")
    print("-" * 40)


def example_streaming():
    """Streaming generation example."""
    print_header("Streaming Generation")

    try:
        copilot = L4D2Copilot(backend="ollama")

        if not copilot.is_available():
            print("Warning: Ollama backend not available. Skipping example.")
            return

        prompt = "Write a simple function to give a player a medkit"
        print(f"Prompt: {prompt}\n")
        print("Streaming output:")
        print("-" * 40)

        full_response = ""
        for chunk in copilot.generate_stream(prompt, max_tokens=256):
            print(chunk, end="", flush=True)
            full_response += chunk

        print("\n" + "-" * 40)
        print(f"\nTotal length: {len(full_response)} characters")

    except Exception as e:
        print(f"Error: {e}")


def example_batch_generation():
    """Batch generation example."""
    print_header("Batch Generation")

    try:
        copilot = L4D2Copilot(backend="ollama")

        if not copilot.is_available():
            print("Warning: Ollama backend not available. Skipping example.")
            return

        prompts = [
            "Write a function to teleport a player",
            "Write a function to spawn a witch",
            "Write a function to check if player is alive",
        ]

        print("Prompts:")
        for i, p in enumerate(prompts, 1):
            print(f"  {i}. {p}")

        print("\nGenerating (parallel)...")
        batch_result = copilot.generate_batch(prompts, max_tokens=128, parallel=True)

        print(f"\nResults:")
        print(f"  Total time: {batch_result.total_time:.2f}s")
        print(f"  Successes: {batch_result.success_count}")
        print(f"  Failures: {batch_result.failure_count}")

        for i, result in enumerate(batch_result.results, 1):
            print(f"\n  Result {i}:")
            print(f"    Tokens: {result.tokens_generated}")
            print(f"    Time: {result.generation_time:.2f}s")
            code_preview = result.code[:100].replace("\n", " ")
            print(f"    Preview: {code_preview}...")

    except Exception as e:
        print(f"Error: {e}")


async def example_async_generation():
    """Async generation example."""
    print_header("Async Generation")

    try:
        copilot = L4D2Copilot(backend="ollama")

        if not copilot.is_available():
            print("Warning: Ollama backend not available. Skipping example.")
            return

        prompt = "Write an async-friendly function for L4D2"
        print(f"Prompt: {prompt}\n")

        print("Generating asynchronously...")
        code = await copilot.generate_async(prompt, max_tokens=256)

        print("Generated Code:")
        print("-" * 40)
        print(code)
        print("-" * 40)

    except Exception as e:
        print(f"Error: {e}")


async def example_async_batch():
    """Async batch generation example."""
    print_header("Async Batch Generation")

    try:
        copilot = L4D2Copilot(backend="ollama")

        if not copilot.is_available():
            print("Warning: Ollama backend not available. Skipping example.")
            return

        prompts = [
            "Function to count survivors",
            "Function to count infected",
        ]

        print("Prompts:", prompts)
        print("\nGenerating asynchronously...")

        batch_result = await copilot.generate_batch_async(prompts, max_tokens=128)

        print(f"\nResults:")
        print(f"  Total time: {batch_result.total_time:.2f}s")
        print(f"  Successes: {batch_result.success_count}")

        for i, result in enumerate(batch_result.results, 1):
            print(f"\n  Result {i}: {result.tokens_generated} tokens")

    except Exception as e:
        print(f"Error: {e}")


def example_different_languages():
    """Generation for different languages."""
    print_header("Different Languages")

    try:
        copilot = L4D2Copilot(backend="ollama")

        if not copilot.is_available():
            print("Warning: Ollama backend not available. Skipping example.")
            return

        # SourcePawn
        prompt = "Create a command to spawn a Tank"
        print(f"SourcePawn prompt: {prompt}")
        sp_code = copilot.generate(prompt, language=Language.SOURCEPAWN, max_tokens=200)
        print("SourcePawn code preview:")
        print(sp_code[:300] + "...")

        # VScript
        prompt = "Create a director script to increase zombie spawns"
        print(f"\nVScript prompt: {prompt}")
        vs_code = copilot.generate(prompt, language=Language.VSCRIPT, max_tokens=200)
        print("VScript code preview:")
        print(vs_code[:300] + "...")

    except Exception as e:
        print(f"Error: {e}")


def example_chat():
    """Chat-style interaction example."""
    print_header("Chat Interaction")

    try:
        copilot = L4D2Copilot(backend="ollama")

        if not copilot.is_available():
            print("Warning: Ollama backend not available. Skipping example.")
            return

        messages = [
            {"role": "system", "content": "You are an expert L4D2 modder."},
            {"role": "user", "content": "How do I detect when a player picks up a weapon?"},
        ]

        print("Simulated chat:")
        for msg in messages:
            print(f"  {msg['role']}: {msg['content']}")

        print("\nGenerating response...")
        response = copilot.chat(messages, max_tokens=256)

        print(f"  assistant: {response}")

    except Exception as e:
        print(f"Error: {e}")


def example_factory_function():
    """Using the factory function."""
    print_header("Factory Function")

    # Create with defaults
    copilot1 = create_copilot()
    print(f"Default copilot: {copilot1}")

    # Create with custom options
    copilot2 = create_copilot(
        backend="ollama",
        ollama_model="l4d2-code-v10plus",
        default_language=Language.VSCRIPT,
        auto_validate=True,
    )
    print(f"Custom copilot: {copilot2}")


def example_error_handling():
    """Error handling examples."""
    print_header("Error Handling")

    # Try invalid backend
    print("Testing invalid backend...")
    try:
        copilot = L4D2Copilot(backend="invalid")
    except ValueError as e:
        print(f"  Caught ValueError: {e}")

    # Try invalid template
    print("\nTesting invalid template...")
    copilot = L4D2Copilot(backend="ollama")
    try:
        copilot.expand_template("nonexistent")
    except ValueError as e:
        print(f"  Caught ValueError: {e}")

    # Test validation of empty code
    print("\nTesting validation of empty code...")
    result = copilot.validate("")
    print(f"  Valid: {result.is_valid}")
    print(f"  Errors: {result.errors}")


def main():
    """Run all examples."""
    print("\n" + "#" * 60)
    print("#  L4D2 Copilot SDK - Example Usage")
    print("#" * 60)

    # Run synchronous examples
    example_basic_generation()
    example_generation_with_metadata()
    example_code_validation()
    example_generation_with_validation()
    example_templates()
    example_streaming()
    example_batch_generation()
    example_different_languages()
    example_chat()
    example_factory_function()
    example_error_handling()

    # Run async examples
    print_header("Running Async Examples")
    asyncio.run(example_async_generation())
    asyncio.run(example_async_batch())

    print("\n" + "=" * 60)
    print("  All examples completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
