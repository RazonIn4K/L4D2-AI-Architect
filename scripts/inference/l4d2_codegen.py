#!/usr/bin/env python3
"""
L4D2 SourcePawn Code Generator - Leverage your fine-tuned model.

This CLI tool uses your V2 fine-tuned model for:
1. Direct code generation from prompts
2. Batch generation for data augmentation
3. Interactive chat mode for plugin development

Usage:
    # Single generation
    python scripts/inference/l4d2_codegen.py generate "Write a plugin that spawns tanks"

    # Batch generation from file (50% cheaper via Batch API)
    python scripts/inference/l4d2_codegen.py batch prompts.txt --output generated/

    # Interactive chat mode
    python scripts/inference/l4d2_codegen.py chat

    # Estimate cost before running
    python scripts/inference/l4d2_codegen.py estimate prompts.txt

Model: ft:gpt-4o-mini-2024-07-18:highencodelearning:l4d2-sourcemod-v2:CuyGSbKT
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_path, safe_write_text

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load .env file if it exists
def load_env():
    """Load environment variables from .env file."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Strip inline comments (e.g., "value  # comment")
                    if '#' in value:
                        value = value.split('#')[0].strip()
                    if key not in os.environ:  # Don't override existing env vars
                        os.environ[key] = value

load_env()

# Model configuration
MODEL_ID = "ft:gpt-4o-mini-2024-07-18:highencodelearning:l4d2-sourcemod-v7:CvTBCVPi"
SYSTEM_PROMPT = """You are an expert SourcePawn and VScript developer for Left 4 Dead 2 SourceMod plugins.
Write clean, well-documented code with proper error handling. Use correct L4D2 APIs and events.

CRITICAL L4D2 API RULES:
- Use GetRandomFloat() and GetRandomInt(), NOT RandomFloat() or RandomInt()
- Use lunge_pounce event for Hunter pounces, NOT pounce
- Use tongue_grab for Smoker, NOT smoker_tongue_grab
- Use player_now_it for bile, NOT boomer_vomit
- Use charger_carry_start for Charger, NOT charger_grab"""

# Pricing (as of 2024) - fine-tuned GPT-4o-mini
PRICING = {
    "input_per_1m": 0.30,   # $0.30 per 1M input tokens
    "output_per_1m": 1.20,  # $1.20 per 1M output tokens
    "batch_discount": 0.50,  # 50% off for batch API
}


def get_client():
    """Get OpenAI client with error handling."""
    try:
        from openai import OpenAI
        return OpenAI()
    except ImportError:
        print("Error: openai package not installed. Run: pip install openai")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Could not initialize OpenAI client: {e}")
        print("Make sure OPENAI_API_KEY environment variable is set")
        sys.exit(1)


def generate_code(prompt: str, temperature: float = 0.1, max_tokens: int = 2048) -> dict:
    """Generate SourcePawn code from a prompt."""
    client = get_client()

    start_time = time.time()
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )
    elapsed = time.time() - start_time

    # Extract usage info
    usage = response.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens

    # Calculate cost
    cost = (input_tokens * PRICING["input_per_1m"] / 1_000_000 +
            output_tokens * PRICING["output_per_1m"] / 1_000_000)

    return {
        "prompt": prompt,
        "response": response.choices[0].message.content,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
        "elapsed_seconds": elapsed,
        "model": MODEL_ID,
        "timestamp": datetime.now().isoformat()
    }


def cmd_generate(args):
    """Handle single generation command."""
    print(f"Generating code for: {args.prompt[:50]}...")
    print(f"Model: {MODEL_ID}")
    print(f"Temperature: {args.temperature}")
    print("-" * 50)

    result = generate_code(args.prompt, temperature=args.temperature)

    print(result["response"])
    print("-" * 50)
    print(f"Tokens: {result['input_tokens']} in / {result['output_tokens']} out")
    print(f"Cost: ${result['cost']:.4f}")
    print(f"Time: {result['elapsed_seconds']:.2f}s")

    # Optionally save to file
    if args.output:
        output_path = safe_path(args.output, PROJECT_ROOT, create_parents=True)
        safe_write_text(output_path, result["response"], PROJECT_ROOT)
        print(f"Saved to: {output_path}")


def cmd_batch(args):
    """Handle batch generation from file."""
    # Read prompts from file
    prompts_path = safe_path(args.prompts_file, PROJECT_ROOT)
    if not prompts_path.exists():
        print(f"Error: Prompts file not found: {prompts_path}")
        sys.exit(1)

    with open(prompts_path, 'r') as f:
        prompts = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    print(f"Found {len(prompts)} prompts")
    print(f"Model: {MODEL_ID}")

    # Estimate cost
    avg_input_tokens = 100  # system + prompt
    avg_output_tokens = 1500  # typical code response
    estimated_cost = len(prompts) * (
        avg_input_tokens * PRICING["input_per_1m"] / 1_000_000 +
        avg_output_tokens * PRICING["output_per_1m"] / 1_000_000
    )

    print(f"Estimated cost: ${estimated_cost:.2f}")
    print(f"Tip: Use OpenAI Batch API for 50% off (${estimated_cost * 0.5:.2f})")

    if not args.yes:
        confirm = input("Proceed? [y/N]: ")
        if confirm.lower() != 'y':
            print("Cancelled")
            return

    # Create output directory
    output_dir = safe_path(args.output, PROJECT_ROOT, create_parents=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_cost = 0

    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Generating...", end=" ", flush=True)
        try:
            result = generate_code(prompt, temperature=args.temperature)
            results.append(result)
            total_cost += result["cost"]

            # Save individual file
            filename = f"generated_{i:03d}.sp"
            output_file = output_dir / filename
            safe_write_text(output_file, result["response"], PROJECT_ROOT)

            print(f"${result['cost']:.4f} - {filename}")

            # Rate limiting
            if i < len(prompts):
                time.sleep(0.5)

        except Exception as e:
            print(f"Error: {e}")
            results.append({"prompt": prompt, "error": str(e)})

    # Save manifest
    manifest = {
        "model": MODEL_ID,
        "timestamp": datetime.now().isoformat(),
        "total_prompts": len(prompts),
        "successful": len([r for r in results if "response" in r]),
        "failed": len([r for r in results if "error" in r]),
        "total_cost": total_cost,
        "results": results
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print("-" * 50)
    print(f"Complete! Generated {manifest['successful']}/{len(prompts)} files")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Output: {output_dir}")
    print(f"Manifest: {manifest_path}")


def cmd_chat(args):
    """Interactive chat mode for plugin development."""
    print("=" * 50)
    print("L4D2 SourcePawn Code Generator - Chat Mode")
    print("=" * 50)
    print(f"Model: {MODEL_ID}")
    print("Type 'quit' to exit, 'clear' to reset conversation")
    print("-" * 50)

    client = get_client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    total_cost = 0

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == 'quit':
            print(f"\nTotal session cost: ${total_cost:.4f}")
            break
        if user_input.lower() == 'clear':
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("Conversation cleared.")
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                max_tokens=2048,
                temperature=args.temperature
            )

            assistant_msg = response.choices[0].message.content
            messages.append({"role": "assistant", "content": assistant_msg})

            # Calculate cost
            usage = response.usage
            cost = (usage.prompt_tokens * PRICING["input_per_1m"] / 1_000_000 +
                   usage.completion_tokens * PRICING["output_per_1m"] / 1_000_000)
            total_cost += cost

            print(f"\nAssistant: {assistant_msg}")
            print(f"\n[Cost: ${cost:.4f} | Total: ${total_cost:.4f}]")

        except Exception as e:
            print(f"\nError: {e}")
            messages.pop()  # Remove failed user message


def cmd_estimate(args):
    """Estimate cost for a batch of prompts."""
    prompts_path = safe_path(args.prompts_file, PROJECT_ROOT)
    if not prompts_path.exists():
        print(f"Error: Prompts file not found: {prompts_path}")
        sys.exit(1)

    with open(prompts_path, 'r') as f:
        prompts = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    # Estimate tokens
    avg_input_tokens = 100  # system prompt + user prompt
    avg_output_tokens = 1500  # typical SourcePawn plugin

    total_input = len(prompts) * avg_input_tokens
    total_output = len(prompts) * avg_output_tokens

    # Calculate costs
    direct_cost = (total_input * PRICING["input_per_1m"] / 1_000_000 +
                   total_output * PRICING["output_per_1m"] / 1_000_000)
    batch_cost = direct_cost * PRICING["batch_discount"]

    print("=" * 50)
    print("Cost Estimation")
    print("=" * 50)
    print(f"Prompts: {len(prompts)}")
    print(f"Estimated input tokens: {total_input:,}")
    print(f"Estimated output tokens: {total_output:,}")
    print("-" * 50)
    print(f"Direct API cost: ${direct_cost:.4f}")
    print(f"Batch API cost:  ${batch_cost:.4f} (50% off)")
    print(f"Savings:         ${direct_cost - batch_cost:.4f}")
    print("-" * 50)
    print("Tip: For batch processing, use OpenAI's Batch API")
    print("     Results delivered within 24 hours at 50% discount")


def cmd_info(args):
    """Show model information and usage tips."""
    print("=" * 60)
    print("L4D2 SourcePawn Fine-Tuned Model Information")
    print("=" * 60)
    print(f"""
MODEL DETAILS:
  ID: {MODEL_ID}
  Base: GPT-4o-mini (fine-tuned)
  Training: 517 quality examples + 15 synthetic
  Validation Score: 9.4/10 (5-prompt battery)

PRICING (Fine-tuned GPT-4o-mini):
  Input:  ${PRICING['input_per_1m']:.2f} per 1M tokens
  Output: ${PRICING['output_per_1m']:.2f} per 1M tokens
  Batch:  50% discount (24-hour delivery)

TYPICAL COSTS:
  Single generation: ~$0.002-0.004
  100 generations:   ~$0.20-0.40 (direct) or $0.10-0.20 (batch)

BEST PRACTICES:
  1. Use temperature 0.1-0.2 for consistent code output (0.3+ increases variance)
  2. Use Batch API for data augmentation (50% savings)
  3. Cache common responses to reduce API calls
  4. Include specific L4D2 context in prompts for best results

CAPABILITIES:
  - Tank/Witch spawn detection
  - Survivor healing/reviving mechanics
  - Player event handling (death, hurt, incap)
  - Timer-based mechanics
  - Console command registration
  - Team-based logic (survivors=2, infected=3)
  - L4D2 chat colors and formatting

EXAMPLE PROMPTS:
  "Write a plugin that spawns a tank every 10 minutes"
  "Create a speed boost when survivors kill special infected"
  "Make a friendly fire prevention plugin during panic events"
""")


def main():
    parser = argparse.ArgumentParser(
        description='L4D2 SourcePawn Code Generator using fine-tuned model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s generate "Write a plugin that heals all survivors"
  %(prog)s batch prompts.txt --output generated/
  %(prog)s chat
  %(prog)s estimate prompts.txt
  %(prog)s info
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate code from a single prompt')
    gen_parser.add_argument('prompt', help='The prompt for code generation')
    gen_parser.add_argument('--temperature', '-t', type=float, default=0.1,
                           help='Temperature for generation (default: 0.1, range: 0.0-1.0)')
    gen_parser.add_argument('--output', '-o', help='Save output to file')
    gen_parser.set_defaults(func=cmd_generate)

    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Generate code from multiple prompts')
    batch_parser.add_argument('prompts_file', help='File with prompts (one per line)')
    batch_parser.add_argument('--output', '-o', default='data/generated',
                             help='Output directory (default: data/generated)')
    batch_parser.add_argument('--temperature', '-t', type=float, default=0.1,
                             help='Temperature for generation (default: 0.1, range: 0.0-1.0)')
    batch_parser.add_argument('--yes', '-y', action='store_true',
                             help='Skip confirmation prompt')
    batch_parser.set_defaults(func=cmd_batch)

    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Interactive chat mode')
    chat_parser.add_argument('--temperature', '-t', type=float, default=0.1,
                            help='Temperature for generation (default: 0.1, range: 0.0-1.0)')
    chat_parser.set_defaults(func=cmd_chat)

    # Estimate command
    est_parser = subparsers.add_parser('estimate', help='Estimate cost for batch prompts')
    est_parser.add_argument('prompts_file', help='File with prompts (one per line)')
    est_parser.set_defaults(func=cmd_estimate)

    # Info command
    info_parser = subparsers.add_parser('info', help='Show model information')
    info_parser.set_defaults(func=cmd_info)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == '__main__':
    main()
