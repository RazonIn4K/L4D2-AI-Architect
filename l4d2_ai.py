#!/usr/bin/env python3
"""
L4D2 AI Management CLI

Unified command-line interface for managing AI bots and directors.

Usage:
    python l4d2_ai.py list                    # List all trained models
    python l4d2_ai.py demo bot aggressive     # Run aggressive bot demo
    python l4d2_ai.py demo director nightmare # Run nightmare director demo
    python l4d2_ai.py eval bot defender 50    # Evaluate defender with 50 episodes
    python l4d2_ai.py train bot medic 1000000 # Train medic for 1M steps
    python l4d2_ai.py status                  # Show training status
    python l4d2_ai.py compare                 # Compare all models
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).parent

def find_bot_model(personality: str) -> Path | None:
    """Find the best model for a bot personality."""
    rl_agents = PROJECT_ROOT / "model_adapters" / "rl_agents"

    # Check for individually trained models first (longer training)
    for d in sorted(rl_agents.glob(f"ppo_{personality}_*"), reverse=True):
        model = d / "final_model.zip"
        if model.exists():
            return d / "final_model"

    # Check batch training folders
    for d in sorted(rl_agents.glob("all_personalities_*"), reverse=True):
        model = d / personality / "final_model.zip"
        if model.exists():
            return d / personality / "final_model"

    return None


def find_director_model(personality: str) -> Path | None:
    """Find the best model for a director personality."""
    director_agents = PROJECT_ROOT / "model_adapters" / "director_agents"

    for d in sorted(director_agents.glob(f"director_{personality}_*"), reverse=True):
        model = d / "final_model.zip"
        if model.exists():
            return d / "final_model"

    return None


def list_models():
    """List all trained models."""
    print("\n" + "=" * 60)
    print("TRAINED MODELS")
    print("=" * 60)

    # Bot models
    print("\n🤖 BOT AGENTS")
    print("-" * 60)

    personalities = ["aggressive", "balanced", "defender", "medic", "speedrunner"]
    for p in personalities:
        model = find_bot_model(p)
        if model:
            # Try to get training info
            info_file = model.parent / "training_info.json"
            if info_file.exists():
                with open(info_file) as f:
                    info = json.load(f)
                reward = info.get("final_mean_reward", 0)
                steps = info.get("total_timesteps", 0)
                print(f"  ✓ {p:<12} | {steps:>10,} steps | {reward:>8.2f} reward")
            else:
                print(f"  ✓ {p:<12} | model found")
        else:
            print(f"  ✗ {p:<12} | not trained")

    # Director models
    print("\n🎮 AI DIRECTOR AGENTS")
    print("-" * 60)

    personalities = ["standard", "relaxed", "intense", "nightmare"]
    for p in personalities:
        model = find_director_model(p)
        if model:
            info_file = model.parent / "training_info.json"
            if info_file.exists():
                with open(info_file) as f:
                    info = json.load(f)
                reward = info.get("final_mean_reward", 0)
                steps = info.get("total_timesteps", 0)
                print(f"  ✓ {p:<12} | {steps:>10,} steps | {reward:>8.2f} reward")
            else:
                print(f"  ✓ {p:<12} | model found")
        else:
            print(f"  ✗ {p:<12} | not trained")

    print()


def demo_bot(personality: str):
    """Run a bot demo."""
    model = find_bot_model(personality)
    if not model:
        print(f"❌ No trained model found for bot personality: {personality}")
        print("Train one with: python l4d2_ai.py train bot", personality, "500000")
        return 1

    print(f"\n🎮 Running {personality} bot demo...")
    print(f"   Model: {model}")
    print("   Press Ctrl+C to stop\n")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "rl_training" / "train_ppo.py"),
        "--mode", "demo",
        "--model", str(model),
        "--personality", personality,
        "--env", "mock"
    ]

    return subprocess.call(cmd)


def demo_director(personality: str):
    """Run a director demo."""
    model = find_director_model(personality)
    if not model:
        print(f"❌ No trained model found for director personality: {personality}")
        print("Train one with: python l4d2_ai.py train director", personality, "500000")
        return 1

    print(f"\n🎮 Running {personality} director demo...")
    print(f"   Model: {model}")
    print("   Running simulation...\n")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "director" / "test_director.py"),
        "--demo"
    ]

    return subprocess.call(cmd)


def eval_bot(personality: str, episodes: int):
    """Evaluate a bot model."""
    model = find_bot_model(personality)
    if not model:
        print(f"❌ No trained model found for bot personality: {personality}")
        return 1

    print(f"\n📊 Evaluating {personality} bot ({episodes} episodes)...")
    print(f"   Model: {model}\n")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "rl_training" / "train_ppo.py"),
        "--mode", "eval",
        "--model", str(model),
        "--personality", personality,
        "--eval-episodes", str(episodes),
        "--env", "mock"
    ]

    return subprocess.call(cmd)


def eval_director(personality: str, episodes: int):
    """Evaluate a director model."""
    model = find_director_model(personality)
    if not model:
        print(f"❌ No trained model found for director personality: {personality}")
        return 1

    print(f"\n📊 Evaluating {personality} director...")
    print(f"   Model: {model}\n")

    # Load and show model info
    try:
        from stable_baselines3 import PPO
        m = PPO.load(str(model) + ".zip")

        info_file = model.parent / "training_info.json"
        if info_file.exists():
            with open(info_file) as f:
                info = json.load(f)
            print(f"   Personality: {info.get('personality', 'unknown')}")
            print(f"   Timesteps: {info.get('total_timesteps', 0):,}")
            print(f"   Final Reward: {info.get('final_mean_reward', 0):.2f}")

        print(f"   Policy: {m.policy.__class__.__name__}")
        print(f"   Observation: {m.observation_space}")
        print(f"   Actions: {m.action_space}")
        print("\n   ✓ Model loaded successfully!")
        return 0
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 1


def train_bot(personality: str, timesteps: int):
    """Train a bot model."""
    print(f"\n🏋️ Training {personality} bot for {timesteps:,} timesteps...")
    print("   This may take a while...\n")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "rl_training" / "train_ppo.py"),
        "--timesteps", str(timesteps),
        "--personality", personality,
        "--n-envs", "4"
    ]

    return subprocess.call(cmd)


def train_director(personality: str, timesteps: int):
    """Train a director model."""
    print(f"\n🏋️ Training {personality} director for {timesteps:,} timesteps...")
    print("   This may take a while...\n")

    cmd = [
        sys.executable, "-m",
        "scripts.director.train_director_rl",
        "--personality", personality,
        "--timesteps", str(timesteps)
    ]

    return subprocess.call(cmd)


def show_status():
    """Show overall training status."""
    print("\n" + "=" * 60)
    print("L4D2 AI TRAINING STATUS")
    print("=" * 60)

    # Count models
    bot_count = sum(1 for p in ["aggressive", "balanced", "defender", "medic", "speedrunner"]
                    if find_bot_model(p))
    dir_count = sum(1 for p in ["standard", "relaxed", "intense", "nightmare"]
                    if find_director_model(p))

    print(f"\n  Bot Models:      {bot_count}/5 trained")
    print(f"  Director Models: {dir_count}/4 trained")

    # Check for TensorBoard logs
    log_dir = PROJECT_ROOT / "data" / "training_logs"
    if log_dir.exists():
        rl_logs = list((log_dir / "rl").glob("*")) if (log_dir / "rl").exists() else []
        print(f"\n  TensorBoard Logs: {len(rl_logs)} runs")
        print(f"  View with: tensorboard --logdir {log_dir} --port 6006")

    # Quick usage guide
    print("\n" + "-" * 60)
    print("QUICK COMMANDS")
    print("-" * 60)
    print("  python l4d2_ai.py list              # List all models")
    print("  python l4d2_ai.py demo bot aggressive")
    print("  python l4d2_ai.py demo director nightmare")
    print("  python l4d2_ai.py eval bot defender 50")
    print("  python l4d2_ai.py train bot balanced 1000000")
    print()


def compare_models():
    """Compare all trained models."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    # Collect bot data
    bot_data = []
    for p in ["aggressive", "balanced", "defender", "medic", "speedrunner"]:
        model = find_bot_model(p)
        if model:
            info_file = model.parent / "training_info.json"
            if info_file.exists():
                with open(info_file) as f:
                    info = json.load(f)
                bot_data.append({
                    "name": p,
                    "reward": info.get("final_mean_reward", 0),
                    "steps": info.get("total_timesteps", 0)
                })

    # Collect director data
    dir_data = []
    for p in ["standard", "relaxed", "intense", "nightmare"]:
        model = find_director_model(p)
        if model:
            info_file = model.parent / "training_info.json"
            if info_file.exists():
                with open(info_file) as f:
                    info = json.load(f)
                dir_data.append({
                    "name": p,
                    "reward": info.get("final_mean_reward", 0),
                    "steps": info.get("total_timesteps", 0)
                })

    # Sort by reward
    bot_data.sort(key=lambda x: x["reward"], reverse=True)
    dir_data.sort(key=lambda x: x["reward"], reverse=True)

    # Print bot comparison
    print("\n🤖 BOT AGENTS (by performance)")
    print("-" * 50)
    for i, b in enumerate(bot_data):
        bar_len = max(0, int((b["reward"] + 100) / 5))  # Scale for display
        bar = "█" * min(bar_len, 30)
        rank = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else "  "))
        print(f"{rank} {b['name']:<12} {b['reward']:>8.2f} |{bar}")

    # Print director comparison
    print("\n🎮 DIRECTORS (by performance)")
    print("-" * 50)
    for i, d in enumerate(dir_data):
        bar_len = int(d["reward"] / 10)  # Scale for display
        bar = "█" * min(bar_len, 30)
        rank = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else "  "))
        print(f"{rank} {d['name']:<12} {d['reward']:>8.2f} |{bar}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="L4D2 AI Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python l4d2_ai.py list
  python l4d2_ai.py demo bot aggressive
  python l4d2_ai.py demo director nightmare
  python l4d2_ai.py eval bot defender 50
  python l4d2_ai.py train bot balanced 1000000
  python l4d2_ai.py status
  python l4d2_ai.py compare
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # list command
    subparsers.add_parser("list", help="List all trained models")

    # status command
    subparsers.add_parser("status", help="Show training status")

    # compare command
    subparsers.add_parser("compare", help="Compare all models")

    # demo command
    demo_parser = subparsers.add_parser("demo", help="Run a model demo")
    demo_parser.add_argument("type", choices=["bot", "director"], help="Model type")
    demo_parser.add_argument("personality", help="Personality name")

    # eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a model")
    eval_parser.add_argument("type", choices=["bot", "director"], help="Model type")
    eval_parser.add_argument("personality", help="Personality name")
    eval_parser.add_argument("episodes", type=int, nargs="?", default=50, help="Number of episodes")

    # train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("type", choices=["bot", "director"], help="Model type")
    train_parser.add_argument("personality", help="Personality name")
    train_parser.add_argument("timesteps", type=int, nargs="?", default=500000, help="Training timesteps")

    args = parser.parse_args()

    if args.command == "list":
        list_models()
    elif args.command == "status":
        show_status()
    elif args.command == "compare":
        compare_models()
    elif args.command == "demo":
        if args.type == "bot":
            return demo_bot(args.personality)
        else:
            return demo_director(args.personality)
    elif args.command == "eval":
        if args.type == "bot":
            return eval_bot(args.personality, args.episodes)
        else:
            return eval_director(args.personality, args.episodes)
    elif args.command == "train":
        if args.type == "bot":
            return train_bot(args.personality, args.timesteps)
        else:
            return train_director(args.personality, args.timesteps)
    else:
        parser.print_help()
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
