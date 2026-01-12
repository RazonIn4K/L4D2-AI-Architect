#!/usr/bin/env python3
"""
Test Script for Enhanced L4D2 Mock Environment

Runs the enhanced environment for a specified number of steps and reports
comprehensive statistics on gameplay metrics, reward distribution, and
environment behavior.

Usage:
    python test_enhanced_env.py
    python test_enhanced_env.py --steps 5000 --episodes 10
    python test_enhanced_env.py --difficulty expert --render
"""

import sys
import argparse
import time
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np

# Import the enhanced environment
sys.path.insert(0, str(Path(__file__).parent))
from enhanced_mock_env import EnhancedL4D2Env, BotAction


@dataclass
class EpisodeStats:
    """Statistics for a single episode."""
    steps: int = 0
    total_reward: float = 0.0
    kills: int = 0
    special_kills: int = 0
    damage_dealt: int = 0
    damage_taken: int = 0
    items_used: int = 0
    teammates_healed: int = 0
    distance_traveled: float = 0.0
    checkpoints_reached: int = 0
    final_health: int = 0
    survived: bool = False
    reached_saferoom: bool = False
    death_step: int = -1
    progress_percent: float = 0.0


@dataclass
class TestResults:
    """Aggregated test results across all episodes."""
    total_steps: int = 0
    total_episodes: int = 0
    episodes: List[EpisodeStats] = field(default_factory=list)

    # Aggregated stats
    total_kills: int = 0
    total_special_kills: int = 0
    total_damage_dealt: int = 0
    total_damage_taken: int = 0
    total_items_used: int = 0
    total_teammates_healed: int = 0
    total_distance: float = 0.0
    total_checkpoints: int = 0
    total_deaths: int = 0
    total_saferoom_reached: int = 0
    total_reward: float = 0.0

    # Action distribution
    action_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))

    # Reward analysis
    rewards: List[float] = field(default_factory=list)

    def add_episode(self, ep: EpisodeStats):
        self.episodes.append(ep)
        self.total_episodes += 1
        self.total_steps += ep.steps
        self.total_kills += ep.kills
        self.total_special_kills += ep.special_kills
        self.total_damage_dealt += ep.damage_dealt
        self.total_damage_taken += ep.damage_taken
        self.total_items_used += ep.items_used
        self.total_teammates_healed += ep.teammates_healed
        self.total_distance += ep.distance_traveled
        self.total_checkpoints += ep.checkpoints_reached
        self.total_reward += ep.total_reward
        if not ep.survived:
            self.total_deaths += 1
        if ep.reached_saferoom:
            self.total_saferoom_reached += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.episodes:
            return {}

        rewards = [ep.total_reward for ep in self.episodes]
        steps = [ep.steps for ep in self.episodes]
        kills = [ep.kills for ep in self.episodes]
        progress = [ep.progress_percent for ep in self.episodes]

        return {
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,

            # Reward stats
            "avg_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "median_reward": np.median(rewards),

            # Episode length
            "avg_steps": np.mean(steps),
            "std_steps": np.std(steps),
            "min_steps": np.min(steps),
            "max_steps": np.max(steps),

            # Combat
            "avg_kills": np.mean(kills),
            "total_kills": self.total_kills,
            "total_special_kills": self.total_special_kills,
            "total_damage_dealt": self.total_damage_dealt,
            "total_damage_taken": self.total_damage_taken,
            "damage_ratio": self.total_damage_dealt / max(1, self.total_damage_taken),

            # Progress
            "avg_progress": np.mean(progress) * 100,
            "max_progress": np.max(progress) * 100,
            "total_checkpoints": self.total_checkpoints,
            "saferoom_rate": self.total_saferoom_reached / max(1, self.total_episodes) * 100,

            # Survival
            "survival_rate": (self.total_episodes - self.total_deaths) / max(1, self.total_episodes) * 100,
            "death_rate": self.total_deaths / max(1, self.total_episodes) * 100,

            # Items and support
            "total_items_used": self.total_items_used,
            "total_teammates_healed": self.total_teammates_healed,
            "avg_distance": self.total_distance / max(1, self.total_episodes),
        }


def run_episode(
    env: EnhancedL4D2Env,
    max_steps: int = 1000,
    policy: str = "random",
    render: bool = False,
    render_freq: int = 50,
) -> EpisodeStats:
    """Run a single episode and collect statistics."""
    obs, info = env.reset()
    stats = EpisodeStats()

    for step in range(max_steps):
        # Select action based on policy
        if policy == "random":
            action = env.action_space.sample()
        elif policy == "forward":
            # Simple forward-moving policy with occasional attacks
            if np.random.random() < 0.1:
                action = BotAction.ATTACK
            else:
                action = BotAction.MOVE_FORWARD
        elif policy == "aggressive":
            # Prioritize attacking, then moving forward
            if info.get("nearby_enemies", 0) > 0 and np.random.random() < 0.7:
                action = BotAction.ATTACK
            elif np.random.random() < 0.6:
                action = BotAction.MOVE_FORWARD
            else:
                action = env.action_space.sample()
        elif policy == "defensive":
            # Prioritize healing and shoving
            if info.get("health", 100) < 50 and np.random.random() < 0.5:
                action = BotAction.HEAL_SELF
            elif info.get("nearby_enemies", 0) > 0 and np.random.random() < 0.3:
                action = BotAction.SHOVE
            elif np.random.random() < 0.5:
                action = BotAction.MOVE_FORWARD
            else:
                action = env.action_space.sample()
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        stats.total_reward += reward
        stats.steps += 1

        if render and step % render_freq == 0:
            env.render()

        if terminated or truncated:
            break

    # Collect final stats
    final_stats = info.get("stats", {})
    stats.kills = final_stats.get("kills", 0)
    stats.special_kills = final_stats.get("special_kills", 0)
    stats.damage_dealt = final_stats.get("damage_dealt", 0)
    stats.damage_taken = final_stats.get("damage_taken", 0)
    stats.items_used = final_stats.get("items_used", 0)
    stats.teammates_healed = final_stats.get("teammates_healed", 0)
    stats.distance_traveled = final_stats.get("distance_traveled", 0.0)
    stats.checkpoints_reached = final_stats.get("checkpoints_reached", 0)

    stats.final_health = info.get("health", 0)
    stats.survived = info.get("is_alive", False)
    stats.reached_saferoom = info.get("in_safe_room", False)
    stats.progress_percent = info.get("progress", 0.0)

    if not stats.survived:
        stats.death_step = stats.steps

    return stats


def run_test(
    num_episodes: int = 10,
    max_steps_per_episode: int = 1000,
    difficulty: str = "normal",
    policy: str = "random",
    render: bool = False,
    render_freq: int = 50,
    seed: int = None,
) -> TestResults:
    """Run multiple episodes and collect aggregated statistics."""
    print(f"\n{'='*60}")
    print(f"Enhanced L4D2 Environment Test")
    print(f"{'='*60}")
    print(f"Episodes: {num_episodes}")
    print(f"Max steps per episode: {max_steps_per_episode}")
    print(f"Difficulty: {difficulty}")
    print(f"Policy: {policy}")
    print(f"{'='*60}\n")

    env = EnhancedL4D2Env(
        max_episode_steps=max_steps_per_episode,
        difficulty=difficulty,
        render_mode="human" if render else None,
        seed=seed,
    )

    results = TestResults()
    start_time = time.time()

    for ep in range(num_episodes):
        ep_start = time.time()
        stats = run_episode(
            env,
            max_steps=max_steps_per_episode,
            policy=policy,
            render=render,
            render_freq=render_freq,
        )
        ep_time = time.time() - ep_start

        results.add_episode(stats)

        # Progress output
        status = "SAFE ROOM" if stats.reached_saferoom else ("SURVIVED" if stats.survived else "DIED")
        print(f"Episode {ep+1:3d}/{num_episodes}: "
              f"Steps={stats.steps:4d}, "
              f"Reward={stats.total_reward:8.2f}, "
              f"Kills={stats.kills:3d}, "
              f"Progress={stats.progress_percent*100:5.1f}%, "
              f"Status={status:10s}, "
              f"Time={ep_time:.2f}s")

    total_time = time.time() - start_time
    env.close()

    # Print summary
    summary = results.get_summary()

    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f}s ({total_time/num_episodes:.3f}s per episode)")
    print(f"Total steps: {results.total_steps:,}")
    print(f"Steps per second: {results.total_steps / total_time:.1f}")

    print(f"\n--- Reward Statistics ---")
    print(f"Average reward:  {summary['avg_reward']:10.2f}")
    print(f"Std reward:      {summary['std_reward']:10.2f}")
    print(f"Min reward:      {summary['min_reward']:10.2f}")
    print(f"Max reward:      {summary['max_reward']:10.2f}")
    print(f"Median reward:   {summary['median_reward']:10.2f}")

    print(f"\n--- Episode Length ---")
    print(f"Average steps:   {summary['avg_steps']:10.1f}")
    print(f"Std steps:       {summary['std_steps']:10.1f}")
    print(f"Min steps:       {summary['min_steps']:10.0f}")
    print(f"Max steps:       {summary['max_steps']:10.0f}")

    print(f"\n--- Combat Statistics ---")
    print(f"Total kills:        {summary['total_kills']:8d}")
    print(f"Special kills:      {summary['total_special_kills']:8d}")
    print(f"Average kills/ep:   {summary['avg_kills']:8.1f}")
    print(f"Damage dealt:       {summary['total_damage_dealt']:8d}")
    print(f"Damage taken:       {summary['total_damage_taken']:8d}")
    print(f"Damage ratio:       {summary['damage_ratio']:8.2f}")

    print(f"\n--- Progress Statistics ---")
    print(f"Average progress:   {summary['avg_progress']:7.1f}%")
    print(f"Max progress:       {summary['max_progress']:7.1f}%")
    print(f"Checkpoints:        {summary['total_checkpoints']:8d}")
    print(f"Safe room rate:     {summary['saferoom_rate']:7.1f}%")

    print(f"\n--- Survival Statistics ---")
    print(f"Survival rate:      {summary['survival_rate']:7.1f}%")
    print(f"Death rate:         {summary['death_rate']:7.1f}%")
    print(f"Total deaths:       {results.total_deaths:8d}")

    print(f"\n--- Support Statistics ---")
    print(f"Items used:         {summary['total_items_used']:8d}")
    print(f"Teammates healed:   {summary['total_teammates_healed']:8d}")
    print(f"Avg distance:       {summary['avg_distance']:8.1f}")

    print(f"\n{'='*60}\n")

    return results


def compare_policies(
    policies: List[str] = None,
    num_episodes: int = 10,
    max_steps: int = 1000,
    difficulty: str = "normal",
):
    """Compare different policies on the environment."""
    if policies is None:
        policies = ["random", "forward", "aggressive", "defensive"]

    print(f"\n{'='*60}")
    print("POLICY COMPARISON")
    print(f"{'='*60}\n")

    all_results = {}
    for policy in policies:
        print(f"\nTesting policy: {policy}")
        print("-" * 40)
        results = run_test(
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps,
            difficulty=difficulty,
            policy=policy,
            render=False,
        )
        all_results[policy] = results.get_summary()

    # Print comparison table
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")

    metrics = ["avg_reward", "avg_kills", "survival_rate", "avg_progress", "saferoom_rate"]
    headers = ["Policy"] + [m.replace("_", " ").title() for m in metrics]

    # Print header
    print(f"{'Policy':<15}", end="")
    for m in metrics:
        print(f"{m.replace('_', ' ').title():>15}", end="")
    print()
    print("-" * (15 + 15 * len(metrics)))

    # Print data
    for policy in policies:
        summary = all_results[policy]
        print(f"{policy:<15}", end="")
        for m in metrics:
            val = summary.get(m, 0)
            if "rate" in m or "progress" in m:
                print(f"{val:>14.1f}%", end="")
            elif isinstance(val, float):
                print(f"{val:>15.2f}", end="")
            else:
                print(f"{val:>15d}", end="")
        print()

    print(f"\n{'='*60}\n")

    return all_results


def run_benchmark(
    total_steps: int = 10000,
    difficulty: str = "normal",
):
    """Run a benchmark to measure environment performance."""
    print(f"\n{'='*60}")
    print("ENVIRONMENT BENCHMARK")
    print(f"{'='*60}")
    print(f"Total steps: {total_steps:,}")
    print(f"Difficulty: {difficulty}")
    print(f"{'='*60}\n")

    env = EnhancedL4D2Env(
        max_episode_steps=total_steps,
        difficulty=difficulty,
    )

    obs, info = env.reset()
    steps = 0
    resets = 0

    start_time = time.time()

    while steps < total_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

        if terminated or truncated:
            obs, info = env.reset()
            resets += 1

        if steps % 1000 == 0:
            elapsed = time.time() - start_time
            sps = steps / elapsed
            print(f"Steps: {steps:6d}/{total_steps}, SPS: {sps:.0f}, Resets: {resets}")

    total_time = time.time() - start_time
    env.close()

    print(f"\n--- Benchmark Results ---")
    print(f"Total steps:        {steps:,}")
    print(f"Total time:         {total_time:.2f}s")
    print(f"Steps per second:   {steps/total_time:,.0f}")
    print(f"Episode resets:     {resets}")
    print(f"Avg episode length: {steps/max(1,resets):.0f} steps")
    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Test Enhanced L4D2 Environment")

    parser.add_argument("--mode", type=str, default="test",
                        choices=["test", "compare", "benchmark"],
                        help="Test mode")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to run")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Max steps per episode (or total for benchmark)")
    parser.add_argument("--difficulty", type=str, default="normal",
                        choices=["easy", "normal", "hard", "expert"],
                        help="Game difficulty")
    parser.add_argument("--policy", type=str, default="random",
                        choices=["random", "forward", "aggressive", "defensive"],
                        help="Agent policy for testing")
    parser.add_argument("--render", action="store_true",
                        help="Render environment output")
    parser.add_argument("--render-freq", type=int, default=50,
                        help="Render frequency (every N steps)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    if args.mode == "test":
        run_test(
            num_episodes=args.episodes,
            max_steps_per_episode=args.steps,
            difficulty=args.difficulty,
            policy=args.policy,
            render=args.render,
            render_freq=args.render_freq,
            seed=args.seed,
        )
    elif args.mode == "compare":
        compare_policies(
            num_episodes=args.episodes,
            max_steps=args.steps,
            difficulty=args.difficulty,
        )
    elif args.mode == "benchmark":
        run_benchmark(
            total_steps=args.steps,
            difficulty=args.difficulty,
        )


if __name__ == "__main__":
    main()
