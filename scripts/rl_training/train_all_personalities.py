#!/usr/bin/env python3
"""
Comprehensive RL Training Script for All L4D2 Bot Personalities

Trains PPO models for all 5 personalities using the EnhancedL4D2Env:
- balanced (500K timesteps)
- aggressive (500K timesteps)
- medic (500K timesteps)
- speedrunner (500K timesteps)
- defender (500K timesteps)

Features:
- GPU acceleration with CUDA when available
- Parallel training with SubprocVecEnv (4 envs per personality)
- Checkpoints every 50K timesteps
- TensorBoard logging
- Evaluation over 100 episodes per personality
- JSON results saved for each personality
- Final comparison report

Estimated training time: ~30-45 min on GPU (Vultr A40/A100)

Usage:
    python scripts/rl_training/train_all_personalities.py
    python scripts/rl_training/train_all_personalities.py --timesteps 100000  # Quick test
    python scripts/rl_training/train_all_personalities.py --personalities balanced aggressive
"""

import os
import sys
import time
import json
import logging
import argparse
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from utils.security import safe_path, safe_write_json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "model_adapters" / "rl_agents"
LOGS_DIR = PROJECT_ROOT / "data" / "training_logs" / "rl"

# Import dependencies (with auto-install)
try:
    import torch
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CheckpointCallback,
        CallbackList,
    )
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor
except ImportError as e:
    logger.error(f"Missing dependency: {e}")
    logger.error("Run: pip install stable-baselines3[extra] gymnasium torch")
    sys.exit(1)

# Import enhanced mock environment
try:
    from enhanced_mock_env import EnhancedL4D2Env
except ImportError as e:
    logger.error(f"Failed to import EnhancedL4D2Env: {e}")
    sys.exit(1)


# ============================================================================
# Personality Configurations
# ============================================================================

PERSONALITIES = {
    "balanced": {
        "description": "Well-rounded survivor that balances combat, healing, and objectives",
        "reward_config": {
            "kill": 1.0,
            "kill_special": 5.0,
            "damage_dealt": 0.1,
            "damage_taken": -0.1,
            "heal_teammate": 5.0,
            "heal_self": 2.0,
            "incapped": -10.0,
            "death": -50.0,
            "safe_room": 100.0,
            "survival": 0.01,
            "proximity_to_team": 0.001,
            "progress": 0.05,
            "checkpoint": 10.0,
            "item_pickup": 1.0,
        },
        "ppo_config": {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "ent_coef": 0.01,
        },
    },
    "aggressive": {
        "description": "Combat-focused survivor that prioritizes killing infected",
        "reward_config": {
            "kill": 3.0,
            "kill_special": 10.0,
            "damage_dealt": 0.3,
            "damage_taken": -0.05,
            "heal_teammate": 1.0,
            "heal_self": 0.5,
            "incapped": -5.0,
            "death": -30.0,
            "safe_room": 50.0,
            "survival": 0.005,
            "proximity_to_team": 0.0,
            "progress": 0.02,
            "checkpoint": 5.0,
            "item_pickup": 0.5,
        },
        "ppo_config": {
            "learning_rate": 3e-4,
            "gamma": 0.95,
            "ent_coef": 0.02,
        },
    },
    "medic": {
        "description": "Support-focused survivor that prioritizes healing teammates",
        "reward_config": {
            "kill": 0.5,
            "kill_special": 3.0,
            "damage_dealt": 0.05,
            "damage_taken": -0.2,
            "heal_teammate": 15.0,
            "heal_self": 5.0,
            "incapped": -15.0,
            "death": -100.0,
            "safe_room": 100.0,
            "survival": 0.02,
            "proximity_to_team": 0.01,
            "progress": 0.05,
            "checkpoint": 10.0,
            "item_pickup": 2.0,
        },
        "ppo_config": {
            "learning_rate": 3e-4,
            "gamma": 0.995,
            "ent_coef": 0.005,
        },
    },
    "speedrunner": {
        "description": "Objective-focused survivor that prioritizes reaching the safe room",
        "reward_config": {
            "kill": 0.2,
            "kill_special": 1.0,
            "damage_dealt": 0.0,
            "damage_taken": -0.05,
            "heal_teammate": 0.5,
            "heal_self": 0.5,
            "incapped": -20.0,
            "death": -50.0,
            "safe_room": 200.0,
            "survival": 0.0,
            "proximity_to_team": -0.001,
            "progress": 0.2,
            "checkpoint": 20.0,
            "item_pickup": 0.2,
        },
        "ppo_config": {
            "learning_rate": 3e-4,
            "gamma": 0.9,
            "ent_coef": 0.03,
        },
    },
    "defender": {
        "description": "Team-focused survivor that protects teammates and holds positions",
        "reward_config": {
            "kill": 2.0,
            "kill_special": 8.0,
            "damage_dealt": 0.2,
            "damage_taken": -0.15,
            "heal_teammate": 8.0,
            "heal_self": 3.0,
            "incapped": -15.0,
            "death": -80.0,
            "safe_room": 80.0,
            "survival": 0.02,
            "proximity_to_team": 0.02,
            "progress": 0.03,
            "checkpoint": 8.0,
            "item_pickup": 1.5,
        },
        "ppo_config": {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "ent_coef": 0.01,
        },
    },
}


# ============================================================================
# Progress Callback
# ============================================================================

class TrainingProgressCallback(BaseCallback):
    """Custom callback for tracking training progress."""

    def __init__(
        self,
        personality: str,
        total_timesteps: int,
        log_freq: int = 10000,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.personality = personality
        self.total_timesteps = total_timesteps
        self.log_freq = log_freq
        self.start_time = None
        self.last_log_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.last_log_time = self.start_time

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            current_time = time.time()
            elapsed = current_time - self.start_time
            steps_per_sec = self.n_calls / elapsed if elapsed > 0 else 0
            progress = self.n_calls / self.total_timesteps * 100
            eta = (self.total_timesteps - self.n_calls) / steps_per_sec if steps_per_sec > 0 else 0

            logger.info(
                f"[{self.personality}] Progress: {progress:.1f}% | "
                f"Steps: {self.n_calls:,}/{self.total_timesteps:,} | "
                f"Speed: {steps_per_sec:.1f} steps/sec | "
                f"ETA: {eta/60:.1f} min"
            )

            self.last_log_time = current_time

        return True


# ============================================================================
# Environment Factory
# ============================================================================

def make_env(
    personality: str,
    seed: int = 0,
    difficulty: str = "normal",
    max_episode_steps: int = 2000,
) -> Callable[[], gym.Env]:
    """Factory function to create environments for SubprocVecEnv."""

    def _init() -> gym.Env:
        env = EnhancedL4D2Env(
            max_episode_steps=max_episode_steps,
            difficulty=difficulty,
            seed=seed,
        )
        # Apply personality reward config
        if personality in PERSONALITIES:
            env.reward_config = PERSONALITIES[personality]["reward_config"].copy()
        return Monitor(env)

    return _init


def create_vectorized_env(
    personality: str,
    n_envs: int = 4,
    use_subproc: bool = True,
    seed: int = 0,
) -> VecMonitor:
    """Create vectorized environment for parallel training."""

    env_fns = [
        make_env(personality, seed=seed + i)
        for i in range(n_envs)
    ]

    if use_subproc and n_envs > 1:
        # Use SubprocVecEnv for true parallelism
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        vec_env = DummyVecEnv(env_fns)

    return VecMonitor(vec_env)


# ============================================================================
# GPU Detection
# ============================================================================

def get_device() -> str:
    """Detect and configure the best available device."""

    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")

        # Optimize CUDA settings
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends, 'cuda'):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        logger.info("CUDA optimizations enabled (cuDNN benchmark, TF32)")

        return device
    else:
        logger.warning("No GPU detected, using CPU (training will be slower)")
        return "cpu"


# ============================================================================
# Training Function
# ============================================================================

def train_personality(
    personality: str,
    total_timesteps: int = 500_000,
    n_envs: int = 4,
    checkpoint_freq: int = 50_000,
    n_eval_episodes: int = 100,
    device: str = "auto",
    seed: int = 42,
    save_dir: Optional[Path] = None,
    log_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Train a single personality and return results.

    Args:
        personality: Name of the personality to train
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        checkpoint_freq: Frequency of checkpoint saves
        n_eval_episodes: Number of evaluation episodes
        device: Training device (cuda/cpu/auto)
        seed: Random seed
        save_dir: Directory to save models
        log_dir: Directory for TensorBoard logs

    Returns:
        Dictionary with training results and statistics
    """

    logger.info("\n" + "=" * 70)
    logger.info(f"Training Personality: {personality.upper()}")
    logger.info(f"Description: {PERSONALITIES[personality]['description']}")
    logger.info("=" * 70)

    # Setup directories
    if save_dir is None:
        save_dir = MODELS_DIR / personality
    save_dir.mkdir(parents=True, exist_ok=True)

    if log_dir is None:
        log_dir = LOGS_DIR / personality
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = save_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Model save path: {save_dir}")
    logger.info(f"Log directory: {log_dir}")

    # Create environments
    logger.info(f"Creating {n_envs} parallel environments...")
    train_env = create_vectorized_env(
        personality, n_envs=n_envs, use_subproc=True, seed=seed
    )

    # Create separate eval environment (single env for stable evaluation)
    eval_env = create_vectorized_env(
        personality, n_envs=1, use_subproc=False, seed=seed + 1000
    )

    # Get PPO config for this personality
    personality_config = PERSONALITIES[personality]["ppo_config"]

    # Create PPO model
    logger.info("Creating PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=personality_config["learning_rate"],
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=personality_config["gamma"],
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=personality_config["ent_coef"],
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        tensorboard_log=str(log_dir),
        device=device,
        seed=seed,
    )

    # Pre-training evaluation
    logger.info("Pre-training evaluation...")
    pre_mean, pre_std = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    logger.info(f"Pre-training reward: {pre_mean:.2f} +/- {pre_std:.2f}")

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq // n_envs,
        save_path=str(checkpoint_dir),
        name_prefix=f"ppo_{personality}",
    )

    progress_callback = TrainingProgressCallback(
        personality=personality,
        total_timesteps=total_timesteps,
        log_freq=10000,
    )

    callbacks = CallbackList([checkpoint_callback, progress_callback])

    # Training
    logger.info(f"Starting training for {total_timesteps:,} timesteps...")
    start_time = time.time()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            tb_log_name=personality,
        )
        training_success = True
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        training_success = False

    training_time = time.time() - start_time

    # Save final model
    final_model_path = save_dir / "final_model"
    model.save(final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")

    # Post-training evaluation (100 episodes)
    logger.info(f"Post-training evaluation ({n_eval_episodes} episodes)...")
    post_mean, post_std = evaluate_policy(
        model, eval_env, n_eval_episodes=n_eval_episodes, deterministic=True
    )
    logger.info(f"Post-training reward: {post_mean:.2f} +/- {post_std:.2f}")

    # Detailed evaluation with statistics
    logger.info("Running detailed evaluation...")
    episode_stats = []

    for ep in range(min(20, n_eval_episodes)):  # Detailed stats for 20 episodes
        obs = eval_env.reset()
        episode_reward = 0
        step_count = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = eval_env.step(action)
            episode_reward += reward[0]
            step_count += 1
            done = dones[0]

        info = infos[0] if infos else {}
        stats = info.get("stats", {})
        episode_stats.append({
            "episode": ep + 1,
            "reward": float(episode_reward),
            "steps": step_count,
            "kills": stats.get("kills", 0),
            "special_kills": stats.get("special_kills", 0),
            "damage_dealt": stats.get("damage_dealt", 0),
            "damage_taken": stats.get("damage_taken", 0),
            "teammates_healed": stats.get("teammates_healed", 0),
            "progress": info.get("progress", 0),
            "in_safe_room": info.get("in_safe_room", False),
        })

    # Compute aggregate statistics
    avg_kills = np.mean([e["kills"] for e in episode_stats])
    avg_special_kills = np.mean([e["special_kills"] for e in episode_stats])
    avg_progress = np.mean([e["progress"] for e in episode_stats])
    avg_damage_dealt = np.mean([e["damage_dealt"] for e in episode_stats])
    avg_damage_taken = np.mean([e["damage_taken"] for e in episode_stats])
    avg_teammates_healed = np.mean([e["teammates_healed"] for e in episode_stats])
    saferoom_rate = np.mean([e["in_safe_room"] for e in episode_stats])

    # Build results dictionary
    results = {
        "personality": personality,
        "description": PERSONALITIES[personality]["description"],
        "training": {
            "total_timesteps": total_timesteps,
            "n_envs": n_envs,
            "training_time_seconds": training_time,
            "training_time_minutes": training_time / 60,
            "steps_per_second": total_timesteps / training_time,
            "success": training_success,
            "device": device,
        },
        "performance": {
            "pre_training": {"mean": float(pre_mean), "std": float(pre_std)},
            "post_training": {"mean": float(post_mean), "std": float(post_std)},
            "improvement": float(post_mean - pre_mean),
            "improvement_percent": float((post_mean - pre_mean) / max(abs(pre_mean), 0.01) * 100),
        },
        "statistics": {
            "avg_kills": float(avg_kills),
            "avg_special_kills": float(avg_special_kills),
            "avg_progress": float(avg_progress),
            "avg_damage_dealt": float(avg_damage_dealt),
            "avg_damage_taken": float(avg_damage_taken),
            "avg_teammates_healed": float(avg_teammates_healed),
            "saferoom_rate": float(saferoom_rate),
        },
        "reward_config": PERSONALITIES[personality]["reward_config"],
        "ppo_config": PERSONALITIES[personality]["ppo_config"],
        "episode_details": episode_stats,
        "model_path": str(final_model_path),
        "timestamp": datetime.now().isoformat(),
    }

    # Save results to JSON
    safe_write_json(
        str(save_dir / "training_results.json"),
        results,
        PROJECT_ROOT,
    )

    # Cleanup
    train_env.close()
    eval_env.close()

    # Print summary
    logger.info("\n" + "-" * 50)
    logger.info(f"[{personality.upper()}] Training Complete")
    logger.info(f"  Time: {training_time/60:.1f} min ({total_timesteps/training_time:.1f} steps/sec)")
    logger.info(f"  Pre-training:  {pre_mean:.2f} +/- {pre_std:.2f}")
    logger.info(f"  Post-training: {post_mean:.2f} +/- {post_std:.2f}")
    logger.info(f"  Improvement:   {post_mean - pre_mean:+.2f}")
    logger.info(f"  Avg Kills:     {avg_kills:.1f}")
    logger.info(f"  Avg Progress:  {avg_progress*100:.1f}%")
    logger.info(f"  Saferoom Rate: {saferoom_rate*100:.1f}%")
    logger.info("-" * 50)

    return results


# ============================================================================
# Comparison Report
# ============================================================================

def generate_comparison_report(
    all_results: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> str:
    """Generate a comprehensive comparison report for all personalities."""

    report_lines = [
        "=" * 80,
        "L4D2 RL TRAINING - PERSONALITY COMPARISON REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # Training Summary
    report_lines.extend([
        "-" * 80,
        "TRAINING SUMMARY",
        "-" * 80,
        "",
        f"{'Personality':<15} {'Timesteps':>12} {'Time (min)':>12} {'Steps/sec':>12} {'Device':>10}",
        "-" * 65,
    ])

    total_time = 0
    total_steps = 0

    for personality, results in all_results.items():
        training = results["training"]
        total_time += training["training_time_minutes"]
        total_steps += training["total_timesteps"]
        report_lines.append(
            f"{personality:<15} {training['total_timesteps']:>12,} "
            f"{training['training_time_minutes']:>12.1f} "
            f"{training['steps_per_second']:>12.1f} "
            f"{training['device']:>10}"
        )

    report_lines.extend([
        "-" * 65,
        f"{'TOTAL':<15} {total_steps:>12,} {total_time:>12.1f}",
        "",
    ])

    # Performance Comparison
    report_lines.extend([
        "-" * 80,
        "PERFORMANCE COMPARISON",
        "-" * 80,
        "",
        f"{'Personality':<15} {'Pre-Train':>12} {'Post-Train':>12} {'Improvement':>12} {'Improv %':>10}",
        "-" * 65,
    ])

    for personality, results in all_results.items():
        perf = results["performance"]
        report_lines.append(
            f"{personality:<15} {perf['pre_training']['mean']:>12.2f} "
            f"{perf['post_training']['mean']:>12.2f} "
            f"{perf['improvement']:>+12.2f} "
            f"{perf['improvement_percent']:>+10.1f}%"
        )

    report_lines.append("")

    # Behavior Statistics
    report_lines.extend([
        "-" * 80,
        "BEHAVIOR STATISTICS",
        "-" * 80,
        "",
        f"{'Personality':<15} {'Kills':>8} {'Special':>8} {'Progress':>10} {'Healed':>8} {'Saferoom':>10}",
        "-" * 65,
    ])

    for personality, results in all_results.items():
        stats = results["statistics"]
        report_lines.append(
            f"{personality:<15} {stats['avg_kills']:>8.1f} "
            f"{stats['avg_special_kills']:>8.1f} "
            f"{stats['avg_progress']*100:>9.1f}% "
            f"{stats['avg_teammates_healed']:>8.1f} "
            f"{stats['saferoom_rate']*100:>9.1f}%"
        )

    report_lines.append("")

    # Personality Rankings
    report_lines.extend([
        "-" * 80,
        "PERSONALITY RANKINGS",
        "-" * 80,
        "",
    ])

    # Rank by different metrics
    metrics = [
        ("Best Improvement", "performance.improvement", True),
        ("Most Kills", "statistics.avg_kills", True),
        ("Most Healing", "statistics.avg_teammates_healed", True),
        ("Best Progress", "statistics.avg_progress", True),
        ("Best Saferoom Rate", "statistics.saferoom_rate", True),
        ("Least Damage Taken", "statistics.avg_damage_taken", False),
    ]

    for metric_name, metric_path, higher_is_better in metrics:
        parts = metric_path.split(".")

        def get_value(r):
            v = r
            for p in parts:
                v = v[p]
            return v

        sorted_personalities = sorted(
            all_results.keys(),
            key=lambda p: get_value(all_results[p]),
            reverse=higher_is_better
        )
        winner = sorted_personalities[0]
        value = get_value(all_results[winner])

        if "progress" in metric_path or "saferoom" in metric_path:
            value_str = f"{value*100:.1f}%"
        else:
            value_str = f"{value:.2f}"

        report_lines.append(f"  {metric_name:<25}: {winner:<15} ({value_str})")

    report_lines.extend([
        "",
        "-" * 80,
        "MODEL PATHS",
        "-" * 80,
        "",
    ])

    for personality, results in all_results.items():
        report_lines.append(f"  {personality:<15}: {results['model_path']}")

    report_lines.extend([
        "",
        "=" * 80,
        "END OF REPORT",
        "=" * 80,
    ])

    report_text = "\n".join(report_lines)

    # Save report
    report_path = output_dir / "comparison_report.txt"
    report_path.write_text(report_text)
    logger.info(f"Comparison report saved to: {report_path}")

    # Also save as JSON for programmatic access
    safe_write_json(
        str(output_dir / "all_results.json"),
        all_results,
        PROJECT_ROOT,
    )

    return report_text


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train PPO agents for all L4D2 bot personalities"
    )

    parser.add_argument(
        "--timesteps", type=int, default=500_000,
        help="Training timesteps per personality (default: 500000)"
    )
    parser.add_argument(
        "--n-envs", type=int, default=4,
        help="Number of parallel environments (default: 4)"
    )
    parser.add_argument(
        "--checkpoint-freq", type=int, default=50_000,
        help="Checkpoint frequency in timesteps (default: 50000)"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=100,
        help="Number of evaluation episodes per personality (default: 100)"
    )
    parser.add_argument(
        "--personalities", type=str, nargs="+",
        choices=list(PERSONALITIES.keys()),
        default=list(PERSONALITIES.keys()),
        help="Specific personalities to train (default: all)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Custom output directory for all models"
    )

    args = parser.parse_args()

    # Print header
    print("\n" + "=" * 80)
    print("L4D2 RL TRAINING - ALL PERSONALITIES")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Personalities: {', '.join(args.personalities)}")
    print(f"Timesteps per personality: {args.timesteps:,}")
    print(f"Parallel environments: {args.n_envs}")
    print(f"Total timesteps: {args.timesteps * len(args.personalities):,}")
    print("=" * 80 + "\n")

    # Detect device
    device = get_device()

    # Estimate training time
    estimated_steps_per_sec = 3000 if device == "cuda" else 500  # Conservative estimates
    estimated_time_per_personality = args.timesteps / estimated_steps_per_sec / 60
    estimated_total_time = estimated_time_per_personality * len(args.personalities)

    print(f"\nEstimated training time:")
    print(f"  Per personality: ~{estimated_time_per_personality:.1f} min")
    print(f"  Total: ~{estimated_total_time:.1f} min ({estimated_total_time/60:.1f} hours)")
    print("")

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = MODELS_DIR / f"all_personalities_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train all personalities
    all_results = {}
    total_start_time = time.time()

    for i, personality in enumerate(args.personalities):
        print(f"\n[{i+1}/{len(args.personalities)}] Training {personality}...")

        personality_save_dir = output_dir / personality
        personality_log_dir = LOGS_DIR / f"all_personalities_{datetime.now().strftime('%Y%m%d_%H%M%S')}" / personality

        results = train_personality(
            personality=personality,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            checkpoint_freq=args.checkpoint_freq,
            n_eval_episodes=args.eval_episodes,
            device=device,
            seed=args.seed + i * 100,  # Different seed per personality
            save_dir=personality_save_dir,
            log_dir=personality_log_dir,
        )

        all_results[personality] = results

    total_time = time.time() - total_start_time

    # Generate comparison report
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON REPORT")
    print("=" * 80)

    report = generate_comparison_report(all_results, output_dir)
    print("\n" + report)

    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total training time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
    print(f"All models saved to: {output_dir}")
    print(f"TensorBoard logs: tensorboard --logdir {LOGS_DIR}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    if mp.get_start_method(allow_none=True) != "spawn":
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # Already set

    main()
