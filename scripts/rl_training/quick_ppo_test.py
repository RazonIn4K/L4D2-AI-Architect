#!/usr/bin/env python3
"""
Quick PPO Training Test for L4D2 RL Pipeline

A minimal test to validate that PPO training works correctly with the
EnhancedL4D2Env before committing Vultr credits to longer training runs.

Features:
- Uses EnhancedL4D2Env (no live L4D2 server required)
- Trains for 10,000 timesteps (quick validation)
- Uses "aggressive" personality reward shaping
- Saves model to model_adapters/rl_test/
- Reports training time and evaluation results

Usage:
    python scripts/rl_training/quick_ppo_test.py
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

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
SAVE_DIR = PROJECT_ROOT / "model_adapters" / "rl_test"


# Personality reward configurations (from train_ppo.py)
PERSONALITIES = {
    "balanced": {
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
    "aggressive": {
        "kill": 3.0,            # Much higher kill reward
        "kill_special": 10.0,   # Very high special kill reward
        "damage_dealt": 0.3,    # Reward damage more
        "damage_taken": -0.05,  # Less penalty for taking damage
        "heal_teammate": 1.0,   # Lower healing priority
        "heal_self": 0.5,       # Low self-heal priority
        "incapped": -5.0,       # Less afraid of getting incapped
        "death": -30.0,
        "safe_room": 50.0,      # Less focused on objective
        "survival": 0.005,
        "proximity_to_team": 0.0,  # Doesn't care about team
        "progress": 0.02,       # Less progress reward
        "checkpoint": 5.0,
        "item_pickup": 0.5,
    },
    "medic": {
        "kill": 0.5,
        "kill_special": 3.0,
        "damage_dealt": 0.05,
        "damage_taken": -0.2,   # Very cautious
        "heal_teammate": 15.0,  # High healing priority
        "heal_self": 5.0,
        "incapped": -15.0,
        "death": -100.0,        # Really doesn't want to die
        "safe_room": 100.0,
        "survival": 0.02,
        "proximity_to_team": 0.01,  # Stays with team
        "progress": 0.05,
        "checkpoint": 10.0,
        "item_pickup": 2.0,
    },
}


def apply_personality(env, personality: str):
    """Apply personality reward shaping to environment."""
    if personality in PERSONALITIES:
        env.reward_config = PERSONALITIES[personality].copy()
        logger.info(f"Applied '{personality}' personality reward config")
    return env


def main():
    """Run quick PPO training test."""
    print("\n" + "=" * 70)
    print("L4D2 RL Pipeline - Quick PPO Training Test")
    print("=" * 70 + "\n")

    # Configuration
    TOTAL_TIMESTEPS = 10_000
    PERSONALITY = "aggressive"
    N_EVAL_EPISODES = 5

    # Check dependencies
    try:
        import gymnasium as gym
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.evaluation import evaluate_policy
        from stable_baselines3.common.monitor import Monitor
        logger.info("All dependencies available")
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Run: pip install stable-baselines3[extra] gymnasium")
        sys.exit(1)

    # Import our enhanced mock environment
    try:
        from enhanced_mock_env import EnhancedL4D2Env
        logger.info("Successfully imported EnhancedL4D2Env")
    except ImportError as e:
        logger.error(f"Failed to import EnhancedL4D2Env: {e}")
        sys.exit(1)

    # Create save directory
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = SAVE_DIR / f"ppo_test_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir = model_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Model will be saved to: {model_dir}")

    # Create environment
    logger.info("Creating training environment...")
    env = EnhancedL4D2Env(
        max_episode_steps=1000,
        difficulty="normal",
    )
    env = apply_personality(env, PERSONALITY)
    env = Monitor(env)

    # Create evaluation environment
    eval_env = EnhancedL4D2Env(
        max_episode_steps=1000,
        difficulty="normal",
    )
    eval_env = apply_personality(eval_env, PERSONALITY)
    eval_env = Monitor(eval_env)

    # Test environment
    logger.info("Testing environment...")
    obs, info = env.reset()
    logger.info(f"  Observation shape: {obs.shape}")
    logger.info(f"  Observation dtype: {obs.dtype}")
    logger.info(f"  Action space: {env.action_space}")

    # Run a few random steps to verify
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
    logger.info("  Environment test passed")
    env.reset()

    # Check if tensorboard is available
    try:
        import tensorboard
        tb_log = str(log_dir)
        logger.info("TensorBoard logging enabled")
    except ImportError:
        tb_log = None
        logger.info("TensorBoard not installed, logging disabled")

    # Create PPO model with hyperparameters tuned for aggressive personality
    logger.info("\nCreating PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=128,           # Smaller for quick test
        batch_size=64,
        n_epochs=10,
        gamma=0.95,            # Slightly shorter horizon for aggressive
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,         # More exploration for aggressive
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=tb_log,
        seed=42,
    )
    logger.info(f"  Policy architecture: MlpPolicy")
    logger.info(f"  Learning rate: 3e-4")
    logger.info(f"  Gamma: 0.95 (aggressive tuning)")

    # Pre-training evaluation
    logger.info("\n" + "-" * 50)
    logger.info("Pre-training evaluation...")
    pre_mean, pre_std = evaluate_policy(
        model, eval_env, n_eval_episodes=N_EVAL_EPISODES, deterministic=True
    )
    logger.info(f"  Pre-training reward: {pre_mean:.2f} +/- {pre_std:.2f}")

    # Training
    logger.info("\n" + "-" * 50)
    logger.info(f"Training for {TOTAL_TIMESTEPS:,} timesteps...")
    logger.info(f"  Personality: {PERSONALITY}")
    logger.info("-" * 50)

    start_time = time.time()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            progress_bar=True,
        )
        training_time = time.time() - start_time
        training_success = True
    except Exception as e:
        logger.error(f"Training failed: {e}")
        training_time = time.time() - start_time
        training_success = False
        import traceback
        traceback.print_exc()

    # Post-training evaluation
    logger.info("\n" + "-" * 50)
    logger.info("Post-training evaluation...")
    post_mean, post_std = evaluate_policy(
        model, eval_env, n_eval_episodes=N_EVAL_EPISODES, deterministic=True
    )
    logger.info(f"  Post-training reward: {post_mean:.2f} +/- {post_std:.2f}")

    # Save model
    model_path = model_dir / "final_model"
    model.save(model_path)
    logger.info(f"Model saved to: {model_path}")

    # Detailed evaluation run
    logger.info("\n" + "-" * 50)
    logger.info("Running detailed evaluation episodes...")

    episode_stats = []
    for ep in range(N_EVAL_EPISODES):
        obs, info = eval_env.reset()
        episode_reward = 0
        step_count = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            step_count += 1

            if terminated or truncated:
                break

        stats = info.get("stats", {})
        episode_stats.append({
            "episode": ep + 1,
            "reward": float(episode_reward),
            "steps": step_count,
            "kills": stats.get("kills", 0),
            "special_kills": stats.get("special_kills", 0),
            "damage_dealt": stats.get("damage_dealt", 0),
            "damage_taken": stats.get("damage_taken", 0),
            "progress": info.get("progress", 0),
            "in_safe_room": info.get("in_safe_room", False),
        })

        logger.info(
            f"  Episode {ep + 1}: reward={episode_reward:.2f}, "
            f"steps={step_count}, kills={stats.get('kills', 0)}, "
            f"progress={info.get('progress', 0)*100:.1f}%"
        )

    # Summary
    avg_kills = sum(e["kills"] for e in episode_stats) / len(episode_stats)
    avg_progress = sum(e["progress"] for e in episode_stats) / len(episode_stats)
    reached_saferoom = sum(1 for e in episode_stats if e["in_safe_room"])

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"\nTraining Configuration:")
    print(f"  Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Personality: {PERSONALITY}")
    print(f"  Training time: {training_time:.1f}s ({TOTAL_TIMESTEPS/training_time:.1f} steps/sec)")
    print(f"  Training success: {'YES' if training_success else 'NO'}")

    print(f"\nPerformance Improvement:")
    print(f"  Pre-training reward:  {pre_mean:.2f} +/- {pre_std:.2f}")
    print(f"  Post-training reward: {post_mean:.2f} +/- {post_std:.2f}")
    improvement = post_mean - pre_mean
    print(f"  Improvement: {improvement:+.2f} ({improvement/max(abs(pre_mean), 0.01)*100:+.1f}%)")

    print(f"\nEvaluation Statistics ({N_EVAL_EPISODES} episodes):")
    print(f"  Average kills: {avg_kills:.1f}")
    print(f"  Average progress: {avg_progress*100:.1f}%")
    print(f"  Reached safe room: {reached_saferoom}/{N_EVAL_EPISODES}")

    print(f"\nModel saved to: {model_path}")
    print(f"TensorBoard logs: {log_dir}")
    print("\n" + "=" * 70)

    # Save test results
    results = {
        "test_timestamp": timestamp,
        "training_config": {
            "total_timesteps": TOTAL_TIMESTEPS,
            "personality": PERSONALITY,
            "n_eval_episodes": N_EVAL_EPISODES,
        },
        "results": {
            "training_time_seconds": training_time,
            "steps_per_second": TOTAL_TIMESTEPS / training_time,
            "training_success": training_success,
            "pre_training_reward": {"mean": float(pre_mean), "std": float(pre_std)},
            "post_training_reward": {"mean": float(post_mean), "std": float(post_std)},
            "improvement": float(improvement),
        },
        "evaluation_stats": {
            "avg_kills": avg_kills,
            "avg_progress": avg_progress,
            "reached_saferoom": reached_saferoom,
        },
        "episode_details": episode_stats,
    }

    safe_write_json(
        str(model_dir / "test_results.json"),
        results,
        PROJECT_ROOT
    )

    # Cleanup
    env.close()
    eval_env.close()

    if training_success and improvement > 0:
        print("\nRL PIPELINE VALIDATION: PASSED")
        print("The training showed improvement, indicating the pipeline is working correctly.")
        return 0
    elif training_success:
        print("\nRL PIPELINE VALIDATION: PARTIAL")
        print("Training completed but reward did not improve. This may need more timesteps.")
        return 0
    else:
        print("\nRL PIPELINE VALIDATION: FAILED")
        print("Training encountered errors. Check the logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
