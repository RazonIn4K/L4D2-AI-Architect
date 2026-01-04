#!/usr/bin/env python3
"""
PPO Training Script for L4D2 Bots

Trains Proximal Policy Optimization agents using Stable-Baselines3
to control L4D2 bots via the Mnemosyne environment.

Supports:
- Single agent training
- Multi-agent (vectorized environments)
- Multiple agent personalities (via reward shaping)
- TensorBoard logging
- Checkpoint saving/resuming

Usage:
    python train_ppo.py --episodes 10000 --save-path models/ppo_agent
    python train_ppo.py --resume models/ppo_agent --episodes 5000
    python train_ppo.py --personality aggressive --episodes 10000
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "model_adapters" / "rl_agents"
LOGS_DIR = PROJECT_ROOT / "data" / "training_logs" / "rl"


def install_dependencies():
    """Install required RL dependencies."""
    deps = [
        "stable-baselines3[extra]",
        "gymnasium",
        "tensorboard",
    ]
    for dep in deps:
        os.system(f"{sys.executable} -m pip install {dep}")


try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import (
        EvalCallback,
        CheckpointCallback,
        CallbackList,
    )
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor
except ImportError:
    logger.info("Installing dependencies...")
    install_dependencies()
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import (
        EvalCallback,
        CheckpointCallback,
        CallbackList,
    )
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor

# Import our environment
sys.path.insert(0, str(Path(__file__).parent))
from mnemosyne_env import MnemosyneEnv


# Agent personality presets (reward shaping configurations)
PERSONALITIES = {
    "balanced": {
        "kill": 1.0,
        "damage_dealt": 0.1,
        "damage_taken": -0.1,
        "heal_teammate": 5.0,
        "incapped": -10.0,
        "death": -50.0,
        "safe_room": 100.0,
        "survival": 0.01,
        "proximity_to_team": 0.001,
    },
    "aggressive": {
        "kill": 3.0,            # Much higher kill reward
        "damage_dealt": 0.3,    # Reward damage more
        "damage_taken": -0.05,  # Less penalty for taking damage
        "heal_teammate": 1.0,   # Lower healing priority
        "incapped": -5.0,       # Less afraid of getting incapped
        "death": -30.0,
        "safe_room": 50.0,      # Less focused on objective
        "survival": 0.005,
        "proximity_to_team": 0.0,  # Doesn't care about team
    },
    "medic": {
        "kill": 0.5,
        "damage_dealt": 0.05,
        "damage_taken": -0.2,   # Very cautious
        "heal_teammate": 15.0,  # High healing priority
        "incapped": -15.0,
        "death": -100.0,        # Really doesn't want to die
        "safe_room": 100.0,
        "survival": 0.02,
        "proximity_to_team": 0.01,  # Stays with team
    },
    "speedrunner": {
        "kill": 0.2,            # Minimal combat
        "damage_dealt": 0.0,
        "damage_taken": -0.05,
        "heal_teammate": 0.5,
        "incapped": -20.0,
        "death": -50.0,
        "safe_room": 200.0,     # High objective priority
        "survival": 0.0,
        "proximity_to_team": -0.001,  # Doesn't wait for team
    },
    "defender": {
        "kill": 2.0,
        "damage_dealt": 0.2,
        "damage_taken": -0.15,
        "heal_teammate": 8.0,
        "incapped": -15.0,
        "death": -80.0,
        "safe_room": 80.0,
        "survival": 0.02,
        "proximity_to_team": 0.02,  # High team cohesion
    },
}


def make_env(
    host: str = "localhost",
    port: int = 27050,
    bot_id: int = 0,
    personality: str = "balanced",
    rank: int = 0,
) -> Callable[[], gym.Env]:
    """Factory function to create environments."""
    def _init() -> gym.Env:
        env = MnemosyneEnv(
            host=host,
            port=port + rank,  # Different port per bot
            bot_id=bot_id + rank,
        )
        # Apply personality
        if personality in PERSONALITIES:
            env.reward_config = PERSONALITIES[personality].copy()
        return Monitor(env)
    return _init


def create_vectorized_env(
    n_envs: int = 4,
    host: str = "localhost",
    base_port: int = 27050,
    personality: str = "balanced",
    use_subproc: bool = False,
) -> VecMonitor:
    """Create vectorized environment for parallel training."""
    env_fns = [
        make_env(host, base_port, i, personality, i)
        for i in range(n_envs)
    ]
    
    if use_subproc and n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)
    
    return VecMonitor(vec_env)


def get_ppo_config(personality: str = "balanced") -> Dict[str, Any]:
    """Get PPO hyperparameters tuned for L4D2."""
    
    # Base configuration
    config = {
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "normalize_advantage": True,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "verbose": 1,
    }
    
    # Personality-specific adjustments
    if personality == "aggressive":
        config["ent_coef"] = 0.02  # More exploration
        config["gamma"] = 0.95     # Less long-term planning
    elif personality == "medic":
        config["gamma"] = 0.995    # More long-term planning
        config["ent_coef"] = 0.005 # More exploitation
    elif personality == "speedrunner":
        config["gamma"] = 0.9      # Short-term focused
        config["ent_coef"] = 0.03  # High exploration
    
    return config


def train(
    total_timesteps: int = 1_000_000,
    n_envs: int = 4,
    personality: str = "balanced",
    save_path: Optional[Path] = None,
    resume_from: Optional[Path] = None,
    host: str = "localhost",
    base_port: int = 27050,
    eval_freq: int = 10000,
    save_freq: int = 50000,
    log_dir: Optional[Path] = None,
):
    """Main training function."""
    
    # Setup directories
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = MODELS_DIR / f"ppo_{personality}_{timestamp}"
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    if log_dir is None:
        log_dir = LOGS_DIR / save_path.name
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Training PPO agent with '{personality}' personality")
    logger.info(f"Model save path: {save_path}")
    logger.info(f"Log directory: {log_dir}")
    
    # Create environments
    logger.info(f"Creating {n_envs} parallel environments...")
    train_env = create_vectorized_env(n_envs, host, base_port, personality)
    eval_env = create_vectorized_env(1, host, base_port + 100, personality)
    
    # Get PPO config
    ppo_config = get_ppo_config(personality)
    ppo_config["tensorboard_log"] = str(log_dir)
    
    # Create or load model
    if resume_from and resume_from.exists():
        logger.info(f"Resuming from {resume_from}")
        model = PPO.load(resume_from, env=train_env)
        model.tensorboard_log = str(log_dir)
    else:
        logger.info("Creating new PPO model...")
        model = PPO(env=train_env, **ppo_config)
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,
        save_path=str(save_path / "checkpoints"),
        name_prefix="ppo_checkpoint",
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path / "best_model"),
        log_path=str(log_dir / "eval"),
        eval_freq=eval_freq // n_envs,
        n_eval_episodes=10,
        deterministic=True,
    )
    
    callbacks = CallbackList([checkpoint_callback, eval_callback])
    
    # Train
    logger.info(f"Starting training for {total_timesteps:,} timesteps...")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    # Save final model
    final_path = save_path / "final_model"
    model.save(final_path)
    logger.info(f"Final model saved to {final_path}")
    
    # Evaluate final model
    logger.info("Evaluating final model...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=20, deterministic=True
    )
    logger.info(f"Final evaluation: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Save training info
    info = {
        "personality": personality,
        "total_timesteps": total_timesteps,
        "n_envs": n_envs,
        "final_mean_reward": float(mean_reward),
        "final_std_reward": float(std_reward),
        "ppo_config": {k: str(v) for k, v in ppo_config.items()},
        "reward_config": PERSONALITIES.get(personality, {}),
        "completed_at": datetime.now().isoformat(),
    }
    
    import json
    with open(save_path / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    return model, save_path


def evaluate(
    model_path: Path,
    n_episodes: int = 100,
    personality: str = "balanced",
    render: bool = False,
):
    """Evaluate a trained model."""
    logger.info(f"Loading model from {model_path}")
    
    env = MnemosyneEnv(render_mode="human" if render else None)
    if personality in PERSONALITIES:
        env.reward_config = PERSONALITIES[personality].copy()
    env = Monitor(env)
    
    model = PPO.load(model_path, env=env)
    
    logger.info(f"Evaluating for {n_episodes} episodes...")
    
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=n_episodes, deterministic=True
    )
    
    logger.info(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    env.close()
    return mean_reward, std_reward


def demo(model_path: Path, personality: str = "balanced"):
    """Run a demo of the trained agent."""
    logger.info(f"Loading model from {model_path}")
    
    env = MnemosyneEnv(render_mode="human")
    if personality in PERSONALITIES:
        env.reward_config = PERSONALITIES[personality].copy()
    
    model = PPO.load(model_path, env=env)
    
    obs, info = env.reset()
    total_reward = 0
    step = 0
    
    print("\n" + "=" * 60)
    print("DEMO MODE - Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            env.render()
            
            if terminated or truncated:
                print(f"\nEpisode ended at step {step}")
                print(f"Total reward: {total_reward:.2f}")
                print("-" * 40)
                
                obs, info = env.reset()
                total_reward = 0
                step = 0
                
    except KeyboardInterrupt:
        print("\nDemo stopped")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for L4D2")
    
    # Mode
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "eval", "demo"],
                        help="Operation mode")
    
    # Training parameters
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--personality", type=str, default="balanced",
                        choices=list(PERSONALITIES.keys()),
                        help="Agent personality preset")
    
    # Paths
    parser.add_argument("--save-path", type=str,
                        help="Path to save model")
    parser.add_argument("--resume", type=str,
                        help="Resume from checkpoint")
    parser.add_argument("--model", type=str,
                        help="Model path for eval/demo")
    
    # Connection
    parser.add_argument("--host", type=str, default="localhost",
                        help="Game server host")
    parser.add_argument("--port", type=int, default=27050,
                        help="Base port for bot connections")
    
    # Evaluation
    parser.add_argument("--eval-episodes", type=int, default=100,
                        help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true",
                        help="Render during evaluation")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            personality=args.personality,
            save_path=Path(args.save_path) if args.save_path else None,
            resume_from=Path(args.resume) if args.resume else None,
            host=args.host,
            base_port=args.port,
        )
    
    elif args.mode == "eval":
        if not args.model:
            logger.error("--model required for evaluation")
            sys.exit(1)
        evaluate(
            model_path=Path(args.model),
            n_episodes=args.eval_episodes,
            personality=args.personality,
            render=args.render,
        )
    
    elif args.mode == "demo":
        if not args.model:
            logger.error("--model required for demo")
            sys.exit(1)
        demo(
            model_path=Path(args.model),
            personality=args.personality,
        )


if __name__ == "__main__":
    main()
