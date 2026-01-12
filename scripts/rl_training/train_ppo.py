#!/usr/bin/env python3
"""
PPO Training Script for L4D2 Bots

Trains Proximal Policy Optimization agents using Stable-Baselines3
to control L4D2 bots. Supports both standalone mock training and
live game server training via the Mnemosyne environment.

Supports:
- Standalone training with mock environment (default, no game server required)
- Live training via Mnemosyne environment (requires L4D2 game server)
- Single agent training
- Multi-agent (vectorized environments)
- Multiple agent personalities (via reward shaping)
- TensorBoard logging
- Checkpoint saving/resuming

Usage:
    # Standalone training with mock environment (default)
    python train_ppo.py --timesteps 100000 --personality balanced
    python train_ppo.py --env mock --timesteps 500000 --n-envs 4

    # Training with live L4D2 game server
    python train_ppo.py --env mnemosyne --host localhost --port 27050

    # Resume training
    python train_ppo.py --resume models/ppo_agent --timesteps 5000

    # Evaluation and demo
    python train_ppo.py --mode eval --model models/ppo_agent --env mock
    python train_ppo.py --mode demo --model models/ppo_agent --env mock
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable

import numpy as np

# Add parent to path for security utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_path, safe_write_json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "model_adapters" / "rl_agents"
LOGS_DIR = PROJECT_ROOT / "data" / "training_logs" / "rl"


def _resolve_path_within_root(path: Path, root: Path) -> Path:
    root_resolved = root.resolve()
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = root_resolved / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root_resolved):
        raise ValueError(f"Refusing path outside project root: {resolved}")
    return resolved


def install_dependencies():
    """Install required RL dependencies."""
    import subprocess
    deps = [
        "stable-baselines3[extra]",
        "gymnasium",
        "tensorboard",
    ]
    for dep in deps:
        # Use subprocess.run without shell=True to prevent command injection
        subprocess.run([sys.executable, "-m", "pip", "install", dep], check=False)


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

# Import our environments
sys.path.insert(0, str(Path(__file__).parent))
from mnemosyne_env import MnemosyneEnv
from enhanced_mock_env import EnhancedL4D2Env


# Agent personality presets (reward shaping configurations)
# Must include all keys expected by EnhancedL4D2Env
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
        "kill_special": 10.0,
        "damage_dealt": 0.3,    # Reward damage more
        "damage_taken": -0.05,  # Less penalty for taking damage
        "heal_teammate": 1.0,   # Lower healing priority
        "heal_self": 0.5,
        "incapped": -5.0,       # Less afraid of getting incapped
        "death": -30.0,
        "safe_room": 50.0,      # Less focused on objective
        "survival": 0.005,
        "proximity_to_team": 0.0,  # Doesn't care about team
        "progress": 0.02,
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
    "speedrunner": {
        "kill": 0.2,            # Minimal combat
        "kill_special": 1.0,
        "damage_dealt": 0.0,
        "damage_taken": -0.05,
        "heal_teammate": 0.5,
        "heal_self": 0.5,
        "incapped": -20.0,
        "death": -50.0,
        "safe_room": 200.0,     # High objective priority
        "survival": 0.0,
        "proximity_to_team": -0.001,  # Doesn't wait for team
        "progress": 0.2,
        "checkpoint": 20.0,
        "item_pickup": 0.2,
    },
    "defender": {
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
        "proximity_to_team": 0.02,  # High team cohesion
        "progress": 0.03,
        "checkpoint": 8.0,
        "item_pickup": 1.5,
    },
}


def make_env(
    host: str = "localhost",
    port: int = 27050,
    bot_id: int = 0,
    personality: str = "balanced",
    rank: int = 0,
    env_type: str = "mock",
) -> Callable[[], gym.Env]:
    """Factory function to create environments.

    Args:
        host: Game server host (only used for mnemosyne env)
        port: Base port for bot connections (only used for mnemosyne env)
        bot_id: Bot identifier (only used for mnemosyne env)
        personality: Agent personality preset for reward shaping
        rank: Environment rank for vectorized training
        env_type: Environment type - "mock" for standalone training, "mnemosyne" for live game
    """
    def _init() -> gym.Env:
        if env_type == "mnemosyne":
            env = MnemosyneEnv(
                host=host,
                port=port + rank,  # Different port per bot
                bot_id=bot_id + rank,
            )
        else:  # mock
            env = EnhancedL4D2Env()

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
    env_type: str = "mock",
) -> VecMonitor:
    """Create vectorized environment for parallel training.

    Args:
        n_envs: Number of parallel environments
        host: Game server host (only used for mnemosyne env)
        base_port: Base port for bot connections (only used for mnemosyne env)
        personality: Agent personality preset for reward shaping
        use_subproc: Use SubprocVecEnv for true parallelism (mnemosyne only)
        env_type: Environment type - "mock" for standalone training, "mnemosyne" for live game

    Note:
        SubprocVecEnv is only used when env_type="mnemosyne" and use_subproc=True.
        Mock environments use DummyVecEnv to avoid pickle issues.
    """
    env_fns = [
        make_env(host, base_port, i, personality, i, env_type)
        for i in range(n_envs)
    ]

    # Use DummyVecEnv for mock env to avoid pickle issues with dataclasses
    # SubprocVecEnv only makes sense for mnemosyne (true parallel game instances)
    if use_subproc and n_envs > 1 and env_type == "mnemosyne":
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
    env_type: str = "mock",
):
    """Main training function.

    Args:
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        personality: Agent personality preset
        save_path: Path to save model
        resume_from: Path to resume training from
        host: Game server host (mnemosyne only)
        base_port: Base port for connections (mnemosyne only)
        eval_freq: Evaluation frequency in timesteps
        save_freq: Checkpoint save frequency in timesteps
        log_dir: Directory for TensorBoard logs
        env_type: Environment type - "mock" (default) or "mnemosyne"
    """

    # Setup directories with path validation
    project_root = Path(__file__).parent.parent.parent.resolve()
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = MODELS_DIR / f"ppo_{personality}_{timestamp}"
    save_path = _resolve_path_within_root(Path(save_path), PROJECT_ROOT)
    save_path.mkdir(parents=True, exist_ok=True)

    if log_dir is None:
        log_dir = LOGS_DIR / save_path.name
    log_dir = _resolve_path_within_root(Path(log_dir), PROJECT_ROOT)
    log_dir.mkdir(parents=True, exist_ok=True)

    if resume_from is not None:
        resume_from = _resolve_path_within_root(Path(resume_from), PROJECT_ROOT)

    logger.info(f"Training PPO agent with '{personality}' personality")
    logger.info(f"Environment type: {env_type}")
    logger.info(f"Model save path: {save_path}")
    logger.info(f"Log directory: {log_dir}")

    # Create environments
    logger.info(f"Creating {n_envs} parallel environments ({env_type})...")
    train_env = create_vectorized_env(n_envs, host, base_port, personality, env_type=env_type)
    eval_env = create_vectorized_env(1, host, base_port + 100, personality, env_type=env_type)
    
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
    
    # Use safe_write_json to combine path validation and file writing
    safe_write_json(
        str(save_path / "training_info.json"),
        info,
        PROJECT_ROOT
    )
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    return model, save_path


def evaluate(
    model_path: Path,
    n_episodes: int = 100,
    personality: str = "balanced",
    render: bool = False,
    env_type: str = "mock",
):
    """Evaluate a trained model.

    Args:
        model_path: Path to the trained model
        n_episodes: Number of evaluation episodes
        personality: Agent personality preset
        render: Whether to render the environment
        env_type: Environment type - "mock" (default) or "mnemosyne"
    """
    model_path = _resolve_path_within_root(model_path, PROJECT_ROOT)
    logger.info(f"Loading model from {model_path}")
    logger.info(f"Environment type: {env_type}")

    if env_type == "mnemosyne":
        env = MnemosyneEnv(render_mode="human" if render else None)
    else:
        env = EnhancedL4D2Env(render_mode="human" if render else None)

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


def demo(model_path: Path, personality: str = "balanced", env_type: str = "mock"):
    """Run a demo of the trained agent.

    Args:
        model_path: Path to the trained model
        personality: Agent personality preset
        env_type: Environment type - "mock" (default) or "mnemosyne"
    """
    model_path = _resolve_path_within_root(model_path, PROJECT_ROOT)
    logger.info(f"Loading model from {model_path}")
    logger.info(f"Environment type: {env_type}")

    if env_type == "mnemosyne":
        env = MnemosyneEnv(render_mode="human")
    else:
        env = EnhancedL4D2Env(render_mode="human")

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

    # Environment selection
    parser.add_argument("--env", type=str, default="mock",
                        choices=["mock", "mnemosyne"],
                        help="Environment type: 'mock' for standalone training (default), "
                             "'mnemosyne' for live L4D2 game server")

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

    # Connection (mnemosyne only)
    parser.add_argument("--host", type=str, default="localhost",
                        help="Game server host (mnemosyne env only)")
    parser.add_argument("--port", type=int, default=27050,
                        help="Base port for bot connections (mnemosyne env only)")

    # Evaluation
    parser.add_argument("--eval-episodes", type=int, default=100,
                        help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true",
                        help="Render during evaluation")

    args = parser.parse_args()

    try:
        if args.mode == "train":
            train(
                total_timesteps=args.timesteps,
                n_envs=args.n_envs,
                personality=args.personality,
                save_path=Path(args.save_path) if args.save_path else None,
                resume_from=Path(args.resume) if args.resume else None,
                host=args.host,
                base_port=args.port,
                env_type=args.env,
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
                env_type=args.env,
            )

        elif args.mode == "demo":
            if not args.model:
                logger.error("--model required for demo")
                sys.exit(1)
            demo(
                model_path=Path(args.model),
                personality=args.personality,
                env_type=args.env,
            )
    except ValueError as e:
        logger.error(str(e))
        sys.exit(2)


if __name__ == "__main__":
    main()
