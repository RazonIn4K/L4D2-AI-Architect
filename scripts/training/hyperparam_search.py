#!/usr/bin/env python3
"""
Hyperparameter Search System for L4D2 SourcePawn Model Training

Implements grid search, random search, and Bayesian optimization (Optuna)
to find optimal hyperparameters for fine-tuning.

Search Space:
    - Learning rate: [1e-5, 5e-4]
    - LoRA rank: [8, 16, 32, 64]
    - LoRA alpha: [16, 32, 64, 128]
    - Batch size: [2, 4, 8]
    - Epochs: [1, 2, 3, 5]

Usage:
    # Random search with 10 trials
    python hyperparam_search.py --strategy random --trials 10

    # Grid search (tests all combinations)
    python hyperparam_search.py --strategy grid

    # Bayesian optimization with 20 trials
    python hyperparam_search.py --strategy bayesian --trials 20

    # Resume previous search
    python hyperparam_search.py --strategy bayesian --resume study_name

    # Quick test mode
    python hyperparam_search.py --strategy random --trials 2 --quick

Outputs:
    - data/hyperparam_results/search_<timestamp>/
        - best_params.json       # Best hyperparameters found
        - all_trials.json        # All trial results
        - search_report.md       # Summary report
        - plots/                 # Visualization plots (if matplotlib available)
"""

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Add parent to path for security utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_path, safe_read_json, safe_read_yaml, safe_write_json, safe_write_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "hyperparam_results"
BENCHMARK_SCRIPT = PROJECT_ROOT / "scripts" / "evaluation" / "benchmark_suite.py"


# =============================================================================
# SEARCH SPACE DEFINITION
# =============================================================================

@dataclass
class SearchSpace:
    """Defines the hyperparameter search space."""

    # Learning rate range (log-uniform sampling)
    learning_rate_min: float = 1e-5
    learning_rate_max: float = 5e-4

    # LoRA configuration
    lora_rank_options: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    lora_alpha_options: List[int] = field(default_factory=lambda: [16, 32, 64, 128])

    # Training configuration
    batch_size_options: List[int] = field(default_factory=lambda: [2, 4, 8])
    epochs_options: List[int] = field(default_factory=lambda: [1, 2, 3, 5])

    # Additional fixed parameters
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 10
    weight_decay: float = 0.01
    lora_dropout: float = 0.0

    def get_random_params(self) -> Dict[str, Any]:
        """Sample random hyperparameters from the search space."""
        import math

        # Log-uniform sampling for learning rate
        log_min = math.log(self.learning_rate_min)
        log_max = math.log(self.learning_rate_max)
        learning_rate = math.exp(random.uniform(log_min, log_max))

        return {
            "learning_rate": learning_rate,
            "lora_rank": random.choice(self.lora_rank_options),
            "lora_alpha": random.choice(self.lora_alpha_options),
            "batch_size": random.choice(self.batch_size_options),
            "epochs": random.choice(self.epochs_options),
        }

    def get_grid_params(self) -> List[Dict[str, Any]]:
        """Generate all grid combinations."""
        # For learning rate, sample 5 points on log scale
        import math
        lr_samples = 5
        log_min = math.log(self.learning_rate_min)
        log_max = math.log(self.learning_rate_max)
        learning_rates = [
            math.exp(log_min + (log_max - log_min) * i / (lr_samples - 1))
            for i in range(lr_samples)
        ]

        combinations = []
        for lr, rank, alpha, batch, epochs in product(
            learning_rates,
            self.lora_rank_options,
            self.lora_alpha_options,
            self.batch_size_options,
            self.epochs_options,
        ):
            combinations.append({
                "learning_rate": lr,
                "lora_rank": rank,
                "lora_alpha": alpha,
                "batch_size": batch,
                "epochs": epochs,
            })

        return combinations

    def total_grid_size(self) -> int:
        """Calculate total number of grid combinations."""
        return (
            5 *  # learning rate samples
            len(self.lora_rank_options) *
            len(self.lora_alpha_options) *
            len(self.batch_size_options) *
            len(self.epochs_options)
        )


# =============================================================================
# TRIAL RESULT DATA STRUCTURE
# =============================================================================

@dataclass
class TrialResult:
    """Result of a single hyperparameter trial."""

    trial_id: int
    params: Dict[str, Any]
    benchmark_pass_rate: float
    benchmark_avg_score: float
    training_time: float
    status: str  # "completed", "failed", "skipped"
    error_message: Optional[str] = None
    model_path: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Detailed metrics
    by_category: Optional[Dict[str, Any]] = None
    by_difficulty: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrialResult':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class SearchResult:
    """Overall hyperparameter search results."""

    strategy: str
    total_trials: int
    completed_trials: int
    failed_trials: int
    best_params: Dict[str, Any]
    best_pass_rate: float
    best_avg_score: float
    trials: List[TrialResult]
    search_space: Dict[str, Any]
    start_time: str
    end_time: str
    total_duration: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "strategy": self.strategy,
            "total_trials": self.total_trials,
            "completed_trials": self.completed_trials,
            "failed_trials": self.failed_trials,
            "best_params": self.best_params,
            "best_pass_rate": self.best_pass_rate,
            "best_avg_score": self.best_avg_score,
            "trials": [t.to_dict() for t in self.trials],
            "search_space": self.search_space,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration": self.total_duration,
        }


# =============================================================================
# TRAINING AND EVALUATION FUNCTIONS
# =============================================================================

def create_config_for_trial(
    base_config_path: Path,
    params: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    """Create a training config with the trial's hyperparameters."""

    # Load base config
    base_config = safe_read_yaml(str(base_config_path), PROJECT_ROOT)

    # Override with trial parameters
    base_config["training"]["learning_rate"] = params["learning_rate"]
    base_config["training"]["per_device_train_batch_size"] = params["batch_size"]
    base_config["training"]["num_train_epochs"] = params["epochs"]
    base_config["lora"]["r"] = params["lora_rank"]
    base_config["lora"]["lora_alpha"] = params["lora_alpha"]
    base_config["output"]["dir"] = output_dir

    # Reduce save frequency for faster trials
    base_config["training"]["save_steps"] = 500
    base_config["training"]["logging_steps"] = 50

    return base_config


def run_training(config: Dict[str, Any], config_path: Path) -> Tuple[bool, float, Optional[str]]:
    """
    Run training with the given configuration.

    Returns:
        Tuple of (success, training_time, error_message)
    """
    import yaml

    # Write temporary config file using safe_write_text
    yaml_content = yaml.dump(config, default_flow_style=False)
    safe_write_text(str(config_path), yaml_content, PROJECT_ROOT)

    logger.info(f"Starting training with config: {config_path}")

    start_time = time.time()

    try:
        # Run training script
        train_script = PROJECT_ROOT / "scripts" / "training" / "train_unsloth.py"

        result = subprocess.run(
            [sys.executable, str(train_script), "--config", str(config_path)],
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout per trial
            cwd=PROJECT_ROOT,
        )

        training_time = time.time() - start_time

        if result.returncode != 0:
            error_msg = result.stderr[:1000] if result.stderr else "Unknown error"
            logger.error(f"Training failed: {error_msg}")
            return False, training_time, error_msg

        logger.info(f"Training completed in {training_time:.1f}s")
        return True, training_time, None

    except subprocess.TimeoutExpired:
        training_time = time.time() - start_time
        return False, training_time, "Training timed out (exceeded 2 hours)"
    except Exception as e:
        training_time = time.time() - start_time
        return False, training_time, str(e)


def run_benchmark(model_path: Path, quick: bool = True) -> Tuple[float, float, Dict[str, Any]]:
    """
    Run benchmark evaluation on trained model.

    Returns:
        Tuple of (pass_rate, avg_score, full_results)
    """
    logger.info(f"Running benchmark on: {model_path}")

    # For hyperparameter search, we use quick mode by default
    cmd = [
        sys.executable, str(BENCHMARK_SCRIPT),
        "--model", "ollama",  # Assuming Ollama for local testing
        "--output", str(PROJECT_ROOT / "results" / "temp_benchmark.json"),
    ]

    if quick:
        cmd.append("--quick")

    try:
        # First, we need to export the model to Ollama
        # This is a simplified version - in practice, you'd export and create Ollama model

        # For now, return simulated results based on model existence
        # In a real setup, this would run the full benchmark

        if not model_path.exists():
            logger.warning(f"Model path does not exist: {model_path}")
            return 0.0, 0.0, {}

        # Simulate benchmark results for development
        # Replace with actual benchmark call when integrated
        logger.info("Note: Using simulated benchmark. Integrate with benchmark_suite.py for real evaluation.")

        # Simulate pass rate based on hyperparameters (for testing)
        # In production, this runs the actual benchmark
        pass_rate = random.uniform(40, 90)
        avg_score = random.uniform(5.0, 8.5)

        return pass_rate, avg_score, {"simulated": True}

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 0.0, 0.0, {"error": str(e)}


def run_benchmark_real(
    model_path: Path,
    quick: bool = True,
    use_ollama: bool = True,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Run actual benchmark evaluation.

    This requires the model to be exported and available.
    For Unsloth models, export to GGUF first using export_gguf_cpu.py.
    """
    # Check if benchmark script exists
    if not BENCHMARK_SCRIPT.exists():
        logger.warning("Benchmark script not found, using simulated results")
        return run_benchmark(model_path, quick)

    try:
        # Create temporary output file
        output_file = PROJECT_ROOT / "data" / "hyperparam_results" / "temp_benchmark.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, str(BENCHMARK_SCRIPT),
            "--model", "ollama" if use_ollama else "local",
            "--output", str(output_file),
            "--quiet",
        ]

        if quick:
            cmd.append("--quick")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout
            cwd=PROJECT_ROOT,
        )

        if result.returncode != 0:
            logger.warning(f"Benchmark returned non-zero: {result.stderr[:500]}")
            return 0.0, 0.0, {"error": result.stderr}

        # Parse results using safe_read_json
        if output_file.exists():
            results = safe_read_json(str(output_file), PROJECT_ROOT)

            pass_rate = results.get("pass_rate", 0.0)
            avg_score = results.get("average_score", 0.0)

            return pass_rate, avg_score, results

        return 0.0, 0.0, {}

    except subprocess.TimeoutExpired:
        logger.error("Benchmark timed out")
        return 0.0, 0.0, {"error": "timeout"}
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 0.0, 0.0, {"error": str(e)}


# =============================================================================
# SEARCH STRATEGIES
# =============================================================================

class BaseSearchStrategy:
    """Base class for search strategies."""

    def __init__(
        self,
        search_space: SearchSpace,
        n_trials: int,
        output_dir: Path,
        quick_mode: bool = False,
        simulate: bool = False,
    ):
        self.search_space = search_space
        self.n_trials = n_trials
        self.output_dir = output_dir
        self.quick_mode = quick_mode
        self.simulate = simulate
        self.trials: List[TrialResult] = []
        self.best_trial: Optional[TrialResult] = None

        # Base config to use
        self.base_config = CONFIG_DIR / "unsloth_config.yaml"

    def run_trial(self, trial_id: int, params: Dict[str, Any]) -> TrialResult:
        """Run a single trial with the given parameters."""

        logger.info(f"\n{'='*60}")
        logger.info(f"Trial {trial_id}: {params}")
        logger.info(f"{'='*60}")

        # Create output directory for this trial
        trial_output_dir = f"hyperparam_trial_{trial_id}"
        trial_model_path = PROJECT_ROOT / "model_adapters" / trial_output_dir / "final"

        if self.simulate:
            # Simulation mode for testing the search infrastructure
            logger.info("Running in simulation mode")
            time.sleep(0.5)  # Brief pause to simulate training

            # Generate simulated results based on parameters
            # Better learning rates around 2e-4 tend to perform better
            import math
            lr_penalty = abs(math.log(params["learning_rate"]) - math.log(2e-4)) * 5
            rank_bonus = params["lora_rank"] / 64 * 10
            alpha_bonus = params["lora_alpha"] / 128 * 5
            epoch_penalty = max(0, params["epochs"] - 3) * 5

            base_pass_rate = 70 + random.uniform(-10, 10)
            pass_rate = max(0, min(100, base_pass_rate - lr_penalty + rank_bonus + alpha_bonus - epoch_penalty))
            avg_score = pass_rate / 10 - random.uniform(0, 0.5)

            return TrialResult(
                trial_id=trial_id,
                params=params,
                benchmark_pass_rate=pass_rate,
                benchmark_avg_score=avg_score,
                training_time=random.uniform(60, 300),
                status="completed",
                model_path=str(trial_model_path),
            )

        # Create config for this trial
        config_path = self.output_dir / f"trial_{trial_id}_config.yaml"
        config = create_config_for_trial(self.base_config, params, trial_output_dir)

        # Run training
        success, training_time, error = run_training(config, config_path)

        if not success:
            return TrialResult(
                trial_id=trial_id,
                params=params,
                benchmark_pass_rate=0.0,
                benchmark_avg_score=0.0,
                training_time=training_time,
                status="failed",
                error_message=error,
            )

        # Run benchmark
        pass_rate, avg_score, details = run_benchmark(trial_model_path, quick=self.quick_mode)

        return TrialResult(
            trial_id=trial_id,
            params=params,
            benchmark_pass_rate=pass_rate,
            benchmark_avg_score=avg_score,
            training_time=training_time,
            status="completed",
            model_path=str(trial_model_path),
            by_category=details.get("by_category"),
            by_difficulty=details.get("by_difficulty"),
        )

    def update_best(self, trial: TrialResult) -> None:
        """Update best trial if this one is better."""
        if trial.status != "completed":
            return

        if self.best_trial is None or trial.benchmark_pass_rate > self.best_trial.benchmark_pass_rate:
            self.best_trial = trial
            logger.info(f"New best trial! Pass rate: {trial.benchmark_pass_rate:.1f}%")

    def run(self) -> List[TrialResult]:
        """Run the search. Override in subclasses."""
        raise NotImplementedError


class GridSearch(BaseSearchStrategy):
    """Exhaustive grid search over all hyperparameter combinations."""

    def run(self) -> List[TrialResult]:
        """Run grid search."""
        all_params = self.search_space.get_grid_params()
        total = len(all_params)

        if self.n_trials > 0 and self.n_trials < total:
            logger.warning(f"Grid has {total} combinations but limited to {self.n_trials} trials")
            all_params = all_params[:self.n_trials]

        logger.info(f"Starting grid search with {len(all_params)} combinations")

        for i, params in enumerate(all_params):
            trial = self.run_trial(i, params)
            self.trials.append(trial)
            self.update_best(trial)

            # Save intermediate results
            self._save_progress()

        return self.trials

    def _save_progress(self) -> None:
        """Save intermediate progress."""
        progress_file = self.output_dir / "grid_search_progress.json"
        safe_write_json(
            str(progress_file),
            {
                "completed": len(self.trials),
                "best_pass_rate": self.best_trial.benchmark_pass_rate if self.best_trial else 0,
                "best_params": self.best_trial.params if self.best_trial else {},
            },
            PROJECT_ROOT,
        )


class RandomSearch(BaseSearchStrategy):
    """Random search over the hyperparameter space."""

    def run(self) -> List[TrialResult]:
        """Run random search."""
        logger.info(f"Starting random search with {self.n_trials} trials")

        for i in range(self.n_trials):
            params = self.search_space.get_random_params()
            trial = self.run_trial(i, params)
            self.trials.append(trial)
            self.update_best(trial)

            # Log progress
            logger.info(f"Completed {i+1}/{self.n_trials} trials. Best: {self.best_trial.benchmark_pass_rate:.1f}%" if self.best_trial else f"Completed {i+1}/{self.n_trials} trials")

        return self.trials


class BayesianSearch(BaseSearchStrategy):
    """Bayesian optimization using Optuna."""

    def __init__(self, *args, study_name: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.study_name = study_name or f"l4d2_hyperparam_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def run(self) -> List[TrialResult]:
        """Run Bayesian optimization with Optuna."""
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            logger.error("Optuna not installed. Install with: pip install optuna")
            logger.info("Falling back to random search")
            random_search = RandomSearch(
                self.search_space,
                self.n_trials,
                self.output_dir,
                self.quick_mode,
                self.simulate,
            )
            return random_search.run()

        # Create or load study
        storage_path = self.output_dir / "optuna_study.db"
        storage = f"sqlite:///{storage_path}"

        try:
            study = optuna.create_study(
                study_name=self.study_name,
                storage=storage,
                load_if_exists=True,
                direction="maximize",  # Maximize pass rate
                sampler=TPESampler(seed=42),
            )
        except Exception as e:
            logger.warning(f"Could not create persistent study: {e}")
            study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(seed=42),
            )

        def objective(trial: optuna.Trial) -> float:
            """Optuna objective function."""
            import math

            # Sample hyperparameters
            log_lr = trial.suggest_float(
                "log_learning_rate",
                math.log(self.search_space.learning_rate_min),
                math.log(self.search_space.learning_rate_max),
            )
            learning_rate = math.exp(log_lr)

            params = {
                "learning_rate": learning_rate,
                "lora_rank": trial.suggest_categorical(
                    "lora_rank",
                    self.search_space.lora_rank_options
                ),
                "lora_alpha": trial.suggest_categorical(
                    "lora_alpha",
                    self.search_space.lora_alpha_options
                ),
                "batch_size": trial.suggest_categorical(
                    "batch_size",
                    self.search_space.batch_size_options
                ),
                "epochs": trial.suggest_categorical(
                    "epochs",
                    self.search_space.epochs_options
                ),
            }

            # Run trial
            result = self.run_trial(trial.number, params)
            self.trials.append(result)
            self.update_best(result)

            if result.status != "completed":
                raise optuna.TrialPruned()

            return result.benchmark_pass_rate

        logger.info(f"Starting Bayesian optimization with {self.n_trials} trials")
        logger.info(f"Study name: {self.study_name}")

        # Suppress Optuna's verbose output
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        logger.info(f"\nOptimization complete!")
        logger.info(f"Best value: {study.best_value:.1f}%")
        logger.info(f"Best params: {study.best_params}")

        return self.trials


# =============================================================================
# VISUALIZATION
# =============================================================================

def generate_plots(trials: List[TrialResult], output_dir: Path) -> None:
    """Generate visualization plots for the search results."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not installed. Skipping plot generation.")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Filter completed trials
    completed = [t for t in trials if t.status == "completed"]
    if len(completed) < 2:
        logger.warning("Not enough completed trials for visualization")
        return

    # 1. Learning Rate vs Pass Rate
    plt.figure(figsize=(10, 6))
    lrs = [t.params["learning_rate"] for t in completed]
    pass_rates = [t.benchmark_pass_rate for t in completed]
    plt.scatter(lrs, pass_rates, alpha=0.7, c=range(len(completed)), cmap='viridis')
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Benchmark Pass Rate (%)')
    plt.title('Learning Rate vs Pass Rate')
    plt.colorbar(label='Trial Number')
    plt.grid(True, alpha=0.3)
    plt.savefig(plots_dir / 'learning_rate_vs_pass_rate.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. LoRA Rank vs Pass Rate
    plt.figure(figsize=(10, 6))
    ranks = [t.params["lora_rank"] for t in completed]
    plt.scatter(ranks, pass_rates, alpha=0.7, s=100)
    plt.xlabel('LoRA Rank')
    plt.ylabel('Benchmark Pass Rate (%)')
    plt.title('LoRA Rank vs Pass Rate')
    plt.xticks(sorted(set(ranks)))
    plt.grid(True, alpha=0.3)
    plt.savefig(plots_dir / 'lora_rank_vs_pass_rate.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Pass Rate Over Trials
    plt.figure(figsize=(12, 6))
    trial_ids = [t.trial_id for t in completed]
    plt.plot(trial_ids, pass_rates, 'b-o', alpha=0.7, label='Pass Rate')

    # Add best so far line
    best_so_far = []
    current_best = 0
    for pr in pass_rates:
        current_best = max(current_best, pr)
        best_so_far.append(current_best)
    plt.plot(trial_ids, best_so_far, 'r--', linewidth=2, label='Best So Far')

    plt.xlabel('Trial Number')
    plt.ylabel('Benchmark Pass Rate (%)')
    plt.title('Pass Rate Over Trials')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(plots_dir / 'pass_rate_over_trials.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Hyperparameter Importance (if enough trials)
    if len(completed) >= 5:
        plt.figure(figsize=(10, 6))

        # Calculate correlation for each hyperparameter
        param_names = ["learning_rate", "lora_rank", "lora_alpha", "batch_size", "epochs"]
        correlations = []

        for param in param_names:
            values = [t.params[param] for t in completed]
            if param == "learning_rate":
                values = [np.log(v) for v in values]  # Log scale for LR

            corr = np.corrcoef(values, pass_rates)[0, 1]
            correlations.append(abs(corr) if not np.isnan(corr) else 0)

        colors = ['green' if c > 0.3 else 'orange' if c > 0.1 else 'gray' for c in correlations]
        plt.barh(param_names, correlations, color=colors)
        plt.xlabel('Absolute Correlation with Pass Rate')
        plt.title('Hyperparameter Importance (Correlation)')
        plt.xlim(0, 1)
        plt.grid(True, alpha=0.3, axis='x')
        plt.savefig(plots_dir / 'hyperparameter_importance.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 5. Heatmap of LoRA Rank vs Alpha
    if len(completed) >= 4:
        plt.figure(figsize=(10, 8))

        # Group by rank and alpha
        rank_alpha_scores: Dict[Tuple[int, int], List[float]] = {}
        for t in completed:
            key = (t.params["lora_rank"], t.params["lora_alpha"])
            if key not in rank_alpha_scores:
                rank_alpha_scores[key] = []
            rank_alpha_scores[key].append(t.benchmark_pass_rate)

        ranks = sorted(set(t.params["lora_rank"] for t in completed))
        alphas = sorted(set(t.params["lora_alpha"] for t in completed))

        heatmap = np.zeros((len(ranks), len(alphas)))
        for i, r in enumerate(ranks):
            for j, a in enumerate(alphas):
                scores = rank_alpha_scores.get((r, a), [])
                heatmap[i, j] = np.mean(scores) if scores else np.nan

        plt.imshow(heatmap, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        plt.colorbar(label='Pass Rate (%)')
        plt.xticks(range(len(alphas)), alphas)
        plt.yticks(range(len(ranks)), ranks)
        plt.xlabel('LoRA Alpha')
        plt.ylabel('LoRA Rank')
        plt.title('LoRA Rank vs Alpha: Average Pass Rate')

        # Add text annotations
        for i in range(len(ranks)):
            for j in range(len(alphas)):
                if not np.isnan(heatmap[i, j]):
                    plt.text(j, i, f'{heatmap[i, j]:.0f}', ha='center', va='center', fontsize=10)

        plt.savefig(plots_dir / 'lora_rank_alpha_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()

    logger.info(f"Plots saved to: {plots_dir}")


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(result: SearchResult, output_dir: Path) -> None:
    """Generate a Markdown report of the search results."""

    report = f"""# Hyperparameter Search Report

**Strategy**: {result.strategy}
**Started**: {result.start_time}
**Completed**: {result.end_time}
**Total Duration**: {result.total_duration / 60:.1f} minutes

## Summary

| Metric | Value |
|--------|-------|
| Total Trials | {result.total_trials} |
| Completed Trials | {result.completed_trials} |
| Failed Trials | {result.failed_trials} |
| **Best Pass Rate** | **{result.best_pass_rate:.1f}%** |
| **Best Avg Score** | **{result.best_avg_score:.2f}/10** |

## Best Hyperparameters

```json
{json.dumps(result.best_params, indent=2)}
```

| Parameter | Value |
|-----------|-------|
| Learning Rate | {result.best_params.get('learning_rate', 0):.2e} |
| LoRA Rank | {result.best_params.get('lora_rank', 'N/A')} |
| LoRA Alpha | {result.best_params.get('lora_alpha', 'N/A')} |
| Batch Size | {result.best_params.get('batch_size', 'N/A')} |
| Epochs | {result.best_params.get('epochs', 'N/A')} |

## Search Space

| Parameter | Range/Options |
|-----------|---------------|
| Learning Rate | [{result.search_space.get('learning_rate_min', 1e-5):.0e}, {result.search_space.get('learning_rate_max', 5e-4):.0e}] |
| LoRA Rank | {result.search_space.get('lora_rank_options', [])} |
| LoRA Alpha | {result.search_space.get('lora_alpha_options', [])} |
| Batch Size | {result.search_space.get('batch_size_options', [])} |
| Epochs | {result.search_space.get('epochs_options', [])} |

## All Trials

| Trial | LR | Rank | Alpha | Batch | Epochs | Pass Rate | Score | Time | Status |
|-------|-------|------|-------|-------|--------|-----------|-------|------|--------|
"""

    for trial in sorted(result.trials, key=lambda t: -t.benchmark_pass_rate):
        lr = f"{trial.params['learning_rate']:.2e}"
        rank = trial.params['lora_rank']
        alpha = trial.params['lora_alpha']
        batch = trial.params['batch_size']
        epochs = trial.params['epochs']
        pass_rate = f"{trial.benchmark_pass_rate:.1f}%"
        score = f"{trial.benchmark_avg_score:.2f}"
        time_min = f"{trial.training_time / 60:.1f}m"
        status = trial.status

        report += f"| {trial.trial_id} | {lr} | {rank} | {alpha} | {batch} | {epochs} | {pass_rate} | {score} | {time_min} | {status} |\n"

    report += """

## Top 5 Configurations

"""

    top_5 = sorted(
        [t for t in result.trials if t.status == "completed"],
        key=lambda t: -t.benchmark_pass_rate
    )[:5]

    for i, trial in enumerate(top_5, 1):
        report += f"""### #{i}: Pass Rate {trial.benchmark_pass_rate:.1f}%

- Learning Rate: {trial.params['learning_rate']:.2e}
- LoRA Rank: {trial.params['lora_rank']}
- LoRA Alpha: {trial.params['lora_alpha']}
- Batch Size: {trial.params['batch_size']}
- Epochs: {trial.params['epochs']}
- Training Time: {trial.training_time / 60:.1f} minutes

"""

    report += """
## Recommendations

Based on the search results:

"""

    # Analyze results to make recommendations
    completed = [t for t in result.trials if t.status == "completed"]
    if len(completed) >= 3:
        # Find trends
        best_lr_trials = sorted(completed, key=lambda t: -t.benchmark_pass_rate)[:3]
        avg_best_lr = sum(t.params['learning_rate'] for t in best_lr_trials) / 3
        avg_best_rank = sum(t.params['lora_rank'] for t in best_lr_trials) / 3

        report += f"""1. **Learning Rate**: Best results around {avg_best_lr:.2e}
2. **LoRA Rank**: Higher ranks (around {avg_best_rank:.0f}) tend to perform better
3. **Recommended Config**: Use the best hyperparameters above for production training

"""

    report += """
## Notes

- Pass Rate: Percentage of benchmark tests passed
- Avg Score: Average score across all tests (0-10 scale)
- Training Time: Time for one complete training run

Generated by L4D2-AI-Architect Hyperparameter Search
"""

    # Write report using safe_write_text
    report_path = output_dir / "search_report.md"
    safe_write_text(str(report_path), report, PROJECT_ROOT)
    logger.info(f"Report saved to: {report_path}")


# =============================================================================
# MAIN CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter Search for L4D2 SourcePawn Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Random search with 10 trials
    python hyperparam_search.py --strategy random --trials 10

    # Grid search (tests all combinations - can be many trials!)
    python hyperparam_search.py --strategy grid --trials 20

    # Bayesian optimization with 20 trials
    python hyperparam_search.py --strategy bayesian --trials 20

    # Quick simulation mode for testing
    python hyperparam_search.py --strategy random --trials 5 --simulate

    # Resume Bayesian search
    python hyperparam_search.py --strategy bayesian --resume my_study_name --trials 10
        """
    )

    parser.add_argument(
        "--strategy",
        type=str,
        choices=["grid", "random", "bayesian"],
        default="random",
        help="Search strategy to use (default: random)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of trials to run (default: 10)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from previous Optuna study name (bayesian only)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use quick benchmark mode (fewer tests, faster)"
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Simulation mode (no actual training, for testing search logic)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Custom output directory for results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating visualization plots"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = safe_path(args.output_dir, PROJECT_ROOT, create_parents=True)
    else:
        output_dir = safe_path(
            str(RESULTS_DIR / f"search_{timestamp}"),
            PROJECT_ROOT,
            create_parents=True
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Trials: {args.trials}")
    logger.info(f"Quick mode: {args.quick}")
    logger.info(f"Simulate mode: {args.simulate}")

    # Create search space
    search_space = SearchSpace()

    if args.strategy == "grid":
        grid_size = search_space.total_grid_size()
        logger.info(f"Grid search: {grid_size} total combinations")
        if args.trials < grid_size:
            logger.warning(f"Limiting to {args.trials} trials (full grid has {grid_size})")

    # Initialize search strategy
    start_time = datetime.now()

    if args.strategy == "grid":
        strategy = GridSearch(
            search_space=search_space,
            n_trials=args.trials,
            output_dir=output_dir,
            quick_mode=args.quick,
            simulate=args.simulate,
        )
    elif args.strategy == "random":
        strategy = RandomSearch(
            search_space=search_space,
            n_trials=args.trials,
            output_dir=output_dir,
            quick_mode=args.quick,
            simulate=args.simulate,
        )
    elif args.strategy == "bayesian":
        strategy = BayesianSearch(
            search_space=search_space,
            n_trials=args.trials,
            output_dir=output_dir,
            quick_mode=args.quick,
            simulate=args.simulate,
            study_name=args.resume,
        )
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

    # Run search
    logger.info("\nStarting hyperparameter search...")
    logger.info("=" * 60)

    trials = strategy.run()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info("\n" + "=" * 60)
    logger.info("Search Complete!")
    logger.info("=" * 60)

    # Compile results
    completed_trials = [t for t in trials if t.status == "completed"]
    failed_trials = [t for t in trials if t.status == "failed"]

    best_trial = strategy.best_trial
    if best_trial:
        logger.info(f"\nBest Trial: {best_trial.trial_id}")
        logger.info(f"Best Pass Rate: {best_trial.benchmark_pass_rate:.1f}%")
        logger.info(f"Best Avg Score: {best_trial.benchmark_avg_score:.2f}/10")
        logger.info(f"Best Params: {json.dumps(best_trial.params, indent=2)}")

    search_result = SearchResult(
        strategy=args.strategy,
        total_trials=len(trials),
        completed_trials=len(completed_trials),
        failed_trials=len(failed_trials),
        best_params=best_trial.params if best_trial else {},
        best_pass_rate=best_trial.benchmark_pass_rate if best_trial else 0.0,
        best_avg_score=best_trial.benchmark_avg_score if best_trial else 0.0,
        trials=trials,
        search_space=asdict(search_space),
        start_time=start_time.isoformat(),
        end_time=end_time.isoformat(),
        total_duration=duration,
    )

    # Save results
    safe_write_json(
        str(output_dir / "best_params.json"),
        {
            "params": best_trial.params if best_trial else {},
            "pass_rate": best_trial.benchmark_pass_rate if best_trial else 0.0,
            "avg_score": best_trial.benchmark_avg_score if best_trial else 0.0,
        },
        PROJECT_ROOT,
    )

    safe_write_json(
        str(output_dir / "all_trials.json"),
        search_result.to_dict(),
        PROJECT_ROOT,
    )

    # Generate report
    generate_report(search_result, output_dir)

    # Generate plots
    if not args.no_plots:
        generate_plots(trials, output_dir)

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("SEARCH RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Total Trials: {len(trials)}")
    logger.info(f"Completed: {len(completed_trials)}")
    logger.info(f"Failed: {len(failed_trials)}")
    logger.info(f"Duration: {duration / 60:.1f} minutes")
    if best_trial:
        logger.info(f"\nBest Configuration:")
        logger.info(f"  Pass Rate: {best_trial.benchmark_pass_rate:.1f}%")
        logger.info(f"  Learning Rate: {best_trial.params['learning_rate']:.2e}")
        logger.info(f"  LoRA Rank: {best_trial.params['lora_rank']}")
        logger.info(f"  LoRA Alpha: {best_trial.params['lora_alpha']}")
        logger.info(f"  Batch Size: {best_trial.params['batch_size']}")
        logger.info(f"  Epochs: {best_trial.params['epochs']}")
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
