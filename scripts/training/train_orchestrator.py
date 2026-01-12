#!/usr/bin/env python3
"""
Multi-Model Training Orchestrator

Orchestrates training multiple base models in sequence or parallel to find
the best model for L4D2 code generation.

Supports:
- Mistral-7B (default, optimized for instruction following)
- CodeLlama-7B (code-focused architecture)
- Qwen-7B (state-of-the-art code generation)
- Llama-3-8B (newer architecture, improved reasoning)

For each model:
1. Loads appropriate config (creates if needed)
2. Runs fine-tuning with specified dataset
3. Exports to GGUF for Ollama
4. Runs benchmark evaluation
5. Uploads results to Object Storage (optional)

Usage:
    # Train specific models
    python train_orchestrator.py --models mistral,codellama --epochs 3

    # Train all models
    python train_orchestrator.py --all-models

    # Compare existing models (skip training)
    python train_orchestrator.py --compare-only

    # Parallel training (requires multiple GPUs)
    python train_orchestrator.py --models mistral,codellama --parallel

    # Dry run (show what would happen)
    python train_orchestrator.py --all-models --dry-run
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent to path for security utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import (
    safe_path,
    safe_read_json,
    safe_read_yaml,
    safe_write_json,
    safe_write_text,
)

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
OUTPUT_DIR = PROJECT_ROOT / "model_adapters"
EXPORTS_DIR = PROJECT_ROOT / "exports"
RESULTS_DIR = PROJECT_ROOT / "results"


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

@dataclass
class ModelSpec:
    """Specification for a base model."""
    name: str
    display_name: str
    unsloth_model: str
    original_model: str  # For GGUF export mapping
    config_file: str
    description: str
    recommended_batch_size: int = 4
    recommended_lr: float = 2e-4
    lora_rank: int = 32
    lora_alpha: int = 64


# Supported base models
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "mistral": ModelSpec(
        name="mistral",
        display_name="Mistral-7B-Instruct",
        unsloth_model="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        original_model="mistralai/Mistral-7B-Instruct-v0.3",
        config_file="unsloth_config.yaml",
        description="Excellent instruction following, balanced performance",
        recommended_batch_size=8,
        recommended_lr=2e-4,
    ),
    "codellama": ModelSpec(
        name="codellama",
        display_name="CodeLlama-7B",
        unsloth_model="unsloth/codellama-7b-bnb-4bit",
        original_model="codellama/CodeLlama-7b-hf",
        config_file="unsloth_config_codellama.yaml",
        description="Code-focused architecture, strong at syntax",
        recommended_batch_size=8,
        recommended_lr=2e-4,
    ),
    "qwen": ModelSpec(
        name="qwen",
        display_name="Qwen2.5-Coder-7B",
        unsloth_model="unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
        original_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        config_file="unsloth_config_qwen.yaml",
        description="State-of-the-art code generation (2024-2025)",
        recommended_batch_size=6,
        recommended_lr=1.5e-4,
        lora_rank=64,
        lora_alpha=128,
    ),
    "llama3": ModelSpec(
        name="llama3",
        display_name="Llama-3-8B-Instruct",
        unsloth_model="unsloth/llama-3-8b-instruct-bnb-4bit",
        original_model="meta-llama/Meta-Llama-3-8B-Instruct",
        config_file="unsloth_config_llama3.yaml",
        description="Newer architecture, improved reasoning",
        recommended_batch_size=6,
        recommended_lr=2e-4,
    ),
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TrainingResult:
    """Result of training a single model."""
    model_name: str
    model_spec: str
    success: bool
    training_time: float
    final_loss: Optional[float] = None
    train_samples: int = 0
    output_dir: Optional[str] = None
    error: Optional[str] = None
    checkpoint_path: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExportResult:
    """Result of exporting a model to GGUF."""
    model_name: str
    success: bool
    gguf_path: Optional[str] = None
    gguf_size_mb: Optional[float] = None
    export_time: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Result of running benchmark evaluation."""
    model_name: str
    success: bool
    pass_rate: float = 0.0
    average_score: float = 0.0
    total_tests: int = 0
    passed_tests: int = 0
    benchmark_time: float = 0.0
    by_category: Dict[str, Dict] = field(default_factory=dict)
    by_difficulty: Dict[str, Dict] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelComparison:
    """Comparison result for a model."""
    model_name: str
    display_name: str
    training_time: float
    final_loss: Optional[float]
    benchmark_score: float
    pass_rate: float
    gguf_size_mb: Optional[float]
    overall_score: float  # Weighted composite score
    recommendation: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OrchestratorReport:
    """Complete orchestration report."""
    timestamp: str
    dataset_version: str
    models_trained: List[str]
    training_results: List[Dict]
    export_results: List[Dict]
    benchmark_results: List[Dict]
    comparison: List[Dict]
    recommended_model: str
    recommendation_reason: str
    total_time: float

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "dataset_version": self.dataset_version,
            "models_trained": self.models_trained,
            "training_results": self.training_results,
            "export_results": self.export_results,
            "benchmark_results": self.benchmark_results,
            "comparison": self.comparison,
            "recommended_model": self.recommended_model,
            "recommendation_reason": self.recommendation_reason,
            "total_time": self.total_time,
        }


# =============================================================================
# CONFIG GENERATION
# =============================================================================

def generate_model_config(
    model_spec: ModelSpec,
    dataset_version: str = "v13",
    epochs: int = 3,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """Generate a training config for a specific model."""
    bs = batch_size or model_spec.recommended_batch_size
    lr = learning_rate or model_spec.recommended_lr

    config = {
        "model": {
            "name": model_spec.unsloth_model,
            "max_seq_length": 4096,
            "dtype": None,  # Auto-detect
            "load_in_4bit": True,
        },
        "lora": {
            "r": model_spec.lora_rank,
            "lora_alpha": model_spec.lora_alpha,
            "lora_dropout": 0,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            "bias": "none",
            "use_gradient_checkpointing": "unsloth",
            "use_rslora": False,
        },
        "training": {
            "num_train_epochs": epochs,
            "per_device_train_batch_size": bs,
            "gradient_accumulation_steps": max(1, 16 // bs),
            "learning_rate": lr,
            "weight_decay": 0.01,
            "warmup_steps": 50,
            "lr_scheduler_type": "cosine",
            "optim": "adamw_8bit",
            "fp16": False,
            "bf16": True,
            "logging_steps": 10,
            "save_steps": 100,
            "save_total_limit": 3,
            "seed": 3407,
            "max_grad_norm": 1.0,
        },
        "data": {
            "train_file": f"l4d2_train_{dataset_version}.jsonl",
            "val_file": "combined_val.jsonl",
            "max_samples": None,
        },
        "output": {
            "dir": f"l4d2-{model_spec.name}-{dataset_version}-lora",
            "push_to_hub": False,
            "hub_model_id": None,
        },
        "advanced": {
            "use_flash_attention_2": True,
            "dataloader_num_workers": 4,
            "dataloader_pin_memory": True,
            "tf32": True,
        },
        "monitoring": {
            "report_to": "tensorboard",
            "logging_dir": "data/training_logs",
            "evaluation_strategy": "steps",
            "eval_steps": 100,
        },
    }

    return config


def ensure_config_exists(model_spec: ModelSpec, dataset_version: str = "v13", **kwargs) -> Path:
    """Ensure config file exists for the model, creating if needed."""
    config_path = CONFIG_DIR / model_spec.config_file

    # Check if config exists
    if config_path.exists():
        logger.info(f"Using existing config: {config_path}")
        return config_path

    # Generate config
    logger.info(f"Generating config for {model_spec.display_name}...")
    config = generate_model_config(model_spec, dataset_version, **kwargs)

    # Write config as YAML
    import yaml
    config_content = f"""# L4D2-AI-Architect: {model_spec.display_name} Training Configuration
#
# Auto-generated by train_orchestrator.py
# Model: {model_spec.unsloth_model}
# Description: {model_spec.description}
#

"""
    config_content += yaml.dump(config, default_flow_style=False, sort_keys=False)

    safe_write_text(str(config_path), config_content, PROJECT_ROOT)
    logger.info(f"Config created: {config_path}")

    return config_path


# =============================================================================
# TRAINING
# =============================================================================

def train_model(
    model_spec: ModelSpec,
    config_path: Path,
    dataset_version: str = "v13",
    resume_from: Optional[str] = None,
    dry_run: bool = False,
) -> TrainingResult:
    """Train a single model."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {model_spec.display_name}")
    logger.info(f"Config: {config_path}")
    logger.info(f"{'='*60}")

    start_time = time.time()

    if dry_run:
        logger.info("[DRY RUN] Would run training...")
        return TrainingResult(
            model_name=model_spec.name,
            model_spec=model_spec.display_name,
            success=True,
            training_time=0.0,
            output_dir=f"model_adapters/l4d2-{model_spec.name}-{dataset_version}-lora",
        )

    # Build command
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "training" / "train_unsloth.py"),
        "--config", str(config_path),
    ]

    if resume_from:
        cmd.extend(["--resume", resume_from])

    try:
        # Run training
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
        )

        elapsed = time.time() - start_time

        if result.returncode != 0:
            logger.error(f"Training failed for {model_spec.name}")
            logger.error(f"STDERR: {result.stderr[-2000:]}")
            return TrainingResult(
                model_name=model_spec.name,
                model_spec=model_spec.display_name,
                success=False,
                training_time=elapsed,
                error=result.stderr[-500:],
            )

        # Parse output for final loss
        final_loss = None
        output_dir = None
        for line in result.stdout.split("\n"):
            if "loss" in line.lower() and ":" in line:
                try:
                    loss_str = line.split(":")[-1].strip()
                    final_loss = float(loss_str.split()[0])
                except (ValueError, IndexError):
                    pass
            if "Model saved to" in line:
                output_dir = line.split("Model saved to")[-1].strip()

        # Load training info if available
        train_samples = 0
        config = safe_read_yaml(str(config_path), PROJECT_ROOT)
        expected_output = OUTPUT_DIR / config["output"]["dir"]
        info_path = expected_output / "training_info.json"

        if info_path.exists():
            info = safe_read_json(str(info_path), PROJECT_ROOT)
            train_samples = info.get("train_samples", 0)

        return TrainingResult(
            model_name=model_spec.name,
            model_spec=model_spec.display_name,
            success=True,
            training_time=elapsed,
            final_loss=final_loss,
            train_samples=train_samples,
            output_dir=str(expected_output),
            checkpoint_path=str(expected_output / "final"),
        )

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return TrainingResult(
            model_name=model_spec.name,
            model_spec=model_spec.display_name,
            success=False,
            training_time=elapsed,
            error="Training timed out after 2 hours",
        )
    except Exception as e:
        elapsed = time.time() - start_time
        return TrainingResult(
            model_name=model_spec.name,
            model_spec=model_spec.display_name,
            success=False,
            training_time=elapsed,
            error=str(e),
        )


# =============================================================================
# EXPORT
# =============================================================================

def export_model(
    model_spec: ModelSpec,
    training_result: TrainingResult,
    dataset_version: str = "v13",
    quantization: str = "q4_k_m",
    dry_run: bool = False,
) -> ExportResult:
    """Export trained model to GGUF format."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Exporting: {model_spec.display_name} to GGUF")
    logger.info(f"{'='*60}")

    start_time = time.time()

    if dry_run:
        logger.info("[DRY RUN] Would export to GGUF...")
        return ExportResult(
            model_name=model_spec.name,
            success=True,
            gguf_path=f"exports/l4d2-{model_spec.name}-{dataset_version}/gguf",
            gguf_size_mb=4000.0,
            export_time=0.0,
        )

    if not training_result.success or not training_result.checkpoint_path:
        return ExportResult(
            model_name=model_spec.name,
            success=False,
            error="No valid checkpoint to export",
        )

    # Build export command
    output_path = EXPORTS_DIR / f"l4d2-{model_spec.name}-{dataset_version}"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "training" / "export_gguf_cpu.py"),
        "--adapter", training_result.checkpoint_path,
        "--output", str(output_path),
        "--quantize", quantization,
        "--create-modelfile",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        elapsed = time.time() - start_time

        if result.returncode != 0:
            logger.error(f"Export failed for {model_spec.name}")
            return ExportResult(
                model_name=model_spec.name,
                success=False,
                export_time=elapsed,
                error=result.stderr[-500:],
            )

        # Find GGUF file and get size
        gguf_dir = output_path / "gguf"
        gguf_size_mb = None
        gguf_path = None

        if gguf_dir.exists():
            for f in gguf_dir.glob("*.gguf"):
                gguf_path = str(f)
                gguf_size_mb = f.stat().st_size / (1024 * 1024)
                break

        return ExportResult(
            model_name=model_spec.name,
            success=True,
            gguf_path=gguf_path or str(gguf_dir),
            gguf_size_mb=gguf_size_mb,
            export_time=elapsed,
        )

    except subprocess.TimeoutExpired:
        return ExportResult(
            model_name=model_spec.name,
            success=False,
            export_time=time.time() - start_time,
            error="Export timed out after 1 hour",
        )
    except Exception as e:
        return ExportResult(
            model_name=model_spec.name,
            success=False,
            export_time=time.time() - start_time,
            error=str(e),
        )


# =============================================================================
# BENCHMARK
# =============================================================================

def run_benchmark(
    model_spec: ModelSpec,
    export_result: ExportResult,
    dataset_version: str = "v13",
    quick: bool = False,
    dry_run: bool = False,
) -> BenchmarkResult:
    """Run benchmark evaluation on exported model."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking: {model_spec.display_name}")
    logger.info(f"{'='*60}")

    start_time = time.time()

    if dry_run:
        logger.info("[DRY RUN] Would run benchmark...")
        return BenchmarkResult(
            model_name=model_spec.name,
            success=True,
            pass_rate=75.0,
            average_score=7.5,
            total_tests=55,
            passed_tests=41,
            benchmark_time=0.0,
        )

    if not export_result.success:
        return BenchmarkResult(
            model_name=model_spec.name,
            success=False,
            error="No valid export to benchmark",
        )

    # Check if model is installed in Ollama
    ollama_model_name = f"l4d2-{model_spec.name}-{dataset_version}"

    # First, try to install to Ollama
    if export_result.gguf_path:
        gguf_dir = Path(export_result.gguf_path).parent
        modelfile_path = gguf_dir / "Modelfile"

        if modelfile_path.exists():
            install_result = subprocess.run(
                ["ollama", "create", ollama_model_name, "-f", str(modelfile_path)],
                capture_output=True,
                text=True,
                cwd=str(gguf_dir),
            )
            if install_result.returncode != 0:
                logger.warning(f"Failed to install {ollama_model_name} to Ollama")

    # Run benchmark
    output_file = RESULTS_DIR / f"benchmark_{model_spec.name}_{dataset_version}.json"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluation" / "benchmark_suite.py"),
        "--model", "ollama",
        "--model-name", ollama_model_name,
        "--output", str(output_file),
    ]

    if quick:
        cmd.append("--quick")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        elapsed = time.time() - start_time

        # Parse results from output file
        if output_file.exists():
            benchmark_data = safe_read_json(str(output_file), PROJECT_ROOT)

            return BenchmarkResult(
                model_name=model_spec.name,
                success=True,
                pass_rate=benchmark_data.get("pass_rate", 0),
                average_score=benchmark_data.get("average_score", 0),
                total_tests=benchmark_data.get("total_tests", 0),
                passed_tests=benchmark_data.get("passed", 0),
                benchmark_time=elapsed,
                by_category=benchmark_data.get("by_category", {}),
                by_difficulty=benchmark_data.get("by_difficulty", {}),
            )
        else:
            return BenchmarkResult(
                model_name=model_spec.name,
                success=False,
                benchmark_time=elapsed,
                error="Benchmark output file not found",
            )

    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            model_name=model_spec.name,
            success=False,
            benchmark_time=time.time() - start_time,
            error="Benchmark timed out after 1 hour",
        )
    except Exception as e:
        return BenchmarkResult(
            model_name=model_spec.name,
            success=False,
            benchmark_time=time.time() - start_time,
            error=str(e),
        )


# =============================================================================
# COMPARISON AND RECOMMENDATION
# =============================================================================

def calculate_overall_score(
    training_result: TrainingResult,
    benchmark_result: BenchmarkResult,
    export_result: ExportResult,
) -> float:
    """Calculate weighted overall score for model comparison."""
    # Weights
    BENCHMARK_WEIGHT = 0.50  # Benchmark score is most important
    PASS_RATE_WEIGHT = 0.25  # Pass rate matters
    TRAINING_EFFICIENCY_WEIGHT = 0.15  # Faster training is better
    SIZE_WEIGHT = 0.10  # Smaller model is slightly preferred

    score = 0.0

    # Benchmark score (0-10, normalize to 0-100)
    if benchmark_result.success:
        score += benchmark_result.average_score * 10 * BENCHMARK_WEIGHT

    # Pass rate (0-100)
    if benchmark_result.success:
        score += benchmark_result.pass_rate * PASS_RATE_WEIGHT

    # Training efficiency (inverse of time, normalized)
    # Assume 2 hours = 0 bonus, 30 min = full bonus
    if training_result.success and training_result.training_time > 0:
        time_score = max(0, min(100, (7200 - training_result.training_time) / 54))
        score += time_score * TRAINING_EFFICIENCY_WEIGHT

    # Size efficiency (smaller is better)
    # Assume 5GB = 0 bonus, 2GB = full bonus
    if export_result.success and export_result.gguf_size_mb:
        size_score = max(0, min(100, (5000 - export_result.gguf_size_mb) / 30))
        score += size_score * SIZE_WEIGHT

    return score


def generate_comparison(
    models: List[str],
    training_results: Dict[str, TrainingResult],
    export_results: Dict[str, ExportResult],
    benchmark_results: Dict[str, BenchmarkResult],
) -> Tuple[List[ModelComparison], str, str]:
    """Generate model comparison and recommendation."""
    comparisons = []

    for model_name in models:
        spec = MODEL_REGISTRY[model_name]
        train_res = training_results.get(model_name)
        export_res = export_results.get(model_name)
        bench_res = benchmark_results.get(model_name)

        if not train_res:
            continue

        overall_score = 0.0
        if train_res and export_res and bench_res:
            overall_score = calculate_overall_score(train_res, bench_res, export_res)

        # Generate recommendation text
        if bench_res and bench_res.success:
            if bench_res.pass_rate >= 80:
                recommendation = "Excellent - highly recommended"
            elif bench_res.pass_rate >= 60:
                recommendation = "Good - suitable for production"
            elif bench_res.pass_rate >= 40:
                recommendation = "Fair - needs improvement"
            else:
                recommendation = "Poor - not recommended"
        else:
            recommendation = "Unable to evaluate"

        comparisons.append(ModelComparison(
            model_name=model_name,
            display_name=spec.display_name,
            training_time=train_res.training_time if train_res else 0,
            final_loss=train_res.final_loss if train_res else None,
            benchmark_score=bench_res.average_score if bench_res and bench_res.success else 0,
            pass_rate=bench_res.pass_rate if bench_res and bench_res.success else 0,
            gguf_size_mb=export_res.gguf_size_mb if export_res and export_res.success else None,
            overall_score=overall_score,
            recommendation=recommendation,
        ))

    # Sort by overall score
    comparisons.sort(key=lambda x: x.overall_score, reverse=True)

    # Determine recommended model
    if comparisons and comparisons[0].overall_score > 0:
        recommended = comparisons[0]
        recommended_model = recommended.model_name
        recommendation_reason = (
            f"{recommended.display_name} scored highest with "
            f"{recommended.pass_rate:.1f}% pass rate, "
            f"{recommended.benchmark_score:.2f}/10 average score, "
            f"trained in {recommended.training_time/60:.1f} minutes"
        )
    else:
        recommended_model = "none"
        recommendation_reason = "Unable to determine recommendation - check benchmark results"

    return comparisons, recommended_model, recommendation_reason


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_markdown_report(report: OrchestratorReport, output_path: Path) -> None:
    """Generate a Markdown comparison report."""
    md = f"""# L4D2 Multi-Model Training Report

**Date**: {report.timestamp}
**Dataset Version**: {report.dataset_version}
**Total Time**: {report.total_time/60:.1f} minutes

## Recommended Model

**{report.recommended_model.upper()}**

{report.recommendation_reason}

## Model Comparison

| Model | Pass Rate | Avg Score | Training Time | GGUF Size | Overall Score |
|-------|-----------|-----------|---------------|-----------|---------------|
"""
    for comp in report.comparison:
        train_time = f"{comp['training_time']/60:.1f}m" if comp['training_time'] > 0 else "N/A"
        gguf_size = f"{comp['gguf_size_mb']:.0f}MB" if comp['gguf_size_mb'] else "N/A"
        md += f"| {comp['display_name']} | {comp['pass_rate']:.1f}% | {comp['benchmark_score']:.2f}/10 | {train_time} | {gguf_size} | {comp['overall_score']:.1f} |\n"

    md += """

## Training Results

| Model | Status | Time | Final Loss | Samples |
|-------|--------|------|------------|---------|
"""
    for result in report.training_results:
        status = "Success" if result['success'] else "Failed"
        time_str = f"{result['training_time']/60:.1f}m"
        loss = f"{result['final_loss']:.4f}" if result['final_loss'] else "N/A"
        samples = result['train_samples'] if result['train_samples'] else "N/A"
        md += f"| {result['model_spec']} | {status} | {time_str} | {loss} | {samples} |\n"

    md += """

## Export Results

| Model | Status | GGUF Size | Export Time |
|-------|--------|-----------|-------------|
"""
    for result in report.export_results:
        status = "Success" if result['success'] else "Failed"
        size = f"{result['gguf_size_mb']:.0f}MB" if result['gguf_size_mb'] else "N/A"
        time_str = f"{result['export_time']/60:.1f}m"
        md += f"| {result['model_name']} | {status} | {size} | {time_str} |\n"

    md += """

## Benchmark Results

| Model | Status | Pass Rate | Avg Score | Tests |
|-------|--------|-----------|-----------|-------|
"""
    for result in report.benchmark_results:
        status = "Success" if result['success'] else "Failed"
        pass_rate = f"{result['pass_rate']:.1f}%" if result['success'] else "N/A"
        avg_score = f"{result['average_score']:.2f}/10" if result['success'] else "N/A"
        tests = f"{result['passed_tests']}/{result['total_tests']}" if result['success'] else "N/A"
        md += f"| {result['model_name']} | {status} | {pass_rate} | {avg_score} | {tests} |\n"

    md += """

## Recommendations

Based on the comparison above:

1. **For Production Use**: Choose the model with the highest pass rate and benchmark score
2. **For Fast Iteration**: Consider training time and model size
3. **For Quality Focus**: Prioritize benchmark score over other metrics

---
*Generated by L4D2-AI-Architect Multi-Model Orchestrator*
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_write_text(str(output_path), md, PROJECT_ROOT)
    logger.info(f"Markdown report saved to: {output_path}")


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class TrainingOrchestrator:
    """Orchestrates training of multiple models."""

    def __init__(
        self,
        models: List[str],
        dataset_version: str = "v13",
        epochs: int = 3,
        parallel: bool = False,
        dry_run: bool = False,
        skip_training: bool = False,
        skip_export: bool = False,
        skip_benchmark: bool = False,
        quick_benchmark: bool = False,
    ):
        self.models = models
        self.dataset_version = dataset_version
        self.epochs = epochs
        self.parallel = parallel
        self.dry_run = dry_run
        self.skip_training = skip_training
        self.skip_export = skip_export
        self.skip_benchmark = skip_benchmark
        self.quick_benchmark = quick_benchmark

        self.training_results: Dict[str, TrainingResult] = {}
        self.export_results: Dict[str, ExportResult] = {}
        self.benchmark_results: Dict[str, BenchmarkResult] = {}

    def run(self) -> OrchestratorReport:
        """Run the full orchestration pipeline."""
        start_time = time.time()

        logger.info(f"\n{'='*60}")
        logger.info("L4D2 Multi-Model Training Orchestrator")
        logger.info(f"{'='*60}")
        logger.info(f"Models: {', '.join(self.models)}")
        logger.info(f"Dataset: {self.dataset_version}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"Parallel: {self.parallel}")
        logger.info(f"Dry Run: {self.dry_run}")

        # Ensure configs exist
        for model_name in self.models:
            spec = MODEL_REGISTRY[model_name]
            ensure_config_exists(spec, self.dataset_version, epochs=self.epochs)

        # Training phase
        if not self.skip_training:
            if self.parallel:
                self._train_parallel()
            else:
                self._train_sequential()
        else:
            logger.info("Skipping training phase")
            self._load_existing_results()

        # Export phase
        if not self.skip_export:
            self._export_all()
        else:
            logger.info("Skipping export phase")

        # Benchmark phase
        if not self.skip_benchmark:
            self._benchmark_all()
        else:
            logger.info("Skipping benchmark phase")

        # Generate comparison
        comparisons, recommended, reason = generate_comparison(
            self.models,
            self.training_results,
            self.export_results,
            self.benchmark_results,
        )

        total_time = time.time() - start_time

        # Generate report
        report = OrchestratorReport(
            timestamp=datetime.now().isoformat(),
            dataset_version=self.dataset_version,
            models_trained=self.models,
            training_results=[r.to_dict() for r in self.training_results.values()],
            export_results=[r.to_dict() for r in self.export_results.values()],
            benchmark_results=[r.to_dict() for r in self.benchmark_results.values()],
            comparison=[c.to_dict() for c in comparisons],
            recommended_model=recommended,
            recommendation_reason=reason,
            total_time=total_time,
        )

        # Save results
        self._save_report(report)

        return report

    def _train_sequential(self) -> None:
        """Train models sequentially."""
        for model_name in self.models:
            spec = MODEL_REGISTRY[model_name]
            config_path = CONFIG_DIR / spec.config_file

            result = train_model(
                spec,
                config_path,
                self.dataset_version,
                dry_run=self.dry_run,
            )
            self.training_results[model_name] = result

    def _train_parallel(self) -> None:
        """Train models in parallel (requires multiple GPUs)."""
        # Note: This is a simplified parallel implementation
        # For true multi-GPU training, you'd need CUDA_VISIBLE_DEVICES management
        logger.warning("Parallel training requires multiple GPUs with proper CUDA device management")

        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            futures = {}
            for i, model_name in enumerate(self.models):
                spec = MODEL_REGISTRY[model_name]
                config_path = CONFIG_DIR / spec.config_file

                # Submit training job
                future = executor.submit(
                    train_model,
                    spec,
                    config_path,
                    self.dataset_version,
                    None,  # resume_from
                    self.dry_run,
                )
                futures[future] = model_name

            # Collect results
            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    result = future.result()
                    self.training_results[model_name] = result
                except Exception as e:
                    self.training_results[model_name] = TrainingResult(
                        model_name=model_name,
                        model_spec=MODEL_REGISTRY[model_name].display_name,
                        success=False,
                        training_time=0,
                        error=str(e),
                    )

    def _load_existing_results(self) -> None:
        """Load existing training results for compare-only mode."""
        for model_name in self.models:
            spec = MODEL_REGISTRY[model_name]
            output_dir = OUTPUT_DIR / f"l4d2-{spec.name}-{self.dataset_version}-lora"
            info_path = output_dir / "training_info.json"

            if info_path.exists():
                info = safe_read_json(str(info_path), PROJECT_ROOT)

                self.training_results[model_name] = TrainingResult(
                    model_name=model_name,
                    model_spec=spec.display_name,
                    success=True,
                    training_time=0,  # Not available from saved info
                    train_samples=info.get("train_samples", 0),
                    output_dir=str(output_dir),
                    checkpoint_path=str(output_dir / "final"),
                )
            else:
                logger.warning(f"No existing results found for {model_name}")

    def _export_all(self) -> None:
        """Export all successfully trained models."""
        for model_name in self.models:
            spec = MODEL_REGISTRY[model_name]
            train_result = self.training_results.get(model_name)

            if train_result and train_result.success:
                result = export_model(
                    spec,
                    train_result,
                    self.dataset_version,
                    dry_run=self.dry_run,
                )
                self.export_results[model_name] = result
            else:
                self.export_results[model_name] = ExportResult(
                    model_name=model_name,
                    success=False,
                    error="Training was not successful",
                )

    def _benchmark_all(self) -> None:
        """Run benchmarks on all exported models."""
        for model_name in self.models:
            spec = MODEL_REGISTRY[model_name]
            export_result = self.export_results.get(model_name)

            if export_result and export_result.success:
                result = run_benchmark(
                    spec,
                    export_result,
                    self.dataset_version,
                    quick=self.quick_benchmark,
                    dry_run=self.dry_run,
                )
                self.benchmark_results[model_name] = result
            else:
                self.benchmark_results[model_name] = BenchmarkResult(
                    model_name=model_name,
                    success=False,
                    error="Export was not successful",
                )

    def _save_report(self, report: OrchestratorReport) -> None:
        """Save orchestration report."""
        # Save JSON report
        json_path = safe_path(
            f"results/orchestrator_{self.dataset_version}.json",
            PROJECT_ROOT,
            create_parents=True,
        )
        safe_write_json(str(json_path), report.to_dict(), PROJECT_ROOT)
        logger.info(f"JSON report saved to: {json_path}")

        # Save Markdown report
        md_path = safe_path(
            f"results/orchestrator_{self.dataset_version}.md",
            PROJECT_ROOT,
            create_parents=True,
        )
        generate_markdown_report(report, md_path)

        # Print summary
        print(f"\n{'='*60}")
        print("ORCHESTRATION COMPLETE")
        print(f"{'='*60}")
        print(f"Models Trained: {len(self.training_results)}")
        print(f"Models Exported: {sum(1 for r in self.export_results.values() if r.success)}")
        print(f"Models Benchmarked: {sum(1 for r in self.benchmark_results.values() if r.success)}")
        print(f"Total Time: {report.total_time/60:.1f} minutes")
        print(f"\nRecommended Model: {report.recommended_model.upper()}")
        print(f"Reason: {report.recommendation_reason}")
        print(f"\nReports saved to:")
        print(f"  - {json_path}")
        print(f"  - {md_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Model Training Orchestrator for L4D2 Code Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train Mistral and CodeLlama
    python train_orchestrator.py --models mistral,codellama --epochs 3

    # Train all supported models
    python train_orchestrator.py --all-models

    # Compare existing models (skip training)
    python train_orchestrator.py --compare-only --models mistral,codellama

    # Dry run to see what would happen
    python train_orchestrator.py --all-models --dry-run

    # Quick benchmark after training
    python train_orchestrator.py --models mistral --quick-benchmark
        """
    )

    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of models to train (mistral,codellama,qwen,llama3)",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Train all supported models",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="v13",
        help="Dataset version to use (default: v13)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Train models in parallel (requires multiple GPUs)",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Skip training, only compare existing models",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip GGUF export phase",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip benchmark evaluation phase",
    )
    parser.add_argument(
        "--quick-benchmark",
        action="store_true",
        help="Run quick benchmark (subset of tests)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without executing",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List supported models and exit",
    )

    args = parser.parse_args()

    # List models mode
    if args.list_models:
        print("\nSupported Models:")
        print("-" * 60)
        for name, spec in MODEL_REGISTRY.items():
            print(f"\n{name}:")
            print(f"  Display Name: {spec.display_name}")
            print(f"  Unsloth Model: {spec.unsloth_model}")
            print(f"  Description: {spec.description}")
            print(f"  Config: {spec.config_file}")
        return

    # Determine models to train
    if args.all_models:
        models = list(MODEL_REGISTRY.keys())
    elif args.models:
        models = [m.strip() for m in args.models.split(",")]
        # Validate models
        for m in models:
            if m not in MODEL_REGISTRY:
                print(f"Error: Unknown model '{m}'")
                print(f"Supported models: {list(MODEL_REGISTRY.keys())}")
                sys.exit(1)
    else:
        print("Error: Specify --models or --all-models")
        parser.print_help()
        sys.exit(1)

    # Create orchestrator
    orchestrator = TrainingOrchestrator(
        models=models,
        dataset_version=args.dataset,
        epochs=args.epochs,
        parallel=args.parallel,
        dry_run=args.dry_run,
        skip_training=args.compare_only,
        skip_export=args.skip_export,
        skip_benchmark=args.skip_benchmark,
        quick_benchmark=args.quick_benchmark,
    )

    # Run orchestration
    report = orchestrator.run()

    # Exit with appropriate code
    if report.recommended_model != "none":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
