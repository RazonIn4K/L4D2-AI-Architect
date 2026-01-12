#!/usr/bin/env python3
"""
L4D2-AI-Architect: Post-Training Automation

Automates the entire post-training pipeline:
1. Verify training completed successfully
2. Export to GGUF for Ollama
3. Run evaluation benchmarks
4. Generate comparison report
5. Package for download

Usage:
    python post_training.py                           # Process all completed models
    python post_training.py --model l4d2-mistral-v15  # Process specific model
    python post_training.py --export-only             # Only export, skip evaluation
    python post_training.py --eval-only               # Only evaluate existing exports
"""

import os
import sys
import json
import argparse
import logging
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_write_json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
ADAPTERS_DIR = PROJECT_ROOT / "model_adapters"
EXPORTS_DIR = PROJECT_ROOT / "exports"
EVAL_DIR = PROJECT_ROOT / "data" / "evaluations"


def find_completed_models() -> List[Path]:
    """Find all models that have completed training."""
    completed = []

    if not ADAPTERS_DIR.exists():
        return completed

    for model_dir in ADAPTERS_DIR.iterdir():
        if not model_dir.is_dir():
            continue

        final_dir = model_dir / "final"
        if final_dir.exists():
            # Check for required files
            has_adapter = (final_dir / "adapter_config.json").exists() or \
                         (final_dir / "config.json").exists()
            if has_adapter:
                completed.append(model_dir)
                logger.info(f"Found completed model: {model_dir.name}")

    return completed


def export_to_gguf(model_dir: Path, quantization: str = "q4_k_m") -> Optional[Path]:
    """Export model to GGUF format."""
    model_name = model_dir.name
    final_path = model_dir / "final"
    export_path = EXPORTS_DIR / model_name

    logger.info(f"Exporting {model_name} to GGUF ({quantization})...")

    export_script = PROJECT_ROOT / "scripts" / "training" / "export_gguf_cpu.py"

    cmd = [
        sys.executable,
        str(export_script),
        "--adapter", str(final_path),
        "--output", str(export_path),
        "--quantize", quantization,
        "--create-modelfile",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Export completed: {export_path}")

        # Find the GGUF file
        gguf_dir = export_path / "gguf"
        if gguf_dir.exists():
            gguf_files = list(gguf_dir.glob("*.gguf"))
            if gguf_files:
                return gguf_files[0]

        return export_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Export failed: {e.stderr}")
        return None


def run_evaluation(model_path: Path, backend: str = "unsloth") -> Optional[Dict]:
    """Run model evaluation."""
    logger.info(f"Running evaluation on {model_path.name}...")

    eval_script = PROJECT_ROOT / "scripts" / "evaluation" / "evaluate_model.py"

    if not eval_script.exists():
        logger.warning("Evaluation script not found, skipping...")
        return None

    output_file = EVAL_DIR / f"{model_path.name}_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(eval_script),
        "--model", str(model_path / "final"),
        "--backend", backend,
        "--output", str(output_file),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if output_file.exists():
            with open(output_file, 'r') as f:
                return json.load(f)
        else:
            logger.warning("Evaluation output not generated")
            return None
    except subprocess.TimeoutExpired:
        logger.error("Evaluation timed out after 10 minutes")
        return None
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return None


def create_ollama_install_script(model_name: str, gguf_path: Path) -> Path:
    """Create a script to install the model in Ollama."""
    script_path = gguf_path.parent / "install_ollama.sh"

    modelfile_path = gguf_path.parent / "Modelfile"

    script_content = f"""#!/bin/bash
# Install {model_name} in Ollama

set -e

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

echo "Installing {model_name} to Ollama..."

if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Please install from https://ollama.ai"
    exit 1
fi

cd "$SCRIPT_DIR"

if [ -f "Modelfile" ]; then
    ollama create {model_name} -f Modelfile
    echo ""
    echo "Success! Run: ollama run {model_name}"
else
    echo "Modelfile not found!"
    exit 1
fi
"""

    with open(script_path, 'w') as f:
        f.write(script_content)

    script_path.chmod(0o755)
    return script_path


def package_for_download(model_name: str) -> Optional[Path]:
    """Create a downloadable package of the model."""
    export_path = EXPORTS_DIR / model_name

    if not export_path.exists():
        logger.warning(f"Export path not found: {export_path}")
        return None

    # Create archive
    archive_name = f"{model_name}_package"
    archive_path = EXPORTS_DIR / archive_name

    logger.info(f"Creating package: {archive_path}.tar.gz")

    try:
        shutil.make_archive(str(archive_path), 'gztar', str(export_path.parent), model_name)
        final_path = Path(f"{archive_path}.tar.gz")
        size_mb = final_path.stat().st_size / 1024 / 1024
        logger.info(f"Package created: {final_path} ({size_mb:.1f} MB)")
        return final_path
    except Exception as e:
        logger.error(f"Packaging failed: {e}")
        return None


def generate_report(models: List[Dict]) -> str:
    """Generate a summary report of all processed models."""
    report = []
    report.append("=" * 70)
    report.append("L4D2-AI-ARCHITECT POST-TRAINING REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)
    report.append("")

    for model in models:
        report.append(f"Model: {model['name']}")
        report.append("-" * 40)
        report.append(f"  Training Status: {'Completed' if model['training_completed'] else 'Incomplete'}")

        if model.get('export_path'):
            report.append(f"  Export: {model['export_path']}")
        else:
            report.append("  Export: Failed or skipped")

        if model.get('evaluation'):
            eval_data = model['evaluation']
            report.append(f"  Evaluation Score: {eval_data.get('average_score', 0):.1%}")
            report.append(f"  Tests Passed: {eval_data.get('passed_tests', 0)}/{eval_data.get('total_tests', 0)}")
        else:
            report.append("  Evaluation: Not run")

        if model.get('package_path'):
            report.append(f"  Package: {model['package_path']}")

        report.append("")

    report.append("=" * 70)

    # Comparison if multiple models
    if len(models) > 1:
        evals = [(m['name'], m.get('evaluation', {}).get('average_score', 0))
                 for m in models if m.get('evaluation')]

        if evals:
            report.append("")
            report.append("MODEL RANKING (by evaluation score):")
            for i, (name, score) in enumerate(sorted(evals, key=lambda x: -x[1]), 1):
                report.append(f"  {i}. {name}: {score:.1%}")

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Post-training automation")
    parser.add_argument("--model", type=str, help="Process specific model")
    parser.add_argument("--export-only", action="store_true", help="Only export, skip evaluation")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate existing exports")
    parser.add_argument("--no-package", action="store_true", help="Skip creating download package")
    parser.add_argument("--quantization", type=str, default="q4_k_m",
                       choices=["q4_0", "q4_k_m", "q5_k_m", "q8_0", "f16"],
                       help="GGUF quantization level")

    args = parser.parse_args()

    print()
    print("=" * 70)
    print("L4D2-AI-ARCHITECT POST-TRAINING AUTOMATION")
    print("=" * 70)
    print()

    # Find models to process
    if args.model:
        model_path = ADAPTERS_DIR / args.model
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            sys.exit(1)
        models_to_process = [model_path]
    else:
        models_to_process = find_completed_models()

    if not models_to_process:
        logger.warning("No completed models found to process")
        sys.exit(0)

    logger.info(f"Processing {len(models_to_process)} model(s)...")

    results = []

    for model_path in models_to_process:
        model_name = model_path.name
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing: {model_name}")
        logger.info(f"{'='*40}")

        model_result = {
            "name": model_name,
            "path": str(model_path),
            "training_completed": (model_path / "final").exists(),
        }

        # Export
        if not args.eval_only:
            gguf_path = export_to_gguf(model_path, args.quantization)
            if gguf_path:
                model_result["export_path"] = str(gguf_path)

                # Create Ollama install script
                create_ollama_install_script(model_name, gguf_path)

        # Evaluate
        if not args.export_only:
            eval_result = run_evaluation(model_path)
            if eval_result:
                model_result["evaluation"] = eval_result

        # Package
        if not args.no_package and not args.eval_only:
            package_path = package_for_download(model_name)
            if package_path:
                model_result["package_path"] = str(package_path)

        results.append(model_result)

    # Generate report
    report = generate_report(results)
    print()
    print(report)

    # Save report
    report_path = PROJECT_ROOT / "data" / "training_logs" / f"post_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)

    logger.info(f"Report saved to {report_path}")

    # Save JSON results
    results_path = report_path.with_suffix('.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
