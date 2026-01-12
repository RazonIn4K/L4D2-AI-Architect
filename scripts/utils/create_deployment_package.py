#!/usr/bin/env python3
"""
Create Deployment Package

Creates self-contained deployment archives for L4D2-AI-Architect:
1. vultr_package.tar.gz - Minimal package for cloud training
2. local_package.tar.gz - Full package for local development

Includes validation, syntax checking, and manifest generation.
"""

import argparse
import hashlib
import json
import os
import py_compile
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add parent to path for security utils
sys.path.insert(0, str(Path(__file__).parent))
from security import safe_path, safe_write_text

PROJECT_ROOT = Path(__file__).parent.parent.parent


def human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def calculate_file_hash(filepath: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]  # Truncated for readability


def validate_jsonl_file(filepath: Path) -> Tuple[bool, int, str]:
    """
    Validate a JSONL file.

    Returns:
        Tuple of (is_valid, line_count, error_message)
    """
    try:
        line_count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        data = json.loads(line)
                        # Validate ChatML format for training data
                        if 'messages' in data:
                            if not isinstance(data['messages'], list):
                                return False, i, f"Line {i}: 'messages' must be a list"
                            for msg in data['messages']:
                                if 'role' not in msg or 'content' not in msg:
                                    return False, i, f"Line {i}: Message missing 'role' or 'content'"
                        line_count += 1
                    except json.JSONDecodeError as e:
                        return False, i, f"Line {i}: Invalid JSON - {e}"
        return True, line_count, "Valid"
    except Exception as e:
        return False, 0, f"Error reading file: {e}"


def validate_python_syntax(filepath: Path) -> Tuple[bool, str]:
    """
    Validate Python file syntax using py_compile.

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        py_compile.compile(str(filepath), doraise=True)
        return True, "Valid"
    except py_compile.PyCompileError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def find_python_scripts(base_dir: Path) -> List[Path]:
    """Find all Python scripts in the scripts directory."""
    scripts = []
    scripts_dir = base_dir / "scripts"
    if scripts_dir.exists():
        for py_file in scripts_dir.rglob("*.py"):
            # Skip __pycache__ and venv
            if "__pycache__" not in str(py_file) and "venv" not in str(py_file):
                scripts.append(py_file)
    return sorted(scripts)


def find_config_files(base_dir: Path) -> List[Path]:
    """Find all YAML config files."""
    configs = []
    config_dir = base_dir / "configs"
    if config_dir.exists():
        for yaml_file in config_dir.glob("*.yaml"):
            configs.append(yaml_file)
    return sorted(configs)


def generate_readme_deploy(
    training_data_file: str,
    training_samples: int,
    configs: List[str],
    timestamp: str
) -> str:
    """Generate deployment quick start guide."""
    config_list = "\n".join(f"  - {c}" for c in configs)

    return f"""# L4D2-AI-Architect Deployment Package

Generated: {timestamp}
Training Data: {training_data_file} ({training_samples:,} samples)

## Quick Start (Vultr A100/A40)

### 1. Upload and Extract
```bash
# Upload to Vultr instance
scp vultr_package.tar.gz root@<VULTR_IP>:/root/

# SSH and extract
ssh root@<VULTR_IP>
cd /root
tar -xzf vultr_package.tar.gz
cd L4D2-AI-Architect
```

### 2. Setup Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements.txt

# Install Unsloth (for QLoRA training)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### 3. Start Training
```bash
# For A100 (40GB)
python scripts/training/train_unsloth.py --config configs/unsloth_config_a100.yaml

# For A40/L40S (48GB)
python scripts/training/train_unsloth.py --config configs/unsloth_config.yaml

# Monitor with TensorBoard
tensorboard --logdir data/training_logs --port 6006
```

## Configuration Files
{config_list}

## Training Data Format (ChatML)
```json
{{"messages": [
  {{"role": "system", "content": "You are an expert SourcePawn developer..."}},
  {{"role": "user", "content": "Write a function to heal survivors"}},
  {{"role": "assistant", "content": "public void HealAllSurvivors()..."}}
]}}
```

## Export Model to Ollama
```bash
# After training completes
python scripts/training/export_gguf_cpu.py \\
    --adapter model_adapters/l4d2-code-lora/final

# Install to Ollama
ollama create l4d2-code -f exports/l4d2-code/gguf/Modelfile
```

## Test the Model
```bash
python scripts/inference/copilot_cli.py ollama \\
    --prompt "Write a plugin that heals all survivors when a tank spawns"
```

## Directory Structure
```
L4D2-AI-Architect/
├── configs/                 # Training configurations
├── data/
│   └── processed/          # Training data (JSONL)
├── scripts/
│   ├── training/           # Training scripts
│   ├── inference/          # Inference/CLI tools
│   ├── rl_training/        # RL bot training
│   └── utils/              # Utility scripts
└── requirements.txt        # Python dependencies
```

## Troubleshooting

### Out of Memory
Reduce batch_size in the config file or use gradient_checkpointing.

### CUDA not found
Ensure PyTorch is installed with CUDA support:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Unsloth import error
Reinstall with compatible versions:
```bash
pip uninstall unsloth -y
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

---
Generated by L4D2-AI-Architect deployment script
"""


def create_vultr_package(
    project_root: Path,
    output_dir: Path,
    training_data: Path,
    configs: List[Path],
    scripts: List[Path],
    requirements: Path,
    verbose: bool = True
) -> Tuple[Path, Dict]:
    """
    Create minimal deployment package for Vultr cloud training.

    Returns:
        Tuple of (archive_path, manifest_dict)
    """
    archive_name = "vultr_package.tar.gz"
    archive_path = output_dir / archive_name

    manifest = {
        "package_type": "vultr",
        "created": datetime.now().isoformat(),
        "files": [],
        "total_size": 0
    }

    # Create tar archive
    with tarfile.open(archive_path, "w:gz") as tar:
        base_name = "L4D2-AI-Architect"

        # Add training data
        if training_data.exists():
            arcname = f"{base_name}/data/processed/{training_data.name}"
            tar.add(training_data, arcname=arcname)
            size = training_data.stat().st_size
            manifest["files"].append({
                "path": arcname,
                "size": size,
                "hash": calculate_file_hash(training_data)
            })
            manifest["total_size"] += size
            if verbose:
                print(f"  + {arcname} ({human_readable_size(size)})")

        # Add config files
        for config in configs:
            arcname = f"{base_name}/configs/{config.name}"
            tar.add(config, arcname=arcname)
            size = config.stat().st_size
            manifest["files"].append({
                "path": arcname,
                "size": size,
                "hash": calculate_file_hash(config)
            })
            manifest["total_size"] += size
            if verbose:
                print(f"  + {arcname} ({human_readable_size(size)})")

        # Add Python scripts (preserving directory structure)
        scripts_dir = project_root / "scripts"
        for script in scripts:
            rel_path = script.relative_to(scripts_dir)
            arcname = f"{base_name}/scripts/{rel_path}"
            tar.add(script, arcname=arcname)
            size = script.stat().st_size
            manifest["files"].append({
                "path": arcname,
                "size": size,
                "hash": calculate_file_hash(script)
            })
            manifest["total_size"] += size
            if verbose:
                print(f"  + {arcname} ({human_readable_size(size)})")

        # Add requirements.txt
        if requirements.exists():
            arcname = f"{base_name}/requirements.txt"
            tar.add(requirements, arcname=arcname)
            size = requirements.stat().st_size
            manifest["files"].append({
                "path": arcname,
                "size": size,
                "hash": calculate_file_hash(requirements)
            })
            manifest["total_size"] += size
            if verbose:
                print(f"  + {arcname} ({human_readable_size(size)})")

        # Generate and add README_DEPLOY.md
        is_valid, line_count, _ = validate_jsonl_file(training_data)
        readme_content = generate_readme_deploy(
            training_data.name,
            line_count if is_valid else 0,
            [c.name for c in configs],
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        # Write README to temp file and add to archive
        readme_temp = output_dir / "README_DEPLOY.md"
        readme_temp.write_text(readme_content)
        arcname = f"{base_name}/README_DEPLOY.md"
        tar.add(readme_temp, arcname=arcname)
        size = readme_temp.stat().st_size
        manifest["files"].append({
            "path": arcname,
            "size": size,
            "type": "generated"
        })
        manifest["total_size"] += size
        readme_temp.unlink()
        if verbose:
            print(f"  + {arcname} (generated)")

        # Create empty directories for output
        for dir_name in ["model_adapters", "data/training_logs", "exports"]:
            # Create a .gitkeep placeholder
            gitkeep_temp = output_dir / ".gitkeep"
            gitkeep_temp.write_text("")
            arcname = f"{base_name}/{dir_name}/.gitkeep"
            tar.add(gitkeep_temp, arcname=arcname)
            gitkeep_temp.unlink()

    return archive_path, manifest


def create_local_package(
    project_root: Path,
    output_dir: Path,
    training_data: Path,
    configs: List[Path],
    scripts: List[Path],
    requirements: Path,
    embeddings_dir: Optional[Path] = None,
    benchmark_results: Optional[List[Path]] = None,
    verbose: bool = True
) -> Tuple[Path, Dict]:
    """
    Create full deployment package for local development.

    Returns:
        Tuple of (archive_path, manifest_dict)
    """
    archive_name = "local_package.tar.gz"
    archive_path = output_dir / archive_name

    manifest = {
        "package_type": "local",
        "created": datetime.now().isoformat(),
        "files": [],
        "total_size": 0
    }

    # Create tar archive
    with tarfile.open(archive_path, "w:gz") as tar:
        base_name = "L4D2-AI-Architect"

        # Add all training data versions (not just v13)
        processed_dir = training_data.parent
        for jsonl_file in sorted(processed_dir.glob("*.jsonl")):
            if "v1" in jsonl_file.name or "combined" in jsonl_file.name:
                arcname = f"{base_name}/data/processed/{jsonl_file.name}"
                tar.add(jsonl_file, arcname=arcname)
                size = jsonl_file.stat().st_size
                manifest["files"].append({
                    "path": arcname,
                    "size": size,
                    "hash": calculate_file_hash(jsonl_file)
                })
                manifest["total_size"] += size
                if verbose:
                    print(f"  + {arcname} ({human_readable_size(size)})")

        # Add config files
        for config in configs:
            arcname = f"{base_name}/configs/{config.name}"
            tar.add(config, arcname=arcname)
            size = config.stat().st_size
            manifest["files"].append({
                "path": arcname,
                "size": size,
                "hash": calculate_file_hash(config)
            })
            manifest["total_size"] += size
            if verbose:
                print(f"  + {arcname} ({human_readable_size(size)})")

        # Add Python scripts
        scripts_dir = project_root / "scripts"
        for script in scripts:
            rel_path = script.relative_to(scripts_dir)
            arcname = f"{base_name}/scripts/{rel_path}"
            tar.add(script, arcname=arcname)
            size = script.stat().st_size
            manifest["files"].append({
                "path": arcname,
                "size": size,
                "hash": calculate_file_hash(script)
            })
            manifest["total_size"] += size
            if verbose:
                print(f"  + {arcname} ({human_readable_size(size)})")

        # Add requirements.txt
        if requirements.exists():
            arcname = f"{base_name}/requirements.txt"
            tar.add(requirements, arcname=arcname)
            size = requirements.stat().st_size
            manifest["files"].append({
                "path": arcname,
                "size": size,
                "hash": calculate_file_hash(requirements)
            })
            manifest["total_size"] += size
            if verbose:
                print(f"  + {arcname} ({human_readable_size(size)})")

        # Add embeddings if available
        if embeddings_dir and embeddings_dir.exists():
            for emb_file in embeddings_dir.iterdir():
                if emb_file.is_file():
                    arcname = f"{base_name}/data/embeddings/{emb_file.name}"
                    tar.add(emb_file, arcname=arcname)
                    size = emb_file.stat().st_size
                    manifest["files"].append({
                        "path": arcname,
                        "size": size,
                        "hash": calculate_file_hash(emb_file)
                    })
                    manifest["total_size"] += size
                    if verbose:
                        print(f"  + {arcname} ({human_readable_size(size)})")

        # Add benchmark results if available
        if benchmark_results:
            for result_file in benchmark_results:
                if result_file.exists():
                    arcname = f"{base_name}/data/{result_file.name}"
                    tar.add(result_file, arcname=arcname)
                    size = result_file.stat().st_size
                    manifest["files"].append({
                        "path": arcname,
                        "size": size,
                        "hash": calculate_file_hash(result_file)
                    })
                    manifest["total_size"] += size
                    if verbose:
                        print(f"  + {arcname} ({human_readable_size(size)})")

        # Generate and add README_DEPLOY.md
        is_valid, line_count, _ = validate_jsonl_file(training_data)
        readme_content = generate_readme_deploy(
            training_data.name,
            line_count if is_valid else 0,
            [c.name for c in configs],
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        readme_temp = output_dir / "README_DEPLOY.md"
        readme_temp.write_text(readme_content)
        arcname = f"{base_name}/README_DEPLOY.md"
        tar.add(readme_temp, arcname=arcname)
        size = readme_temp.stat().st_size
        manifest["files"].append({
            "path": arcname,
            "size": size,
            "type": "generated"
        })
        manifest["total_size"] += size
        readme_temp.unlink()
        if verbose:
            print(f"  + {arcname} (generated)")

    return archive_path, manifest


def validate_package_contents(project_root: Path, verbose: bool = True) -> Dict:
    """
    Validate all files that will be included in the package.

    Returns:
        Dictionary with validation results
    """
    results = {
        "training_data": {"valid": False, "samples": 0, "error": None},
        "configs": {"valid": True, "count": 0, "errors": []},
        "scripts": {"valid": True, "count": 0, "errors": []},
        "requirements": {"valid": False, "error": None}
    }

    # Validate training data
    training_data = project_root / "data" / "processed" / "l4d2_train_v13.jsonl"
    if training_data.exists():
        is_valid, line_count, error = validate_jsonl_file(training_data)
        results["training_data"]["valid"] = is_valid
        results["training_data"]["samples"] = line_count
        if not is_valid:
            results["training_data"]["error"] = error
        if verbose:
            status = "OK" if is_valid else "FAIL"
            print(f"  [{status}] Training data: {line_count} samples")
    else:
        results["training_data"]["error"] = "File not found"
        if verbose:
            print(f"  [FAIL] Training data: File not found")

    # Validate config files
    configs = find_config_files(project_root)
    results["configs"]["count"] = len(configs)
    for config in configs:
        try:
            import yaml
            with open(config, 'r') as f:
                yaml.safe_load(f)
            if verbose:
                print(f"  [OK] Config: {config.name}")
        except Exception as e:
            results["configs"]["valid"] = False
            results["configs"]["errors"].append(f"{config.name}: {e}")
            if verbose:
                print(f"  [FAIL] Config: {config.name} - {e}")

    # Validate Python scripts
    scripts = find_python_scripts(project_root)
    results["scripts"]["count"] = len(scripts)
    for script in scripts:
        is_valid, error = validate_python_syntax(script)
        if not is_valid:
            results["scripts"]["valid"] = False
            results["scripts"]["errors"].append(f"{script.name}: {error}")
            if verbose:
                print(f"  [FAIL] Script: {script.name}")
        elif verbose:
            print(f"  [OK] Script: {script.name}")

    # Validate requirements.txt
    requirements = project_root / "requirements.txt"
    if requirements.exists():
        results["requirements"]["valid"] = True
        if verbose:
            print(f"  [OK] requirements.txt")
    else:
        results["requirements"]["error"] = "File not found"
        if verbose:
            print(f"  [FAIL] requirements.txt: File not found")

    return results


def generate_manifest(
    vultr_manifest: Dict,
    local_manifest: Dict,
    validation_results: Dict,
    output_path: Path
) -> None:
    """Generate MANIFEST.txt listing all included files."""

    lines = [
        "=" * 70,
        "L4D2-AI-Architect Deployment Package Manifest",
        "=" * 70,
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "=" * 70,
        "VULTR PACKAGE (vultr_package.tar.gz)",
        "=" * 70,
        f"Total Size: {human_readable_size(vultr_manifest['total_size'])}",
        f"File Count: {len(vultr_manifest['files'])}",
        "",
        "Files:",
    ]

    for f in vultr_manifest["files"]:
        size_str = human_readable_size(f["size"]) if "size" in f else "N/A"
        hash_str = f.get("hash", "generated")[:8] if f.get("hash") else "generated"
        lines.append(f"  {f['path']}")
        lines.append(f"    Size: {size_str}, Hash: {hash_str}")

    lines.extend([
        "",
        "=" * 70,
        "LOCAL PACKAGE (local_package.tar.gz)",
        "=" * 70,
        f"Total Size: {human_readable_size(local_manifest['total_size'])}",
        f"File Count: {len(local_manifest['files'])}",
        "",
        "Files:",
    ])

    for f in local_manifest["files"]:
        size_str = human_readable_size(f["size"]) if "size" in f else "N/A"
        hash_str = f.get("hash", "generated")[:8] if f.get("hash") else "generated"
        lines.append(f"  {f['path']}")
        lines.append(f"    Size: {size_str}, Hash: {hash_str}")

    lines.extend([
        "",
        "=" * 70,
        "VALIDATION SUMMARY",
        "=" * 70,
        "",
        f"Training Data: {'VALID' if validation_results['training_data']['valid'] else 'INVALID'}",
        f"  Samples: {validation_results['training_data']['samples']}",
        "",
        f"Config Files: {'VALID' if validation_results['configs']['valid'] else 'INVALID'}",
        f"  Count: {validation_results['configs']['count']}",
    ])

    if validation_results['configs']['errors']:
        lines.append("  Errors:")
        for err in validation_results['configs']['errors']:
            lines.append(f"    - {err}")

    lines.extend([
        "",
        f"Python Scripts: {'VALID' if validation_results['scripts']['valid'] else 'INVALID'}",
        f"  Count: {validation_results['scripts']['count']}",
    ])

    if validation_results['scripts']['errors']:
        lines.append("  Errors:")
        for err in validation_results['scripts']['errors']:
            lines.append(f"    - {err}")

    lines.extend([
        "",
        f"Requirements: {'VALID' if validation_results['requirements']['valid'] else 'INVALID'}",
        "",
        "=" * 70,
    ])

    output_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Create deployment packages for L4D2-AI-Architect"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="deployment",
        help="Output directory for packages (default: deployment)"
    )
    parser.add_argument(
        "--training-data",
        type=str,
        default="data/processed/l4d2_train_v13.jsonl",
        help="Path to training data file"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip file validation"
    )
    parser.add_argument(
        "--vultr-only",
        action="store_true",
        help="Create only Vultr package"
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Create only local package"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Verbose output"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet output"
    )

    args = parser.parse_args()
    verbose = not args.quiet and args.verbose

    # Resolve paths
    project_root = PROJECT_ROOT
    output_dir = safe_path(args.output_dir, project_root, create_parents=True)
    # Ensure the output directory itself exists
    output_dir.mkdir(parents=True, exist_ok=True)
    training_data = safe_path(args.training_data, project_root)

    print("=" * 70)
    print("L4D2-AI-Architect Deployment Package Creator")
    print("=" * 70)
    print(f"\nProject Root: {project_root}")
    print(f"Output Directory: {output_dir}")
    print(f"Training Data: {training_data}")
    print()

    # Validate files
    if not args.skip_validation:
        print("Validating package contents...")
        print("-" * 40)
        validation_results = validate_package_contents(project_root, verbose=verbose)
        print()

        # Check for critical errors
        if not validation_results["training_data"]["valid"]:
            print("ERROR: Training data validation failed!")
            print(f"  {validation_results['training_data']['error']}")
            if not training_data.exists():
                print("\nPlease run the data preparation pipeline first:")
                print("  python scripts/training/generate_v13_massive.py")
                sys.exit(1)

        if not validation_results["scripts"]["valid"]:
            print("WARNING: Some Python scripts have syntax errors:")
            for err in validation_results["scripts"]["errors"][:5]:
                print(f"  - {err}")
            print()
    else:
        validation_results = {
            "training_data": {"valid": True, "samples": 0, "error": None},
            "configs": {"valid": True, "count": 0, "errors": []},
            "scripts": {"valid": True, "count": 0, "errors": []},
            "requirements": {"valid": True, "error": None}
        }

    # Find files to include
    configs = find_config_files(project_root)
    scripts = find_python_scripts(project_root)
    requirements = project_root / "requirements.txt"
    embeddings_dir = project_root / "data" / "embeddings"

    # Find benchmark results
    benchmark_results = list((project_root / "data").glob("eval_results*.json"))

    print(f"Found {len(configs)} config files")
    print(f"Found {len(scripts)} Python scripts")
    print(f"Found {len(benchmark_results)} benchmark result files")
    print()

    vultr_manifest = {}
    local_manifest = {}

    # Create Vultr package
    if not args.local_only:
        print("Creating Vultr package...")
        print("-" * 40)
        vultr_path, vultr_manifest = create_vultr_package(
            project_root=project_root,
            output_dir=output_dir,
            training_data=training_data,
            configs=configs,
            scripts=scripts,
            requirements=requirements,
            verbose=verbose
        )
        vultr_size = vultr_path.stat().st_size
        print(f"\nCreated: {vultr_path}")
        print(f"Size: {human_readable_size(vultr_size)}")
        print()

    # Create local package
    if not args.vultr_only:
        print("Creating local package...")
        print("-" * 40)
        local_path, local_manifest = create_local_package(
            project_root=project_root,
            output_dir=output_dir,
            training_data=training_data,
            configs=configs,
            scripts=scripts,
            requirements=requirements,
            embeddings_dir=embeddings_dir,
            benchmark_results=benchmark_results,
            verbose=verbose
        )
        local_size = local_path.stat().st_size
        print(f"\nCreated: {local_path}")
        print(f"Size: {human_readable_size(local_size)}")
        print()

    # Generate manifest
    manifest_path = output_dir / "MANIFEST.txt"
    generate_manifest(
        vultr_manifest=vultr_manifest or {"files": [], "total_size": 0},
        local_manifest=local_manifest or {"files": [], "total_size": 0},
        validation_results=validation_results,
        output_path=manifest_path
    )
    print(f"Created: {manifest_path}")
    print()

    # Summary
    print("=" * 70)
    print("PACKAGE SUMMARY")
    print("=" * 70)

    if not args.local_only and vultr_manifest:
        print(f"\nVultr Package: {output_dir / 'vultr_package.tar.gz'}")
        print(f"  Files: {len(vultr_manifest['files'])}")
        print(f"  Size: {human_readable_size(vultr_manifest['total_size'])} (uncompressed)")
        print(f"  Compressed: {human_readable_size((output_dir / 'vultr_package.tar.gz').stat().st_size)}")

    if not args.vultr_only and local_manifest:
        print(f"\nLocal Package: {output_dir / 'local_package.tar.gz'}")
        print(f"  Files: {len(local_manifest['files'])}")
        print(f"  Size: {human_readable_size(local_manifest['total_size'])} (uncompressed)")
        print(f"  Compressed: {human_readable_size((output_dir / 'local_package.tar.gz').stat().st_size)}")

    print(f"\nManifest: {manifest_path}")
    print()

    # Validation summary
    all_valid = (
        validation_results["training_data"]["valid"] and
        validation_results["configs"]["valid"] and
        validation_results["scripts"]["valid"] and
        validation_results["requirements"]["valid"]
    )

    if all_valid:
        print("Validation: ALL CHECKS PASSED")
    else:
        print("Validation: SOME CHECKS FAILED")
        if not validation_results["training_data"]["valid"]:
            print(f"  - Training data: {validation_results['training_data']['error']}")
        if not validation_results["scripts"]["valid"]:
            print(f"  - Scripts: {len(validation_results['scripts']['errors'])} errors")

    print()
    print("Deployment packages created successfully!")

    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
