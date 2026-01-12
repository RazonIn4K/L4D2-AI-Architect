#!/usr/bin/env python3
"""
Project Statistics Generator for L4D2-AI-Architect

Scans the entire project and generates comprehensive statistics including:
- Code statistics (Python files, lines of code per module)
- Data statistics (JSONL files, training examples, embeddings)
- Model statistics (LoRA adapters, GGUF exports, RL agents)
- Config statistics (YAML configs, training parameters)
- Test statistics (test files, test functions)
- Documentation statistics (Markdown files, word counts)

Usage:
    python scripts/utils/project_stats.py --output-all
    python scripts/utils/project_stats.py --console-only
    python scripts/utils/project_stats.py --json-only
    python scripts/utils/project_stats.py --markdown-only
"""

import argparse
import ast
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent to path for security utils
sys.path.insert(0, str(Path(__file__).parent))

try:
    from security import safe_path, safe_write_json, safe_write_text
except ImportError:
    # Fallback if security module not available
    def safe_path(path: str, base_dir: Path, create_parents: bool = False) -> Path:
        result = (base_dir / path).resolve()
        if create_parents:
            result.parent.mkdir(parents=True, exist_ok=True)
        return result

    def safe_write_json(path: str, data: Any, base_dir: Path, indent: int = 2, **kwargs) -> Path:
        validated = safe_path(path, base_dir, create_parents=True)
        with open(validated, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, **kwargs)
        return validated

    def safe_write_text(path: str, content: str, base_dir: Path, **kwargs) -> Path:
        validated = safe_path(path, base_dir, create_parents=True)
        with open(validated, "w", encoding="utf-8") as f:
            f.write(content)
        return validated

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Directories to exclude from scanning
EXCLUDED_DIRS = {"venv", ".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "env"}


@dataclass
class CodeStats:
    """Statistics about Python code in the project."""
    total_files: int = 0
    total_lines: int = 0
    total_blank_lines: int = 0
    total_comment_lines: int = 0
    total_code_lines: int = 0
    lines_by_module: Dict[str, int] = field(default_factory=dict)
    files_by_module: Dict[str, int] = field(default_factory=dict)
    largest_files: List[Tuple[str, int]] = field(default_factory=list)


@dataclass
class DataStats:
    """Statistics about data files in the project."""
    jsonl_files: List[Dict[str, Any]] = field(default_factory=list)
    total_training_examples: int = 0
    total_jsonl_files: int = 0
    examples_by_version: Dict[str, int] = field(default_factory=dict)
    embedding_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelStats:
    """Statistics about trained models in the project."""
    lora_adapters: List[Dict[str, Any]] = field(default_factory=list)
    gguf_exports: List[Dict[str, Any]] = field(default_factory=list)
    rl_agents: List[Dict[str, Any]] = field(default_factory=list)
    total_adapters: int = 0
    total_gguf: int = 0
    total_rl_agents: int = 0


@dataclass
class ConfigStats:
    """Statistics about configuration files."""
    yaml_configs: List[Dict[str, Any]] = field(default_factory=list)
    total_configs: int = 0
    training_params_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestStats:
    """Statistics about test files."""
    test_files: List[Dict[str, Any]] = field(default_factory=list)
    total_test_files: int = 0
    total_test_functions: int = 0
    total_test_classes: int = 0


@dataclass
class DocStats:
    """Statistics about documentation files."""
    markdown_files: List[Dict[str, Any]] = field(default_factory=list)
    total_md_files: int = 0
    total_words: int = 0
    total_lines: int = 0


@dataclass
class ProjectStats:
    """Complete project statistics."""
    generated_at: str = ""
    project_name: str = "L4D2-AI-Architect"
    code: CodeStats = field(default_factory=CodeStats)
    data: DataStats = field(default_factory=DataStats)
    models: ModelStats = field(default_factory=ModelStats)
    configs: ConfigStats = field(default_factory=ConfigStats)
    tests: TestStats = field(default_factory=TestStats)
    docs: DocStats = field(default_factory=DocStats)


def should_skip_dir(path: Path) -> bool:
    """Check if a directory should be skipped during scanning."""
    return any(excluded in path.parts for excluded in EXCLUDED_DIRS)


def count_python_lines(filepath: Path) -> Tuple[int, int, int, int]:
    """
    Count lines in a Python file.

    Returns:
        Tuple of (total_lines, blank_lines, comment_lines, code_lines)
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except (IOError, OSError):
        return 0, 0, 0, 0

    total = len(lines)
    blank = 0
    comment = 0
    in_docstring = False
    docstring_char = None

    for line in lines:
        stripped = line.strip()

        # Check for docstring start/end
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                docstring_char = stripped[:3]
                if stripped.count(docstring_char) >= 2 and len(stripped) > 6:
                    # Single-line docstring
                    comment += 1
                else:
                    in_docstring = True
                    comment += 1
                continue
        else:
            comment += 1
            if docstring_char in stripped:
                in_docstring = False
            continue

        if not stripped:
            blank += 1
        elif stripped.startswith("#"):
            comment += 1

    code = total - blank - comment
    return total, blank, comment, code


def count_test_functions(filepath: Path) -> Tuple[int, int]:
    """
    Count test functions and test classes in a Python file.

    Returns:
        Tuple of (test_functions, test_classes)
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        tree = ast.parse(content)
    except (SyntaxError, IOError, OSError):
        return 0, 0

    test_funcs = 0
    test_classes = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith("test_"):
                test_funcs += 1
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("Test"):
                test_classes += 1

    return test_funcs, test_classes


def count_jsonl_examples(filepath: Path) -> int:
    """Count the number of examples in a JSONL file."""
    count = 0
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip():
                    count += 1
    except (IOError, OSError):
        pass
    return count


def get_file_size_mb(filepath: Path) -> float:
    """Get file size in MB."""
    try:
        return filepath.stat().st_size / (1024 * 1024)
    except (IOError, OSError):
        return 0.0


def count_markdown_words(filepath: Path) -> Tuple[int, int]:
    """
    Count words and lines in a Markdown file.

    Returns:
        Tuple of (words, lines)
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except (IOError, OSError):
        return 0, 0

    lines = content.count("\n") + 1
    # Remove code blocks for word count
    content_no_code = re.sub(r"```[\s\S]*?```", "", content)
    words = len(content_no_code.split())
    return words, lines


def get_module_name(filepath: Path) -> str:
    """Extract the module name from a file path."""
    try:
        rel_path = filepath.relative_to(PROJECT_ROOT)
        parts = rel_path.parts

        # Return first meaningful directory
        if len(parts) > 1:
            if parts[0] == "scripts" and len(parts) > 2:
                return f"scripts/{parts[1]}"
            return parts[0]
        return "root"
    except ValueError:
        return "unknown"


def collect_code_stats() -> CodeStats:
    """Collect statistics about Python code."""
    stats = CodeStats()
    file_sizes: List[Tuple[str, int]] = []

    for py_file in PROJECT_ROOT.rglob("*.py"):
        if should_skip_dir(py_file):
            continue

        total, blank, comment, code = count_python_lines(py_file)

        stats.total_files += 1
        stats.total_lines += total
        stats.total_blank_lines += blank
        stats.total_comment_lines += comment
        stats.total_code_lines += code

        module = get_module_name(py_file)
        stats.lines_by_module[module] = stats.lines_by_module.get(module, 0) + total
        stats.files_by_module[module] = stats.files_by_module.get(module, 0) + 1

        rel_path = str(py_file.relative_to(PROJECT_ROOT))
        file_sizes.append((rel_path, total))

    # Get top 10 largest files
    file_sizes.sort(key=lambda x: x[1], reverse=True)
    stats.largest_files = file_sizes[:10]

    return stats


def collect_data_stats() -> DataStats:
    """Collect statistics about data files."""
    stats = DataStats()
    data_dir = PROJECT_ROOT / "data"

    if not data_dir.exists():
        return stats

    # Scan all JSONL files
    for jsonl_file in data_dir.rglob("*.jsonl"):
        if should_skip_dir(jsonl_file):
            continue

        example_count = count_jsonl_examples(jsonl_file)
        size_mb = get_file_size_mb(jsonl_file)
        rel_path = str(jsonl_file.relative_to(PROJECT_ROOT))

        file_info = {
            "path": rel_path,
            "name": jsonl_file.name,
            "examples": example_count,
            "size_mb": round(size_mb, 2)
        }

        stats.jsonl_files.append(file_info)
        stats.total_jsonl_files += 1
        stats.total_training_examples += example_count

        # Track examples by version
        name = jsonl_file.name.lower()
        version_match = re.search(r"v(\d+)", name)
        if version_match:
            version = f"v{version_match.group(1)}"
            stats.examples_by_version[version] = stats.examples_by_version.get(version, 0) + example_count

    # Sort by example count
    stats.jsonl_files.sort(key=lambda x: x["examples"], reverse=True)

    # Check for embeddings
    embeddings_dir = data_dir / "embeddings"
    if embeddings_dir.exists():
        embedding_info: Dict[str, Any] = {"exists": True, "files": []}

        for emb_file in embeddings_dir.iterdir():
            if emb_file.is_file():
                file_info = {
                    "name": emb_file.name,
                    "size_mb": round(get_file_size_mb(emb_file), 2)
                }

                # Try to get dimensions from numpy files
                if emb_file.suffix == ".npy":
                    try:
                        import numpy as np
                        arr = np.load(emb_file)
                        file_info["shape"] = list(arr.shape)
                        file_info["dtype"] = str(arr.dtype)
                    except Exception:
                        pass

                # Check FAISS index
                if emb_file.name == "faiss_index.bin":
                    try:
                        import faiss
                        index = faiss.read_index(str(emb_file))
                        file_info["num_vectors"] = index.ntotal
                        file_info["dimension"] = index.d
                    except Exception:
                        pass

                embedding_info["files"].append(file_info)

        stats.embedding_info = embedding_info
    else:
        stats.embedding_info = {"exists": False}

    return stats


def collect_model_stats() -> ModelStats:
    """Collect statistics about trained models."""
    stats = ModelStats()
    model_adapters_dir = PROJECT_ROOT / "model_adapters"
    exports_dir = PROJECT_ROOT / "exports"

    # Scan LoRA adapters
    if model_adapters_dir.exists():
        for adapter_dir in model_adapters_dir.iterdir():
            if adapter_dir.is_dir() and not adapter_dir.name.startswith("."):
                adapter_info: Dict[str, Any] = {
                    "name": adapter_dir.name,
                    "path": str(adapter_dir.relative_to(PROJECT_ROOT))
                }

                # Check for final adapter
                final_dir = adapter_dir / "final"
                if final_dir.exists():
                    adapter_info["has_final"] = True

                    # Check for adapter config
                    config_file = final_dir / "adapter_config.json"
                    if config_file.exists():
                        try:
                            with open(config_file, "r") as f:
                                config = json.load(f)
                            adapter_info["lora_rank"] = config.get("r")
                            adapter_info["lora_alpha"] = config.get("lora_alpha")
                            adapter_info["base_model"] = config.get("base_model_name_or_path", "unknown")
                        except Exception:
                            pass

                    # Check for safetensors
                    safetensors = list(final_dir.glob("*.safetensors"))
                    if safetensors:
                        adapter_info["safetensors_size_mb"] = round(
                            sum(get_file_size_mb(f) for f in safetensors), 2
                        )

                # Check for training info
                training_info_file = adapter_dir / "training_info.json"
                if training_info_file.exists():
                    try:
                        with open(training_info_file, "r") as f:
                            training_info = json.load(f)
                        adapter_info["training_samples"] = training_info.get("num_train_samples")
                        adapter_info["epochs"] = training_info.get("num_epochs")
                    except Exception:
                        pass

                # Check for RL agents
                if adapter_dir.name in ["rl_test", "director_agents", "rl_agents"]:
                    for agent_dir in adapter_dir.rglob("*.zip"):
                        agent_info = {
                            "name": agent_dir.stem,
                            "path": str(agent_dir.relative_to(PROJECT_ROOT)),
                            "size_mb": round(get_file_size_mb(agent_dir), 2)
                        }
                        stats.rl_agents.append(agent_info)
                        stats.total_rl_agents += 1
                else:
                    stats.lora_adapters.append(adapter_info)
                    stats.total_adapters += 1

    # Scan GGUF exports
    if exports_dir.exists():
        for gguf_file in exports_dir.rglob("*.gguf"):
            gguf_info = {
                "name": gguf_file.name,
                "path": str(gguf_file.relative_to(PROJECT_ROOT)),
                "size_mb": round(get_file_size_mb(gguf_file), 2)
            }
            stats.gguf_exports.append(gguf_info)
            stats.total_gguf += 1

    return stats


def collect_config_stats() -> ConfigStats:
    """Collect statistics about configuration files."""
    stats = ConfigStats()
    configs_dir = PROJECT_ROOT / "configs"

    if not configs_dir.exists():
        return stats

    training_params: Dict[str, List[Any]] = defaultdict(list)

    for yaml_file in configs_dir.glob("*.yaml"):
        config_info: Dict[str, Any] = {
            "name": yaml_file.name,
            "path": str(yaml_file.relative_to(PROJECT_ROOT)),
            "size_bytes": yaml_file.stat().st_size
        }

        try:
            import yaml
            with open(yaml_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            if config:
                config_info["sections"] = list(config.keys())

                # Extract training parameters for summary
                if "training" in config and isinstance(config["training"], dict):
                    for key, value in config["training"].items():
                        if value is not None:
                            training_params[key].append(value)

                if "model" in config and isinstance(config["model"], dict):
                    config_info["model_name"] = config["model"].get("name", "unknown")
                    config_info["max_seq_length"] = config["model"].get("max_seq_length")

                if "lora" in config and isinstance(config["lora"], dict):
                    config_info["lora_rank"] = config["lora"].get("r")
        except Exception:
            config_info["parse_error"] = True

        stats.yaml_configs.append(config_info)
        stats.total_configs += 1

    # Summarize training parameters
    for key, values in training_params.items():
        if all(isinstance(v, (int, float)) for v in values):
            stats.training_params_summary[key] = {
                "min": min(values),
                "max": max(values),
                "values": list(set(values))
            }
        else:
            stats.training_params_summary[key] = {"values": list(set(str(v) for v in values))}

    return stats


def collect_test_stats() -> TestStats:
    """Collect statistics about test files."""
    stats = TestStats()
    tests_dir = PROJECT_ROOT / "tests"
    scripts_dir = PROJECT_ROOT / "scripts"

    # Scan tests directory
    if tests_dir.exists():
        for py_file in tests_dir.rglob("*.py"):
            if should_skip_dir(py_file):
                continue

            test_funcs, test_classes = count_test_functions(py_file)
            total_lines, _, _, _ = count_python_lines(py_file)

            test_info = {
                "name": py_file.name,
                "path": str(py_file.relative_to(PROJECT_ROOT)),
                "test_functions": test_funcs,
                "test_classes": test_classes,
                "lines": total_lines
            }

            stats.test_files.append(test_info)
            stats.total_test_files += 1
            stats.total_test_functions += test_funcs
            stats.total_test_classes += test_classes

    # Also scan for test files in scripts directory
    if scripts_dir.exists():
        for py_file in scripts_dir.rglob("test_*.py"):
            if should_skip_dir(py_file):
                continue

            test_funcs, test_classes = count_test_functions(py_file)
            total_lines, _, _, _ = count_python_lines(py_file)

            test_info = {
                "name": py_file.name,
                "path": str(py_file.relative_to(PROJECT_ROOT)),
                "test_functions": test_funcs,
                "test_classes": test_classes,
                "lines": total_lines
            }

            stats.test_files.append(test_info)
            stats.total_test_files += 1
            stats.total_test_functions += test_funcs
            stats.total_test_classes += test_classes

    return stats


def collect_doc_stats() -> DocStats:
    """Collect statistics about documentation files."""
    stats = DocStats()

    # Scan for markdown files (excluding venv and other excluded dirs)
    for md_file in PROJECT_ROOT.rglob("*.md"):
        if should_skip_dir(md_file):
            continue

        words, lines = count_markdown_words(md_file)

        doc_info = {
            "name": md_file.name,
            "path": str(md_file.relative_to(PROJECT_ROOT)),
            "words": words,
            "lines": lines
        }

        stats.markdown_files.append(doc_info)
        stats.total_md_files += 1
        stats.total_words += words
        stats.total_lines += lines

    # Sort by word count
    stats.markdown_files.sort(key=lambda x: x["words"], reverse=True)

    return stats


def collect_all_stats() -> ProjectStats:
    """Collect all project statistics."""
    stats = ProjectStats()
    stats.generated_at = datetime.now().isoformat()

    print("Collecting code statistics...")
    stats.code = collect_code_stats()

    print("Collecting data statistics...")
    stats.data = collect_data_stats()

    print("Collecting model statistics...")
    stats.models = collect_model_stats()

    print("Collecting config statistics...")
    stats.configs = collect_config_stats()

    print("Collecting test statistics...")
    stats.tests = collect_test_stats()

    print("Collecting documentation statistics...")
    stats.docs = collect_doc_stats()

    return stats


def format_console_output(stats: ProjectStats) -> str:
    """Format statistics for console output."""
    lines = []
    lines.append("=" * 70)
    lines.append("L4D2-AI-ARCHITECT PROJECT STATISTICS")
    lines.append(f"Generated: {stats.generated_at}")
    lines.append("=" * 70)

    # Code Statistics
    lines.append("\n[CODE STATISTICS]")
    lines.append(f"  Total Python files: {stats.code.total_files}")
    lines.append(f"  Total lines: {stats.code.total_lines:,}")
    lines.append(f"    - Code lines: {stats.code.total_code_lines:,}")
    lines.append(f"    - Comment lines: {stats.code.total_comment_lines:,}")
    lines.append(f"    - Blank lines: {stats.code.total_blank_lines:,}")
    lines.append("\n  Lines by module:")
    for module, count in sorted(stats.code.lines_by_module.items(), key=lambda x: -x[1]):
        files = stats.code.files_by_module.get(module, 0)
        lines.append(f"    {module}: {count:,} lines ({files} files)")

    if stats.code.largest_files:
        lines.append("\n  Largest files:")
        for path, line_count in stats.code.largest_files[:5]:
            lines.append(f"    {path}: {line_count:,} lines")

    # Data Statistics
    lines.append("\n[DATA STATISTICS]")
    lines.append(f"  Total JSONL files: {stats.data.total_jsonl_files}")
    lines.append(f"  Total training examples: {stats.data.total_training_examples:,}")

    if stats.data.examples_by_version:
        lines.append("\n  Examples by version:")
        for version, count in sorted(stats.data.examples_by_version.items()):
            lines.append(f"    {version}: {count:,}")

    lines.append("\n  Top JSONL files by example count:")
    for file_info in stats.data.jsonl_files[:10]:
        lines.append(f"    {file_info['name']}: {file_info['examples']:,} examples ({file_info['size_mb']:.2f} MB)")

    if stats.data.embedding_info.get("exists"):
        lines.append("\n  Embeddings:")
        for file_info in stats.data.embedding_info.get("files", []):
            info_str = f"    {file_info['name']}: {file_info['size_mb']:.2f} MB"
            if "shape" in file_info:
                info_str += f" (shape: {file_info['shape']})"
            if "num_vectors" in file_info:
                info_str += f" ({file_info['num_vectors']} vectors, dim={file_info['dimension']})"
            lines.append(info_str)

    # Model Statistics
    lines.append("\n[MODEL STATISTICS]")
    lines.append(f"  LoRA adapters: {stats.models.total_adapters}")
    for adapter in stats.models.lora_adapters:
        info_str = f"    - {adapter['name']}"
        if adapter.get("lora_rank"):
            info_str += f" (rank={adapter['lora_rank']})"
        if adapter.get("safetensors_size_mb"):
            info_str += f" [{adapter['safetensors_size_mb']:.2f} MB]"
        lines.append(info_str)

    lines.append(f"\n  GGUF exports: {stats.models.total_gguf}")
    for gguf in stats.models.gguf_exports:
        lines.append(f"    - {gguf['name']}: {gguf['size_mb']:.2f} MB")

    lines.append(f"\n  RL agents: {stats.models.total_rl_agents}")
    for agent in stats.models.rl_agents:
        lines.append(f"    - {agent['name']}: {agent['size_mb']:.2f} MB")

    # Config Statistics
    lines.append("\n[CONFIG STATISTICS]")
    lines.append(f"  YAML config files: {stats.configs.total_configs}")
    for config in stats.configs.yaml_configs:
        info_str = f"    - {config['name']}"
        if config.get("model_name"):
            info_str += f" (model: {config['model_name'][:40]}...)"
        lines.append(info_str)

    if stats.configs.training_params_summary:
        lines.append("\n  Training parameters summary:")
        for param, info in stats.configs.training_params_summary.items():
            if "min" in info and "max" in info:
                if info["min"] == info["max"]:
                    lines.append(f"    {param}: {info['min']}")
                else:
                    lines.append(f"    {param}: {info['min']} - {info['max']}")

    # Test Statistics
    lines.append("\n[TEST STATISTICS]")
    lines.append(f"  Test files: {stats.tests.total_test_files}")
    lines.append(f"  Test functions: {stats.tests.total_test_functions}")
    lines.append(f"  Test classes: {stats.tests.total_test_classes}")
    for test in stats.tests.test_files:
        lines.append(f"    - {test['name']}: {test['test_functions']} functions, {test['test_classes']} classes")

    # Documentation Statistics
    lines.append("\n[DOCUMENTATION STATISTICS]")
    lines.append(f"  Markdown files: {stats.docs.total_md_files}")
    lines.append(f"  Total words: {stats.docs.total_words:,}")
    lines.append(f"  Total lines: {stats.docs.total_lines:,}")
    lines.append("\n  Top documentation files:")
    for doc in stats.docs.markdown_files[:10]:
        lines.append(f"    - {doc['name']}: {doc['words']:,} words")

    lines.append("\n" + "=" * 70)

    return "\n".join(lines)


def format_markdown_output(stats: ProjectStats) -> str:
    """Format statistics as Markdown."""
    lines = []
    lines.append("# L4D2-AI-Architect Project Statistics")
    lines.append("")
    lines.append(f"*Generated: {stats.generated_at}*")
    lines.append("")

    # Summary Table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Category | Metric | Value |")
    lines.append("|----------|--------|-------|")
    lines.append(f"| Code | Python files | {stats.code.total_files} |")
    lines.append(f"| Code | Total lines | {stats.code.total_lines:,} |")
    lines.append(f"| Data | JSONL files | {stats.data.total_jsonl_files} |")
    lines.append(f"| Data | Training examples | {stats.data.total_training_examples:,} |")
    lines.append(f"| Models | LoRA adapters | {stats.models.total_adapters} |")
    lines.append(f"| Models | GGUF exports | {stats.models.total_gguf} |")
    lines.append(f"| Models | RL agents | {stats.models.total_rl_agents} |")
    lines.append(f"| Config | YAML configs | {stats.configs.total_configs} |")
    lines.append(f"| Tests | Test files | {stats.tests.total_test_files} |")
    lines.append(f"| Tests | Test functions | {stats.tests.total_test_functions} |")
    lines.append(f"| Docs | Markdown files | {stats.docs.total_md_files} |")
    lines.append(f"| Docs | Total words | {stats.docs.total_words:,} |")
    lines.append("")

    # Code Statistics
    lines.append("## Code Statistics")
    lines.append("")
    lines.append(f"- **Total Python files:** {stats.code.total_files}")
    lines.append(f"- **Total lines:** {stats.code.total_lines:,}")
    lines.append(f"  - Code lines: {stats.code.total_code_lines:,}")
    lines.append(f"  - Comment lines: {stats.code.total_comment_lines:,}")
    lines.append(f"  - Blank lines: {stats.code.total_blank_lines:,}")
    lines.append("")

    lines.append("### Lines by Module")
    lines.append("")
    lines.append("| Module | Lines | Files |")
    lines.append("|--------|-------|-------|")
    for module, count in sorted(stats.code.lines_by_module.items(), key=lambda x: -x[1]):
        files = stats.code.files_by_module.get(module, 0)
        lines.append(f"| {module} | {count:,} | {files} |")
    lines.append("")

    # Data Statistics
    lines.append("## Data Statistics")
    lines.append("")
    lines.append(f"- **Total JSONL files:** {stats.data.total_jsonl_files}")
    lines.append(f"- **Total training examples:** {stats.data.total_training_examples:,}")
    lines.append("")

    if stats.data.examples_by_version:
        lines.append("### Examples by Version")
        lines.append("")
        lines.append("| Version | Examples |")
        lines.append("|---------|----------|")
        for version, count in sorted(stats.data.examples_by_version.items()):
            lines.append(f"| {version} | {count:,} |")
        lines.append("")

    lines.append("### Top JSONL Files")
    lines.append("")
    lines.append("| File | Examples | Size (MB) |")
    lines.append("|------|----------|-----------|")
    for file_info in stats.data.jsonl_files[:15]:
        lines.append(f"| {file_info['name']} | {file_info['examples']:,} | {file_info['size_mb']:.2f} |")
    lines.append("")

    # Model Statistics
    lines.append("## Model Statistics")
    lines.append("")

    lines.append("### LoRA Adapters")
    lines.append("")
    if stats.models.lora_adapters:
        lines.append("| Name | LoRA Rank | Size (MB) |")
        lines.append("|------|-----------|-----------|")
        for adapter in stats.models.lora_adapters:
            rank = adapter.get("lora_rank", "-")
            size = adapter.get("safetensors_size_mb", "-")
            lines.append(f"| {adapter['name']} | {rank} | {size} |")
    else:
        lines.append("*No LoRA adapters found*")
    lines.append("")

    lines.append("### GGUF Exports")
    lines.append("")
    if stats.models.gguf_exports:
        lines.append("| Name | Size (MB) |")
        lines.append("|------|-----------|")
        for gguf in stats.models.gguf_exports:
            lines.append(f"| {gguf['name']} | {gguf['size_mb']:.2f} |")
    else:
        lines.append("*No GGUF exports found*")
    lines.append("")

    lines.append("### RL Agents")
    lines.append("")
    if stats.models.rl_agents:
        lines.append("| Name | Size (MB) |")
        lines.append("|------|-----------|")
        for agent in stats.models.rl_agents:
            lines.append(f"| {agent['name']} | {agent['size_mb']:.2f} |")
    else:
        lines.append("*No RL agents found*")
    lines.append("")

    # Config Statistics
    lines.append("## Configuration Statistics")
    lines.append("")
    lines.append("| Config File | Model | Sections |")
    lines.append("|-------------|-------|----------|")
    for config in stats.configs.yaml_configs:
        model = config.get("model_name", "-")
        if model and len(model) > 40:
            model = model[:37] + "..."
        sections = ", ".join(config.get("sections", [])[:4])
        lines.append(f"| {config['name']} | {model} | {sections} |")
    lines.append("")

    # Test Statistics
    lines.append("## Test Statistics")
    lines.append("")
    lines.append(f"- **Test files:** {stats.tests.total_test_files}")
    lines.append(f"- **Test functions:** {stats.tests.total_test_functions}")
    lines.append(f"- **Test classes:** {stats.tests.total_test_classes}")
    lines.append("")

    if stats.tests.test_files:
        lines.append("| File | Functions | Classes | Lines |")
        lines.append("|------|-----------|---------|-------|")
        for test in stats.tests.test_files:
            lines.append(f"| {test['name']} | {test['test_functions']} | {test['test_classes']} | {test['lines']} |")
    lines.append("")

    # Documentation Statistics
    lines.append("## Documentation Statistics")
    lines.append("")
    lines.append(f"- **Markdown files:** {stats.docs.total_md_files}")
    lines.append(f"- **Total words:** {stats.docs.total_words:,}")
    lines.append("")

    lines.append("### Documentation Files")
    lines.append("")
    lines.append("| File | Words | Lines |")
    lines.append("|------|-------|-------|")
    for doc in stats.docs.markdown_files[:20]:
        lines.append(f"| {doc['name']} | {doc['words']:,} | {doc['lines']} |")
    lines.append("")

    return "\n".join(lines)


def stats_to_dict(stats: ProjectStats) -> Dict[str, Any]:
    """Convert ProjectStats to a dictionary for JSON serialization."""
    return {
        "generated_at": stats.generated_at,
        "project_name": stats.project_name,
        "code": asdict(stats.code),
        "data": asdict(stats.data),
        "models": asdict(stats.models),
        "configs": asdict(stats.configs),
        "tests": asdict(stats.tests),
        "docs": asdict(stats.docs)
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive project statistics for L4D2-AI-Architect"
    )
    parser.add_argument(
        "--output-all", action="store_true",
        help="Output to console, JSON, and Markdown"
    )
    parser.add_argument(
        "--console-only", action="store_true",
        help="Only output to console"
    )
    parser.add_argument(
        "--json-only", action="store_true",
        help="Only output JSON file"
    )
    parser.add_argument(
        "--markdown-only", action="store_true",
        help="Only output Markdown file"
    )

    args = parser.parse_args()

    # Default to output-all if no specific flag
    if not any([args.output_all, args.console_only, args.json_only, args.markdown_only]):
        args.output_all = True

    # Collect statistics
    print("Scanning project...")
    stats = collect_all_stats()
    print("Done collecting statistics.\n")

    # Console output
    if args.output_all or args.console_only:
        console_output = format_console_output(stats)
        print(console_output)

    # JSON output
    if args.output_all or args.json_only:
        json_path = "data/project_stats.json"
        stats_dict = stats_to_dict(stats)
        safe_write_json(json_path, stats_dict, PROJECT_ROOT, indent=2)
        print(f"\nJSON report saved to: {PROJECT_ROOT / json_path}")

    # Markdown output
    if args.output_all or args.markdown_only:
        md_path = "docs/PROJECT_STATISTICS.md"
        markdown_output = format_markdown_output(stats)
        safe_write_text(md_path, markdown_output, PROJECT_ROOT)
        print(f"Markdown report saved to: {PROJECT_ROOT / md_path}")


if __name__ == "__main__":
    main()
