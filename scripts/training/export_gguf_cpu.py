#!/usr/bin/env python3
"""
CPU-Compatible GGUF Export Script

Merges LoRA adapter with base model and converts to GGUF format.
Works on macOS without CUDA using llama-cpp-python.

Usage:
    python export_gguf_cpu.py --adapter model_adapters/l4d2-mistral-v15-lora/final --output exports/l4d2-v15
    python export_gguf_cpu.py --adapter model_adapters/l4d2-mistral-v10plus-lora/final --output exports/l4d2-v10plus
"""

import os
import sys
import json
import argparse
import logging
import subprocess
import shutil
import re
from pathlib import Path
from typing import Optional, Tuple

# Add parent to path for security utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_path, safe_write_text, safe_read_json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# GGUF quantization options
QUANT_OPTIONS = ["f16", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q4_k_s", "q3_k_m", "q2_k"]

# Trusted model sources for trust_remote_code (security allowlist)
TRUSTED_MODEL_PREFIXES = [
    "mistralai/",
    "meta-llama/",
    "codellama/",
    "Qwen/",
    "unsloth/",
    "huggingface/",
    "google/",
    "microsoft/",
    "bigcode/",
    "deepseek-ai/",
    "stabilityai/",
]

# Default model name if version cannot be determined
DEFAULT_MODEL_NAME = "l4d2-code"


def is_trusted_model(model_name: str) -> bool:
    """Check if model is from a trusted source."""
    return any(model_name.startswith(prefix) or prefix.rstrip('/') in model_name.lower()
               for prefix in TRUSTED_MODEL_PREFIXES)


def extract_model_info(adapter_path: Path) -> Tuple[str, str]:
    """
    Extract model name and version from adapter path or training info.

    Returns:
        Tuple of (model_name, version_string) e.g., ("l4d2-mistral", "v15")
    """
    # Try to read training_info.json if it exists (one level up from final/)
    training_info_path = adapter_path.parent / "training_info.json"
    if not training_info_path.exists() and adapter_path.name == "final":
        training_info_path = adapter_path.parent.parent / "training_info.json"

    model_name = DEFAULT_MODEL_NAME
    version = ""

    # Try to extract from training_info.json
    if training_info_path.exists():
        try:
            info = safe_read_json(str(training_info_path), PROJECT_ROOT)
            if "model_name" in info:
                model_name = info["model_name"]
            if "version" in info:
                version = info["version"]
        except Exception as e:
            logger.debug(f"Could not read training_info.json: {e}")

    # If version not found, try to extract from path
    if not version:
        # Look for version pattern in path (v9, v10, v10plus, v15, etc.)
        path_str = str(adapter_path)
        version_match = re.search(r'-(v\d+(?:plus|[a-z]*)?)-', path_str, re.IGNORECASE)
        if version_match:
            version = version_match.group(1).lower()

    # Try to extract base model type from path
    if model_name == DEFAULT_MODEL_NAME:
        path_str = str(adapter_path).lower()
        if "mistral" in path_str:
            model_name = "l4d2-mistral"
        elif "codellama" in path_str:
            model_name = "l4d2-codellama"
        elif "llama" in path_str:
            model_name = "l4d2-llama"
        elif "qwen" in path_str:
            model_name = "l4d2-qwen"
        elif "deepseek" in path_str:
            model_name = "l4d2-deepseek"

    return model_name, version


def get_gguf_filename(adapter_path: Path, quantization: str) -> str:
    """Generate GGUF filename based on adapter path and quantization."""
    model_name, version = extract_model_info(adapter_path)

    if version:
        return f"{model_name}-{version}-{quantization}.gguf"
    else:
        return f"{model_name}-{quantization}.gguf"


def check_dependencies():
    """Check required dependencies."""
    missing = []

    try:
        import torch
        logger.info(f"PyTorch: {torch.__version__}")
    except ImportError:
        missing.append("torch")

    try:
        import transformers
        logger.info(f"Transformers: {transformers.__version__}")
    except ImportError:
        missing.append("transformers")

    try:
        import peft
        logger.info(f"PEFT: {peft.__version__}")
    except ImportError:
        missing.append("peft")

    try:
        import llama_cpp
        logger.info(f"llama-cpp-python: available")
    except ImportError:
        missing.append("llama-cpp-python")

    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        logger.info("Install with: pip install torch transformers peft llama-cpp-python")
        return False

    return True


def load_adapter_config(adapter_path: Path) -> dict:
    """Load adapter configuration with path validation."""
    # Validate adapter path is within project
    validated_adapter = safe_path(str(adapter_path), PROJECT_ROOT)
    config_path = validated_adapter / "adapter_config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Adapter config not found: {config_path}")

    # Use safe_read_json for secure file reading
    return safe_read_json(str(config_path), PROJECT_ROOT)


def get_base_model_name(adapter_config: dict) -> str:
    """Get the appropriate base model name for CPU loading."""
    base_model = adapter_config.get("base_model_name_or_path", "")

    # Map Unsloth 4-bit models to their original equivalents
    model_mapping = {
        # Mistral models
        "unsloth/mistral-7b-instruct-v0.3-bnb-4bit": "mistralai/Mistral-7B-Instruct-v0.3",
        "unsloth/mistral-7b-instruct-v0.2-bnb-4bit": "mistralai/Mistral-7B-Instruct-v0.2",
        "unsloth/mistral-7b-bnb-4bit": "mistralai/Mistral-7B-v0.1",
        # Llama 2 models
        "unsloth/llama-2-7b-bnb-4bit": "meta-llama/Llama-2-7b-hf",
        "unsloth/llama-2-13b-bnb-4bit": "meta-llama/Llama-2-13b-hf",
        # Llama 3/3.1 models
        "unsloth/llama-3-8b-bnb-4bit": "meta-llama/Meta-Llama-3-8B",
        "unsloth/llama-3-8b-instruct-bnb-4bit": "meta-llama/Meta-Llama-3-8B-Instruct",
        "unsloth/llama-3.1-8b-bnb-4bit": "meta-llama/Meta-Llama-3.1-8B",
        "unsloth/llama-3.1-8b-instruct-bnb-4bit": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "unsloth/Meta-Llama-3.1-8B-bnb-4bit": "meta-llama/Meta-Llama-3.1-8B",
        # CodeLlama models
        "unsloth/codellama-7b-bnb-4bit": "codellama/CodeLlama-7b-hf",
        "unsloth/codellama-13b-bnb-4bit": "codellama/CodeLlama-13b-hf",
        "unsloth/codellama-34b-bnb-4bit": "codellama/CodeLlama-34b-hf",
        # Qwen models
        "unsloth/Qwen2-7B-bnb-4bit": "Qwen/Qwen2-7B",
        "unsloth/Qwen2-7B-Instruct-bnb-4bit": "Qwen/Qwen2-7B-Instruct",
        "unsloth/Qwen2.5-7B-bnb-4bit": "Qwen/Qwen2.5-7B",
        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit": "Qwen/Qwen2.5-7B-Instruct",
        "unsloth/Qwen2.5-Coder-7B-bnb-4bit": "Qwen/Qwen2.5-Coder-7B",
        "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit": "Qwen/Qwen2.5-Coder-7B-Instruct",
        # DeepSeek models
        "unsloth/deepseek-coder-6.7b-base-bnb-4bit": "deepseek-ai/deepseek-coder-6.7b-base",
        "unsloth/deepseek-coder-6.7b-instruct-bnb-4bit": "deepseek-ai/deepseek-coder-6.7b-instruct",
    }

    if base_model in model_mapping:
        return model_mapping[base_model]

    # Generic fallback: remove common suffixes
    result = base_model
    for suffix in ["-bnb-4bit", "-bnb-8bit", "-4bit", "-8bit"]:
        if result.endswith(suffix):
            result = result[:-len(suffix)]
            break

    return result


def merge_lora_adapter(
    adapter_path: Path,
    output_path: Path,
    base_model_name: Optional[str] = None
) -> Path:
    """Merge LoRA adapter with base model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    adapter_config = load_adapter_config(adapter_path)

    if base_model_name is None:
        base_model_name = get_base_model_name(adapter_config)

    logger.info(f"Base model: {base_model_name}")

    # Security check: validate model is from trusted source before enabling remote code
    if not is_trusted_model(base_model_name):
        logger.error(f"Untrusted model source: {base_model_name}")
        logger.error(f"Trusted sources: {TRUSTED_MODEL_PREFIXES}")
        raise ValueError(f"Model '{base_model_name}' is not from a trusted source. "
                        "Add to TRUSTED_MODEL_PREFIXES if this is a legitimate model.")

    logger.info(f"Loading base model (this may take a while on first run)...")

    # Load base model in float16 for merging
    # trust_remote_code is safe here because we validated the model source above
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    logger.info("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, str(adapter_path))

    logger.info("Merging LoRA weights...")
    model = model.merge_and_unload()

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    merged_path = output_path / "merged"
    merged_path.mkdir(exist_ok=True)

    logger.info(f"Saving merged model to {merged_path}...")
    model.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)

    logger.info("Merge complete!")
    return merged_path


def convert_to_gguf(
    model_path: Path,
    output_path: Path,
    adapter_path: Path,
    quantization: str = "q4_k_m"
) -> Optional[Path]:
    """
    Convert HuggingFace model to GGUF format using llama.cpp's convert script.

    Args:
        model_path: Path to the merged HuggingFace model
        output_path: Base output directory
        adapter_path: Original adapter path (used for naming)
        quantization: GGUF quantization method

    Returns:
        Path to the gguf directory containing the converted model, or None if conversion failed
    """
    # Find llama.cpp convert script or use built-in conversion
    gguf_path = output_path / "gguf"
    gguf_path.mkdir(parents=True, exist_ok=True)

    # Generate dynamic filename based on adapter
    gguf_filename = get_gguf_filename(adapter_path, quantization)
    output_file = gguf_path / gguf_filename

    logger.info(f"Target GGUF file: {output_file}")

    # Try using llama-cpp-python's convert functionality
    try:
        # First, try the python conversion approach
        logger.info("Converting to GGUF format...")

        # Use huggingface-hub's gguf conversion if available
        try:
            from transformers import AutoModelForCausalLM

            # Check if model has GGUF conversion support
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                device_map="cpu",
                low_cpu_mem_usage=True,
            )

            # Some models support direct GGUF export
            if hasattr(model, 'save_pretrained_gguf'):
                logger.info("Using native GGUF export...")
                model.save_pretrained_gguf(str(gguf_path), quantization_method=quantization)
                return gguf_path
        except Exception as e:
            logger.debug(f"Native GGUF export not available: {e}")

        # Fall back to llama.cpp convert.py script approach
        logger.info("Using llama.cpp conversion approach...")

        # Check for llama.cpp installation
        llama_cpp_path = shutil.which("convert-hf-to-gguf") or shutil.which("convert.py")

        if llama_cpp_path:
            logger.info(f"Found llama.cpp converter at {llama_cpp_path}")
            result = subprocess.run(
                [llama_cpp_path, str(model_path), "--outfile", str(output_file)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info(f"GGUF saved to {output_file}")
                return gguf_path
            else:
                logger.warning(f"Conversion failed: {result.stderr}")

        # Try pip-installed converter
        try:
            result = subprocess.run(
                [sys.executable, "-m", "llama_cpp.convert", str(model_path),
                 "--outfile", str(output_file)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return gguf_path
        except Exception:
            pass

        # Provide manual instructions
        logger.warning("Automatic GGUF conversion not available.")
        logger.info("To convert manually:")
        logger.info("1. Clone llama.cpp: git clone https://github.com/ggml-org/llama.cpp")
        logger.info("2. Install requirements: pip install -r llama.cpp/requirements.txt")
        logger.info(f"3. Run: python llama.cpp/convert_hf_to_gguf.py {model_path} --outfile {output_file}")
        logger.info(f"4. Optionally quantize: llama-quantize {output_file} {output_file.with_suffix('')}-{quantization}.gguf {quantization}")

        return None

    except Exception as e:
        logger.error(f"GGUF conversion error: {e}")
        return None


def create_ollama_modelfile(
    model_path: Path,
    model_name: str = "l4d2-code"
) -> Optional[Path]:
    """
    Create an Ollama Modelfile.

    Args:
        model_path: Path to the directory containing GGUF file(s)
        model_name: Name for the Ollama model

    Returns:
        Path to the created Modelfile, or None if no GGUF files found
    """
    # Find GGUF file (prefer the one with the model name)
    gguf_files = list(model_path.glob("*.gguf"))
    if not gguf_files:
        # Check subdirectories
        gguf_files = list(model_path.glob("**/*.gguf"))

    if not gguf_files:
        logger.error("No GGUF files found in {model_path}")
        return None

    # If multiple files, try to find the best match
    gguf_file = gguf_files[0]
    if len(gguf_files) > 1:
        # Prefer q4_k_m quantization as default
        for f in gguf_files:
            if "q4_k_m" in f.name:
                gguf_file = f
                break
        logger.info(f"Multiple GGUF files found, using: {gguf_file.name}")

    # Determine context size based on model
    ctx_size = 4096
    if "qwen" in model_name.lower() or "llama-3" in model_name.lower():
        ctx_size = 8192  # Newer models support longer context

    modelfile_content = f'''# Ollama Modelfile for L4D2 Code Assistant
# Model: {model_name}
# Generated by L4D2-AI-Architect

FROM {gguf_file.name}

# System prompt for L4D2 code generation
SYSTEM """You are an expert SourcePawn and VScript developer specializing in Left 4 Dead 2 modding. You write clean, efficient, and well-documented code following best practices. You understand the Source Engine, SourceMod API, and L4D2 VScript system."""

# Parameters optimized for code generation
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx {ctx_size}
PARAMETER stop "<|im_end|>"

# Template (ChatML format)
TEMPLATE """{{{{- if .System }}}}
<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{- end }}}}
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
"""

# License
LICENSE """
MIT License - L4D2-AI-Architect
Fine-tuned for Left 4 Dead 2 modding
Model: {model_name}
"""
'''

    modelfile_path = model_path / "Modelfile"
    safe_write_text(str(modelfile_path), modelfile_content, PROJECT_ROOT)

    logger.info(f"Modelfile created at {modelfile_path}")
    logger.info(f"To install in Ollama:")
    logger.info(f"  cd {model_path}")
    logger.info(f"  ollama create {model_name} -f Modelfile")

    return modelfile_path


def install_to_ollama(model_path: Path, model_name: str) -> bool:
    """Install the model to Ollama."""
    modelfile_path = model_path / "Modelfile"

    if not modelfile_path.exists():
        logger.error("Modelfile not found")
        return False

    # Check Ollama
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Ollama not available")
            return False
    except FileNotFoundError:
        logger.error("Ollama not installed")
        return False

    # Create model
    logger.info(f"Creating Ollama model: {model_name}")
    result = subprocess.run(
        ["ollama", "create", model_name, "-f", str(modelfile_path)],
        cwd=str(model_path),
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        logger.info(f"Model installed! Run with: ollama run {model_name}")
        return True
    else:
        logger.error(f"Failed: {result.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Export LoRA adapter to GGUF format for Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export v15 model
  python export_gguf_cpu.py --adapter model_adapters/l4d2-mistral-v15-lora/final

  # Export with custom output directory
  python export_gguf_cpu.py --adapter model_adapters/l4d2-mistral-v15-lora/final --output exports/l4d2-v15

  # Export and install to Ollama
  python export_gguf_cpu.py --adapter model_adapters/l4d2-mistral-v15-lora/final --install-ollama l4d2-code-v15
        """
    )
    parser.add_argument("--adapter", type=str, required=True,
                        help="Path to LoRA adapter directory (e.g., model_adapters/l4d2-mistral-v15-lora/final)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: auto-generated from adapter name)")
    parser.add_argument("--base-model", type=str,
                        help="Override base model name (auto-detected from adapter config)")
    parser.add_argument("--quantize", type=str, default="q4_k_m",
                        choices=QUANT_OPTIONS,
                        help="Quantization method (default: q4_k_m)")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Skip merge step (use existing merged model)")
    parser.add_argument("--create-modelfile", action="store_true",
                        help="Create Ollama Modelfile")
    parser.add_argument("--install-ollama", type=str,
                        help="Install to Ollama with given name")

    args = parser.parse_args()

    if not check_dependencies():
        sys.exit(1)

    # Validate adapter path
    adapter_path = safe_path(args.adapter, PROJECT_ROOT)
    if not adapter_path.exists():
        logger.error(f"Adapter not found: {adapter_path}")
        sys.exit(1)

    # Validate adapter has required files
    adapter_config_path = adapter_path / "adapter_config.json"
    if not adapter_config_path.exists():
        logger.error(f"Invalid adapter directory: {adapter_path}")
        logger.error("Expected adapter_config.json not found")
        logger.info("Make sure you're pointing to the 'final' subdirectory of the adapter")
        sys.exit(1)

    # Auto-generate output path from adapter name if not specified
    if args.output is None:
        model_name, version = extract_model_info(adapter_path)
        if version:
            output_dir_name = f"{model_name}-{version}".replace("l4d2-", "l4d2-")
        else:
            output_dir_name = model_name
        args.output = f"exports/{output_dir_name}"
        logger.info(f"Auto-generated output directory: {args.output}")

    output_path = safe_path(args.output, PROJECT_ROOT, create_parents=True)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Merge LoRA with base model
    merged_path = output_path / "merged"
    if not args.skip_merge or not merged_path.exists():
        try:
            merged_path = merge_lora_adapter(adapter_path, output_path, args.base_model)
        except Exception as e:
            logger.error(f"Failed to merge LoRA adapter: {e}")
            logger.info("Check that you have enough RAM (16GB+ recommended for 7B models)")
            sys.exit(1)
    else:
        logger.info(f"Using existing merged model at {merged_path}")

    # Step 2: Convert to GGUF
    gguf_path = convert_to_gguf(merged_path, output_path, adapter_path, args.quantize)

    if gguf_path:
        # Determine model name for Ollama
        model_name, version = extract_model_info(adapter_path)
        ollama_model_name = args.install_ollama
        if not ollama_model_name:
            ollama_model_name = f"{model_name}-{version}" if version else model_name

        # Step 3: Create Modelfile
        if args.create_modelfile or args.install_ollama:
            create_ollama_modelfile(gguf_path, ollama_model_name)

        # Step 4: Install to Ollama
        if args.install_ollama:
            install_to_ollama(gguf_path, args.install_ollama)
    else:
        logger.warning("GGUF conversion was not completed automatically")
        logger.info("The merged model is available for manual conversion")

    # Summary
    print("\n" + "=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)
    print(f"Adapter: {adapter_path}")
    print(f"Output:  {output_path}")
    if merged_path and merged_path.exists():
        print(f"Merged:  {merged_path}")
    if gguf_path and gguf_path.exists():
        gguf_files = list(gguf_path.glob("*.gguf"))
        if gguf_files:
            for f in gguf_files:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"GGUF:    {f.name} ({size_mb:.1f} MB)")
        else:
            print("GGUF:    No GGUF files created (see manual instructions above)")
    else:
        print("GGUF:    Conversion not completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
