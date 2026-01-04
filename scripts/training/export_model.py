#!/usr/bin/env python3
"""
Model Export Script

Exports fine-tuned LoRA adapters to various formats:
- GGUF for llama.cpp / Ollama
- Merged safetensors
- HuggingFace Hub

Usage:
    python export_model.py --input model_adapters/l4d2-code-lora/final --format gguf --quantize q4_k_m
    python export_model.py --input model_adapters/l4d2-code-lora/final --push-to-hub username/model-name
"""

import os
import sys
import json
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "model_adapters"

# GGUF quantization options (from llama.cpp)
QUANTIZATION_OPTIONS = {
    "f16": "16-bit float, no quantization (largest, best quality)",
    "q8_0": "8-bit quantization (good balance)",
    "q6_k": "6-bit quantization (smaller, good quality)",
    "q5_k_m": "5-bit quantization medium",
    "q5_k_s": "5-bit quantization small",
    "q4_k_m": "4-bit quantization medium (recommended)",
    "q4_k_s": "4-bit quantization small",
    "q3_k_m": "3-bit quantization medium",
    "q3_k_s": "3-bit quantization small",
    "q2_k": "2-bit quantization (smallest, lowest quality)",
}


def check_unsloth():
    """Check if Unsloth is available."""
    try:
        from unsloth import FastLanguageModel
        return True
    except ImportError:
        logger.error("Unsloth not installed. Run: pip install unsloth")
        return False


def export_to_gguf_unsloth(
    model_path: Path,
    output_path: Path,
    quantization: str = "q4_k_m"
) -> Path:
    """Export model to GGUF using Unsloth's built-in method."""
    from unsloth import FastLanguageModel
    
    logger.info(f"Loading model from {model_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=2048,
        load_in_4bit=True,
    )
    
    output_path.mkdir(parents=True, exist_ok=True)
    gguf_path = output_path / f"model-{quantization}.gguf"
    
    logger.info(f"Exporting to GGUF with {quantization} quantization...")
    
    # Unsloth's built-in GGUF export
    model.save_pretrained_gguf(
        str(output_path),
        tokenizer,
        quantization_method=quantization,
    )
    
    logger.info(f"GGUF saved to {output_path}")
    return output_path


def export_merged_model(
    model_path: Path,
    output_path: Path,
    push_to_hub: Optional[str] = None
) -> Path:
    """Export merged model (base + LoRA) to safetensors."""
    from unsloth import FastLanguageModel
    
    logger.info(f"Loading model from {model_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=2048,
        load_in_4bit=True,
    )
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Merging LoRA weights and saving...")
    
    # Save merged 16-bit model
    model.save_pretrained_merged(
        str(output_path),
        tokenizer,
        save_method="merged_16bit",
    )
    
    if push_to_hub:
        logger.info(f"Pushing to HuggingFace Hub: {push_to_hub}")
        model.push_to_hub_merged(
            push_to_hub,
            tokenizer,
            save_method="merged_16bit",
        )
    
    logger.info(f"Merged model saved to {output_path}")
    return output_path


def export_lora_only(
    model_path: Path,
    output_path: Path,
    push_to_hub: Optional[str] = None
) -> Path:
    """Export only the LoRA adapter weights."""
    from unsloth import FastLanguageModel
    
    logger.info(f"Loading model from {model_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=2048,
        load_in_4bit=True,
    )
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Saving LoRA adapter...")
    
    model.save_pretrained_merged(
        str(output_path),
        tokenizer,
        save_method="lora",
    )
    
    if push_to_hub:
        logger.info(f"Pushing LoRA to HuggingFace Hub: {push_to_hub}")
        model.push_to_hub_merged(
            push_to_hub,
            tokenizer,
            save_method="lora",
        )
    
    logger.info(f"LoRA adapter saved to {output_path}")
    return output_path


def create_ollama_modelfile(
    model_path: Path,
    model_name: str,
    system_prompt: Optional[str] = None
) -> Path:
    """Create an Ollama Modelfile for the exported model."""
    
    if system_prompt is None:
        system_prompt = """You are an expert SourcePawn and VScript developer specializing in Left 4 Dead 2 modding. You write clean, efficient, and well-documented code following best practices. You understand the Source Engine, SourceMod API, and L4D2 VScript system."""
    
    # Find the GGUF file
    gguf_files = list(model_path.glob("*.gguf"))
    if not gguf_files:
        logger.error(f"No GGUF files found in {model_path}")
        return None
    
    gguf_file = gguf_files[0]
    
    modelfile_content = f'''# Ollama Modelfile for {model_name}
# Generated by L4D2-AI-Architect

FROM {gguf_file.name}

# System prompt for L4D2 code generation
SYSTEM """
{system_prompt}
"""

# Parameters optimized for code generation
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048

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
"""
'''
    
    modelfile_path = model_path / "Modelfile"
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    
    logger.info(f"Ollama Modelfile created at {modelfile_path}")
    logger.info(f"To use with Ollama:")
    logger.info(f"  cd {model_path}")
    logger.info(f"  ollama create {model_name} -f Modelfile")
    
    return modelfile_path


def install_to_ollama(model_path: Path, model_name: str) -> bool:
    """Install the model to Ollama."""
    modelfile_path = model_path / "Modelfile"
    
    if not modelfile_path.exists():
        logger.error("Modelfile not found. Create it first with --create-modelfile")
        return False
    
    # Check if Ollama is available
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Ollama not found. Install from https://ollama.ai")
            return False
    except FileNotFoundError:
        logger.error("Ollama not found. Install from https://ollama.ai")
        return False
    
    # Create model in Ollama
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
        logger.error(f"Failed to create Ollama model: {result.stderr}")
        return False


def list_quantization_options():
    """Print available quantization options."""
    print("\nAvailable quantization options:")
    print("-" * 60)
    for quant, desc in QUANTIZATION_OPTIONS.items():
        print(f"  {quant:12} - {desc}")
    print("-" * 60)
    print("\nRecommended: q4_k_m (good balance of size and quality)")


def main():
    parser = argparse.ArgumentParser(description="Export fine-tuned model")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to fine-tuned model directory")
    parser.add_argument("--output", type=str,
                        help="Output directory (default: input_dir/export)")
    parser.add_argument("--format", type=str, default="gguf",
                        choices=["gguf", "merged", "lora", "all"],
                        help="Export format")
    parser.add_argument("--quantize", type=str, default="q4_k_m",
                        help="GGUF quantization method")
    parser.add_argument("--push-to-hub", type=str,
                        help="Push to HuggingFace Hub (username/model-name)")
    parser.add_argument("--create-modelfile", action="store_true",
                        help="Create Ollama Modelfile")
    parser.add_argument("--install-ollama", type=str,
                        help="Install to Ollama with given name")
    parser.add_argument("--list-quants", action="store_true",
                        help="List quantization options")
    
    args = parser.parse_args()
    
    if args.list_quants:
        list_quantization_options()
        return
    
    if not check_unsloth():
        sys.exit(1)
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path not found: {input_path}")
        sys.exit(1)
    
    output_path = Path(args.output) if args.output else input_path / "export"
    
    # Export based on format
    if args.format in ["gguf", "all"]:
        gguf_output = output_path / "gguf"
        export_to_gguf_unsloth(input_path, gguf_output, args.quantize)
        
        if args.create_modelfile or args.install_ollama:
            create_ollama_modelfile(gguf_output, args.install_ollama or "l4d2-code")
        
        if args.install_ollama:
            install_to_ollama(gguf_output, args.install_ollama)
    
    if args.format in ["merged", "all"]:
        merged_output = output_path / "merged"
        export_merged_model(input_path, merged_output, args.push_to_hub)
    
    if args.format in ["lora", "all"]:
        lora_output = output_path / "lora"
        export_lora_only(input_path, lora_output, args.push_to_hub)
    
    logger.info("Export complete!")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)
    print(f"Source: {input_path}")
    print(f"Output: {output_path}")
    
    if (output_path / "gguf").exists():
        gguf_files = list((output_path / "gguf").glob("*.gguf"))
        for f in gguf_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"GGUF: {f.name} ({size_mb:.1f} MB)")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
