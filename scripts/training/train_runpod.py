#!/usr/bin/env python3
"""
RunPod/Vultr-compatible LoRA Training Script

Works without Unsloth - uses standard HuggingFace transformers + PEFT + TRL.
Tested on RunPod A40 GPU and Vultr A100.

Auto-detects GPU type and optimizes settings:
  - A100: Uses BF16, higher batch size, TF32 acceleration
  - A40/Other: Uses FP16 with standard settings

Usage:
    python train_runpod.py                    # Use defaults (TinyLlama)
    python train_runpod.py --model mistral    # Use Mistral-7B
    python train_runpod.py --max-steps 500    # Limit training steps
    python train_runpod.py --model mistral --batch-size 8  # A100 recommended
"""

import argparse
import sys
import torch
from pathlib import Path

# Add parent to path for security utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_path

# Project root for path validation
PROJECT_ROOT = Path(__file__).parent.parent.parent


def check_flash_attention():
    """Check if Flash Attention 2 is available."""
    try:
        import flash_attn
        return True
    except ImportError:
        return False


def detect_gpu_type():
    """Detect GPU and return optimized settings."""
    if not torch.cuda.is_available():
        return {
            "name": "CPU",
            "vram_gb": 0,
            "use_bf16": False,
            "use_tf32": False,
            "use_flash_attn": False,
            "recommended_batch_size": 1,
            "dataloader_workers": 0,
        }

    gpu_name = torch.cuda.get_device_name(0).lower()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    flash_available = check_flash_attention()

    # A100 detection - supports BF16 natively, TF32, and Flash Attention 2
    if "a100" in gpu_name:
        return {
            "name": torch.cuda.get_device_name(0),
            "vram_gb": vram_gb,
            "use_bf16": True,  # Native BF16 support
            "use_tf32": True,  # TF32 acceleration
            "use_flash_attn": flash_available,  # Flash Attention 2 if installed
            "recommended_batch_size": 8,
            "dataloader_workers": 4,
        }
    # H100/H200 - also supports BF16, TF32, and Flash Attention 2
    elif "h100" in gpu_name or "h200" in gpu_name:
        return {
            "name": torch.cuda.get_device_name(0),
            "vram_gb": vram_gb,
            "use_bf16": True,
            "use_tf32": True,
            "use_flash_attn": flash_available,
            "recommended_batch_size": 16,
            "dataloader_workers": 4,
        }
    # A40, L40S, RTX 4090 - use FP16, Flash Attention if available
    else:
        return {
            "name": torch.cuda.get_device_name(0),
            "vram_gb": vram_gb,
            "use_bf16": False,  # FP16 is safer
            "use_tf32": "a40" in gpu_name or "l40" in gpu_name,
            "use_flash_attn": flash_available and ("a40" in gpu_name or "l40" in gpu_name or "4090" in gpu_name),
            "recommended_batch_size": 4,
            "dataloader_workers": 2,
        }


def main():
    parser = argparse.ArgumentParser(description="LoRA Training for L4D2")
    parser.add_argument("--model", choices=["tiny", "mistral"], default="tiny",
                       help="Base model: tiny (1.1B) or mistral (7B)")
    parser.add_argument("--max-steps", type=int, default=None,
                       help="Max training steps (None = full epochs)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--output", type=str, default="model_adapters/l4d2-mistral-v10plus-lora",
                       help="Output directory")
    args = parser.parse_args()

    # Import dependencies
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset

    print("=" * 60)
    print("L4D2 LoRA Training")
    print("=" * 60)

    # Select model
    if args.model == "tiny":
        BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        use_quantization = False
    else:
        BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
        use_quantization = True  # 4-bit for Mistral to fit in memory

    print(f"Base Model: {BASE_MODEL}")
    print(f"Output: {args.output}")

    # Detect GPU and get optimized settings
    gpu_info = detect_gpu_type()
    print(f"GPU: {gpu_info['name']} ({gpu_info['vram_gb']:.1f} GB)")

    if gpu_info["vram_gb"] == 0:
        print("WARNING: No GPU detected!")
    else:
        precision = "BF16" if gpu_info["use_bf16"] else "FP16"
        print(f"  Precision: {precision}")
        if gpu_info["use_tf32"]:
            print(f"  TF32: Enabled")
        if gpu_info["use_flash_attn"]:
            print(f"  Flash Attention 2: Enabled")
        else:
            print(f"  Flash Attention 2: Not available (install with: pip install flash-attn --no-build-isolation)")
        print(f"  Recommended batch size: {gpu_info['recommended_batch_size']}")

        # Enable TF32 if supported
        if gpu_info["use_tf32"]:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # Load model with GPU-optimized dtype and attention implementation
    compute_dtype = torch.bfloat16 if gpu_info["use_bf16"] else torch.float16
    attn_impl = "flash_attention_2" if gpu_info["use_flash_attn"] else "sdpa"  # sdpa is default efficient attention
    print(f"\nLoading model (dtype: {compute_dtype}, attention: {attn_impl})...")

    if use_quantization:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation=attn_impl,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=compute_dtype,
            device_map="auto",
            attn_implementation=attn_impl,
        )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    print("\nLoading dataset...")
    train_path = "data/processed/combined_train.jsonl"
    val_path = "data/processed/combined_val.jsonl"

    train_dataset = load_dataset("json", data_files=train_path, split="train")
    val_dataset = load_dataset("json", data_files=val_path, split="train") if Path(val_path).exists() else None

    def format_example(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

    train_dataset = train_dataset.map(format_example)
    if val_dataset:
        val_dataset = val_dataset.map(format_example)

    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")

    # Training config - sanitize output path to prevent path traversal
    output_dir = safe_path(args.output, PROJECT_ROOT, create_parents=True)

    # Sanitize logging directory path
    logging_dir = safe_path(
        f"data/training_logs/{output_dir.name}",
        PROJECT_ROOT,
        create_parents=True
    )

    # Determine effective batch size (use provided or GPU-recommended)
    effective_batch_size = args.batch_size
    if args.batch_size == 4 and gpu_info["recommended_batch_size"] > 4:
        # User didn't specify, use recommended
        effective_batch_size = gpu_info["recommended_batch_size"]
        print(f"Using recommended batch size: {effective_batch_size}")

    # Adjust gradient accumulation to maintain effective batch of 16
    grad_accum = max(1, 16 // effective_batch_size)

    config = SFTConfig(
        output_dir=str(output_dir),
        dataset_text_field="text",
        max_length=1024 if args.model == "tiny" else 2048,
        per_device_train_batch_size=effective_batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=args.epochs if not args.max_steps else 1,
        max_steps=args.max_steps if args.max_steps else -1,
        learning_rate=2e-4,
        warmup_steps=50,
        logging_steps=10,
        save_steps=100,
        eval_steps=100 if val_dataset else None,
        eval_strategy="steps" if val_dataset else "no",
        # Use BF16 on A100/H100, FP16 on others
        fp16=not gpu_info["use_bf16"],
        bf16=gpu_info["use_bf16"],
        report_to="tensorboard",
        logging_dir=str(logging_dir),
        save_total_limit=3,
        # Dataloader optimizations
        dataloader_num_workers=gpu_info["dataloader_workers"],
        dataloader_pin_memory=True if gpu_info["vram_gb"] > 0 else False,
    )

    # Train
    print("\nStarting training...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        args=config,
    )

    trainer.train()
    trainer.save_model()

    print(f"\nTraining complete! Model saved to {output_dir}")
    print("\nTo test inference:")
    print(f"  python scripts/inference/test_lora.py --adapter {output_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
