#!/usr/bin/env python3
"""
Unsloth QLoRA Fine-Tuning Script

Fine-tunes a language model on SourcePawn/VScript code using Unsloth for
2x faster training with 60% less VRAM.

Usage:
    python train_unsloth.py --config configs/unsloth_config.yaml
    python train_unsloth.py --model mistral-7b --dataset data/processed/combined_train.jsonl
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add parent to path for security utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_path, safe_read_yaml, safe_write_json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "model_adapters"

def _resolve_data_path(path_str: str) -> Path:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return safe_path(str(candidate), PROJECT_ROOT)
    return safe_path(str(DATA_DIR / candidate), PROJECT_ROOT)


def _resolve_output_dir(dir_name: str) -> Path:
    return safe_path(str(OUTPUT_DIR / dir_name), PROJECT_ROOT, create_parents=True)


def check_dependencies():
    """Check and install required dependencies."""
    try:
        import torch
        import unsloth
        import yaml
        from transformers import TrainingArguments
        from trl import SFTTrainer
        from datasets import load_dataset

        _ = (torch, unsloth, yaml, TrainingArguments, SFTTrainer, load_dataset)
        logger.info("All dependencies available")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Installing dependencies...")
        
        # Install Unsloth and dependencies
        install_cmds = [
            f"{sys.executable} -m pip install --upgrade pip",
            f"{sys.executable} -m pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'",
            f"{sys.executable} -m pip install --no-deps trl peft accelerate bitsandbytes",
            f"{sys.executable} -m pip install datasets transformers sentencepiece protobuf pyyaml",
        ]
        
        for cmd in install_cmds:
            subprocess.run(cmd.split(), check=False)

        # Optional acceleration library - may not have wheels on all platforms
        xformers_result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "xformers"],
            check=False
        )
        if xformers_result.returncode != 0:
            logger.warning("xformers install failed; continuing without it")
        
        return False


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load training configuration from YAML file with path validation."""
    if config_path and config_path.exists():
        # Use safe_read_yaml which combines path validation and YAML loading
        config = safe_read_yaml(str(config_path), PROJECT_ROOT)
        logger.info(f"Loaded config from {config_path}")
        return config
    
    # Default configuration
    return {
        "model": {
            "name": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
            "max_seq_length": 2048,
            "dtype": None,  # Auto-detect
            "load_in_4bit": True,
        },
        "lora": {
            "r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "bias": "none",
            "use_gradient_checkpointing": "unsloth",
            "use_rslora": False,
        },
        "training": {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
            "warmup_steps": 10,
            "lr_scheduler_type": "linear",
            "optim": "adamw_8bit",
            "fp16": False,
            "bf16": True,  # Use bfloat16 for Ampere+
            "logging_steps": 10,
            "save_steps": 100,
            "save_total_limit": 3,
            "seed": 3407,
        },
        "data": {
            "train_file": "combined_train.jsonl",
            "val_file": "combined_val.jsonl",
            "max_samples": None,  # Use all
        },
        "output": {
            "dir": "l4d2-mistral-v10plus-lora",
            "push_to_hub": False,
            "hub_model_id": None,
        }
    }


def log_gpu_memory(label: str):
    """Log GPU memory statistics for debugging."""
    import torch
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"[{label}] GPU Memory: {allocated:.2f}GB allocated, "
                   f"{reserved:.2f}GB reserved, {max_allocated:.2f}GB peak")


def setup_model(config: Dict[str, Any]):
    """Load and configure the model with LoRA."""
    from unsloth import FastLanguageModel
    
    model_config = config["model"]
    lora_config = config["lora"]
    advanced_config = config.get("advanced", {})
    
    logger.info(f"Loading model: {model_config['name']}")
    
    # Load base model
    model_load_kwargs: Dict[str, Any] = {}
    if advanced_config.get("use_flash_attention_2"):
        try:
            import flash_attn

            _ = flash_attn
            model_load_kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:
            logger.warning("Flash Attention 2 requested but not available; continuing without it")

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_config["name"],
            max_seq_length=model_config["max_seq_length"],
            dtype=model_config["dtype"],
            load_in_4bit=model_config["load_in_4bit"],
            **model_load_kwargs,
        )
    except TypeError:
        if "attn_implementation" not in model_load_kwargs:
            raise
        logger.warning("attn_implementation not supported by FastLanguageModel; retrying without it")
        model_load_kwargs.pop("attn_implementation", None)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_config["name"],
            max_seq_length=model_config["max_seq_length"],
            dtype=model_config["dtype"],
            load_in_4bit=model_config["load_in_4bit"],
            **model_load_kwargs,
        )
    
    logger.info("Applying LoRA adapters...")
    
    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        target_modules=lora_config["target_modules"],
        bias=lora_config["bias"],
        use_gradient_checkpointing=lora_config["use_gradient_checkpointing"],
        random_state=config["training"]["seed"],
        use_rslora=lora_config["use_rslora"],
    )
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model, tokenizer


def load_dataset_from_jsonl(file_path: Path, tokenizer, max_samples: Optional[int] = None):
    """Load and format dataset from JSONL file."""
    from datasets import Dataset
    
    logger.info(f"Loading dataset from {file_path}")
    
    data = []
    skipped = 0
    total_lines = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            total_lines = line_num
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                skipped += 1
                if skipped <= 5:  # Log first few errors
                    logger.warning(f"Line {line_num}: Invalid JSON - {e}")

    # Report skipped lines
    if skipped > 0:
        skip_pct = 100 * skipped / total_lines if total_lines > 0 else 0
        logger.warning(f"Skipped {skipped} malformed lines ({skip_pct:.1f}%)")
        if skip_pct > 10:  # >10% failure rate is suspicious
            raise ValueError(f"Too many malformed lines: {skipped}/{total_lines} ({skip_pct:.1f}%). Check dataset integrity.")

    if max_samples:
        data = data[:max_samples]

    logger.info(f"Loaded {len(data)} examples")
    
    # Format for training
    def format_example(example):
        """Format example using chat template."""
        messages = example.get("messages", [])
        if not messages:
            # Handle Alpaca format
            text = f"### Instruction:\n{example.get('instruction', '')}\n\n### Response:\n{example.get('output', '')}"
        else:
            # Handle ChatML format
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}
    
    dataset = Dataset.from_list(data)
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    
    return dataset


def train(config: Dict[str, Any], resume_from: Optional[str] = None):
    """Run the training loop."""
    from transformers import TrainingArguments
    from trl import SFTTrainer

    # Validate training data exists BEFORE loading model (saves GPU time if missing)
    data_config = config["data"]
    train_path = _resolve_data_path(data_config["train_file"])

    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        logger.error("Please ensure training data exists before starting expensive GPU training.")
        raise FileNotFoundError(f"Training data not found: {train_path}")

    if train_path.stat().st_size == 0:
        logger.error(f"Training data file is empty: {train_path}")
        raise ValueError(f"Training data file is empty: {train_path}")

    logger.info(f"Training data validated: {train_path} ({train_path.stat().st_size / 1024:.1f} KB)")

    # Setup model
    model, tokenizer = setup_model(config)
    log_gpu_memory("After model load")

    # Load datasets
    train_dataset = load_dataset_from_jsonl(
        train_path,
        tokenizer,
        data_config.get("max_samples")
    )
    
    val_dataset = None
    if data_config.get("val_file"):
        val_path = _resolve_data_path(data_config["val_file"])
        if val_path.exists():
            val_dataset = load_dataset_from_jsonl(val_path, tokenizer)
            logger.info(f"Validation data loaded: {len(val_dataset)} samples")
        else:
            logger.warning(f"Validation file specified but not found: {val_path}")
            logger.warning("Training will continue without validation data.")
    else:
        logger.info("No validation file specified, skipping evaluation during training.")
    
    # Output directory
    output_dir = _resolve_output_dir(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training arguments
    train_config = config["training"]
    monitoring_config = config.get("monitoring", {})
    advanced_config = config.get("advanced", {})

    logging_base = monitoring_config.get("logging_dir", "data/training_logs")
    logging_dir = safe_path(
        str(Path(logging_base) / config["output"]["dir"]),
        PROJECT_ROOT,
        create_parents=True,
    )
    logging_dir.mkdir(parents=True, exist_ok=True)

    eval_strategy_value = "no"
    eval_steps = None
    if val_dataset:
        eval_strategy_value = monitoring_config.get("evaluation_strategy", "steps")
        if eval_strategy_value == "steps":
            eval_steps = monitoring_config.get("eval_steps", train_config["save_steps"])

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_config["num_train_epochs"],
        per_device_train_batch_size=train_config["per_device_train_batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        learning_rate=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
        warmup_steps=train_config["warmup_steps"],
        lr_scheduler_type=train_config["lr_scheduler_type"],
        optim=train_config["optim"],
        fp16=train_config["fp16"],
        bf16=train_config["bf16"],
        max_grad_norm=train_config.get("max_grad_norm", 1.0),
        logging_steps=train_config["logging_steps"],
        save_steps=train_config["save_steps"],
        save_total_limit=train_config["save_total_limit"],
        seed=train_config["seed"],
        eval_strategy=eval_strategy_value,  # Renamed from evaluation_strategy in transformers 4.46+
        eval_steps=eval_steps,
        report_to=monitoring_config.get("report_to", "tensorboard"),
        logging_dir=str(logging_dir),
        dataloader_num_workers=advanced_config.get("dataloader_num_workers", 0),
        dataloader_pin_memory=advanced_config.get("dataloader_pin_memory", True),
        tf32=advanced_config.get("tf32", False),
    )
    
    # Initialize trainer with optional packing for better GPU utilization
    packing_enabled = advanced_config.get("packing", False)
    if packing_enabled:
        logger.info("Dataset packing ENABLED - combining short samples for better GPU utilization")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=config["model"]["max_seq_length"],
        packing=packing_enabled,
        args=training_args,
    )
    
    # Train with OOM detection
    logger.info("Starting training...")
    log_gpu_memory("Before training")
    start_time = datetime.now()

    try:
        if resume_from:
            trainer.train(resume_from_checkpoint=resume_from)
        else:
            trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            log_gpu_memory("OOM occurred")
            logger.error("=" * 60)
            logger.error("GPU OUT OF MEMORY ERROR")
            logger.error("=" * 60)
            logger.error("Suggestions to fix:")
            logger.error("  1. Reduce per_device_train_batch_size (try halving it)")
            logger.error("  2. Increase gradient_accumulation_steps to compensate")
            logger.error("  3. Reduce max_seq_length if possible")
            logger.error("  4. Use a '_fast' config variant if available")
            logger.error("  5. Ensure no other processes are using GPU memory")
            logger.error("=" * 60)
        raise

    elapsed = datetime.now() - start_time
    log_gpu_memory("After training")
    logger.info(f"Training completed in {elapsed}")
    
    # Save final model (LoRA adapter)
    logger.info("Saving LoRA adapter...")
    final_path = output_dir / "final"
    final_path.mkdir(parents=True, exist_ok=True)

    try:
        # Primary method: save_pretrained for LoRA adapters
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        logger.info(f"LoRA adapter saved to {final_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        # Fallback: try PEFT's save method directly
        try:
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(final_path)
                tokenizer.save_pretrained(final_path)
                logger.info(f"Model saved using fallback method to {final_path}")
        except Exception as e2:
            logger.error(f"Fallback save also failed: {e2}")
            raise

    # Optionally save merged model for easier export (if space permits)
    try:
        merged_path = output_dir / "merged_16bit"
        logger.info(f"Attempting to save merged 16-bit model to {merged_path}...")
        model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
        logger.info(f"Merged model saved to {merged_path}")
    except Exception as e:
        logger.warning(f"Could not save merged model (this is optional): {e}")
        logger.info("LoRA adapter was saved successfully - you can merge later during export.")
    
    # Push to hub if configured
    if config["output"]["push_to_hub"] and config["output"]["hub_model_id"]:
        logger.info(f"Pushing to hub: {config['output']['hub_model_id']}")
        model.push_to_hub(config["output"]["hub_model_id"], tokenizer)
    
    # Save training info
    info = {
        "config": config,
        "training_time": str(elapsed),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset) if val_dataset else 0,
        "completed_at": datetime.now().isoformat(),
    }

    # Save training info using safe_write_json
    safe_write_json(
        str(output_dir / "training_info.json"),
        info,
        PROJECT_ROOT
    )
    
    log_gpu_memory("After save")
    logger.info(f"Model saved to {output_dir}")
    return output_dir


def test_inference(model_path: Path, prompt: str):
    """Test the trained model with a sample prompt."""
    import torch
    from unsloth import FastLanguageModel

    logger.info(f"Loading model from {model_path}")

    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("CUDA not available, using CPU (this will be slow)")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=2048,
        load_in_4bit=True if device == "cuda" else False,
    )

    FastLanguageModel.for_inference(model)

    # Format prompt
    messages = [
        {"role": "system", "content": "You are an expert SourcePawn developer specializing in Left 4 Dead 2."},
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)
    
    # Generate
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n" + "=" * 50)
    print("GENERATED RESPONSE:")
    print("=" * 50)
    print(response)
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM with Unsloth")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--model", type=str, help="Model name override")
    parser.add_argument("--dataset", type=str, help="Training dataset path")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--test", type=str, help="Test with prompt after training")
    parser.add_argument("--test-only", type=str, help="Only test existing model")
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        logger.info("Dependencies installed. Please restart the script.")
        sys.exit(0)
    
    # Test only mode
    if args.test_only:
        test_inference(Path(args.test_only), args.test or "Write a function to heal all survivors")
        return
    
    # Load config
    config_path = Path(args.config) if args.config else CONFIG_DIR / "unsloth_config.yaml"
    config = load_config(config_path if config_path.exists() else None)
    
    # Override with CLI args
    if args.model:
        config["model"]["name"] = args.model
    if args.dataset:
        config["data"]["train_file"] = args.dataset
    if args.epochs:
        config["training"]["num_train_epochs"] = args.epochs
    if args.batch_size:
        config["training"]["per_device_train_batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    
    # Verify GPU
    import torch
    if not torch.cuda.is_available():
        logger.error("CUDA not available! This script requires a GPU.")
        sys.exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Train
    output_dir = train(config, resume_from=args.resume)
    
    # Test if requested
    if args.test:
        test_inference(output_dir / "final", args.test)


if __name__ == "__main__":
    main()
