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
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

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


def check_dependencies():
    """Check and install required dependencies."""
    try:
        import torch
        import unsloth
        from transformers import TrainingArguments
        from trl import SFTTrainer
        from datasets import load_dataset
        logger.info("All dependencies available")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Installing dependencies...")
        
        # Install Unsloth and dependencies
        install_cmds = [
            f"{sys.executable} -m pip install --upgrade pip",
            f"{sys.executable} -m pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'",
            f"{sys.executable} -m pip install --no-deps xformers trl peft accelerate bitsandbytes",
            f"{sys.executable} -m pip install datasets transformers",
        ]
        
        for cmd in install_cmds:
            os.system(cmd)
        
        return False


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    if config_path and config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
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
            "dir": "l4d2-code-lora",
            "push_to_hub": False,
            "hub_model_id": None,
        }
    }


def setup_model(config: Dict[str, Any]):
    """Load and configure the model with LoRA."""
    from unsloth import FastLanguageModel
    
    model_config = config["model"]
    lora_config = config["lora"]
    
    logger.info(f"Loading model: {model_config['name']}")
    
    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config["name"],
        max_seq_length=model_config["max_seq_length"],
        dtype=model_config["dtype"],
        load_in_4bit=model_config["load_in_4bit"],
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
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError:
                continue
    
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
    from unsloth import FastLanguageModel
    
    # Setup model
    model, tokenizer = setup_model(config)
    
    # Load datasets
    data_config = config["data"]
    train_dataset = load_dataset_from_jsonl(
        DATA_DIR / data_config["train_file"],
        tokenizer,
        data_config.get("max_samples")
    )
    
    val_dataset = None
    if data_config.get("val_file"):
        val_path = DATA_DIR / data_config["val_file"]
        if val_path.exists():
            val_dataset = load_dataset_from_jsonl(val_path, tokenizer)
    
    # Output directory
    output_dir = OUTPUT_DIR / config["output"]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training arguments
    train_config = config["training"]
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
        logging_steps=train_config["logging_steps"],
        save_steps=train_config["save_steps"],
        save_total_limit=train_config["save_total_limit"],
        seed=train_config["seed"],
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=train_config["save_steps"] if val_dataset else None,
        report_to="tensorboard",
        logging_dir=str(output_dir / "logs"),
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=config["model"]["max_seq_length"],
        args=training_args,
    )
    
    # Train
    logger.info("Starting training...")
    start_time = datetime.now()
    
    if resume_from:
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()
    
    elapsed = datetime.now() - start_time
    logger.info(f"Training completed in {elapsed}")
    
    # Save final model
    logger.info("Saving final model...")
    model.save_pretrained(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")
    
    # Save LoRA adapter only (smaller file)
    logger.info("Saving LoRA adapter...")
    model.save_pretrained_merged(
        output_dir / "lora_adapter",
        tokenizer,
        save_method="lora",
    )
    
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
    
    with open(output_dir / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"Model saved to {output_dir}")
    return output_dir


def test_inference(model_path: Path, prompt: str):
    """Test the trained model with a sample prompt."""
    from unsloth import FastLanguageModel
    
    logger.info(f"Loading model from {model_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=2048,
        load_in_4bit=True,
    )
    
    FastLanguageModel.for_inference(model)
    
    # Format prompt
    messages = [
        {"role": "system", "content": "You are an expert SourcePawn developer."},
        {"role": "user", "content": prompt},
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
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
