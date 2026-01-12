#!/usr/bin/env python3
"""
Generate test plugins using the trained LoRA model for validation.
"""

import argparse
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Add parent to path for security utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_write_text

PROJECT_ROOT = Path(__file__).parent.parent.parent

TEST_PROMPTS = [
    {
        "prompt": "Write a complete L4D2 SourcePawn plugin that gives survivors a 30% speed boost for 5 seconds after killing a special infected. Use m_flLaggedMovementValue property.",
        "filename": "TestSpeedBoost.sp"
    },
    {
        "prompt": "Write a complete L4D2 SourcePawn plugin that tracks zombie kills per player using the infected_death event and displays kills on round end.",
        "filename": "TestKillTracker.sp"
    },
    {
        "prompt": "Write a complete L4D2 SourcePawn plugin that heals survivors by 20 health when they enter a saferoom using player_entered_checkpoint event.",
        "filename": "TestSaferoomHeal.sp"
    },
]

def generate_plugin(model, tokenizer, prompt: str, device: str, max_tokens: int = 512) -> str:
    """Generate a plugin from a prompt."""
    messages = [
        {"role": "system", "content": "You are an expert SourcePawn developer for Left 4 Dead 2. Generate complete, working plugin code with proper includes, plugin info, and all required functions."},
        {"role": "user", "content": prompt}
    ]

    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_text, return_tensors="pt")

    if device == "mps":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.5,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract assistant response
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()

    return response

def main():
    parser = argparse.ArgumentParser(description="Generate test plugins")
    parser.add_argument("--adapter", type=str, default="model_adapters/l4d2-mistral-v10plus-lora/final")
    parser.add_argument("--output-dir", type=str, default="data/generated_test")
    parser.add_argument("--max-tokens", type=int, default=800)
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("L4D2 Test Plugin Generator")
    print("=" * 60)

    # Detect device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Load model
    base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Loading base model: {base_model}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    )

    if device == "mps":
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    print(f"\nGenerating {len(TEST_PROMPTS)} test plugins...")
    print("-" * 60)

    for i, test in enumerate(TEST_PROMPTS, 1):
        print(f"\n[{i}/{len(TEST_PROMPTS)}] Generating: {test['filename']}")

        code = generate_plugin(model, tokenizer, test["prompt"], device, args.max_tokens)

        # Save to file using secure write
        output_path = output_dir / test["filename"]
        safe_write_text(output_path, code, PROJECT_ROOT)
        print(f"  Saved to: {output_path}")
        print(f"  Length: {len(code)} chars, {len(code.splitlines())} lines")

    print("\n" + "=" * 60)
    print(f"Generated {len(TEST_PROMPTS)} plugins to {output_dir}")
    print("Run validation: python scripts/evaluation/validate_generated_code.py --input data/generated_test")

if __name__ == "__main__":
    main()
