#!/usr/bin/env python3
"""
Test inference with trained LoRA adapter

Usage:
    python test_lora.py                                                     # Use default adapter
    python test_lora.py --adapter model_adapters/l4d2-mistral-v10plus-lora/final  # Specify adapter
    python test_lora.py --base mistral                                      # Use Mistral base model
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="Test LoRA inference")
    parser.add_argument("--adapter", type=str, default="model_adapters/l4d2-tiny-v15-lora256",
                       help="Path to LoRA adapter (best: l4d2-tiny-v15-lora256)")
    parser.add_argument("--base", choices=["tiny", "mistral"], default="tiny",
                       help="Base model: tiny or mistral")
    args = parser.parse_args()

    print("=" * 50)
    print("L4D2 LoRA Inference Test")
    print("=" * 50)

    # Select base model
    if args.base == "tiny":
        base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    else:
        base_model = "mistralai/Mistral-7B-Instruct-v0.3"

    adapter_path = args.adapter

    # Detect device
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA GPU")
    else:
        device = "cpu"
        print("Using CPU")

    # Load base model
    print(f"\nLoading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    if device == "mps":
        model = model.to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Load LoRA adapter
    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    print("\nModel loaded successfully!")
    print("-" * 50)

    # Test prompts
    test_prompts = [
        "Write a SourcePawn function to heal all survivors to full health",
        "Create a timer that spawns zombies every 30 seconds",
        "Write code to detect when a player picks up a health kit",
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[Test {i}] {prompt}")
        print("-" * 40)

        # Format as chat
        messages = [
            {"role": "system", "content": "You are an expert SourcePawn developer for Left 4 Dead 2."},
            {"role": "user", "content": prompt}
        ]

        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_text, return_tensors="pt")

        if device == "mps":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the assistant response
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()

        print(f"Response:\n{response[:500]}")

    print("\n" + "=" * 50)
    print("Inference test complete!")

if __name__ == "__main__":
    main()
