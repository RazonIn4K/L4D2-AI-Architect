#!/usr/bin/env python3
"""
Evaluate Local LoRA Models vs OpenAI Models
Uses OpenAI API for comparative evaluation with free credits

Usage:
    # Compare best LoRA against GPT-4
    python scripts/evaluation/run_lora_evaluation.py --lora-adapter model_adapters/l4d2-tiny-v15-lora256

    # Compare all LoRA models
    python scripts/evaluation/run_lora_evaluation.py --compare-all

    # Use specific OpenAI model
    python scripts/evaluation/run_lora_evaluation.py --openai-model gpt-4o-mini
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent.parent

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)


# L4D2 Test Prompts
TEST_PROMPTS = [
    "Write a SourcePawn function to heal all survivors to full health",
    "Create a plugin that announces when a Tank spawns with its health",
    "Write code to detect when a player picks up a health kit",
    "Create a timer that spawns zombies every 30 seconds",
    "Write a function to give all survivors infinite ammo",
    "Create a plugin that tracks Hunter pounce damage",
    "Write code to teleport all survivors to the safe room",
    "Create a function to spawn a Witch at a random location",
    "Write a plugin that announces top damage dealers when Tank dies",
    "Create code to give survivors a speed boost when they kill special infected"
]


def load_local_lora(adapter_path: str):
    """Load local LoRA adapter."""
    print(f"Loading LoRA adapter: {adapter_path}")
    
    base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Detect device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    
    if device == "mps":
        model = model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    return model, tokenizer, device


def generate_local(model, tokenizer, device, prompt: str) -> str:
    """Generate response from local LoRA model."""
    messages = [
        {"role": "system", "content": "You are an expert SourcePawn developer for Left 4 Dead 2."},
        {"role": "user", "content": prompt}
    ]
    
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_text, return_tensors="pt")
    
    if device == "mps":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    
    return response


def generate_openai(client: OpenAI, model: str, prompt: str) -> str:
    """Generate response from OpenAI model."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert SourcePawn developer for Left 4 Dead 2."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.7
    )
    return response.choices[0].message.content


def evaluate_response(response: str) -> Dict[str, Any]:
    """Evaluate quality of generated code."""
    metrics = {
        "has_code": "```" in response or "public" in response or "Action" in response,
        "has_sourcepawn": any(kw in response for kw in ["public", "Action", "Handle", "ConVar"]),
        "has_l4d2_api": any(api in response for api in ["GetClientTeam", "GetEntProp", "PrintToChatAll", "CreateTimer"]),
        "line_count": len([l for l in response.split('\n') if l.strip()]),
        "char_count": len(response)
    }
    
    # Quality score (0-100)
    score = 0
    if metrics["has_code"]: score += 30
    if metrics["has_sourcepawn"]: score += 30
    if metrics["has_l4d2_api"]: score += 40
    
    metrics["quality_score"] = score
    return metrics


def run_evaluation(lora_adapter: str, openai_model: str = "gpt-4o-mini"):
    """Run comparative evaluation."""
    print("=" * 60)
    print("L4D2 LoRA vs OpenAI Evaluation")
    print("=" * 60)
    
    # Initialize OpenAI
    client = OpenAI()
    
    # Load local LoRA
    model, tokenizer, device = load_local_lora(lora_adapter)
    
    results = []
    
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n[{i}/{len(TEST_PROMPTS)}] Testing: {prompt[:60]}...")
        
        # Generate from local LoRA
        print("  → Local LoRA generating...")
        lora_response = generate_local(model, tokenizer, device, prompt)
        lora_metrics = evaluate_response(lora_response)
        
        # Generate from OpenAI
        print("  → OpenAI generating...")
        openai_response = generate_openai(client, openai_model, prompt)
        openai_metrics = evaluate_response(openai_response)
        
        result = {
            "prompt": prompt,
            "lora": {
                "response": lora_response,
                "metrics": lora_metrics
            },
            "openai": {
                "response": openai_response,
                "metrics": openai_metrics
            }
        }
        results.append(result)
        
        print(f"  ✓ LoRA Score: {lora_metrics['quality_score']}/100")
        print(f"  ✓ OpenAI Score: {openai_metrics['quality_score']}/100")
    
    # Calculate summary
    lora_avg = sum(r["lora"]["metrics"]["quality_score"] for r in results) / len(results)
    openai_avg = sum(r["openai"]["metrics"]["quality_score"] for r in results) / len(results)
    
    summary = {
        "lora_adapter": lora_adapter,
        "openai_model": openai_model,
        "test_count": len(TEST_PROMPTS),
        "lora_avg_score": lora_avg,
        "openai_avg_score": openai_avg,
        "winner": "LoRA" if lora_avg > openai_avg else "OpenAI",
        "timestamp": datetime.now().isoformat()
    }
    
    # Save results
    output_dir = PROJECT_ROOT / "data" / "evaluation_results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"lora_vs_openai_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "summary": summary,
            "results": results
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"LoRA Average Score:   {lora_avg:.1f}/100")
    print(f"OpenAI Average Score: {openai_avg:.1f}/100")
    print(f"Winner: {summary['winner']}")
    print(f"\nResults saved to: {output_file}")
    
    return summary, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA vs OpenAI models")
    parser.add_argument("--lora-adapter", type=str, default="model_adapters/l4d2-tiny-v15-lora256",
                       help="Path to LoRA adapter")
    parser.add_argument("--openai-model", type=str, default="gpt-4o-mini",
                       help="OpenAI model to compare against")
    parser.add_argument("--compare-all", action="store_true",
                       help="Compare all available LoRA models")
    
    args = parser.parse_args()
    
    if args.compare_all:
        lora_models = [
            "model_adapters/l4d2-tiny-v15-lora256",
            "model_adapters/l4d2-tiny-v15-lora128",
            "model_adapters/l4d2-tiny-v15-lora64-long",
            "model_adapters/l4d2-tiny-v15-lora"
        ]
        
        all_summaries = []
        for lora_adapter in lora_models:
            if Path(lora_adapter).exists():
                print(f"\n{'='*60}")
                print(f"Evaluating: {lora_adapter}")
                print(f"{'='*60}")
                summary, _ = run_evaluation(lora_adapter, args.openai_model)
                all_summaries.append(summary)
        
        # Print comparison
        print("\n" + "=" * 60)
        print("ALL MODELS COMPARISON")
        print("=" * 60)
        for s in all_summaries:
            adapter_name = Path(s["lora_adapter"]).name
            print(f"{adapter_name:30} {s['lora_avg_score']:>6.1f}/100")
    else:
        run_evaluation(args.lora_adapter, args.openai_model)


if __name__ == "__main__":
    main()
