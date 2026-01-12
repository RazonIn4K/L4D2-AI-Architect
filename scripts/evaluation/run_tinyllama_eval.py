#!/usr/bin/env python3
"""
Run TinyLlama LoRA evaluation on Vultr instance.
Saves results for comparison with OpenAI fine-tuned model.
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_inference(prompt: str) -> str:
    """Run TinyLlama LoRA inference on Vultr."""
    cmd = [
        "python", "scripts/inference/test_lora.py",
        "--adapter", "model_adapters/l4d2-mistral-v10plus-lora/final",
        "--base", "mistral",
        "--prompt", prompt
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            # Extract the generated response
            output = result.stdout
            # Remove the prompt if echoed
            if prompt in output:
                response = output.split(prompt)[-1].strip()
            else:
                response = output.strip()
            return response
        else:
            return f"ERROR: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "ERROR: Timeout"
    except Exception as e:
        return f"ERROR: {str(e)}"

def score_response(prompt: str, response: str) -> dict:
    """Simple scoring for TinyLlama responses."""
    score = {
        'has_code': 0,
        'has_sourcepawn': 0,
        'is_relevant': 0,
        'is_complete': 0,
        'total': 0
    }
    
    # Check for SourcePawn elements
    if '#include' in response:
        score['has_code'] += 1
    if 'public ' in response or 'stock ' in response:
        score['has_sourcepawn'] += 1
        
    # Check relevance
    prompt_lower = prompt.lower()
    response_lower = response.lower()
    
    keywords = {
        'tank': ['tank', 'boss', 'health'],
        'heal': ['health', 'heal', 'revive'],
        'spawn': ['spawn', 'create', 'entity'],
        'damage': ['damage', 'hurt', 'takedamage'],
        'timer': ['timer', 'createtimer'],
        'command': ['command', 'regcmd'],
        'teleport': ['teleport', 'position'],
        'leaderboard': ['score', 'track', 'stats']
    }
    
    for key, words in keywords.items():
        if key in prompt_lower and any(w in response_lower for w in words):
            score['is_relevant'] += 1
            break
            
    # Check completeness
    if '{' in response and '}' in response:
        score['is_complete'] += 1
    if len(response.split('\n')) > 10:
        score['is_complete'] += 1
        
    score['total'] = sum(score.values())
    return score

def main():
    print("=" * 60)
    print("TinyLlama LoRA Evaluation")
    print("=" * 60)
    
    # Load test cases
    test_cases = []
    with open("data/eval_test_cases.jsonl", "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                test_cases.append(data["item"])
    
    print(f"\nLoaded {len(test_cases)} test cases")
    print("\nRunning evaluation...\n")
    
    results = []
    total_score = 0
    
    for i, case in enumerate(test_cases, 1):
        prompt = case["input"]
        print(f"[{i}/{len(test_cases)}] {prompt[:50]}...")
        
        response = run_inference(prompt)
        score = score_response(prompt, response)
        
        result = {
            'test_case': prompt,
            'response': response,
            'score': score
        }
        results.append(result)
        total_score += score['total']
        
        print(f"  Score: {score['total']}/20")
        print(f"  Response preview: {response[:100]}...")
        print()
    
    # Calculate statistics
    avg_score = total_score / len(test_cases)
    max_score = len(test_cases) * 20
    
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total score: {total_score}/{max_score}")
    print(f"Average score: {avg_score:.1f}/20 ({avg_score*5:.1f}%)")
    
    # Count good responses (score >= 10)
    good_responses = sum(1 for r in results if r['score']['total'] >= 10)
    print(f"Good responses (≥10): {good_responses}/{len(test_cases)} ({good_responses/len(test_cases)*100:.0f}%)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"data/tinyllama_eval_results_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump({
            'timestamp': timestamp,
            'model': 'TinyLlama-1.1B + LoRA',
            'total_score': total_score,
            'max_score': max_score,
            'average_score': avg_score,
            'good_responses': good_responses,
            'total_tests': len(test_cases),
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()
