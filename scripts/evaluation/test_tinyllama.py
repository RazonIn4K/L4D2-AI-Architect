#!/usr/bin/env python3
"""
Simple script to test TinyLlama LoRA model on Vultr instance.
"""

import json
from pathlib import Path

# Load test cases
test_cases = []
with open("data/eval_test_cases.jsonl", "r") as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            test_cases.append(data["item"])

print("# TinyLlama LoRA Evaluation Test Cases")
print("# Copy these commands to run on Vultr instance:")
print()
print("# SSH into Vultr:")
print("ssh root@108.61.127.209")
print()
print("# Activate environment:")
print("cd /root/L4D2-AI-Architect")
print("source venv/bin/activate")
print()
print("# Run inference script for each test case:")
print()

for i, case in enumerate(test_cases, 1):
    prompt = case["input"]
    escaped_prompt = prompt.replace('"', '\\"')
    print(f"# Test {i}: {prompt[:50]}...")
    print(f'python scripts/inference/test_lora.py --adapter model_adapters/l4d2-lora --base tiny --prompt "{escaped_prompt}"')
    print()

print("# Or run all tests at once:")
print("python scripts/evaluation/run_tinyllama_eval.py")
