#!/usr/bin/env python3
"""
L4D2 SourcePawn Code Generator - Web UI

A modern web interface for generating L4D2 SourceMod plugins using the V7 fine-tuned model.

Usage:
    python scripts/inference/web_ui.py
    python scripts/inference/web_ui.py --port 8080
    
    # With Doppler for API key management
    doppler run --project local-mac-work --config dev_personal -- python scripts/inference/web_ui.py
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_read_text

PROJECT_ROOT = Path(__file__).parent.parent.parent

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("FastAPI not installed. Run: pip install fastapi uvicorn")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("OpenAI not installed. Run: pip install openai")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model Configuration
AVAILABLE_MODELS = {
    "v7": {
        "id": "ft:gpt-4o-mini-2024-07-18:highencodelearning:l4d2-sourcemod-v7:CvTBCVPi",
        "name": "V7",
        "description": "96.7% pass rate, 34.7/40 avg score"
    },
    # V8 and V9 will be added automatically when training completes
}

# Check for V8 model ID file
V8_MODEL_FILE = PROJECT_ROOT / "data" / "processed" / "v8_model_id.txt"
if V8_MODEL_FILE.exists():
    v8_id = V8_MODEL_FILE.read_text().strip()
    AVAILABLE_MODELS["v8"] = {
        "id": v8_id,
        "name": "V8",
        "description": "779 examples including gap-filling synthetic data"
    }

# Check for V9 model ID file
V9_MODEL_FILE = PROJECT_ROOT / "data" / "processed" / "v9_model_id.txt"
if V9_MODEL_FILE.exists():
    v9_id = V9_MODEL_FILE.read_text().strip()
    AVAILABLE_MODELS["v9"] = {
        "id": v9_id,
        "name": "V9 (Latest)",
        "description": "671 examples with expanded map_events and special_infected coverage"
    }

# Default to latest available model
if "v9" in AVAILABLE_MODELS:
    MODEL_ID = AVAILABLE_MODELS["v9"]["id"]
    DEFAULT_VERSION = "v9"
elif "v8" in AVAILABLE_MODELS:
    MODEL_ID = AVAILABLE_MODELS["v8"]["id"]
    DEFAULT_VERSION = "v8"
else:
    MODEL_ID = AVAILABLE_MODELS["v7"]["id"]
    DEFAULT_VERSION = "v7"

SYSTEM_PROMPT = """You are an expert SourcePawn and VScript developer for Left 4 Dead 2 SourceMod plugins.
Write clean, well-documented code with proper error handling. Use correct L4D2 APIs and events.

CRITICAL L4D2 API RULES:
- Use GetRandomFloat() and GetRandomInt(), NOT RandomFloat() or RandomInt()
- Use lunge_pounce event for Hunter pounces, NOT pounce
- Use tongue_grab for Smoker, NOT smoker_tongue_grab
- Use player_now_it for bile, NOT boomer_vomit
- Use charger_carry_start for Charger, NOT charger_grab
- Use m_flLaggedMovementValue for speed, NOT m_flSpeed or m_flMaxSpeed"""

# Pricing
PRICING = {
    "input_per_1m": 0.30,
    "output_per_1m": 1.20,
}

# Preset prompts for quick access
PRESET_PROMPTS = [
    {"name": "Tank Announcer", "prompt": "Write a plugin that announces when a Tank spawns and shows its health to all players"},
    {"name": "Speed Boost", "prompt": "Write a plugin that gives survivors a 20% speed boost for 5 seconds when they kill a special infected"},
    {"name": "No Friendly Fire", "prompt": "Write a plugin that prevents friendly fire damage during panic events"},
    {"name": "Auto Heal", "prompt": "Write a plugin that automatically heals survivors when they enter a saferoom"},
    {"name": "Kill Counter", "prompt": "Write a plugin that tracks and displays each player's zombie kill count per round"},
    {"name": "Witch Warning", "prompt": "Write a plugin that plays a warning sound and shows a message when players get close to a Witch"},
    {"name": "Hunter Pounce Tracker", "prompt": "Write a plugin that tracks Hunter pounce damage and announces high-damage pounces"},
    {"name": "Custom Tank HP", "prompt": "Write a plugin that sets Tank health based on the number of alive survivors"},
]


class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 0.1
    max_tokens: int = 2048
    model: Optional[str] = None  # v7, v8, or None for default


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: float = 0.1
    max_tokens: int = 2048
    model: Optional[str] = None  # v7, v8, or None for default


def get_model_id(model_key: Optional[str] = None) -> str:
    """Get model ID from key, defaulting to latest available."""
    if model_key and model_key in AVAILABLE_MODELS:
        return AVAILABLE_MODELS[model_key]["id"]
    return MODEL_ID


# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>L4D2 SourcePawn Code Generator</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <style>
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --accent: #58a6ff;
            --accent-hover: #79b8ff;
            --success: #3fb950;
            --warning: #d29922;
            --error: #f85149;
            --border: #30363d;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            text-align: center;
            padding: 30px 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 30px;
        }
        
        header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(135deg, var(--accent), var(--success));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        header p {
            color: var(--text-secondary);
            font-size: 1.1rem;
        }
        
        .model-badge {
            display: inline-block;
            background: var(--bg-tertiary);
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            color: var(--success);
            margin-top: 10px;
            border: 1px solid var(--border);
        }
        
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }
        
        @media (max-width: 1024px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
        }
        
        .panel {
            background: var(--bg-secondary);
            border-radius: 12px;
            border: 1px solid var(--border);
            overflow: hidden;
        }
        
        .panel-header {
            padding: 15px 20px;
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .panel-header h2 {
            font-size: 1.1rem;
            font-weight: 600;
        }
        
        .panel-body {
            padding: 20px;
        }
        
        .tabs {
            display: flex;
            gap: 5px;
            margin-bottom: 20px;
        }
        
        .tab {
            padding: 10px 20px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 8px;
            cursor: pointer;
            color: var(--text-secondary);
            transition: all 0.2s;
        }
        
        .tab:hover {
            background: var(--bg-primary);
            color: var(--text-primary);
        }
        
        .tab.active {
            background: var(--accent);
            color: white;
            border-color: var(--accent);
        }
        
        textarea {
            width: 100%;
            min-height: 150px;
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 15px;
            color: var(--text-primary);
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 14px;
            resize: vertical;
            transition: border-color 0.2s;
        }
        
        textarea:focus {
            outline: none;
            border-color: var(--accent);
        }
        
        .presets {
            margin-bottom: 20px;
        }
        
        .presets-label {
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-bottom: 10px;
            display: block;
        }
        
        .preset-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        
        .preset-btn {
            padding: 6px 12px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 6px;
            cursor: pointer;
            color: var(--text-secondary);
            font-size: 0.85rem;
            transition: all 0.2s;
        }
        
        .preset-btn:hover {
            background: var(--bg-primary);
            color: var(--accent);
            border-color: var(--accent);
        }
        
        .controls {
            display: flex;
            gap: 15px;
            align-items: center;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        
        .control-group {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .control-group label {
            font-size: 0.9rem;
            color: var(--text-secondary);
        }
        
        input[type="range"] {
            width: 100px;
            accent-color: var(--accent);
        }
        
        .temp-value {
            font-family: monospace;
            color: var(--accent);
            min-width: 30px;
        }
        
        .generate-btn {
            padding: 12px 30px;
            background: linear-gradient(135deg, var(--accent), #2ea043);
            border: none;
            border-radius: 8px;
            color: white;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            margin-left: auto;
        }
        
        .generate-btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(88, 166, 255, 0.3);
        }
        
        .generate-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .output-area {
            min-height: 400px;
            max-height: 600px;
            overflow-y: auto;
        }
        
        .code-block {
            background: var(--bg-primary);
            border-radius: 8px;
            overflow: hidden;
        }
        
        .code-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 15px;
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border);
        }
        
        .code-lang {
            font-size: 0.85rem;
            color: var(--text-secondary);
        }
        
        .copy-btn {
            padding: 5px 12px;
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 5px;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 0.85rem;
            transition: all 0.2s;
        }
        
        .copy-btn:hover {
            background: var(--accent);
            color: white;
            border-color: var(--accent);
        }
        
        .code-content {
            padding: 15px;
            overflow-x: auto;
        }
        
        .code-content pre {
            margin: 0;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 13px;
            line-height: 1.5;
        }
        
        .stats {
            display: flex;
            gap: 20px;
            padding: 15px;
            background: var(--bg-tertiary);
            border-top: 1px solid var(--border);
            font-size: 0.9rem;
        }
        
        .stat {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .stat-label {
            color: var(--text-secondary);
        }
        
        .stat-value {
            color: var(--success);
            font-family: monospace;
        }
        
        .chat-container {
            display: none;
            flex-direction: column;
            height: 500px;
        }
        
        .chat-container.active {
            display: flex;
        }
        
        .generate-container.active {
            display: block;
        }
        
        .generate-container {
            display: none;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 15px;
            background: var(--bg-primary);
            border-radius: 8px;
            margin-bottom: 15px;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 12px 15px;
            border-radius: 8px;
        }
        
        .message.user {
            background: var(--accent);
            color: white;
            margin-left: 50px;
        }
        
        .message.assistant {
            background: var(--bg-tertiary);
            margin-right: 50px;
        }
        
        .message pre {
            background: var(--bg-primary);
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            margin-top: 10px;
        }
        
        .chat-input-area {
            display: flex;
            gap: 10px;
        }
        
        .chat-input {
            flex: 1;
            padding: 12px 15px;
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-primary);
            font-size: 1rem;
        }
        
        .chat-input:focus {
            outline: none;
            border-color: var(--accent);
        }
        
        .send-btn {
            padding: 12px 25px;
            background: var(--accent);
            border: none;
            border-radius: 8px;
            color: white;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .send-btn:hover:not(:disabled) {
            background: var(--accent-hover);
        }
        
        .loading {
            display: flex;
            align-items: center;
            gap: 10px;
            color: var(--text-secondary);
            padding: 20px;
        }
        
        .loading-spinner {
            width: 20px;
            height: 20px;
            border: 2px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .placeholder {
            text-align: center;
            padding: 60px 20px;
            color: var(--text-secondary);
        }
        
        .placeholder-icon {
            font-size: 3rem;
            margin-bottom: 15px;
        }
        
        footer {
            text-align: center;
            padding: 30px;
            margin-top: 40px;
            border-top: 1px solid var(--border);
            color: var(--text-secondary);
            font-size: 0.9rem;
        }
        
        footer a {
            color: var(--accent);
            text-decoration: none;
        }
        
        footer a:hover {
            text-decoration: underline;
        }
        
        .error-message {
            background: rgba(248, 81, 73, 0.1);
            border: 1px solid var(--error);
            color: var(--error);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>L4D2 SourcePawn Generator</h1>
            <p>AI-powered code generation for Left 4 Dead 2 SourceMod plugins</p>
            <span class="model-badge" id="model-badge">Model: Loading...</span>
        </header>
        
        <div class="main-grid">
            <!-- Input Panel -->
            <div class="panel">
                <div class="panel-header">
                    <h2>Input</h2>
                    <div class="tabs">
                        <button class="tab active" data-tab="generate">Generate</button>
                        <button class="tab" data-tab="chat">Chat</button>
                    </div>
                </div>
                <div class="panel-body">
                    <!-- Generate Mode -->
                    <div class="generate-container active" id="generate-mode">
                        <div class="presets">
                            <span class="presets-label">Quick Presets:</span>
                            <div class="preset-buttons" id="preset-buttons"></div>
                        </div>
                        
                        <textarea id="prompt-input" placeholder="Describe the L4D2 plugin you want to create...

Example: Write a plugin that gives survivors a speed boost when they kill a special infected"></textarea>
                        
                        <div class="controls">
                            <div class="control-group">
                                <label>Temperature:</label>
                                <input type="range" id="temperature" min="0" max="1" step="0.1" value="0.1">
                                <span class="temp-value" id="temp-display">0.1</span>
                            </div>
                            <button class="generate-btn" id="generate-btn" onclick="generateCode()">
                                Generate Code
                            </button>
                        </div>
                    </div>
                    
                    <!-- Chat Mode -->
                    <div class="chat-container" id="chat-mode">
                        <div class="chat-messages" id="chat-messages">
                            <div class="message assistant">
                                Hello! I'm your L4D2 SourcePawn assistant. Ask me to write plugins, explain code, or help debug issues. What would you like to create?
                            </div>
                        </div>
                        <div class="chat-input-area">
                            <input type="text" class="chat-input" id="chat-input" 
                                   placeholder="Ask me to write a plugin..." 
                                   onkeypress="if(event.key==='Enter') sendChat()">
                            <button class="send-btn" onclick="sendChat()">Send</button>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Output Panel -->
            <div class="panel">
                <div class="panel-header">
                    <h2>Generated Code</h2>
                </div>
                <div class="panel-body">
                    <div class="output-area" id="output-area">
                        <div class="placeholder">
                            <div class="placeholder-icon">&#128187;</div>
                            <p>Your generated SourcePawn code will appear here</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <footer>
            <p>
                Powered by <a href="https://openai.com" target="_blank">OpenAI</a> fine-tuned model |
                <span id="footer-model-info">Loading model info...</span> |
                <a href="/docs" target="_blank">API Docs</a>
            </p>
        </footer>
    </div>
    
    <script>
        // Preset prompts
        const presets = PRESET_DATA;
        
        // State
        let chatMessages = [
            {role: "system", content: "You are an expert SourcePawn developer for L4D2."}
        ];
        let totalCost = 0;
        
        // State for models
        let availableModels = {};
        let currentModel = null;

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            initPresets();
            initTabs();
            initTemperature();
            loadModels();
        });

        async function loadModels() {
            try {
                const response = await fetch('/api/models');
                const data = await response.json();
                availableModels = data.models;
                currentModel = data.default;

                // Update model badge
                const model = availableModels[currentModel];
                document.getElementById('model-badge').textContent =
                    `Model: ${model.name} Fine-tuned GPT-4o-mini`;

                // Update footer
                document.getElementById('footer-model-info').textContent =
                    `${model.name}: ${model.description}`;

            } catch (err) {
                console.error('Failed to load models:', err);
                document.getElementById('model-badge').textContent = 'Model: Error loading';
            }
        }
        
        function initPresets() {
            const container = document.getElementById('preset-buttons');
            presets.forEach(preset => {
                const btn = document.createElement('button');
                btn.className = 'preset-btn';
                btn.textContent = preset.name;
                btn.onclick = () => {
                    document.getElementById('prompt-input').value = preset.prompt;
                };
                container.appendChild(btn);
            });
        }
        
        function initTabs() {
            document.querySelectorAll('.tab').forEach(tab => {
                tab.onclick = () => {
                    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                    tab.classList.add('active');
                    
                    const mode = tab.dataset.tab;
                    document.getElementById('generate-mode').classList.toggle('active', mode === 'generate');
                    document.getElementById('chat-mode').classList.toggle('active', mode === 'chat');
                };
            });
        }
        
        function initTemperature() {
            const slider = document.getElementById('temperature');
            const display = document.getElementById('temp-display');
            slider.oninput = () => {
                display.textContent = slider.value;
            };
        }
        
        async function generateCode() {
            const prompt = document.getElementById('prompt-input').value.trim();
            if (!prompt) return;
            
            const btn = document.getElementById('generate-btn');
            const output = document.getElementById('output-area');
            
            btn.disabled = true;
            btn.textContent = 'Generating...';
            
            output.innerHTML = `
                <div class="loading">
                    <div class="loading-spinner"></div>
                    <span>Generating your L4D2 plugin...</span>
                </div>
            `;
            
            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        prompt: prompt,
                        temperature: parseFloat(document.getElementById('temperature').value),
                        max_tokens: 2048
                    })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    output.innerHTML = `<div class="error-message">${data.error}</div>`;
                    return;
                }
                
                displayCode(data.code, data.tokens, data.cost, data.elapsed);
                
            } catch (err) {
                output.innerHTML = `<div class="error-message">Error: ${err.message}</div>`;
            } finally {
                btn.disabled = false;
                btn.textContent = 'Generate Code';
            }
        }
        
        function displayCode(code, tokens, cost, elapsed) {
            const output = document.getElementById('output-area');
            
            // Escape HTML
            const escapedCode = code
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');
            
            output.innerHTML = `
                <div class="code-block">
                    <div class="code-header">
                        <span class="code-lang">SourcePawn</span>
                        <button class="copy-btn" onclick="copyCode()">Copy Code</button>
                    </div>
                    <div class="code-content">
                        <pre><code class="language-cpp" id="code-output">${escapedCode}</code></pre>
                    </div>
                    <div class="stats">
                        <div class="stat">
                            <span class="stat-label">Tokens:</span>
                            <span class="stat-value">${tokens}</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Cost:</span>
                            <span class="stat-value">$${cost.toFixed(4)}</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Time:</span>
                            <span class="stat-value">${elapsed.toFixed(2)}s</span>
                        </div>
                    </div>
                </div>
            `;
            
            // Apply syntax highlighting
            hljs.highlightElement(document.getElementById('code-output'));
        }
        
        function copyCode() {
            const code = document.getElementById('code-output').textContent;
            navigator.clipboard.writeText(code).then(() => {
                const btn = document.querySelector('.copy-btn');
                btn.textContent = 'Copied!';
                setTimeout(() => btn.textContent = 'Copy Code', 2000);
            });
        }
        
        async function sendChat() {
            const input = document.getElementById('chat-input');
            const message = input.value.trim();
            if (!message) return;
            
            input.value = '';
            
            // Add user message to UI
            const messagesDiv = document.getElementById('chat-messages');
            messagesDiv.innerHTML += `<div class="message user">${escapeHtml(message)}</div>`;
            
            // Add to chat history
            chatMessages.push({role: "user", content: message});
            
            // Show loading
            messagesDiv.innerHTML += `
                <div class="message assistant loading-msg">
                    <div class="loading">
                        <div class="loading-spinner"></div>
                        <span>Thinking...</span>
                    </div>
                </div>
            `;
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        messages: chatMessages.slice(1), // Exclude system message
                        temperature: 0.1,
                        max_tokens: 2048
                    })
                });
                
                const data = await response.json();
                
                // Remove loading message
                document.querySelector('.loading-msg')?.remove();
                
                if (data.error) {
                    messagesDiv.innerHTML += `<div class="message assistant error-message">${data.error}</div>`;
                    return;
                }
                
                // Add assistant response
                chatMessages.push({role: "assistant", content: data.response});
                
                // Format code blocks
                const formatted = formatMessage(data.response);
                messagesDiv.innerHTML += `<div class="message assistant">${formatted}</div>`;
                
                // Update total cost
                totalCost += data.cost;
                
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
                
            } catch (err) {
                document.querySelector('.loading-msg')?.remove();
                messagesDiv.innerHTML += `<div class="message assistant error-message">Error: ${err.message}</div>`;
            }
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        function formatMessage(text) {
            // Convert code blocks
            return text.replace(/```(\\w*)\\n([\\s\\S]*?)```/g, (match, lang, code) => {
                const escaped = escapeHtml(code.trim());
                return `<pre><code class="language-${lang || 'cpp'}">${escaped}</code></pre>`;
            }).replace(/\\n/g, '<br>');
        }
    </script>
</body>
</html>
"""


class WebUI:
    """Web UI for L4D2 Code Generator"""
    
    def __init__(self):
        self.client = None
        self._init_client()
        
        self.app = FastAPI(
            title="L4D2 SourcePawn Generator",
            description="Web UI for generating L4D2 SourceMod plugins",
            version="1.0.0"
        )
        
        # Add CORS - restricted to localhost for security
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:8000", "http://localhost:3000", "http://127.0.0.1:8000", "http://127.0.0.1:3000"],
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )
        
        self._setup_routes()
    
    def _init_client(self):
        """Initialize OpenAI client"""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            # Try loading from .env using secure file read
            env_path = PROJECT_ROOT / ".env"
            if env_path.exists():
                try:
                    env_content = safe_read_text(env_path, PROJECT_ROOT)
                    for line in env_content.splitlines():
                        if line.startswith("OPENAI_API_KEY="):
                            api_key = line.split("=", 1)[1].strip()
                            break
                except (ValueError, FileNotFoundError):
                    logger.warning("Could not read .env file")

        if not api_key:
            logger.warning("OPENAI_API_KEY not set - API calls will fail")
            logger.info("Set OPENAI_API_KEY environment variable or add to .env file")
        else:
            self.client = OpenAI(api_key=api_key)
            logger.info(f"OpenAI client initialized, using model: {MODEL_ID}")
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * PRICING["input_per_1m"] / 1_000_000 +
                output_tokens * PRICING["output_per_1m"] / 1_000_000)
    
    def _setup_routes(self):
        
        @self.app.get("/", response_class=HTMLResponse)
        async def index():
            # Inject preset data into HTML
            html = HTML_TEMPLATE.replace(
                "PRESET_DATA",
                json.dumps(PRESET_PROMPTS)
            )
            return html
        
        @self.app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "model": MODEL_ID,
                "client_ready": self.client is not None
            }

        @self.app.get("/api/models")
        async def list_models():
            """List available models."""
            return {
                "models": AVAILABLE_MODELS,
                "default": DEFAULT_VERSION
            }

        @self.app.post("/api/generate")
        async def generate(request: GenerateRequest):
            if not self.client:
                raise HTTPException(status_code=500, detail="OpenAI client not initialized")
            
            try:
                start_time = time.time()
                
                response = self.client.chat.completions.create(
                    model=get_model_id(request.model),
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": request.prompt}
                    ],
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                )
                
                elapsed = time.time() - start_time
                usage = response.usage
                cost = self._calculate_cost(usage.prompt_tokens, usage.completion_tokens)
                
                return {
                    "code": response.choices[0].message.content,
                    "tokens": usage.total_tokens,
                    "cost": cost,
                    "elapsed": elapsed
                }
                
            except Exception as e:
                logger.error(f"Generation error: {e}")
                return {"error": str(e)}
        
        @self.app.post("/api/chat")
        async def chat(request: ChatRequest):
            if not self.client:
                raise HTTPException(status_code=500, detail="OpenAI client not initialized")
            
            try:
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                messages.extend([{"role": m.role, "content": m.content} for m in request.messages])
                
                response = self.client.chat.completions.create(
                    model=get_model_id(request.model),
                    messages=messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                )
                
                usage = response.usage
                cost = self._calculate_cost(usage.prompt_tokens, usage.completion_tokens)
                
                return {
                    "response": response.choices[0].message.content,
                    "tokens": usage.total_tokens,
                    "cost": cost
                }
                
            except Exception as e:
                logger.error(f"Chat error: {e}")
                return {"error": str(e)}
    
    def run(self, host: str = "127.0.0.1", port: int = 8000):
        logger.info(f"Starting L4D2 Web UI on http://{host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


def main():
    parser = argparse.ArgumentParser(description="L4D2 SourcePawn Generator Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()
    
    ui = WebUI()
    ui.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
