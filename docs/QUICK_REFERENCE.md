# L4D2-AI-Architect Quick Reference Card

## Instant Start (Copy & Paste)

### Web UI
```bash
cd /Users/davidortiz/left4dead-model/L4D2-AI-Architect
export OPENAI_API_KEY="sk-your-key"
./start_web_ui.sh
# Open http://localhost:8000
```

### CLI Tool
```bash
cd /Users/davidortiz/left4dead-model/L4D2-AI-Architect
export OPENAI_API_KEY="sk-your-key"

# Generate code
python scripts/inference/l4d2_codegen.py generate "Write a tank announcer plugin"

# Interactive chat
python scripts/inference/l4d2_codegen.py chat
```

### API Server
```bash
cd /Users/davidortiz/left4dead-model/L4D2-AI-Architect
export OPENAI_API_KEY="sk-your-key"
python scripts/inference/copilot_server_openai.py --port 8000

# Then use:
curl -X POST http://localhost:8000/v1/complete \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a tank health announcer"}'
```

---

## All Available Tools

| Tool | Command | Purpose |
|------|---------|---------|
| **Web UI** | `./start_web_ui.sh` | Browser-based code generation |
| **CLI Generate** | `python scripts/inference/l4d2_codegen.py generate "prompt"` | Single code generation |
| **CLI Chat** | `python scripts/inference/l4d2_codegen.py chat` | Interactive development |
| **CLI Batch** | `python scripts/inference/l4d2_codegen.py batch prompts.txt` | Bulk generation |
| **CLI Info** | `python scripts/inference/l4d2_codegen.py info` | Model info & pricing |
| **API Server** | `python scripts/inference/copilot_server_openai.py` | REST API endpoints |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/health` | GET | Health check |
| `/v1/complete` | POST | Code completion |
| `/v1/chat/completions` | POST | Chat (OpenAI-compatible) |
| `/v1/generate-plugin` | POST | Full plugin generation |

---

## Model Info

```
Model ID: ft:gpt-4o-mini-2024-07-18:highencodelearning:l4d2-sourcemod-v7:CvTBCVPi
Pass Rate: 80%
API Correctness: 100%
Cost per generation: ~$0.002-0.004
Temperature (recommended): 0.1
```

---

## Common Plugin Prompts

```
# Tank Mechanics
"Write a plugin that announces when a Tank spawns and shows its health"
"Create a plugin that sets Tank health based on number of survivors"

# Survivor Abilities  
"Write a plugin that gives speed boost when killing special infected"
"Create a plugin that auto-heals survivors in saferooms"

# Game Events
"Write a plugin that tracks zombie kill counts per round"
"Create a plugin that prevents friendly fire during panic events"

# Special Infected
"Write a plugin that tracks Hunter pounce damage"
"Create a plugin that plays warning sound near Witches"
```

---

## Correct L4D2 APIs (Model Enforced)

| Use This | NOT This |
|----------|----------|
| `GetRandomFloat()` | `RandomFloat()` |
| `GetRandomInt()` | `RandomInt()` |
| `lunge_pounce` | `pounce` |
| `tongue_grab` | `smoker_tongue_grab` |
| `player_now_it` | `boomer_vomit` |
| `charger_carry_start` | `charger_grab` |
| `m_flLaggedMovementValue` | `m_flSpeed` |

---

## Cost Estimates

| Action | Cost |
|--------|------|
| Single generation | ~$0.002-0.004 |
| 10 generations | ~$0.02-0.04 |
| 100 generations | ~$0.20-0.40 |
| Batch API (50% off) | Half of above |

---

## Key Directories

```
L4D2-AI-Architect/
├── scripts/inference/     # All inference tools
│   ├── web_ui.py         # Web interface
│   ├── l4d2_codegen.py   # CLI tool
│   └── copilot_server_openai.py  # API server
├── docs/
│   ├── ARCHITECTURE.md   # Full architecture guide
│   └── QUICK_REFERENCE.md # This file
└── start_web_ui.sh       # Easy startup script
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "OPENAI_API_KEY not set" | `export OPENAI_API_KEY="sk-..."` |
| "openai not found" | `pip install openai` |
| "fastapi not found" | `pip install fastapi uvicorn` |
| Empty/wrong output | Use temperature 0.1 |

---

*Keep this card handy for quick reference!*
