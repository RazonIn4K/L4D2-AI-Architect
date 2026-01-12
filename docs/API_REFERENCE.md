# L4D2-AI-Architect API Reference

This document provides comprehensive API documentation for all major components of the L4D2-AI-Architect project.

## Table of Contents

1. [Training Scripts API](#training-scripts-api)
2. [Inference API](#inference-api)
3. [RL Training API](#rl-training-api)
4. [Director API](#director-api)
5. [Evaluation API](#evaluation-api)
6. [Utility Functions](#utility-functions)

---

## Training Scripts API

### train_unsloth.py

Fine-tunes LLMs using Unsloth with QLoRA for L4D2 code generation.

#### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | str | `configs/unsloth_config.yaml` | Path to training configuration file |
| `--test-only` | str | None | Path to adapter for testing only (skip training) |
| `--resume` | str | None | Resume training from checkpoint |

#### Usage Examples

```bash
# Standard training
python scripts/training/train_unsloth.py --config configs/unsloth_config.yaml

# Resume from checkpoint
python scripts/training/train_unsloth.py --resume model_adapters/l4d2-code-lora/checkpoint-200

# Test a trained model
python scripts/training/train_unsloth.py --test-only model_adapters/l4d2-mistral-v10plus-lora/final
```

#### Configuration (unsloth_config.yaml)

```yaml
model:
  name: "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"  # Base model
  max_seq_length: 2048                               # Context length
  dtype: null                                        # Auto-detect (bf16/fp16)
  load_in_4bit: true                                 # Enable 4-bit quantization

lora:
  r: 32                           # LoRA rank
  lora_alpha: 64                  # Alpha scaling
  lora_dropout: 0                 # Dropout (0 for Unsloth optimization)
  target_modules:                 # Modules to apply LoRA
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  use_gradient_checkpointing: "unsloth"  # Memory optimization

training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  warmup_steps: 10
  optim: "adamw_8bit"             # Memory-efficient optimizer
  bf16: true                      # BFloat16 precision

data:
  train_file: "combined_train.jsonl"
  val_file: "combined_val.jsonl"

output:
  dir: "l4d2-mistral-v10plus-lora"
  push_to_hub: false
```

---

### prepare_dataset.py

Prepares training data from scraped sources into ChatML format.

#### Main Functions

##### `prepare_sourcepawn_data()`
Processes SourcePawn plugin files into training examples.

```python
from scripts.training.prepare_dataset import prepare_sourcepawn_data

# Prepare SourcePawn training data
stats = prepare_sourcepawn_data(
    input_dir="data/raw/github_plugins",
    output_file="data/processed/sourcepawn_train.jsonl",
    min_lines=20,           # Minimum code lines
    max_lines=500,          # Maximum code lines
    quality_filter=True     # Enable quality filtering
)
print(f"Processed {stats['total']} files, kept {stats['kept']}")
```

##### `prepare_vscript_data()`
Processes VScript examples from Valve Developer Wiki.

```python
from scripts.training.prepare_dataset import prepare_vscript_data

stats = prepare_vscript_data(
    input_dir="data/raw/valve_wiki",
    output_file="data/processed/vscript_train.jsonl"
)
```

##### `combine_datasets()`
Merges multiple dataset files with optional shuffling.

```python
from scripts.training.prepare_dataset import combine_datasets

combine_datasets(
    input_files=[
        "data/processed/sourcepawn_train.jsonl",
        "data/processed/vscript_train.jsonl"
    ],
    output_file="data/processed/combined_train.jsonl",
    shuffle=True,
    seed=42
)
```

#### Data Format

Training data uses ChatML format:

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert SourcePawn developer..."},
    {"role": "user", "content": "Write a function to heal all survivors"},
    {"role": "assistant", "content": "public void HealAllSurvivors() {...}"}
  ]
}
```

---

### export_gguf_cpu.py

Exports trained LoRA adapters to GGUF format for CPU inference via Ollama.

#### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--adapter` | str | Required | Path to LoRA adapter directory |
| `--output` | str | `exports/` | Output directory for GGUF files |
| `--quantization` | str | `q4_k_m` | Quantization type |
| `--model-name` | str | Auto | Name for the exported model |

#### Quantization Options

| Type | Size | Quality | Use Case |
|------|------|---------|----------|
| `q4_k_m` | ~4GB | Good | Balanced (recommended) |
| `q5_k_m` | ~5GB | Better | Higher quality |
| `q8_0` | ~8GB | Best | Maximum quality |
| `q3_k_m` | ~3GB | Acceptable | Memory constrained |

#### Usage Examples

```bash
# Export with default settings
python scripts/training/export_gguf_cpu.py \
    --adapter model_adapters/l4d2-mistral-v10plus-lora/final

# Custom quantization and output
python scripts/training/export_gguf_cpu.py \
    --adapter model_adapters/l4d2-mistral-v10plus-lora/final \
    --output exports/l4d2-v10plus/gguf \
    --quantization q5_k_m \
    --model-name l4d2-code-v10plus

# Install to Ollama after export
ollama create l4d2-code-v10plus -f exports/l4d2-v10plus/gguf/Modelfile
```

---

## Inference API

### copilot_server.py

FastAPI server providing code completion endpoints.

#### Endpoints

##### POST `/v1/complete`
Generate code completion from a prompt.

**Request:**
```json
{
  "prompt": "Write a SourcePawn function to heal all survivors",
  "max_tokens": 512,
  "temperature": 0.3,
  "stop": ["```", "\n\n\n"]
}
```

**Response:**
```json
{
  "completion": "public void HealAllSurvivors() {\n    for (int i = 1; i <= MaxClients; i++) {...}",
  "tokens_used": 128,
  "finish_reason": "stop"
}
```

##### POST `/v1/chat`
Multi-turn chat completion.

**Request:**
```json
{
  "messages": [
    {"role": "system", "content": "You are an expert SourcePawn developer."},
    {"role": "user", "content": "How do I hook player damage?"}
  ],
  "max_tokens": 1024,
  "temperature": 0.3
}
```

**Response:**
```json
{
  "message": {
    "role": "assistant",
    "content": "To hook player damage in SourcePawn, use SDKHook..."
  },
  "tokens_used": 256
}
```

##### GET `/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model": "l4d2-mistral-v10plus-lora",
  "device": "cuda:0",
  "uptime_seconds": 3600
}
```

#### Starting the Server

```bash
# Default settings
python scripts/inference/copilot_server.py

# Custom port and model
python scripts/inference/copilot_server.py \
    --port 8080 \
    --adapter model_adapters/l4d2-mistral-v10plus-lora/final

# With SSL
python scripts/inference/copilot_server.py \
    --ssl-cert /path/to/cert.pem \
    --ssl-key /path/to/key.pem
```

---

### copilot_cli.py

Command-line interface for code generation.

#### Commands

##### `ollama` - Generate with Ollama (Recommended)

```bash
# Single prompt
python scripts/inference/copilot_cli.py ollama \
    --prompt "Write a function to spawn a Tank"

# With custom model
python scripts/inference/copilot_cli.py ollama \
    --prompt "Create a menu system" \
    --model l4d2-code-v10plus
```

##### `complete` - Server-based completion

```bash
python scripts/inference/copilot_cli.py complete \
    --prompt "Hook the player_spawn event" \
    --server http://localhost:8000
```

##### `chat` - Interactive mode

```bash
# Start interactive session
python scripts/inference/copilot_cli.py chat

# With Ollama backend
python scripts/inference/copilot_cli.py chat --backend ollama
```

##### `template` - Code scaffolding

```bash
# List available templates
python scripts/inference/copilot_cli.py template --list

# Generate from template
python scripts/inference/copilot_cli.py template \
    --template plugin_base \
    --output my_plugin.sp

# Available templates:
#   - plugin_base: Basic plugin structure
#   - admin_command: Admin command handler
#   - event_handler: Event hook template
#   - timer_system: Repeating timer setup
#   - menu_system: Interactive menu
```

---

### rag_copilot.py

RAG-enhanced code generation with documentation retrieval.

#### Endpoints

##### POST `/v1/rag/complete`
Generate completion with relevant documentation context.

**Request:**
```json
{
  "prompt": "How do I use L4D_GetPlayerZombieClass?",
  "max_docs": 5,
  "include_sources": true
}
```

**Response:**
```json
{
  "completion": "L4D_GetPlayerZombieClass returns the zombie class...",
  "sources": [
    {"title": "L4D2 Scripting API", "url": "...", "relevance": 0.92},
    {"title": "Left4DHooks Documentation", "url": "...", "relevance": 0.87}
  ]
}
```

##### POST `/v1/rag/index`
Index new documentation for retrieval.

**Request:**
```json
{
  "documents": [
    {
      "title": "Custom Plugin Guide",
      "content": "...",
      "url": "https://example.com/guide"
    }
  ]
}
```

#### RAG Workflow

```python
from scripts.inference.rag_copilot import RAGCopilot

# Initialize RAG system
copilot = RAGCopilot(
    model_path="model_adapters/l4d2-mistral-v10plus-lora/final",
    index_path="data/rag_index"
)

# Build index from documentation
copilot.build_index(
    docs_path="data/raw/valve_wiki",
    chunk_size=512,
    chunk_overlap=64
)

# Query with RAG
response = copilot.complete(
    prompt="How do I detect when a player enters a saferoom?",
    max_docs=5
)
print(response.completion)
print("Sources:", response.sources)
```

---

## RL Training API

### MnemosyneEnv

Gymnasium-compatible environment for L4D2 bot training via the Mnemosyne SourceMod plugin.

#### Observation Space (20D)

| Index | Name | Range | Description |
|-------|------|-------|-------------|
| 0 | health | 0-1 | Player health normalized |
| 1 | is_alive | 0/1 | Alive status |
| 2 | is_incapped | 0/1 | Incapacitated status |
| 3-5 | position | -1 to 1 | X, Y, Z coordinates |
| 6-8 | velocity | -1 to 1 | Movement velocity |
| 9-10 | angle | -1 to 1 | Pitch and yaw |
| 11 | primary_weapon | 0-1 | Weapon type |
| 12 | ammo | 0-1 | Ammo percentage |
| 13 | nearby_enemies | 0-1 | Enemy count (0-10) |
| 14 | nearest_enemy_dist | 0-1 | Distance to nearest enemy |
| 15 | nearest_teammate_dist | 0-1 | Distance to nearest teammate |
| 16 | teammates_alive | 0-1 | Count of alive teammates |
| 17 | teammates_incapped | 0-1 | Count of incapped teammates |
| 18 | in_safe_room | 0/1 | Safe room status |
| 19 | near_objective | 0/1 | Near checkpoint/objective |

#### Action Space (14 discrete actions)

| Action | ID | Description |
|--------|------|-------------|
| IDLE | 0 | No action |
| MOVE_FORWARD | 1 | Move forward |
| MOVE_BACKWARD | 2 | Move backward |
| MOVE_LEFT | 3 | Strafe left |
| MOVE_RIGHT | 4 | Strafe right |
| ATTACK | 5 | Primary attack |
| USE | 6 | Use/interact |
| RELOAD | 7 | Reload weapon |
| CROUCH | 8 | Crouch |
| JUMP | 9 | Jump |
| SHOVE | 10 | Melee shove |
| HEAL_SELF | 11 | Use health item on self |
| HEAL_OTHER | 12 | Heal teammate |
| THROW_ITEM | 13 | Throw grenade/pipe bomb |

#### Usage Example

```python
from scripts.rl_training.mnemosyne_env import MnemosyneEnv

# Connect to live L4D2 server
env = MnemosyneEnv(
    host="127.0.0.1",
    port=27050,
    timeout=5.0
)

obs, info = env.reset()
for _ in range(1000):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

---

### EnhancedL4D2Env

Self-contained simulation environment for training without a live server.

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_episode_steps` | int | 5000 | Maximum steps per episode |
| `render_mode` | str | None | "human" for text output |
| `difficulty` | str | "normal" | easy/normal/hard/expert |
| `seed` | int | None | Random seed for reproducibility |

#### Features

- **Zombie AI**: Different behaviors per infected type
  - Common: Simple approach and attack
  - Smoker: Tongue pull from distance
  - Hunter: Pounce attacks
  - Boomer: Bile blindness and horde attraction
  - Tank: Relentless pursuit with heavy damage

- **Map Progression**: Distance-based progress toward saferoom
- **Item System**: Medkits, pills, throwables, ammo
- **Teammate AI**: Follow player, provide cover fire

#### Difficulty Settings

| Difficulty | Damage Mult | Spawn Mult | Special Mult |
|------------|-------------|------------|--------------|
| easy | 0.5 | 0.5 | 0.3 |
| normal | 1.0 | 1.0 | 1.0 |
| hard | 1.5 | 1.5 | 1.5 |
| expert | 2.0 | 2.0 | 2.0 |

#### Usage Example

```python
from scripts.rl_training.enhanced_mock_env import EnhancedL4D2Env

env = EnhancedL4D2Env(
    difficulty="hard",
    max_episode_steps=5000,
    render_mode="human"
)

obs, info = env.reset(seed=42)
total_reward = 0

while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated or truncated:
        print(f"Episode ended. Reward: {total_reward}")
        print(f"Stats: {info['stats']}")
        break
```

---

### train_ppo.py

PPO training script with personality-based reward shaping.

#### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--timesteps` | int | 1000000 | Total training timesteps |
| `--personality` | str | "balanced" | Bot personality type |
| `--learning-rate` | float | 3e-4 | PPO learning rate |
| `--batch-size` | int | 64 | Batch size |
| `--n-epochs` | int | 10 | PPO epochs per update |
| `--output` | str | Auto | Output directory |
| `--resume` | str | None | Resume from checkpoint |

#### Usage Examples

```bash
# Train balanced personality
python scripts/rl_training/train_ppo.py \
    --timesteps 1000000 \
    --personality balanced

# Train aggressive personality with custom LR
python scripts/rl_training/train_ppo.py \
    --timesteps 2000000 \
    --personality aggressive \
    --learning-rate 1e-4

# Resume training
python scripts/rl_training/train_ppo.py \
    --resume model_adapters/rl_agents/balanced_1000000.zip
```

---

### train_all_personalities.py

Train multiple bot personalities in sequence or parallel.

#### Bot Personalities

| Personality | Description | Key Reward Weights |
|-------------|-------------|-------------------|
| **balanced** | Well-rounded survivalist | kill: 1.0, heal: 5.0, progress: 0.05 |
| **aggressive** | Damage-focused | kill: 3.0, kill_special: 10.0, damage_dealt: 0.3 |
| **medic** | Team healer | heal_teammate: 15.0, team_proximity: 0.05 |
| **speedrunner** | Objective-focused | safe_room: 200.0, progress: 0.2, survival: 0.005 |
| **defender** | Team protector | proximity_to_team: 0.02, heal_teammate: 8.0 |

#### Personality Configuration

```python
PERSONALITIES = {
    "balanced": {
        "reward_config": {
            "kill": 1.0,
            "kill_special": 5.0,
            "heal_teammate": 5.0,
            "heal_self": 2.0,
            "damage_dealt": 0.1,
            "damage_taken": -0.1,
            "incapped": -10.0,
            "death": -50.0,
            "safe_room": 100.0,
            "survival": 0.01,
            "progress": 0.05,
        },
        "ppo_config": {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
        }
    },
    "aggressive": {
        "reward_config": {
            "kill": 3.0,
            "kill_special": 10.0,
            "damage_dealt": 0.3,
            "heal_teammate": 1.0,
            "safe_room": 50.0,
        },
        # ...
    },
    # Additional personalities...
}
```

#### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--personalities` | str | "all" | Comma-separated list or "all" |
| `--timesteps` | int | 1000000 | Timesteps per personality |
| `--parallel` | flag | False | Train in parallel (requires multi-GPU) |
| `--output-dir` | str | `model_adapters/rl_agents` | Output directory |

#### Usage Examples

```bash
# Train all personalities
python scripts/rl_training/train_all_personalities.py \
    --personalities all \
    --timesteps 1000000

# Train specific personalities
python scripts/rl_training/train_all_personalities.py \
    --personalities balanced,aggressive,medic

# Parallel training (multi-GPU)
python scripts/rl_training/train_all_personalities.py \
    --parallel \
    --timesteps 2000000
```

---

## Director API

### DirectorEnv

Gymnasium environment for training the AI Director.

#### Observation Space (16D)

| Index | Name | Range | Description |
|-------|------|-------|-------------|
| 0 | avg_survivor_stress | 0-1 | Average stress level |
| 1 | survivor_health_avg | 0-1 | Average health |
| 2 | survivor_health_min | 0-1 | Minimum health |
| 3 | survivors_alive | 0-1 | Count of alive survivors |
| 4 | survivors_incapped | 0-1 | Count of incapped survivors |
| 5 | threat_level | 0-1 | Current threat intensity |
| 6 | active_hordes | 0-1 | Number of active hordes |
| 7 | active_special_infected | 0-1 | Special infected count |
| 8 | tank_active | 0/1 | Tank is alive |
| 9 | witch_nearby | 0/1 | Witch within range |
| 10 | map_progress | 0-1 | Progress percentage |
| 11 | time_since_last_event | 0-1 | Normalized time |
| 12 | team_spacing | 0-1 | Team spread distance |
| 13 | ammo_level_avg | 0-1 | Average ammo |
| 14 | health_items_available | 0-1 | Health item count |
| 15 | stress_momentum | -1 to 1 | Stress change rate |

#### Action Space (15 discrete actions)

```python
class DirectorAction(IntEnum):
    IDLE = 0                  # No action
    SPAWN_COMMONS_LOW = 1     # Spawn 5-10 commons
    SPAWN_COMMONS_MEDIUM = 2  # Spawn 10-20 commons
    SPAWN_COMMONS_HIGH = 3    # Spawn 20-30 commons
    SPAWN_HORDE = 4           # Trigger horde event
    SPAWN_SPECIAL_RANDOM = 5  # Random special infected
    SPAWN_SMOKER = 6
    SPAWN_BOOMER = 7
    SPAWN_HUNTER = 8
    SPAWN_SPITTER = 9
    SPAWN_CHARGER = 10
    SPAWN_JOCKEY = 11
    SPAWN_TANK = 12          # Spawn Tank (major event)
    SPAWN_WITCH = 13         # Place Witch
    DROP_AMMO = 14           # Drop ammo pile
```

---

### train_director_rl.py

Train the AI Director using PPO.

#### Director Personalities

| Personality | Target Stress | Spawn Multiplier | Description |
|-------------|---------------|------------------|-------------|
| **relaxed** | 0.3 | 0.6 | Casual gameplay |
| **standard** | 0.5 | 1.0 | Balanced experience |
| **intense** | 0.7 | 1.5 | High action |
| **nightmare** | 0.85 | 2.0 | Maximum challenge |

#### Personality Configuration

```python
DIRECTOR_PERSONALITIES = {
    "relaxed": {
        "target_stress": 0.3,
        "spawn_multiplier": 0.6,
        "tank_chance": 0.001,
        "witch_chance": 0.002,
        "horde_cooldown": 120.0,
    },
    "standard": {
        "target_stress": 0.5,
        "spawn_multiplier": 1.0,
        "tank_chance": 0.005,
        "witch_chance": 0.01,
        "horde_cooldown": 60.0,
    },
    "intense": {
        "target_stress": 0.7,
        "spawn_multiplier": 1.5,
        "tank_chance": 0.01,
        "witch_chance": 0.02,
        "horde_cooldown": 45.0,
    },
    "nightmare": {
        "target_stress": 0.85,
        "spawn_multiplier": 2.0,
        "tank_chance": 0.02,
        "witch_chance": 0.03,
        "horde_cooldown": 30.0,
    },
}
```

#### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--timesteps` | int | 500000 | Training timesteps |
| `--personality` | str | "standard" | Director personality |
| `--learning-rate` | float | 3e-4 | Learning rate |
| `--output` | str | Auto | Output path |

#### Usage Examples

```bash
# Train standard director
python scripts/director/train_director_rl.py \
    --timesteps 500000 \
    --personality standard

# Train nightmare director
python scripts/director/train_director_rl.py \
    --timesteps 1000000 \
    --personality nightmare \
    --learning-rate 1e-4
```

---

## Evaluation API

### benchmark_suite.py

Comprehensive benchmark for evaluating SourcePawn code generation models.

#### Test Categories

| Category | Tests | Description |
|----------|-------|-------------|
| `basic_syntax` | 10 | Plugin structure, commands, ConVars |
| `l4d2_api` | 15 | L4D2-specific APIs and functions |
| `event_handling` | 10 | Game event hooks |
| `special_infected` | 10 | SI mechanics and events |
| `advanced_patterns` | 10 | Complex plugin patterns |

#### TestCase Structure

```python
@dataclass
class TestCase:
    id: str                              # Unique identifier
    prompt: str                          # Test prompt
    expected_patterns: List[str]         # Required patterns
    forbidden_patterns: List[str]        # Patterns that fail the test
    category: Category                   # Test category
    difficulty: Difficulty               # easy/medium/hard
    description: str                     # Human-readable description
    expected_patterns_any: List[List[str]]  # Alternative valid patterns
    min_code_lines: int                  # Minimum code length
    requires_includes: List[str]         # Required includes
```

#### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | "ollama" | Model type (ollama/openai/base) |
| `--model-id` | str | None | OpenAI model ID |
| `--model-name` | str | None | Ollama model name |
| `--output` | str | `results/benchmark.json` | Output file |
| `--markdown` | str | None | Generate Markdown report |
| `--quick` | flag | False | Run subset of tests |
| `--category` | str | None | Filter by category |
| `--difficulty` | str | None | Filter by difficulty |
| `--compare` | flag | False | Compare multiple models |
| `--models` | str | None | Comma-separated model list |
| `--list-tests` | flag | False | List all test cases |

#### Usage Examples

```bash
# Run full benchmark with Ollama
python scripts/evaluation/benchmark_suite.py \
    --model ollama \
    --model-name l4d2-code-v10plus \
    --output results/benchmark_v10plus.json

# Quick test (subset)
python scripts/evaluation/benchmark_suite.py \
    --model ollama \
    --quick

# Filter by category
python scripts/evaluation/benchmark_suite.py \
    --model ollama \
    --category l4d2_api

# Compare multiple models
python scripts/evaluation/benchmark_suite.py \
    --compare \
    --models ollama,openai,base \
    --model-id ft:gpt-4o-mini-2024-07-18:...

# Generate Markdown report
python scripts/evaluation/benchmark_suite.py \
    --model ollama \
    --markdown results/benchmark_report.md

# List all tests
python scripts/evaluation/benchmark_suite.py --list-tests
```

#### Adding Custom Test Cases

```python
from scripts.evaluation.benchmark_suite import TestCase, Category, Difficulty

custom_test = TestCase(
    id="custom_tank_announce",
    prompt="Write a plugin that announces Tank health when spawned",
    expected_patterns=["HookEvent", "tank_spawn", "GetEntProp", "m_iHealth"],
    forbidden_patterns=["tank_health_changed"],
    category=Category.SPECIAL_INFECTED,
    difficulty=Difficulty.MEDIUM,
    description="Tank spawn announcement with health display",
    min_code_lines=10,
    requires_includes=["sourcemod", "sdktools"],
)

# Add to test suite
BENCHMARK_TESTS.append(custom_test)
```

#### Benchmark Report Structure

```python
@dataclass
class BenchmarkReport:
    model_name: str           # Model identifier
    model_type: str           # ollama/openai/base
    timestamp: str            # ISO timestamp
    total_tests: int          # Number of tests run
    passed: int               # Passing tests
    failed: int               # Failing tests
    pass_rate: float          # Pass percentage
    average_score: float      # Average score (0-10)
    by_category: Dict         # Results grouped by category
    by_difficulty: Dict       # Results grouped by difficulty
    test_results: List[Dict]  # Individual test results
    execution_time: float     # Total time in seconds
    common_issues: List       # Most frequent issues
```

---

## Utility Functions

### security.py

Security utilities for path sanitization and URL validation.

#### safe_path()

Validates and sanitizes file paths to prevent path traversal attacks.

```python
from scripts.utils.security import safe_path
from pathlib import Path

PROJECT_ROOT = Path("/path/to/project")

# Valid path - returns resolved Path
validated = safe_path("data/output.json", PROJECT_ROOT)
# Returns: /path/to/project/data/output.json

# Path traversal attempt - raises ValueError
try:
    safe_path("../../../etc/passwd", PROJECT_ROOT)
except ValueError as e:
    print(f"Blocked: {e}")

# Create parent directories
validated = safe_path("new/nested/file.json", PROJECT_ROOT, create_parents=True)
```

#### safe_read_json() / safe_write_json()

Safe file I/O with path validation.

```python
from scripts.utils.security import safe_read_json, safe_write_json
from pathlib import Path

PROJECT_ROOT = Path("/path/to/project")

# Read JSON safely
config = safe_read_json("configs/settings.json", PROJECT_ROOT)

# Write JSON safely
data = {"key": "value", "count": 42}
output_path = safe_write_json(
    "data/output.json",
    data,
    PROJECT_ROOT,
    indent=2
)
print(f"Written to: {output_path}")
```

#### safe_read_yaml()

Safe YAML file reading.

```python
from scripts.utils.security import safe_read_yaml
from pathlib import Path

PROJECT_ROOT = Path("/path/to/project")

config = safe_read_yaml("configs/unsloth_config.yaml", PROJECT_ROOT)
print(f"Model: {config['model']['name']}")
```

#### safe_write_text() / safe_read_text()

Safe text file I/O.

```python
from scripts.utils.security import safe_write_text, safe_read_text
from pathlib import Path

PROJECT_ROOT = Path("/path/to/project")

# Write text file
safe_write_text(
    "exports/model.txt",
    "Model exported successfully",
    PROJECT_ROOT
)

# Read text file
content = safe_read_text("exports/model.txt", PROJECT_ROOT)
```

#### safe_write_jsonl()

Write JSONL (JSON Lines) files safely.

```python
from scripts.utils.security import safe_write_jsonl
from pathlib import Path

PROJECT_ROOT = Path("/path/to/project")

items = [
    {"prompt": "Question 1", "response": "Answer 1"},
    {"prompt": "Question 2", "response": "Answer 2"},
]

safe_write_jsonl("data/processed/dataset.jsonl", items, PROJECT_ROOT)
```

#### validate_url()

Validates URLs against an allowlist to prevent SSRF attacks.

```python
from scripts.utils.security import validate_url

# Default allowed domains (GitHub, Valve Developer Wiki)
safe_url = validate_url("https://api.github.com/repos/user/repo")

# Custom allowed domains
ALLOWED = {"example.com", "api.example.com"}
safe_url = validate_url(
    "https://api.example.com/data",
    allowed_domains=ALLOWED
)

# Blocked - internal IP
try:
    validate_url("http://192.168.1.1/admin")
except ValueError as e:
    print(f"Blocked: {e}")

# Blocked - not in allowlist
try:
    validate_url("https://malicious-site.com/payload")
except ValueError as e:
    print(f"Blocked: {e}")
```

#### sanitize_filename()

Sanitize user-provided filenames.

```python
from scripts.utils.security import sanitize_filename

# Remove dangerous characters
safe_name = sanitize_filename("../../../etc/passwd")
# Returns: "etc_passwd"

safe_name = sanitize_filename("file<with>bad:chars?.txt")
# Returns: "file_with_bad_chars_.txt"

# Truncate long names
safe_name = sanitize_filename("a" * 300, max_length=100)
# Returns: "aaa..." (100 chars)
```

---

### object_storage_sync.py

Synchronize files with Vultr S3-compatible object storage.

#### ObjectStorageSync Class

```python
from scripts.utils.object_storage_sync import ObjectStorageSync

# Initialize with credentials
sync = ObjectStorageSync(
    endpoint="https://ewr1.vultrobjects.com",
    access_key="YOUR_ACCESS_KEY",
    secret_key="YOUR_SECRET_KEY",
    bucket="l4d2-models"
)

# Upload a file
sync.upload_file(
    local_path="model_adapters/l4d2-mistral-v10plus-lora/final",
    remote_path="models/v10plus/final"
)

# Upload directory
sync.upload_directory(
    local_dir="model_adapters/l4d2-mistral-v10plus-lora",
    remote_prefix="models/v10plus"
)

# Download file
sync.download_file(
    remote_path="models/v10plus/final/adapter_model.bin",
    local_path="downloads/adapter_model.bin"
)

# List remote files
files = sync.list_files(prefix="models/")
for f in files:
    print(f"{f['key']}: {f['size']} bytes")

# Sync with checksum verification
sync.sync_directory(
    local_dir="model_adapters",
    remote_prefix="backups/models",
    delete_orphaned=False  # Don't delete remote files not in local
)
```

#### Environment Variables

```bash
# Set credentials via environment
export VULTR_S3_ACCESS_KEY="your-access-key"
export VULTR_S3_SECRET_KEY="your-secret-key"
export VULTR_S3_ENDPOINT="https://ewr1.vultrobjects.com"
export VULTR_S3_BUCKET="l4d2-models"
```

#### CLI Usage

```bash
# Upload model
python scripts/utils/object_storage_sync.py upload \
    --local model_adapters/l4d2-mistral-v10plus-lora \
    --remote models/v10plus

# Download model
python scripts/utils/object_storage_sync.py download \
    --remote models/v10plus \
    --local downloaded_models/v10plus

# List files
python scripts/utils/object_storage_sync.py list \
    --prefix models/

# Sync directory
python scripts/utils/object_storage_sync.py sync \
    --local model_adapters \
    --remote backups/models
```

---

## Appendix: Common Patterns

### Training a New Model End-to-End

```bash
# 1. Collect data
./run_scraping.sh

# 2. Prepare dataset
python scripts/training/prepare_dataset.py

# 3. Train model
python scripts/training/train_unsloth.py --config configs/unsloth_config.yaml

# 4. Export to GGUF
python scripts/training/export_gguf_cpu.py \
    --adapter model_adapters/l4d2-mistral-v10plus-lora/final

# 5. Install to Ollama
ollama create l4d2-code-v10plus -f exports/l4d2-v10plus/gguf/Modelfile

# 6. Run benchmark
python scripts/evaluation/benchmark_suite.py \
    --model ollama \
    --model-name l4d2-code-v10plus \
    --output results/benchmark.json
```

### Training All RL Agents

```bash
# Train all bot personalities
python scripts/rl_training/train_all_personalities.py \
    --personalities all \
    --timesteps 1000000

# Train all director personalities
for personality in relaxed standard intense nightmare; do
    python scripts/director/train_director_rl.py \
        --personality $personality \
        --timesteps 500000
done
```

### Secure File Handling Template

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import (
    safe_path,
    safe_read_json,
    safe_write_json,
    validate_url
)

PROJECT_ROOT = Path(__file__).parent.parent.parent

def process_data(input_path: str, output_path: str):
    """Process data with secure file handling."""
    # Validate and read input
    data = safe_read_json(input_path, PROJECT_ROOT)

    # Process data
    processed = transform(data)

    # Safely write output
    safe_write_json(output_path, processed, PROJECT_ROOT, indent=2)
```
