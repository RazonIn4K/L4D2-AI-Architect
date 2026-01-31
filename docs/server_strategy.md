# L4D2 Server Strategy & AI Integration

## 1. Hosting Custom Maps & Mods

To host a server where you can play custom maps and mods with others, you have two primary paths: **Listen Server** (hosting from your own game client) or **Dedicated Server** (standalone). For a persistent, modded experience, a **Dedicated Server** is recommended.

### Workshop Collections (The Modern Way)

Instead of manually distributing files, use Steam Workshop Collections.

1.  **Create a Collection**: On Steam Workshop, create a "Server Collection" adding all the maps and mods you want.
2.  **Server Config**: Add the collection ID to your server's startup arguments: `+host_workshop_collection [COLLECTION_ID]`.
3.  **Client Sync**: When players join, they will be prompted to download the matching add-ons from the Workshop automatically.

### Handling "Other People's Mods" (Consistency)

You asked: _"Would I just enforce the servers mods only?"_
This is controlled by `sv_consistency` and `sv_pure`.

- **`sv_pure 1` (Strict)**: Forces clients to use _only_ the files specified by the server. This prevents players from using their own "cheat" hacks or drastically different models that might conflict.
- **`sv_consistency 1` (Standard)**: Checks if critical model specifications match. If a player has a "Shrek Tank" mod and the server has a "Thanos Tank" mod, they will get kicked with a consistency error.
- **Recommended Strategy**:
  - Set `sv_pure 0` or `sv_consistency 0` if you want to allow friends to use their own weapon skins/HUDs (Client-side mods).
  - Set `sv_consistency 1` if you want to ensure everyone sees exactly what the server dictates (e.g., custom infected models).

## 2. AI Integration Strategy

Your `L4D2-AI-Architect` project interacts with the game primarily via **SourceMod** (Server-side) and **VScript** (Server-side).

### How the AI "Plays"

The AI doesn't "watch" the screen like a human. It runs as logic on the server.

1.  **Reinforcement Learning (RL) Agents**:
    - The `train_ppo.py` script trains a model offline.
    - To "deploy" this on a live server, you need a bridge. The repository contains `l4d2_ai_bridge_v2.sp` (SourceMod plugin).
    - **Workflow**: The SourceMod plugin creates a TCP socket server. Your Python inference script connects to it. The plugin sends game state (zombie positions, health) to Python -> Python decides action -> Python sends command back to SourceMod -> SourceMod executes input on the bot.

### Deployment Architecture

- **Game Server**: Runs L4D2 Dedicated Server + SourceMod + `Mnemosyne` Plugin.
- **AI Brain**: Runs a Python script (`copilot_server.py` or `model_server.py`) adjacent to the server (or on the same machine).
- **Connection**: They talk via localhost networking.

### Does the AI need "Mods"?

- **No Client Downloads Needed**: The AI logic is purely server-side. Players joining your server do **not** need to download your Python scripts or AI models. They just see the bots acting smarter (or weirder).
- **Custom Maps**: If the AI is trained on navigating generic nav meshes, it should work on custom maps immediately, provided the custom map has a valid `.nav` file (which most do).

## 3. Next Steps

1.  **Set up a local Dedicated Server** to test the "Workshop Collection" approach.
2.  **Install SourceMod** on that server.
3.  **Deploy the `l4d2_ai_bridge` plugin** to your server.
4.  **Run the Python Inference Server** and verify it can control a bot on your custom server.
