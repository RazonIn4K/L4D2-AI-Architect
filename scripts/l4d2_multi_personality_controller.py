#!/usr/bin/env python3
"""
L4D2 Multi-Personality AI Controller

Each bot gets a DIFFERENT trained PPO personality:
- Rochelle = Medic (prioritizes healing teammates)
- Ellis = Speedrunner (rushes forward aggressively)
- Nick = Defender (protects the team, stays close)
- Coach = Balanced (adaptive behavior)

This creates a unique squad dynamic where each AI has distinct behavior!
"""

import json
import time
import os
import sys
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import IntEnum
from collections import defaultdict

# Action space (must match SourceMod plugin)
class BotAction(IntEnum):
    IDLE = 0
    MOVE_FORWARD = 1
    MOVE_BACKWARD = 2
    MOVE_LEFT = 3
    MOVE_RIGHT = 4
    ATTACK = 5
    USE = 6
    RELOAD = 7
    CROUCH = 8
    JUMP = 9
    SHOVE = 10
    HEAL_SELF = 11
    HEAL_OTHER = 12
    THROW_ITEM = 13

ACTION_NAMES = {
    0: "IDLE", 1: "FORWARD", 2: "BACK", 3: "LEFT", 4: "RIGHT",
    5: "ATTACK", 6: "USE", 7: "RELOAD", 8: "CROUCH", 9: "JUMP",
    10: "SHOVE", 11: "HEAL_SELF", 12: "HEAL_OTHER", 13: "THROW"
}

# Personality assignments by bot name
BOT_PERSONALITIES = {
    "Rochelle": "medic",
    "Ellis": "speedrunner",
    "Nick": "defender",
    "Coach": "balanced",
    # Fallbacks for any name
    "default": "balanced"
}

# Personality descriptions for display
PERSONALITY_DESC = {
    "medic": "HEALER - Prioritizes keeping team alive",
    "speedrunner": "RUSHER - Pushes forward aggressively",
    "defender": "GUARDIAN - Protects teammates, holds position",
    "balanced": "ADAPTIVE - Balanced combat and survival"
}


@dataclass
class SurvivorState:
    id: int
    name: str
    health: int
    alive: bool
    incapped: bool
    is_bot: bool
    pos: List[float]
    ang: List[float]
    vel: List[float]
    weapon: str


@dataclass
class GameState:
    time: float
    map_name: str
    survivors: List[SurvivorState]
    infected_common: int
    infected_witches: int
    infected_tanks: int


class MultiPersonalityController:
    """
    Advanced AI Controller with multiple trained personalities.
    Each bot runs a different trained PPO model!
    """

    def __init__(
        self,
        state_file: str,
        command_file: str,
        model_dir: str,
        debug: bool = True
    ):
        self.state_file = Path(state_file)
        self.command_file = Path(command_file)
        self.model_dir = Path(model_dir)
        self.debug = debug

        self.models = {}  # personality -> PPO model
        self.last_state_time = 0.0
        self.stats = defaultdict(lambda: {"actions": 0, "kills": 0, "heals": 0})
        self.last_actions = {}  # bot_id -> last action for display

        # Load all available models
        self.load_all_models()

    def load_all_models(self):
        """Load all trained personality models."""
        try:
            from stable_baselines3 import PPO

            personalities = ["medic", "speedrunner", "defender", "balanced"]

            for personality in personalities:
                # Find the model file
                pattern = f"ppo_{personality}_*/final_model.zip"
                matches = list(self.model_dir.glob(pattern))

                if matches:
                    model_path = matches[-1]  # Use most recent
                    try:
                        self.models[personality] = PPO.load(str(model_path))
                        print(f"[AI] Loaded {personality.upper()} model from {model_path.name}")
                    except Exception as e:
                        print(f"[AI] Warning: Could not load {personality} model: {e}")
                else:
                    print(f"[AI] No model found for {personality}")

            print(f"[AI] Loaded {len(self.models)} personality models")

        except ImportError:
            print("[AI] stable-baselines3 not available, using rule-based AI")

    def get_personality(self, bot_name: str) -> str:
        """Get personality for a bot based on their name."""
        return BOT_PERSONALITIES.get(bot_name, BOT_PERSONALITIES["default"])

    def state_to_observation(self, state: GameState, bot_id: int) -> np.ndarray:
        """Convert game state to 20D observation vector."""
        bot = None
        for s in state.survivors:
            if s.id == bot_id:
                bot = s
                break

        if not bot:
            return np.zeros(20, dtype=np.float32)

        # Calculate teammate distances
        min_teammate_dist = 9999.0
        teammates_alive = 0
        teammates_incapped = 0

        for s in state.survivors:
            if s.id != bot_id and s.alive:
                teammates_alive += 1
                if s.incapped:
                    teammates_incapped += 1
                dist = np.sqrt(sum((a - b)**2 for a, b in zip(bot.pos, s.pos)))
                min_teammate_dist = min(min_teammate_dist, dist)

        # Weapon mapping
        weapon_ids = {
            'weapon_pistol': 0, 'weapon_pistol_magnum': 1,
            'weapon_smg': 2, 'weapon_smg_silenced': 3,
            'weapon_pumpshotgun': 4, 'weapon_shotgun_chrome': 5,
            'weapon_rifle': 6, 'weapon_rifle_ak47': 7,
            'weapon_hunting_rifle': 8, 'weapon_sniper_military': 9,
            'weapon_autoshotgun': 10, 'weapon_shotgun_spas': 11,
        }
        weapon_id = weapon_ids.get(bot.weapon, 0)

        obs = np.array([
            bot.health / 100.0,
            1.0 if bot.alive else 0.0,
            1.0 if bot.incapped else 0.0,
            bot.pos[0] / 10000.0,
            bot.pos[1] / 10000.0,
            bot.pos[2] / 1000.0,
            bot.vel[0] / 500.0,
            bot.vel[1] / 500.0,
            bot.vel[2] / 500.0,
            bot.ang[0] / 90.0,
            bot.ang[1] / 180.0,
            weapon_id / 15.0,
            1.0,  # Ammo placeholder
            min(state.infected_common, 30) / 30.0,
            500.0 / max(500.0, min_teammate_dist),
            min_teammate_dist / 1000.0,
            teammates_alive / 3.0,
            teammates_incapped / 3.0,
            0.0,  # Safe room placeholder
            0.0,  # Objective placeholder
        ], dtype=np.float32)

        return obs

    def get_model_action(self, state: GameState, bot_id: int, personality: str) -> int:
        """Get action from trained model."""
        if personality in self.models:
            obs = self.state_to_observation(state, bot_id)
            action, _ = self.models[personality].predict(obs, deterministic=True)
            return int(action)
        return self.get_rule_based_action(state, bot_id, personality)

    def get_rule_based_action(self, state: GameState, bot_id: int, personality: str) -> int:
        """Fallback rule-based action."""
        bot = None
        for s in state.survivors:
            if s.id == bot_id:
                bot = s
                break

        if not bot or not bot.alive:
            return BotAction.IDLE

        if personality == "speedrunner":
            # Rush forward, attack only if many enemies
            if state.infected_common > 3:
                return BotAction.ATTACK
            return BotAction.MOVE_FORWARD

        elif personality == "medic":
            # Check for wounded teammates
            for s in state.survivors:
                if s.id != bot_id and s.alive:
                    if s.incapped:
                        return BotAction.USE  # Help up
                    if s.health < 40:
                        return BotAction.HEAL_OTHER
            if bot.health < 50:
                return BotAction.HEAL_SELF
            if state.infected_common > 0:
                return BotAction.ATTACK
            return BotAction.MOVE_FORWARD

        elif personality == "defender":
            # Stay with team, attack threats
            if state.infected_common > 2:
                return BotAction.ATTACK
            if state.infected_tanks > 0:
                return BotAction.ATTACK
            return BotAction.SHOVE  # Push back zombies

        else:  # balanced
            if state.infected_common > 0:
                return BotAction.ATTACK
            if bot.health < 30:
                return BotAction.HEAL_SELF
            return BotAction.MOVE_FORWARD

    def read_game_state(self) -> Optional[GameState]:
        """Read current game state from file."""
        if not self.state_file.exists():
            return None

        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)

            survivors = []
            for s in data.get('survivors', []):
                survivors.append(SurvivorState(
                    id=s['id'],
                    name=s.get('name', f"Bot_{s['id']}"),
                    health=s.get('health', 100),
                    alive=s.get('alive', True),
                    incapped=s.get('incapped', False),
                    is_bot=s.get('bot', True),
                    pos=s.get('pos', [0, 0, 0]),
                    ang=s.get('ang', [0, 0]),
                    vel=s.get('vel', [0, 0, 0]),
                    weapon=s.get('weapon', 'weapon_pistol')
                ))

            infected = data.get('infected', {})

            return GameState(
                time=data.get('time', 0),
                map_name=data.get('map', 'unknown'),
                survivors=survivors,
                infected_common=infected.get('common', 0),
                infected_witches=infected.get('witches', 0),
                infected_tanks=infected.get('tanks', 0)
            )
        except Exception as e:
            return None

    def run_once(self) -> bool:
        """Run single control iteration."""
        state = self.read_game_state()
        if not state:
            return False

        if state.time == self.last_state_time:
            return True
        self.last_state_time = state.time

        # Clear command file
        if self.command_file.exists():
            self.command_file.unlink()

        commands = []
        display_lines = []

        for survivor in state.survivors:
            if survivor.is_bot and survivor.alive:
                personality = self.get_personality(survivor.name)
                action = self.get_model_action(state, survivor.id, personality)

                commands.append(f"{survivor.id},{action}")
                self.stats[survivor.name]["actions"] += 1
                self.last_actions[survivor.id] = action

                # Track special actions
                if action == BotAction.ATTACK:
                    self.stats[survivor.name]["kills"] += 0.1  # Approximate
                elif action in [BotAction.HEAL_SELF, BotAction.HEAL_OTHER]:
                    self.stats[survivor.name]["heals"] += 1

                action_name = ACTION_NAMES.get(action, "?")
                display_lines.append(
                    f"  {survivor.name:10} [{personality:11}] HP:{survivor.health:3} -> {action_name}"
                )

        if commands:
            with open(self.command_file, 'w') as f:
                f.write("\n".join(commands))

        if self.debug and display_lines:
            print(f"\n[AI] Time: {state.time:.1f}s | Map: {state.map_name} | Zombies: {state.infected_common}")
            for line in display_lines:
                print(line)

        return True

    def run(self, hz: float = 10.0):
        """Main control loop."""
        print("\n" + "="*60)
        print("   L4D2 MULTI-PERSONALITY AI CONTROLLER")
        print("="*60)
        print("\nPersonality Assignments:")
        for name, personality in BOT_PERSONALITIES.items():
            if name != "default":
                desc = PERSONALITY_DESC.get(personality, "")
                print(f"  {name:10} -> {personality.upper():12} | {desc}")
        print("\nModels Loaded:", list(self.models.keys()) if self.models else "Rule-based fallback")
        print(f"State File: {self.state_file}")
        print(f"Update Rate: {hz} Hz")
        print("\nWaiting for game state...\n")

        interval = 1.0 / hz
        iteration = 0

        try:
            while True:
                start = time.time()
                self.run_once()
                iteration += 1

                elapsed = time.time() - start
                if elapsed < interval:
                    time.sleep(interval - elapsed)

        except KeyboardInterrupt:
            print("\n\n" + "="*60)
            print("   SESSION STATISTICS")
            print("="*60)
            for name, stats in self.stats.items():
                print(f"  {name}: {stats['actions']} actions, ~{stats['kills']:.0f} kills, {stats['heals']} heals")
            print("="*60)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-file", default="/tmp/sm_install/addons/sourcemod/data/l4d2_state.json")
    parser.add_argument("--command-file", default="/tmp/sm_install/addons/sourcemod/data/l4d2_commands.txt")
    parser.add_argument("--model-dir", default="/opt/l4d2_ai/models")
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--debug", action="store_true", default=True)
    args = parser.parse_args()

    controller = MultiPersonalityController(
        state_file=args.state_file,
        command_file=args.command_file,
        model_dir=args.model_dir,
        debug=args.debug
    )
    controller.run(hz=args.hz)


if __name__ == "__main__":
    main()
