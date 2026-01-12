#!/usr/bin/env python3
"""
L4D2 AI Controller

Reads game state from file, runs PPO models, writes commands back.
Works with the l4d2_ai_file_bridge SourceMod plugin.

Usage:
    python l4d2_ai_controller.py --personality balanced
    python l4d2_ai_controller.py --personality aggressive --debug
"""

import argparse
import json
import time
import os
import sys
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import IntEnum

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


@dataclass
class SurvivorState:
    """State of a single survivor."""
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
    """Complete game state."""
    time: float
    map_name: str
    survivors: List[SurvivorState]
    infected_common: int
    infected_witches: int
    infected_tanks: int


class L4D2AIController:
    """
    AI Controller for L4D2 bots.

    Reads game state, runs inference, writes commands.
    """

    def __init__(
        self,
        state_file: str = "/tmp/l4d2_state.json",
        command_file: str = "/tmp/l4d2_commands.txt",
        model_path: Optional[str] = None,
        personality: str = "balanced",
        debug: bool = False
    ):
        self.state_file = Path(state_file)
        self.command_file = Path(command_file)
        self.model_path = model_path
        self.personality = personality
        self.debug = debug

        self.model = None
        self.last_state_time = 0.0
        self.actions_sent = 0

        # Load model if provided
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load a trained PPO model."""
        try:
            from stable_baselines3 import PPO
            self.model = PPO.load(model_path)
            print(f"[AI] Loaded model from {model_path}")
        except Exception as e:
            print(f"[AI] Warning: Could not load model: {e}")
            print("[AI] Using rule-based fallback")
            self.model = None

    def read_game_state(self) -> Optional[GameState]:
        """Read the current game state from file."""
        if not self.state_file.exists():
            return None

        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)

            # Parse survivors
            survivors = []
            for s in data.get('survivors', []):
                survivors.append(SurvivorState(
                    id=s['id'],
                    name=s.get('name', f"Bot_{s['id']}"),
                    health=s['health'],
                    alive=s['alive'],
                    incapped=s['incapped'],
                    is_bot=s['bot'],
                    pos=s['pos'],
                    ang=s['ang'],
                    vel=s.get('vel', [0, 0, 0]),
                    weapon=s['weapon']
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
            if self.debug:
                print(f"[AI] Error reading state: {e}")
            return None

    def state_to_observation(self, state: GameState, bot_id: int) -> np.ndarray:
        """
        Convert game state to observation vector for the model.

        Observation space (20 dimensions):
        [health, alive, incapped, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
         angle_pitch, angle_yaw, primary_weapon, ammo, enemies,
         enemy_dist, teammate_dist, teammates_alive, teammates_incapped,
         in_safe_room, near_objective]
        """
        # Find this bot
        bot = None
        for s in state.survivors:
            if s.id == bot_id:
                bot = s
                break

        if not bot:
            return np.zeros(20, dtype=np.float32)

        # Calculate distances
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

        # Weapon ID (simple mapping)
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
            bot.health / 100.0,  # Normalized health
            1.0 if bot.alive else 0.0,
            1.0 if bot.incapped else 0.0,
            bot.pos[0] / 10000.0,  # Normalized position
            bot.pos[1] / 10000.0,
            bot.pos[2] / 1000.0,
            bot.vel[0] / 500.0,  # Normalized velocity
            bot.vel[1] / 500.0,
            bot.vel[2] / 500.0,
            bot.ang[0] / 90.0,  # Normalized angles
            bot.ang[1] / 180.0,
            weapon_id / 15.0,
            1.0,  # Ammo (placeholder)
            min(state.infected_common, 30) / 30.0,  # Enemy count
            500.0 / max(500.0, min_teammate_dist),  # Enemy distance estimate
            min_teammate_dist / 1000.0,  # Teammate distance
            teammates_alive / 3.0,
            teammates_incapped / 3.0,
            0.0,  # In safe room (placeholder)
            0.0,  # Near objective (placeholder)
        ], dtype=np.float32)

        return obs

    def get_action(self, state: GameState, bot_id: int) -> int:
        """Get action for a bot using the model or fallback rules."""
        if self.model:
            obs = self.state_to_observation(state, bot_id)
            action, _ = self.model.predict(obs, deterministic=True)
            return int(action)

        # Rule-based fallback based on personality
        return self.get_rule_based_action(state, bot_id)

    def get_rule_based_action(self, state: GameState, bot_id: int) -> int:
        """Simple rule-based action selection."""
        bot = None
        for s in state.survivors:
            if s.id == bot_id:
                bot = s
                break

        if not bot or not bot.alive:
            return BotAction.IDLE

        # Personality-based behavior
        if self.personality == "aggressive":
            # Always attack if enemies present
            if state.infected_common > 0:
                return BotAction.ATTACK
            return BotAction.MOVE_FORWARD

        elif self.personality == "medic":
            # Check for wounded teammates
            for s in state.survivors:
                if s.id != bot_id and s.alive and s.health < 50:
                    return BotAction.HEAL_OTHER
            if bot.health < 40:
                return BotAction.HEAL_SELF
            return BotAction.MOVE_FORWARD

        elif self.personality == "defender":
            # Stay near teammates, attack when needed
            if state.infected_common > 5:
                return BotAction.ATTACK
            return BotAction.IDLE

        elif self.personality == "speedrunner":
            # Always move forward
            return BotAction.MOVE_FORWARD

        else:  # balanced
            # Mix of actions
            if state.infected_common > 0:
                return BotAction.ATTACK
            if bot.health < 30:
                return BotAction.HEAL_SELF
            return BotAction.MOVE_FORWARD

    def write_command(self, bot_id: int, action: int):
        """Write a command to the command file."""
        with open(self.command_file, 'a') as f:
            f.write(f"{bot_id},{action}\n")
        self.actions_sent += 1

    def run_once(self):
        """Run a single iteration of the control loop."""
        state = self.read_game_state()
        if not state:
            return False

        # Skip if state hasn't changed
        if state.time == self.last_state_time:
            return True
        self.last_state_time = state.time

        # Clear command file
        if self.command_file.exists():
            self.command_file.unlink()

        # Process each bot
        for survivor in state.survivors:
            if survivor.is_bot and survivor.alive:
                action = self.get_action(state, survivor.id)
                self.write_command(survivor.id, action)

                if self.debug:
                    print(f"[AI] Bot {survivor.name} (id={survivor.id}): "
                          f"health={survivor.health}, action={BotAction(action).name}")

        return True

    def run(self, hz: float = 10.0):
        """Run the control loop."""
        print(f"[AI] Starting L4D2 AI Controller")
        print(f"[AI] Personality: {self.personality}")
        print(f"[AI] State file: {self.state_file}")
        print(f"[AI] Command file: {self.command_file}")
        print(f"[AI] Update rate: {hz} Hz")
        print(f"[AI] Model: {'Loaded' if self.model else 'Rule-based fallback'}")
        print("[AI] Press Ctrl+C to stop")
        print()

        interval = 1.0 / hz

        try:
            while True:
                start = time.time()

                success = self.run_once()

                if self.debug and self.actions_sent % 100 == 0:
                    print(f"[AI] Actions sent: {self.actions_sent}")

                # Sleep to maintain update rate
                elapsed = time.time() - start
                if elapsed < interval:
                    time.sleep(interval - elapsed)

        except KeyboardInterrupt:
            print(f"\n[AI] Stopped. Total actions sent: {self.actions_sent}")


def main():
    parser = argparse.ArgumentParser(description="L4D2 AI Controller")
    parser.add_argument("--state-file", default="/tmp/l4d2_state.json",
                        help="Path to game state file")
    parser.add_argument("--command-file", default="/tmp/l4d2_commands.txt",
                        help="Path to command file")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained PPO model (.zip)")
    parser.add_argument("--personality", default="balanced",
                        choices=["aggressive", "balanced", "medic", "defender", "speedrunner"],
                        help="Bot personality")
    parser.add_argument("--hz", type=float, default=10.0,
                        help="Update rate in Hz")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")

    args = parser.parse_args()

    controller = L4D2AIController(
        state_file=args.state_file,
        command_file=args.command_file,
        model_path=args.model,
        personality=args.personality,
        debug=args.debug
    )

    controller.run(hz=args.hz)


if __name__ == "__main__":
    main()
