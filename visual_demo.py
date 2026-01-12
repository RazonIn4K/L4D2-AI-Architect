#!/usr/bin/env python3
"""
L4D2 AI Visual Demo

Watch the trained AI agents play in a visual simulation.
No game server required - this visualizes the mock environment.

Usage:
    python visual_demo.py                    # Default: aggressive bot
    python visual_demo.py --personality medic
    python visual_demo.py --speed 2          # 2x speed
    python visual_demo.py --director nightmare

Controls:
    SPACE - Pause/Resume
    R - Reset episode
    1-5 - Switch personality (1=aggressive, 2=balanced, 3=defender, 4=medic, 5=speedrunner)
    D - Toggle director mode
    ESC/Q - Quit
"""

import argparse
import math
import os
import sys
import random
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pygame
import numpy as np

# Initialize pygame
pygame.init()

# Screen settings
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (50, 50, 50)
DARK_GRAY = (30, 30, 30)
RED = (255, 80, 80)
GREEN = (80, 255, 80)
BLUE = (80, 150, 255)
YELLOW = (255, 255, 80)
PURPLE = (200, 80, 255)
ORANGE = (255, 165, 0)
CYAN = (0, 255, 255)

# Game area
GAME_AREA = pygame.Rect(50, 50, 800, 600)
MAP_SCALE = 0.15  # Scale factor for positions

# Action names
ACTION_NAMES = [
    "IDLE", "FORWARD", "BACKWARD", "LEFT", "RIGHT",
    "ATTACK", "USE", "RELOAD", "CROUCH", "JUMP",
    "SHOVE", "HEAL_SELF", "HEAL_OTHER", "THROW"
]


class Survivor:
    """Visual representation of a survivor."""

    def __init__(self, x, y, is_player=False, name="Bot"):
        self.x = x
        self.y = y
        self.angle = 0
        self.health = 100
        self.alive = True
        self.incapped = False
        self.is_player = is_player
        self.name = name
        self.color = BLUE if is_player else GREEN
        self.kills = 0
        self.last_action = "IDLE"

    def draw(self, screen, offset_x=0, offset_y=0):
        if not self.alive:
            return

        # Position on screen
        sx = GAME_AREA.x + (self.x * MAP_SCALE) + offset_x
        sy = GAME_AREA.y + (self.y * MAP_SCALE) + offset_y

        # Keep in bounds
        sx = max(GAME_AREA.left + 10, min(GAME_AREA.right - 10, sx))
        sy = max(GAME_AREA.top + 10, min(GAME_AREA.bottom - 10, sy))

        # Draw body
        color = YELLOW if self.incapped else (self.color if self.is_player else GREEN)
        pygame.draw.circle(screen, color, (int(sx), int(sy)), 15)

        # Draw direction indicator
        dx = math.cos(math.radians(self.angle)) * 20
        dy = math.sin(math.radians(self.angle)) * 20
        pygame.draw.line(screen, WHITE, (sx, sy), (sx + dx, sy + dy), 3)

        # Health bar
        bar_width = 30
        bar_height = 4
        health_pct = self.health / 100
        pygame.draw.rect(screen, RED, (sx - bar_width//2, sy - 25, bar_width, bar_height))
        pygame.draw.rect(screen, GREEN, (sx - bar_width//2, sy - 25, int(bar_width * health_pct), bar_height))

        # Name
        font = pygame.font.Font(None, 18)
        if self.is_player:
            text = font.render(f"AI ({self.last_action})", True, CYAN)
        else:
            text = font.render(self.name, True, WHITE)
        screen.blit(text, (sx - text.get_width()//2, sy + 20))


class Zombie:
    """Visual representation of a zombie."""

    def __init__(self, x, y, zombie_type="common"):
        self.x = x
        self.y = y
        self.zombie_type = zombie_type
        self.health = 50 if zombie_type == "common" else 500
        self.alive = True

        # Colors based on type
        self.color = {
            "common": (100, 80, 60),
            "hunter": PURPLE,
            "smoker": (100, 150, 100),
            "boomer": (150, 150, 50),
            "tank": (150, 50, 50),
            "witch": (200, 200, 200),
        }.get(zombie_type, RED)

        self.size = 20 if zombie_type == "tank" else (12 if zombie_type == "common" else 15)

    def draw(self, screen, offset_x=0, offset_y=0):
        if not self.alive:
            return

        sx = GAME_AREA.x + (self.x * MAP_SCALE) + offset_x
        sy = GAME_AREA.y + (self.y * MAP_SCALE) + offset_y

        # Keep in bounds
        sx = max(GAME_AREA.left, min(GAME_AREA.right, sx))
        sy = max(GAME_AREA.top, min(GAME_AREA.bottom, sy))

        pygame.draw.circle(screen, self.color, (int(sx), int(sy)), self.size)

        if self.zombie_type != "common":
            font = pygame.font.Font(None, 14)
            text = font.render(self.zombie_type[0].upper(), True, WHITE)
            screen.blit(text, (sx - 4, sy - 5))


class SafeRoom:
    """Visual representation of the safe room goal."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, screen, offset_x=0, offset_y=0):
        sx = GAME_AREA.x + (self.x * MAP_SCALE) + offset_x
        sy = GAME_AREA.y + (self.y * MAP_SCALE) + offset_y

        # Draw safe room
        pygame.draw.rect(screen, GREEN, (sx - 30, sy - 30, 60, 60), 3)
        font = pygame.font.Font(None, 20)
        text = font.render("SAFE", True, GREEN)
        screen.blit(text, (sx - 15, sy - 8))


class VisualDemo:
    """Main visual demo class."""

    def __init__(self, personality="aggressive", speed=1.0, director_mode="standard"):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(f"L4D2 AI Visual Demo - {personality}")
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        self.speed = speed
        self.frame = 0

        self.personality = personality
        self.director_mode = director_mode

        # Load the trained model
        self.model = self.load_model(personality)

        # Game state
        self.reset_game()

        # Stats
        self.total_kills = 0
        self.episodes = 0
        self.episode_reward = 0

        # Camera offset (for scrolling)
        self.camera_x = 0
        self.camera_y = 0

    def load_model(self, personality):
        """Load trained PPO model."""
        try:
            from stable_baselines3 import PPO

            # Find model
            rl_agents = PROJECT_ROOT / "model_adapters" / "rl_agents"

            # Check individual training first
            for d in sorted(rl_agents.glob(f"ppo_{personality}_*"), reverse=True):
                model_path = d / "final_model.zip"
                if model_path.exists():
                    print(f"Loading model: {model_path}")
                    return PPO.load(str(model_path))

            # Check batch training
            for d in sorted(rl_agents.glob("all_personalities_*"), reverse=True):
                model_path = d / personality / "final_model.zip"
                if model_path.exists():
                    print(f"Loading model: {model_path}")
                    return PPO.load(str(model_path))

            print(f"Warning: No trained model found for {personality}, using random actions")
            return None

        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def reset_game(self):
        """Reset the game state."""
        # Player (AI-controlled survivor)
        self.player = Survivor(400, 300, is_player=True, name="AI Bot")

        # Teammates
        self.teammates = [
            Survivor(350, 280, name="Coach"),
            Survivor(450, 280, name="Ellis"),
            Survivor(400, 350, name="Rochelle"),
        ]

        # Zombies
        self.zombies = []
        self.spawn_zombies(20)

        # Safe room (goal)
        self.safe_room = SafeRoom(2500, 300)

        # State
        self.game_time = 0
        self.episode_reward = 0
        self.distance_to_safe = 5000

    def spawn_zombies(self, count, near_player=False):
        """Spawn zombies."""
        for _ in range(count):
            if near_player:
                x = self.player.x + random.uniform(-300, 300)
                y = self.player.y + random.uniform(-300, 300)
            else:
                x = random.uniform(0, 3000)
                y = random.uniform(0, 600)
            self.zombies.append(Zombie(x, y, "common"))

    def get_observation(self):
        """Build observation vector for the AI."""
        # 20D observation matching EnhancedL4D2Env
        obs = np.zeros(20, dtype=np.float32)

        # Basic state
        obs[0] = self.player.health / 100  # health
        obs[1] = 1.0 if self.player.alive else 0.0  # alive
        obs[2] = 1.0 if self.player.incapped else 0.0  # incapped

        # Position (normalized)
        obs[3] = self.player.x / 5000
        obs[4] = self.player.y / 600
        obs[5] = 0.0  # z

        # Velocity (simplified)
        obs[6] = 0.0
        obs[7] = 0.0
        obs[8] = 0.0

        # Angles
        obs[9] = self.player.angle / 360
        obs[10] = 0.0

        # Weapon/ammo
        obs[11] = 1.0  # has weapon
        obs[12] = 0.5  # ammo

        # Nearby enemies
        nearby = sum(1 for z in self.zombies if z.alive and
                     abs(z.x - self.player.x) < 200 and abs(z.y - self.player.y) < 200)
        obs[13] = min(nearby / 20, 1.0)

        # Nearest enemy distance
        min_dist = 9999
        for z in self.zombies:
            if z.alive:
                dist = math.sqrt((z.x - self.player.x)**2 + (z.y - self.player.y)**2)
                min_dist = min(min_dist, dist)
        obs[14] = min(min_dist / 500, 1.0)

        # Team
        alive_teammates = sum(1 for t in self.teammates if t.alive)
        obs[15] = 0.0  # nearest teammate dist
        obs[16] = alive_teammates / 3
        obs[17] = 0.0  # incapped teammates

        # Objective
        obs[18] = 1.0 if self.distance_to_safe < 100 else 0.0  # in safe room
        obs[19] = max(0, 1.0 - self.distance_to_safe / 5000)  # progress

        return obs

    def step_ai(self):
        """Get and execute AI action."""
        obs = self.get_observation()

        # Get action from model
        if self.model:
            action, _ = self.model.predict(obs, deterministic=True)
        else:
            action = random.randint(0, 13)

        # Execute action
        self.execute_action(int(action))
        self.player.last_action = ACTION_NAMES[int(action)]

        return action

    def execute_action(self, action):
        """Execute the AI's chosen action."""
        speed = 5 * self.speed

        if action == 1:  # Forward
            self.player.x += speed * math.cos(math.radians(self.player.angle))
            self.player.y += speed * math.sin(math.radians(self.player.angle))
        elif action == 2:  # Backward
            self.player.x -= speed * math.cos(math.radians(self.player.angle))
            self.player.y -= speed * math.sin(math.radians(self.player.angle))
        elif action == 3:  # Left
            self.player.angle -= 5
        elif action == 4:  # Right
            self.player.angle += 5
        elif action == 5:  # Attack
            self.attack()
        elif action == 6:  # Use
            pass
        elif action == 11:  # Heal self
            if self.player.health < 100:
                self.player.health = min(100, self.player.health + 20)
                self.episode_reward += 2

        # Keep in bounds
        self.player.x = max(0, min(5000, self.player.x))
        self.player.y = max(0, min(600, self.player.y))

    def attack(self):
        """Attack nearby zombies."""
        attack_range = 100
        for z in self.zombies:
            if z.alive:
                dist = math.sqrt((z.x - self.player.x)**2 + (z.y - self.player.y)**2)
                if dist < attack_range:
                    z.health -= 30
                    if z.health <= 0:
                        z.alive = False
                        self.player.kills += 1
                        self.total_kills += 1
                        self.episode_reward += 3
                    break

    def update_zombies(self):
        """Update zombie behavior."""
        for z in self.zombies:
            if not z.alive:
                continue

            # Move toward player
            dx = self.player.x - z.x
            dy = self.player.y - z.y
            dist = math.sqrt(dx*dx + dy*dy)

            if dist > 0 and dist < 500:
                speed = 2 * self.speed
                z.x += (dx / dist) * speed
                z.y += (dy / dist) * speed

                # Attack player
                if dist < 30:
                    self.player.health -= 0.5 * self.speed
                    if self.player.health <= 0:
                        self.player.alive = False

        # Spawn more zombies periodically
        if self.frame % (60 * 3) == 0:  # Every 3 seconds
            self.spawn_zombies(5, near_player=True)

    def update_camera(self):
        """Update camera to follow player."""
        target_x = self.player.x * MAP_SCALE - GAME_AREA.width / 2
        target_y = self.player.y * MAP_SCALE - GAME_AREA.height / 2

        # Smooth camera
        self.camera_x += (target_x - self.camera_x) * 0.1
        self.camera_y += (target_y - self.camera_y) * 0.1

    def update(self):
        """Update game state."""
        if self.paused or not self.player.alive:
            return

        self.frame += 1
        self.game_time += 1/60 * self.speed

        # AI step
        self.step_ai()

        # Update zombies
        self.update_zombies()

        # Update camera
        self.update_camera()

        # Update distance to safe room
        self.distance_to_safe = math.sqrt(
            (self.safe_room.x - self.player.x)**2 +
            (self.safe_room.y - self.player.y)**2
        )

        # Check win condition
        if self.distance_to_safe < 100:
            self.episode_reward += 100
            self.episodes += 1
            self.reset_game()

        # Survival reward
        self.episode_reward += 0.01 * self.speed

    def draw_hud(self):
        """Draw heads-up display."""
        font = pygame.font.Font(None, 24)
        large_font = pygame.font.Font(None, 36)

        # Title
        title = large_font.render(f"L4D2 AI Demo - {self.personality.upper()}", True, CYAN)
        self.screen.blit(title, (GAME_AREA.x, 15))

        # Stats panel (right side)
        panel_x = GAME_AREA.right + 20
        panel_y = GAME_AREA.top

        # Background
        pygame.draw.rect(self.screen, DARK_GRAY, (panel_x, panel_y, 300, 400), border_radius=10)
        pygame.draw.rect(self.screen, GRAY, (panel_x, panel_y, 300, 400), 2, border_radius=10)

        # Stats
        stats = [
            ("Personality", self.personality.upper()),
            ("", ""),
            ("Health", f"{int(self.player.health)}%"),
            ("Kills", str(self.player.kills)),
            ("Episode Reward", f"{self.episode_reward:.1f}"),
            ("", ""),
            ("Distance to Safe", f"{int(self.distance_to_safe)}m"),
            ("Zombies Alive", str(sum(1 for z in self.zombies if z.alive))),
            ("", ""),
            ("Current Action", self.player.last_action),
            ("Game Time", f"{self.game_time:.1f}s"),
            ("Episodes", str(self.episodes)),
            ("Total Kills", str(self.total_kills)),
        ]

        y = panel_y + 15
        for label, value in stats:
            if label:
                text = font.render(f"{label}:", True, GRAY if label == "" else WHITE)
                self.screen.blit(text, (panel_x + 15, y))
                value_text = font.render(str(value), True, CYAN)
                self.screen.blit(value_text, (panel_x + 180, y))
            y += 25

        # Controls
        controls_y = panel_y + 420
        pygame.draw.rect(self.screen, DARK_GRAY, (panel_x, controls_y, 300, 180), border_radius=10)

        controls = [
            "CONTROLS:",
            "SPACE - Pause/Resume",
            "R - Reset Episode",
            "1-5 - Switch Personality",
            "D - Toggle Director",
            "ESC/Q - Quit",
        ]

        for i, text in enumerate(controls):
            color = YELLOW if i == 0 else GRAY
            t = font.render(text, True, color)
            self.screen.blit(t, (panel_x + 15, controls_y + 15 + i * 25))

        # Status bar
        status = "PAUSED" if self.paused else ("DEAD - Press R" if not self.player.alive else "RUNNING")
        status_color = YELLOW if self.paused else (RED if not self.player.alive else GREEN)
        status_text = font.render(status, True, status_color)
        self.screen.blit(status_text, (GAME_AREA.x + GAME_AREA.width - 100, 15))

    def draw_minimap(self):
        """Draw minimap."""
        mm_x = GAME_AREA.right + 20
        mm_y = GAME_AREA.bottom - 150
        mm_width = 300
        mm_height = 100

        pygame.draw.rect(self.screen, DARK_GRAY, (mm_x, mm_y, mm_width, mm_height), border_radius=5)
        pygame.draw.rect(self.screen, GRAY, (mm_x, mm_y, mm_width, mm_height), 1, border_radius=5)

        # Scale for minimap
        mm_scale = mm_width / 5000

        # Draw safe room
        sx = mm_x + self.safe_room.x * mm_scale
        sy = mm_y + mm_height / 2
        pygame.draw.rect(self.screen, GREEN, (sx - 3, sy - 3, 6, 6))

        # Draw zombies
        for z in self.zombies:
            if z.alive:
                zx = mm_x + z.x * mm_scale
                zy = mm_y + (z.y / 600) * mm_height
                pygame.draw.circle(self.screen, RED, (int(zx), int(zy)), 1)

        # Draw player
        px = mm_x + self.player.x * mm_scale
        py = mm_y + (self.player.y / 600) * mm_height
        pygame.draw.circle(self.screen, CYAN, (int(px), int(py)), 3)

        # Label
        font = pygame.font.Font(None, 18)
        text = font.render("MINIMAP", True, GRAY)
        self.screen.blit(text, (mm_x + 5, mm_y + 5))

    def draw(self):
        """Draw everything."""
        self.screen.fill(BLACK)

        # Draw game area background
        pygame.draw.rect(self.screen, DARK_GRAY, GAME_AREA)
        pygame.draw.rect(self.screen, GRAY, GAME_AREA, 2)

        # Calculate offset for camera
        offset_x = -self.camera_x
        offset_y = -self.camera_y

        # Draw safe room
        self.safe_room.draw(self.screen, offset_x, offset_y)

        # Draw zombies
        for z in self.zombies:
            z.draw(self.screen, offset_x, offset_y)

        # Draw teammates
        for t in self.teammates:
            t.draw(self.screen, offset_x, offset_y)

        # Draw player
        self.player.draw(self.screen, offset_x, offset_y)

        # Draw HUD
        self.draw_hud()
        self.draw_minimap()

        pygame.display.flip()

    def handle_events(self):
        """Handle input events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.reset_game()
                elif event.key == pygame.K_1:
                    self.switch_personality("aggressive")
                elif event.key == pygame.K_2:
                    self.switch_personality("balanced")
                elif event.key == pygame.K_3:
                    self.switch_personality("defender")
                elif event.key == pygame.K_4:
                    self.switch_personality("medic")
                elif event.key == pygame.K_5:
                    self.switch_personality("speedrunner")

    def switch_personality(self, personality):
        """Switch to a different personality."""
        self.personality = personality
        self.model = self.load_model(personality)
        pygame.display.set_caption(f"L4D2 AI Visual Demo - {personality}")
        print(f"Switched to {personality} personality")

    def run(self):
        """Main game loop."""
        print("\n" + "=" * 50)
        print("L4D2 AI VISUAL DEMO")
        print("=" * 50)
        print(f"Personality: {self.personality}")
        print("Press SPACE to pause, R to reset, 1-5 to switch personality")
        print("=" * 50 + "\n")

        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="L4D2 AI Visual Demo")
    parser.add_argument("--personality", "-p", default="aggressive",
                        choices=["aggressive", "balanced", "defender", "medic", "speedrunner"],
                        help="Bot personality to use")
    parser.add_argument("--speed", "-s", type=float, default=1.0,
                        help="Game speed multiplier")
    parser.add_argument("--director", "-d", default="standard",
                        choices=["standard", "relaxed", "intense", "nightmare"],
                        help="Director mode")

    args = parser.parse_args()

    demo = VisualDemo(
        personality=args.personality,
        speed=args.speed,
        director_mode=args.director
    )
    demo.run()


if __name__ == "__main__":
    main()
