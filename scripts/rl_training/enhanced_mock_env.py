#!/usr/bin/env python3
"""
Enhanced Mock Environment for L4D2 RL Training

A self-contained simulation environment that provides realistic game mechanics
for training PPO agents without requiring a live L4D2 server.

Features:
1. Realistic state transitions (physics-based movement)
2. Zombie AI behavior (approach player, attack when close)
3. Map progress simulation (distance to saferoom)
4. Item pickup mechanics (medkits, pills, weapons)
5. Teammate simulation (basic follow/support behavior)
6. Special infected patterns (smoker, boomer, hunter, tank)

Usage:
    from enhanced_mock_env import EnhancedL4D2Env

    env = EnhancedL4D2Env()
    obs, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field
from enum import IntEnum
import logging
import math

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BotAction(IntEnum):
    """Discrete action space for L4D2 bots."""
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


class ZombieType(IntEnum):
    """Types of infected in L4D2."""
    COMMON = 0
    SMOKER = 1
    BOOMER = 2
    HUNTER = 3
    SPITTER = 4
    CHARGER = 5
    JOCKEY = 6
    TANK = 7


class ItemType(IntEnum):
    """Pickup item types."""
    NONE = 0
    MEDKIT = 1
    PILLS = 2
    ADRENALINE = 3
    PIPE_BOMB = 4
    MOLOTOV = 5
    BILE_BOMB = 6
    AMMO_PILE = 7
    PRIMARY_WEAPON = 8


@dataclass
class Vector3:
    """Simple 3D vector with utility methods."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> 'Vector3':
        mag = self.magnitude()
        if mag < 0.001:
            return Vector3(0, 0, 0)
        return Vector3(self.x / mag, self.y / mag, self.z / mag)

    def distance_to(self, other: 'Vector3') -> float:
        return (self - other).magnitude()

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class Zombie:
    """Represents an infected entity."""
    zombie_type: ZombieType
    position: Vector3
    health: int
    is_alive: bool = True
    is_attacking: bool = False
    target_id: int = 0  # Player/teammate being targeted
    special_state: int = 0  # For special infected abilities
    cooldown: float = 0.0  # Ability cooldown

    @property
    def damage(self) -> int:
        """Damage per attack based on zombie type."""
        damages = {
            ZombieType.COMMON: 5,
            ZombieType.SMOKER: 3,  # Pull damage per tick
            ZombieType.BOOMER: 0,  # No direct damage, but attracts horde
            ZombieType.HUNTER: 10,  # Pounce damage per tick
            ZombieType.SPITTER: 8,  # Acid damage
            ZombieType.CHARGER: 15,  # Charge + pound
            ZombieType.JOCKEY: 4,  # Ride damage per tick
            ZombieType.TANK: 25,  # Heavy punch
        }
        return damages.get(self.zombie_type, 5)

    @property
    def speed(self) -> float:
        """Movement speed based on zombie type."""
        speeds = {
            ZombieType.COMMON: 150.0,
            ZombieType.SMOKER: 100.0,
            ZombieType.BOOMER: 80.0,
            ZombieType.HUNTER: 200.0,
            ZombieType.SPITTER: 120.0,
            ZombieType.CHARGER: 180.0,
            ZombieType.JOCKEY: 180.0,
            ZombieType.TANK: 120.0,
        }
        return speeds.get(self.zombie_type, 150.0)

    @property
    def attack_range(self) -> float:
        """Attack range based on zombie type."""
        ranges = {
            ZombieType.COMMON: 50.0,
            ZombieType.SMOKER: 750.0,  # Tongue range
            ZombieType.BOOMER: 200.0,  # Bile range
            ZombieType.HUNTER: 500.0,  # Pounce range
            ZombieType.SPITTER: 400.0,  # Spit range
            ZombieType.CHARGER: 100.0,  # Charge initiation
            ZombieType.JOCKEY: 100.0,  # Jump range
            ZombieType.TANK: 80.0,  # Punch range
        }
        return ranges.get(self.zombie_type, 50.0)


@dataclass
class Teammate:
    """Represents an AI teammate survivor."""
    teammate_id: int
    position: Vector3
    health: int = 100
    is_alive: bool = True
    is_incapped: bool = False
    has_medkit: bool = True
    is_being_attacked: bool = False
    incap_timer: float = 0.0  # Time until death if incapped

    def update(self, player_pos: Vector3, dt: float):
        """Update teammate behavior - follow player loosely."""
        if not self.is_alive:
            return

        # Follow player at a distance
        ideal_distance = 150.0 + self.teammate_id * 50  # Spread out
        current_dist = self.position.distance_to(player_pos)

        if current_dist > ideal_distance + 100:
            # Move toward player
            direction = (player_pos - self.position).normalized()
            move_speed = 200.0 * dt
            self.position = self.position + direction * move_speed
        elif current_dist < ideal_distance - 50:
            # Move away slightly
            direction = (self.position - player_pos).normalized()
            move_speed = 100.0 * dt
            self.position = self.position + direction * move_speed

        # Update incap timer
        if self.is_incapped:
            self.incap_timer -= dt
            if self.incap_timer <= 0:
                self.is_alive = False


@dataclass
class ItemPickup:
    """Represents a pickup item in the world."""
    item_type: ItemType
    position: Vector3
    is_available: bool = True


@dataclass
class MapState:
    """Represents map/level state."""
    total_distance: float = 5000.0  # Distance to saferoom
    current_progress: float = 0.0  # How far player has progressed
    saferoom_position: Vector3 = field(default_factory=lambda: Vector3(5000, 0, 0))
    checkpoints: List[float] = field(default_factory=lambda: [1000, 2000, 3000, 4000])
    checkpoints_reached: List[bool] = field(default_factory=lambda: [False, False, False, False])

    @property
    def progress_percent(self) -> float:
        return min(1.0, self.current_progress / self.total_distance)

    @property
    def distance_to_saferoom(self) -> float:
        return max(0, self.total_distance - self.current_progress)


@dataclass
class PlayerState:
    """Full player state."""
    position: Vector3 = field(default_factory=Vector3)
    velocity: Vector3 = field(default_factory=Vector3)
    angle: Tuple[float, float] = (0.0, 0.0)  # pitch, yaw
    health: int = 100
    is_alive: bool = True
    is_incapped: bool = False

    # Inventory
    primary_weapon: int = 1  # 0=none, 1=smg, 2=shotgun, etc.
    secondary_weapon: int = 1  # Pistol
    throwable: int = 0  # 0=none, 1=pipe, 2=molotov, 3=bile
    health_item: int = 1  # 0=none, 1=medkit, 2=pills, 3=adrenaline
    ammo: int = 100

    # Status effects
    is_grabbed: bool = False  # By smoker/hunter/etc.
    grabbed_by: Optional[ZombieType] = None
    boomer_blind_timer: float = 0.0
    incap_count: int = 0  # Number of times incapped this level

    def take_damage(self, amount: int) -> bool:
        """Apply damage and return True if died/incapped."""
        if not self.is_alive:
            return False

        self.health -= amount

        if self.health <= 0:
            if self.is_incapped or self.incap_count >= 2:
                # Already incapped or too many incaps = death
                self.is_alive = False
                return True
            else:
                # First incap
                self.is_incapped = True
                self.health = 300  # Incap health pool
                self.incap_count += 1
                return True
        return False

    def heal(self, amount: int, full_heal: bool = False):
        """Heal the player."""
        if full_heal:
            self.is_incapped = False
            self.health = min(100, self.health + amount)
        else:
            self.health = min(100 if not self.is_incapped else 300, self.health + amount)


class EnhancedL4D2Env(gym.Env):
    """
    Enhanced mock environment for L4D2 RL training.

    Provides realistic game mechanics simulation including:
    - Physics-based movement
    - Zombie AI with different behaviors per type
    - Map progression toward saferoom
    - Item pickups and usage
    - Teammate AI
    - Special infected abilities
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # Game constants
    PLAYER_SPEED = 220.0  # Units per second
    ATTACK_RANGE = 100.0
    ATTACK_DAMAGE = 20
    SHOVE_RANGE = 75.0
    SHOVE_COOLDOWN = 0.7
    HEAL_TIME = 5.0
    USE_RANGE = 100.0

    # Spawn rates (per second)
    COMMON_SPAWN_RATE = 0.5
    SPECIAL_SPAWN_RATE = 0.02
    HORDE_CHANCE = 0.002  # Chance per step of horde event

    def __init__(
        self,
        max_episode_steps: int = 5000,
        render_mode: Optional[str] = None,
        difficulty: str = "normal",  # easy, normal, hard, expert
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.difficulty = difficulty

        # Difficulty multipliers
        self.difficulty_settings = {
            "easy": {"damage_mult": 0.5, "spawn_mult": 0.5, "special_mult": 0.3},
            "normal": {"damage_mult": 1.0, "spawn_mult": 1.0, "special_mult": 1.0},
            "hard": {"damage_mult": 1.5, "spawn_mult": 1.5, "special_mult": 1.5},
            "expert": {"damage_mult": 2.0, "spawn_mult": 2.0, "special_mult": 2.0},
        }
        self.diff = self.difficulty_settings.get(difficulty, self.difficulty_settings["normal"])

        # Initialize random state
        self.np_random = np.random.default_rng(seed)

        # Simulation time step
        self.dt = 1.0 / 30.0  # 30 FPS simulation

        # Define spaces (same as original for compatibility)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(20,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(BotAction))

        # Reward configuration
        self.reward_config = {
            "kill": 1.0,
            "kill_special": 5.0,
            "damage_dealt": 0.1,
            "damage_taken": -0.1,
            "heal_teammate": 5.0,
            "heal_self": 2.0,
            "incapped": -10.0,
            "death": -50.0,
            "safe_room": 100.0,
            "survival": 0.01,
            "proximity_to_team": 0.001,
            "progress": 0.05,  # Reward for map progress
            "checkpoint": 10.0,  # Bonus for reaching checkpoint
            "item_pickup": 1.0,
        }

        # State variables (initialized in reset)
        self.current_step = 0
        self.episode_reward = 0.0
        self.player: Optional[PlayerState] = None
        self.zombies: List[Zombie] = []
        self.teammates: List[Teammate] = []
        self.items: List[ItemPickup] = []
        self.map_state: Optional[MapState] = None

        # Action cooldowns
        self.shove_cooldown = 0.0
        self.heal_progress = 0.0
        self.is_healing = False

        # Statistics
        self.stats = {
            "kills": 0,
            "special_kills": 0,
            "damage_dealt": 0,
            "damage_taken": 0,
            "items_used": 0,
            "teammates_healed": 0,
            "distance_traveled": 0.0,
            "checkpoints_reached": 0,
        }

    def _spawn_zombies(self):
        """Spawn zombies based on game state."""
        # Common infected spawn
        if self.np_random.random() < self.COMMON_SPAWN_RATE * self.diff["spawn_mult"] * self.dt:
            spawn_count = self.np_random.integers(1, 4)
            for _ in range(spawn_count):
                # Spawn ahead of player, to the sides
                offset = Vector3(
                    self.np_random.uniform(200, 500),
                    self.np_random.uniform(-300, 300),
                    0
                )
                spawn_pos = self.player.position + offset
                self.zombies.append(Zombie(
                    zombie_type=ZombieType.COMMON,
                    position=spawn_pos,
                    health=50,
                ))

        # Special infected spawn (less frequent)
        if self.np_random.random() < self.SPECIAL_SPAWN_RATE * self.diff["special_mult"] * self.dt:
            special_types = [
                ZombieType.SMOKER,
                ZombieType.BOOMER,
                ZombieType.HUNTER,
                ZombieType.SPITTER,
                ZombieType.CHARGER,
                ZombieType.JOCKEY,
            ]
            # Very rare tank spawn
            if self.np_random.random() < 0.01 and self.map_state.progress_percent > 0.5:
                special_types.append(ZombieType.TANK)

            zombie_type = self.np_random.choice(special_types)
            spawn_pos = self.player.position + Vector3(
                self.np_random.uniform(300, 600),
                self.np_random.uniform(-400, 400),
                0
            )

            health_map = {
                ZombieType.SMOKER: 250,
                ZombieType.BOOMER: 50,
                ZombieType.HUNTER: 250,
                ZombieType.SPITTER: 100,
                ZombieType.CHARGER: 600,
                ZombieType.JOCKEY: 325,
                ZombieType.TANK: 6000,
            }

            self.zombies.append(Zombie(
                zombie_type=zombie_type,
                position=spawn_pos,
                health=health_map.get(zombie_type, 250),
            ))

        # Horde event
        if self.np_random.random() < self.HORDE_CHANCE:
            self._spawn_horde()

    def _spawn_horde(self, size: int = 20):
        """Spawn a horde of common infected."""
        for _ in range(size):
            angle = self.np_random.uniform(0, 2 * math.pi)
            distance = self.np_random.uniform(300, 600)
            offset = Vector3(
                math.cos(angle) * distance,
                math.sin(angle) * distance,
                0
            )
            self.zombies.append(Zombie(
                zombie_type=ZombieType.COMMON,
                position=self.player.position + offset,
                health=50,
            ))

    def _spawn_items(self):
        """Spawn item pickups along the path."""
        # Spawn items ahead of player occasionally
        if self.np_random.random() < 0.01 * self.dt:
            item_types = [
                ItemType.MEDKIT,
                ItemType.PILLS,
                ItemType.PIPE_BOMB,
                ItemType.MOLOTOV,
                ItemType.AMMO_PILE,
            ]
            weights = [0.15, 0.25, 0.2, 0.2, 0.2]
            item_type = self.np_random.choice(item_types, p=weights)

            spawn_pos = self.player.position + Vector3(
                self.np_random.uniform(100, 400),
                self.np_random.uniform(-200, 200),
                0
            )
            self.items.append(ItemPickup(
                item_type=item_type,
                position=spawn_pos,
            ))

    def _update_zombies(self):
        """Update zombie AI behavior."""
        for zombie in self.zombies:
            if not zombie.is_alive:
                continue

            zombie.cooldown = max(0, zombie.cooldown - self.dt)

            # Find nearest target (player or teammate)
            targets = [(self.player.position, 0, self.player.is_alive and not self.player.is_grabbed)]
            for i, tm in enumerate(self.teammates):
                targets.append((tm.position, i + 1, tm.is_alive and not tm.is_being_attacked))

            # Filter valid targets and find nearest
            valid_targets = [(pos, tid) for pos, tid, valid in targets if valid]
            if not valid_targets:
                continue

            nearest_pos, target_id = min(valid_targets, key=lambda t: zombie.position.distance_to(t[0]))
            zombie.target_id = target_id
            distance = zombie.position.distance_to(nearest_pos)

            # Behavior based on zombie type
            if zombie.zombie_type == ZombieType.COMMON:
                self._update_common_zombie(zombie, nearest_pos, distance)
            elif zombie.zombie_type == ZombieType.SMOKER:
                self._update_smoker(zombie, nearest_pos, distance)
            elif zombie.zombie_type == ZombieType.BOOMER:
                self._update_boomer(zombie, nearest_pos, distance)
            elif zombie.zombie_type == ZombieType.HUNTER:
                self._update_hunter(zombie, nearest_pos, distance)
            elif zombie.zombie_type == ZombieType.TANK:
                self._update_tank(zombie, nearest_pos, distance)
            else:
                # Generic special infected behavior
                self._update_common_zombie(zombie, nearest_pos, distance)

    def _update_common_zombie(self, zombie: Zombie, target_pos: Vector3, distance: float):
        """Common infected: approach and attack."""
        if distance <= zombie.attack_range:
            zombie.is_attacking = True
            self._apply_zombie_damage(zombie)
        else:
            zombie.is_attacking = False
            # Move toward target
            direction = (target_pos - zombie.position).normalized()
            zombie.position = zombie.position + direction * (zombie.speed * self.dt)

    def _update_smoker(self, zombie: Zombie, target_pos: Vector3, distance: float):
        """Smoker: tongue pull from distance."""
        if zombie.special_state == 1:  # Currently pulling
            # Apply pull damage
            if zombie.target_id == 0 and self.player.is_grabbed:
                self.player.take_damage(int(zombie.damage * self.diff["damage_mult"]))
                self.stats["damage_taken"] += zombie.damage
                # Pull player toward smoker
                direction = (zombie.position - self.player.position).normalized()
                self.player.position = self.player.position + direction * (50 * self.dt)
        elif distance <= zombie.attack_range and zombie.cooldown <= 0:
            # Initiate tongue attack
            if zombie.target_id == 0 and not self.player.is_grabbed:
                self.player.is_grabbed = True
                self.player.grabbed_by = ZombieType.SMOKER
                zombie.special_state = 1
        else:
            # Move closer but keep distance
            if distance > 400:
                direction = (target_pos - zombie.position).normalized()
                zombie.position = zombie.position + direction * (zombie.speed * self.dt)

    def _update_boomer(self, zombie: Zombie, target_pos: Vector3, distance: float):
        """Boomer: approach and explode/bile."""
        if distance <= zombie.attack_range and zombie.cooldown <= 0:
            # Bile attack (blind and attract horde)
            if zombie.target_id == 0:
                self.player.boomer_blind_timer = 5.0  # 5 seconds of blindness
            # Spawn mini-horde attracted to biled player
            self._spawn_horde(size=10)
            zombie.cooldown = 30.0  # Long cooldown
        else:
            direction = (target_pos - zombie.position).normalized()
            zombie.position = zombie.position + direction * (zombie.speed * self.dt)

    def _update_hunter(self, zombie: Zombie, target_pos: Vector3, distance: float):
        """Hunter: pounce from distance."""
        if zombie.special_state == 1:  # Currently pouncing
            if zombie.target_id == 0 and self.player.is_grabbed:
                self.player.take_damage(int(zombie.damage * self.diff["damage_mult"]))
                self.stats["damage_taken"] += zombie.damage
        elif distance <= zombie.attack_range and distance > 100 and zombie.cooldown <= 0:
            # Initiate pounce
            if zombie.target_id == 0 and not self.player.is_grabbed:
                self.player.is_grabbed = True
                self.player.grabbed_by = ZombieType.HUNTER
                zombie.position = target_pos  # Jump to target
                zombie.special_state = 1
        else:
            # Crouch and stalk
            if distance > zombie.attack_range:
                direction = (target_pos - zombie.position).normalized()
                zombie.position = zombie.position + direction * (zombie.speed * self.dt)

    def _update_tank(self, zombie: Zombie, target_pos: Vector3, distance: float):
        """Tank: relentless pursuit and heavy hits."""
        if distance <= zombie.attack_range and zombie.cooldown <= 0:
            self._apply_zombie_damage(zombie)
            zombie.cooldown = 1.5  # Slower attack rate
        else:
            direction = (target_pos - zombie.position).normalized()
            zombie.position = zombie.position + direction * (zombie.speed * self.dt)

    def _apply_zombie_damage(self, zombie: Zombie):
        """Apply damage from zombie to its target."""
        damage = int(zombie.damage * self.diff["damage_mult"])

        if zombie.target_id == 0:
            self.player.take_damage(damage)
            self.stats["damage_taken"] += damage
        else:
            tm_idx = zombie.target_id - 1
            if tm_idx < len(self.teammates):
                tm = self.teammates[tm_idx]
                tm.health -= damage
                if tm.health <= 0:
                    if tm.is_incapped:
                        tm.is_alive = False
                    else:
                        tm.is_incapped = True
                        tm.health = 300
                        tm.incap_timer = 60.0  # 60 seconds to revive

    def _update_teammates(self):
        """Update teammate AI."""
        for tm in self.teammates:
            tm.update(self.player.position, self.dt)

            # Teammates can kill zombies occasionally
            if tm.is_alive and not tm.is_incapped:
                for zombie in self.zombies:
                    if not zombie.is_alive:
                        continue
                    dist = tm.position.distance_to(zombie.position)
                    if dist < 150 and self.np_random.random() < 0.1 * self.dt:
                        zombie.health -= 30
                        if zombie.health <= 0:
                            zombie.is_alive = False

    def _process_action(self, action: int) -> float:
        """Process player action and return immediate reward."""
        reward = 0.0

        if action == BotAction.IDLE:
            pass

        elif action == BotAction.MOVE_FORWARD:
            self._move_player(1, 0)

        elif action == BotAction.MOVE_BACKWARD:
            self._move_player(-0.5, 0)  # Slower backward

        elif action == BotAction.MOVE_LEFT:
            self._move_player(0, -1)

        elif action == BotAction.MOVE_RIGHT:
            self._move_player(0, 1)

        elif action == BotAction.ATTACK:
            reward += self._do_attack()

        elif action == BotAction.USE:
            reward += self._do_use()

        elif action == BotAction.RELOAD:
            self.player.ammo = min(100, self.player.ammo + 30)

        elif action == BotAction.CROUCH:
            pass  # Could add crouch mechanics

        elif action == BotAction.JUMP:
            pass  # Could add jump mechanics

        elif action == BotAction.SHOVE:
            reward += self._do_shove()

        elif action == BotAction.HEAL_SELF:
            reward += self._do_heal_self()

        elif action == BotAction.HEAL_OTHER:
            reward += self._do_heal_other()

        elif action == BotAction.THROW_ITEM:
            reward += self._do_throw()

        return reward

    def _move_player(self, forward: float, right: float):
        """Move the player in the given direction."""
        if self.player.is_incapped or self.player.is_grabbed:
            return

        # Calculate movement
        move_speed = self.PLAYER_SPEED * self.dt

        # Simplified: forward is +X, right is +Y
        delta = Vector3(forward * move_speed, right * move_speed, 0)

        old_pos = self.player.position
        self.player.position = self.player.position + delta

        # Update map progress
        progress_delta = max(0, self.player.position.x - old_pos.x)
        self.map_state.current_progress += progress_delta
        self.stats["distance_traveled"] += delta.magnitude()

        # Check checkpoints
        for i, checkpoint in enumerate(self.map_state.checkpoints):
            if not self.map_state.checkpoints_reached[i] and self.map_state.current_progress >= checkpoint:
                self.map_state.checkpoints_reached[i] = True
                self.stats["checkpoints_reached"] += 1

    def _do_attack(self) -> float:
        """Perform attack action."""
        reward = 0.0

        if self.player.ammo <= 0:
            return 0.0

        self.player.ammo -= 1

        # Find zombies in attack range
        for zombie in self.zombies:
            if not zombie.is_alive:
                continue

            dist = self.player.position.distance_to(zombie.position)
            if dist <= self.ATTACK_RANGE:
                # Hit!
                damage = self.ATTACK_DAMAGE
                zombie.health -= damage
                reward += self.reward_config["damage_dealt"]
                self.stats["damage_dealt"] += damage

                if zombie.health <= 0:
                    zombie.is_alive = False

                    # Free player if this zombie was grabbing them
                    if self.player.is_grabbed and self.player.grabbed_by == zombie.zombie_type:
                        self.player.is_grabbed = False
                        self.player.grabbed_by = None

                    if zombie.zombie_type == ZombieType.COMMON:
                        reward += self.reward_config["kill"]
                        self.stats["kills"] += 1
                    else:
                        reward += self.reward_config["kill_special"]
                        self.stats["special_kills"] += 1

                break  # Only hit one zombie per attack

        return reward

    def _do_shove(self) -> float:
        """Perform shove/melee action."""
        if self.shove_cooldown > 0:
            return 0.0

        self.shove_cooldown = self.SHOVE_COOLDOWN
        reward = 0.0

        # Shove pushes zombies back and can free grabbed player
        for zombie in self.zombies:
            if not zombie.is_alive:
                continue

            dist = self.player.position.distance_to(zombie.position)
            if dist <= self.SHOVE_RANGE:
                # Push zombie back
                direction = (zombie.position - self.player.position).normalized()
                zombie.position = zombie.position + direction * 100

                # Cancel special infected abilities
                if zombie.special_state > 0:
                    zombie.special_state = 0
                    if self.player.grabbed_by == zombie.zombie_type:
                        self.player.is_grabbed = False
                        self.player.grabbed_by = None
                        reward += 1.0  # Reward for self-rescue

        return reward

    def _do_use(self) -> float:
        """Use/interact action - pick up items, revive teammates."""
        reward = 0.0

        # Check for item pickups
        for item in self.items:
            if not item.is_available:
                continue

            dist = self.player.position.distance_to(item.position)
            if dist <= self.USE_RANGE:
                item.is_available = False
                reward += self.reward_config["item_pickup"]
                self.stats["items_used"] += 1

                if item.item_type == ItemType.MEDKIT:
                    self.player.health_item = 1
                elif item.item_type == ItemType.PILLS:
                    self.player.health_item = 2
                elif item.item_type == ItemType.PIPE_BOMB:
                    self.player.throwable = 1
                elif item.item_type == ItemType.MOLOTOV:
                    self.player.throwable = 2
                elif item.item_type == ItemType.AMMO_PILE:
                    self.player.ammo = 100

                break

        # Check for incapped teammates to revive
        for tm in self.teammates:
            if not tm.is_alive or not tm.is_incapped:
                continue

            dist = self.player.position.distance_to(tm.position)
            if dist <= self.USE_RANGE:
                tm.is_incapped = False
                tm.health = 30  # Low health after revive
                reward += self.reward_config["heal_teammate"]
                self.stats["teammates_healed"] += 1
                break

        return reward

    def _do_heal_self(self) -> float:
        """Heal self with health item."""
        if self.player.health_item == 0:
            return 0.0

        if self.player.health >= 100 and not self.player.is_incapped:
            return 0.0  # No need to heal

        heal_amount = 80 if self.player.health_item == 1 else 50  # Medkit vs pills
        self.player.heal(heal_amount, full_heal=(self.player.health_item == 1))
        self.player.health_item = 0
        self.stats["items_used"] += 1

        return self.reward_config["heal_self"]

    def _do_heal_other(self) -> float:
        """Heal a nearby teammate."""
        if self.player.health_item == 0:
            return 0.0

        for tm in self.teammates:
            if not tm.is_alive:
                continue

            dist = self.player.position.distance_to(tm.position)
            if dist <= self.USE_RANGE and (tm.health < 50 or tm.is_incapped):
                heal_amount = 80 if self.player.health_item == 1 else 50
                tm.health = min(100, tm.health + heal_amount)
                if tm.is_incapped and self.player.health_item == 1:
                    tm.is_incapped = False

                self.player.health_item = 0
                self.stats["items_used"] += 1
                self.stats["teammates_healed"] += 1
                return self.reward_config["heal_teammate"]

        return 0.0

    def _do_throw(self) -> float:
        """Throw throwable item."""
        if self.player.throwable == 0:
            return 0.0

        reward = 0.0

        if self.player.throwable == 1:  # Pipe bomb
            # Attracts and kills zombies
            kills = 0
            for zombie in self.zombies:
                if zombie.is_alive and zombie.zombie_type == ZombieType.COMMON:
                    dist = self.player.position.distance_to(zombie.position)
                    if dist < 500:
                        zombie.is_alive = False
                        kills += 1
            reward += kills * self.reward_config["kill"]
            self.stats["kills"] += kills

        elif self.player.throwable == 2:  # Molotov
            # Area damage
            for zombie in self.zombies:
                if not zombie.is_alive:
                    continue
                dist = self.player.position.distance_to(zombie.position)
                if dist < 300:
                    zombie.health -= 100
                    if zombie.health <= 0:
                        zombie.is_alive = False
                        reward += self.reward_config["kill"]
                        self.stats["kills"] += 1

        self.player.throwable = 0
        self.stats["items_used"] += 1
        return reward

    def _build_observation(self) -> np.ndarray:
        """Build observation vector from current state."""
        # Count nearby enemies
        nearby_enemies = 0
        nearest_enemy_dist = 2000.0
        for zombie in self.zombies:
            if not zombie.is_alive:
                continue
            dist = self.player.position.distance_to(zombie.position)
            if dist < 1000:
                nearby_enemies += 1
            nearest_enemy_dist = min(nearest_enemy_dist, dist)

        # Find nearest alive teammate
        nearest_teammate_dist = 2000.0
        teammates_alive = 0
        teammates_incapped = 0
        for tm in self.teammates:
            if tm.is_alive:
                teammates_alive += 1
                if tm.is_incapped:
                    teammates_incapped += 1
                dist = self.player.position.distance_to(tm.position)
                nearest_teammate_dist = min(nearest_teammate_dist, dist)

        # Check if in safe room
        in_safe_room = self.map_state.distance_to_saferoom < 100

        # Check if near objective (checkpoint or safe room)
        near_objective = self.map_state.distance_to_saferoom < 500

        obs = np.array([
            self.player.health / 100.0,
            float(self.player.is_alive),
            float(self.player.is_incapped),
            self.player.position.x / 10000.0,
            self.player.position.y / 10000.0,
            self.player.position.z / 1000.0,
            self.player.velocity.x / 500.0,
            self.player.velocity.y / 500.0,
            self.player.velocity.z / 500.0,
            self.player.angle[0] / 180.0,
            self.player.angle[1] / 180.0,
            self.player.primary_weapon / 20.0,
            self.player.ammo / 100.0,
            min(nearby_enemies, 10) / 10.0,
            min(nearest_enemy_dist, 2000.0) / 2000.0,
            min(nearest_teammate_dist, 2000.0) / 2000.0,
            teammates_alive / 3.0,
            teammates_incapped / 3.0,
            float(in_safe_room),
            float(near_objective),
        ], dtype=np.float32)

        return np.clip(obs, -1.0, 1.0)

    def _calculate_step_reward(self, action: int, action_reward: float) -> float:
        """Calculate total reward for the step."""
        reward = action_reward

        # Survival reward
        if self.player.is_alive:
            reward += self.reward_config["survival"]

        # Progress reward
        if hasattr(self, '_prev_progress'):
            progress_delta = self.map_state.current_progress - self._prev_progress
            if progress_delta > 0:
                reward += self.reward_config["progress"] * (progress_delta / 100.0)
        self._prev_progress = self.map_state.current_progress

        # Checkpoint reward
        if hasattr(self, '_prev_checkpoints'):
            new_checkpoints = sum(self.map_state.checkpoints_reached) - self._prev_checkpoints
            if new_checkpoints > 0:
                reward += self.reward_config["checkpoint"] * new_checkpoints
        self._prev_checkpoints = sum(self.map_state.checkpoints_reached)

        # Team proximity reward
        for tm in self.teammates:
            if tm.is_alive:
                dist = self.player.position.distance_to(tm.position)
                if dist < 500:
                    reward += self.reward_config["proximity_to_team"]

        # Penalty for damage taken this step
        if hasattr(self, '_prev_health'):
            health_diff = self.player.health - self._prev_health
            if health_diff < 0:
                reward += self.reward_config["damage_taken"] * abs(health_diff) / 10.0
        self._prev_health = self.player.health

        # Incap/death penalties
        if hasattr(self, '_was_incapped'):
            if not self._was_incapped and self.player.is_incapped:
                reward += self.reward_config["incapped"]
        self._was_incapped = self.player.is_incapped

        if hasattr(self, '_was_alive'):
            if self._was_alive and not self.player.is_alive:
                reward += self.reward_config["death"]
        self._was_alive = self.player.is_alive

        # Safe room bonus
        if self.map_state.distance_to_saferoom < 100:
            reward += self.reward_config["safe_room"]

        return reward

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)

        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # Reset step counter
        self.current_step = 0
        self.episode_reward = 0.0

        # Initialize player
        self.player = PlayerState(
            position=Vector3(0, 0, 0),
            health=100,
            is_alive=True,
            ammo=100,
            health_item=1,  # Start with medkit
            primary_weapon=1,
        )

        # Initialize teammates
        self.teammates = [
            Teammate(teammate_id=i, position=Vector3(-50 - i*30, (i-1)*50, 0))
            for i in range(3)
        ]

        # Initialize map
        self.map_state = MapState()

        # Clear entities
        self.zombies = []
        self.items = []

        # Reset cooldowns
        self.shove_cooldown = 0.0

        # Reset tracking variables
        self._prev_progress = 0.0
        self._prev_checkpoints = 0
        self._prev_health = 100
        self._was_incapped = False
        self._was_alive = True

        # Reset statistics
        self.stats = {
            "kills": 0,
            "special_kills": 0,
            "damage_dealt": 0,
            "damage_taken": 0,
            "items_used": 0,
            "teammates_healed": 0,
            "distance_traveled": 0.0,
            "checkpoints_reached": 0,
        }

        observation = self._build_observation()
        info = {
            "health": self.player.health,
            "progress": self.map_state.progress_percent,
            "stats": self.stats.copy(),
        }

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        self.current_step += 1

        # Update cooldowns
        self.shove_cooldown = max(0, self.shove_cooldown - self.dt)
        self.player.boomer_blind_timer = max(0, self.player.boomer_blind_timer - self.dt)

        # Process player action
        action_reward = self._process_action(action)

        # Update game state
        self._spawn_zombies()
        self._spawn_items()
        self._update_zombies()
        self._update_teammates()

        # Clean up dead zombies
        self.zombies = [z for z in self.zombies if z.is_alive]
        self.items = [i for i in self.items if i.is_available]

        # Calculate reward
        reward = self._calculate_step_reward(action, action_reward)
        self.episode_reward += reward

        # Check termination conditions
        terminated = False

        # Death = termination
        if not self.player.is_alive:
            terminated = True

        # Reaching safe room = success termination
        if self.map_state.distance_to_saferoom < 100:
            terminated = True

        # Truncation (max steps)
        truncated = self.current_step >= self.max_episode_steps

        # Build observation
        observation = self._build_observation()

        info = {
            "health": self.player.health,
            "is_alive": self.player.is_alive,
            "is_incapped": self.player.is_incapped,
            "step": self.current_step,
            "episode_reward": self.episode_reward,
            "progress": self.map_state.progress_percent,
            "distance_to_saferoom": self.map_state.distance_to_saferoom,
            "nearby_enemies": len([z for z in self.zombies if z.is_alive]),
            "teammates_alive": sum(1 for tm in self.teammates if tm.is_alive),
            "in_safe_room": self.map_state.distance_to_saferoom < 100,
            "stats": self.stats.copy(),
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            print(f"\n--- Step {self.current_step} ---")
            print(f"Health: {self.player.health} | Alive: {self.player.is_alive} | Incapped: {self.player.is_incapped}")
            print(f"Position: ({self.player.position.x:.1f}, {self.player.position.y:.1f})")
            print(f"Progress: {self.map_state.progress_percent*100:.1f}% | Distance to safe room: {self.map_state.distance_to_saferoom:.0f}")
            print(f"Zombies: {len([z for z in self.zombies if z.is_alive])} | Teammates alive: {sum(1 for tm in self.teammates if tm.is_alive)}")
            print(f"Kills: {self.stats['kills']} | Special kills: {self.stats['special_kills']}")
            print(f"Ammo: {self.player.ammo} | Health item: {self.player.health_item} | Throwable: {self.player.throwable}")
            print(f"Episode reward: {self.episode_reward:.2f}")

    def close(self):
        """Clean up resources."""
        pass


# Register environment
try:
    gym.register(
        id="L4D2-Enhanced-v0",
        entry_point="enhanced_mock_env:EnhancedL4D2Env",
    )
except:
    pass


if __name__ == "__main__":
    # Quick test
    print("Testing EnhancedL4D2Env...")

    env = EnhancedL4D2Env(render_mode="human")
    obs, info = env.reset()

    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if i % 20 == 0:
            env.render()

        if terminated or truncated:
            print(f"\nEpisode ended at step {i}")
            print(f"Final stats: {info['stats']}")
            break

    print(f"\nTotal reward: {total_reward:.2f}")
    env.close()
