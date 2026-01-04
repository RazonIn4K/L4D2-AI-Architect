#!/usr/bin/env python3
"""
Mnemosyne L4D2 Gymnasium Environment

A Gymnasium-compatible environment wrapper for training RL agents
on Left 4 Dead 2 through the Mnemosyne bot control system.

This environment connects to a running L4D2 dedicated server with
the Mnemosyne SourceMod plugin installed.

Usage:
    from mnemosyne_env import MnemosyneEnv
    
    env = MnemosyneEnv(host="localhost", port=27050)
    obs, info = env.reset()
    
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
"""

import time
import socket
import struct
import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from enum import IntEnum

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


@dataclass
class GameState:
    """Represents the current game state for an agent."""
    # Bot state
    bot_id: int
    health: int
    is_alive: bool
    is_incapped: bool
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    angle: Tuple[float, float]
    
    # Inventory
    primary_weapon: int
    secondary_weapon: int
    throwable: int
    health_item: int
    ammo: int
    
    # Game context
    nearby_enemies: int
    nearest_enemy_dist: float
    nearest_teammate_dist: float
    teammates_alive: int
    teammates_incapped: int
    
    # Objectives
    in_safe_room: bool
    near_objective: bool
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'GameState':
        """Parse game state from Mnemosyne protocol."""
        if len(data) < 64:
            raise ValueError(f"Invalid data length: {len(data)}")
        
        # Unpack according to Mnemosyne protocol
        # This matches the 26-byte base protocol plus extensions
        fmt = "<BHBBfffffffffffBBBBHBfBfBBBB"
        
        try:
            values = struct.unpack(fmt, data[:struct.calcsize(fmt)])
        except struct.error:
            # Fallback for minimal data
            return cls.default()
        
        return cls(
            bot_id=values[0],
            health=values[1],
            is_alive=bool(values[2]),
            is_incapped=bool(values[3]),
            position=(values[4], values[5], values[6]),
            velocity=(values[7], values[8], values[9]),
            angle=(values[10], values[11]),
            primary_weapon=values[12],
            secondary_weapon=values[13],
            throwable=values[14],
            health_item=values[15],
            ammo=values[16],
            nearby_enemies=values[17],
            nearest_enemy_dist=values[18],
            teammates_alive=values[19],
            nearest_teammate_dist=values[20],
            teammates_incapped=values[21],
            in_safe_room=bool(values[22]),
            near_objective=bool(values[23]),
        )
    
    @classmethod
    def default(cls) -> 'GameState':
        """Return a default game state."""
        return cls(
            bot_id=0,
            health=100,
            is_alive=True,
            is_incapped=False,
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            angle=(0.0, 0.0),
            primary_weapon=0,
            secondary_weapon=0,
            throwable=0,
            health_item=0,
            ammo=0,
            nearby_enemies=0,
            nearest_enemy_dist=1000.0,
            nearest_teammate_dist=100.0,
            teammates_alive=3,
            teammates_incapped=0,
            in_safe_room=True,
            near_objective=False,
        )
    
    def to_observation(self) -> np.ndarray:
        """Convert to numpy observation vector."""
        return np.array([
            self.health / 100.0,
            float(self.is_alive),
            float(self.is_incapped),
            self.position[0] / 10000.0,
            self.position[1] / 10000.0,
            self.position[2] / 1000.0,
            self.velocity[0] / 500.0,
            self.velocity[1] / 500.0,
            self.velocity[2] / 500.0,
            self.angle[0] / 180.0,
            self.angle[1] / 180.0,
            self.primary_weapon / 20.0,
            self.ammo / 100.0,
            self.nearby_enemies / 10.0,
            min(self.nearest_enemy_dist, 2000.0) / 2000.0,
            min(self.nearest_teammate_dist, 2000.0) / 2000.0,
            self.teammates_alive / 3.0,
            self.teammates_incapped / 3.0,
            float(self.in_safe_room),
            float(self.near_objective),
        ], dtype=np.float32)


class MnemosyneEnv(gym.Env):
    """
    Gymnasium environment for L4D2 bot control via Mnemosyne.
    
    Observation Space (20D continuous):
        - Health (normalized)
        - Alive/Incapped status
        - Position (x, y, z normalized)
        - Velocity (x, y, z normalized)
        - View angles (pitch, yaw normalized)
        - Weapon/ammo info
        - Enemy proximity info
        - Teammate status
        - Objective proximity
    
    Action Space (14 discrete actions):
        See BotAction enum
    
    Rewards:
        - +1.0 for killing an enemy
        - +5.0 for healing a teammate
        - +0.1 for dealing damage
        - -0.1 for taking damage
        - -10.0 for getting incapped
        - -50.0 for dying
        - +100.0 for reaching safe room
        - +0.01 per timestep survived
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 27050,
        bot_id: int = 0,
        max_episode_steps: int = 10000,
        render_mode: Optional[str] = None,
        timeout: float = 5.0,
    ):
        super().__init__()
        
        self.host = host
        self.port = port
        self.bot_id = bot_id
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.timeout = timeout
        
        # Socket connection
        self.socket: Optional[socket.socket] = None
        self.connected = False
        
        # Episode state
        self.current_step = 0
        self.prev_state: Optional[GameState] = None
        self.current_state: Optional[GameState] = None
        self.episode_reward = 0.0
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(20,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(len(BotAction))
        
        # Reward shaping parameters
        self.reward_config = {
            "kill": 1.0,
            "damage_dealt": 0.1,
            "damage_taken": -0.1,
            "heal_teammate": 5.0,
            "incapped": -10.0,
            "death": -50.0,
            "safe_room": 100.0,
            "survival": 0.01,
            "proximity_to_team": 0.001,
        }
    
    def _connect(self) -> bool:
        """Establish connection to Mnemosyne server."""
        if self.connected:
            return True
        
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.settimeout(self.timeout)
            
            # Send connection request
            connect_msg = struct.pack("<BB", 0x01, self.bot_id)
            self.socket.sendto(connect_msg, (self.host, self.port))
            
            # Wait for acknowledgment
            data, addr = self.socket.recvfrom(1024)
            if data[0] == 0x01:
                self.connected = True
                logger.info(f"Connected to Mnemosyne at {self.host}:{self.port}")
                return True
            
        except socket.timeout:
            logger.warning("Connection timeout - using simulation mode")
        except Exception as e:
            logger.warning(f"Connection failed: {e} - using simulation mode")
        
        return False
    
    def _disconnect(self):
        """Close connection to server."""
        if self.socket:
            try:
                disconnect_msg = struct.pack("<BB", 0x02, self.bot_id)
                self.socket.sendto(disconnect_msg, (self.host, self.port))
            except:
                pass
            self.socket.close()
            self.socket = None
        self.connected = False
    
    def _send_action(self, action: int) -> bool:
        """Send action to the game server."""
        if not self.connected:
            return False
        
        try:
            action_msg = struct.pack("<BBB", 0x03, self.bot_id, action)
            self.socket.sendto(action_msg, (self.host, self.port))
            return True
        except Exception as e:
            logger.error(f"Failed to send action: {e}")
            return False
    
    def _receive_state(self) -> Optional[GameState]:
        """Receive game state from server."""
        if not self.connected:
            return self._simulate_state()
        
        try:
            data, addr = self.socket.recvfrom(1024)
            if data[0] == 0x04:  # State update message
                return GameState.from_bytes(data[1:])
        except socket.timeout:
            logger.warning("State receive timeout")
        except Exception as e:
            logger.error(f"Failed to receive state: {e}")
        
        return None
    
    def _simulate_state(self) -> GameState:
        """Generate simulated state for testing without game connection."""
        if self.current_state is None:
            return GameState.default()
        
        # Simple simulation: random walk with some enemy encounters
        state = self.current_state
        
        # Simulate movement
        new_pos = (
            state.position[0] + np.random.randn() * 10,
            state.position[1] + np.random.randn() * 10,
            state.position[2],
        )
        
        # Simulate combat
        health = state.health
        if np.random.random() < 0.05:  # 5% chance of taking damage
            health = max(0, health - np.random.randint(5, 20))
        
        # Simulate enemy spawning
        nearby_enemies = max(0, state.nearby_enemies + np.random.randint(-1, 2))
        
        return GameState(
            bot_id=self.bot_id,
            health=health,
            is_alive=health > 0,
            is_incapped=health <= 0 and health > -100,
            position=new_pos,
            velocity=state.velocity,
            angle=state.angle,
            primary_weapon=state.primary_weapon,
            secondary_weapon=state.secondary_weapon,
            throwable=state.throwable,
            health_item=state.health_item,
            ammo=max(0, state.ammo - 1) if nearby_enemies > 0 else state.ammo,
            nearby_enemies=nearby_enemies,
            nearest_enemy_dist=500.0 / max(1, nearby_enemies),
            nearest_teammate_dist=state.nearest_teammate_dist,
            teammates_alive=state.teammates_alive,
            teammates_incapped=state.teammates_incapped,
            in_safe_room=False,
            near_objective=np.random.random() < 0.01,
        )
    
    def _calculate_reward(self, prev_state: GameState, curr_state: GameState, action: int) -> float:
        """Calculate reward based on state transition."""
        reward = 0.0
        
        # Survival reward
        if curr_state.is_alive:
            reward += self.reward_config["survival"]
        
        # Health change
        health_diff = curr_state.health - prev_state.health
        if health_diff < 0:
            reward += self.reward_config["damage_taken"] * abs(health_diff)
        
        # Incap/Death penalties
        if not prev_state.is_incapped and curr_state.is_incapped:
            reward += self.reward_config["incapped"]
        
        if prev_state.is_alive and not curr_state.is_alive:
            reward += self.reward_config["death"]
        
        # Kill rewards (estimated from enemy count change)
        enemy_diff = prev_state.nearby_enemies - curr_state.nearby_enemies
        if enemy_diff > 0 and action == BotAction.ATTACK:
            reward += self.reward_config["kill"] * enemy_diff
        
        # Healing rewards
        if action == BotAction.HEAL_OTHER:
            if prev_state.teammates_incapped > curr_state.teammates_incapped:
                reward += self.reward_config["heal_teammate"]
        
        # Safe room reward
        if not prev_state.in_safe_room and curr_state.in_safe_room:
            reward += self.reward_config["safe_room"]
        
        # Team proximity reward (encourage staying with team)
        if curr_state.nearest_teammate_dist < 500:
            reward += self.reward_config["proximity_to_team"]
        
        return reward
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Try to connect if not connected
        if not self.connected:
            self._connect()
        
        # Reset episode state
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Get initial state
        if self.connected:
            # Send reset command to game
            reset_msg = struct.pack("<BB", 0x05, self.bot_id)
            self.socket.sendto(reset_msg, (self.host, self.port))
            time.sleep(0.1)
            self.current_state = self._receive_state()
        else:
            self.current_state = GameState.default()
        
        self.prev_state = self.current_state
        
        observation = self.current_state.to_observation()
        info = {
            "connected": self.connected,
            "bot_id": self.bot_id,
            "health": self.current_state.health,
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        self.current_step += 1
        
        # Send action
        if self.connected:
            self._send_action(action)
            time.sleep(0.01)  # 100Hz tick rate
            self.current_state = self._receive_state()
        else:
            # Simulate action effects
            self.current_state = self._simulate_state()
        
        if self.current_state is None:
            self.current_state = self.prev_state
        
        # Calculate reward
        reward = self._calculate_reward(self.prev_state, self.current_state, action)
        self.episode_reward += reward
        
        # Check termination
        terminated = not self.current_state.is_alive
        truncated = self.current_step >= self.max_episode_steps
        
        # Also terminate on safe room reached (success!)
        if self.current_state.in_safe_room:
            terminated = True
        
        observation = self.current_state.to_observation()
        
        info = {
            "health": self.current_state.health,
            "is_alive": self.current_state.is_alive,
            "is_incapped": self.current_state.is_incapped,
            "step": self.current_step,
            "episode_reward": self.episode_reward,
            "in_safe_room": self.current_state.in_safe_room,
            "nearby_enemies": self.current_state.nearby_enemies,
        }
        
        self.prev_state = self.current_state
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment (for debugging)."""
        if self.render_mode == "human":
            state = self.current_state
            print(f"\n--- Step {self.current_step} ---")
            print(f"Health: {state.health} | Alive: {state.is_alive}")
            print(f"Position: ({state.position[0]:.1f}, {state.position[1]:.1f}, {state.position[2]:.1f})")
            print(f"Enemies nearby: {state.nearby_enemies}")
            print(f"Episode reward: {self.episode_reward:.2f}")
    
    def close(self):
        """Clean up resources."""
        self._disconnect()


# Register environment with Gymnasium
try:
    gym.register(
        id="L4D2-Mnemosyne-v0",
        entry_point="mnemosyne_env:MnemosyneEnv",
    )
except:
    pass


if __name__ == "__main__":
    # Test the environment
    print("Testing MnemosyneEnv in simulation mode...")
    
    env = MnemosyneEnv(render_mode="human")
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
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")
    env.close()
