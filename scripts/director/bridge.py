#!/usr/bin/env python3
"""
Game Bridge Module

Handles communication between Python and the L4D2 SourceMod plugin.
Supports TCP, UDP, and HTTP protocols.
"""

import json
import time
import logging
import socket
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import queue

logger = logging.getLogger(__name__)


@dataclass
class BridgeConfig:
    """Configuration for the game bridge"""
    host: str = "localhost"
    port: int = 27050
    protocol: str = "tcp"  # tcp, udp, or http
    timeout: float = 5.0
    retry_interval: float = 1.0
    max_retries: int = 5


class GameBridge:
    """Bridge for communicating with L4D2 game server"""
    
    def __init__(self, host: str = "localhost", port: int = 27050, 
                 protocol: str = "tcp", timeout: float = 5.0):
        self.config = BridgeConfig(host=host, port=port, protocol=protocol, timeout=timeout)
        self.socket = None
        self.is_connected = False
        self.game_state = {}
        self.state_callbacks: List[Callable] = []
        self.command_queue = queue.Queue()
        
        # Threading
        self.receive_thread = None
        self.send_thread = None
        self.running = False
        
    def connect(self) -> bool:
        """Connect to the game server"""
        if self.is_connected:
            return True
        
        try:
            if self.config.protocol == "tcp":
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(self.config.timeout)
                self.socket.connect((self.config.host, self.config.port))
            elif self.config.protocol == "udp":
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.socket.settimeout(self.config.timeout)
            else:
                raise ValueError(f"Unsupported protocol: {self.config.protocol}")
            
            self.is_connected = True
            self.running = True
            
            # Start communication threads
            self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.receive_thread.start()
            
            self.send_thread = threading.Thread(target=self._send_loop, daemon=True)
            self.send_thread.start()
            
            logger.info(f"Connected to game server at {self.config.host}:{self.config.port}")
            
            # Send handshake
            self._send_handshake()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to game server: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the game server"""
        self.running = False
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        
        self.is_connected = False
        logger.info("Disconnected from game server")
    
    def _send_handshake(self):
        """Send initial handshake"""
        handshake = {
            "type": "handshake",
            "version": "1.0.0",
            "timestamp": time.time()
        }
        self._send_json(handshake)
    
    def _receive_loop(self):
        """Thread for receiving data from game server"""
        buffer = ""
        
        while self.running and self.is_connected:
            try:
                if self.config.protocol == "tcp":
                    data = self.socket.recv(4096).decode('utf-8')
                    if not data:
                        break
                    
                    buffer += data
                    
                    # Process complete JSON messages
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if line:
                            self._handle_message(line)
                
                elif self.config.protocol == "udp":
                    data, addr = self.socket.recvfrom(4096)
                    if data:
                        self._handle_message(data.decode('utf-8'))
                
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Error receiving data: {e}")
                break
        
        self.is_connected = False
    
    def _send_loop(self):
        """Thread for sending commands to game server"""
        while self.running and self.is_connected:
            try:
                # Get command from queue
                command = self.command_queue.get(timeout=1.0)
                self._send_json(command)
                self.command_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error sending command: {e}")
                break
    
    def _handle_message(self, message: str):
        """Handle incoming message from game server"""
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")
            
            if msg_type == "game_state":
                self.game_state = data
                # Notify callbacks
                for callback in self.state_callbacks:
                    try:
                        callback(data)
                    except Exception as e:
                        logger.error(f"Error in state callback: {e}")
            
            elif msg_type == "event":
                self._handle_event(data)
            
            elif msg_type == "response":
                # Handle command responses
                pass
            
            else:
                logger.debug(f"Unknown message type: {msg_type}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON received: {e}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    def _handle_event(self, event: Dict[str, Any]):
        """Handle game events"""
        event_type = event.get("event_type", "")
        
        if event_type == "player_death":
            logger.info(f"Player died: {event}")
        elif event_type == "round_start":
            logger.info("Round started")
        elif event_type == "round_end":
            logger.info("Round ended")
        elif event_type == "tank_spawned":
            logger.info("Tank spawned")
        elif event_type == "witch_spawned":
            logger.info("Witch spawned")
    
    def _send_json(self, data: Dict[str, Any]):
        """Send JSON data to game server"""
        try:
            message = json.dumps(data) + '\n'
            
            if self.config.protocol == "tcp":
                self.socket.send(message.encode('utf-8'))
            elif self.config.protocol == "udp":
                self.socket.sendto(message.encode('utf-8'), 
                                (self.config.host, self.config.port))
            
        except Exception as e:
            logger.error(f"Error sending data: {e}")
            self.is_connected = False
    
    def get_game_state(self) -> Optional[Dict[str, Any]]:
        """Get current game state"""
        return self.game_state.copy() if self.game_state else None
    
    def send_bot_action(self, bot_id: int, action: str, **kwargs):
        """Send action command for a bot"""
        command = {
            "type": "bot_action",
            "bot_id": bot_id,
            "action": action,
            "parameters": kwargs,
            "timestamp": time.time()
        }
        self.command_queue.put(command)
    
    def send_director_command(self, command_type: str, parameters: Dict[str, Any]):
        """Send director command to game server"""
        command = {
            "type": "director_command",
            "command": command_type,
            "parameters": parameters,
            "timestamp": time.time()
        }
        self.command_queue.put(command)
    
    def reset_episode(self):
        """Reset the current episode"""
        command = {
            "type": "reset_episode",
            "timestamp": time.time()
        }
        self.command_queue.put(command)
    
    def add_state_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for game state updates"""
        self.state_callbacks.append(callback)
    
    def remove_state_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Remove state callback"""
        if callback in self.state_callbacks:
            self.state_callbacks.remove(callback)


class MockBridge(GameBridge):
    """Mock bridge for testing without game server"""
    
    def __init__(self):
        super().__init__()
        self.mock_state = {
            "gameTime": 0.0,
            "roundTime": 0.0,
            "survivors": [
                {
                    "id": 1,
                    "health": 100,
                    "tempHealth": 0,
                    "position": [0, 0, 0],
                    "angle": [0, 0, 0],
                    "weapon": "pistol",
                    "isIncapped": False,
                    "isDead": False
                },
                {
                    "id": 2,
                    "health": 80,
                    "tempHealth": 20,
                    "position": [100, 0, 0],
                    "angle": [0, 90, 0],
                    "weapon": "shotgun",
                    "isIncapped": False,
                    "isDead": False
                }
            ],
            "commonInfected": 5,
            "specialInfected": [0, 1, 0, 0, 0],
            "witchCount": 0,
            "tankCount": 0,
            "itemsAvailable": 3,
            "healthPacksUsed": 1,
            "recentDeaths": 0,
            "panicActive": False,
            "tankActive": False
        }
        self.simulation_thread = None
    
    def connect(self) -> bool:
        """Mock connection"""
        self.is_connected = True
        self.running = True
        
        # Start simulation thread
        self.simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.simulation_thread.start()
        
        logger.info("Connected to mock game server")
        return True
    
    def _simulation_loop(self):
        """Simulate game state changes"""
        while self.running:
            # Update time
            self.mock_state["gameTime"] = time.time()
            self.mock_state["roundTime"] += 0.1
            
            # Simulate some changes
            if np.random.random() < 0.1:
                self.mock_state["commonInfected"] = max(0, 
                    self.mock_state["commonInfected"] + np.random.randint(-2, 5))
            
            if np.random.random() < 0.05:
                idx = np.random.randint(0, 5)
                self.mock_state["specialInfected"][idx] = max(0, 
                    self.mock_state["specialInfected"][idx] + np.random.randint(-1, 2))
            
            # Update survivors
            for survivor in self.mock_state["survivors"]:
                if np.random.random() < 0.1:
                    # Random movement
                    survivor["position"][0] += np.random.randint(-10, 11)
                    survivor["position"][1] += np.random.randint(-10, 11)
                
                if np.random.random() < 0.05:
                    # Health change
                    survivor["health"] = max(0, min(100, 
                        survivor["health"] + np.random.randint(-10, 5)))
            
            # Notify callbacks
            self.game_state = self.mock_state.copy()
            for callback in self.state_callbacks:
                try:
                    callback(self.game_state)
                except:
                    pass
            
            time.sleep(0.1)
    
    def _send_json(self, data: Dict[str, Any]):
        """Mock send - just log"""
        logger.debug(f"Mock sending: {data}")


# Import numpy for mock bridge
try:
    import numpy as np
except ImportError:
    np = None


def main():
    """Test the bridge"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Game Bridge")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=27050, help="Server port")
    parser.add_argument("--protocol", choices=["tcp", "udp"], default="tcp", help="Protocol")
    parser.add_argument("--mock", action="store_true", help="Use mock bridge")
    
    args = parser.parse_args()
    
    # Create bridge
    if args.mock:
        bridge = MockBridge()
    else:
        bridge = GameBridge(args.host, args.port, args.protocol)
    
    # Add state callback
    def on_state_update(state):
        print(f"State update: {len(state.get('survivors', []))} survivors, "
              f"{state.get('commonInfected', 0)} common infected")
    
    bridge.add_state_callback(on_state_update)
    
    # Connect
    if bridge.connect():
        print("Connected successfully!")
        
        try:
            # Test commands
            time.sleep(1)
            
            # Send bot action
            bridge.send_bot_action(1, "move_forward")
            
            # Send director command
            bridge.send_director_command("spawn_common", {"count": 5})
            
            # Keep running
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nShutting down...")
            bridge.disconnect()
    else:
        print("Failed to connect")


if __name__ == "__main__":
    main()
