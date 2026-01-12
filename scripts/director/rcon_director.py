#!/usr/bin/env python3
"""
L4D2 RCON-Based AI Director

A simple AI Director that spawns special infected via RCON commands.
Works with the laoyutang/l4d2 Docker image which has SourceMod pre-installed.

This is a standalone director that doesn't require the Mnemosyne plugin.
It uses sm_cvar commands to bypass sv_cheats restrictions.

Usage:
    # On the server (inside Docker or via SSH)
    python rcon_director.py

    # Or remotely (change RCON_HOST to server IP)
    python rcon_director.py --host 104.248.183.166

Server Requirements:
    - Docker Image: laoyutang/l4d2:latest
    - SourceMod: 1.11.0.6968 (pre-installed)
    - RCON enabled with known password
"""

import socket
import struct
import time
import random
import argparse
from typing import Tuple, Optional

# Default Configuration
DEFAULT_RCON_HOST = "127.0.1.1"  # Host networking binds here
DEFAULT_RCON_PORT = 27015
DEFAULT_RCON_PASSWORD = "ai2026"

# Spawn Configuration - Conservative defaults to prevent crashes
SPECIAL_INFECTED = ["hunter", "smoker", "boomer", "charger", "spitter", "jockey"]

# Stable spawn limits (tested values that don't crash)
STABLE_LIMITS = {
    "z_hunter_limit": 2,
    "z_smoker_limit": 2,
    "z_boomer_limit": 2,
    "z_charger_limit": 2,
    "z_spitter_limit": 2,
    "z_jockey_limit": 2,
    "z_special_spawn_interval": 25,
    "z_common_limit": 25,
}


class RCONClient:
    """Simple RCON client with UTF-8 support."""

    def __init__(self, host: str, port: int, password: str):
        self.host = host
        self.port = port
        self.password = password
        self.sock: Optional[socket.socket] = None
        self.req_id = 1

    def connect(self) -> bool:
        """Connect and authenticate to RCON."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            self.sock.settimeout(10)

            # Authenticate
            self._send(3, self.password)  # 3 = AUTH
            self._recv()  # Empty response
            result = self._recv()  # Auth result
            return result is not None
        except Exception as e:
            print(f"[RCON] Connection failed: {e}")
            return False

    def _send(self, req_type: int, body: str) -> None:
        """Send RCON packet."""
        if not self.sock:
            raise RuntimeError("Not connected")
        body_bytes = body.encode('utf-8') + b'\x00\x00'
        size = 4 + 4 + len(body_bytes)
        packet = struct.pack('<iii', size, self.req_id, req_type) + body_bytes
        self.sock.send(packet)
        self.req_id += 1

    def _recv(self) -> Optional[str]:
        """Receive RCON response."""
        if not self.sock:
            return None
        try:
            data = self.sock.recv(4)
            if len(data) < 4:
                return None
            size = struct.unpack('<i', data)[0]
            data = self.sock.recv(size)
            body = data[8:-2].decode('utf-8', errors='replace')
            return body
        except Exception:
            return None

    def execute(self, cmd: str) -> Optional[str]:
        """Execute RCON command and return response."""
        try:
            self._send(2, cmd)  # 2 = EXECCOMMAND
            return self._recv()
        except Exception as e:
            print(f"[RCON] Command failed: {e}")
            return None

    def close(self) -> None:
        """Close connection."""
        if self.sock:
            self.sock.close()
            self.sock = None


class RCONDirector:
    """AI Director that spawns special infected via RCON."""

    def __init__(self, host: str, port: int, password: str):
        self.host = host
        self.port = port
        self.password = password
        self.rcon: Optional[RCONClient] = None
        self.last_spawn_time = 0.0
        self.last_tank_time = 0.0
        self.active = False
        self.intense_mode = False
        self.spawn_count = 0

        # Configurable intervals
        self.spawn_interval_min = 25
        self.spawn_interval_max = 45
        self.tank_interval = 180  # Every 3 minutes in intense mode

    def connect(self) -> bool:
        """Connect to the server."""
        try:
            self.rcon = RCONClient(self.host, self.port, self.password)
            if self.rcon.connect():
                print(f"[Director] Connected to {self.host}:{self.port}")
                return True
        except Exception as e:
            print(f"[Director] Connection failed: {e}")
        return False

    def get_player_count(self) -> Tuple[int, int]:
        """Get current player count (humans, bots)."""
        if not self.rcon:
            return 0, 0

        status = self.rcon.execute('status')
        if not status:
            return 0, 0

        humans = 0
        bots = 0
        for line in status.split('\n'):
            if 'humans' in line and 'bots' in line:
                import re
                match = re.search(r'(\d+)\s+humans?,\s+(\d+)\s+bots?', line)
                if match:
                    humans = int(match.group(1))
                    bots = int(match.group(2))
                    break

        return humans, bots

    def apply_stable_settings(self) -> None:
        """Apply stable spawn settings that won't crash the server."""
        if not self.rcon:
            return

        print("[Director] Applying stable spawn settings...")
        for cvar, value in STABLE_LIMITS.items():
            self.rcon.execute(f'sm_cvar {cvar} {value}')

        # Enable cheats for z_spawn commands
        self.rcon.execute('sm_cvar sv_cheats 1')
        self.rcon.execute('sm_cvar sv_hibernate_when_empty 0')
        print("[Director] Settings applied")

    def spawn_special(self, infected_type: str) -> None:
        """Spawn a special infected."""
        if not self.rcon:
            return

        self.rcon.execute(f'z_spawn {infected_type}')
        self.spawn_count += 1
        print(f"[Director] Spawned {infected_type} (total: {self.spawn_count})")

    def spawn_tank(self) -> None:
        """Spawn a Tank."""
        if not self.rcon:
            return

        self.rcon.execute('z_spawn tank')
        print("[Director] *** TANK SPAWNED ***")

    def spawn_horde(self) -> None:
        """Trigger a mini-horde."""
        if not self.rcon:
            return

        self.rcon.execute('z_spawn mob')
        print("[Director] Horde incoming!")

    def run(self) -> None:
        """Main director loop."""
        print("\n" + "=" * 60)
        print("   L4D2 RCON-BASED AI DIRECTOR")
        print("   For laoyutang/l4d2 Docker + SourceMod")
        print("=" * 60)
        print(f"\nConfig:")
        print(f"  Host: {self.host}:{self.port}")
        print(f"  Spawn interval: {self.spawn_interval_min}-{self.spawn_interval_max}s")
        print(f"  Tank interval: {self.tank_interval}s (intense mode)")
        print(f"  Special infected: {', '.join(SPECIAL_INFECTED)}")
        print("\nConnecting to server...")

        if not self.connect():
            print("[Director] Failed to connect. Exiting.")
            return

        # Apply stable settings on startup
        self.apply_stable_settings()

        print("\n[Director] Monitoring started. Waiting for players...\n")

        iteration = 0
        while True:
            try:
                iteration += 1
                current_time = time.time()

                # Check player count
                humans, bots = self.get_player_count()
                total = humans + bots

                # Update mode
                was_active = self.active
                was_intense = self.intense_mode

                self.active = humans >= 1
                self.intense_mode = total >= 4

                # Log mode changes
                if self.active and not was_active:
                    print(f"[Director] ACTIVATED - {humans} human(s) detected!")
                elif not self.active and was_active:
                    print("[Director] DEACTIVATED - No humans in game")

                if self.intense_mode and not was_intense:
                    print(f"[Director] INTENSE MODE - Full lobby ({total} players)!")
                    # Welcome spawns (careful - only 2)
                    self.spawn_special(random.choice(SPECIAL_INFECTED))
                    time.sleep(1)
                    self.spawn_special(random.choice(SPECIAL_INFECTED))

                # Periodic status
                if iteration % 30 == 0:
                    mode = "INTENSE" if self.intense_mode else ("ACTIVE" if self.active else "IDLE")
                    print(f"[Director] Status: {mode} | Humans: {humans} | Bots: {bots} | Spawns: {self.spawn_count}")

                # Spawn logic
                if self.active:
                    spawn_interval = random.randint(self.spawn_interval_min, self.spawn_interval_max)
                    if self.intense_mode:
                        spawn_interval = spawn_interval // 2

                    if current_time - self.last_spawn_time >= spawn_interval:
                        infected = random.choice(SPECIAL_INFECTED)
                        self.spawn_special(infected)

                        # Small chance of double spawn in intense mode
                        if self.intense_mode and random.random() < 0.2:
                            time.sleep(1)
                            self.spawn_special(random.choice(SPECIAL_INFECTED))

                        self.last_spawn_time = current_time

                    # Tank spawning (intense mode only)
                    if self.intense_mode:
                        if current_time - self.last_tank_time >= self.tank_interval:
                            self.spawn_tank()
                            self.last_tank_time = current_time

                    # Random horde (5% chance each minute in intense mode)
                    if self.intense_mode and iteration % 60 == 0 and random.random() < 0.05:
                        self.spawn_horde()

                time.sleep(1)

            except KeyboardInterrupt:
                print("\n\n[Director] Shutting down...")
                print(f"[Director] Total spawns: {self.spawn_count}")
                break
            except Exception as e:
                print(f"[Director] Error: {e}")
                time.sleep(5)
                self.connect()


def main():
    parser = argparse.ArgumentParser(description="L4D2 RCON-Based AI Director")
    parser.add_argument("--host", default=DEFAULT_RCON_HOST, help="RCON host address")
    parser.add_argument("--port", type=int, default=DEFAULT_RCON_PORT, help="RCON port")
    parser.add_argument("--password", default=DEFAULT_RCON_PASSWORD, help="RCON password")
    parser.add_argument("--spawn-min", type=int, default=25, help="Minimum spawn interval (seconds)")
    parser.add_argument("--spawn-max", type=int, default=45, help="Maximum spawn interval (seconds)")
    parser.add_argument("--tank-interval", type=int, default=180, help="Tank spawn interval (seconds)")

    args = parser.parse_args()

    director = RCONDirector(args.host, args.port, args.password)
    director.spawn_interval_min = args.spawn_min
    director.spawn_interval_max = args.spawn_max
    director.tank_interval = args.tank_interval
    director.run()


if __name__ == "__main__":
    main()
