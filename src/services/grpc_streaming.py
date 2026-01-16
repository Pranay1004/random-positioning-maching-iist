"""
gRPC Service Definitions for RPM Digital Twin
==============================================
Provides real-time streaming services for UI components.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional, AsyncIterator
import struct

from loguru import logger

# Note: These protobuf definitions would normally be compiled from .proto files
# For now, we define simple message classes

class Vector3:
    """3D vector message."""
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x = x
        self.y = y
        self.z = z
    
    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "z": self.z}
    
    def to_bytes(self) -> bytes:
        return struct.pack('<fff', self.x, self.y, self.z)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'Vector3':
        x, y, z = struct.unpack('<fff', data[:12])
        return cls(x, y, z)


class MotorCommand:
    """Motor command message."""
    def __init__(
        self,
        motor_id: int,
        rpm: float,
        acceleration: float = 100.0
    ):
        self.motor_id = motor_id
        self.rpm = rpm
        self.acceleration = acceleration
    
    def to_bytes(self) -> bytes:
        return struct.pack('<Bff', self.motor_id, self.rpm, self.acceleration)


class RPMStateMessage:
    """RPM state update message."""
    def __init__(
        self,
        timestamp_ms: int,
        inner_position_rad: float,
        inner_velocity_rad_s: float,
        outer_position_rad: float,
        outer_velocity_rad_s: float,
        acceleration: Vector3,
        instantaneous_g: float,
        time_averaged_g: float
    ):
        self.timestamp_ms = timestamp_ms
        self.inner_position_rad = inner_position_rad
        self.inner_velocity_rad_s = inner_velocity_rad_s
        self.outer_position_rad = outer_position_rad
        self.outer_velocity_rad_s = outer_velocity_rad_s
        self.acceleration = acceleration
        self.instantaneous_g = instantaneous_g
        self.time_averaged_g = time_averaged_g
    
    def to_dict(self) -> dict:
        return {
            "timestamp_ms": self.timestamp_ms,
            "inner_position_rad": self.inner_position_rad,
            "inner_velocity_rad_s": self.inner_velocity_rad_s,
            "outer_position_rad": self.outer_position_rad,
            "outer_velocity_rad_s": self.outer_velocity_rad_s,
            "acceleration": self.acceleration.to_dict(),
            "instantaneous_g": self.instantaneous_g,
            "time_averaged_g": self.time_averaged_g
        }
    
    def to_bytes(self) -> bytes:
        return struct.pack(
            '<Qffffffff',
            self.timestamp_ms,
            self.inner_position_rad,
            self.inner_velocity_rad_s,
            self.outer_position_rad,
            self.outer_velocity_rad_s,
            self.acceleration.x,
            self.acceleration.y,
            self.acceleration.z,
            self.instantaneous_g
        )


class GRPCStreamServer:
    """
    gRPC-like streaming server for real-time data.
    
    Provides WebSocket-based streaming for Unity UI and other clients.
    """
    
    def __init__(self, host: str = "localhost", port: int = 50051):
        """
        Initialize gRPC stream server.
        
        Args:
            host: Server host address
            port: Server port
        """
        self.host = host
        self.port = port
        self._running = False
        self._clients: list = []
        self._state_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        
        logger.info(f"GRPCStreamServer initialized on {host}:{port}")
    
    async def start(self) -> None:
        """Start the streaming server."""
        self._running = True
        logger.info(f"gRPC Stream Server started on {self.host}:{self.port}")
    
    async def stop(self) -> None:
        """Stop the streaming server."""
        self._running = False
        # Close all client connections
        for client in self._clients:
            try:
                await client.close()
            except Exception:
                pass
        self._clients.clear()
        logger.info("gRPC Stream Server stopped")
    
    async def broadcast_state(self, state: RPMStateMessage) -> None:
        """
        Broadcast state update to all connected clients.
        
        Args:
            state: Current RPM state
        """
        if not self._running:
            return
        
        # Put in queue for any listeners
        try:
            self._state_queue.put_nowait(state)
        except asyncio.QueueFull:
            # Drop oldest
            try:
                self._state_queue.get_nowait()
                self._state_queue.put_nowait(state)
            except:
                pass
    
    async def stream_states(self) -> AsyncIterator[RPMStateMessage]:
        """
        Async generator for streaming states.
        
        Yields:
            RPMStateMessage objects as they arrive
        """
        while self._running:
            try:
                state = await asyncio.wait_for(
                    self._state_queue.get(),
                    timeout=0.1
                )
                yield state
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break


class GRPCClient:
    """
    gRPC client for connecting to RPM Digital Twin server.
    
    Used by Unity UI or other visualization clients.
    """
    
    def __init__(self, host: str = "localhost", port: int = 50051):
        """
        Initialize gRPC client.
        
        Args:
            host: Server host
            port: Server port
        """
        self.host = host
        self.port = port
        self._connected = False
        
    async def connect(self) -> bool:
        """
        Connect to server.
        
        Returns:
            True if connected successfully
        """
        # In real implementation, this would establish gRPC channel
        self._connected = True
        logger.info(f"Connected to gRPC server at {self.host}:{self.port}")
        return True
    
    async def disconnect(self) -> None:
        """Disconnect from server."""
        self._connected = False
        logger.info("Disconnected from gRPC server")
    
    async def send_motor_command(self, command: MotorCommand) -> bool:
        """
        Send motor command to server.
        
        Args:
            command: Motor command message
            
        Returns:
            True if sent successfully
        """
        if not self._connected:
            logger.warning("Not connected to server")
            return False
        
        # In real implementation, this would send via gRPC
        logger.debug(f"Motor command: id={command.motor_id}, rpm={command.rpm}")
        return True
    
    async def subscribe_to_states(
        self,
        callback
    ) -> None:
        """
        Subscribe to state updates.
        
        Args:
            callback: Function to call with each state update
        """
        if not self._connected:
            logger.warning("Not connected to server")
            return
        
        # In real implementation, this would stream from server
        logger.info("Subscribed to state updates")
