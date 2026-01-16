"""
RPM Digital Twin - Real-Time Web Dashboard Server v2.0
======================================================
SpaceX Dragon / Tesla Autopilot inspired UI backend.

Fixed gravity calculations based on proper microgravity science:
- Time-averaged gravity VECTOR (not magnitude)
- Proper rotation matrices from research paper
- Rectangular frame support with configurable dimensions

Commercial-grade WebSocket streaming for live visualization.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger
from pydantic import BaseModel


# =============================================================================
# CONSTANTS
# =============================================================================
EARTH_GRAVITY = 9.80665  # m/s²


# =============================================================================
# DATA MODELS
# =============================================================================

class FrameDimensions(BaseModel):
    """Rectangular frame dimensions."""
    inner_length: float = 0.30  # m
    inner_breadth: float = 0.20  # m
    outer_length: float = 0.50  # m
    outer_breadth: float = 0.35  # m


class AxisInclination(BaseModel):
    """Axis inclination angles (degrees) - tilt of rotation axes."""
    inner_axis_tilt: float = 0.0  # Tilt of inner frame rotation axis from Z
    outer_axis_tilt: float = 0.0  # Tilt of outer frame rotation axis from Y


class SimulationConfig(BaseModel):
    inner_rpm: float = 2.0
    outer_rpm: float = 2.0
    mode: str = "clinostat_3d"
    frame_dimensions: Optional[FrameDimensions] = None
    axis_inclination: Optional[AxisInclination] = None


class ConnectionInfo(BaseModel):
    """Hardware connection information."""
    type: str  # serial, network, usb, gpio
    port: str
    status: str  # connected, disconnected, error
    device_name: str
    baudrate: Optional[int] = None
    ip_address: Optional[str] = None


class SimulationState(BaseModel):
    timestamp: float
    time_elapsed: float
    # Frame rotation angles (phi and psi from paper)
    phi_deg: float  # Inner frame angle (Z-axis rotation)
    psi_deg: float  # Outer frame angle (Y-axis rotation)
    omega_phi_rpm: float  # Inner frame angular velocity
    omega_psi_rpm: float  # Outer frame angular velocity
    # Axis inclination angles
    inner_axis_tilt_deg: float
    outer_axis_tilt_deg: float
    # Gravity vector components (normalized, in sample frame) - from Eq. (2)
    gravity_x: float
    gravity_y: float
    gravity_z: float
    # Time-averaged gravity vector (KEY METRIC)
    avg_gravity_x: float
    avg_gravity_y: float
    avg_gravity_z: float
    # Microgravity quality metrics
    instantaneous_g: float
    mean_g: float  # ||time-averaged gravity vector|| - TRUE MICROGRAVITY METRIC (taSMG)
    min_g: float
    max_g: float
    samples: int
    status: str
    # Frame dimensions
    inner_length: float
    inner_breadth: float
    outer_length: float
    outer_breadth: float
    # Velocity ratio gamma = omega_phi / omega_psi (Eq. 5)
    gamma: float


# =============================================================================
# ROTATION MATRIX CALCULATIONS - FROM PAPER Eq. (1)
# =============================================================================

class RotationMatrices:
    """
    Proper rotation matrices based on research paper equations.
    
    From the paper:
    - R_Y (outer frame): rotation around Y-axis by angle φ (psi)
    - R_Z (inner frame): rotation around Z-axis by angle φ (phi)
    
    R_Y = | cos(ψ)   0   sin(ψ) |    R_Z = | cos(φ)  -sin(φ)  0 |
          |   0      1     0    |          | sin(φ)   cos(φ)  0 |
          |-sin(ψ)   0   cos(ψ) |          |   0        0     1 |
    
    The gravity vector in sample frame (Eq. 2):
    ē_g = -sin(φ)cos(ψ)ē_x + sin(φ)sin(ψ)ē_y + cos(φ)ē_z
    
    Where gravity in lab frame points along -Z direction.
    """
    
    @staticmethod
    def R_Y(psi: float) -> np.ndarray:
        """
        Rotation around Y-axis (outer frame) - Eq from paper.
        psi (ψ): outer frame angle in radians
        """
        c, s = np.cos(psi), np.sin(psi)
        return np.array([
            [c,  0, s],
            [0,  1, 0],
            [-s, 0, c]
        ], dtype=np.float64)
    
    @staticmethod
    def R_Z(phi: float) -> np.ndarray:
        """
        Rotation around Z-axis (inner frame) - Eq from paper.
        phi (φ): inner frame angle in radians
        """
        c, s = np.cos(phi), np.sin(phi)
        return np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ], dtype=np.float64)
    
    @staticmethod
    def R_X(theta: float) -> np.ndarray:
        """Rotation around X-axis (for axis tilting)."""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [1,  0,  0],
            [0,  c, -s],
            [0,  s,  c]
        ], dtype=np.float64)
    
    @staticmethod
    def compute_gravity_vector(phi: float, psi: float) -> np.ndarray:
        """
        Compute gravity vector in sample frame using Eq. (2) from paper.
        
        ē_g = -sin(φ)cos(ψ)ē_x + sin(φ)sin(ψ)ē_y + cos(φ)ē_z
        
        Args:
            phi: inner frame angle (radians)
            psi: outer frame angle (radians)
        
        Returns:
            Normalized gravity vector [gx, gy, gz]
        """
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        sin_psi = np.sin(psi)
        cos_psi = np.cos(psi)
        
        gx = -sin_phi * cos_psi
        gy = sin_phi * sin_psi
        gz = cos_phi
        
        return np.array([gx, gy, gz], dtype=np.float64)


# =============================================================================
# MICROGRAVITY PHYSICS ENGINE - Based on Paper Equations
# =============================================================================

class MicrogravitySimulator:
    """
    Proper microgravity simulation based on RPM physics from the paper.
    
    Key equations:
    - Gravity vector (Eq. 2): ē_g = -sin(φ)cos(ψ)ē_x + sin(φ)sin(ψ)ē_y + cos(φ)ē_z
    - Angular velocities: ω_φ = dφ/dt (inner), ω_ψ = dψ/dt (outer)
    - Velocity ratio (Eq. 5): γ = ω_φ / ω_ψ
    
    For good microgravity (low taSMG):
    - γ should be irrational (e.g., √2, π/e) to avoid repetitive paths
    - Velocities between 0.05-0.4 rad/s (0.5-4 RPM)
    
    Critical: The INSTANTANEOUS gravity magnitude is ALWAYS 1g.
    Microgravity is achieved through TIME-AVERAGED VECTOR approaching zero.
    """
    
    def __init__(self):
        # Frame angles (radians) - φ (phi) and ψ (psi) from paper
        self.phi = 0.0      # Inner frame angle (Z-axis rotation)
        self.psi = 0.0      # Outer frame angle (Y-axis rotation)
        
        # Angular velocities (rad/s) - ω_φ and ω_ψ
        self.omega_phi = 0.0    # Inner frame angular velocity
        self.omega_psi = 0.0    # Outer frame angular velocity
        
        # Setpoints
        self.omega_phi_setpoint = 0.0
        self.omega_psi_setpoint = 0.0
        
        # Axis inclination angles (for tilted rotation axes)
        self.inner_axis_tilt = 0.0  # Tilt of inner frame axis (radians)
        self.outer_axis_tilt = 0.0  # Tilt of outer frame axis (radians)
        
        # Time tracking
        self.time = 0.0
        self.start_time = 0.0
        
        # Gravity vector history for proper time-averaging
        self.gravity_history: List[np.ndarray] = []
        self.max_history = 6000  # 2 minutes at 50Hz
        
        # Running sum for efficient averaging
        self.gravity_sum = np.zeros(3)
        
        # Frame dimensions (rectangular)
        self.inner_length = 0.30
        self.inner_breadth = 0.20
        self.outer_length = 0.50
        self.outer_breadth = 0.35
        
        # Min/Max tracking for taSMG
        self.min_g = float('inf')
        self.max_g = 0.0
        
        logger.info("Microgravity Simulator initialized with paper equations")
    
    def set_velocities(self, inner_rpm: float, outer_rpm: float):
        """Set target velocities in RPM."""
        self.omega_phi_setpoint = inner_rpm * 2 * np.pi / 60  # rad/s
        self.omega_psi_setpoint = outer_rpm * 2 * np.pi / 60
        logger.info(f"Velocities set: ω_φ={inner_rpm:.2f} RPM, ω_ψ={outer_rpm:.2f} RPM")
    
    def set_axis_inclination(self, inner_tilt_deg: float, outer_tilt_deg: float):
        """Set axis inclination angles in degrees."""
        self.inner_axis_tilt = np.radians(inner_tilt_deg)
        self.outer_axis_tilt = np.radians(outer_tilt_deg)
        logger.info(f"Axis inclination: inner={inner_tilt_deg}°, outer={outer_tilt_deg}°")
    
    def set_frame_dimensions(self, inner_l: float, inner_b: float, 
                             outer_l: float, outer_b: float):
        """Set rectangular frame dimensions in meters."""
        self.inner_length = inner_l
        self.inner_breadth = inner_b
        self.outer_length = outer_l
        self.outer_breadth = outer_b
    
    def reset(self):
        """Reset simulation state."""
        self.phi = 0.0
        self.psi = 0.0
        self.omega_phi = 0.0
        self.omega_psi = 0.0
        self.time = 0.0
        self.start_time = time.time()
        self.gravity_history.clear()
        self.gravity_sum = np.zeros(3)
        self.min_g = float('inf')
        self.max_g = 0.0
        logger.info("Simulation reset")
    
    def get_gamma(self) -> float:
        """Calculate velocity ratio γ = ω_φ / ω_ψ (Eq. 5)."""
        if abs(self.omega_psi) < 1e-9:
            return float('inf') if self.omega_phi > 0 else 0.0
        return self.omega_phi / self.omega_psi
    
    def step(self, dt: float) -> dict:
        """
        Advance simulation by dt seconds.
        
        Uses proper physics from paper:
        - Update angles: φ += ω_φ·dt, ψ += ω_ψ·dt
        - Compute gravity using Eq. (2)
        
        Returns dict with all simulation state.
        """
        # First-order velocity dynamics (motor response)
        tau = 0.3  # Time constant
        self.omega_phi += (self.omega_phi_setpoint - self.omega_phi) * dt / tau
        self.omega_psi += (self.omega_psi_setpoint - self.omega_psi) * dt / tau
        
        # Update angles (continuous rotation - don't wrap!)
        self.phi += self.omega_phi * dt
        self.psi += self.omega_psi * dt
        
        # Compute gravity in sample frame using Eq. (2) from paper
        # Apply axis inclination if set
        if abs(self.inner_axis_tilt) > 1e-6 or abs(self.outer_axis_tilt) > 1e-6:
            # Modified rotation with tilted axes
            # First apply outer frame rotation around tilted Y-axis
            R_tilt_outer = RotationMatrices.R_X(self.outer_axis_tilt)
            R_outer = R_tilt_outer @ RotationMatrices.R_Y(self.psi) @ R_tilt_outer.T
            
            # Then inner frame rotation around tilted Z-axis
            R_tilt_inner = RotationMatrices.R_X(self.inner_axis_tilt)
            R_inner = R_tilt_inner @ RotationMatrices.R_Z(self.phi) @ R_tilt_inner.T
            
            # Combined rotation applied to lab gravity [0, 0, -1]
            R_total = R_inner @ R_outer
            g_sample = R_total @ np.array([0, 0, -1])
        else:
            # Standard formula from Eq. (2)
            g_sample = RotationMatrices.compute_gravity_vector(self.phi, self.psi)
        
        # Ensure unit vector
        g_mag = np.linalg.norm(g_sample)
        if g_mag > 0:
            g_sample = g_sample / g_mag
        
        # Update history (for time-averaged vector)
        self.gravity_sum += g_sample
        self.gravity_history.append(g_sample.copy())
        
        if len(self.gravity_history) > self.max_history:
            oldest = self.gravity_history.pop(0)
            self.gravity_sum -= oldest
        
        # Compute time-averaged gravity vector (taSMG)
        n_samples = len(self.gravity_history)
        avg_gravity = self.gravity_sum / n_samples if n_samples > 0 else np.zeros(3)
        
        # TRUE MICROGRAVITY METRIC (taSMG): magnitude of time-averaged vector
        mean_g = np.linalg.norm(avg_gravity)
        
        # Track min/max of taSMG
        if n_samples > 50:  # Wait for samples to stabilize
            self.min_g = min(self.min_g, mean_g)
            self.max_g = max(self.max_g, mean_g)
        
        self.time += dt
        
        # Velocity ratio γ
        gamma = self.get_gamma()
        
        return {
            "phi": self.phi,
            "psi": self.psi,
            "omega_phi": self.omega_phi,
            "omega_psi": self.omega_psi,
            "inner_axis_tilt": self.inner_axis_tilt,
            "outer_axis_tilt": self.outer_axis_tilt,
            "gravity_sample": g_sample,
            "avg_gravity": avg_gravity,
            "instant_g": 1.0,  # Always 1g instantaneously
            "mean_g": mean_g,  # taSMG
            "min_g": self.min_g if self.min_g != float('inf') else mean_g,
            "max_g": self.max_g if self.max_g > 0 else mean_g,
            "n_samples": n_samples,
            "time": self.time,
            "gamma": gamma
        }


# =============================================================================
# CONNECTION MANAGER
# =============================================================================

class HardwareCommandQueue:
    """
    Thread-safe command queue for hardware communication.
    Prevents bottlenecks by batching/throttling commands.
    """
    
    def __init__(self, max_rate: float = 50.0):
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self.max_rate = max_rate  # Max commands per second
        self.min_interval = 1.0 / max_rate
        self.last_send_time = 0.0
        self._running = False
        self._task = None
        
        # Latest values (for coalescing rapid updates)
        self._latest_velocities = None
        self._latest_tilt = None
        
    async def start(self):
        """Start the command processor."""
        self._running = True
        self._task = asyncio.create_task(self._process_queue())
        logger.info("Hardware command queue started")
    
    async def stop(self):
        """Stop the command processor."""
        self._running = False
        if self._task:
            self._task.cancel()
            
    async def _process_queue(self):
        """Process commands from queue with rate limiting."""
        while self._running:
            try:
                # Wait for command
                cmd = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                
                # Rate limiting
                now = time.time()
                elapsed = now - self.last_send_time
                if elapsed < self.min_interval:
                    await asyncio.sleep(self.min_interval - elapsed)
                
                # Send to hardware
                await self._send_to_hardware(cmd)
                self.last_send_time = time.time()
                
            except asyncio.TimeoutError:
                # Check for coalesced commands
                await self._send_coalesced_commands()
            except Exception as e:
                logger.error(f"Command queue error: {e}")
    
    async def _send_coalesced_commands(self):
        """Send any coalesced (batched) commands."""
        if self._latest_velocities:
            await self._send_to_hardware({
                'type': 'velocity',
                'data': self._latest_velocities
            })
            self._latest_velocities = None
            
        if self._latest_tilt:
            await self._send_to_hardware({
                'type': 'tilt',
                'data': self._latest_tilt
            })
            self._latest_tilt = None
    
    async def _send_to_hardware(self, cmd: dict):
        """
        Actual hardware communication.
        Override this for real hardware integration.
        """
        cmd_type = cmd.get('type', 'unknown')
        data = cmd.get('data', {})
        
        # Log the command (in production, send to serial/network)
        logger.debug(f"HW CMD [{cmd_type}]: {data}")
        
        # Placeholder for real hardware communication:
        # if self.serial_connection:
        #     self.serial_connection.write(json.dumps(cmd).encode())
        # elif self.network_connection:
        #     await self.network_connection.send(json.dumps(cmd))
    
    def queue_velocity(self, inner_rpm: float, outer_rpm: float):
        """Queue velocity command (coalesces rapid updates)."""
        self._latest_velocities = {
            'inner_rpm': inner_rpm,
            'outer_rpm': outer_rpm,
            'timestamp': time.time()
        }
        
    def queue_tilt(self, inner_tilt: float, outer_tilt: float):
        """Queue tilt command (coalesces rapid updates)."""
        self._latest_tilt = {
            'inner_tilt': inner_tilt,
            'outer_tilt': outer_tilt,
            'timestamp': time.time()
        }
        
    async def queue_immediate(self, cmd_type: str, data: dict):
        """Queue a command that needs immediate processing."""
        try:
            self.queue.put_nowait({
                'type': cmd_type,
                'data': data,
                'timestamp': time.time()
            })
        except asyncio.QueueFull:
            logger.warning(f"Command queue full, dropping {cmd_type}")


class ConnectionManager:
    """Manages hardware connections (serial, network, etc.)."""
    
    def __init__(self):
        self.connections: Dict[str, ConnectionInfo] = {}
        self._scan_interval = 5.0  # seconds
        self._last_scan = 0.0
        self.command_queue = HardwareCommandQueue()
    
    async def scan_connections(self) -> Dict[str, ConnectionInfo]:
        """Scan for available hardware connections."""
        import os
        import glob
        
        connections = {}
        
        # Scan serial ports
        try:
            if sys.platform == 'darwin':
                serial_ports = glob.glob('/dev/tty.usb*') + glob.glob('/dev/cu.usb*')
            elif sys.platform.startswith('linux'):
                serial_ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
            else:
                serial_ports = [f'COM{i}' for i in range(1, 20)]
            
            for port in serial_ports:
                conn_id = f"serial_{port.replace('/', '_')}"
                connections[conn_id] = ConnectionInfo(
                    type="serial",
                    port=port,
                    status="available",
                    device_name="Serial Device",
                    baudrate=115200
                )
        except Exception as e:
            logger.warning(f"Serial scan error: {e}")
        
        # Scan network (localhost RPM service)
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex(('localhost', 5000))
            sock.close()
            
            connections["network_localhost_5000"] = ConnectionInfo(
                type="network",
                port="5000",
                status="connected" if result == 0 else "disconnected",
                device_name="RPM Control Service",
                ip_address="127.0.0.1"
            )
        except Exception as e:
            logger.debug(f"Network scan: {e}")
        
        # gRPC service check
        connections["grpc_localhost_50051"] = ConnectionInfo(
            type="network",
            port="50051",
            status="disconnected",  # Would need actual check
            device_name="gRPC Streaming Service",
            ip_address="127.0.0.1"
        )
        
        # Simulated hardware connections
        connections["usb_rpm_controller"] = ConnectionInfo(
            type="usb",
            port="USB0",
            status="simulated",
            device_name="RPM Motor Controller (Simulated)"
        )
        
        connections["gpio_raspberry_pi"] = ConnectionInfo(
            type="gpio",
            port="GPIO",
            status="not_available",
            device_name="Raspberry Pi GPIO"
        )
        
        self.connections = connections
        return connections
    
    def get_connections_list(self) -> List[dict]:
        """Get connections as list of dicts."""
        return [conn.model_dump() for conn in self.connections.values()]


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

class SimulationEngine:
    """
    Real-time simulation manager with WebSocket broadcasting.
    
    Features:
    - 50Hz physics simulation
    - 20Hz client broadcast (reduces network load)
    - Hardware command queueing (prevents bottlenecks)
    - Thread-safe state management
    """
    
    def __init__(self):
        self.simulator = MicrogravitySimulator()
        self.connection_manager = ConnectionManager()
        
        self.running = False
        self.dt = 0.02  # 50 Hz simulation
        self.broadcast_rate = 20  # 20 Hz to clients
        
        # Connected clients
        self.clients: Set[WebSocket] = set()
        
        # Current config - using paper notation
        self.omega_phi_rpm = 2.0   # Inner frame velocity (φ rotation around Z)
        self.omega_psi_rpm = 2.0   # Outer frame velocity (ψ rotation around Y)
        
        # Axis inclination
        self.inner_tilt_deg = 0.0
        self.outer_tilt_deg = 0.0
        
        # Target values for smooth transitions (server-side too)
        self._target_inner_rpm = 2.0
        self._target_outer_rpm = 2.0
        self._target_inner_tilt = 0.0
        self._target_outer_tilt = 0.0
        
        self.start_time = 0.0
        
        logger.info("Simulation engine initialized with paper equations (Eq. 2)")
    
    async def start_hardware_queue(self):
        """Start the hardware command queue processor."""
        await self.connection_manager.command_queue.start()
    
    async def stop_hardware_queue(self):
        """Stop the hardware command queue processor."""
        await self.connection_manager.command_queue.stop()
    
    def set_velocities(self, inner_rpm: float, outer_rpm: float):
        """
        Update target velocities in RPM.
        Queues command to hardware to prevent bottlenecks.
        """
        self.omega_phi_rpm = inner_rpm
        self.omega_psi_rpm = outer_rpm
        self.simulator.set_velocities(inner_rpm, outer_rpm)
        
        # Queue for hardware (coalesces rapid updates)
        self.connection_manager.command_queue.queue_velocity(inner_rpm, outer_rpm)
        logger.info(f"Velocities set: ω_φ={inner_rpm:.2f} RPM, ω_ψ={outer_rpm:.2f} RPM")
    
    def set_axis_inclination(self, inner_tilt: float, outer_tilt: float):
        """
        Set axis inclination angles in degrees.
        Queues command to hardware to prevent bottlenecks.
        """
        self.inner_tilt_deg = inner_tilt
        self.outer_tilt_deg = outer_tilt
        self.simulator.set_axis_inclination(inner_tilt, outer_tilt)
        
        # Queue for hardware (coalesces rapid updates)
        self.connection_manager.command_queue.queue_tilt(inner_tilt, outer_tilt)
        logger.info(f"Axis inclination: inner={inner_tilt}°, outer={outer_tilt}°")
    
    def set_frame_dimensions(self, inner_l: float, inner_b: float,
                            outer_l: float, outer_b: float):
        """Set frame dimensions."""
        self.simulator.set_frame_dimensions(inner_l, inner_b, outer_l, outer_b)
    
    def start(self):
        """Start simulation."""
        self.running = True
        self.start_time = time.time()
        self.simulator.start_time = self.start_time
        self.set_velocities(self.omega_phi_rpm, self.omega_psi_rpm)
        logger.info("Simulation started")
    
    def stop(self):
        """Stop simulation."""
        self.running = False
        logger.info("Simulation stopped")
    
    def reset(self):
        """Reset simulation state."""
        self.simulator.reset()
        self.start_time = time.time()
        self.set_velocities(self.omega_phi_rpm, self.omega_psi_rpm)
        logger.info("Simulation reset")
    
    def step(self) -> SimulationState:
        """Execute one simulation step using paper physics."""
        result = self.simulator.step(self.dt)
        
        # Convert angles to display degrees (mod 360 for display)
        phi_display = float(np.degrees(result["phi"])) % 360
        psi_display = float(np.degrees(result["psi"])) % 360
        
        return SimulationState(
            timestamp=time.time(),
            time_elapsed=time.time() - self.start_time,
            # Use phi/psi notation from paper
            phi_deg=phi_display,
            psi_deg=psi_display,
            omega_phi_rpm=float(result["omega_phi"] * 30 / np.pi),
            omega_psi_rpm=float(result["omega_psi"] * 30 / np.pi),
            # Axis inclination
            inner_axis_tilt_deg=float(np.degrees(result["inner_axis_tilt"])),
            outer_axis_tilt_deg=float(np.degrees(result["outer_axis_tilt"])),
            # Gravity components
            gravity_x=float(result["gravity_sample"][0]),
            gravity_y=float(result["gravity_sample"][1]),
            gravity_z=float(result["gravity_sample"][2]),
            avg_gravity_x=float(result["avg_gravity"][0]),
            avg_gravity_y=float(result["avg_gravity"][1]),
            avg_gravity_z=float(result["avg_gravity"][2]),
            # Microgravity metrics
            instantaneous_g=float(result["instant_g"]),
            mean_g=float(result["mean_g"]),
            min_g=float(result["min_g"]),
            max_g=float(result["max_g"]),
            # Velocity ratio γ = ω_φ/ω_ψ
            gamma=float(result["gamma"]) if result["gamma"] != float('inf') else 999.9,
            samples=int(result["n_samples"]),
            status="running" if self.running else "stopped",
            # Frame dimensions
            inner_length=self.simulator.inner_length,
            inner_breadth=self.simulator.inner_breadth,
            outer_length=self.simulator.outer_length,
            outer_breadth=self.simulator.outer_breadth
        )
    
    async def broadcast(self, state: SimulationState):
        """Send state to all connected clients."""
        if not self.clients:
            return
        
        message = json.dumps(state.model_dump())
        disconnected = set()
        
        for client in self.clients:
            try:
                await client.send_text(message)
            except Exception:
                disconnected.add(client)
        
        self.clients -= disconnected
    
    async def run_loop(self):
        """Main simulation loop with broadcasting."""
        broadcast_interval = 1.0 / self.broadcast_rate
        last_broadcast = 0.0
        
        while True:
            if self.running:
                state = self.step()
                
                now = time.time()
                if now - last_broadcast >= broadcast_interval:
                    await self.broadcast(state)
                    last_broadcast = now
                
                await asyncio.sleep(self.dt)
            else:
                await asyncio.sleep(0.1)


# Global simulation engine
engine = SimulationEngine()


# =============================================================================
# APPLICATION LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    await engine.start_hardware_queue()
    asyncio.create_task(engine.run_loop())
    logger.info("RPM Digital Twin server started")
    logger.info("Hardware command queue active - ready for sensor connections")
    yield
    # Shutdown
    await engine.stop_hardware_queue()
    logger.info("RPM Digital Twin server stopped")


app = FastAPI(
    title="RPM Digital Twin",
    description="Commercial-grade Random Positioning Machine Control System",
    version="2.0.0",
    lifespan=lifespan
)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main dashboard."""
    html_path = Path(__file__).parent / "static" / "index.html"
    return FileResponse(html_path)


@app.get("/api/status")
async def get_status():
    """Get current simulation status."""
    return {
        "running": engine.running,
        "inner_rpm": engine.inner_rpm,
        "outer_rpm": engine.outer_rpm,
        "samples": engine.simulator.gravity_history.__len__(),
        "clients": len(engine.clients)
    }


@app.get("/api/connections")
async def get_connections():
    """Get available hardware connections."""
    connections = await engine.connection_manager.scan_connections()
    return {
        "connections": engine.connection_manager.get_connections_list(),
        "timestamp": time.time()
    }


@app.post("/api/start")
async def start_simulation():
    """Start the simulation."""
    engine.start()
    return {"status": "started"}


@app.post("/api/stop")
async def stop_simulation():
    """Stop the simulation."""
    engine.stop()
    return {"status": "stopped"}


@app.post("/api/reset")
async def reset_simulation():
    """Reset the simulation."""
    engine.reset()
    return {"status": "reset"}


@app.post("/api/config")
async def update_config(config: SimulationConfig):
    """Update simulation configuration."""
    engine.set_velocities(config.inner_rpm, config.outer_rpm)
    
    if config.frame_dimensions:
        engine.set_frame_dimensions(
            config.frame_dimensions.inner_length,
            config.frame_dimensions.inner_breadth,
            config.frame_dimensions.outer_length,
            config.frame_dimensions.outer_breadth
        )
    
    if config.axis_inclination:
        engine.set_axis_inclination(
            config.axis_inclination.inner_axis_tilt,
            config.axis_inclination.outer_axis_tilt
        )
    
    return {"status": "updated", "config": config.model_dump()}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data streaming."""
    await websocket.accept()
    engine.clients.add(websocket)
    logger.info(f"Client connected. Total clients: {len(engine.clients)}")
    
    try:
        while True:
            # Handle incoming commands
            data = await websocket.receive_text()
            try:
                cmd = json.loads(data)
                action = cmd.get("action")
                
                if action == "start":
                    engine.start()
                elif action == "stop":
                    engine.stop()
                elif action == "reset":
                    engine.reset()
                elif action == "config":
                    engine.set_velocities(
                        cmd.get("inner_rpm", 2.0),
                        cmd.get("outer_rpm", 2.0)
                    )
                elif action == "frame_dimensions":
                    engine.set_frame_dimensions(
                        cmd.get("inner_length", 0.30),
                        cmd.get("inner_breadth", 0.20),
                        cmd.get("outer_length", 0.50),
                        cmd.get("outer_breadth", 0.35)
                    )
                elif action == "axis_inclination":
                    engine.set_axis_inclination(
                        cmd.get("inner_tilt", 0.0),
                        cmd.get("outer_tilt", 0.0)
                    )
                elif action == "scan_connections":
                    connections = await engine.connection_manager.scan_connections()
                    await websocket.send_text(json.dumps({
                        "type": "connections",
                        "data": engine.connection_manager.get_connections_list()
                    }))
                    
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        engine.clients.discard(websocket)
        logger.info(f"Client disconnected. Total clients: {len(engine.clients)}")


# =============================================================================
# MAIN ENTRY
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
