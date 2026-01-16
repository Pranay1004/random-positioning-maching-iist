"""
RPM Physics Simulation Engine
==============================
Core physics engine for Random Positioning Machine digital twin simulation.

This module implements the mathematical model for:
- 3D rotation kinematics with nested gimbals
- Gravity vector computation at any point in space
- Time-averaged microgravity calculation
- Centrifugal acceleration effects

Based on theoretical framework from:
"A New Random Positioning Machine Modification Applied for Microgravity 
Simulation in Laboratory Experiments with Rats" - Yotov et al.

Mathematical Model:
-------------------
The RPM consists of two frames:
- Outer frame: Rotates around Y-axis with angular velocity ω_o
- Inner frame: Rotates around X-axis with angular velocity ω_i

The effective gravity vector at any point is computed using
rotation matrices and includes centrifugal accelerations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Callable
from enum import Enum
import time
from scipy.spatial.transform import Rotation
from scipy.integrate import solve_ivp
from loguru import logger


# Physical constants
EARTH_GRAVITY = 9.80665  # m/s²


class OperationMode(Enum):
    """RPM operation modes."""
    CLINOSTAT_2D = "clinostat_2d"
    CLINOSTAT_3D = "clinostat_3d"
    RANDOM = "random"
    PARTIAL_GRAVITY = "partial_gravity"
    CUSTOM = "custom"


@dataclass
class RPMGeometry:
    """
    Physical geometry parameters of the RPM.
    
    Attributes:
        inner_frame_radius: Radius of inner frame (m)
        outer_frame_radius: Radius of outer frame (m)
        sample_mount_offset: Offset from center to sample mount (m)
    """
    inner_frame_radius: float = 0.15  # m
    outer_frame_radius: float = 0.25  # m
    sample_mount_offset: NDArray = field(default_factory=lambda: np.array([0.0, 0.0, 0.10]))
    
    # Mass properties for inertia calculations
    inner_frame_mass: float = 2.0  # kg
    outer_frame_mass: float = 3.0  # kg
    sample_mass: float = 0.5  # kg
    
    def __post_init__(self):
        self.sample_mount_offset = np.asarray(self.sample_mount_offset)


@dataclass
class RPMState:
    """
    Complete state of the RPM at a given instant.
    
    All angles are in radians, angular velocities in rad/s.
    """
    # Time
    time: float = 0.0
    
    # Inner frame (rotates around local X-axis)
    theta_inner: float = 0.0  # rad
    omega_inner: float = 0.0  # rad/s
    alpha_inner: float = 0.0  # rad/s² (acceleration)
    
    # Outer frame (rotates around fixed Y-axis)
    theta_outer: float = 0.0  # rad
    omega_outer: float = 0.0  # rad/s
    alpha_outer: float = 0.0  # rad/s²
    
    # Setpoints
    omega_inner_setpoint: float = 0.0
    omega_outer_setpoint: float = 0.0
    
    def to_array(self) -> NDArray:
        """Convert state to numpy array [θi, ωi, θo, ωo]."""
        return np.array([
            self.theta_inner, self.omega_inner,
            self.theta_outer, self.omega_outer
        ])
    
    @classmethod
    def from_array(cls, arr: NDArray, time: float = 0.0) -> RPMState:
        """Create state from numpy array."""
        return cls(
            time=time,
            theta_inner=arr[0],
            omega_inner=arr[1],
            theta_outer=arr[2],
            omega_outer=arr[3]
        )


class RotationMatrices:
    """
    Rotation matrix utilities for RPM kinematics.
    
    Coordinate System:
    - X: Points right
    - Y: Points up (vertical)
    - Z: Points out of page
    
    Gravity acts in -Y direction.
    """
    
    @staticmethod
    def Rx(theta: float) -> NDArray:
        """
        Rotation matrix around X-axis (inner frame).
        
        Args:
            theta: Rotation angle in radians
            
        Returns:
            3x3 rotation matrix
        """
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    
    @staticmethod
    def Ry(theta: float) -> NDArray:
        """
        Rotation matrix around Y-axis (outer frame).
        
        Args:
            theta: Rotation angle in radians
            
        Returns:
            3x3 rotation matrix
        """
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    
    @staticmethod
    def Rz(theta: float) -> NDArray:
        """
        Rotation matrix around Z-axis.
        
        Args:
            theta: Rotation angle in radians
            
        Returns:
            3x3 rotation matrix
        """
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    
    @staticmethod
    def combined_rotation(theta_inner: float, theta_outer: float) -> NDArray:
        """
        Combined rotation matrix for RPM.
        
        R_total = R_y(θ_outer) @ R_x(θ_inner)
        
        This represents the orientation of the sample platform
        in the lab frame.
        
        Args:
            theta_inner: Inner frame angle (rad)
            theta_outer: Outer frame angle (rad)
            
        Returns:
            3x3 combined rotation matrix
        """
        return RotationMatrices.Ry(theta_outer) @ RotationMatrices.Rx(theta_inner)


class GravityVectorCalculator:
    """
    Computes the effective gravity vector experienced at any point on the RPM.
    
    The effective gravity includes:
    1. Gravitational acceleration transformed to rotating frame
    2. Centrifugal acceleration from frame rotations
    
    For microgravity simulation, we want the time-averaged effective
    gravity magnitude to approach zero.
    """
    
    def __init__(self, geometry: RPMGeometry):
        """
        Initialize calculator with RPM geometry.
        
        Args:
            geometry: Physical parameters of the RPM
        """
        self.geometry = geometry
        self.g = np.array([0, -EARTH_GRAVITY, 0])  # Gravity in lab frame
        
    def compute_gravity_in_sample_frame(
        self,
        theta_inner: float,
        theta_outer: float
    ) -> NDArray:
        """
        Compute gravity vector as experienced in the sample (rotating) frame.
        
        This is the gravitational acceleration transformed from lab frame
        to the rotating sample frame.
        
        Args:
            theta_inner: Inner frame angle (rad)
            theta_outer: Outer frame angle (rad)
            
        Returns:
            Gravity vector in sample frame (m/s²)
        """
        # Rotation matrix from lab to sample frame is transpose of lab←sample
        R = RotationMatrices.combined_rotation(theta_inner, theta_outer)
        g_sample = R.T @ self.g
        return g_sample
    
    def compute_centrifugal_acceleration(
        self,
        position: NDArray,
        omega_inner: float,
        omega_outer: float,
        theta_inner: float,
        theta_outer: float
    ) -> NDArray:
        """
        Compute centrifugal acceleration at a point.
        
        a_cent = -ω × (ω × r)
        
        For the RPM with two axes, this is more complex due to
        the nested rotation.
        
        Args:
            position: Point position in sample frame (m)
            omega_inner: Inner frame angular velocity (rad/s)
            omega_outer: Outer frame angular velocity (rad/s)
            theta_inner: Inner frame angle (rad)
            theta_outer: Outer frame angle (rad)
            
        Returns:
            Centrifugal acceleration vector (m/s²)
        """
        # Angular velocity of inner frame in lab frame
        omega_i_lab = np.array([omega_inner, 0, 0])
        
        # Angular velocity of outer frame in lab frame
        omega_o_lab = np.array([0, omega_outer, 0])
        
        # Transform inner angular velocity through outer frame rotation
        R_outer = RotationMatrices.Ry(theta_outer)
        omega_i_lab_transformed = R_outer @ omega_i_lab
        
        # Total angular velocity in lab frame
        omega_total = omega_i_lab_transformed + omega_o_lab
        
        # Position in lab frame
        R_combined = RotationMatrices.combined_rotation(theta_inner, theta_outer)
        position_lab = R_combined @ position
        
        # Centrifugal acceleration: -ω × (ω × r)
        a_cent_lab = -np.cross(omega_total, np.cross(omega_total, position_lab))
        
        # Transform back to sample frame
        a_cent_sample = R_combined.T @ a_cent_lab
        
        return a_cent_sample
    
    def compute_effective_gravity(
        self,
        state: RPMState,
        position: Optional[NDArray] = None
    ) -> NDArray:
        """
        Compute total effective gravity (including centrifugal effects).
        
        Args:
            state: Current RPM state
            position: Point position in sample frame (default: sample mount)
            
        Returns:
            Effective gravity vector in sample frame (m/s²)
        """
        if position is None:
            position = self.geometry.sample_mount_offset
            
        # Gravitational component
        g_sample = self.compute_gravity_in_sample_frame(
            state.theta_inner, state.theta_outer
        )
        
        # Centrifugal component
        a_cent = self.compute_centrifugal_acceleration(
            position,
            state.omega_inner,
            state.omega_outer,
            state.theta_inner,
            state.theta_outer
        )
        
        return g_sample + a_cent
    
    def compute_g_magnitude(
        self,
        state: RPMState,
        position: Optional[NDArray] = None
    ) -> float:
        """
        Compute magnitude of effective gravity in g-units.
        
        Args:
            state: Current RPM state
            position: Point position
            
        Returns:
            Effective gravity magnitude in g-units (1g = 9.81 m/s²)
        """
        g_eff = self.compute_effective_gravity(state, position)
        return np.linalg.norm(g_eff) / EARTH_GRAVITY


class RPMSimulator:
    """
    Main RPM physics simulator.
    
    Simulates the time evolution of the RPM and computes
    microgravity metrics.
    
    Usage:
        simulator = RPMSimulator()
        simulator.set_mode(OperationMode.RANDOM)
        
        # Run simulation
        for state in simulator.simulate(duration=60.0, dt=0.01):
            g = simulator.get_instantaneous_g()
            print(f"t={state.time:.2f}, g={g:.4f}")
    """
    
    def __init__(
        self,
        geometry: Optional[RPMGeometry] = None,
        mode: OperationMode = OperationMode.CLINOSTAT_3D
    ):
        """
        Initialize RPM simulator.
        
        Args:
            geometry: RPM physical parameters
            mode: Operation mode
        """
        self.geometry = geometry or RPMGeometry()
        self.mode = mode
        self.state = RPMState()
        self.gravity_calc = GravityVectorCalculator(self.geometry)
        
        # Mode-specific parameters
        self._mode_params = {
            OperationMode.CLINOSTAT_3D: {
                "omega_inner": 2.0 * np.pi / 60,  # 1 RPM
                "omega_outer": 2.0 * np.pi / 60,  # 1 RPM
            },
            OperationMode.RANDOM: {
                "omega_range": (0.5 * np.pi / 30, 5.0 * np.pi / 30),  # 1-10 RPM in rad/s
                "direction_change_interval": (2.0, 10.0),
            }
        }
        
        # History for time-averaging
        self._gravity_history: List[Tuple[float, float]] = []
        self._history_window = 60.0  # seconds - ADAPTIVE when motors stop
        self._motor_stop_threshold = 0.01  # rad/s - below this is "stopped"
        self._was_moving = False  # Track state transition
        
        # Random mode state
        self._next_direction_change = 0.0
        self._rng = np.random.default_rng()
        
        logger.info(f"RPM Simulator initialized in {mode.value} mode")
    
    def set_mode(self, mode: OperationMode, params: Optional[dict] = None) -> None:
        """
        Set operation mode.
        
        Args:
            mode: Operation mode
            params: Optional mode-specific parameters
        """
        self.mode = mode
        if params:
            self._mode_params[mode] = params
        
        # Initialize mode
        if mode == OperationMode.CLINOSTAT_3D:
            p = self._mode_params[mode]
            self.state.omega_inner_setpoint = p["omega_inner"]
            self.state.omega_outer_setpoint = p["omega_outer"]
            
        logger.info(f"Mode set to {mode.value}")
    
    def set_velocities(
        self,
        omega_inner: float,
        omega_outer: float,
        unit: str = "rpm"
    ) -> None:
        """
        Set frame angular velocities.
        
        Args:
            omega_inner: Inner frame velocity
            omega_outer: Outer frame velocity
            unit: "rpm" or "rad/s"
        """
        if unit == "rpm":
            omega_inner = omega_inner * 2 * np.pi / 60
            omega_outer = omega_outer * 2 * np.pi / 60
            
        self.state.omega_inner_setpoint = omega_inner
        self.state.omega_outer_setpoint = omega_outer
        
        logger.debug(f"Velocities set: inner={omega_inner:.3f} rad/s, outer={omega_outer:.3f} rad/s")
    
    def step(self, dt: float) -> RPMState:
        """
        Advance simulation by one timestep.
        
        Args:
            dt: Time step (seconds)
            
        Returns:
            Updated state
        """
        # Update mode-specific behavior
        self._update_mode_behavior(dt)
        
        # Simple velocity control (first-order dynamics)
        tau = 0.5  # Time constant for velocity response
        
        self.state.omega_inner += (self.state.omega_inner_setpoint - self.state.omega_inner) * dt / tau
        self.state.omega_outer += (self.state.omega_outer_setpoint - self.state.omega_outer) * dt / tau
        
        # Update positions
        self.state.theta_inner += self.state.omega_inner * dt
        self.state.theta_outer += self.state.omega_outer * dt
        
        # Wrap angles to [-π, π]
        self.state.theta_inner = np.arctan2(
            np.sin(self.state.theta_inner),
            np.cos(self.state.theta_inner)
        )
        self.state.theta_outer = np.arctan2(
            np.sin(self.state.theta_outer),
            np.cos(self.state.theta_outer)
        )
        
        # Update time
        self.state.time += dt
        
        # Record gravity for averaging
        g = self.get_instantaneous_g()
        self._gravity_history.append((self.state.time, g))
        
        # **ADAPTIVE HISTORY MANAGEMENT**
        # Detect if motors just stopped
        is_moving = (abs(self.state.omega_inner) > self._motor_stop_threshold or 
                     abs(self.state.omega_outer) > self._motor_stop_threshold)
        
        if not is_moving and self._was_moving:
            # Motors just stopped - clear old history to show immediate 1g
            logger.info("Motors stopped - clearing history for immediate gravity recovery")
            self._gravity_history = [(self.state.time, g)]
            self._was_moving = False
        elif is_moving and not self._was_moving:
            # Motors just started - keep some baseline
            logger.info("Motors started - resuming 60s averaging window")
            self._was_moving = True
        elif is_moving:
            # Motors moving - use full 60s window
            self._was_moving = True
            cutoff = self.state.time - self._history_window
            self._gravity_history = [(t, g) for t, g in self._gravity_history if t > cutoff]
        else:
            # Motors stopped - maintain short-term history only (5 seconds)
            # This shows steady 1g instead of decaying from old rotations
            cutoff = self.state.time - 5.0
            self._gravity_history = [(t, g) for t, g in self._gravity_history if t > cutoff]
        
        return self.state
    
    def _update_mode_behavior(self, dt: float) -> None:
        """Update mode-specific behavior (e.g., random direction changes)."""
        if self.mode == OperationMode.RANDOM:
            if self.state.time >= self._next_direction_change:
                params = self._mode_params[OperationMode.RANDOM]
                omega_range = params["omega_range"]
                interval_range = params["direction_change_interval"]
                
                # Random velocities with random directions
                self.state.omega_inner_setpoint = self._rng.uniform(*omega_range) * self._rng.choice([-1, 1])
                self.state.omega_outer_setpoint = self._rng.uniform(*omega_range) * self._rng.choice([-1, 1])
                
                # Schedule next change
                self._next_direction_change = self.state.time + self._rng.uniform(*interval_range)
                
                logger.trace(f"Random direction change: ωi={self.state.omega_inner_setpoint:.3f}, ωo={self.state.omega_outer_setpoint:.3f}")
    
    def simulate(
        self,
        duration: float,
        dt: float = 0.01,
        callback: Optional[Callable[[RPMState], None]] = None
    ) -> List[RPMState]:
        """
        Run simulation for specified duration.
        
        Args:
            duration: Simulation duration (seconds)
            dt: Time step (seconds)
            callback: Optional callback called each step
            
        Yields:
            State at each timestep
        """
        states = []
        n_steps = int(duration / dt)
        
        for i in range(n_steps):
            state = self.step(dt)
            states.append(RPMState(
                time=state.time,
                theta_inner=state.theta_inner,
                theta_outer=state.theta_outer,
                omega_inner=state.omega_inner,
                omega_outer=state.omega_outer
            ))
            
            if callback:
                callback(state)
                
        return states
    
    def get_instantaneous_g(self, position: Optional[NDArray] = None) -> float:
        """
        Get instantaneous effective gravity magnitude.
        
        Args:
            position: Point in sample frame (m)
            
        Returns:
            Effective gravity in g-units
        """
        return self.gravity_calc.compute_g_magnitude(self.state, position)
    
    def get_gravity_vector(self, position: Optional[NDArray] = None) -> NDArray:
        """
        Get instantaneous effective gravity vector.
        
        Args:
            position: Point in sample frame (m)
            
        Returns:
            Gravity vector in sample frame (m/s²)
        """
        return self.gravity_calc.compute_effective_gravity(self.state, position)
    
    def get_time_averaged_g(self, window: Optional[float] = None) -> float:
        """
        Compute time-averaged effective gravity magnitude.
        
        This is the key metric for microgravity quality.
        For good simulation, this should be < 0.01g.
        
        Args:
            window: Averaging window in seconds (default: full history)
            
        Returns:
            Time-averaged gravity in g-units
        """
        if not self._gravity_history:
            return 0.0
            
        if window:
            cutoff = self.state.time - window
            history = [(t, g) for t, g in self._gravity_history if t > cutoff]
        else:
            history = self._gravity_history
            
        if not history:
            return 0.0
            
        # Simple time average
        g_values = [g for _, g in history]
        return np.mean(g_values)
    
    def get_gravity_std(self, window: Optional[float] = None) -> float:
        """
        Get standard deviation of gravity magnitude.
        
        Args:
            window: Averaging window in seconds
            
        Returns:
            Standard deviation of gravity in g-units
        """
        if not self._gravity_history:
            return 0.0
            
        if window:
            cutoff = self.state.time - window
            history = [(t, g) for t, g in self._gravity_history if t > cutoff]
        else:
            history = self._gravity_history
            
        if len(history) < 2:
            return 0.0
            
        g_values = [g for _, g in history]
        return np.std(g_values)
    
    def analyze_point(
        self,
        x: float,
        y: float,
        z: float,
        duration: float = 60.0,
        dt: float = 0.01
    ) -> dict:
        """
        Analyze microgravity conditions at a specific 3D point.
        
        This simulates the RPM and computes gravity statistics
        at the specified point.
        
        Args:
            x, y, z: Point coordinates in sample frame (m)
            duration: Analysis duration (s)
            dt: Time step (s)
            
        Returns:
            Dictionary with analysis results
        """
        position = np.array([x, y, z])
        
        # Reset simulator state
        initial_state = RPMState(
            omega_inner=self.state.omega_inner_setpoint,
            omega_outer=self.state.omega_outer_setpoint
        )
        self.state = initial_state
        self._gravity_history = []
        
        # Run simulation
        g_values = []
        g_vectors = []
        
        n_steps = int(duration / dt)
        for _ in range(n_steps):
            self.step(dt)
            g_values.append(self.get_instantaneous_g(position))
            g_vectors.append(self.get_gravity_vector(position))
        
        g_values = np.array(g_values)
        g_vectors = np.array(g_vectors)
        
        # Compute statistics
        results = {
            "position": {"x": x, "y": y, "z": z},
            "duration_s": duration,
            "samples": len(g_values),
            "mean_g": float(np.mean(g_values)),
            "std_g": float(np.std(g_values)),
            "min_g": float(np.min(g_values)),
            "max_g": float(np.max(g_values)),
            "median_g": float(np.median(g_values)),
            "mean_vector": g_vectors.mean(axis=0).tolist(),
            "mean_vector_magnitude": float(np.linalg.norm(g_vectors.mean(axis=0)) / EARTH_GRAVITY),
            "quality": "excellent" if np.mean(g_values) < 0.01 else 
                      "good" if np.mean(g_values) < 0.05 else
                      "acceptable" if np.mean(g_values) < 0.1 else "poor"
        }
        
        return results
    
    def compute_gravity_field(
        self,
        x_range: Tuple[float, float] = (-0.15, 0.15),
        y_range: Tuple[float, float] = (-0.15, 0.15),
        z_range: Tuple[float, float] = (0.0, 0.20),
        resolution: float = 0.02,
        duration: float = 60.0
    ) -> dict:
        """
        Compute gravity field over a 3D grid.
        
        This is useful for visualizing the microgravity quality
        throughout the sample volume.
        
        Args:
            x_range, y_range, z_range: Grid bounds (m)
            resolution: Grid spacing (m)
            duration: Simulation duration for each point (s)
            
        Returns:
            Dictionary with grid data
        """
        x = np.arange(x_range[0], x_range[1] + resolution, resolution)
        y = np.arange(y_range[0], y_range[1] + resolution, resolution)
        z = np.arange(z_range[0], z_range[1] + resolution, resolution)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        G = np.zeros_like(X)
        
        total_points = X.size
        logger.info(f"Computing gravity field for {total_points} points...")
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    result = self.analyze_point(X[i,j,k], Y[i,j,k], Z[i,j,k], duration=duration)
                    G[i,j,k] = result["mean_g"]
        
        return {
            "x": x.tolist(),
            "y": y.tolist(),
            "z": z.tolist(),
            "gravity_field": G.tolist(),
            "min_g": float(G.min()),
            "max_g": float(G.max()),
            "mean_g": float(G.mean()),
        }


# =============================================================================
# COMPARISON UTILITIES
# =============================================================================

class MicrogravityValidator:
    """
    Validates simulation results against experimental data.
    
    Provides methods for comparing numerical predictions with
    actual sensor measurements.
    """
    
    def __init__(self, simulator: RPMSimulator):
        """
        Initialize validator.
        
        Args:
            simulator: RPM simulator instance
        """
        self.simulator = simulator
        
    def compare_with_experiment(
        self,
        experimental_data: NDArray,
        experimental_timestamps: NDArray,
        position: NDArray
    ) -> dict:
        """
        Compare simulation with experimental measurements.
        
        Args:
            experimental_data: Measured gravity values (g-units)
            experimental_timestamps: Time stamps (s)
            position: Sensor position (m)
            
        Returns:
            Comparison statistics
        """
        # Run simulation for same duration
        duration = experimental_timestamps[-1] - experimental_timestamps[0]
        dt = np.mean(np.diff(experimental_timestamps))
        
        sim_states = self.simulator.simulate(duration, dt)
        sim_g = np.array([
            self.simulator.gravity_calc.compute_g_magnitude(state, position)
            for state in sim_states
        ])
        
        # Interpolate to match timestamps if needed
        if len(sim_g) != len(experimental_data):
            from scipy.interpolate import interp1d
            sim_times = np.linspace(0, duration, len(sim_g))
            f = interp1d(sim_times, sim_g, kind='linear', fill_value='extrapolate')
            sim_g_interp = f(experimental_timestamps - experimental_timestamps[0])
        else:
            sim_g_interp = sim_g
            
        # Compute comparison metrics
        error = sim_g_interp - experimental_data
        
        return {
            "mean_absolute_error": float(np.mean(np.abs(error))),
            "max_absolute_error": float(np.max(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "correlation": float(np.corrcoef(sim_g_interp, experimental_data)[0, 1]),
            "simulation_mean_g": float(np.mean(sim_g_interp)),
            "experimental_mean_g": float(np.mean(experimental_data)),
            "relative_error_percent": float(100 * np.abs(
                np.mean(sim_g_interp) - np.mean(experimental_data)
            ) / np.mean(experimental_data)) if np.mean(experimental_data) > 0 else 0,
        }
