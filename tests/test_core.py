"""
Test Suite for RPM Digital Twin
================================
Comprehensive tests for all subsystems.
"""

import pytest
import numpy as np
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

# Import modules to test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hardware_interface import (
    RPMStateSnapshot,
    IMUDataPacket,
    EncoderDataPacket,
    MotorCommandPacket,
    MotorState,
    ConnectionStatus,
)
from simulation import (
    RPMSimulator,
    RPMGeometry,
    RPMState,
    OperationMode,
    GravityVectorCalculator,
    RotationMatrices,
)
from data_pipeline import StateManager, DataPipeline


class TestRotationMatrices:
    """Tests for rotation matrix calculations."""
    
    def test_identity_at_zero_angles(self):
        """Rotation matrix should be identity when both angles are zero."""
        R = RotationMatrices.combined_rotation(0, 0)
        np.testing.assert_array_almost_equal(R, np.eye(3), decimal=10)
    
    def test_inner_rotation_90_degrees(self):
        """Test 90-degree rotation of inner frame (around X)."""
        R = RotationMatrices.inner_frame_rotation(np.pi / 2)
        
        # Expected: rotation around X by 90 degrees
        expected = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        np.testing.assert_array_almost_equal(R, expected, decimal=10)
    
    def test_outer_rotation_90_degrees(self):
        """Test 90-degree rotation of outer frame (around Y)."""
        R = RotationMatrices.outer_frame_rotation(np.pi / 2)
        
        # Expected: rotation around Y by 90 degrees
        expected = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ])
        np.testing.assert_array_almost_equal(R, expected, decimal=10)
    
    def test_combined_rotation_commutative_check(self):
        """Combined rotation should match Ry * Rx order."""
        theta_i = np.pi / 4
        theta_o = np.pi / 6
        
        R_combined = RotationMatrices.combined_rotation(theta_i, theta_o)
        
        Rx = RotationMatrices.inner_frame_rotation(theta_i)
        Ry = RotationMatrices.outer_frame_rotation(theta_o)
        R_manual = Ry @ Rx
        
        np.testing.assert_array_almost_equal(R_combined, R_manual, decimal=10)
    
    def test_rotation_orthogonality(self):
        """Rotation matrices should be orthogonal (R^T * R = I)."""
        for _ in range(10):
            theta_i = np.random.uniform(-np.pi, np.pi)
            theta_o = np.random.uniform(-np.pi, np.pi)
            
            R = RotationMatrices.combined_rotation(theta_i, theta_o)
            
            np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=10)
            np.testing.assert_almost_equal(np.linalg.det(R), 1.0, decimal=10)


class TestGravityVectorCalculator:
    """Tests for gravity vector calculations."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.geometry = RPMGeometry(
            inner_frame_radius=0.15,
            outer_frame_radius=0.25,
            sample_mount_offset=np.array([0, 0, 0.1])
        )
        self.calculator = GravityVectorCalculator(self.geometry)
    
    def test_gravity_at_zero_rotation(self):
        """Gravity should point in -Y direction when no rotation."""
        g = self.calculator.compute_gravity_in_sample_frame(0, 0)
        
        # At zero rotation, gravity in sample frame = lab frame = [0, -g, 0]
        np.testing.assert_almost_equal(g[0], 0.0, decimal=5)
        np.testing.assert_almost_equal(g[1], -9.80665, decimal=3)
        np.testing.assert_almost_equal(g[2], 0.0, decimal=5)
    
    def test_gravity_magnitude_constant(self):
        """Gravity magnitude should be approximately constant (ignoring centrifugal)."""
        g0 = 9.80665
        
        for _ in range(20):
            theta_i = np.random.uniform(0, 2 * np.pi)
            theta_o = np.random.uniform(0, 2 * np.pi)
            
            g = self.calculator.compute_gravity_in_sample_frame(theta_i, theta_o)
            mag = np.linalg.norm(g)
            
            # Should be close to 1g when not rotating
            np.testing.assert_almost_equal(mag, g0, decimal=3)
    
    def test_time_averaged_gravity_clinostat(self):
        """Time-averaged gravity should approach zero in clinostat mode."""
        duration = 60.0  # 60 seconds
        dt = 0.01  # 10ms steps
        omega = 2 * np.pi / 60  # 1 RPM in rad/s
        
        g_sum = np.zeros(3)
        n_steps = int(duration / dt)
        
        for i in range(n_steps):
            t = i * dt
            theta_i = omega * t
            theta_o = omega * t
            g = self.calculator.compute_gravity_in_sample_frame(theta_i, theta_o)
            g_sum += g
        
        g_avg = g_sum / n_steps
        g_avg_mag = np.linalg.norm(g_avg) / 9.80665  # in g units
        
        # Time-averaged g should be small for clinostat mode
        assert g_avg_mag < 0.1, f"Time-averaged g too high: {g_avg_mag:.4f} g"


class TestRPMSimulator:
    """Tests for the RPM simulator."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.geometry = RPMGeometry()
        self.simulator = RPMSimulator(geometry=self.geometry)
    
    def test_initial_state(self):
        """Simulator should start at rest."""
        state = self.simulator.state
        
        assert state.theta_inner == 0.0
        assert state.theta_outer == 0.0
        assert state.omega_inner == 0.0
        assert state.omega_outer == 0.0
    
    def test_setpoint_update(self):
        """Setpoints should be stored correctly."""
        omega_i = 2.0
        omega_o = 3.0
        
        self.simulator.set_velocities(omega_i, omega_o, unit="rad/s")
        
        assert self.simulator.state.omega_inner_setpoint == omega_i
        assert self.simulator.state.omega_outer_setpoint == omega_o
    
    def test_simulation_step(self):
        """Simulation should advance state over time."""
        self.simulator.set_velocities(1.0, 1.0, unit="rad/s")
        
        # Run several steps
        for _ in range(100):
            state = self.simulator.step(0.01)
        
        # Velocities should approach setpoints
        assert abs(state.omega_inner) > 0
        assert abs(state.omega_outer) > 0
    
    def test_emergency_stop(self):
        """Emergency stop should zero setpoints."""
        self.simulator.set_velocities(5.0, 5.0, unit="rad/s")
        self.simulator.emergency_stop()
        
        assert self.simulator.state.omega_inner_setpoint == 0.0
        assert self.simulator.state.omega_outer_setpoint == 0.0
    
    def test_angle_wrapping(self):
        """Angles should stay within [-pi, pi]."""
        self.simulator.set_velocities(10.0, 10.0, unit="rad/s")  # High speed
        
        # Run long simulation
        for _ in range(10000):
            state = self.simulator.step(0.01)
        
        assert -np.pi <= state.theta_inner <= np.pi
        assert -np.pi <= state.theta_outer <= np.pi
    
    def test_clinostat_3d_mode(self):
        """Test clinostat 3D operation mode."""
        self.simulator = RPMSimulator(
            geometry=self.geometry,
            mode=OperationMode.CLINOSTAT_3D
        )
        
        self.simulator.set_velocities(2 * np.pi / 60, 2 * np.pi / 60, unit="rad/s")  # 1 RPM each
        
        g_values = []
        for _ in range(1000):
            self.simulator.step(0.01)
            g = self.simulator.compute_gravity_at_sample()
            g_values.append(np.linalg.norm(g) / 9.80665)
        
        mean_g = np.mean(g_values)
        # Clinostat should achieve low mean g
        assert mean_g < 1.5, f"Mean g too high: {mean_g}"


class TestStateManager:
    """Tests for the state manager."""
    
    def test_initial_state(self):
        """State manager should have valid initial state."""
        manager = StateManager()
        state = manager.current_state
        
        assert isinstance(state, RPMStateSnapshot)
        assert state.inner_frame_position_rad == 0.0
    
    def test_state_update(self):
        """State updates should be reflected."""
        manager = StateManager()
        
        manager.update_state(
            inner_frame_position_rad=1.5,
            inner_frame_velocity_rad_s=0.5
        )
        
        state = manager.current_state
        assert state.inner_frame_position_rad == 1.5
        assert state.inner_frame_velocity_rad_s == 0.5
    
    def test_history_tracking(self):
        """State history should be maintained."""
        manager = StateManager()
        
        for i in range(10):
            manager.update_state(inner_frame_position_rad=float(i))
        
        history = manager.get_history()
        assert len(history) == 10
    
    def test_subscriber_notification(self):
        """Subscribers should be notified of updates."""
        manager = StateManager()
        callback_data = []
        
        def callback(state):
            callback_data.append(state)
        
        manager.subscribe(callback)
        manager.update_state(inner_frame_position_rad=1.0)
        
        assert len(callback_data) == 1
        assert callback_data[0].inner_frame_position_rad == 1.0


class TestDataModels:
    """Tests for Pydantic data models."""
    
    def test_imu_packet_creation(self):
        """IMU data packet should initialize correctly."""
        packet = IMUDataPacket(
            device_id="imu_001",
            accel_x=0.1,
            accel_y=-9.8,
            accel_z=0.05,
            gyro_x=0.01,
            gyro_y=0.02,
            gyro_z=0.03
        )
        
        assert packet.device_id == "imu_001"
        assert packet.accel_y == -9.8
    
    def test_imu_packet_computed_fields(self):
        """IMU packet should compute magnitude correctly."""
        packet = IMUDataPacket(
            device_id="test",
            accel_x=3.0,
            accel_y=4.0,
            accel_z=0.0,
            gyro_x=0,
            gyro_y=0,
            gyro_z=0
        )
        
        assert abs(packet.acceleration_magnitude - 5.0) < 0.001
    
    def test_encoder_packet_conversion(self):
        """Encoder packet should convert units correctly."""
        packet = EncoderDataPacket(
            device_id="enc_inner",
            raw_count=1000,
            position_rad=np.pi,
            velocity_rad_s=2 * np.pi  # 1 revolution per second = 60 RPM
        )
        
        assert abs(packet.velocity_rpm - 60.0) < 0.001
    
    def test_motor_command_creation(self):
        """Motor command packet should validate correctly."""
        cmd = MotorCommandPacket(
            motor_id=0,
            target_rpm=5.0,
            acceleration_rpm_s=100.0
        )
        
        assert cmd.motor_id == 0
        assert cmd.target_velocity_rad_s == pytest.approx(5.0 * 2 * np.pi / 60)
    
    def test_state_snapshot_serialization(self):
        """State snapshot should serialize to dict."""
        state = RPMStateSnapshot(
            inner_frame_position_rad=1.0,
            inner_frame_velocity_rad_s=0.5,
            instantaneous_g=0.02
        )
        
        data = state.to_dict()
        assert "inner_frame_position_rad" in data
        assert data["inner_frame_position_rad"] == 1.0


class TestIntegration:
    """Integration tests for multiple components."""
    
    @pytest.mark.asyncio
    async def test_simulation_to_state_update(self):
        """Test data flow from simulation to state manager."""
        geometry = RPMGeometry()
        simulator = RPMSimulator(geometry=geometry)
        state_manager = StateManager()
        
        simulator.set_velocities(1.0, 1.0)
        
        for _ in range(100):
            sim_state = simulator.step(0.01)
            g = simulator.compute_gravity_at_sample()
            g_mag = np.linalg.norm(g) / 9.80665
            
            state_manager.update_from_simulation(sim_state, g_mag, g_mag)
        
        current = state_manager.current_state
        assert current.inner_frame_velocity_rad_s > 0
    
    def test_microgravity_quality(self):
        """Test that system can achieve good microgravity."""
        geometry = RPMGeometry(
            inner_frame_radius=0.15,
            outer_frame_radius=0.25,
            sample_mount_offset=np.array([0, 0, 0.1])
        )
        simulator = RPMSimulator(
            geometry=geometry,
            mode=OperationMode.CLINOSTAT_3D
        )
        
        # 2 RPM on each axis
        omega = 2 * np.pi / 30
        simulator.set_velocities(omega, omega, unit="rad/s")
        
        # Run for 60 seconds
        duration = 60.0
        dt = 0.01
        g_values = []
        
        for _ in range(int(duration / dt)):
            simulator.step(dt)
            g = simulator.compute_gravity_at_sample()
            g_values.append(np.linalg.norm(g) / 9.80665)
        
        mean_g = np.mean(g_values)
        std_g = np.std(g_values)
        
        print(f"\nMicrogravity Quality Test:")
        print(f"  Mean g: {mean_g:.5f} g")
        print(f"  Std g: {std_g:.5f} g")
        
        # Should achieve reasonable microgravity
        assert mean_g < 0.5, f"Mean g too high: {mean_g}"


# Fixtures

@pytest.fixture
def sample_geometry():
    """Provide standard RPM geometry."""
    return RPMGeometry(
        inner_frame_radius=0.15,
        outer_frame_radius=0.25,
        sample_mount_offset=np.array([0, 0, 0.1])
    )


@pytest.fixture
def sample_simulator(sample_geometry):
    """Provide initialized simulator."""
    return RPMSimulator(geometry=sample_geometry)


@pytest.fixture
def sample_state_manager():
    """Provide state manager instance."""
    return StateManager()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
