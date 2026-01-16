"""
Hardware Interface - Data Models
=================================
Pydantic models for hardware data packets and configuration.

These models ensure type safety and validation for all hardware communications.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
import numpy as np


class DeviceType(str, Enum):
    """Supported hardware device types."""
    ARDUINO = "arduino"
    RASPBERRY_PI = "raspberry_pi"
    MOTOR_CONTROLLER = "motor_controller"
    IMU_SENSOR = "imu"
    ENCODER = "encoder"
    LOAD_CELL = "load_cell"
    CUSTOM = "custom"


class ConnectionStatus(str, Enum):
    """Device connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    TIMEOUT = "timeout"


class MotorState(str, Enum):
    """Motor operational state."""
    IDLE = "idle"
    RUNNING = "running"
    ACCELERATING = "accelerating"
    DECELERATING = "decelerating"
    HOLDING = "holding"
    ERROR = "error"
    DISABLED = "disabled"


class VelocityUnit(str, Enum):
    """Supported velocity units."""
    RPM = "rpm"
    RAD_PER_SEC = "rad/s"
    DEG_PER_SEC = "deg/s"


# =============================================================================
# DATA PACKET MODELS
# =============================================================================

class SensorDataPacket(BaseModel):
    """
    Generic sensor data packet received from hardware.
    
    This is the fundamental data unit transmitted from sensors to the software.
    Each packet is timestamped and contains raw or processed sensor readings.
    """
    timestamp: datetime = Field(default_factory=datetime.now)
    device_id: str = Field(..., description="Unique device identifier")
    device_type: DeviceType
    sequence_number: int = Field(ge=0, description="Packet sequence for ordering")
    
    # Sensor readings
    data: Dict[str, float] = Field(default_factory=dict)
    
    # Packet metadata
    latency_ms: Optional[float] = Field(None, ge=0)
    checksum: Optional[str] = None
    is_valid: bool = True
    error_code: Optional[int] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class IMUDataPacket(SensorDataPacket):
    """
    IMU (Inertial Measurement Unit) specific data packet.
    
    Contains accelerometer, gyroscope, and optionally magnetometer data.
    """
    device_type: DeviceType = DeviceType.IMU_SENSOR
    
    # Accelerometer (m/s²)
    accel_x: float = Field(0.0, description="X-axis acceleration")
    accel_y: float = Field(0.0, description="Y-axis acceleration")
    accel_z: float = Field(0.0, description="Z-axis acceleration")
    
    # Gyroscope (rad/s)
    gyro_x: float = Field(0.0, description="X-axis angular velocity")
    gyro_y: float = Field(0.0, description="Y-axis angular velocity")
    gyro_z: float = Field(0.0, description="Z-axis angular velocity")
    
    # Magnetometer (µT) - optional
    mag_x: Optional[float] = None
    mag_y: Optional[float] = None
    mag_z: Optional[float] = None
    
    # Temperature (°C)
    temperature: Optional[float] = None
    
    @property
    def acceleration_vector(self) -> np.ndarray:
        """Get acceleration as numpy array."""
        return np.array([self.accel_x, self.accel_y, self.accel_z])
    
    @property
    def angular_velocity_vector(self) -> np.ndarray:
        """Get angular velocity as numpy array."""
        return np.array([self.gyro_x, self.gyro_y, self.gyro_z])
    
    @property
    def acceleration_magnitude(self) -> float:
        """Calculate total acceleration magnitude."""
        return np.linalg.norm(self.acceleration_vector)


class EncoderDataPacket(SensorDataPacket):
    """
    Rotary encoder data packet.
    
    Contains position and velocity feedback from frame encoders.
    """
    device_type: DeviceType = DeviceType.ENCODER
    
    # Position
    position_counts: int = Field(0, description="Raw encoder counts")
    position_rad: float = Field(0.0, description="Position in radians")
    position_deg: float = Field(0.0, description="Position in degrees")
    
    # Velocity
    velocity_rad_s: float = Field(0.0, description="Angular velocity (rad/s)")
    velocity_rpm: float = Field(0.0, description="Angular velocity (RPM)")
    
    # Index pulse
    index_count: int = Field(0, description="Number of index pulses detected")
    
    @validator('position_deg', always=True)
    def compute_position_deg(cls, v, values):
        """Auto-compute degrees from radians if not provided."""
        if v == 0.0 and 'position_rad' in values:
            return np.degrees(values['position_rad'])
        return v


class MotorCommandPacket(BaseModel):
    """
    Motor command packet sent to motor controllers.
    
    Contains setpoints and control parameters for NEMA motors.
    """
    timestamp: datetime = Field(default_factory=datetime.now)
    motor_id: str = Field(..., description="Target motor identifier")
    command_id: int = Field(..., description="Unique command identifier")
    
    # Setpoint
    target_velocity: float = Field(..., description="Target velocity")
    velocity_unit: VelocityUnit = VelocityUnit.RPM
    
    # Motion profile
    acceleration: Optional[float] = Field(None, description="Acceleration rate")
    deceleration: Optional[float] = Field(None, description="Deceleration rate")
    
    # Control flags
    enable: bool = True
    emergency_stop: bool = False
    
    def to_rad_per_sec(self) -> float:
        """Convert target velocity to rad/s."""
        if self.velocity_unit == VelocityUnit.RPM:
            return self.target_velocity * (2 * np.pi / 60)
        elif self.velocity_unit == VelocityUnit.DEG_PER_SEC:
            return np.radians(self.target_velocity)
        return self.target_velocity


class MotorStatusPacket(SensorDataPacket):
    """
    Motor status feedback packet.
    
    Contains current state and telemetry from motor controllers.
    """
    device_type: DeviceType = DeviceType.MOTOR_CONTROLLER
    
    # State
    state: MotorState = MotorState.IDLE
    
    # Current values
    current_velocity_rpm: float = 0.0
    current_position_rad: float = 0.0
    
    # Setpoint tracking
    target_velocity_rpm: float = 0.0
    velocity_error_rpm: float = 0.0
    
    # Electrical
    current_draw_a: Optional[float] = None
    voltage_v: Optional[float] = None
    power_w: Optional[float] = None
    
    # Thermal
    temperature_c: Optional[float] = None
    
    # Faults
    fault_code: int = 0
    fault_description: Optional[str] = None


# =============================================================================
# DEVICE CONFIGURATION MODELS
# =============================================================================

class DeviceConfig(BaseModel):
    """Base configuration for hardware devices."""
    device_id: str
    device_type: DeviceType
    name: str
    enabled: bool = True
    
    # Connection parameters
    connection_type: str = "serial"  # serial, i2c, spi, tcp, can
    
    # Polling
    poll_rate_hz: float = Field(100.0, gt=0)
    
    # Timeout
    timeout_s: float = Field(1.0, gt=0)


class ArduinoConfig(DeviceConfig):
    """Arduino-specific configuration."""
    device_type: DeviceType = DeviceType.ARDUINO
    connection_type: str = "serial"
    
    port: str = "auto"
    baudrate: int = Field(115200, gt=0)
    
    # Protocol settings
    packet_start_byte: int = 0xAA
    packet_end_byte: int = 0x55


class RaspberryPiConfig(DeviceConfig):
    """Raspberry Pi configuration."""
    device_type: DeviceType = DeviceType.RASPBERRY_PI
    connection_type: str = "tcp"
    
    host: str = "localhost"
    port: int = Field(5000, gt=0, lt=65536)
    
    # Optional SSH for remote management
    ssh_enabled: bool = False
    ssh_user: Optional[str] = None
    ssh_key_path: Optional[str] = None


class MotorConfig(BaseModel):
    """NEMA motor configuration."""
    motor_id: str
    name: str
    motor_type: str = "NEMA23"
    
    # Stepper configuration
    steps_per_revolution: int = 200
    microstepping: int = 16
    
    # Limits
    max_velocity_rpm: float = Field(60.0, gt=0)
    max_acceleration_rpm_s: float = Field(100.0, gt=0)
    
    # GPIO pins (for direct control)
    direction_pin: Optional[int] = None
    step_pin: Optional[int] = None
    enable_pin: Optional[int] = None
    
    # Feedback
    encoder_enabled: bool = True
    encoder_ppr: int = 2000
    
    @property
    def steps_per_radian(self) -> float:
        """Calculate steps per radian."""
        total_steps = self.steps_per_revolution * self.microstepping
        return total_steps / (2 * np.pi)


# =============================================================================
# AGGREGATE DATA MODELS
# =============================================================================

class RPMStateSnapshot(BaseModel):
    """
    Complete RPM system state at a point in time.
    
    This is the primary data structure for storing and transmitting
    the complete state of the Random Positioning Machine.
    """
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Frame positions and velocities
    inner_frame_position_rad: float = 0.0
    inner_frame_velocity_rad_s: float = 0.0
    outer_frame_position_rad: float = 0.0
    outer_frame_velocity_rad_s: float = 0.0
    
    # Setpoints
    inner_frame_setpoint_rpm: float = 0.0
    outer_frame_setpoint_rpm: float = 0.0
    
    # Motor states
    inner_motor_state: MotorState = MotorState.IDLE
    outer_motor_state: MotorState = MotorState.IDLE
    
    # IMU data (at sample position)
    acceleration_x: float = 0.0
    acceleration_y: float = 0.0
    acceleration_z: float = 0.0
    
    # Computed gravity metrics
    instantaneous_g: float = 0.0
    time_averaged_g: Optional[float] = None
    
    # System status
    is_running: bool = False
    operation_mode: str = "idle"
    
    # Health
    all_devices_connected: bool = False
    active_faults: List[str] = Field(default_factory=list)
    
    @property
    def inner_frame_velocity_rpm(self) -> float:
        """Get inner frame velocity in RPM."""
        return self.inner_frame_velocity_rad_s * 60 / (2 * np.pi)
    
    @property
    def outer_frame_velocity_rpm(self) -> float:
        """Get outer frame velocity in RPM."""
        return self.outer_frame_velocity_rad_s * 60 / (2 * np.pi)
    
    @property
    def acceleration_magnitude(self) -> float:
        """Get total acceleration magnitude in m/s²."""
        return np.sqrt(
            self.acceleration_x**2 + 
            self.acceleration_y**2 + 
            self.acceleration_z**2
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()


class ExperimentMetadata(BaseModel):
    """
    Metadata for an experimental run.
    
    Stores all parameters and conditions for reproducibility.
    """
    experiment_id: str
    name: str
    description: Optional[str] = None
    
    # Timing
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_s: Optional[float] = None
    
    # Operating parameters
    operation_mode: str
    inner_frame_speed_rpm: float
    outer_frame_speed_rpm: float
    
    # Sample information
    sample_description: Optional[str] = None
    sample_position_m: List[float] = [0.0, 0.0, 0.1]
    
    # Environmental conditions
    ambient_temperature_c: Optional[float] = None
    humidity_percent: Optional[float] = None
    
    # Results summary
    mean_g_achieved: Optional[float] = None
    std_g: Optional[float] = None
    
    # Data files
    data_file_paths: List[str] = Field(default_factory=list)
    
    # Tags for searchability
    tags: List[str] = Field(default_factory=list)
    
    # Operator
    operator_name: Optional[str] = None
    notes: Optional[str] = None
