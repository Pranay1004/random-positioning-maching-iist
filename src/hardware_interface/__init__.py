"""
Hardware Interface Package
===========================
Provides communication layer for all hardware devices including:
- Arduino microcontrollers
- Raspberry Pi edge computers
- NEMA stepper motors
- IMU sensors
- Rotary encoders

This package abstracts hardware-specific protocols and provides
a unified interface for the rest of the application.
"""

from .models import (
    DeviceType,
    ConnectionStatus,
    MotorState,
    VelocityUnit,
    SensorDataPacket,
    IMUDataPacket,
    EncoderDataPacket,
    MotorCommandPacket,
    MotorStatusPacket,
    DeviceConfig,
    ArduinoConfig,
    RaspberryPiConfig,
    MotorConfig,
    RPMStateSnapshot,
    ExperimentMetadata,
)

# higher-level managers
from .serial_manager import SerialManager

__all__ = [
    "DeviceType",
    "ConnectionStatus",
    "MotorState",
    "VelocityUnit",
    "SensorDataPacket",
    "IMUDataPacket",
    "EncoderDataPacket",
    "MotorCommandPacket",
    "MotorStatusPacket",
    "DeviceConfig",
    "ArduinoConfig",
    "RaspberryPiConfig",
    "MotorConfig",
    "RPMStateSnapshot",
    "ExperimentMetadata",
    "SerialManager",
]
