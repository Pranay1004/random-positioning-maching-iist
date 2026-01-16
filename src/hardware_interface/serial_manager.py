"""
Serial Communication Manager
==============================
Handles serial port communication with Arduino and other serial devices.

Features:
- Auto-detection of serial ports
- Async read/write operations
- Packet parsing with checksums
- Automatic reconnection
- Thread-safe operations
"""

from __future__ import annotations

import asyncio
import struct
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Optional, Dict, List, Any, Tuple
import threading

import serial
import serial.tools.list_ports
from loguru import logger

from .models import (
    ConnectionStatus,
    SensorDataPacket,
    IMUDataPacket,
    EncoderDataPacket,
    MotorStatusPacket,
    ArduinoConfig,
    DeviceType,
)


class PacketType(Enum):
    """Serial packet types for protocol."""
    HEARTBEAT = 0x01
    IMU_DATA = 0x02
    ENCODER_DATA = 0x03
    MOTOR_STATUS = 0x04
    MOTOR_COMMAND = 0x05
    CONFIG = 0x06
    ACK = 0x07
    ERROR = 0xFF


@dataclass
class SerialPacket:
    """
    Binary serial packet structure.
    
    Packet Format:
    [START][LENGTH][TYPE][SEQUENCE][PAYLOAD...][CHECKSUM][END]
    
    - START: 1 byte (0xAA)
    - LENGTH: 2 bytes (payload length)
    - TYPE: 1 byte (PacketType)
    - SEQUENCE: 2 bytes (packet counter)
    - PAYLOAD: N bytes
    - CHECKSUM: 2 bytes (CRC16)
    - END: 1 byte (0x55)
    """
    packet_type: PacketType
    sequence: int
    payload: bytes
    timestamp: datetime = field(default_factory=datetime.now)
    checksum: int = 0
    is_valid: bool = True
    
    START_BYTE = 0xAA
    END_BYTE = 0x55
    HEADER_SIZE = 6  # START + LENGTH(2) + TYPE + SEQUENCE(2)
    FOOTER_SIZE = 3  # CHECKSUM(2) + END
    
    @classmethod
    def create(cls, packet_type: PacketType, payload: bytes, sequence: int = 0) -> SerialPacket:
        """Create a new packet with computed checksum."""
        packet = cls(
            packet_type=packet_type,
            sequence=sequence,
            payload=payload
        )
        packet.checksum = packet._compute_checksum()
        return packet
    
    def _compute_checksum(self) -> int:
        """Compute CRC16-CCITT checksum."""
        data = struct.pack('<BHB', self.START_BYTE, len(self.payload), self.packet_type.value)
        data += struct.pack('<H', self.sequence)
        data += self.payload
        
        crc = 0xFFFF
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ 0x1021
                else:
                    crc <<= 1
                crc &= 0xFFFF
        return crc
    
    def to_bytes(self) -> bytes:
        """Serialize packet to bytes for transmission."""
        header = struct.pack(
            '<BHB H',
            self.START_BYTE,
            len(self.payload),
            self.packet_type.value,
            self.sequence
        )
        footer = struct.pack('<H B', self.checksum, self.END_BYTE)
        return header + self.payload + footer
    
    @classmethod
    def from_bytes(cls, data: bytes) -> Optional[SerialPacket]:
        """Deserialize packet from bytes."""
        if len(data) < cls.HEADER_SIZE + cls.FOOTER_SIZE:
            return None
            
        try:
            # Parse header
            start, length, ptype, sequence = struct.unpack('<BHB H', data[:cls.HEADER_SIZE])
            
            if start != cls.START_BYTE:
                return None
                
            # Extract payload
            payload_end = cls.HEADER_SIZE + length
            payload = data[cls.HEADER_SIZE:payload_end]
            
            # Parse footer
            checksum, end = struct.unpack('<H B', data[payload_end:payload_end + cls.FOOTER_SIZE])
            
            if end != cls.END_BYTE:
                return None
            
            packet = cls(
                packet_type=PacketType(ptype),
                sequence=sequence,
                payload=payload,
                checksum=checksum
            )
            
            # Verify checksum
            computed = packet._compute_checksum()
            packet.is_valid = (checksum == computed)
            
            return packet
            
        except (struct.error, ValueError) as e:
            logger.warning(f"Failed to parse packet: {e}")
            return None


class SerialManager:
    """
    Manages serial communication with hardware devices.
    
    Features:
    - Automatic port detection
    - Asynchronous read/write
    - Packet buffering and parsing
    - Connection health monitoring
    - Automatic reconnection
    
    Usage:
        manager = SerialManager(config)
        await manager.connect()
        manager.register_callback(PacketType.IMU_DATA, handle_imu)
        await manager.start()
    """
    
    def __init__(self, config: ArduinoConfig):
        """
        Initialize serial manager.
        
        Args:
            config: Arduino configuration parameters
        """
        self.config = config
        self._serial: Optional[serial.Serial] = None
        self._status = ConnectionStatus.DISCONNECTED
        self._running = False
        
        # Packet handling
        self._callbacks: Dict[PacketType, List[Callable]] = {pt: [] for pt in PacketType}
        self._sequence_counter = 0
        self._last_sequence_received = 0
        
        # Statistics
        self._packets_sent = 0
        self._packets_received = 0
        self._packets_dropped = 0
        self._last_packet_time: Optional[datetime] = None
        
        # Buffers
        self._read_buffer = bytearray()
        self._write_queue: deque = deque(maxlen=1000)
        
        # Threading
        self._lock = threading.Lock()
        self._read_task: Optional[asyncio.Task] = None
        self._write_task: Optional[asyncio.Task] = None
        
        logger.info(f"SerialManager initialized for device: {config.device_id}")
    
    @property
    def status(self) -> ConnectionStatus:
        """Get current connection status."""
        return self._status
    
    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._status == ConnectionStatus.CONNECTED
    
    @property
    def statistics(self) -> Dict[str, Any]:
        """Get communication statistics."""
        return {
            "status": self._status.value,
            "packets_sent": self._packets_sent,
            "packets_received": self._packets_received,
            "packets_dropped": self._packets_dropped,
            "last_packet_time": self._last_packet_time.isoformat() if self._last_packet_time else None,
            "latency_ms": self._calculate_latency(),
        }
    
    def _calculate_latency(self) -> Optional[float]:
        """Calculate communication latency."""
        if self._last_packet_time:
            delta = datetime.now() - self._last_packet_time
            return delta.total_seconds() * 1000
        return None
    
    @staticmethod
    def list_available_ports() -> List[Dict[str, str]]:
        """
        List all available serial ports.
        
        Returns:
            List of port information dictionaries
        """
        ports = []
        for port in serial.tools.list_ports.comports():
            ports.append({
                "device": port.device,
                "name": port.name,
                "description": port.description,
                "hwid": port.hwid,
                "manufacturer": port.manufacturer or "Unknown",
            })
        return ports
    
    def _find_arduino_port(self) -> Optional[str]:
        """
        Auto-detect Arduino serial port.
        
        Returns:
            Port device path or None if not found
        """
        for port in serial.tools.list_ports.comports():
            # Common Arduino identifiers
            if any(id in port.description.lower() for id in ['arduino', 'ch340', 'cp210', 'ftdi']):
                logger.info(f"Auto-detected Arduino on {port.device}")
                return port.device
            if any(id in (port.manufacturer or '').lower() for id in ['arduino', 'wch']):
                logger.info(f"Auto-detected Arduino on {port.device}")
                return port.device
        return None
    
    async def connect(self) -> bool:
        """
        Establish serial connection.
        
        Returns:
            True if connection successful
        """
        self._status = ConnectionStatus.CONNECTING
        logger.info(f"Connecting to serial device...")
        
        try:
            # Determine port
            port = self.config.port
            if port == "auto":
                port = self._find_arduino_port()
                if not port:
                    logger.error("Could not auto-detect Arduino port")
                    self._status = ConnectionStatus.ERROR
                    return False
            
            # Open serial connection
            self._serial = serial.Serial(
                port=port,
                baudrate=self.config.baudrate,
                timeout=self.config.timeout_s,
                write_timeout=self.config.timeout_s,
            )
            
            # Wait for Arduino reset
            await asyncio.sleep(2.0)
            
            # Clear buffers
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()
            
            self._status = ConnectionStatus.CONNECTED
            logger.success(f"Connected to {port} at {self.config.baudrate} baud")
            return True
            
        except serial.SerialException as e:
            logger.error(f"Serial connection failed: {e}")
            self._status = ConnectionStatus.ERROR
            return False
    
    async def disconnect(self) -> None:
        """Close serial connection."""
        self._running = False
        
        if self._read_task:
            self._read_task.cancel()
        if self._write_task:
            self._write_task.cancel()
            
        if self._serial and self._serial.is_open:
            self._serial.close()
            
        self._status = ConnectionStatus.DISCONNECTED
        logger.info("Serial connection closed")
    
    def register_callback(self, packet_type: PacketType, callback: Callable) -> None:
        """
        Register callback for specific packet type.
        
        Args:
            packet_type: Type of packet to handle
            callback: Function to call when packet received
        """
        self._callbacks[packet_type].append(callback)
        logger.debug(f"Registered callback for {packet_type.name}")
    
    def unregister_callback(self, packet_type: PacketType, callback: Callable) -> None:
        """Remove a registered callback."""
        if callback in self._callbacks[packet_type]:
            self._callbacks[packet_type].remove(callback)
    
    async def send_packet(self, packet: SerialPacket) -> bool:
        """
        Send packet to device.
        
        Args:
            packet: Packet to send
            
        Returns:
            True if sent successfully
        """
        if not self.is_connected:
            logger.warning("Cannot send packet: not connected")
            return False
            
        try:
            data = packet.to_bytes()
            with self._lock:
                self._serial.write(data)
                self._serial.flush()
            
            self._packets_sent += 1
            logger.trace(f"Sent packet: {packet.packet_type.name}, seq={packet.sequence}")
            return True
            
        except serial.SerialException as e:
            logger.error(f"Failed to send packet: {e}")
            return False
    
    async def send_motor_command(
        self,
        motor_id: str,
        velocity_rpm: float,
        acceleration: Optional[float] = None
    ) -> bool:
        """
        Send motor command.
        
        Args:
            motor_id: Target motor ID ("inner" or "outer")
            velocity_rpm: Target velocity in RPM
            acceleration: Optional acceleration rate
            
        Returns:
            True if command sent
        """
        # Pack motor command payload
        motor_byte = 0x01 if motor_id == "inner" else 0x02
        payload = struct.pack('<B f', motor_byte, velocity_rpm)
        
        if acceleration is not None:
            payload += struct.pack('<f', acceleration)
        
        self._sequence_counter = (self._sequence_counter + 1) % 65536
        packet = SerialPacket.create(
            PacketType.MOTOR_COMMAND,
            payload,
            self._sequence_counter
        )
        
        return await self.send_packet(packet)
    
    async def start(self) -> None:
        """Start async read/write loops."""
        if not self.is_connected:
            raise RuntimeError("Must connect before starting")
            
        self._running = True
        self._read_task = asyncio.create_task(self._read_loop())
        self._write_task = asyncio.create_task(self._write_loop())
        
        logger.info("Serial communication started")
    
    async def _read_loop(self) -> None:
        """Continuous read loop."""
        while self._running:
            try:
                if self._serial.in_waiting:
                    with self._lock:
                        data = self._serial.read(self._serial.in_waiting)
                    self._read_buffer.extend(data)
                    await self._process_buffer()
                else:
                    await asyncio.sleep(0.001)  # 1ms polling
                    
            except serial.SerialException as e:
                logger.error(f"Read error: {e}")
                self._status = ConnectionStatus.ERROR
                await asyncio.sleep(1.0)
    
    async def _write_loop(self) -> None:
        """Process write queue."""
        while self._running:
            if self._write_queue:
                packet = self._write_queue.popleft()
                await self.send_packet(packet)
            await asyncio.sleep(0.001)
    
    async def _process_buffer(self) -> None:
        """Process read buffer and extract packets."""
        while len(self._read_buffer) >= SerialPacket.HEADER_SIZE + SerialPacket.FOOTER_SIZE:
            # Find start byte
            start_idx = self._read_buffer.find(SerialPacket.START_BYTE)
            
            if start_idx == -1:
                self._read_buffer.clear()
                return
                
            if start_idx > 0:
                # Discard bytes before start
                del self._read_buffer[:start_idx]
                self._packets_dropped += start_idx
                
            # Check if we have enough bytes for header
            if len(self._read_buffer) < SerialPacket.HEADER_SIZE:
                return
                
            # Get payload length
            length = struct.unpack('<H', self._read_buffer[1:3])[0]
            total_size = SerialPacket.HEADER_SIZE + length + SerialPacket.FOOTER_SIZE
            
            # Wait for complete packet
            if len(self._read_buffer) < total_size:
                return
                
            # Extract packet bytes
            packet_data = bytes(self._read_buffer[:total_size])
            del self._read_buffer[:total_size]
            
            # Parse packet
            packet = SerialPacket.from_bytes(packet_data)
            
            if packet and packet.is_valid:
                self._packets_received += 1
                self._last_packet_time = datetime.now()
                await self._dispatch_packet(packet)
            else:
                self._packets_dropped += 1
                logger.warning("Received invalid packet")
    
    async def _dispatch_packet(self, packet: SerialPacket) -> None:
        """Dispatch packet to registered callbacks."""
        # Parse based on packet type
        data_packet = self._parse_payload(packet)
        
        # Call registered callbacks
        for callback in self._callbacks[packet.packet_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data_packet)
                else:
                    callback(data_packet)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def _parse_payload(self, packet: SerialPacket) -> SensorDataPacket:
        """Parse packet payload into appropriate data model."""
        if packet.packet_type == PacketType.IMU_DATA:
            return self._parse_imu_payload(packet)
        elif packet.packet_type == PacketType.ENCODER_DATA:
            return self._parse_encoder_payload(packet)
        elif packet.packet_type == PacketType.MOTOR_STATUS:
            return self._parse_motor_status_payload(packet)
        else:
            return SensorDataPacket(
                device_id=self.config.device_id,
                device_type=DeviceType.ARDUINO,
                sequence_number=packet.sequence,
                data={}
            )
    
    def _parse_imu_payload(self, packet: SerialPacket) -> IMUDataPacket:
        """Parse IMU data payload."""
        # Expected format: 6 floats (ax, ay, az, gx, gy, gz) + optional temp
        try:
            values = struct.unpack('<6f', packet.payload[:24])
            
            return IMUDataPacket(
                device_id=self.config.device_id,
                sequence_number=packet.sequence,
                accel_x=values[0],
                accel_y=values[1],
                accel_z=values[2],
                gyro_x=values[3],
                gyro_y=values[4],
                gyro_z=values[5],
                temperature=struct.unpack('<f', packet.payload[24:28])[0] if len(packet.payload) >= 28 else None
            )
        except struct.error as e:
            logger.error(f"Failed to parse IMU payload: {e}")
            return IMUDataPacket(device_id=self.config.device_id, sequence_number=packet.sequence)
    
    def _parse_encoder_payload(self, packet: SerialPacket) -> EncoderDataPacket:
        """Parse encoder data payload."""
        try:
            # Expected: encoder_id (1), counts (4), position_rad (4), velocity (4)
            encoder_id, counts, position, velocity = struct.unpack('<B i f f', packet.payload[:13])
            
            return EncoderDataPacket(
                device_id=f"{self.config.device_id}_encoder_{encoder_id}",
                sequence_number=packet.sequence,
                position_counts=counts,
                position_rad=position,
                velocity_rad_s=velocity,
                velocity_rpm=velocity * 60 / (2 * 3.14159)
            )
        except struct.error as e:
            logger.error(f"Failed to parse encoder payload: {e}")
            return EncoderDataPacket(device_id=self.config.device_id, sequence_number=packet.sequence)
    
    def _parse_motor_status_payload(self, packet: SerialPacket) -> MotorStatusPacket:
        """Parse motor status payload."""
        try:
            # Expected: motor_id(1), state(1), velocity(4), position(4), current(4), temp(4)
            motor_id, state, velocity, position, current, temp = struct.unpack('<B B f f f f', packet.payload[:18])
            
            from .models import MotorState
            
            return MotorStatusPacket(
                device_id=f"{self.config.device_id}_motor_{motor_id}",
                sequence_number=packet.sequence,
                state=MotorState(state) if state < len(MotorState) else MotorState.ERROR,
                current_velocity_rpm=velocity,
                current_position_rad=position,
                current_draw_a=current,
                temperature_c=temp
            )
        except struct.error as e:
            logger.error(f"Failed to parse motor status payload: {e}")
            return MotorStatusPacket(device_id=self.config.device_id, sequence_number=packet.sequence)
