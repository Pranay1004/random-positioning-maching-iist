"""
RPM Digital Twin - Hardware Test Base Framework
================================================
Unified test infrastructure for all hardware components.

Features:
- Dual display: terminal (colored) + GUI (web) modes
- Packet-level hex dump logging
- Connection status tracking
- CRC16-CCITT validation
- Auto-detection of serial ports
- Structured test results with pass/fail/warn states

Usage:
    python -m tests.hardware.test_base --list
    python -m tests.hardware.test_base --component pico_w
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import struct
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("hw_test")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================

class TestStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"
    RUNNING = "RUNNING"
    PENDING = "PENDING"


class ComponentType(str, Enum):
    RPI_PICO_W = "rpi_pico_w"
    RPI_5 = "rpi_5"
    NEMA_23_INNER = "nema_23_inner"
    NEMA_23_OUTER = "nema_23_outer"
    NEMA_24 = "nema_24"
    SLIP_RING_INNER = "slip_ring_inner"
    SLIP_RING_OUTER = "slip_ring_outer"


class ConnectionType(str, Enum):
    SERIAL = "serial"
    USB = "usb"
    GPIO = "gpio"
    NETWORK = "network"
    I2C = "i2c"
    SPI = "spi"


@dataclass
class TestResult:
    """Individual test result."""
    name: str
    component: ComponentType
    status: TestStatus = TestStatus.PENDING
    message: str = ""
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    packets_sent: int = 0
    packets_received: int = 0
    packets_failed: int = 0
    hex_dump: List[str] = field(default_factory=list)


@dataclass
class ComponentStatus:
    """Hardware component connection status."""
    component: ComponentType
    name: str
    connection_type: ConnectionType
    is_connected: bool = False
    port: str = ""
    status_msg: str = "Not tested"
    last_seen: Optional[datetime] = None
    firmware_version: str = "unknown"
    test_results: List[TestResult] = field(default_factory=list)
    latency_ms: float = -1.0
    packets_per_sec: float = 0.0


# =============================================================================
# TERMINAL DISPLAY (Colored Output)
# =============================================================================

class TerminalColors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_CYAN = "\033[46m"
    
    # Status colors
    @staticmethod
    def status_color(status: TestStatus) -> str:
        mapping = {
            TestStatus.PASS: TerminalColors.GREEN,
            TestStatus.FAIL: TerminalColors.RED,
            TestStatus.WARN: TerminalColors.YELLOW,
            TestStatus.SKIP: TerminalColors.GRAY,
            TestStatus.RUNNING: TerminalColors.CYAN,
            TestStatus.PENDING: TerminalColors.DIM,
        }
        return mapping.get(status, TerminalColors.WHITE)


C = TerminalColors


class TerminalDisplay:
    """Rich terminal output for hardware tests."""
    
    HEADER_WIDTH = 72
    
    @staticmethod
    def banner(title: str) -> None:
        print(f"\n{C.CYAN}{C.BOLD}{'=' * TerminalDisplay.HEADER_WIDTH}")
        print(f"  RPM DIGITAL TWIN - {title}")
        print(f"{'=' * TerminalDisplay.HEADER_WIDTH}{C.RESET}")
        print(f"{C.DIM}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
              f"{platform.node()}  |  {platform.system()} {platform.release()}{C.RESET}\n")
    
    @staticmethod
    def section(title: str) -> None:
        print(f"\n{C.BOLD}{C.BLUE}--- {title} {'─' * (TerminalDisplay.HEADER_WIDTH - len(title) - 5)}{C.RESET}")
    
    @staticmethod
    def test_result(result: TestResult) -> None:
        color = C.status_color(result.status)
        status_str = f"[{result.status.value}]"
        duration = f"{result.duration_ms:.1f}ms" if result.duration_ms > 0 else ""
        
        print(f"  {color}{C.BOLD}{status_str:8s}{C.RESET} {result.name:40s} "
              f"{C.DIM}{duration:>10s}{C.RESET}")
        
        if result.message:
            print(f"           {C.DIM}{result.message}{C.RESET}")
        
        if result.packets_sent > 0 or result.packets_received > 0:
            print(f"           {C.GRAY}TX: {result.packets_sent}  |  "
                  f"RX: {result.packets_received}  |  "
                  f"ERR: {result.packets_failed}{C.RESET}")
    
    @staticmethod
    def hex_dump(data: bytes, direction: str = "TX") -> str:
        """Format bytes as hex dump with ASCII representation."""
        if not data:
            return ""
        
        lines = []
        hex_chars = data.hex().upper()
        for i in range(0, len(data), 16):
            chunk = data[i:i+16]
            hex_part = " ".join(f"{b:02X}" for b in chunk)
            ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            addr = f"{i:04X}"
            lines.append(f"    {C.GRAY}{addr}{C.RESET}  {C.CYAN if direction == 'TX' else C.GREEN}"
                         f"{hex_part:<48s}{C.RESET}  |{ascii_part}|")
        
        header_color = C.CYAN if direction == "TX" else C.GREEN
        header = f"    {header_color}{C.BOLD}[{direction}]{C.RESET} {len(data)} bytes"
        return header + "\n" + "\n".join(lines)
    
    @staticmethod
    def connection_status(component: ComponentStatus) -> None:
        if component.is_connected:
            icon = f"{C.GREEN}{C.BOLD}●{C.RESET}"
            status = f"{C.GREEN}CONNECTED{C.RESET}"
        else:
            icon = f"{C.RED}{C.BOLD}○{C.RESET}"
            status = f"{C.RED}DISCONNECTED{C.RESET}"
        
        print(f"  {icon} {component.name:30s} {status:>20s}  "
              f"{C.DIM}{component.port}{C.RESET}")
        if component.latency_ms > 0:
            print(f"    {C.DIM}Latency: {component.latency_ms:.1f}ms  |  "
                  f"Throughput: {component.packets_per_sec:.0f} pkt/s{C.RESET}")
    
    @staticmethod
    def summary(results: List[TestResult]) -> None:
        passed = sum(1 for r in results if r.status == TestStatus.PASS)
        failed = sum(1 for r in results if r.status == TestStatus.FAIL)
        warned = sum(1 for r in results if r.status == TestStatus.WARN)
        skipped = sum(1 for r in results if r.status == TestStatus.SKIP)
        total = len(results)
        
        print(f"\n{C.BOLD}{'=' * TerminalDisplay.HEADER_WIDTH}")
        print(f"  TEST SUMMARY")
        print(f"{'=' * TerminalDisplay.HEADER_WIDTH}{C.RESET}")
        print(f"  {C.GREEN}PASS: {passed}{C.RESET}  |  "
              f"{C.RED}FAIL: {failed}{C.RESET}  |  "
              f"{C.YELLOW}WARN: {warned}{C.RESET}  |  "
              f"{C.GRAY}SKIP: {skipped}{C.RESET}  |  "
              f"TOTAL: {total}")
        
        if failed == 0 and warned == 0:
            print(f"\n  {C.GREEN}{C.BOLD}ALL TESTS PASSED{C.RESET}\n")
        elif failed > 0:
            print(f"\n  {C.RED}{C.BOLD}TESTS FAILED - CHECK HARDWARE{C.RESET}\n")
        else:
            print(f"\n  {C.YELLOW}{C.BOLD}TESTS PASSED WITH WARNINGS{C.RESET}\n")
    
    @staticmethod
    def packet_trace(direction: str, packet_type: str, seq: int,
                     payload_size: int, crc: int, valid: bool) -> None:
        """Display a single packet trace line."""
        color = C.CYAN if direction == "TX" else C.GREEN
        valid_str = f"{C.GREEN}OK{C.RESET}" if valid else f"{C.RED}BAD CRC{C.RESET}"
        
        print(f"    {color}[{direction}]{C.RESET} "
              f"Type={packet_type:15s} Seq={seq:05d} "
              f"Len={payload_size:4d}B CRC=0x{crc:04X} {valid_str}")


# =============================================================================
# CRC16-CCITT (matching serial_manager.py protocol)
# =============================================================================

def crc16_ccitt(data: bytes) -> int:
    """Compute CRC16-CCITT checksum (matching firmware protocol)."""
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


# =============================================================================
# SERIAL PACKET UTILITIES (matching serial_manager.py protocol)
# =============================================================================

PACKET_START = 0xAA
PACKET_END = 0x55


def build_packet(packet_type: int, payload: bytes, sequence: int = 0) -> bytes:
    """
    Build a binary serial packet matching firmware protocol.
    
    Format: [START=0xAA][LENGTH:2][TYPE:1][SEQUENCE:2][PAYLOAD:N][CRC16:2][END=0x55]
    """
    header = struct.pack('<BHB H', PACKET_START, len(payload), packet_type, sequence)
    data_for_crc = header + payload
    crc = crc16_ccitt(data_for_crc)
    footer = struct.pack('<H B', crc, PACKET_END)
    return header + payload + footer


def parse_packet(data: bytes) -> Optional[Dict[str, Any]]:
    """
    Parse a binary serial packet.
    
    Returns dict with type, sequence, payload, crc, valid fields, or None on failure.
    """
    HEADER_SIZE = 6
    FOOTER_SIZE = 3
    
    if len(data) < HEADER_SIZE + FOOTER_SIZE:
        return None
    
    try:
        start, length, ptype, sequence = struct.unpack('<BHB H', data[:HEADER_SIZE])
        if start != PACKET_START:
            return None
        
        payload_end = HEADER_SIZE + length
        payload = data[HEADER_SIZE:payload_end]
        
        checksum, end = struct.unpack('<H B', data[payload_end:payload_end + FOOTER_SIZE])
        if end != PACKET_END:
            return None
        
        # Verify CRC
        crc_data = data[:payload_end]
        computed_crc = crc16_ccitt(crc_data)
        
        return {
            "type": ptype,
            "sequence": sequence,
            "payload": payload,
            "crc": checksum,
            "valid": checksum == computed_crc,
            "raw": data[:payload_end + FOOTER_SIZE],
        }
    except struct.error:
        return None


# =============================================================================
# PORT SCANNER
# =============================================================================

class PortScanner:
    """Scan for serial/USB devices across platforms."""
    
    # Known device identifiers
    PICO_W_IDS = ["2e8a", "raspberry pi", "pico", "rp2040"]
    ARDUINO_IDS = ["arduino", "ch340", "cp210", "ftdi", "2341"]
    MOTOR_DRIVER_IDS = ["tb6600", "dm542", "dm556", "motor"]
    
    @staticmethod
    def scan_all() -> List[Dict[str, str]]:
        """Scan all available serial ports with device identification."""
        if not HAS_SERIAL:
            return []
        
        ports = []
        for port in serial.tools.list_ports.comports():
            device_class = PortScanner._identify_device(port)
            ports.append({
                "device": port.device,
                "name": port.name,
                "description": port.description,
                "hwid": port.hwid,
                "vid": f"0x{port.vid:04X}" if port.vid else "N/A",
                "pid": f"0x{port.pid:04X}" if port.pid else "N/A",
                "manufacturer": port.manufacturer or "Unknown",
                "serial_number": port.serial_number or "N/A",
                "device_class": device_class,
            })
        return ports
    
    @staticmethod
    def find_pico_w() -> Optional[str]:
        """Auto-detect RPi Pico W serial port."""
        if not HAS_SERIAL:
            return None
        for port in serial.tools.list_ports.comports():
            desc_lower = (port.description or "").lower()
            hwid_lower = (port.hwid or "").lower()
            mfr_lower = (port.manufacturer or "").lower()
            combined = f"{desc_lower} {hwid_lower} {mfr_lower}"
            if any(pid in combined for pid in PortScanner.PICO_W_IDS):
                return port.device
        return None
    
    @staticmethod
    def find_arduino() -> Optional[str]:
        """Auto-detect Arduino serial port."""
        if not HAS_SERIAL:
            return None
        for port in serial.tools.list_ports.comports():
            desc_lower = (port.description or "").lower()
            mfr_lower = (port.manufacturer or "").lower()
            combined = f"{desc_lower} {mfr_lower}"
            if any(pid in combined for pid in PortScanner.ARDUINO_IDS):
                return port.device
        return None
    
    @staticmethod
    def _identify_device(port) -> str:
        """Identify device type from port info."""
        combined = f"{port.description} {port.hwid} {port.manufacturer}".lower()
        
        if any(pid in combined for pid in PortScanner.PICO_W_IDS):
            return "RPi Pico W"
        if any(pid in combined for pid in PortScanner.ARDUINO_IDS):
            return "Arduino"
        if any(pid in combined for pid in PortScanner.MOTOR_DRIVER_IDS):
            return "Motor Driver"
        return "Unknown"


# =============================================================================
# BASE TEST CLASS
# =============================================================================

class HardwareTestBase(ABC):
    """
    Base class for all hardware component tests.
    
    Subclass and implement:
    - component_type: ComponentType
    - component_name: str
    - run_tests(): List[TestResult]
    """
    
    def __init__(self, verbose: bool = False, gui_mode: bool = False):
        self.verbose = verbose
        self.gui_mode = gui_mode
        self.display = TerminalDisplay()
        self.results: List[TestResult] = []
        self._serial: Optional[Any] = None  # serial.Serial when connected
        self._sequence = 0
    
    @property
    @abstractmethod
    def component_type(self) -> ComponentType:
        ...
    
    @property
    @abstractmethod
    def component_name(self) -> str:
        ...
    
    @abstractmethod
    async def run_tests(self) -> List[TestResult]:
        """Execute all tests for this component. Return list of TestResult."""
        ...
    
    def make_result(self, name: str, status: TestStatus = TestStatus.PENDING,
                    message: str = "") -> TestResult:
        """Create a TestResult pre-filled with component info."""
        return TestResult(
            name=name,
            component=self.component_type,
            status=status,
            message=message,
        )
    
    def timed_test(self, name: str) -> _TimedTestContext:
        """Context manager for timing a test."""
        return _TimedTestContext(name, self.component_type, self)
    
    # --- Serial helpers ---
    
    def open_serial(self, port: str, baudrate: int = 115200,
                    timeout: float = 2.0) -> bool:
        """Open serial port. Returns True on success."""
        if not HAS_SERIAL:
            return False
        try:
            self._serial = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=timeout,
                write_timeout=timeout,
            )
            time.sleep(2.0)  # Wait for device reset
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()
            return True
        except serial.SerialException as e:
            logger.error(f"Failed to open {port}: {e}")
            return False
    
    def close_serial(self) -> None:
        """Close serial port."""
        if self._serial and self._serial.is_open:
            self._serial.close()
            self._serial = None
    
    def send_packet(self, packet_type: int, payload: bytes = b"") -> Optional[bytes]:
        """
        Send a binary packet and return raw bytes sent.
        Displays hex dump in verbose mode.
        """
        if not self._serial:
            return None
        
        self._sequence = (self._sequence + 1) % 65536
        raw = build_packet(packet_type, payload, self._sequence)
        
        try:
            self._serial.write(raw)
            self._serial.flush()
            
            if self.verbose:
                print(TerminalDisplay.hex_dump(raw, "TX"))
                TerminalDisplay.packet_trace(
                    "TX",
                    f"0x{packet_type:02X}",
                    self._sequence,
                    len(payload),
                    crc16_ccitt(raw[:6 + len(payload)]),
                    True,
                )
            return raw
        except serial.SerialException as e:
            logger.error(f"Send failed: {e}")
            return None
    
    def receive_packet(self, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
        """
        Wait for and parse one incoming packet.
        Displays hex dump in verbose mode.
        """
        if not self._serial:
            return None
        
        old_timeout = self._serial.timeout
        self._serial.timeout = timeout
        
        try:
            # Read until we find start byte
            start_time = time.time()
            while time.time() - start_time < timeout:
                byte = self._serial.read(1)
                if not byte:
                    continue
                if byte[0] == PACKET_START:
                    # Read length (2 bytes)
                    length_bytes = self._serial.read(2)
                    if len(length_bytes) < 2:
                        continue
                    length = struct.unpack('<H', length_bytes)[0]
                    
                    # Read type(1) + sequence(2) + payload(length) + crc(2) + end(1)
                    remaining = 1 + 2 + length + 2 + 1
                    rest = self._serial.read(remaining)
                    if len(rest) < remaining:
                        continue
                    
                    full_packet = bytes([PACKET_START]) + length_bytes + rest
                    parsed = parse_packet(full_packet)
                    
                    if parsed and self.verbose:
                        print(TerminalDisplay.hex_dump(full_packet, "RX"))
                        TerminalDisplay.packet_trace(
                            "RX",
                            f"0x{parsed['type']:02X}",
                            parsed["sequence"],
                            len(parsed["payload"]),
                            parsed["crc"],
                            parsed["valid"],
                        )
                    
                    return parsed
            
            return None
        except serial.SerialException as e:
            logger.error(f"Receive failed: {e}")
            return None
        finally:
            self._serial.timeout = old_timeout
    
    def ping_device(self, retries: int = 3) -> Tuple[bool, float]:
        """
        Send heartbeat packet and measure round-trip latency.
        Returns (success, latency_ms).
        """
        for attempt in range(retries):
            start = time.perf_counter()
            sent = self.send_packet(0x01, b"\x00")  # HEARTBEAT
            if not sent:
                continue
            
            response = self.receive_packet(timeout=2.0)
            elapsed = (time.perf_counter() - start) * 1000
            
            if response and response["valid"]:
                return True, elapsed
        
        return False, -1.0
    
    async def execute(self) -> List[TestResult]:
        """Run all tests and display results."""
        self.display.banner(f"Testing: {self.component_name}")
        
        try:
            self.results = await self.run_tests()
        except Exception as e:
            result = self.make_result("Test execution", TestStatus.FAIL, str(e))
            self.results = [result]
        
        # Display results
        self.display.section("Results")
        for r in self.results:
            self.display.test_result(r)
        
        self.display.summary(self.results)
        
        return self.results
    
    def to_json(self) -> Dict[str, Any]:
        """Export results as JSON-serializable dict."""
        return {
            "component": self.component_type.value,
            "name": self.component_name,
            "timestamp": datetime.now().isoformat(),
            "results": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "message": r.message,
                    "duration_ms": r.duration_ms,
                    "packets_sent": r.packets_sent,
                    "packets_received": r.packets_received,
                    "packets_failed": r.packets_failed,
                    "data": r.data,
                }
                for r in self.results
            ],
        }


class _TimedTestContext:
    """Context manager for timing a test."""
    
    def __init__(self, name: str, component: ComponentType, parent: HardwareTestBase):
        self.result = TestResult(name=name, component=component)
        self.parent = parent
        self._start = 0.0
    
    def __enter__(self) -> TestResult:
        self._start = time.perf_counter()
        self.result.status = TestStatus.RUNNING
        return self.result
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.result.duration_ms = (time.perf_counter() - self._start) * 1000
        if exc_type:
            self.result.status = TestStatus.FAIL
            self.result.message = str(exc_val)
        elif self.result.status == TestStatus.RUNNING:
            self.result.status = TestStatus.PASS
        self.parent.results.append(self.result)
        return True  # Suppress exceptions


# =============================================================================
# GPIO UTILITIES (for RPi 5 direct pin testing)
# =============================================================================

class GPIOHelper:
    """GPIO pin control helper - works on RPi 5 with gpiod or lgpio."""
    
    _lib = None
    _chip = None
    
    @classmethod
    def available(cls) -> bool:
        """Check if GPIO is available (running on RPi)."""
        try:
            import lgpio
            cls._lib = "lgpio"
            return True
        except ImportError:
            pass
        try:
            import gpiod
            cls._lib = "gpiod"
            return True
        except ImportError:
            pass
        return False
    
    @classmethod
    def test_pin_output(cls, pin: int, duration: float = 0.1) -> bool:
        """Briefly toggle a GPIO pin HIGH then LOW. Returns True on success."""
        if cls._lib == "lgpio":
            import lgpio
            h = lgpio.gpiochip_open(0)
            try:
                lgpio.gpio_claim_output(h, pin)
                lgpio.gpio_write(h, pin, 1)
                time.sleep(duration)
                lgpio.gpio_write(h, pin, 0)
                return True
            except Exception as e:
                logger.error(f"GPIO pin {pin} test failed: {e}")
                return False
            finally:
                lgpio.gpiochip_close(h)
        
        elif cls._lib == "gpiod":
            import gpiod
            chip = gpiod.Chip("gpiochip0")
            try:
                line = chip.get_line(pin)
                line.request(consumer="rpm_test", type=gpiod.LINE_REQ_DIR_OUT)
                line.set_value(1)
                time.sleep(duration)
                line.set_value(0)
                line.release()
                return True
            except Exception as e:
                logger.error(f"GPIO pin {pin} test failed: {e}")
                return False
        
        return False
    
    @classmethod
    def read_pin(cls, pin: int) -> Optional[int]:
        """Read a GPIO pin value. Returns 0 or 1, or None on failure."""
        if cls._lib == "lgpio":
            import lgpio
            h = lgpio.gpiochip_open(0)
            try:
                lgpio.gpio_claim_input(h, pin)
                val = lgpio.gpio_read(h, pin)
                return val
            except Exception:
                return None
            finally:
                lgpio.gpiochip_close(h)
        return None


# =============================================================================
# MAIN ENTRY FOR LISTING AVAILABLE HARDWARE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RPM Hardware Test Framework")
    parser.add_argument("--list", "-l", action="store_true", help="List all serial ports")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    if args.list:
        display = TerminalDisplay()
        display.banner("PORT SCANNER")
        
        display.section("Available Serial Ports")
        ports = PortScanner.scan_all()
        
        if not ports:
            print(f"  {C.YELLOW}No serial ports detected{C.RESET}")
            print(f"  {C.DIM}Ensure devices are connected via USB{C.RESET}")
        else:
            for p in ports:
                icon = C.GREEN + "●" + C.RESET if p["device_class"] != "Unknown" else C.GRAY + "○" + C.RESET
                print(f"  {icon} {p['device']:30s} {C.CYAN}{p['device_class']:15s}{C.RESET} "
                      f"{C.DIM}{p['description']}{C.RESET}")
                if args.verbose:
                    print(f"      VID={p['vid']} PID={p['pid']} "
                          f"SN={p['serial_number']} MFR={p['manufacturer']}")
        
        display.section("Auto-Detection")
        pico = PortScanner.find_pico_w()
        arduino = PortScanner.find_arduino()
        
        print(f"  RPi Pico W:  {C.GREEN + pico + C.RESET if pico else C.RED + 'Not found' + C.RESET}")
        print(f"  Arduino:     {C.GREEN + arduino + C.RESET if arduino else C.RED + 'Not found' + C.RESET}")
        print()
