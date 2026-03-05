"""
RPM Digital Twin - Raspberry Pi Pico W Hardware Test
====================================================
Comprehensive test suite for RPi Pico W microcontroller.

Tests:
1. USB serial connection detection
2. Serial port open/close
3. Heartbeat ping (latency measurement)
4. Firmware version query
5. Packet protocol validation (CRC16-CCITT)
6. Bidirectional data transfer
7. WiFi status query (Pico W specific)
8. Sustained throughput test
9. Error recovery test

Usage:
    python -m tests.hardware.test_pico_w
    python -m tests.hardware.test_pico_w --port /dev/tty.usbmodem1101
    python -m tests.hardware.test_pico_w --verbose
"""

from __future__ import annotations

import asyncio
import struct
import sys
import time
from pathlib import Path
from typing import List, Optional

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tests.hardware.test_base import (
    HardwareTestBase,
    ComponentType,
    TestResult,
    TestStatus,
    PortScanner,
    TerminalDisplay,
    TerminalColors as C,
    build_packet,
    parse_packet,
    crc16_ccitt,
    PACKET_START,
    PACKET_END,
    HAS_SERIAL,
)

# Packet types matching firmware protocol
HEARTBEAT = 0x01
IMU_DATA = 0x02
ENCODER_DATA = 0x03
MOTOR_STATUS = 0x04
MOTOR_COMMAND = 0x05
CONFIG = 0x06
ACK = 0x07
FW_VERSION = 0x10
WIFI_STATUS = 0x11
SELF_TEST = 0x12
ERROR = 0xFF


class PicoWTest(HardwareTestBase):
    """Complete test suite for RPi Pico W microcontroller."""
    
    def __init__(self, port: Optional[str] = None, baudrate: int = 115200,
                 verbose: bool = False):
        super().__init__(verbose=verbose)
        self._port = port
        self._baudrate = baudrate
        self._detected_port: Optional[str] = None
    
    @property
    def component_type(self) -> ComponentType:
        return ComponentType.RPI_PICO_W
    
    @property
    def component_name(self) -> str:
        return "Raspberry Pi Pico W"
    
    async def run_tests(self) -> List[TestResult]:
        results = []
        
        # Test 1: Port detection
        with self.timed_test("USB Port Auto-Detection") as t:
            if self._port:
                self._detected_port = self._port
                t.status = TestStatus.PASS
                t.message = f"Using specified port: {self._port}"
            else:
                self._detected_port = PortScanner.find_pico_w()
                if self._detected_port:
                    t.status = TestStatus.PASS
                    t.message = f"Detected on {self._detected_port}"
                else:
                    # List what was found
                    all_ports = PortScanner.scan_all()
                    if all_ports:
                        ports_str = ", ".join(p["device"] for p in all_ports)
                        t.status = TestStatus.FAIL
                        t.message = f"Pico W not found. Available ports: {ports_str}"
                    else:
                        t.status = TestStatus.FAIL
                        t.message = "No serial ports detected. Check USB connection."
        
        if not self._detected_port:
            # Can't continue without a port
            remaining = [
                "Serial Connection", "Heartbeat Ping", "Firmware Version",
                "Packet Protocol CRC16", "Bidirectional Data Transfer",
                "WiFi Status", "Sustained Throughput", "Error Recovery"
            ]
            for name in remaining:
                r = self.make_result(name, TestStatus.SKIP, "No port detected")
                results.append(r)
            # Results already added by timed_test context
            return self.results + results
        
        # Test 2: Serial connection
        with self.timed_test("Serial Connection") as t:
            if not HAS_SERIAL:
                t.status = TestStatus.FAIL
                t.message = "pyserial not installed. Run: pip install pyserial"
            elif self.open_serial(self._detected_port, self._baudrate):
                t.status = TestStatus.PASS
                t.message = f"Connected at {self._baudrate} baud"
            else:
                t.status = TestStatus.FAIL
                t.message = f"Failed to open {self._detected_port}"
        
        if not self._serial:
            remaining = [
                "Heartbeat Ping", "Firmware Version", "Packet Protocol CRC16",
                "Bidirectional Data Transfer", "WiFi Status",
                "Sustained Throughput", "Error Recovery"
            ]
            for name in remaining:
                r = self.make_result(name, TestStatus.SKIP, "Serial not connected")
                results.append(r)
            return self.results + results
        
        try:
            # Test 3: Heartbeat ping
            with self.timed_test("Heartbeat Ping") as t:
                success, latency = self.ping_device(retries=3)
                if success:
                    t.status = TestStatus.PASS
                    t.message = f"RTT: {latency:.1f}ms"
                    t.data["latency_ms"] = latency
                    t.packets_sent = 1
                    t.packets_received = 1
                else:
                    t.status = TestStatus.FAIL
                    t.message = "No heartbeat response after 3 attempts"
                    t.packets_sent = 3
                    t.packets_failed = 3
            
            # Test 4: Firmware version query
            with self.timed_test("Firmware Version Query") as t:
                sent = self.send_packet(FW_VERSION, b"\x00")
                if sent:
                    t.packets_sent = 1
                    response = self.receive_packet(timeout=3.0)
                    if response and response["valid"]:
                        t.packets_received = 1
                        payload = response["payload"]
                        if len(payload) >= 4:
                            major, minor, patch = struct.unpack('<BBB', payload[:3])
                            fw_str = f"v{major}.{minor}.{patch}"
                            t.status = TestStatus.PASS
                            t.message = f"Firmware: {fw_str}"
                            t.data["firmware"] = fw_str
                        else:
                            t.status = TestStatus.WARN
                            t.message = f"Response received but unexpected payload size: {len(payload)}"
                    else:
                        t.status = TestStatus.WARN
                        t.message = "No firmware version response (command may not be implemented)"
                else:
                    t.status = TestStatus.FAIL
                    t.message = "Failed to send firmware query"
            
            # Test 5: Packet protocol CRC16 validation
            with self.timed_test("Packet Protocol CRC16") as t:
                test_payloads = [
                    b"\x00",
                    b"\x01\x02\x03\x04",
                    struct.pack('<6f', 0.1, 0.2, 9.8, 0.01, 0.02, 0.03),  # IMU-like
                    bytes(range(64)),  # 64 bytes
                ]
                
                all_valid = True
                for i, payload in enumerate(test_payloads):
                    raw = build_packet(0x01, payload, sequence=i)
                    parsed = parse_packet(raw)
                    if not parsed or not parsed["valid"]:
                        all_valid = False
                        t.message = f"CRC mismatch on test case {i+1}"
                        break
                
                if all_valid:
                    t.status = TestStatus.PASS
                    t.message = f"All {len(test_payloads)} CRC test cases passed"
                else:
                    t.status = TestStatus.FAIL
            
            # Test 6: Bidirectional data transfer
            with self.timed_test("Bidirectional Data Transfer") as t:
                # Send a config packet with echo request
                echo_data = struct.pack('<I', 0xDEADBEEF)
                sent = self.send_packet(CONFIG, b"\x01" + echo_data)  # Echo request
                
                if sent:
                    t.packets_sent = 1
                    response = self.receive_packet(timeout=3.0)
                    if response and response["valid"]:
                        t.packets_received = 1
                        t.status = TestStatus.PASS
                        t.message = f"Echo response: {len(response['payload'])} bytes"
                    elif response:
                        t.packets_received = 1
                        t.packets_failed = 1
                        t.status = TestStatus.WARN
                        t.message = "Response received but CRC invalid"
                    else:
                        t.status = TestStatus.WARN
                        t.message = "No echo response (echo may not be implemented)"
                else:
                    t.status = TestStatus.FAIL
                    t.message = "Failed to send echo request"
            
            # Test 7: WiFi status
            with self.timed_test("WiFi Status (Pico W)") as t:
                sent = self.send_packet(WIFI_STATUS, b"\x00")
                if sent:
                    t.packets_sent = 1
                    response = self.receive_packet(timeout=3.0)
                    if response and response["valid"]:
                        t.packets_received = 1
                        payload = response["payload"]
                        if len(payload) >= 1:
                            wifi_connected = payload[0] == 0x01
                            t.status = TestStatus.PASS if wifi_connected else TestStatus.WARN
                            t.message = "WiFi: Connected" if wifi_connected else "WiFi: Not connected"
                            
                            # Parse SSID if available
                            if len(payload) > 5:
                                rssi = struct.unpack('<b', payload[1:2])[0]
                                ssid_len = payload[2]
                                ssid = payload[3:3+ssid_len].decode('utf-8', errors='replace')
                                t.data["rssi"] = rssi
                                t.data["ssid"] = ssid
                                t.message += f" | SSID: {ssid} | RSSI: {rssi}dBm"
                        else:
                            t.status = TestStatus.WARN
                            t.message = "Empty WiFi status response"
                    else:
                        t.status = TestStatus.WARN
                        t.message = "WiFi query not supported in current firmware"
                else:
                    t.status = TestStatus.FAIL
                    t.message = "Failed to send WiFi query"
            
            # Test 8: Sustained throughput
            with self.timed_test("Sustained Throughput (100 packets)") as t:
                num_packets = 100
                tx_count = 0
                rx_count = 0
                start = time.perf_counter()
                
                for i in range(num_packets):
                    payload = struct.pack('<I', i)
                    sent = self.send_packet(0x01, payload)
                    if sent:
                        tx_count += 1
                    # Brief delay to avoid overwhelming the Pico
                    time.sleep(0.005)
                
                # Read any responses (non-blocking drain)
                if self._serial:
                    self._serial.timeout = 0.1
                    drain_start = time.time()
                    while time.time() - drain_start < 2.0:
                        byte = self._serial.read(1)
                        if not byte:
                            break
                        if byte[0] == PACKET_START:
                            length_bytes = self._serial.read(2)
                            if len(length_bytes) == 2:
                                length = struct.unpack('<H', length_bytes)[0]
                                rest = self._serial.read(1 + 2 + length + 2 + 1)
                                if rest:
                                    rx_count += 1
                
                elapsed = (time.perf_counter() - start)
                pps = tx_count / elapsed if elapsed > 0 else 0
                
                t.packets_sent = tx_count
                t.packets_received = rx_count
                t.data["packets_per_sec"] = round(pps, 1)
                t.data["elapsed_sec"] = round(elapsed, 3)
                
                if tx_count == num_packets:
                    t.status = TestStatus.PASS
                    t.message = f"TX: {tx_count} pkts in {elapsed:.2f}s ({pps:.0f} pkt/s)"
                else:
                    t.status = TestStatus.WARN
                    t.message = f"TX: {tx_count}/{num_packets} pkts ({pps:.0f} pkt/s)"
            
            # Test 9: Error recovery
            with self.timed_test("Error Recovery (malformed packet)") as t:
                # Send garbage data, then a valid heartbeat
                if self._serial:
                    self._serial.write(b"\xFF\xFE\xFD\xFC\xFB")
                    self._serial.flush()
                    time.sleep(0.1)
                    
                    # Now send valid heartbeat
                    success, latency = self.ping_device(retries=3)
                    if success:
                        t.status = TestStatus.PASS
                        t.message = f"Recovered after garbage data. RTT: {latency:.1f}ms"
                    else:
                        t.status = TestStatus.WARN
                        t.message = "Device did not respond after garbage data"
                else:
                    t.status = TestStatus.SKIP
                    t.message = "Serial not available"
        
        finally:
            self.close_serial()
        
        return self.results


# =============================================================================
# CLI ENTRY
# =============================================================================

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="RPi Pico W Hardware Test")
    parser.add_argument("--port", "-p", type=str, default=None,
                        help="Serial port (default: auto-detect)")
    parser.add_argument("--baudrate", "-b", type=int, default=115200,
                        help="Baud rate (default: 115200)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show packet hex dumps")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    args = parser.parse_args()
    
    test = PicoWTest(port=args.port, baudrate=args.baudrate, verbose=args.verbose)
    results = await test.execute()
    
    if args.json:
        import json
        print(json.dumps(test.to_json(), indent=2))
    
    # Exit code: 0 if all pass, 1 if any fail
    has_failures = any(r.status == TestStatus.FAIL for r in results)
    sys.exit(1 if has_failures else 0)


if __name__ == "__main__":
    asyncio.run(main())
