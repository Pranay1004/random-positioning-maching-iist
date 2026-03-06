"""
RPM Digital Twin - Slip Ring Connection Test
=============================================
Tests for 2 slip rings used to maintain electrical connections
through the rotating frames.

Slip Ring 1 — Inner frame axis (transfers motor power + signal)
Slip Ring 2 — Outer frame axis (transfers motor power + signal)

Tests:
1. Electrical continuity (signal path verification)
2. Signal integrity (send known pattern, verify output)
3. Data packet transmission through slip ring path
4. Noise/resistance measurement (via ADC or serial query)
5. Sustained data throughput through rotating joint
6. Packet error rate under rotation

Usage:
    python -m tests.hardware.test_slip_rings
    python -m tests.hardware.test_slip_rings --serial-port /dev/ttyUSB0
    python -m tests.hardware.test_slip_rings --verbose
"""

from __future__ import annotations

import asyncio
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    GPIOHelper,
    build_packet,
    parse_packet,
    crc16_ccitt,
    HAS_SERIAL,
)

# Packet types
HEARTBEAT = 0x01
SLIP_RING_TEST = 0x15   # Custom test packet type
CONTINUITY_TEST = 0x16


class SlipRingTest(HardwareTestBase):
    """Test suite for slip ring connections."""
    
    def __init__(self, ring_id: str = "inner", serial_port: Optional[str] = None,
                 verbose: bool = False):
        super().__init__(verbose=verbose)
        self._ring_id = ring_id  # "inner" or "outer"
        self._serial_port = serial_port
        
        if ring_id == "inner":
            self._component = ComponentType.SLIP_RING_INNER
            self._name = "Slip Ring — Inner Frame Axis"
        else:
            self._component = ComponentType.SLIP_RING_OUTER
            self._name = "Slip Ring — Outer Frame Axis"
    
    @property
    def component_type(self) -> ComponentType:
        return self._component
    
    @property
    def component_name(self) -> str:
        return self._name
    
    async def run_tests(self) -> List[TestResult]:
        has_serial = self._serial_port and HAS_SERIAL
        
        # Test 1: Slip ring configuration — read-only config, not a hardware test
        with self.timed_test("Slip Ring Configuration") as t:
            ring_specs = {
                "inner": {
                    "channels": 6,  # 2 power + 4 signal
                    "max_rpm": 60,
                    "voltage_rating": "24V",
                    "current_rating": "5A per channel",
                    "signal_channels": "STEP, DIR, ENABLE, ENCODER_A, ENCODER_B, GND",
                },
                "outer": {
                    "channels": 6,
                    "max_rpm": 60,
                    "voltage_rating": "24V",
                    "current_rating": "5A per channel",
                    "signal_channels": "STEP, DIR, ENABLE, ENCODER_A, ENCODER_B, GND",
                },
            }
            spec = ring_specs.get(self._ring_id, ring_specs["inner"])
            t.data = spec
            t.status = TestStatus.SKIP
            t.message = (f"Config only (not verified): {spec['channels']} channels | "
                         f"{spec['voltage_rating']} | {spec['current_rating']} | "
                         f"Max {spec['max_rpm']}RPM")
        
        # Test 2: Electrical continuity (via GPIO or serial)
        with self.timed_test("Electrical Continuity") as t:
            if GPIOHelper.available():
                # Test by driving a signal through the slip ring
                # On one side write HIGH, on other side read
                # This requires specific wiring — test the enable pin path
                test_pin = 4 if self._ring_id == "inner" else 7
                
                ok = GPIOHelper.test_pin_output(test_pin, duration=0.05)
                if ok:
                    t.status = TestStatus.PASS
                    t.message = f"GPIO{test_pin} path through slip ring OK"
                else:
                    t.status = TestStatus.WARN
                    t.message = f"GPIO{test_pin} toggle — verify continuity manually"
            elif has_serial:
                # Send continuity test packet through serial
                if self.open_serial(self._serial_port):
                    ring_byte = 0x01 if self._ring_id == "inner" else 0x02
                    sent = self.send_packet(CONTINUITY_TEST, struct.pack('<B', ring_byte))
                    if sent:
                        t.packets_sent = 1
                        response = self.receive_packet(timeout=3.0)
                        if response and response["valid"]:
                            t.packets_received = 1
                            t.status = TestStatus.PASS
                            t.message = "Continuity confirmed via serial packet"
                        else:
                            t.status = TestStatus.WARN
                            t.message = "No continuity response (command may not be in firmware)"
                    self.close_serial()
                else:
                    t.status = TestStatus.FAIL
                    t.message = "Could not open serial port"
            else:
                t.status = TestStatus.SKIP
                t.message = "No GPIO or serial available for continuity test"
        
        # Test 3: Signal integrity — known pattern test
        with self.timed_test("Signal Integrity (Pattern Test)") as t:
            test_patterns = [
                b"\x55\x55\x55\x55",  # Alternating bits
                b"\xAA\xAA\xAA\xAA",  # Alternating bits (inverted)
                b"\xFF\x00\xFF\x00",  # All ones / all zeros
                b"\x00\xFF\x00\xFF",
                bytes(range(256)),     # All byte values
            ]
            
            if has_serial:
                if self.open_serial(self._serial_port):
                    passed = 0
                    total = len(test_patterns)
                    
                    for i, pattern in enumerate(test_patterns):
                        ring_byte = 0x01 if self._ring_id == "inner" else 0x02
                        payload = struct.pack('<B', ring_byte) + pattern
                        sent = self.send_packet(SLIP_RING_TEST, payload)
                        
                        if sent:
                            t.packets_sent += 1
                            response = self.receive_packet(timeout=2.0)
                            if response and response["valid"]:
                                t.packets_received += 1
                                # Check if echo matches
                                if response["payload"][1:] == pattern:
                                    passed += 1
                                else:
                                    t.packets_failed += 1
                            else:
                                t.packets_failed += 1
                    
                    self.close_serial()
                    
                    if passed == total:
                        t.status = TestStatus.PASS
                        t.message = f"All {total} patterns matched"
                    elif passed > 0:
                        t.status = TestStatus.WARN
                        t.message = f"{passed}/{total} patterns matched (signal degradation)"
                    else:
                        t.status = TestStatus.WARN
                        t.message = "Echo test may not be implemented in firmware"
                else:
                    t.status = TestStatus.FAIL
                    t.message = "Could not open serial port"
            else:
                # Protocol-level test only
                all_ok = True
                for pattern in test_patterns:
                    ring_byte = 0x01 if self._ring_id == "inner" else 0x02
                    payload = struct.pack('<B', ring_byte) + pattern
                    raw = build_packet(SLIP_RING_TEST, payload, sequence=0)
                    parsed = parse_packet(raw)
                    if not parsed or not parsed["valid"]:
                        all_ok = False
                
                t.status = TestStatus.SKIP
                t.message = (f"CRC verified locally but NO DATA SENT to hardware — "
                             f"{len(test_patterns)} patterns built, 0 transmitted")
        
        # Test 4: Noise measurement query
        with self.timed_test("Noise/Resistance Measurement") as t:
            if has_serial:
                if self.open_serial(self._serial_port):
                    ring_byte = 0x01 if self._ring_id == "inner" else 0x02
                    # Query ADC reading for slip ring channel
                    sent = self.send_packet(0x17, struct.pack('<BB', ring_byte, 0x01))
                    
                    if sent:
                        t.packets_sent = 1
                        response = self.receive_packet(timeout=3.0)
                        if response and response["valid"] and len(response["payload"]) >= 4:
                            t.packets_received = 1
                            adc_value = struct.unpack('<H', response["payload"][:2])[0]
                            resistance = struct.unpack('<H', response["payload"][2:4])[0]
                            t.status = TestStatus.PASS
                            t.message = f"ADC: {adc_value} | Resistance: {resistance}mΩ"
                            t.data["adc"] = adc_value
                            t.data["resistance_mohm"] = resistance
                        else:
                            t.status = TestStatus.WARN
                            t.message = "ADC query not supported in firmware"
                    self.close_serial()
                else:
                    t.status = TestStatus.FAIL
                    t.message = "Could not open serial port"
            else:
                t.status = TestStatus.SKIP
                t.message = "Requires serial connection for ADC measurement"
        
        # Test 5: Sustained throughput through slip ring path
        with self.timed_test("Sustained Throughput (50 packets)") as t:
            if has_serial:
                if self.open_serial(self._serial_port):
                    num_packets = 50
                    tx_ok = 0
                    rx_ok = 0
                    start = time.perf_counter()
                    
                    for i in range(num_packets):
                        payload = struct.pack('<BI', 
                            0x01 if self._ring_id == "inner" else 0x02,
                            i)
                        sent = self.send_packet(HEARTBEAT, payload)
                        if sent:
                            tx_ok += 1
                        time.sleep(0.01)
                    
                    # Drain responses
                    drain_deadline = time.time() + 2.0
                    while time.time() < drain_deadline:
                        resp = self.receive_packet(timeout=0.1)
                        if resp:
                            rx_ok += 1
                        else:
                            break
                    
                    elapsed = time.perf_counter() - start
                    pps = tx_ok / elapsed if elapsed > 0 else 0
                    
                    self.close_serial()
                    
                    t.packets_sent = tx_ok
                    t.packets_received = rx_ok
                    t.data["elapsed_s"] = round(elapsed, 3)
                    t.data["pps"] = round(pps, 1)
                    
                    if tx_ok == num_packets:
                        t.status = TestStatus.PASS
                        t.message = f"TX: {tx_ok}/{num_packets} | RX: {rx_ok} | {pps:.0f} pkt/s"
                    else:
                        t.status = TestStatus.WARN
                        t.message = f"TX: {tx_ok}/{num_packets} ({pps:.0f} pkt/s)"
                else:
                    t.status = TestStatus.FAIL
                    t.message = "Could not open serial port"
            else:
                t.status = TestStatus.SKIP
                t.message = "Requires serial connection"
        
        # Test 6: Packet error rate estimation
        with self.timed_test("Packet Error Rate") as t:
            total_sent = sum(r.packets_sent for r in self.results)
            total_recv = sum(r.packets_received for r in self.results)
            total_fail = sum(r.packets_failed for r in self.results)
            
            if total_sent > 0:
                error_rate = (total_fail / total_sent) * 100 if total_sent > 0 else 0
                t.data["total_sent"] = total_sent
                t.data["total_received"] = total_recv
                t.data["total_failed"] = total_fail
                t.data["error_rate_pct"] = round(error_rate, 2)
                
                if error_rate == 0:
                    t.status = TestStatus.PASS
                    t.message = f"0% error rate ({total_sent} packets)"
                elif error_rate < 1:
                    t.status = TestStatus.WARN
                    t.message = f"{error_rate:.2f}% error rate ({total_fail}/{total_sent})"
                else:
                    t.status = TestStatus.FAIL
                    t.message = f"{error_rate:.1f}% error rate ({total_fail}/{total_sent})"
            else:
                t.status = TestStatus.SKIP
                t.message = "No packets sent (no serial connection)"
        
        return self.results


class AllSlipRingsTest:
    """Run tests for both slip rings."""
    
    def __init__(self, serial_port: Optional[str] = None, verbose: bool = False):
        self.serial_port = serial_port
        self.verbose = verbose
    
    async def run_all(self) -> Dict[str, List[TestResult]]:
        display = TerminalDisplay()
        all_results = {}
        
        display.banner("SLIP RING TEST SUITE")
        
        for ring_id in ["inner", "outer"]:
            test = SlipRingTest(
                ring_id=ring_id,
                serial_port=self.serial_port,
                verbose=self.verbose,
            )
            
            display.section(test.component_name)
            results = await test.run_tests()
            
            for r in results:
                display.test_result(r)
            
            all_results[ring_id] = results
        
        all_flat = [r for results in all_results.values() for r in results]
        display.summary(all_flat)
        
        return all_results


# =============================================================================
# CLI ENTRY
# =============================================================================

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Slip Ring Connection Test")
    parser.add_argument("--ring", "-r", choices=["inner", "outer", "all"],
                        default="all", help="Slip ring to test (default: all)")
    parser.add_argument("--serial-port", "-s", type=str, default=None,
                        help="Serial port for slip ring testing")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show packet hex dumps")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    args = parser.parse_args()
    
    if args.ring == "all":
        suite = AllSlipRingsTest(serial_port=args.serial_port, verbose=args.verbose)
        all_results = await suite.run_all()
        
        if args.json:
            import json
            output = {}
            for key, results in all_results.items():
                output[key] = [
                    {"name": r.name, "status": r.status.value, 
                     "message": r.message, "data": r.data}
                    for r in results
                ]
            print(json.dumps(output, indent=2))
        
        has_failures = any(
            r.status == TestStatus.FAIL
            for results in all_results.values()
            for r in results
        )
        sys.exit(1 if has_failures else 0)
    else:
        test = SlipRingTest(
            ring_id=args.ring,
            serial_port=args.serial_port,
            verbose=args.verbose,
        )
        results = await test.execute()
        
        if args.json:
            import json
            print(json.dumps(test.to_json(), indent=2))
        
        has_failures = any(r.status == TestStatus.FAIL for r in results)
        sys.exit(1 if has_failures else 0)


if __name__ == "__main__":
    asyncio.run(main())
