"""
RPM Digital Twin - NEMA Stepper Motor Test Suite
=================================================
Tests for NEMA 23 (x2) and NEMA 24 (x1) stepper motors.

Hardware Configuration (from config/main_config.yaml):
  Inner Frame — NEMA 23, TB6600 driver: DIR=GPIO2, STEP=GPIO3, EN=GPIO4
  Outer Frame — NEMA 23, TB6600 driver: DIR=GPIO5, STEP=GPIO6, EN=GPIO7
  Third Axis  — NEMA 24, TB6600 driver: DIR=GPIO22, STEP=GPIO23, EN=GPIO24

Tests per motor:
1. GPIO pin existence & access
2. Enable/disable pin toggle
3. Direction pin toggle
4. Step pulse generation (10 steps)
5. Encoder feedback (if connected via serial)
6. Speed ramp test (acceleration profile)
7. Motor command packet (serial protocol)
8. Emergency stop response

Usage:
    python -m tests.hardware.test_motors
    python -m tests.hardware.test_motors --motor inner
    python -m tests.hardware.test_motors --motor outer
    python -m tests.hardware.test_motors --motor nema24
    python -m tests.hardware.test_motors --verbose
    python -m tests.hardware.test_motors --dry-run    # no GPIO writes
"""

from __future__ import annotations

import asyncio
import struct
import sys
import time
from dataclasses import dataclass
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
    HAS_SERIAL,
)

# Packet types
MOTOR_COMMAND = 0x05
MOTOR_STATUS = 0x04
ENCODER_DATA = 0x03
HEARTBEAT = 0x01


@dataclass
class MotorPinConfig:
    """GPIO pin assignments for a single motor."""
    name: str
    motor_type: str  # NEMA23 or NEMA24
    direction_pin: int
    step_pin: int
    enable_pin: int
    component_type: ComponentType
    steps_per_rev: int = 200
    microstepping: int = 16
    max_rpm: float = 60.0


# Motor configurations from the project
MOTOR_CONFIGS = {
    "inner": MotorPinConfig(
        name="Inner Frame Motor",
        motor_type="NEMA23",
        direction_pin=2,
        step_pin=3,
        enable_pin=4,
        component_type=ComponentType.NEMA_23_INNER,
    ),
    "outer": MotorPinConfig(
        name="Outer Frame Motor",
        motor_type="NEMA23",
        direction_pin=5,
        step_pin=6,
        enable_pin=7,
        component_type=ComponentType.NEMA_23_OUTER,
    ),
    "nema24": MotorPinConfig(
        name="Third Axis Motor",
        motor_type="NEMA24",
        direction_pin=22,
        step_pin=23,
        enable_pin=24,
        component_type=ComponentType.NEMA_24,
        steps_per_rev=200,
        microstepping=16,
        max_rpm=40.0,
    ),
}


class MotorTest(HardwareTestBase):
    """Test suite for a single NEMA stepper motor."""
    
    def __init__(self, motor_key: str, serial_port: Optional[str] = None,
                 dry_run: bool = False, verbose: bool = False):
        super().__init__(verbose=verbose)
        self._motor_key = motor_key
        self._config = MOTOR_CONFIGS[motor_key]
        self._serial_port = serial_port
        self._dry_run = dry_run
        self._has_gpio = GPIOHelper.available()
        self._gpio_lib = None
    
    @property
    def component_type(self) -> ComponentType:
        return self._config.component_type
    
    @property
    def component_name(self) -> str:
        return f"{self._config.name} ({self._config.motor_type})"
    
    def _step_motor(self, steps: int, delay_us: float = 500.0) -> bool:
        """
        Generate step pulses via GPIO.
        delay_us: microsecond delay between step edges (controls speed).
        """
        if self._dry_run or not self._has_gpio:
            return True
        
        delay_s = delay_us / 1_000_000
        
        try:
            if GPIOHelper._lib == "lgpio":
                import lgpio
                h = lgpio.gpiochip_open(0)
                try:
                    lgpio.gpio_claim_output(h, self._config.step_pin)
                    for _ in range(steps):
                        lgpio.gpio_write(h, self._config.step_pin, 1)
                        time.sleep(delay_s)
                        lgpio.gpio_write(h, self._config.step_pin, 0)
                        time.sleep(delay_s)
                    return True
                finally:
                    lgpio.gpiochip_close(h)
            return False
        except Exception as e:
            if self.verbose:
                print(f"    {C.RED}Step error: {e}{C.RESET}")
            return False
    
    async def run_tests(self) -> List[TestResult]:
        
        pins = {
            "DIRECTION": self._config.direction_pin,
            "STEP": self._config.step_pin,
            "ENABLE": self._config.enable_pin,
        }
        
        # Test 1: GPIO pin existence
        with self.timed_test("GPIO Pin Configuration") as t:
            t.data["pins"] = {k: v for k, v in pins.items()}
            t.data["motor_type"] = self._config.motor_type
            t.data["steps_per_rev"] = self._config.steps_per_rev
            t.data["microstepping"] = self._config.microstepping
            
            total_steps = self._config.steps_per_rev * self._config.microstepping
            pin_info = (f"DIR=GPIO{pins['DIRECTION']} STEP=GPIO{pins['STEP']} "
                        f"EN=GPIO{pins['ENABLE']} | {total_steps} steps/rev")
            if self._has_gpio:
                t.status = TestStatus.PASS
                t.message = pin_info
            else:
                t.status = TestStatus.SKIP
                t.message = f"Config: {pin_info} — NOT verified (no GPIO)"
        
        # Test 2: GPIO access
        with self.timed_test("GPIO Access") as t:
            if self._dry_run:
                t.status = TestStatus.SKIP
                t.message = "Dry-run mode — GPIO NOT tested"
            elif self._has_gpio:
                t.status = TestStatus.PASS
                t.message = f"GPIO available via {GPIOHelper._lib}"
            else:
                t.status = TestStatus.FAIL
                t.message = "GPIO not available (not on RPi or libraries missing)"
        
        # Test 3: Enable pin toggle
        with self.timed_test("Enable Pin Toggle") as t:
            if self._dry_run:
                t.status = TestStatus.SKIP
                t.message = "Dry-run: EN pin NOT tested"
            elif self._has_gpio:
                ok = GPIOHelper.test_pin_output(self._config.enable_pin, duration=0.05)
                t.status = TestStatus.PASS if ok else TestStatus.FAIL
                t.message = f"GPIO{self._config.enable_pin} {'toggled OK' if ok else 'FAILED'}"
            else:
                t.status = TestStatus.SKIP
                t.message = "GPIO not available"
        
        # Test 4: Direction pin toggle
        with self.timed_test("Direction Pin Toggle") as t:
            if self._dry_run:
                t.status = TestStatus.SKIP
                t.message = "Dry-run: DIR pin NOT tested"
            elif self._has_gpio:
                ok = GPIOHelper.test_pin_output(self._config.direction_pin, duration=0.05)
                t.status = TestStatus.PASS if ok else TestStatus.FAIL
                t.message = f"GPIO{self._config.direction_pin} {'toggled OK' if ok else 'FAILED'}"
            else:
                t.status = TestStatus.SKIP
                t.message = "GPIO not available"
        
        # Test 5: Step pulse generation (10 micro-steps)
        with self.timed_test("Step Pulse Generation (10 steps)") as t:
            if self._dry_run:
                t.status = TestStatus.SKIP
                t.message = "Dry-run: step pulses NOT generated"
            elif self._has_gpio:
                # Enable motor first
                GPIOHelper.test_pin_output(self._config.enable_pin, duration=0.01)
                
                ok = self._step_motor(10, delay_us=500)
                
                # Disable motor
                if GPIOHelper._lib == "lgpio":
                    import lgpio
                    h = lgpio.gpiochip_open(0)
                    lgpio.gpio_claim_output(h, self._config.enable_pin)
                    lgpio.gpio_write(h, self._config.enable_pin, 1)  # HIGH = disabled for TB6600
                    lgpio.gpiochip_close(h)
                
                t.status = TestStatus.PASS if ok else TestStatus.FAIL
                t.message = "10 step pulses generated" if ok else "Step pulse generation failed"
            else:
                t.status = TestStatus.SKIP
                t.message = "GPIO not available"
        
        # Test 6: Encoder feedback via serial
        with self.timed_test("Encoder Feedback (Serial)") as t:
            if self._serial_port and HAS_SERIAL:
                if self.open_serial(self._serial_port):
                    # Request encoder data
                    motor_byte = 0x01 if self._motor_key == "inner" else 0x02
                    if self._motor_key == "nema24":
                        motor_byte = 0x03
                    
                    sent = self.send_packet(ENCODER_DATA, struct.pack('<B', motor_byte))
                    if sent:
                        t.packets_sent = 1
                        response = self.receive_packet(timeout=2.0)
                        if response and response["valid"]:
                            t.packets_received = 1
                            try:
                                _, counts, position, velocity = struct.unpack(
                                    '<B i f f', response["payload"][:13]
                                )
                                t.status = TestStatus.PASS
                                t.message = (f"Counts: {counts} | Pos: {position:.3f} rad | "
                                             f"Vel: {velocity:.3f} rad/s")
                                t.data["counts"] = counts
                                t.data["position_rad"] = position
                                t.data["velocity_rad_s"] = velocity
                            except struct.error:
                                t.status = TestStatus.WARN
                                t.message = "Response received but payload format unexpected"
                        else:
                            t.status = TestStatus.WARN
                            t.message = "No encoder response (may not be implemented)"
                    else:
                        t.status = TestStatus.FAIL
                        t.message = "Failed to send encoder query"
                    self.close_serial()
                else:
                    t.status = TestStatus.FAIL
                    t.message = f"Could not open serial port {self._serial_port}"
            else:
                t.status = TestStatus.SKIP
                t.message = "No serial port specified (use --serial-port)"
        
        # Test 7: Speed ramp calculation (software-only — not hardware verified)
        with self.timed_test("Speed Ramp Profile (Calculated)") as t:
            max_rpm = self._config.max_rpm
            accel = 100.0  # RPM/s from config
            ramp_time = max_rpm / accel
            
            total_steps = self._config.steps_per_rev * self._config.microstepping
            steps_per_sec_max = (max_rpm / 60.0) * total_steps
            min_step_delay_us = 1_000_000 / steps_per_sec_max if steps_per_sec_max > 0 else 0
            
            t.data["max_rpm"] = max_rpm
            t.data["ramp_time_s"] = round(ramp_time, 3)
            t.data["max_steps_per_sec"] = round(steps_per_sec_max, 0)
            t.data["min_step_delay_us"] = round(min_step_delay_us, 1)
            
            t.status = TestStatus.SKIP
            t.message = (f"[Calculation only] 0→{max_rpm}RPM in {ramp_time:.2f}s | "
                         f"{steps_per_sec_max:.0f} steps/s max | "
                         f"{min_step_delay_us:.0f}µs min delay — NOT verified on motor")
        
        # Test 8: Motor command packet (protocol test)
        with self.timed_test("Motor Command Packet (Protocol)") as t:
            motor_byte = 0x01 if self._motor_key == "inner" else 0x02
            if self._motor_key == "nema24":
                motor_byte = 0x03
            
            velocity_rpm = 2.0
            acceleration = 100.0
            payload = struct.pack('<B f f', motor_byte, velocity_rpm, acceleration)
            raw = build_packet(MOTOR_COMMAND, payload, sequence=1)
            parsed = parse_packet(raw)
            
            if parsed and parsed["valid"]:
                t.status = TestStatus.PASS
                t.message = (f"Packet built: {len(raw)} bytes | "
                             f"Motor=0x{motor_byte:02X} RPM={velocity_rpm} Accel={acceleration}")
                t.data["packet_hex"] = raw.hex().upper()
                
                if self.verbose:
                    print(TerminalDisplay.hex_dump(raw, "TX"))
            else:
                t.status = TestStatus.FAIL
                t.message = "Packet build/parse failed (CRC error)"
        
        # Test 9: Emergency stop packet
        with self.timed_test("Emergency Stop Packet (Protocol)") as t:
            estop_payload = struct.pack('<B', 0x00)  # E-stop all
            raw = build_packet(0x21, estop_payload, sequence=99)
            parsed = parse_packet(raw)
            
            if parsed and parsed["valid"]:
                t.status = TestStatus.PASS
                t.message = f"E-Stop packet: {len(raw)} bytes, CRC valid"
                
                if self._serial_port and HAS_SERIAL:
                    # Don't actually send E-stop unless we have a reason
                    t.message += " (not sent — test only)"
            else:
                t.status = TestStatus.FAIL
                t.message = "E-Stop packet build failed"
        
        return self.results


class AllMotorsTest:
    """Run tests for all configured motors."""
    
    def __init__(self, serial_port: Optional[str] = None,
                 dry_run: bool = False, verbose: bool = False):
        self.serial_port = serial_port
        self.dry_run = dry_run
        self.verbose = verbose
    
    async def run_all(self) -> Dict[str, List[TestResult]]:
        display = TerminalDisplay()
        all_results = {}
        
        display.banner("NEMA MOTOR TEST SUITE")
        
        for key in ["inner", "outer", "nema24"]:
            config = MOTOR_CONFIGS[key]
            display.section(f"{config.name} ({config.motor_type})")
            
            test = MotorTest(
                motor_key=key,
                serial_port=self.serial_port,
                dry_run=self.dry_run,
                verbose=self.verbose,
            )
            results = await test.run_tests()
            
            for r in results:
                display.test_result(r)
            
            all_results[key] = results
        
        # Combined summary
        all_flat = [r for results in all_results.values() for r in results]
        display.summary(all_flat)
        
        return all_results


# =============================================================================
# CLI ENTRY
# =============================================================================

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="NEMA Motor Test Suite")
    parser.add_argument("--motor", "-m", choices=["inner", "outer", "nema24", "all"],
                        default="all", help="Motor to test (default: all)")
    parser.add_argument("--serial-port", "-s", type=str, default=None,
                        help="Serial port for encoder feedback")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write to GPIO pins")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show packet hex dumps")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    args = parser.parse_args()
    
    if args.motor == "all":
        suite = AllMotorsTest(
            serial_port=args.serial_port,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        all_results = await suite.run_all()
        
        if args.json:
            import json
            output = {}
            for key, results in all_results.items():
                config = MOTOR_CONFIGS[key]
                output[key] = {
                    "name": config.name,
                    "motor_type": config.motor_type,
                    "results": [
                        {"name": r.name, "status": r.status.value,
                         "message": r.message, "data": r.data}
                        for r in results
                    ],
                }
            print(json.dumps(output, indent=2))
        
        has_failures = any(
            r.status == TestStatus.FAIL
            for results in all_results.values()
            for r in results
        )
        sys.exit(1 if has_failures else 0)
    else:
        test = MotorTest(
            motor_key=args.motor,
            serial_port=args.serial_port,
            dry_run=args.dry_run,
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
