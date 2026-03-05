"""
RPM Digital Twin - Unified Hardware Connection Dashboard
========================================================
All 7 hardware components on a single status display.
Dual mode: terminal-based (colored) AND web-based (GUI).

Components monitored:
  1. RPi Pico W          — USB Serial
  2. RPi 5 (8GB)         — Self (localhost)
  3. NEMA 23 Inner Frame — GPIO
  4. NEMA 23 Outer Frame — GPIO
  5. NEMA 24 Third Axis  — GPIO
  6. Slip Ring Inner      — Signal path
  7. Slip Ring Outer      — Signal path

Features:
- Real-time connection status with colored indicators
- Packet sniffer with hex dump display
- Data transfer rate monitoring
- JSON report export
- Auto-refresh in terminal mode
- Web dashboard endpoint at /api/hardware-status

Usage:
    python -m tests.hardware.run_all                # Run all tests once
    python -m tests.hardware.run_all --watch        # Auto-refresh every 10s
    python -m tests.hardware.run_all --json         # JSON output
    python -m tests.hardware.run_all --verbose      # Hex dumps enabled
    python -m tests.hardware.run_all --report       # Save report to file
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tests.hardware.test_base import (
    ComponentType,
    ComponentStatus,
    ConnectionType,
    TestResult,
    TestStatus,
    PortScanner,
    TerminalDisplay,
    TerminalColors as C,
    GPIOHelper,
)
from tests.hardware.test_pico_w import PicoWTest
from tests.hardware.test_rpi5 import RPi5DiagnosticsTest
from tests.hardware.test_motors import MotorTest, MOTOR_CONFIGS
from tests.hardware.test_slip_rings import SlipRingTest


class UnifiedDashboard:
    """
    Unified hardware status dashboard — all 7 components.
    Provides both terminal display and JSON API output.
    """
    
    def __init__(self, serial_port: Optional[str] = None,
                 pico_port: Optional[str] = None,
                 dry_run: bool = False,
                 verbose: bool = False):
        self.serial_port = serial_port
        self.pico_port = pico_port
        self.dry_run = dry_run
        self.verbose = verbose
        self.display = TerminalDisplay()
        
        # Component status tracking
        self.components: Dict[str, ComponentStatus] = {
            "rpi_pico_w": ComponentStatus(
                component=ComponentType.RPI_PICO_W,
                name="RPi Pico W",
                connection_type=ConnectionType.SERIAL,
            ),
            "rpi_5": ComponentStatus(
                component=ComponentType.RPI_5,
                name="RPi 5 (8GB)",
                connection_type=ConnectionType.USB,
            ),
            "nema_23_inner": ComponentStatus(
                component=ComponentType.NEMA_23_INNER,
                name="NEMA 23 — Inner Frame",
                connection_type=ConnectionType.GPIO,
            ),
            "nema_23_outer": ComponentStatus(
                component=ComponentType.NEMA_23_OUTER,
                name="NEMA 23 — Outer Frame",
                connection_type=ConnectionType.GPIO,
            ),
            "nema_24": ComponentStatus(
                component=ComponentType.NEMA_24,
                name="NEMA 24 — Third Axis",
                connection_type=ConnectionType.GPIO,
            ),
            "slip_ring_inner": ComponentStatus(
                component=ComponentType.SLIP_RING_INNER,
                name="Slip Ring — Inner",
                connection_type=ConnectionType.GPIO,
            ),
            "slip_ring_outer": ComponentStatus(
                component=ComponentType.SLIP_RING_OUTER,
                name="Slip Ring — Outer",
                connection_type=ConnectionType.GPIO,
            ),
        }
        
        self._all_results: Dict[str, List[TestResult]] = {}
        self._run_timestamp: Optional[datetime] = None
    
    async def run_all_tests(self) -> Dict[str, List[TestResult]]:
        """Run all hardware tests and update component status."""
        self._run_timestamp = datetime.now()
        self._all_results = {}
        
        # 1. RPi Pico W
        pico = PicoWTest(port=self.pico_port, verbose=self.verbose)
        pico_results = await pico.run_tests()
        self._all_results["rpi_pico_w"] = pico_results
        self._update_status("rpi_pico_w", pico_results, pico._detected_port or "N/A")
        
        # 2. RPi 5
        rpi5 = RPi5DiagnosticsTest(verbose=self.verbose)
        rpi5_results = await rpi5.run_tests()
        self._all_results["rpi_5"] = rpi5_results
        self._update_status("rpi_5", rpi5_results, "localhost")
        
        # 3-5. Motors
        for motor_key in ["inner", "outer", "nema24"]:
            comp_key = {
                "inner": "nema_23_inner",
                "outer": "nema_23_outer",
                "nema24": "nema_24",
            }[motor_key]
            
            motor = MotorTest(
                motor_key=motor_key,
                serial_port=self.serial_port,
                dry_run=self.dry_run,
                verbose=self.verbose,
            )
            motor_results = await motor.run_tests()
            self._all_results[comp_key] = motor_results
            
            config = MOTOR_CONFIGS[motor_key]
            self._update_status(
                comp_key, motor_results,
                f"GPIO DIR={config.direction_pin} STEP={config.step_pin} EN={config.enable_pin}"
            )
        
        # 6-7. Slip rings
        for ring_id in ["inner", "outer"]:
            comp_key = f"slip_ring_{ring_id}"
            
            slip = SlipRingTest(
                ring_id=ring_id,
                serial_port=self.serial_port,
                verbose=self.verbose,
            )
            slip_results = await slip.run_tests()
            self._all_results[comp_key] = slip_results
            self._update_status(comp_key, slip_results, f"{ring_id} axis")
        
        return self._all_results
    
    def _update_status(self, key: str, results: List[TestResult], port: str) -> None:
        """Update component status from test results. Only mark CONNECTED when hardware is actually verified."""
        comp = self.components[key]
        comp.port = port
        comp.test_results = results
        comp.last_seen = datetime.now()
        
        fails = sum(1 for r in results if r.status == TestStatus.FAIL)
        passes = sum(1 for r in results if r.status == TestStatus.PASS)
        skips = sum(1 for r in results if r.status == TestStatus.SKIP)
        warns = sum(1 for r in results if r.status == TestStatus.WARN)
        total = len(results)
        
        # Only mark CONNECTED if there are actual hardware-verified PASS results
        # If ALL non-skip tests passed AND at least one was a real hardware test
        if fails > 0:
            comp.is_connected = False
            comp.status_msg = f"{fails} tests FAILED"
        elif passes > 0 and skips < total:
            # Some tests passed — but were they real hardware tests?
            # Check if any PASS test doesn't have "Config" or "Calculation" or "Software" in name
            real_hw_passes = sum(
                1 for r in results 
                if r.status == TestStatus.PASS and
                not any(kw in r.name.lower() for kw in ["config", "calculat", "protocol", "software"])
            )
            if real_hw_passes > 0:
                comp.is_connected = True
                comp.status_msg = f"{passes} passed ({real_hw_passes} hardware-verified)"
            else:
                comp.is_connected = False
                comp.status_msg = f"{passes} software-only, {skips} skipped — hardware NOT verified"
        else:
            comp.is_connected = False
            comp.status_msg = "All tests skipped — hardware NOT available"
        
        # Extract latency if available
        for r in results:
            if "latency_ms" in r.data:
                comp.latency_ms = r.data["latency_ms"]
            if "packets_per_sec" in r.data:
                comp.packets_per_sec = r.data["packets_per_sec"]
    
    def display_terminal(self) -> None:
        """Render the full terminal dashboard."""
        self.display.banner("HARDWARE CONNECTION STATUS")
        
        # Connection status table
        self.display.section("Component Status")
        for comp in self.components.values():
            self.display.connection_status(comp)
        
        # Detailed results per component
        for key, results in self._all_results.items():
            comp = self.components[key]
            self.display.section(comp.name)
            for r in results:
                self.display.test_result(r)
        
        # Overall summary
        all_results = [r for results in self._all_results.values() for r in results]
        self.display.summary(all_results)
        
        # Packet transfer summary
        self.display.section("Data Transfer Summary")
        total_tx = sum(r.packets_sent for r in all_results)
        total_rx = sum(r.packets_received for r in all_results)
        total_err = sum(r.packets_failed for r in all_results)
        
        print(f"  {C.CYAN}TX:{C.RESET} {total_tx:6d} packets   "
              f"{C.GREEN}RX:{C.RESET} {total_rx:6d} packets   "
              f"{C.RED}ERR:{C.RESET} {total_err:5d} packets")
        
        if total_tx > 0:
            success_rate = ((total_tx - total_err) / total_tx) * 100
            print(f"  Success rate: {success_rate:.1f}%")
        print()
    
    def to_json(self) -> Dict[str, Any]:
        """Export complete dashboard state as JSON."""
        return {
            "timestamp": self._run_timestamp.isoformat() if self._run_timestamp else None,
            "components": {
                key: {
                    "name": comp.name,
                    "type": comp.component.value,
                    "connection": comp.connection_type.value,
                    "connected": comp.is_connected,
                    "port": comp.port,
                    "status": comp.status_msg,
                    "latency_ms": comp.latency_ms if comp.latency_ms > 0 else None,
                    "packets_per_sec": comp.packets_per_sec if comp.packets_per_sec > 0 else None,
                    "tests": [
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
                        for r in comp.test_results
                    ],
                }
                for key, comp in self.components.items()
            },
            "summary": {
                "total_components": len(self.components),
                "connected": sum(1 for c in self.components.values() if c.is_connected),
                "disconnected": sum(1 for c in self.components.values() if not c.is_connected),
                "total_tests": sum(len(r) for r in self._all_results.values()),
                "passed": sum(
                    1 for results in self._all_results.values()
                    for r in results if r.status == TestStatus.PASS
                ),
                "failed": sum(
                    1 for results in self._all_results.values()
                    for r in results if r.status == TestStatus.FAIL
                ),
                "warnings": sum(
                    1 for results in self._all_results.values()
                    for r in results if r.status == TestStatus.WARN
                ),
            },
        }
    
    def save_report(self, filepath: Optional[str] = None) -> str:
        """Save test report to JSON file."""
        if filepath is None:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = PROJECT_ROOT / "logs"
            report_dir.mkdir(exist_ok=True)
            filepath = str(report_dir / f"hardware_report_{timestamp_str}.json")
        
        with open(filepath, 'w') as f:
            json.dump(self.to_json(), f, indent=2, default=str)
        
        print(f"\n  {C.GREEN}Report saved:{C.RESET} {filepath}")
        return filepath


# =============================================================================
# CLI ENTRY
# =============================================================================

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RPM Digital Twin - Unified Hardware Connection Dashboard"
    )
    parser.add_argument("--serial-port", "-s", type=str, default=None,
                        help="Serial port for motor/slip ring tests")
    parser.add_argument("--pico-port", type=str, default=None,
                        help="Serial port for RPi Pico W (default: auto-detect)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write to GPIO pins")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show packet hex dumps and traces")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON instead of terminal display")
    parser.add_argument("--report", action="store_true",
                        help="Save report to logs/ directory")
    parser.add_argument("--watch", "-w", action="store_true",
                        help="Auto-refresh every 10 seconds")
    parser.add_argument("--interval", type=int, default=10,
                        help="Watch interval in seconds (default: 10)")
    args = parser.parse_args()
    
    dashboard = UnifiedDashboard(
        serial_port=args.serial_port,
        pico_port=args.pico_port,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    
    if args.watch:
        print(f"{C.CYAN}Auto-refresh mode: every {args.interval}s  (Ctrl+C to stop){C.RESET}\n")
        try:
            while True:
                # Clear screen
                print("\033[2J\033[H", end="")
                
                await dashboard.run_all_tests()
                dashboard.display_terminal()
                
                if args.report:
                    dashboard.save_report()
                
                await asyncio.sleep(args.interval)
        except KeyboardInterrupt:
            print(f"\n{C.YELLOW}Stopped.{C.RESET}")
    else:
        await dashboard.run_all_tests()
        
        if args.json:
            print(json.dumps(dashboard.to_json(), indent=2, default=str))
        else:
            dashboard.display_terminal()
        
        if args.report:
            dashboard.save_report()
    
    # Exit code
    total_fails = sum(
        1 for results in dashboard._all_results.values()
        for r in results if r.status == TestStatus.FAIL
    )
    sys.exit(1 if total_fails > 0 else 0)


if __name__ == "__main__":
    asyncio.run(main())
