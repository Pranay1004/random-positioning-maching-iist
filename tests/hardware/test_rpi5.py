"""
RPM Digital Twin - Raspberry Pi 5 Self-Diagnostics
===================================================
System diagnostics for RPi 5 8GB — the RPM brain.

Tests:
1. System identification (RPi 5 8GB verification)
2. CPU info (Cortex-A76 4-core 2.4GHz)
3. Memory (8GB LPDDR4X)
4. Storage health
5. Temperature monitoring
6. GPIO availability
7. Network interfaces (for web dashboard access)
8. Python environment
9. Required packages (FastAPI, uvicorn, numpy, etc.)
10. Motor GPIO pin connectivity (NEMA pins from config)
11. I2C bus scan (IMU sensor detection)
12. SPI bus availability
13. USB devices connected
14. FastAPI server readiness

Usage:
    python -m tests.hardware.test_rpi5
    python -m tests.hardware.test_rpi5 --verbose
    python -m tests.hardware.test_rpi5 --json
"""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tests.hardware.test_base import (
    HardwareTestBase,
    ComponentType,
    TestResult,
    TestStatus,
    TerminalDisplay,
    TerminalColors as C,
    GPIOHelper,
)


def _run_cmd(cmd: str, timeout: float = 10.0) -> Tuple[int, str]:
    """Run a shell command safely and return (returncode, stdout)."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return -1, ""


class RPi5DiagnosticsTest(HardwareTestBase):
    """Self-diagnostics for Raspberry Pi 5 8GB."""
    
    @property
    def component_type(self) -> ComponentType:
        return ComponentType.RPI_5
    
    @property
    def component_name(self) -> str:
        return "Raspberry Pi 5 (8GB)"
    
    def _is_rpi(self) -> bool:
        """Check if running on a Raspberry Pi."""
        try:
            with open("/proc/device-tree/model", "r") as f:
                model = f.read().strip()
            return "raspberry pi" in model.lower()
        except (FileNotFoundError, PermissionError):
            return False
    
    async def run_tests(self) -> List[TestResult]:
        is_rpi = self._is_rpi()
        
        # Test 1: System identification
        with self.timed_test("System Identification") as t:
            uname = platform.uname()
            t.data["system"] = uname.system
            t.data["node"] = uname.node
            t.data["release"] = uname.release
            t.data["machine"] = uname.machine
            
            if is_rpi:
                with open("/proc/device-tree/model", "r") as f:
                    model = f.read().strip('\x00').strip()
                t.data["model"] = model
                
                if "pi 5" in model.lower():
                    t.status = TestStatus.PASS
                    t.message = f"Confirmed: {model}"
                elif "pi" in model.lower():
                    t.status = TestStatus.WARN
                    t.message = f"Raspberry Pi detected but not Pi 5: {model}"
                else:
                    t.status = TestStatus.WARN
                    t.message = f"Unknown model: {model}"
            else:
                t.status = TestStatus.WARN
                t.message = (f"Not running on RPi ({uname.system} {uname.machine}). "
                             f"Remote diagnostics only.")
        
        # Test 2: CPU info
        with self.timed_test("CPU Information") as t:
            if is_rpi:
                rc, cpuinfo = _run_cmd("cat /proc/cpuinfo | grep 'model name' | head -1")
                rc2, cpu_count = _run_cmd("nproc")
                rc3, cpu_freq = _run_cmd("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null")
                
                cpu_name = cpuinfo.split(":")[-1].strip() if rc == 0 else "Unknown"
                cores = cpu_count if rc2 == 0 else "?"
                freq_mhz = int(cpu_freq) // 1000 if rc3 == 0 and cpu_freq.isdigit() else 0
                
                t.data["cpu"] = cpu_name
                t.data["cores"] = cores
                t.data["freq_mhz"] = freq_mhz
                
                t.status = TestStatus.PASS
                t.message = f"{cpu_name} | {cores} cores | {freq_mhz}MHz"
            else:
                t.status = TestStatus.SKIP
                t.message = f"Host CPU: {platform.processor()} | {os.cpu_count()} cores — NOT RPi 5"
                t.data["cpu"] = platform.processor()
                t.data["cores"] = os.cpu_count()
        
        # Test 3: Memory
        with self.timed_test("Memory (Target: 8GB)") as t:
            if is_rpi:
                rc, meminfo = _run_cmd("grep MemTotal /proc/meminfo")
                if rc == 0 and meminfo:
                    mem_kb = int(meminfo.split()[1])
                    mem_gb = mem_kb / (1024 * 1024)
                    t.data["memory_gb"] = round(mem_gb, 2)
                    
                    if mem_gb >= 7.5:
                        t.status = TestStatus.PASS
                        t.message = f"{mem_gb:.1f} GB RAM"
                    elif mem_gb >= 3.5:
                        t.status = TestStatus.WARN
                        t.message = f"{mem_gb:.1f} GB RAM (recommend 8GB for production)"
                    else:
                        t.status = TestStatus.FAIL
                        t.message = f"{mem_gb:.1f} GB RAM (insufficient for RPM Digital Twin)"
                else:
                    t.status = TestStatus.WARN
                    t.message = "Could not read memory info"
            else:
                try:
                    import psutil
                    mem = psutil.virtual_memory()
                    mem_gb = mem.total / (1024**3)
                except ImportError:
                    # Fallback: read from sysctl on macOS or /proc on Linux
                    rc, output = _run_cmd("sysctl -n hw.memsize 2>/dev/null")
                    if rc == 0 and output.isdigit():
                        mem_gb = int(output) / (1024**3)
                    else:
                        mem_gb = 0
                t.status = TestStatus.PASS if mem_gb > 0 else TestStatus.WARN
                t.message = f"{mem_gb:.1f} GB RAM (host machine — NOT RPi 5)" if mem_gb > 0 else "Could not determine memory"
                t.data["memory_gb"] = round(mem_gb, 2)
        
        # Test 4: Storage health
        with self.timed_test("Storage Health") as t:
            usage = shutil.disk_usage("/")
            total_gb = usage.total / (1024**3)
            free_gb = usage.free / (1024**3)
            used_pct = (usage.used / usage.total) * 100
            
            t.data["total_gb"] = round(total_gb, 2)
            t.data["free_gb"] = round(free_gb, 2)
            t.data["used_pct"] = round(used_pct, 1)
            
            if free_gb > 2.0:
                t.status = TestStatus.PASS
            elif free_gb > 0.5:
                t.status = TestStatus.WARN
            else:
                t.status = TestStatus.FAIL
            t.message = f"Total: {total_gb:.1f}GB  Free: {free_gb:.1f}GB  Used: {used_pct:.0f}%"
        
        # Test 5: Temperature
        with self.timed_test("CPU Temperature") as t:
            if is_rpi:
                rc, temp = _run_cmd("cat /sys/class/thermal/thermal_zone0/temp")
                if rc == 0 and temp.isdigit():
                    temp_c = int(temp) / 1000
                    t.data["temp_c"] = temp_c
                    
                    if temp_c < 60:
                        t.status = TestStatus.PASS
                    elif temp_c < 75:
                        t.status = TestStatus.WARN
                    else:
                        t.status = TestStatus.FAIL
                    t.message = f"{temp_c:.1f}°C"
                else:
                    t.status = TestStatus.WARN
                    t.message = "Temperature sensor not readable"
            else:
                t.status = TestStatus.SKIP
                t.message = "Not running on RPi"
        
        # Test 6: GPIO availability
        with self.timed_test("GPIO Interface") as t:
            if GPIOHelper.available():
                t.status = TestStatus.PASS
                t.message = f"GPIO available via {GPIOHelper._lib}"
            elif is_rpi:
                t.status = TestStatus.FAIL
                t.message = "GPIO library not found. Install: pip install lgpio"
            else:
                t.status = TestStatus.SKIP
                t.message = "GPIO not available on this platform"
        
        # Test 7: Network interfaces
        with self.timed_test("Network Interfaces") as t:
            hostname = socket.gethostname()
            
            # Get all IPs
            ips = []
            try:
                # Get the IP that would route to external
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                primary_ip = s.getsockname()[0]
                s.close()
                ips.append(primary_ip)
            except Exception:
                primary_ip = "127.0.0.1"
            
            if is_rpi:
                # Check wlan0 and eth0
                rc_wlan, wlan = _run_cmd("ip -4 addr show wlan0 2>/dev/null | grep inet | awk '{print $2}'")
                rc_eth, eth = _run_cmd("ip -4 addr show eth0 2>/dev/null | grep inet | awk '{print $2}'")
                
                interfaces = []
                if wlan:
                    interfaces.append(f"wlan0: {wlan}")
                if eth:
                    interfaces.append(f"eth0: {eth}")
                
                t.data["interfaces"] = interfaces
                t.data["hostname"] = hostname
                
                if interfaces:
                    t.status = TestStatus.PASS
                    t.message = f"{hostname} | " + " | ".join(interfaces)
                else:
                    t.status = TestStatus.WARN
                    t.message = f"{hostname} | No network interfaces"
            else:
                t.status = TestStatus.PASS
                t.message = f"{hostname} | {primary_ip}"
                t.data["hostname"] = hostname
                t.data["primary_ip"] = primary_ip
        
        # Test 8: Python environment
        with self.timed_test("Python Environment") as t:
            py_version = sys.version
            py_path = sys.executable
            venv = os.environ.get("VIRTUAL_ENV", "None")
            
            t.data["version"] = py_version
            t.data["executable"] = py_path
            t.data["venv"] = venv
            
            major, minor = sys.version_info[:2]
            if major >= 3 and minor >= 11:
                t.status = TestStatus.PASS
            elif major >= 3 and minor >= 9:
                t.status = TestStatus.WARN
            else:
                t.status = TestStatus.FAIL
            t.message = f"Python {major}.{minor} | {py_path}"
        
        # Test 9: Required packages
        with self.timed_test("Required Python Packages") as t:
            required = [
                "fastapi", "uvicorn", "numpy", "pydantic",
                "loguru", "serial", "yaml",
            ]
            missing = []
            found = []
            
            for pkg in required:
                try:
                    __import__(pkg)
                    found.append(pkg)
                except ImportError:
                    missing.append(pkg)
            
            t.data["found"] = found
            t.data["missing"] = missing
            
            if not missing:
                t.status = TestStatus.PASS
                t.message = f"All {len(required)} packages installed"
            else:
                t.status = TestStatus.FAIL
                t.message = f"Missing: {', '.join(missing)}"
        
        # Test 10: Motor GPIO pins
        with self.timed_test("Motor GPIO Pins (from config)") as t:
            # Load config for GPIO pin assignments
            import yaml
            config_path = PROJECT_ROOT / "config" / "main_config.yaml"
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                motors = config.get("hardware", {}).get("motors", {})
                pins_info = []
                
                for frame_name, motor_cfg in motors.items():
                    dir_pin = motor_cfg.get("direction_pin")
                    step_pin = motor_cfg.get("step_pin")
                    enable_pin = motor_cfg.get("enable_pin")
                    pins_info.append(f"{frame_name}: DIR={dir_pin} STEP={step_pin} EN={enable_pin}")
                
                if GPIOHelper.available() and is_rpi:
                    # Actually test the enable pins (safe — just toggle briefly)
                    all_ok = True
                    for frame_name, motor_cfg in motors.items():
                        enable_pin = motor_cfg.get("enable_pin")
                        if enable_pin is not None:
                            ok = GPIOHelper.test_pin_output(enable_pin, duration=0.01)
                            if not ok:
                                all_ok = False
                    
                    t.status = TestStatus.PASS if all_ok else TestStatus.WARN
                    t.message = " | ".join(pins_info)
                else:
                    t.status = TestStatus.SKIP
                    t.message = "Config loaded: " + " | ".join(pins_info) + " — NOT verified (no GPIO)"
                
                t.data["pins"] = pins_info
            else:
                t.status = TestStatus.WARN
                t.message = "Config file not found"
        
        # Test 11: I2C bus (IMU sensor)
        with self.timed_test("I2C Bus Scan (IMU)") as t:
            if is_rpi:
                rc, i2c_devices = _run_cmd(r"i2cdetect -y 1 2>/dev/null | grep -E '^\d' | awk '{for(i=2;i<=NF;i++) if($i!=""--"") print $i}'")
                if rc == 0:
                    devices = [d.strip() for d in i2c_devices.split('\n') if d.strip()]
                    t.data["devices"] = devices
                    
                    # MPU9250 is at 0x68 (config setting)
                    if "68" in devices:
                        t.status = TestStatus.PASS
                        t.message = f"MPU9250 found at 0x68. Total devices: {len(devices)}"
                    elif devices:
                        t.status = TestStatus.WARN
                        t.message = f"I2C devices found: {', '.join('0x'+d for d in devices)} (MPU9250 not at 0x68)"
                    else:
                        t.status = TestStatus.WARN
                        t.message = "No I2C devices detected"
                else:
                    t.status = TestStatus.WARN
                    t.message = "i2cdetect not available"
            else:
                t.status = TestStatus.SKIP
                t.message = "I2C not available on this platform"
        
        # Test 12: SPI bus
        with self.timed_test("SPI Bus") as t:
            if is_rpi:
                spi_exists = Path("/dev/spidev0.0").exists()
                t.status = TestStatus.PASS if spi_exists else TestStatus.WARN
                t.message = "SPI0 available" if spi_exists else "SPI not enabled (raspi-config)"
            else:
                t.status = TestStatus.SKIP
                t.message = "SPI not available on this platform"
        
        # Test 13: Connected USB devices
        with self.timed_test("USB Devices") as t:
            if is_rpi or platform.system() == "Darwin":
                if platform.system() == "Darwin":
                    rc, usb_list = _run_cmd("system_profiler SPUSBDataType 2>/dev/null | grep -E '(Product ID|Vendor ID|Serial Number|:$)' | head -30")
                else:
                    rc, usb_list = _run_cmd("lsusb 2>/dev/null")
                
                if rc == 0 and usb_list:
                    lines = [l.strip() for l in usb_list.split('\n') if l.strip()]
                    t.status = TestStatus.PASS
                    t.message = f"{len(lines)} USB entries detected"
                    t.data["devices"] = lines[:10]
                else:
                    t.status = TestStatus.SKIP
                    t.message = "USB enumeration not available"
            else:
                t.status = TestStatus.SKIP
                t.message = "USB listing not available"
        
        # Test 14: FastAPI server readiness
        with self.timed_test("FastAPI Server Readiness") as t:
            server_py = PROJECT_ROOT / "src" / "webapp" / "server.py"
            index_html = PROJECT_ROOT / "src" / "webapp" / "static" / "index.html"
            
            checks = {
                "server.py": server_py.exists(),
                "index.html": index_html.exists(),
            }
            
            # Check if port 8080 is available
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1)
                result = s.connect_ex(("0.0.0.0", 8080))
                s.close()
                port_in_use = (result == 0)
            except Exception:
                port_in_use = False
            
            checks["port_8080"] = not port_in_use
            
            all_ok = all(checks.values())
            t.data["checks"] = {k: "OK" if v else "FAIL" for k, v in checks.items()}
            
            if all_ok:
                t.status = TestStatus.PASS
                t.message = "All server files present. Port 8080 available."
            elif checks["server.py"] and checks["index.html"]:
                t.status = TestStatus.WARN
                t.message = "Server files OK. Port 8080 in use (server already running?)"
            else:
                t.status = TestStatus.FAIL
                missing = [k for k, v in checks.items() if not v]
                t.message = f"Missing: {', '.join(missing)}"
        
        return self.results


# =============================================================================
# CLI ENTRY
# =============================================================================

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="RPi 5 Self-Diagnostics")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()
    
    test = RPi5DiagnosticsTest(verbose=args.verbose)
    results = await test.execute()
    
    if args.json:
        import json
        print(json.dumps(test.to_json(), indent=2))
    
    has_failures = any(r.status == TestStatus.FAIL for r in results)
    sys.exit(1 if has_failures else 0)


if __name__ == "__main__":
    asyncio.run(main())
