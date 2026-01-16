"""
RPM Digital Twin - Main Application Entry Point
================================================
Commercial-grade Random Positioning Machine control and simulation software.

This is the main entry point for the Python backend of the RPM Digital Twin.
It initializes all subsystems and provides a unified interface for:
- Hardware communication (Arduino, RPi, sensors)
- Real-time simulation
- Data acquisition and storage
- Visualization dashboards

Target Customers: ISRO, NASA, AXIOM SPACE, IITs, Space Research Institutions

Copyright (c) 2024 RPM Digital Twin Team
"""

from __future__ import annotations

import asyncio
import signal
import sys
from pathlib import Path
from typing import Optional
import argparse

import numpy as np
import yaml
from loguru import logger

# Add src to path for imports
SRC_DIR = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

# Import subsystems
from hardware_interface import SerialManager, RPMStateSnapshot
from simulation import RPMSimulator, RPMState, RPMGeometry, OperationMode
from data_pipeline import StateManager, DataPipeline, ExperimentRecorder
from visualization import RPMDashboard


class RPMApplication:
    """
    Main application class for RPM Digital Twin.
    
    Orchestrates all subsystems and provides unified control interface.
    """
    
    VERSION = "1.0.0"
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize RPM application.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = config_path or PROJECT_ROOT / "config" / "main_config.yaml"
        self.config = self._load_config()
        
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        # Subsystem instances (initialized in start())
        self.state_manager: Optional[StateManager] = None
        self.data_pipeline: Optional[DataPipeline] = None
        self.serial_manager: Optional[SerialManager] = None
        self.simulator: Optional[RPMSimulator] = None
        self.experiment_recorder: Optional[ExperimentRecorder] = None
        self.dashboard: Optional[RPMDashboard] = None
        
        # Mode flags
        self._simulation_mode = False
        self._hardware_mode = False
        
        logger.info(f"RPM Digital Twin v{self.VERSION} initialized")
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
                return config
        else:
            logger.warning(f"Config file not found: {self.config_path}")
            return {}
    
    async def start(
        self,
        mode: str = "simulation",
        enable_dashboard: bool = True,
        enable_recording: bool = False
    ) -> None:
        """
        Start the RPM Digital Twin application.
        
        Args:
            mode: Operation mode - "simulation", "hardware", or "hybrid"
            enable_dashboard: Whether to launch visualization dashboard
            enable_recording: Whether to enable experiment recording
        """
        logger.info(f"Starting RPM Digital Twin in {mode} mode")
        
        # Initialize state manager
        self.state_manager = StateManager()
        
        # Initialize data pipeline
        self.data_pipeline = DataPipeline(self.state_manager)
        await self.data_pipeline.start()
        
        # Initialize based on mode
        if mode in ["simulation", "hybrid"]:
            self._simulation_mode = True
            self._init_simulator()
        
        if mode in ["hardware", "hybrid"]:
            self._hardware_mode = True
            await self._init_hardware()
        
        # Initialize experiment recorder if requested
        if enable_recording:
            self.experiment_recorder = ExperimentRecorder(self.state_manager)
        
        # Initialize dashboard if requested
        if enable_dashboard:
            self.dashboard = RPMDashboard(self.state_manager)
        
        self._running = True
        logger.info("RPM Digital Twin started successfully")
        
        # Print status
        self._print_status()
    
    def _init_simulator(self) -> None:
        """Initialize the simulation subsystem."""
        sim_config = self.config.get("simulation", {})
        
        # Create geometry from config or defaults
        geometry = RPMGeometry(
            inner_frame_radius=sim_config.get("inner_frame_radius", 0.15),
            outer_frame_radius=sim_config.get("outer_frame_radius", 0.25),
            sample_mount_offset=np.array(sim_config.get("sample_position", [0, 0, 0.1]))
        )
        
        # Get operation mode
        mode_str = sim_config.get("default_mode", "clinostat_3d")
        mode_map = {
            "clinostat_3d": OperationMode.CLINOSTAT_3D,
            "random": OperationMode.RANDOM,
            "partial_gravity": OperationMode.PARTIAL_GRAVITY
        }
        mode = mode_map.get(mode_str, OperationMode.CLINOSTAT_3D)
        
        self.simulator = RPMSimulator(geometry=geometry, mode=mode)
        
        logger.info(f"Simulator initialized in {mode.value} mode")
    
    async def _init_hardware(self) -> None:
        """Initialize hardware communication."""
        hw_config = self.config.get("hardware", {})
        arduino_config = hw_config.get("arduino", {})
        
        port = arduino_config.get("port", "/dev/tty.usbmodem*")
        baudrate = arduino_config.get("baudrate", 115200)
        
        self.serial_manager = SerialManager(port=port, baudrate=baudrate)
        
        # Register callbacks
        self.serial_manager.register_callback(0x01, self._on_imu_data)
        self.serial_manager.register_callback(0x02, self._on_encoder_data)
        self.serial_manager.register_callback(0x03, self._on_motor_status)
        
        # Try to connect
        connected = await self.serial_manager.connect()
        if connected:
            logger.info(f"Connected to Arduino on {port}")
        else:
            logger.warning("Failed to connect to Arduino, running in simulation-only mode")
            self._hardware_mode = False
    
    def _on_imu_data(self, data: bytes) -> None:
        """Handle incoming IMU data from Arduino."""
        # Parse and push to data pipeline
        # This will be processed and update state
        pass
    
    def _on_encoder_data(self, data: bytes) -> None:
        """Handle incoming encoder data from Arduino."""
        pass
    
    def _on_motor_status(self, data: bytes) -> None:
        """Handle incoming motor status from Arduino."""
        pass
    
    async def run_simulation(
        self,
        duration_s: float,
        inner_rpm: float,
        outer_rpm: float,
        dt: float = 0.01
    ) -> dict:
        """
        Run a simulation and return results.
        
        Args:
            duration_s: Simulation duration in seconds
            inner_rpm: Inner frame rotation speed in RPM
            outer_rpm: Outer frame rotation speed in RPM
            dt: Time step in seconds
            
        Returns:
            Dictionary with simulation results
        """
        if not self._simulation_mode:
            raise RuntimeError("Simulation mode not enabled")
        
        import numpy as np
        
        # Convert RPM to rad/s
        omega_inner = inner_rpm * np.pi / 30
        omega_outer = outer_rpm * np.pi / 30
        
        # Set simulator velocities
        self.simulator.set_velocities(omega_inner, omega_outer)
        
        # Run simulation
        num_steps = int(duration_s / dt)
        g_values = []
        
        logger.info(f"Running simulation: {duration_s}s at {inner_rpm}/{outer_rpm} RPM")
        
        for i in range(num_steps):
            state = self.simulator.step(dt)
            g = self.simulator.gravity_calc.compute_gravity_in_sample_frame(state.theta_inner, state.theta_outer)
            g_values.append(np.linalg.norm(g) / 9.80665)
            
            # Update state manager
            if self.state_manager:
                self.state_manager.update_from_simulation(
                    state, g_values[-1], np.mean(g_values)
                )
        
        # Compute statistics
        g_array = np.array(g_values)
        results = {
            "duration_s": duration_s,
            "inner_rpm": inner_rpm,
            "outer_rpm": outer_rpm,
            "mean_g": float(np.mean(g_array)),
            "std_g": float(np.std(g_array)),
            "max_g": float(np.max(g_array)),
            "min_g": float(np.min(g_array)),
            "samples": len(g_values)
        }
        
        logger.info(f"Simulation complete. Mean g: {results['mean_g']:.5f}")
        return results
    
    async def set_motor_speed(self, motor: str, rpm: float) -> bool:
        """
        Set motor speed (hardware mode).
        
        Args:
            motor: "inner" or "outer"
            rpm: Target speed in RPM
            
        Returns:
            True if command sent successfully
        """
        if not self._hardware_mode or not self.serial_manager:
            logger.warning("Hardware mode not available")
            return False
        
        import struct
        
        motor_id = 0 if motor == "inner" else 1
        acceleration = 100.0  # Default acceleration
        
        payload = struct.pack('<Bff', motor_id, rpm, acceleration)
        return await self.serial_manager.send_packet(0x20, payload)
    
    async def emergency_stop(self) -> None:
        """Emergency stop all motors."""
        logger.warning("EMERGENCY STOP triggered")
        
        if self.serial_manager and self._hardware_mode:
            # Send emergency stop command
            await self.serial_manager.send_packet(0x21, bytes([0x00]))
        
        if self.simulator and self._simulation_mode:
            self.simulator.emergency_stop()
    
    def start_recording(
        self,
        name: str,
        description: Optional[str] = None
    ) -> str:
        """
        Start recording an experiment.
        
        Args:
            name: Experiment name
            description: Optional description
            
        Returns:
            Experiment ID
        """
        if not self.experiment_recorder:
            raise RuntimeError("Recording not enabled")
        
        return self.experiment_recorder.start_recording(name, description)
    
    def stop_recording(self) -> dict:
        """Stop recording and return summary."""
        if not self.experiment_recorder:
            raise RuntimeError("Recording not enabled")
        
        return self.experiment_recorder.stop_recording()
    
    def export_data(self, filepath: str, format: str = "csv") -> None:
        """
        Export recorded data.
        
        Args:
            filepath: Output file path
            format: "csv" or "json"
        """
        if not self.experiment_recorder:
            raise RuntimeError("Recording not enabled")
        
        if format == "csv":
            self.experiment_recorder.export_to_csv(filepath)
        elif format == "json":
            self.experiment_recorder.export_to_json(filepath)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def launch_dashboard(self) -> None:
        """Launch the visualization dashboard."""
        if self.dashboard:
            self.dashboard.show()
        else:
            logger.warning("Dashboard not initialized")
    
    def _print_status(self) -> None:
        """Print current application status."""
        print("\n" + "=" * 60)
        print(f"  RPM Digital Twin v{self.VERSION}")
        print("=" * 60)
        print(f"  Simulation Mode: {'Enabled' if self._simulation_mode else 'Disabled'}")
        print(f"  Hardware Mode:   {'Enabled' if self._hardware_mode else 'Disabled'}")
        print(f"  Recording:       {'Enabled' if self.experiment_recorder else 'Disabled'}")
        print(f"  Dashboard:       {'Enabled' if self.dashboard else 'Disabled'}")
        print("=" * 60 + "\n")
    
    async def stop(self) -> None:
        """Stop the application and cleanup."""
        logger.info("Shutting down RPM Digital Twin...")
        
        self._running = False
        
        # Stop subsystems
        if self.data_pipeline:
            await self.data_pipeline.stop()
        
        if self.serial_manager:
            await self.serial_manager.disconnect()
        
        logger.info("RPM Digital Twin stopped")
    
    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()
    
    def request_shutdown(self) -> None:
        """Request application shutdown."""
        self._shutdown_event.set()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    logger.remove()  # Remove default handler
    
    level = "DEBUG" if verbose else "INFO"
    
    # Console handler with custom format
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True
    )
    
    # File handler for debug logs
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / "rpm_digital_twin_{time}.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG"
    )


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RPM Digital Twin - Commercial-grade Random Positioning Machine Software"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["simulation", "hardware", "hybrid"],
        default="simulation",
        help="Operation mode (default: simulation)"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        default=True,
        help="Enable visualization dashboard"
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_false",
        dest="dashboard",
        help="Disable visualization dashboard"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--quick-sim",
        action="store_true",
        help="Run a quick simulation and exit"
    )
    parser.add_argument(
        "--inner-rpm",
        type=float,
        default=2.0,
        help="Inner frame RPM for quick simulation"
    )
    parser.add_argument(
        "--outer-rpm",
        type=float,
        default=2.0,
        help="Outer frame RPM for quick simulation"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Simulation duration in seconds"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Create application
    app = RPMApplication(config_path=args.config)
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        app.request_shutdown()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start application
        await app.start(
            mode=args.mode,
            enable_dashboard=args.dashboard and not args.quick_sim
        )
        
        if args.quick_sim:
            # Run quick simulation
            results = await app.run_simulation(
                duration_s=args.duration,
                inner_rpm=args.inner_rpm,
                outer_rpm=args.outer_rpm
            )
            print("\n=== Simulation Results ===")
            for key, value in results.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value}")
        else:
            # Run until shutdown requested
            print("RPM Digital Twin is running. Press Ctrl+C to stop.")
            
            if args.dashboard:
                app.launch_dashboard()
            
            await app.wait_for_shutdown()
    
    finally:
        await app.stop()


if __name__ == "__main__":
    asyncio.run(main())
