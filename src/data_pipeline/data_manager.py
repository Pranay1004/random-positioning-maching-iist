"""
Data Pipeline - Real-time Data Manager
========================================
Central hub for data flow management in the RPM Digital Twin.

Responsibilities:
- Coordinate data from multiple hardware sources
- Manage real-time data streaming to UI
- Handle data persistence to databases
- Provide unified state access
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Any, Callable, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
from loguru import logger

# Import internal modules (absolute imports for module packages)
# When running as part of src package, use absolute imports from src root
try:
    from hardware_interface import (
        RPMStateSnapshot,
        IMUDataPacket,
        EncoderDataPacket,
        MotorStatusPacket,
        ConnectionStatus,
        MotorState,
    )
    from simulation import RPMSimulator, RPMState
except ImportError:
    # Fallback to relative imports if needed
    from ..hardware_interface import (
        RPMStateSnapshot,
        IMUDataPacket,
        EncoderDataPacket,
        MotorStatusPacket,
        ConnectionStatus,
        MotorState,
    )
    from ..simulation import RPMSimulator, RPMState


@dataclass
class DataStreamConfig:
    """Configuration for data streaming."""
    buffer_size: int = 10000
    publish_rate_hz: float = 100.0
    storage_rate_hz: float = 10.0  # Rate for database writes
    enable_persistence: bool = True


class StateManager:
    """
    Centralized state manager for the entire RPM system.
    
    Provides thread-safe access to the current system state
    and maintains state history for analysis.
    """
    
    def __init__(self, config: Optional[DataStreamConfig] = None):
        """
        Initialize state manager.
        
        Args:
            config: Stream configuration
        """
        self.config = config or DataStreamConfig()
        
        # Current state
        self._current_state = RPMStateSnapshot()
        self._state_lock = threading.RLock()
        
        # State history
        self._history: deque = deque(maxlen=self.config.buffer_size)
        
        # Subscribers for state updates
        self._subscribers: List[Callable[[RPMStateSnapshot], None]] = []
        
        # Statistics
        self._update_count = 0
        self._last_update_time: Optional[datetime] = None
        self._updates_per_second = 0.0
        
        logger.info("StateManager initialized")
    
    @property
    def current_state(self) -> RPMStateSnapshot:
        """Get current system state (thread-safe)."""
        with self._state_lock:
            return self._current_state
    
    def update_state(self, **kwargs) -> None:
        """
        Update current state with new values.
        
        Args:
            **kwargs: State attributes to update
        """
        with self._state_lock:
            # Update timestamp
            self._current_state.timestamp = datetime.now()
            
            # Update provided fields
            for key, value in kwargs.items():
                if hasattr(self._current_state, key):
                    setattr(self._current_state, key, value)
            
            # Add to history
            self._history.append(self._current_state.model_copy())
            
            # Update statistics
            self._update_count += 1
            now = datetime.now()
            if self._last_update_time:
                dt = (now - self._last_update_time).total_seconds()
                if dt > 0:
                    # Exponential moving average
                    alpha = 0.1
                    instant_rate = 1.0 / dt
                    self._updates_per_second = alpha * instant_rate + (1 - alpha) * self._updates_per_second
            self._last_update_time = now
        
        # Notify subscribers (outside lock)
        self._notify_subscribers()
    
    def update_from_imu(self, packet: IMUDataPacket) -> None:
        """
        Update state from IMU data packet.
        
        Args:
            packet: IMU sensor data
        """
        self.update_state(
            acceleration_x=packet.accel_x,
            acceleration_y=packet.accel_y,
            acceleration_z=packet.accel_z,
            instantaneous_g=packet.acceleration_magnitude / 9.80665
        )
    
    def update_from_encoder(self, packet: EncoderDataPacket, frame: str) -> None:
        """
        Update state from encoder data.
        
        Args:
            packet: Encoder data packet
            frame: "inner" or "outer"
        """
        if frame == "inner":
            self.update_state(
                inner_frame_position_rad=packet.position_rad,
                inner_frame_velocity_rad_s=packet.velocity_rad_s
            )
        else:
            self.update_state(
                outer_frame_position_rad=packet.position_rad,
                outer_frame_velocity_rad_s=packet.velocity_rad_s
            )
    
    def update_from_motor_status(self, packet: MotorStatusPacket, frame: str) -> None:
        """
        Update state from motor status.
        
        Args:
            packet: Motor status packet
            frame: "inner" or "outer"
        """
        if frame == "inner":
            self.update_state(inner_motor_state=packet.state)
        else:
            self.update_state(outer_motor_state=packet.state)
    
    def update_from_simulation(self, sim_state: RPMState, instantaneous_g: float, time_averaged_g: float) -> None:
        """
        Update state from simulation.
        
        Args:
            sim_state: Simulator state
            instantaneous_g: Current effective gravity
            time_averaged_g: Time-averaged gravity
        """
        self.update_state(
            inner_frame_position_rad=sim_state.theta_inner,
            inner_frame_velocity_rad_s=sim_state.omega_inner,
            outer_frame_position_rad=sim_state.theta_outer,
            outer_frame_velocity_rad_s=sim_state.omega_outer,
            inner_frame_setpoint_rpm=sim_state.omega_inner_setpoint * 60 / (2 * np.pi),
            outer_frame_setpoint_rpm=sim_state.omega_outer_setpoint * 60 / (2 * np.pi),
            instantaneous_g=instantaneous_g,
            time_averaged_g=time_averaged_g
        )
    
    def subscribe(self, callback: Callable[[RPMStateSnapshot], None]) -> None:
        """
        Subscribe to state updates.
        
        Args:
            callback: Function to call on state update
        """
        self._subscribers.append(callback)
        logger.debug(f"Added state subscriber, total: {len(self._subscribers)}")
    
    def unsubscribe(self, callback: Callable) -> None:
        """Remove a subscriber."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)
    
    def _notify_subscribers(self) -> None:
        """Notify all subscribers of state update."""
        state = self.current_state
        for callback in self._subscribers:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"Subscriber callback error: {e}")
    
    def get_history(
        self,
        duration_s: Optional[float] = None,
        max_samples: Optional[int] = None
    ) -> List[RPMStateSnapshot]:
        """
        Get state history.
        
        Args:
            duration_s: Maximum age of samples
            max_samples: Maximum number of samples
            
        Returns:
            List of historical states
        """
        with self._state_lock:
            history = list(self._history)
        
        if duration_s:
            cutoff = datetime.now().timestamp() - duration_s
            history = [s for s in history if s.timestamp.timestamp() > cutoff]
        
        if max_samples:
            history = history[-max_samples:]
        
        return history
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get state manager statistics."""
        return {
            "update_count": self._update_count,
            "updates_per_second": round(self._updates_per_second, 1),
            "history_size": len(self._history),
            "subscriber_count": len(self._subscribers),
            "last_update": self._last_update_time.isoformat() if self._last_update_time else None
        }


class DataPipeline:
    """
    Main data pipeline coordinator.
    
    Manages data flow from hardware -> processing -> storage -> UI.
    """
    
    def __init__(
        self,
        state_manager: Optional[StateManager] = None,
        config: Optional[DataStreamConfig] = None
    ):
        """
        Initialize data pipeline.
        
        Args:
            state_manager: Shared state manager
            config: Pipeline configuration
        """
        self.config = config or DataStreamConfig()
        self.state_manager = state_manager or StateManager(config)
        
        # Processing queues
        self._raw_data_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self._processed_data_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        
        # Task management
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Database clients (to be initialized)
        self._influx_client = None
        self._postgres_client = None
        
        # Statistics
        self._packets_processed = 0
        self._processing_latency_ms = 0.0
        
        logger.info("DataPipeline initialized")
    
    async def start(self) -> None:
        """Start the data pipeline."""
        if self._running:
            logger.warning("Pipeline already running")
            return
        
        self._running = True
        
        # Start processing tasks
        self._tasks.append(asyncio.create_task(self._processing_loop()))
        self._tasks.append(asyncio.create_task(self._persistence_loop()))
        
        logger.info("DataPipeline started")
    
    async def stop(self) -> None:
        """Stop the data pipeline."""
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        
        # Shutdown executor
        self._executor.shutdown(wait=False)
        
        logger.info("DataPipeline stopped")
    
    async def push_raw_data(self, data: Any) -> None:
        """
        Push raw data into the pipeline.
        
        Args:
            data: Raw sensor data packet
        """
        try:
            await asyncio.wait_for(
                self._raw_data_queue.put(data),
                timeout=0.1
            )
        except asyncio.TimeoutError:
            logger.warning("Raw data queue full, dropping packet")
    
    async def _processing_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            try:
                # Get raw data
                data = await asyncio.wait_for(
                    self._raw_data_queue.get(),
                    timeout=0.1
                )
                
                start_time = time.perf_counter()
                
                # Process based on data type
                processed = await self._process_data(data)
                
                # Update state
                if processed:
                    await self._processed_data_queue.put(processed)
                
                # Update statistics
                self._packets_processed += 1
                latency = (time.perf_counter() - start_time) * 1000
                self._processing_latency_ms = 0.1 * latency + 0.9 * self._processing_latency_ms
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Processing error: {e}")
    
    async def _process_data(self, data: Any) -> Optional[Dict[str, Any]]:
        """
        Process a single data packet.
        
        Args:
            data: Raw data packet
            
        Returns:
            Processed data dictionary
        """
        if isinstance(data, IMUDataPacket):
            self.state_manager.update_from_imu(data)
            return {
                "type": "imu",
                "timestamp": data.timestamp.isoformat(),
                "accel": [data.accel_x, data.accel_y, data.accel_z],
                "gyro": [data.gyro_x, data.gyro_y, data.gyro_z],
                "g_magnitude": data.acceleration_magnitude / 9.80665
            }
            
        elif isinstance(data, EncoderDataPacket):
            frame = "inner" if "inner" in data.device_id else "outer"
            self.state_manager.update_from_encoder(data, frame)
            return {
                "type": "encoder",
                "frame": frame,
                "timestamp": data.timestamp.isoformat(),
                "position_rad": data.position_rad,
                "velocity_rpm": data.velocity_rpm
            }
            
        elif isinstance(data, MotorStatusPacket):
            frame = "inner" if "inner" in data.device_id else "outer"
            self.state_manager.update_from_motor_status(data, frame)
            return {
                "type": "motor_status",
                "frame": frame,
                "timestamp": data.timestamp.isoformat(),
                "state": data.state.value,
                "velocity_rpm": data.current_velocity_rpm,
                "current_a": data.current_draw_a,
                "temp_c": data.temperature_c
            }
        
        return None
    
    async def _persistence_loop(self) -> None:
        """Database persistence loop."""
        if not self.config.enable_persistence:
            return
        
        interval = 1.0 / self.config.storage_rate_hz
        
        while self._running:
            try:
                # Collect data for batch write
                batch = []
                while not self._processed_data_queue.empty() and len(batch) < 100:
                    try:
                        data = self._processed_data_queue.get_nowait()
                        batch.append(data)
                    except asyncio.QueueEmpty:
                        break
                
                if batch:
                    await self._write_to_database(batch)
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Persistence error: {e}")
    
    async def _write_to_database(self, batch: List[Dict]) -> None:
        """
        Write batch of processed data to database.
        
        Args:
            batch: List of processed data dictionaries
        """
        # TODO: Implement actual database writes
        # For now, just log
        logger.trace(f"Would write {len(batch)} records to database")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "running": self._running,
            "packets_processed": self._packets_processed,
            "processing_latency_ms": round(self._processing_latency_ms, 2),
            "raw_queue_size": self._raw_data_queue.qsize(),
            "processed_queue_size": self._processed_data_queue.qsize(),
            **self.state_manager.get_statistics()
        }


class ExperimentRecorder:
    """
    Records experiment data with metadata.
    
    Provides structured experiment recording with start/stop,
    metadata capture, and data export.
    """
    
    def __init__(self, state_manager: StateManager):
        """
        Initialize recorder.
        
        Args:
            state_manager: Shared state manager
        """
        self.state_manager = state_manager
        
        self._recording = False
        self._experiment_id: Optional[str] = None
        self._start_time: Optional[datetime] = None
        self._recorded_states: List[RPMStateSnapshot] = []
        self._metadata: Dict[str, Any] = {}
        
        logger.info("ExperimentRecorder initialized")
    
    @property
    def is_recording(self) -> bool:
        """Check if recording is active."""
        return self._recording
    
    def start_recording(
        self,
        experiment_name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start recording an experiment.
        
        Args:
            experiment_name: Name of the experiment
            description: Optional description
            metadata: Additional metadata
            
        Returns:
            Experiment ID
        """
        if self._recording:
            raise RuntimeError("Already recording")
        
        self._experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._start_time = datetime.now()
        self._recorded_states = []
        self._metadata = {
            "id": self._experiment_id,
            "name": experiment_name,
            "description": description,
            "start_time": self._start_time.isoformat(),
            **(metadata or {})
        }
        
        # Subscribe to state updates
        self.state_manager.subscribe(self._on_state_update)
        self._recording = True
        
        logger.info(f"Started recording experiment: {experiment_name}")
        return self._experiment_id
    
    def stop_recording(self) -> Dict[str, Any]:
        """
        Stop recording and return summary.
        
        Returns:
            Experiment summary dictionary
        """
        if not self._recording:
            raise RuntimeError("Not recording")
        
        self._recording = False
        self.state_manager.unsubscribe(self._on_state_update)
        
        end_time = datetime.now()
        duration = (end_time - self._start_time).total_seconds()
        
        # Compute summary statistics
        g_values = [s.instantaneous_g for s in self._recorded_states if s.instantaneous_g is not None]
        
        summary = {
            **self._metadata,
            "end_time": end_time.isoformat(),
            "duration_s": duration,
            "sample_count": len(self._recorded_states),
            "sample_rate_hz": len(self._recorded_states) / duration if duration > 0 else 0,
            "mean_g": np.mean(g_values) if g_values else None,
            "std_g": np.std(g_values) if g_values else None,
            "min_g": np.min(g_values) if g_values else None,
            "max_g": np.max(g_values) if g_values else None,
        }
        
        logger.info(f"Stopped recording. Duration: {duration:.1f}s, Samples: {len(self._recorded_states)}")
        return summary
    
    def _on_state_update(self, state: RPMStateSnapshot) -> None:
        """Callback for state updates."""
        if self._recording:
            self._recorded_states.append(state.model_copy())
    
    def export_to_csv(self, filepath: str) -> None:
        """
        Export recorded data to CSV.
        
        Args:
            filepath: Output file path
        """
        import csv
        
        if not self._recorded_states:
            logger.warning("No data to export")
            return
        
        fieldnames = [
            "timestamp",
            "inner_position_rad",
            "inner_velocity_rad_s",
            "outer_position_rad",
            "outer_velocity_rad_s",
            "accel_x",
            "accel_y",
            "accel_z",
            "instantaneous_g",
            "time_averaged_g"
        ]
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for state in self._recorded_states:
                writer.writerow({
                    "timestamp": state.timestamp.isoformat(),
                    "inner_position_rad": state.inner_frame_position_rad,
                    "inner_velocity_rad_s": state.inner_frame_velocity_rad_s,
                    "outer_position_rad": state.outer_frame_position_rad,
                    "outer_velocity_rad_s": state.outer_frame_velocity_rad_s,
                    "accel_x": state.acceleration_x,
                    "accel_y": state.acceleration_y,
                    "accel_z": state.acceleration_z,
                    "instantaneous_g": state.instantaneous_g,
                    "time_averaged_g": state.time_averaged_g,
                })
        
        logger.info(f"Exported {len(self._recorded_states)} samples to {filepath}")
    
    def export_to_json(self, filepath: str) -> None:
        """
        Export recorded data to JSON.
        
        Args:
            filepath: Output file path
        """
        data = {
            "metadata": self._metadata,
            "data": [s.to_dict() for s in self._recorded_states]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Exported experiment to {filepath}")
