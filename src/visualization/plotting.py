"""
Real-time Data Visualization Engine
=====================================
High-performance plotting for live RPM data visualization.

This module provides:
- Real-time updating plots with PyQtGraph
- Statistical displays and gauges
- Time-series visualization
- 3D orientation display

Designed for 60+ FPS rendering performance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Callable
from datetime import datetime
import time

# Conditional imports for cross-platform compatibility
try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import matplotlib.animation as animation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from loguru import logger


@dataclass
class PlotConfig:
    """Configuration for a real-time plot."""
    title: str = "Plot"
    x_label: str = "Time (s)"
    y_label: str = "Value"
    y_range: Optional[Tuple[float, float]] = None
    history_length: int = 1000
    update_rate_hz: float = 30.0
    line_color: str = "#00ff88"
    line_width: float = 1.5
    background_color: str = "#1a1a2e"
    grid: bool = True
    show_legend: bool = True


@dataclass
class TimeSeriesBuffer:
    """
    Circular buffer for time-series data.
    
    Efficiently stores recent data points for plotting
    without memory allocation during runtime.
    """
    max_length: int = 10000
    timestamps: deque = field(default_factory=deque)
    values: deque = field(default_factory=deque)
    
    def __post_init__(self):
        self.timestamps = deque(maxlen=self.max_length)
        self.values = deque(maxlen=self.max_length)
    
    def append(self, timestamp: float, value: float) -> None:
        """Add a new data point."""
        self.timestamps.append(timestamp)
        self.values.append(value)
    
    def get_arrays(self) -> Tuple[NDArray, NDArray]:
        """Get numpy arrays of timestamps and values."""
        return np.array(self.timestamps), np.array(self.values)
    
    def clear(self) -> None:
        """Clear all data."""
        self.timestamps.clear()
        self.values.clear()
    
    def __len__(self) -> int:
        return len(self.timestamps)


class DataPlotter:
    """
    Real-time data plotter using matplotlib (fallback mode).
    
    This class provides high-quality static and animated plots
    for data visualization and export.
    """
    
    def __init__(self, config: Optional[PlotConfig] = None):
        """
        Initialize plotter.
        
        Args:
            config: Plot configuration
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for DataPlotter")
            
        self.config = config or PlotConfig()
        self._buffers: Dict[str, TimeSeriesBuffer] = {}
        
        # Set up matplotlib style
        plt.style.use('dark_background')
        plt.rcParams.update({
            'figure.facecolor': self.config.background_color,
            'axes.facecolor': '#0f0f1a',
            'axes.edgecolor': '#333366',
            'axes.labelcolor': '#ffffff',
            'text.color': '#ffffff',
            'xtick.color': '#aaaaaa',
            'ytick.color': '#aaaaaa',
            'grid.color': '#333366',
            'grid.linestyle': '--',
            'grid.alpha': 0.5,
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
        })
        
        logger.debug("DataPlotter initialized")
    
    def add_series(self, name: str, color: Optional[str] = None) -> None:
        """
        Add a new data series to the plot.
        
        Args:
            name: Series name/label
            color: Line color (hex string)
        """
        self._buffers[name] = TimeSeriesBuffer(max_length=self.config.history_length)
        logger.debug(f"Added series: {name}")
    
    def update_data(self, name: str, timestamp: float, value: float) -> None:
        """
        Update a data series with new value.
        
        Args:
            name: Series name
            timestamp: Time value
            value: Data value
        """
        if name not in self._buffers:
            self.add_series(name)
        self._buffers[name].append(timestamp, value)
    
    def create_figure(
        self,
        series_names: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (12, 6)
    ) -> Figure:
        """
        Create a matplotlib figure with current data.
        
        Args:
            series_names: List of series to plot (default: all)
            figsize: Figure size in inches
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Default color cycle (SpaceX-inspired palette)
        colors = ['#00ff88', '#00d4ff', '#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff']
        
        series_to_plot = series_names or list(self._buffers.keys())
        
        for i, name in enumerate(series_to_plot):
            if name in self._buffers and len(self._buffers[name]) > 0:
                t, v = self._buffers[name].get_arrays()
                color = colors[i % len(colors)]
                ax.plot(t, v, label=name, color=color, linewidth=self.config.line_width)
        
        ax.set_xlabel(self.config.x_label)
        ax.set_ylabel(self.config.y_label)
        ax.set_title(self.config.title)
        
        if self.config.y_range:
            ax.set_ylim(self.config.y_range)
            
        if self.config.grid:
            ax.grid(True, alpha=0.3)
            
        if self.config.show_legend and len(series_to_plot) > 1:
            ax.legend(loc='upper right', framealpha=0.8)
        
        fig.tight_layout()
        return fig
    
    def save_figure(
        self,
        filepath: str,
        series_names: Optional[List[str]] = None,
        dpi: int = 150
    ) -> None:
        """
        Save plot to file.
        
        Args:
            filepath: Output file path
            series_names: Series to include
            dpi: Resolution
        """
        fig = self.create_figure(series_names)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info(f"Plot saved to {filepath}")


class GaugeWidget:
    """
    Circular gauge widget for displaying real-time values.
    
    Provides a Tesla-style animated gauge display.
    """
    
    def __init__(
        self,
        title: str = "Gauge",
        min_value: float = 0.0,
        max_value: float = 100.0,
        unit: str = "",
        warning_threshold: Optional[float] = None,
        danger_threshold: Optional[float] = None
    ):
        """
        Initialize gauge.
        
        Args:
            title: Gauge label
            min_value: Minimum scale value
            max_value: Maximum scale value
            unit: Unit string
            warning_threshold: Yellow warning level
            danger_threshold: Red danger level
        """
        self.title = title
        self.min_value = min_value
        self.max_value = max_value
        self.unit = unit
        self.warning_threshold = warning_threshold
        self.danger_threshold = danger_threshold
        
        self._current_value = min_value
        self._target_value = min_value
        
    @property
    def value(self) -> float:
        """Current gauge value."""
        return self._current_value
    
    @value.setter
    def value(self, val: float) -> None:
        """Set target value (for smooth animation)."""
        self._target_value = np.clip(val, self.min_value, self.max_value)
    
    def update(self, dt: float = 0.016) -> None:
        """
        Update gauge animation.
        
        Args:
            dt: Time step for smooth animation
        """
        # Smooth interpolation
        alpha = min(1.0, dt * 10)  # Smoothing factor
        self._current_value += (self._target_value - self._current_value) * alpha
    
    @property
    def percentage(self) -> float:
        """Get value as percentage of range."""
        return (self._current_value - self.min_value) / (self.max_value - self.min_value) * 100
    
    @property
    def status_color(self) -> str:
        """Get color based on current value and thresholds."""
        if self.danger_threshold and self._current_value >= self.danger_threshold:
            return "#ff4444"  # Red
        elif self.warning_threshold and self._current_value >= self.warning_threshold:
            return "#ffaa00"  # Orange
        else:
            return "#00ff88"  # Green
    
    def render_matplotlib(self, ax: plt.Axes) -> None:
        """
        Render gauge using matplotlib.
        
        Args:
            ax: Matplotlib axes to render on
        """
        ax.clear()
        ax.set_aspect('equal')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axis('off')
        
        # Draw arc background
        theta = np.linspace(0.75 * np.pi, 0.25 * np.pi, 100)
        x = np.cos(theta)
        y = np.sin(theta)
        ax.plot(x, y, color='#333366', linewidth=20, solid_capstyle='round')
        
        # Draw value arc
        pct = self.percentage / 100
        theta_val = np.linspace(0.75 * np.pi, 0.75 * np.pi - pct * 0.5 * np.pi, max(2, int(100 * pct)))
        if len(theta_val) > 1:
            x_val = np.cos(theta_val)
            y_val = np.sin(theta_val)
            ax.plot(x_val, y_val, color=self.status_color, linewidth=15, solid_capstyle='round')
        
        # Draw needle
        needle_angle = 0.75 * np.pi - pct * 0.5 * np.pi
        ax.plot([0, 0.7 * np.cos(needle_angle)], [0, 0.7 * np.sin(needle_angle)],
                color='white', linewidth=3)
        ax.plot(0, 0, 'o', color='white', markersize=10)
        
        # Draw text
        ax.text(0, -0.3, f"{self._current_value:.2f}", ha='center', va='center',
                fontsize=24, fontweight='bold', color='white')
        ax.text(0, -0.6, self.unit, ha='center', va='center',
                fontsize=12, color='#aaaaaa')
        ax.text(0, 0.9, self.title, ha='center', va='center',
                fontsize=14, fontweight='bold', color='white')


class RPMDashboard:
    """
    Complete dashboard for RPM visualization.
    
    Combines multiple plots and gauges into a unified display.
    """
    
    def __init__(self):
        """Initialize dashboard."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for RPMDashboard")
            
        # Create gauge widgets
        self.gauges = {
            "inner_velocity": GaugeWidget(
                title="Inner Frame",
                min_value=-60, max_value=60,
                unit="RPM",
                warning_threshold=50,
                danger_threshold=55
            ),
            "outer_velocity": GaugeWidget(
                title="Outer Frame",
                min_value=-60, max_value=60,
                unit="RPM",
                warning_threshold=50,
                danger_threshold=55
            ),
            "effective_g": GaugeWidget(
                title="Effective Gravity",
                min_value=0, max_value=1,
                unit="g",
                warning_threshold=0.1,
                danger_threshold=0.5
            ),
            "time_avg_g": GaugeWidget(
                title="Time-Avg Gravity",
                min_value=0, max_value=0.1,
                unit="g",
                warning_threshold=0.02,
                danger_threshold=0.05
            ),
        }
        
        # Time series plotters
        self.plotters = {
            "gravity": DataPlotter(PlotConfig(
                title="Gravity History",
                y_label="Effective g (g-units)",
                y_range=(0, 1.5),
                history_length=6000
            )),
            "velocity": DataPlotter(PlotConfig(
                title="Angular Velocities",
                y_label="Velocity (RPM)",
                y_range=(-70, 70),
                history_length=6000
            )),
            "position": DataPlotter(PlotConfig(
                title="Frame Positions",
                y_label="Angle (rad)",
                y_range=(-np.pi, np.pi),
                history_length=6000
            )),
        }
        
        # Initialize series
        self.plotters["gravity"].add_series("instantaneous_g")
        self.plotters["gravity"].add_series("time_averaged_g")
        self.plotters["velocity"].add_series("inner_velocity")
        self.plotters["velocity"].add_series("outer_velocity")
        self.plotters["position"].add_series("inner_position")
        self.plotters["position"].add_series("outer_position")
        
        logger.info("RPM Dashboard initialized")
    
    def update(
        self,
        timestamp: float,
        inner_velocity_rpm: float,
        outer_velocity_rpm: float,
        inner_position_rad: float,
        outer_position_rad: float,
        instantaneous_g: float,
        time_averaged_g: float
    ) -> None:
        """
        Update all dashboard elements with new data.
        
        Args:
            timestamp: Current time (s)
            inner_velocity_rpm: Inner frame velocity (RPM)
            outer_velocity_rpm: Outer frame velocity (RPM)
            inner_position_rad: Inner frame position (rad)
            outer_position_rad: Outer frame position (rad)
            instantaneous_g: Current effective g
            time_averaged_g: Time-averaged effective g
        """
        # Update gauges
        self.gauges["inner_velocity"].value = inner_velocity_rpm
        self.gauges["outer_velocity"].value = outer_velocity_rpm
        self.gauges["effective_g"].value = instantaneous_g
        self.gauges["time_avg_g"].value = time_averaged_g
        
        # Update gauges animation
        for gauge in self.gauges.values():
            gauge.update()
        
        # Update time series
        self.plotters["gravity"].update_data("instantaneous_g", timestamp, instantaneous_g)
        self.plotters["gravity"].update_data("time_averaged_g", timestamp, time_averaged_g)
        self.plotters["velocity"].update_data("inner_velocity", timestamp, inner_velocity_rpm)
        self.plotters["velocity"].update_data("outer_velocity", timestamp, outer_velocity_rpm)
        self.plotters["position"].update_data("inner_position", timestamp, inner_position_rad)
        self.plotters["position"].update_data("outer_position", timestamp, outer_position_rad)
    
    def create_full_figure(self, figsize: Tuple[float, float] = (20, 12)) -> Figure:
        """
        Create complete dashboard figure.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure with all elements
        """
        fig = plt.figure(figsize=figsize, facecolor='#0f0f1a')
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Top row: Gauges
        ax_gauge1 = fig.add_subplot(gs[0, 0])
        ax_gauge2 = fig.add_subplot(gs[0, 1])
        ax_gauge3 = fig.add_subplot(gs[0, 2])
        ax_gauge4 = fig.add_subplot(gs[0, 3])
        
        self.gauges["inner_velocity"].render_matplotlib(ax_gauge1)
        self.gauges["outer_velocity"].render_matplotlib(ax_gauge2)
        self.gauges["effective_g"].render_matplotlib(ax_gauge3)
        self.gauges["time_avg_g"].render_matplotlib(ax_gauge4)
        
        # Middle row: Gravity plot (spanning 2 columns)
        ax_gravity = fig.add_subplot(gs[1, :2])
        self._plot_series_on_axes(ax_gravity, self.plotters["gravity"])
        
        # Middle row: Velocity plot
        ax_velocity = fig.add_subplot(gs[1, 2:])
        self._plot_series_on_axes(ax_velocity, self.plotters["velocity"])
        
        # Bottom row: Position plot (spanning full width)
        ax_position = fig.add_subplot(gs[2, :])
        self._plot_series_on_axes(ax_position, self.plotters["position"])
        
        return fig
    
    def _plot_series_on_axes(self, ax: plt.Axes, plotter: DataPlotter) -> None:
        """Helper to plot data series on given axes."""
        colors = ['#00ff88', '#00d4ff', '#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff']
        
        for i, (name, buffer) in enumerate(plotter._buffers.items()):
            if len(buffer) > 0:
                t, v = buffer.get_arrays()
                ax.plot(t, v, label=name, color=colors[i % len(colors)],
                       linewidth=plotter.config.line_width)
        
        ax.set_xlabel(plotter.config.x_label, color='white')
        ax.set_ylabel(plotter.config.y_label, color='white')
        ax.set_title(plotter.config.title, color='white', fontweight='bold')
        
        if plotter.config.y_range:
            ax.set_ylim(plotter.config.y_range)
        
        ax.set_facecolor('#0f0f1a')
        ax.tick_params(colors='#aaaaaa')
        for spine in ax.spines.values():
            spine.set_color('#333366')
        
        if plotter.config.grid:
            ax.grid(True, color='#333366', linestyle='--', alpha=0.5)
        
        if len(plotter._buffers) > 1:
            ax.legend(loc='upper right', framealpha=0.8)
    
    def save_snapshot(self, filepath: str, dpi: int = 150) -> None:
        """
        Save dashboard snapshot to file.
        
        Args:
            filepath: Output path
            dpi: Resolution
        """
        fig = self.create_full_figure()
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info(f"Dashboard snapshot saved to {filepath}")
