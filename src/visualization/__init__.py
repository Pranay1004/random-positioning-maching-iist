"""
Visualization Package
======================
Real-time and static visualization for RPM data.
"""

from .plotting import (
    PlotConfig,
    TimeSeriesBuffer,
    DataPlotter,
    GaugeWidget,
    RPMDashboard,
)

__all__ = [
    "PlotConfig",
    "TimeSeriesBuffer",
    "DataPlotter",
    "GaugeWidget",
    "RPMDashboard",
]
