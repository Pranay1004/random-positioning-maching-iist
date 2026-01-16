"""
RPM Digital Twin - Core Package
================================
Commercial-grade software for Random Positioning Machine control and digital twin simulation.

Modules:
--------
- hardware: Hardware interface layer
- simulation: Physics simulation engine
- data: Data acquisition and processing
- analysis: Microgravity analysis
- ui: User interface components

Copyright (c) 2024 RPM Digital Twin Team
All rights reserved.
"""

__version__ = "1.0.0"
__author__ = "RPM Digital Twin Team"
__license__ = "Commercial"

from pathlib import Path

# Package root directory
PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

# Configuration paths
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, LOGS_DIR, DATA_DIR / "experiments", DATA_DIR / "simulations"]:
    directory.mkdir(parents=True, exist_ok=True)
