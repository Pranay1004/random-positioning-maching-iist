# Random Positioning Machine - Digital Twin Simulator

A web-based simulator for Random Positioning Machines (RPM) used in microgravity research. Real-time 3D visualization, paper-accurate physics, and comprehensive analytics.

**Institution:** Department of Aerospace Engineering, IIST  
**License:** MIT  
**Version:** 3.1.1

---

## Quick Navigation

**New Users:**
1. See Folder Structure below
2. Read [docs/README.txt](docs/README.txt) for complete guide
3. See Installation section

**Developers:**
1. Check [src/](src/) for source code
2. Review [docs/Formulas.txt](docs/Formulas.txt) for physics

**Researchers:**
1. See [docs/REFERENCES.txt](docs/REFERENCES.txt) for citations
2. Read [docs/SAMPLES_METRIC_EXPLAINED.txt](docs/SAMPLES_METRIC_EXPLAINED.txt)

---

## Folder Structure

```
Digital_Twin/
├── README.md              <- You are here
├── README.txt             <- Comprehensive documentation
├── LICENSE                <- MIT License
├── DISCLAIMER.md          <- Legal information
├── requirements.txt       <- Python dependencies
│
├── docs/                  <- DOCUMENTATION
│   ├── README.txt         <- Complete user guide
│   ├── Formulas.txt       <- All physics equations
│   ├── SAMPLES_METRIC_EXPLAINED.txt
│   ├── CAD_REFERENCE.txt
│   ├── REFERENCES.txt
│   └── FUTURE_FEATURES.txt
│
├── src/                   <- SOURCE CODE
│   ├── main.py            <- Application entry point
│   ├── simulation/        <- Physics engine
│   ├── webapp/            <- Web frontend & server
│   ├── hardware_interface/
│   ├── data_pipeline/
│   └── visualization/
│
├── tests/                 <- TESTING
│   └── test_core.py
│
├── config/                <- CONFIGURATION
│   └── main_config.yaml
│
├── firmware/              <- HARDWARE CONTROL
│   └── arduino/
│
├── Figure/                <- REFERENCE IMAGES
│
└── database/              <- DATA STORAGE
```

---

## Purpose of Each Folder

| Folder | Purpose | Who Uses It |
|--------|---------|-------------|
| docs/ | Documentation, guides, physics equations | Everyone |
| src/ | Python source code | Developers |
| tests/ | Unit and integration tests | QA, Developers |
| config/ | Settings and parameters | DevOps |
| database/ | Simulation results, logs | Data analysts |
| firmware/ | Motor control code | Hardware engineers |
| Figure/ | Reference images and diagrams | Everyone |

---

## Features

- Real-time 3D visualization with Three.js
- Paper-accurate physics based on Yotov et al. research
- Live microgravity metrics (taSMG, time-averaged gravity)
- Interactive controls for motor speeds and frame dimensions
- WebSocket streaming for low-latency updates (50 Hz physics, 20 Hz display)
- Responsive dark-theme UI with professional dashboard

---

## Installation

### Prerequisites
- Python 3.10+
- pip (Python package manager)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Quick Start

```bash
# 1. Navigate to project
cd Digital_Twin

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run server
python src/main.py

# 5. Open browser at http://localhost:8080
```

See [docs/README.txt](docs/README.txt) for detailed setup.

---

## Documentation

| Document | What You'll Learn |
|----------|-------------------|
| [docs/README.txt](docs/README.txt) | Complete user guide, physics theory, API reference |
| [docs/Formulas.txt](docs/Formulas.txt) | All mathematical equations |
| [docs/SAMPLES_METRIC_EXPLAINED.txt](docs/SAMPLES_METRIC_EXPLAINED.txt) | What "Samples" means |
| [docs/CAD_REFERENCE.txt](docs/CAD_REFERENCE.txt) | Mechanical design |
| [docs/REFERENCES.txt](docs/REFERENCES.txt) | Research citations |
| [DISCLAIMER.md](DISCLAIMER.md) | Legal info, licensing |

---

## What is SAMPLE?

The Samples metric in the dashboard shows:
- Count of measurements collected during rotation
- Convergence indicator - higher samples = more accurate results
- Quality metric - when physics has settled to stable values

**Real-world analogy:** Like averaging 100 photos to get a clear image.

```
Low Samples (<50):     Physics still calculating, results preliminary
Medium Samples (50-200): Good stability, typical for analysis  
High Samples (200+):   Excellent stability, safe for publication
```

For details: [docs/SAMPLES_METRIC_EXPLAINED.txt](docs/SAMPLES_METRIC_EXPLAINED.txt)

---

## License and Legal

- **License:** MIT License (see [LICENSE](LICENSE) file)
- **Copyright:** Department of Aerospace Engineering, IIST
- **AI-Generated Code:** ~80% AI-assisted, ~20% human-directed
- **Warranty:** NONE - Use at your own risk

See [DISCLAIMER.md](DISCLAIMER.md) for complete legal information.

---

## For New Developers

**First Time Setup:**
1. Read [docs/README.txt](docs/README.txt) (sections 1-3)
2. Understand [docs/Formulas.txt](docs/Formulas.txt) (physics basics)
3. Explore [src/simulation/physics_engine.py](src/simulation/physics_engine.py)
4. Run `python src/main.py` and test in browser

**Code Structure:**
```
src/
├── main.py                <- Entry point, starts server
├── simulation/
│   └── physics_engine.py  <- Rotation matrices, gravity
├── webapp/
│   ├── server.py          <- FastAPI backend
│   └── static/            <- HTML/CSS/JavaScript
└── hardware_interface/    <- Motor control
```

---

## Project Status

| Aspect | Status |
|--------|--------|
| Physics Engine | Complete (v3.1.1) |
| Web Dashboard | Complete |
| 3D Visualization | Complete |
| API | Complete |
| Hardware Integration | In Progress (v3.2.0) |
| Unit Tests | Complete |

---

## Important Notes

- **Research Use Only:** This is a physics research project
- **AI-Generated:** Code uses AI assistance (see DISCLAIMER.md)
- **No Warranty:** Use at your own risk for research purposes
- **Citation Required:** Please cite docs/REFERENCES.txt

---

## Support

- **Usage:** See [docs/README.txt](docs/README.txt)
- **Physics:** See [docs/Formulas.txt](docs/Formulas.txt)
- **Bugs:** Open an Issue on GitHub
- **Legal:** Read [DISCLAIMER.md](DISCLAIMER.md)

---

**Last Updated:** January 16, 2026  
**Version:** 3.1.1  
**License:** MIT
