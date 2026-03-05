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
│   ├── test_core.py
│   └── hardware/          <- Hardware validation tests
│       ├── run_all.py         <- Run full hardware test suite
│       ├── test_base.py       <- Base test class
│       ├── test_pico_w.py     <- RPi Pico W connectivity
│       ├── test_rpi5.py       <- Raspberry Pi 5 diagnostics
│       ├── test_motors.py     <- NEMA motor GPIO tests
│       └── test_slip_rings.py <- Slip ring signal tests
│
├── config/                <- CONFIGURATION
│   └── main_config.yaml   <- Pico W, RPi5, NEMA24, slip ring params
│
├── deploy/                <- DEPLOYMENT SCRIPTS
│   └── rpi5_setup.sh      <- One-shot RPi5 setup script
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
| tests/hardware/ | Real hardware validation (SKIP when disconnected) | Hardware Engineers |
| config/ | Settings and parameters | DevOps |
| deploy/ | Deployment and setup scripts for RPi5 | DevOps, Hardware Engineers |
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
- **Hardware Test Suite:** 7-component validation (RPi Pico W, RPi 5, NEMA 23×2, NEMA 24, Slip Rings×2)
- **Honest test results:** SKIP when hardware disconnected, no hardcoded or fabricated data
- **Web Hardware Dashboard:** Live test status at `/api/hardware/tests/dashboard`
- **RPi5 deployment script:** `deploy/rpi5_setup.sh` for one-shot on-hardware setup

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
│   ├── hardware_tests.py  <- Hardware test API + dashboard router
│   └── static/            <- HTML/CSS/JavaScript
└── hardware_interface/    <- Motor control

tests/hardware/
├── run_all.py             <- Full suite runner
├── test_pico_w.py         <- USB/serial Pico W checks
├── test_rpi5.py           <- RPi5 OS/GPIO/network diagnostics
├── test_motors.py         <- NEMA23/24 GPIO validation
└── test_slip_rings.py     <- Slip ring signal integrity
```

---

## Deployment Options

**Option 1: Local Web App (FastAPI)**
```bash
python src/main.py
# Visit http://localhost:8080
# Hardware dashboard: http://localhost:8080/api/hardware/tests/dashboard
```

**Option 1b: Deploy on Raspberry Pi 5**
```bash
# On the RPi5:
bash deploy/rpi5_setup.sh
# Then run hardware tests:
python -m tests.hardware.run_all
```

**Option 2: Streamlit Cloud (Recommended for quick demo)**
```bash
streamlit run streamlit_app.py
# Or deploy to: https://streamlit.io/cloud
```

See [STREAMLIT_DEPLOYMENT.txt](STREAMLIT_DEPLOYMENT.txt) for cloud deployment guide.

---

## Project Status

| Aspect | Status |
|--------|--------|
| Physics Engine | Complete (v3.1.1) |
| Web Dashboard | Complete |
| 3D Visualization | Complete |
| API | Complete |
| Streamlit App | Complete |
| Hardware Integration | In Progress (v3.2.0) |
| Hardware Test Suite | Complete (7 components) |
| Hardware Test Dashboard | Complete |
| RPi5 Deployment Script | Complete |
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

**Last Updated:** June 2025  
**Version:** 3.2.0  
**License:** MIT
