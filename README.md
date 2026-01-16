# Random Positioning Machine - Digital Twin Simulator

A web-based simulator for Random Positioning Machines (RPM) used in microgravity research. Real-time 3D visualization, paper-accurate physics, and comprehensive analytics.

**Institution:** Department of Aerospace Engineering, IIST  
**License:** MIT  
**Version:** 3.1.1

---

## 📚 Quick Navigation

**New Users:**
1. See [Folder Structure](#-folder-structure) below
2. Read [docs/README.txt](docs/README.txt) for complete guide
3. See **Installation** section

**Developers:**
1. Check [src/](src/) for source code
2. Review [docs/Formulas.txt](docs/Formulas.txt) for physics
3. See [Architecture](docs/README.txt#3-architecture) section

**Researchers:**
1. See [docs/REFERENCES.txt](docs/REFERENCES.txt) for citations
2. Read [docs/SAMPLES_METRIC_EXPLAINED.txt](docs/SAMPLES_METRIC_EXPLAINED.txt)
3. Check [docs/CAD_REFERENCE.txt](docs/CAD_REFERENCE.txt)

---


--
heck [docs/CAD_REFERENCE.txt](docs/CAD_REFERENCE.txt)
ETRIC_EXPLAINED.txt)
y research. Real-timl_y research. Real-timl_y reseaME.md (YOU ARE HERE)
├── 📄 README.txt (Comprehensive documentation)
├── 📄 LICENSE (MIT License)
├── 📄 DISCLAIMER.md (Legal information)
│
├── 📂 docs/ ← DOCUMEN├── 📂 docs/ ← DOCUMEN├── 📂 docs/ ← DOde├── 📂 do details)
│   ├── Formulas.txt (All physics equations)
│   ├── SAMPLES_METRIC_EXPLAINED.txt (Understandi│   ├── SAMPLES_METRIC_EXPLAINED.txt (Understandinica�│   ├── SAMPLES_METRIC��─ REFERENCES.tx│   ├── SAMPLES_METRIC_EXPLA�│   ├── SAMPLES_METRIC_EXPLAINED.txt (Understandi─ │   ├── SAMPLES_METRIC_EXPLAINED.txt (Understandi│plication entry poin│   ├── SAMPLES_METRIC_EXPLAINED.txt (Understandi│  )
│   ├── SAMPLES_METRIC_EXPLAINED.txt (Understandi│   ├── SAMP� │   ├── SAMPLES_METRIC_EXPLAINED.txt (Understand� server.py (FastAPI server)
│       └── static/
│      │      │      x.�│      │      │      x.�│      │      │      x.�│              �├── css/ (Styling)
│           └── a│           └── a│       �│           └── a│           └─── test_physics.py (Physics validation tests)
│   ├── test_websocket.py (Server tests)
│   └── fixtures/ (Te│   └── fixtures/ (Te│   └── fixATION FOLDER
│   ├── settings.yaml (Application settings)
│   ├── motor_config.json (Motor parameters)
│   └── frame_dimensions.json (RPM geometry)
│
├── 📂 database/ ← DATA STORAGE FOLDER
│   ├── measurements.db (Simulation results)
│   └── logs/ (Simulation logs)
│
├── 📂 firmware/ ← HARDWARE CONTROL FOLDER
│   ├── motor_driver.py (NEMA mot│   ├── motor_driver.py (NEMA mot│   ├── motor_driver.py (NEMA mot│   ├── motor_driver.py (NEMA mot│   ├── motor_driver.py (NEMA mre│   ├── mo��
├─�├─�├─�├─�├─�├─�├─�├─�├─�├─�├─�├─�├─�├─�├─�├─�├─�├─�├─�├Installation script)
�└── .gitignore (Git configuration)
```

---

## 🎯 Purpose of Each Folder

| Folder | Purpose | Who Uses It |
|--------|---------|------------|
| **docs/** | Documentation, guides| **docs/** | Documentation, guides| **docs/** | Documentation, guides| **docs/** | Docus/** | Unit & integration tests | QA, Developers |
| **config/** | Settings & parameters (YAML, JSON) | DevOps, Configuration |
| **database/** | Simulation results,| **database/** | Simulation results,| **databa c| **database/** | Simulation results,| **database/*Re| **database/** | Simulation results,|

---
---
database/** | Simulation results,| **database/* with Three.js
- **Paper-accurate physics** based on Yotov et al. research
- **Live microgravity metrics** (taSMG, time-averaged gravity)
- **Interactive controls** for motor speeds and frame dimensions
- **WebSocket streaming** for low-latency updates (50 Hz physics, 20 Hz display)
- **Responsive dark-theme UI** with professional dashboard

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- pip (Python package manager- - M- pip (Python package managerref- pip (Python package manager- - t
- pip (Python package manager- -t
- pip (Python package manager- -t
- pip (Python package managerref- pip (Python package manager- - t
 | Simulation results,| **database/*Re| **database/** | Simulation resu
pip install -r requirements.txt
pip install -r requirements-visualization.txt
pip install -r requirements-visualization.txt
base/** | Simulation resu
(Python package manager- - t
README.txt](docs/README.txt) for detailed setup and troubleshooting.

---

## ##��##oc## ##��##oc## ##��##oc## ##��##oc## ##��##oc## ##��##oc## ##---------------|
| [docs/README.txt](docs/README.txt) | Complete user guide, physics theory, API reference |
| [docs/Formulas| [docs/Formulas| [docs/Formulas| [docs/Formulas| [docs/Formulaimulator |
| [docs/SAMPLES_METRIC_EXPLAINED.txt](docs/SAMPLES_METRIC_EXPLAINED.txt)| [docs/SAMPLES_METRIC_EXPLAINED.txt](docs/SAMPLES_METRICREFERENCE.txt](docs/CAD_REFERENCE.txt) | Mechanical design and | [docs/SAMPLES_METRIC_EXPLAINED.txt](docs/SAMPLES_METRIC_EXPLAINED.txt)| [docs/SAMPLES_METRIC_EXPLur| [docs/SAMPLES_METRIFEATURES.txt]| [docs/SAMPLES_METRIC_EXPLAINEDnned features and roadmap |
| [DISCLAIMER.md](DISCLAIMER.md) | Legal | [DISCLAIMER.md](DISCLAIMER.md) | Legal | [DISCLAIMER.md]What is SAMPLE| [DISCLAIMER.md](DISCLAIMER.mhe dashboard shows:
- **Count of measurements** collected during rotation
- **Convergence indicator** - higher samples = more accurate results
- **Qu- **Qu- **Qu- **Qu- **Qu- **Qu- **Qu- **Qu- **Qu- **Qu- **Qu- **Qu-l-world analogy:** Like averaging 100 photos to get a clear image.

```
Low Samples (<50):    PhLow Samples (<50):    PhLow Samples (<50):    PhLum Samples (50-200): Good stability, typical for analysis  
High Samples (200+):   Excellent stability, safe for publication
```

For complete details, see: [docs/SAMPLES_METRIC_EXPLAINED.txt](docs/SAMPLES_METRIC_EXPLAINED.txt)

---

## ⚖️ License & Legal

- **License:** MIT License (see [LICENSE](LICENSE) file)
- **Copyright:** Department of Aerospace Engineering, IIST
- **AI-Generated Code:** ~80% AI-assisted, ~20% human-directed
- **Warranty:** NONE - Use at your own risk

**See:** [DISCLAIMER.md](DISCLAIMER**See:** [DISCLAIMER.md](DISCLAIMER**See:** [DISCLAIMER.md](DISCLAIMER**See:** [DISCLAIMER.md](DISCLAIMER**See:** [DISCLAIMER.md](DISCLAIMER**See:** [DISCLAIMER.-i**See:** [DISCLAIMER.md](DISCLAIMER**See:** [DISCLAI.com/Pranay1004/random-positioning-machine-iist/issues)

---

## 💡 F## 💡 F## 💡 F## 💡 F## 💡 F## 💡 F## 💡 F## 💡 F## 💡 F## 💡 F## 💡 F## 💡 F## 💡 F## 💡 F## 💡 F## 💡 F## s/Formu## 💡 F## 💡 F## 💡 F## 💡 F## 💡 F## _engine.py](src/physics_engine.py) (main algorithm)
4. Run: `python src/main.py` and test in browser

**Understanding the Code Structure:**
```
src/
�├── main.py → Entry point, starts server
├── physics_engine.py → Rotation matrices, graty calculations
�├── websocket_server.py → Real-time data broadcast
└── webapp/
    ├── server.py → FastAPI backend
    └── static/ → HTML/CSS/JavaScript frontend
```

See [Folder Structure](#-folder-structure) above for detaSee [Folder Structure](#-folder-structure) above for detaSee [Folder Structure](#-folder-structure) above for detaSee [Folder Structure](#-folder-structure) above for detaSee [Folder Structure](#-folder-structure) above for detaSee [Folder Structure](#-folder-structure) above for detaSee [Folder Structu-


e e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e �e Ime �e �e �e �e �e �e �e �e �e �e �e � a physics research project
- **AI-Genera- d:** Co- **AI-Genera- d:** Co- **AI-Genera- d:** Co- *LAIMER.md))
- **No Warranty:** Use at your own risk for research purposes
- **Citation Required:** Please cite [docs/REFERENCES.txt](docs/REFERENCES.txt)

---

## ��📞 Support

For questions abouFor questige:** See [docs/README.txt](docs/README.txt)
- **Physics:** See [docs/Formulas.txt](docs/Formulas.txt)
- **Bugs:** Open an [Issue](https://github.com/Pranay1004/random-positioning-machine-iist/issues)
- **Legal:** Read [DISCLAIMER.md](DISCLAIMER.md)

---

**Last Updated:** January 16, 2026  
**Version:** 3.1.1  
**License:** MIT
