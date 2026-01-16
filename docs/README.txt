================================================================================
          RPM DIGITAL TWIN - MICROGRAVITY SIMULATION PLATFORM
================================================================================

Version: 3.1.0
Last Updated: 15 January 2026
Platform: Python 3.10+ / Web Browser

================================================================================
                              TABLE OF CONTENTS
================================================================================

1. OVERVIEW
2. PHYSICS & THEORY
3. ARCHITECTURE
4. INSTALLATION
5. USAGE
6. VALIDATION & VERIFICATION
7. API REFERENCE

================================================================================
                              1. OVERVIEW
================================================================================

WHAT IS AN RPM?
---------------
A Random Positioning Machine (RPM) is a laboratory device used to simulate
microgravity conditions on Earth. It consists of two independently rotating
frames that continuously change the orientation of a sample relative to
Earth's gravity vector.

KEY FEATURES
------------
* Paper-accurate physics - Implements Eq. 2 rotation matrices
* Real-time 3D visualization - Three.js rendering at 50Hz
* Time-averaged microgravity (taSMG) - Correct ||<g>|| calculation
* SpaceX/Tesla-style UI - Dark theme professional dashboard
* Hardware-ready architecture - Command queue for NEMA motor control
* WebSocket streaming - Low-latency data broadcast at 20Hz

================================================================================
                           2. PHYSICS & THEORY
================================================================================

REFERENCE FRAMES
----------------
+-------------+---------------------------+---------------------+
| Frame       | Description               | Gravity Direction   |
+-------------+---------------------------+---------------------+
| Lab Frame   | Fixed to Earth            | Always (0, -1, 0)   |
| Sample Frame| Attached to payload       | Changes continuously|
+-------------+---------------------------+---------------------+

"Gravity Vector (Sample Frame)" means: The direction gravity appears to come
from, as seen by the sample/payload inside the RPM. Because the RPM rotates,
this direction constantly changes - which is how we simulate microgravity!


COORDINATE SYSTEM
-----------------
        Y (up)
        |
        |
        |
        +-------- X (right)
       /
      /
     Z (toward viewer)


ROTATION MATRICES (Equation 2)
------------------------------
The gravity vector transformation from lab frame to sample frame:

    g_sample = R_Z(phi) * R_Y(psi) * g_lab

Where:
  - phi (φ) = Inner frame angle (rotates around Z-axis)
  - psi (ψ) = Outer frame angle (rotates around Y-axis)
  - g_lab = (0, -1, 0) = Earth's gravity in lab frame

R_Y(psi) - Outer Frame Rotation Matrix:
    [ cos(psi)   0   sin(psi) ]
    [    0       1      0     ]
    [-sin(psi)   0   cos(psi) ]

R_Z(phi) - Inner Frame Rotation Matrix:
    [ cos(phi)  -sin(phi)  0 ]
    [ sin(phi)   cos(phi)  0 ]
    [    0          0      1 ]


GRAVITY COMPONENTS (Gx, Gy, Gz)
-------------------------------
Applying the rotation matrices to g_lab = (0, -1, 0):

    Gx = -sin(phi) * cos(psi)
    Gy = -cos(phi)
    Gz = sin(phi) * sin(psi)

These values oscillate between -1 and +1 as the frames rotate.

NOTE: The magnitude ||g|| = sqrt(Gx^2 + Gy^2 + Gz^2) = 1 ALWAYS
      (gravity magnitude is constant, only direction changes)


AVERAGE GRAVITY (<Gx>, <Gy>, <Gz>)
----------------------------------
Time-averaged gravity components over the simulation:

    <Gx> = (1/N) * SUM(Gx_i)   for i = 1 to N
    <Gy> = (1/N) * SUM(Gy_i)   for i = 1 to N
    <Gz> = (1/N) * SUM(Gz_i)   for i = 1 to N

For perfect microgravity, all three averages should approach ZERO.


G* (taSMG) - THE KEY METRIC
---------------------------
G* is the time-averaged Simulated Microgravity magnitude:

    G* = ||<g>|| = sqrt(<Gx>^2 + <Gy>^2 + <Gz>^2)

CRITICAL DISTINCTION:
    WRONG:   <||g||>  = average of magnitudes = always ~1
    CORRECT: ||<g>||  = magnitude of averaged vector -> 0

This is the most important number! It tells you how good your
microgravity simulation is.

TARGET VALUES (from research literature):
    Excellent: 0.0018g - 0.0033g
    Good:      < 0.01g
    Fair:      < 0.05g


VELOCITY RATIO (gamma)
----------------------
    gamma = omega_phi / omega_psi = (Inner RPM) / (Outer RPM)

For optimal microgravity:
  - gamma ≈ 1 (similar speeds)
  - gamma should be IRRATIONAL for non-repeating trajectories
  - Recommended: 2.0 / 2.1 RPM (ratio = 0.952...)


TRAJECTORY ON UNIT SPHERE
-------------------------
Since ||g|| = 1 always, the gravity vector tip traces a path on a
unit sphere (radius = 1).

The "Trajectory on Unit Sphere" visualization shows:
  - Each point = instantaneous gravity direction (Gx, Gy, Gz)
  - The line = path traced over time
  - GOOD microgravity = uniform coverage of entire sphere
  - POOR microgravity = trajectory clusters in one region


================================================================================
                            3. ARCHITECTURE
================================================================================

SYSTEM DIAGRAM
--------------

+------------------------------------------------------------------+
|                       FRONTEND (Browser)                          |
|  +-------------+  +-------------+  +-------------------------+    |
|  |  Three.js   |  |  Chart.js   |  |    Control Panels       |    |
|  |  3D Scene   |  |   Graphs    |  | Motors, Dimensions...   |    |
|  +------+------+  +------+------+  +-----------+-------------+    |
|         |                |                     |                  |
|         +----------------+---------------------+                  |
|                          | WebSocket                              |
+--------------------------|----------------------------------------+
                           |
+--------------------------|----------------------------------------+
|                    BACKEND (Python/FastAPI)                       |
|  +------------------------------------------------------------+  |
|  |                 WebSocket Server (:8080)                    |  |
|  |               ConnectionManager (broadcast)                 |  |
|  +---------------------------+--------------------------------+  |
|                              |                                    |
|  +---------------------------+--------------------------------+  |
|  |                   SimulationEngine                          |  |
|  |  * 50Hz physics loop                                        |  |
|  |  * 20Hz WebSocket broadcast                                 |  |
|  |  * Angle integration: phi += omega_phi*dt                   |  |
|  +---------------------------+--------------------------------+  |
|                              |                                    |
|  +---------------------------+--------------------------------+  |
|  |                 MicrogravitySimulator                       |  |
|  |  * Rotation matrices R_Y(psi), R_Z(phi)                    |  |
|  |  * Gravity calculation (Eq. 2)                              |  |
|  |  * Time-averaging & G* calculation                          |  |
|  +---------------------------+--------------------------------+  |
|                              |                                    |
|  +---------------------------+--------------------------------+  |
|  |                 HardwareCommandQueue                        |  |
|  |  * Rate-limited commands (50/sec)                           |  |
|  |  * Command coalescing                                       |  |
|  |  * Future: NEMA motor control                               |  |
|  +------------------------------------------------------------+  |
+-------------------------------------------------------------------+


FILE STRUCTURE
--------------
Digital_Twin/
|-- README.md                    # Documentation (Markdown)
|-- README.txt                   # Documentation (Plain text)
|-- requirements.txt             # Python dependencies
|-- docs/
|   |-- FUTURE_FEATURES.txt      # Planned features roadmap
|   +-- CAD_REFERENCE.txt        # CAD design documentation
|-- src/
|   +-- webapp/
|       |-- server.py            # FastAPI backend server
|       +-- static/
|           +-- index.html       # Frontend dashboard
+-- venv/                        # Python virtual environment


================================================================================
                            4. INSTALLATION
================================================================================

PREREQUISITES
-------------
* Python 3.10 or higher
* Modern web browser (Chrome, Firefox, Safari, Edge)

SETUP STEPS
-----------
1. Navigate to project directory:
   cd /path/to/Digital_Twin

2. Create virtual environment:
   python3 -m venv venv

3. Activate virtual environment:
   source venv/bin/activate        (macOS/Linux)
   venv\Scripts\activate           (Windows)

4. Install dependencies:
   pip install fastapi uvicorn websockets numpy

5. Run the server:
   cd src/webapp
   python server.py

6. Open browser:
   http://localhost:8080


================================================================================
                              5. USAGE
================================================================================

BASIC CONTROLS
--------------
+---------------------+----------------------------------------+
| Control             | Function                               |
+---------------------+----------------------------------------+
| Inner Motor Speed   | Set inner frame rotation (0-20 RPM)    |
| Outer Motor Speed   | Set outer frame rotation (0-20 RPM)    |
| START               | Begin simulation                       |
| STOP                | Pause simulation                       |
| RESET               | Clear all data, reset angles           |
+---------------------+----------------------------------------+

FRAME DIMENSIONS (Defaults)
---------------------------
* Inner Frame:  80 cm x 80 cm
* Outer Frame:  150 cm x 150 cm
* Payload:      50 cm x 50 cm x 50 cm

VISUALIZATION DISPLAYS
----------------------
* 3D View         - Real-time RPM rotation
* Gravity Vector  - Instantaneous Gx, Gy, Gz
* Avg Gravity     - Time-averaged <Gx>, <Gy>, <Gz>
* taSMG (G*)      - Main metric (should converge to ~0.003g)
* Convergence     - G* over time chart
* Components      - Gx, Gy, Gz oscillation chart
* Sphere          - Gravity vector trajectory on unit sphere


================================================================================
                      6. VALIDATION & VERIFICATION
================================================================================

A. PHYSICS VALIDATION TESTS
---------------------------

+------------------+-------------------------+----------------------+
| Test             | Expected Result         | How to Verify        |
+------------------+-------------------------+----------------------+
| Static (0 RPM)   | Gy=-1, Gx=Gz=0         | Stop both motors     |
| Inner only       | Gx,Gy oscillate; Gz=0  | Set outer = 0        |
| Outer only       | Gx,Gz oscillate        | Set inner = 0        |
| Equal speeds     | G* -> 0.003g           | Run 30+ minutes      |
| Irrational gamma | Best sphere coverage   | Use 2.0/2.1 RPM      |
+------------------+-------------------------+----------------------+


B. MATHEMATICAL VERIFICATION
----------------------------
Test at known angles (phi=45°, psi=30°):

    Gx = -sin(45°) * cos(30°) = -0.612
    Gy = -cos(45°)            = -0.707
    Gz = sin(45°) * sin(30°)  = +0.354
    
    Magnitude = sqrt(0.612^2 + 0.707^2 + 0.354^2) = 1.000


C. VISUAL VERIFICATION CHECKLIST
--------------------------------
[ ] Sphere trajectory covers entire sphere uniformly
[ ] Gravity chart shows symmetric sinusoidal oscillations
[ ] G* decreases over time, approaching 0
[ ] 3D frames rotate smoothly on correct axes
[ ] Outer frame rotates around Y (vertical)
[ ] Inner frame rotates around Z (horizontal)


D. BENCHMARK VALUES
-------------------
+------------+-------------+---------+
| Duration   | Expected G* | Samples |
+------------+-------------+---------+
| 1 minute   | < 0.1g      | ~3,000  |
| 10 minutes | < 0.02g     | ~30,000 |
| 30 minutes | < 0.005g    | ~90,000 |
| 60 minutes | < 0.003g    | ~180,000|
+------------+-------------+---------+


================================================================================
                           7. API REFERENCE
================================================================================

WEBSOCKET COMMANDS
------------------

Start Simulation:
    {"action": "start"}

Stop Simulation:
    {"action": "stop"}

Reset Simulation:
    {"action": "reset"}

Set Motor Speeds:
    {
      "action": "config",
      "inner_rpm": 2.0,
      "outer_rpm": 2.0
    }

Set Frame Dimensions (in meters):
    {
      "action": "frame_dimensions",
      "inner_length": 0.80,
      "inner_breadth": 0.80,
      "outer_length": 1.50,
      "outer_breadth": 1.50
    }


WEBSOCKET RESPONSE FORMAT
-------------------------
    {
      "phi_deg": 45.0,           // Inner frame angle (degrees)
      "psi_deg": 30.0,           // Outer frame angle (degrees)
      "omega_phi_rpm": 2.0,      // Inner motor speed (RPM)
      "omega_psi_rpm": 2.0,      // Outer motor speed (RPM)
      "gamma": 1.0,              // Velocity ratio
      "gravity_x": -0.612,       // Instantaneous Gx
      "gravity_y": -0.707,       // Instantaneous Gy
      "gravity_z": 0.354,        // Instantaneous Gz
      "avg_gravity_x": 0.001,    // Time-averaged <Gx>
      "avg_gravity_y": -0.002,   // Time-averaged <Gy>
      "avg_gravity_z": 0.001,    // Time-averaged <Gz>
      "mean_g": 0.0024,          // G* (taSMG) - MAIN METRIC
      "instantaneous_g": 1.0,    // ||g|| (always 1)
      "min_g": 0.0018,           // Minimum G* achieved
      "max_g": 0.0033,           // Maximum G* achieved
      "samples": 50000,          // Total samples collected
      "time_elapsed": 1000.5,    // Simulation time (seconds)
      "status": "running"        // "running" or "stopped"
    }


================================================================================
                              QUICK REFERENCE
================================================================================

KEY FORMULAS
------------
    Gx = -sin(phi) * cos(psi)
    Gy = -cos(phi)
    Gz = sin(phi) * sin(psi)
    
    G* = sqrt(<Gx>^2 + <Gy>^2 + <Gz>^2)
    
    gamma = omega_phi / omega_psi

SYMBOL GLOSSARY
---------------
    phi (φ)    = Inner frame angle
    psi (ψ)    = Outer frame angle
    omega_phi  = Inner frame angular velocity (rad/s or RPM)
    omega_psi  = Outer frame angular velocity (rad/s or RPM)
    gamma (γ)  = Velocity ratio
    G*         = Time-averaged simulated microgravity (taSMG)
    <x>        = Time average of x

TARGET MICROGRAVITY
-------------------
    Excellent: G* < 0.003g
    Good:      G* < 0.01g
    Fair:      G* < 0.05g

================================================================================
                              END OF README
================================================================================
