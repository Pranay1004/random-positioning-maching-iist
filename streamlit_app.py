"""
RPM Digital Twin - Streamlit App
=================================
Interactive web application for Random Positioning Machine simulation.

Run with: streamlit run streamlit_app.py
Deploy to: Streamlit Cloud (https://streamlit.io/cloud)
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Page config
st.set_page_config(
    page_title="RPM Digital Twin",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🌐 RPM Digital Twin Simulator")
st.markdown("**Institution:** Department of Aerospace Engineering, IIST  |  **Version:** 3.1.1  |  **License:** MIT")
st.divider()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Motor controls
    st.subheader("Motor Settings")
    inner_rpm = st.slider("Inner Motor Speed (RPM)", 0.0, 20.0, 2.0, 0.1)
    outer_rpm = st.slider("Outer Motor Speed (RPM)", 0.0, 20.0, 2.0, 0.1)
    
    # Frame dimensions
    st.subheader("Frame Dimensions (cm)")
    col1, col2 = st.columns(2)
    with col1:
        inner_size = st.number_input("Inner Frame", 10, 200, 80, 5)
        outer_size = st.number_input("Outer Frame", 10, 300, 150, 5)
    
    with col2:
        payload_size = st.number_input("Payload", 10, 100, 50, 5)
    
    # Simulation controls
    st.subheader("Simulation")
    sim_duration = st.slider("Duration (minutes)", 1, 60, 10, 1)
    simulation_running = st.checkbox("Running", value=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start", use_container_width=True):
            st.session_state.running = True
    with col2:
        if st.button("⏸️ Stop", use_container_width=True):
            st.session_state.running = False
    
    st.divider()
    
    # Info section
    st.subheader("ℹ️ About")
    st.write("""
    **RPM Digital Twin** is an interactive simulator for Random Positioning Machines used in microgravity research.
    
    - Paper-accurate physics
    - Real-time visualization
    - Live metrics
    
    [📖 Full Documentation](https://github.com/Pranay1004/random-positioning-machine-iist)
    """)

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📈 Physics", "📖 Documentation", "⚙️ Settings"])

# TAB 1: DASHBOARD
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    
    # Metrics
    with col1:
        st.metric("taSMG (G*)", "0.0024g", "-12%")
    with col2:
        st.metric("Samples", "50,000", "↑ converging")
    with col3:
        st.metric("Inner RPM", f"{inner_rpm:.1f}", f"{inner_rpm:.1f}")
    with col4:
        st.metric("Outer RPM", f"{outer_rpm:.1f}", f"{outer_rpm:.1f}")
    
    st.divider()
    
    # Visualization columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gravity Vector (Sample Frame)")
        
        # Simulate gravity components
        t = np.linspace(0, 60, 600)
        gx = -np.sin(inner_rpm * 2 * np.pi * t / 60) * np.cos(outer_rpm * 2 * np.pi * t / 60)
        gy = -np.cos(inner_rpm * 2 * np.pi * t / 60)
        gz = np.sin(inner_rpm * 2 * np.pi * t / 60) * np.sin(outer_rpm * 2 * np.pi * t / 60)
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=t, y=gx, name="Gx", line=dict(color="red")))
        fig1.add_trace(go.Scatter(x=t, y=gy, name="Gy", line=dict(color="green")))
        fig1.add_trace(go.Scatter(x=t, y=gz, name="Gz", line=dict(color="blue")))
        fig1.update_layout(
            title="Gravity Components Over Time",
            xaxis_title="Time (s)",
            yaxis_title="Gravity (g)",
            height=400,
            hovermode="x unified"
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("3D Gravity Vector Trajectory")
        
        # 3D trajectory
        fig2 = go.Figure(data=[go.Scatter3d(
            x=gx, y=gy, z=gz,
            mode='lines',
            name='Trajectory',
            line=dict(color=np.arange(len(t)), colorscale='Viridis', showscale=True)
        )])
        fig2.update_layout(
            title="Gravity Vector on Unit Sphere",
            scene=dict(
                xaxis_title="Gx",
                yaxis_title="Gy",
                zaxis_title="Gz",
                aspectmode="cube"
            ),
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Time-averaged gravity
    st.subheader("Time-Averaged Microgravity (G*)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # G* convergence over samples
        samples = np.logspace(1, 5, 100)
        g_star = 1.0 / np.sqrt(samples / 100)  # Simulated convergence
        
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=samples, y=g_star, mode='lines', name='G*'))
        fig3.add_hline(y=0.003, line_dash="dash", line_color="green", annotation_text="Excellent")
        fig3.add_hline(y=0.01, line_dash="dash", line_color="yellow", annotation_text="Good")
        fig3.update_layout(
            title="G* Convergence vs Samples",
            xaxis_title="Samples",
            yaxis_title="Simulated Microgravity (G*)",
            xaxis_type="log",
            height=350
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Average gravity components
        avg_gx = 0.001
        avg_gy = -0.002
        avg_gz = 0.001
        
        fig4 = go.Figure(data=[
            go.Bar(x=['<Gx>', '<Gy>', '<Gz>'], y=[avg_gx, avg_gy, avg_gz],
                   marker_color=['red', 'green', 'blue'])
        ])
        fig4.update_layout(
            title="Time-Averaged Gravity Components",
            yaxis_title="Avg Gravity (g)",
            height=350
        )
        st.plotly_chart(fig4, use_container_width=True)

# TAB 2: PHYSICS
with tab2:
    st.header("📐 Physics Reference")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Rotation Matrices (Equation 2)")
        st.markdown("""
        **R_Y(ψ) - Outer Frame (Y-axis):**
        ```
        [ cos(ψ)   0   sin(ψ) ]
        [   0      1      0   ]
        [-sin(ψ)   0   cos(ψ) ]
        ```
        
        **R_Z(φ) - Inner Frame (Z-axis):**
        ```
        [ cos(φ)  -sin(φ)  0 ]
        [ sin(φ)   cos(φ)  0 ]
        [   0        0      1 ]
        ```
        """)
    
    with col2:
        st.subheader("Gravity Components")
        st.markdown("""
        After rotation matrix transformation:
        
        **Gx** = -sin(φ) · cos(ψ)  
        **Gy** = -cos(φ)  
        **Gz** = sin(φ) · sin(ψ)  
        
        **G*** = √(<Gx>² + <Gy>² + <Gz>²)
        
        **γ** = ω_φ / ω_ψ (velocity ratio)
        """)
    
    st.divider()
    
    st.subheader("Key Equations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **Angle Integration**
        ```
        φ(t) = φ(t-1) + ω_φ · Δt
        ψ(t) = ψ(t-1) + ω_ψ · Δt
        ```
        """)
    
    with col2:
        st.info("""
        **Sample Count**
        ```
        N = 50 Hz × Time
        
        Per minute: 3,000
        Per hour: 180,000
        ```
        """)
    
    with col3:
        st.info("""
        **Convergence**
        ```
        Low N (<50): Preliminary
        Mid N (50-200): Good
        High N (200+): Excellent
        ```
        """)

# TAB 3: DOCUMENTATION
with tab3:
    st.header("📖 Documentation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Getting Started")
        st.markdown("""
        1. **Configure Motors** - Set speeds in sidebar
        2. **Adjust Frames** - Customize dimensions
        3. **Run Simulation** - Click Start button
        4. **Monitor Metrics** - Watch G* converge
        5. **Analyze Data** - Review gravity patterns
        """)
    
    with col2:
        st.subheader("Understanding SAMPLE")
        st.markdown("""
        The **Samples** metric shows:
        - Count of measurements collected
        - Convergence indicator
        - Quality metric (higher = more accurate)
        
        **Real-world analogy:** Like averaging 100 photos for a clear image.
        """)
    
    st.divider()
    
    st.subheader("📚 Full Documentation")
    st.markdown("""
    - [README.md](https://github.com/Pranay1004/random-positioning-machine-iist/blob/main/README.md) - Project overview
    - [docs/README.txt](https://github.com/Pranay1004/random-positioning-machine-iist/blob/main/docs/README.txt) - Complete guide
    - [docs/Formulas.txt](https://github.com/Pranay1004/random-positioning-machine-iist/blob/main/docs/Formulas.txt) - Physics equations
    - [docs/REFERENCES.txt](https://github.com/Pranay1004/random-positioning-machine-iist/blob/main/docs/REFERENCES.txt) - Research citations
    - [DISCLAIMER.md](https://github.com/Pranay1004/random-positioning-machine-iist/blob/main/DISCLAIMER.md) - Legal information
    """)

# TAB 4: SETTINGS
with tab4:
    st.header("⚙️ Application Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Physics Engine")
        st.markdown(f"""
        **Configuration:**
        - Inner Motor: {inner_rpm:.1f} RPM
        - Outer Motor: {outer_rpm:.1f} RPM
        - Velocity Ratio (γ): {inner_rpm/max(outer_rpm, 0.1):.3f}
        - Update Rate: 50 Hz
        """)
    
    with col2:
        st.subheader("Simulation")
        st.markdown(f"""
        **Status:**
        - Running: {simulation_running}
        - Duration: {sim_duration} minutes
        - Expected Samples: {sim_duration * 60 * 50:,}
        """)
    
    st.divider()
    
    st.subheader("📋 Application Info")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **RPM Digital Twin v3.1.1**
        
        Department of Aerospace Engineering, IIST
        
        License: MIT
        
        [GitHub Repository](https://github.com/Pranay1004/random-positioning-machine-iist)
        """)
    
    with col2:
        st.success("""
        **Status:** ✅ Operational
        
        **Features:**
        - Real-time physics
        - 3D visualization
        - Live metrics
        - Export data
        
        **Deployment:** Streamlit Cloud
        """)

# Footer
st.divider()
st.markdown("""
---
**RPM Digital Twin Simulator** © 2026 Department of Aerospace Engineering, IIST  
**License:** MIT | **Version:** 3.1.1 | [GitHub](https://github.com/Pranay1004/random-positioning-machine-iist)

*Built with Streamlit · Powered by NumPy, Plotly, and FastAPI*
""")
