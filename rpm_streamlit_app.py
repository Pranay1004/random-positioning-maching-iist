"""
RPM Digital Twin - Streamlit Application
=========================================
Interactive Streamlit app for Random Positioning Machine simulation
using the actual physics engine from the research project.

Features:
- Real-time microgravity simulation based on Yotov et al. paper
- 3D visualization with Plotly
- Live metrics dashboard (taSMG, gravity components)
- Motor controls (inner/outer RPM)
- Frame dimension controls
- Axis inclination settings
- Sample convergence tracking

Run: streamlit run rpm_streamlit_app.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
from datetime import datetime
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the actual physics engine
from webapp.server import MicrogravitySimulator, RotationMatrices

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
        padding-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .physics-card {
        background-color: #0f0f1a;
        color: white;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333366;
        margin: 10px 0;
    }
    .stNumberInput > div > div > input {
        background-color: #1e1e2e;
        color: white;
        border: 1px solid #444;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'simulator' not in st.session_state:
    st.session_state.simulator = MicrogravitySimulator()
    st.session_state.running = False
    st.session_state.last_update = time.time()
    st.session_state.update_counter = 0

# Header
st.title("🌐 RPM Digital Twin - Microgravity Simulator")
st.markdown("**Institution:** Department of Aerospace Engineering, IIST | **Version:** 3.1.1 | **Based on:** Yotov et al. Research")
st.divider()

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Simulation Controls")
    
    # Simulation control buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start", use_container_width=True, type="primary"):
            st.session_state.running = True
            st.rerun()
    with col2:
        if st.button("⏸️ Stop", use_container_width=True):
            st.session_state.running = False
            st.rerun()
    
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.simulator.reset()
        st.session_state.running = False
        st.session_state.update_counter = 0
        st.rerun()
    
    st.divider()
    
    # Motor Settings
    st.subheader("🔧 Motor Settings")
    col1, col2 = st.columns(2)
    with col1:
        inner_rpm = st.number_input(
            "Inner Frame (RPM)", 
            min_value=0.0, max_value=20.0, value=2.0, step=0.1,
            key="inner_rpm"
        )
    with col2:
        outer_rpm = st.number_input(
            "Outer Frame (RPM)", 
            min_value=0.0, max_value=20.0, value=2.0, step=0.1,
            key="outer_rpm"
        )
    
    # Update simulator velocities
    st.session_state.simulator.set_velocities(inner_rpm, outer_rpm)
    
    # Frame Dimensions
    st.subheader("📐 Frame Dimensions (m)")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Inner Frame:**")
        inner_length = st.number_input("Length", min_value=0.1, max_value=1.0, value=0.30, step=0.01, key="inner_l")
        inner_breadth = st.number_input("Breadth", min_value=0.1, max_value=1.0, value=0.20, step=0.01, key="inner_b")
    
    with col2:
        st.markdown("**Outer Frame:**")
        outer_length = st.number_input("Length", min_value=0.1, max_value=1.0, value=0.50, step=0.01, key="outer_l")
        outer_breadth = st.number_input("Breadth", min_value=0.1, max_value=1.0, value=0.35, step=0.01, key="outer_b")
    
    # Update frame dimensions
    st.session_state.simulator.set_frame_dimensions(inner_length, inner_breadth, outer_length, outer_breadth)
    
    st.divider()
    
    # Axis Inclination
    st.subheader("📊 Axis Inclination")
    inner_tilt = st.slider("Inner Axis Tilt (°)", -30.0, 30.0, 0.0, 0.5)
    outer_tilt = st.slider("Outer Axis Tilt (°)", -30.0, 30.0, 0.0, 0.5)
    
    # Update axis inclination
    st.session_state.simulator.set_axis_inclination(inner_tilt, outer_tilt)
    
    st.divider()
    
    # Simulation Parameters
    st.subheader("⚡ Update Rate")
    update_rate = st.select_slider("Hz", options=[10, 20, 30, 50], value=20, key="update_rate")
    
    st.divider()
    
    # Info
    st.subheader("ℹ️ About")
    st.markdown("""
    **RPM Digital Twin** simulates microgravity using the exact physics from:
    
    *"A New Random Positioning Machine Modification Applied for Microgravity Simulation in Laboratory Experiments with Rats"* - Yotov et al.
    
    Key metrics:
    - **taSMG**: Time-averaged specific microgravity
    - **Samples**: Convergence indicator  
    - **γ (Gamma)**: Velocity ratio ω_φ/ω_ψ
    """)

# Auto-update mechanism
if st.session_state.running:
    dt = 1.0 / update_rate
    current_time = time.time()
    if current_time - st.session_state.last_update >= dt:
        # Step the physics simulation
        state = st.session_state.simulator.step(dt)
        st.session_state.last_update = current_time
        st.session_state.update_counter += 1
        
        # Auto-rerun to update display
        if st.session_state.update_counter % (update_rate // 4) == 0:  # Update display 4 times per second
            st.rerun()

# Get current simulation state
sim_state = st.session_state.simulator.step(0.0)  # Get state without advancing

# Main Content Area
tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Dashboard", "🧮 Physics Engine", "📈 Analysis", "⚙️ Settings"])

# TAB 1: LIVE DASHBOARD
with tab1:
    # Status indicator
    status_color = "🟢" if st.session_state.running else "🔴"
    st.markdown(f"**Status:** {status_color} {'Running' if st.session_state.running else 'Stopped'}")
    
    # Main metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "taSMG (G*)", 
            f"{sim_state['mean_g']:.5f}g",
            delta=f"{sim_state['mean_g'] - sim_state['min_g']:.5f}" if sim_state['n_samples'] > 50 else None
        )
    with col2:
        st.metric("Samples", f"{sim_state['n_samples']:,}", "↑ converging" if st.session_state.running else "stopped")
    with col3:
        st.metric("Inner RPM", f"{sim_state['omega_phi'] * 30/np.pi:.2f}", f"Target: {inner_rpm:.1f}")
    with col4:
        st.metric("Outer RPM", f"{sim_state['omega_psi'] * 30/np.pi:.2f}", f"Target: {outer_rpm:.1f}")
    with col5:
        gamma = sim_state['gamma']
        gamma_str = f"{gamma:.3f}" if abs(gamma) < 1000 else "∞"
        st.metric("γ (Gamma)", gamma_str, "Velocity ratio")
    
    st.divider()
    
    # Visualization row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌍 Gravity Vector Components")
        
        # Real-time gravity components
        gx, gy, gz = sim_state['gravity_sample']
        avg_gx, avg_gy, avg_gz = sim_state['avg_gravity']
        
        # Create time series data (simulate history)
        if 'gravity_history' not in st.session_state:
            st.session_state.gravity_history = {'t': [], 'gx': [], 'gy': [], 'gz': [], 'avg_gx': [], 'avg_gy': [], 'avg_gz': []}
        
        # Add current data point
        current_time = sim_state['time']
        history = st.session_state.gravity_history
        history['t'].append(current_time)
        history['gx'].append(gx)
        history['gy'].append(gy)
        history['gz'].append(gz)
        history['avg_gx'].append(avg_gx)
        history['avg_gy'].append(avg_gy)
        history['avg_gz'].append(avg_gz)
        
        # Keep only last 300 points
        max_points = 300
        for key in history:
            if len(history[key]) > max_points:
                history[key] = history[key][-max_points:]
        
        if len(history['t']) > 1:
            fig_gravity = go.Figure()
            
            # Instantaneous components
            fig_gravity.add_trace(go.Scatter(x=history['t'], y=history['gx'], name='Gx (instantaneous)', line=dict(color='red', width=1)))
            fig_gravity.add_trace(go.Scatter(x=history['t'], y=history['gy'], name='Gy (instantaneous)', line=dict(color='green', width=1)))
            fig_gravity.add_trace(go.Scatter(x=history['t'], y=history['gz'], name='Gz (instantaneous)', line=dict(color='blue', width=1)))
            
            # Time-averaged components (thicker lines)
            fig_gravity.add_trace(go.Scatter(x=history['t'], y=history['avg_gx'], name='<Gx> (avg)', line=dict(color='red', width=3, dash='dash')))
            fig_gravity.add_trace(go.Scatter(x=history['t'], y=history['avg_gy'], name='<Gy> (avg)', line=dict(color='green', width=3, dash='dash')))
            fig_gravity.add_trace(go.Scatter(x=history['t'], y=history['avg_gz'], name='<Gz> (avg)', line=dict(color='blue', width=3, dash='dash')))
            
            fig_gravity.update_layout(
                title="Gravity Components vs Time",
                xaxis_title="Time (s)",
                yaxis_title="Gravity (g)",
                height=400,
                template="plotly_dark",
                showlegend=True
            )
            st.plotly_chart(fig_gravity, use_container_width=True)
        else:
            st.info("Start simulation to see gravity components")
    
    with col2:
        st.subheader("🌐 3D Gravity Trajectory")
        
        if len(history['t']) > 10:
            # 3D trajectory of gravity vector on unit sphere
            fig_3d = go.Figure()
            
            # Recent trajectory (last 100 points)
            n_recent = min(100, len(history['gx']))
            recent_gx = history['gx'][-n_recent:]
            recent_gy = history['gy'][-n_recent:]
            recent_gz = history['gz'][-n_recent:]
            
            # Color-coded by time
            fig_3d.add_trace(go.Scatter3d(
                x=recent_gx, y=recent_gy, z=recent_gz,
                mode='lines+markers',
                name='Gravity Trajectory',
                line=dict(color=np.arange(len(recent_gx)), colorscale='Viridis', width=4),
                marker=dict(size=2)
            ))
            
            # Unit sphere (gravity magnitude = 1)
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x_sphere = np.outer(np.cos(u), np.sin(v)) * 0.3
            y_sphere = np.outer(np.sin(u), np.sin(v)) * 0.3  
            z_sphere = np.outer(np.ones(np.size(u)), np.cos(v)) * 0.3
            
            fig_3d.add_trace(go.Surface(
                x=x_sphere, y=y_sphere, z=z_sphere,
                opacity=0.1, colorscale='Blues', showscale=False, name='Unit Sphere'
            ))
            
            # Current position
            fig_3d.add_trace(go.Scatter3d(
                x=[gx], y=[gy], z=[gz],
                mode='markers',
                name='Current',
                marker=dict(size=8, color='red')
            ))
            
            # Average position
            fig_3d.add_trace(go.Scatter3d(
                x=[avg_gx], y=[avg_gy], z=[avg_gz],
                mode='markers',
                name=f'Average (taSMG={sim_state["mean_g"]:.5f})',
                marker=dict(size=10, color='yellow', symbol='diamond')
            ))
            
            fig_3d.update_layout(
                title="Gravity Vector on Unit Sphere",
                scene=dict(
                    xaxis_title="Gx",
                    yaxis_title="Gy", 
                    zaxis_title="Gz",
                    aspectmode="cube",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                height=400,
                template="plotly_dark"
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.info("Start simulation to see 3D trajectory")
    
    st.divider()
    
    # Bottom metrics row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 taSMG Convergence")
        
        if 'tasmg_history' not in st.session_state:
            st.session_state.tasmg_history = {'samples': [], 'tasmg': []}
        
        # Add current taSMG value
        tasmg_hist = st.session_state.tasmg_history
        tasmg_hist['samples'].append(sim_state['n_samples'])
        tasmg_hist['tasmg'].append(sim_state['mean_g'])
        
        # Keep reasonable history
        if len(tasmg_hist['samples']) > 500:
            tasmg_hist['samples'] = tasmg_hist['samples'][-500:]
            tasmg_hist['tasmg'] = tasmg_hist['tasmg'][-500:]
        
        if len(tasmg_hist['samples']) > 5:
            fig_tasmg = go.Figure()
            fig_tasmg.add_trace(go.Scatter(
                x=tasmg_hist['samples'], y=tasmg_hist['tasmg'],
                mode='lines', name='taSMG',
                line=dict(color='cyan', width=2)
            ))
            
            # Quality thresholds
            fig_tasmg.add_hline(y=0.003, line_dash="dash", line_color="green", annotation_text="Excellent (<0.003g)")
            fig_tasmg.add_hline(y=0.01, line_dash="dash", line_color="yellow", annotation_text="Good (<0.01g)")
            fig_tasmg.add_hline(y=0.03, line_dash="dash", line_color="red", annotation_text="Poor (>0.03g)")
            
            fig_tasmg.update_layout(
                title="taSMG Convergence vs Samples",
                xaxis_title="Samples",
                yaxis_title="taSMG (g)",
                height=300,
                template="plotly_dark"
            )
            st.plotly_chart(fig_tasmg, use_container_width=True)
        else:
            st.info("Collecting samples for convergence analysis...")
    
    with col2:
        st.subheader("📋 Current State")
        
        # Physics state display
        st.markdown(f"""
        **Frame Angles:**
        - φ (Inner): {np.degrees(sim_state['phi']):.1f}°
        - ψ (Outer): {np.degrees(sim_state['psi']):.1f}°
        
        **Velocities:**
        - ω_φ: {sim_state['omega_phi']:.3f} rad/s
        - ω_ψ: {sim_state['omega_psi']:.3f} rad/s
        - Ratio γ: {gamma_str}
        
        **Gravity (Instantaneous):**
        - Gx: {gx:.4f}
        - Gy: {gy:.4f}  
        - Gz: {gz:.4f}
        
        **Time-Averaged Gravity:**
        - <Gx>: {avg_gx:.6f}
        - <Gy>: {avg_gy:.6f}
        - <Gz>: {avg_gz:.6f}
        
        **Quality Metrics:**
        - taSMG: {sim_state['mean_g']:.6f}g
        - Min: {sim_state['min_g']:.6f}g
        - Max: {sim_state['max_g']:.6f}g
        - Runtime: {sim_state['time']:.1f}s
        """)

# TAB 2: PHYSICS ENGINE
with tab2:
    st.header("🧮 Physics Engine - Research Implementation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📐 Rotation Matrices (Eq. 1)")
        
        # Show actual rotation matrices
        phi = sim_state['phi'] 
        psi = sim_state['psi']
        
        R_Y = RotationMatrices.R_Y(psi)
        R_Z = RotationMatrices.R_Z(phi)
        
        st.markdown("**R_Y(ψ) - Outer Frame:**")
        st.code(f"""
[ {R_Y[0,0]:6.3f}  {R_Y[0,1]:6.3f}  {R_Y[0,2]:6.3f} ]
[ {R_Y[1,0]:6.3f}  {R_Y[1,1]:6.3f}  {R_Y[1,2]:6.3f} ]
[ {R_Y[2,0]:6.3f}  {R_Y[2,1]:6.3f}  {R_Y[2,2]:6.3f} ]
        """)
        
        st.markdown("**R_Z(φ) - Inner Frame:**")
        st.code(f"""
[ {R_Z[0,0]:6.3f}  {R_Z[0,1]:6.3f}  {R_Z[0,2]:6.3f} ]
[ {R_Z[1,0]:6.3f}  {R_Z[1,1]:6.3f}  {R_Z[1,2]:6.3f} ]
[ {R_Z[2,0]:6.3f}  {R_Z[2,1]:6.3f}  {R_Z[2,2]:6.3f} ]
        """)
    
    with col2:
        st.subheader("📊 Key Equations")
        
        st.markdown("""
        **Gravity Vector (Eq. 2):**
        ```
        ē_g = -sin(φ)cos(ψ)ē_x + sin(φ)sin(ψ)ē_y + cos(φ)ē_z
        ```
        
        **Velocity Ratio (Eq. 5):**
        ```
        γ = ω_φ / ω_ψ
        ```
        
        **Time-Averaged Microgravity:**
        ```
        taSMG = ||<ē_g>|| = ||∑ē_g / N||
        ```
        
        **Current Values:**
        """)
        
        st.code(f"""
φ = {phi:.4f} rad = {np.degrees(phi):6.1f}°
ψ = {psi:.4f} rad = {np.degrees(psi):6.1f}°
ω_φ = {sim_state['omega_phi']:6.3f} rad/s
ω_ψ = {sim_state['omega_psi']:6.3f} rad/s
γ = {gamma_str}
        """)
    
    st.divider()
    
    st.subheader("🔬 Research Reference")
    st.info("""
    **Paper:** "A New Random Positioning Machine Modification Applied for Microgravity Simulation in Laboratory Experiments with Rats"
    
    **Authors:** Yotov et al.
    
    **Implementation:** This simulator uses the exact equations from the research paper, ensuring scientific accuracy for microgravity research applications.
    
    **Key Insight:** True microgravity is achieved when the time-averaged gravity vector approaches zero, not when instantaneous gravity is minimized.
    """)

# TAB 3: ANALYSIS  
with tab3:
    st.header("📈 Data Analysis & Export")
    
    # Data export section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Data Export")
        
        if st.button("📊 Export Current Session", type="primary"):
            # Prepare data for export
            if len(history['t']) > 0:
                import pandas as pd
                
                df = pd.DataFrame({
                    'time_s': history['t'],
                    'phi_rad': [sim_state['phi']] * len(history['t']),
                    'psi_rad': [sim_state['psi']] * len(history['t']),
                    'gx_instant': history['gx'],
                    'gy_instant': history['gy'],
                    'gz_instant': history['gz'],
                    'gx_avg': history['avg_gx'],
                    'gy_avg': history['avg_gy'],
                    'gz_avg': history['avg_gz'],
                    'tasmg': tasmg_hist['tasmg'] if 'tasmg_history' in st.session_state else [sim_state['mean_g']] * len(history['t'])
                })
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"rpm_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime='text/csv'
                )
                
                st.success(f"Ready to export {len(df)} data points")
            else:
                st.warning("No data to export. Start simulation first.")
    
    with col2:
        st.subheader("📈 Quality Assessment")
        
        if sim_state['n_samples'] > 0:
            # Quality indicators
            tasmg = sim_state['mean_g']
            
            if tasmg < 0.003:
                quality = "🟢 Excellent"
                quality_desc = "Suitable for sensitive microgravity experiments"
            elif tasmg < 0.01:
                quality = "🟡 Good"  
                quality_desc = "Good for general microgravity research"
            elif tasmg < 0.03:
                quality = "🟠 Fair"
                quality_desc = "Marginal microgravity quality"
            else:
                quality = "🔴 Poor"
                quality_desc = "Insufficient microgravity quality"
            
            st.metric("Quality Rating", quality)
            st.write(quality_desc)
            
            # Convergence status
            if sim_state['n_samples'] < 50:
                conv_status = "🟡 Converging"
                conv_desc = "Still collecting samples"
            elif sim_state['n_samples'] < 200:
                conv_status = "🟢 Good"
                conv_desc = "Sufficient samples for analysis"
            else:
                conv_status = "🟢 Excellent"
                conv_desc = "High-confidence results"
            
            st.metric("Convergence", conv_status)
            st.write(conv_desc)
        else:
            st.info("Start simulation to see quality metrics")

# TAB 4: SETTINGS
with tab4:
    st.header("⚙️ Application Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 Simulation Parameters")
        
        st.markdown(f"""
        **Current Configuration:**
        - Inner RPM: {inner_rpm:.1f}
        - Outer RPM: {outer_rpm:.1f}
        - Update Rate: {update_rate} Hz
        - Frame Dims: {inner_length:.2f}×{inner_breadth:.2f} / {outer_length:.2f}×{outer_breadth:.2f} m
        - Axis Tilts: {inner_tilt:.1f}° / {outer_tilt:.1f}°
        
        **Physics Constants:**
        - Max History: {st.session_state.simulator.max_history} samples
        - Earth Gravity: 9.80665 m/s²
        - Motor Time Constant: 0.3s
        """)
        
        if st.button("🗑️ Clear All History"):
            st.session_state.gravity_history = {'t': [], 'gx': [], 'gy': [], 'gz': [], 'avg_gx': [], 'avg_gy': [], 'avg_gz': []}
            st.session_state.tasmg_history = {'samples': [], 'tasmg': []}
            st.success("History cleared")
            st.rerun()
    
    with col2:
        st.subheader("📱 App Information")
        
        st.info(f"""
        **RPM Digital Twin v3.1.1**
        
        Department of Aerospace Engineering, IIST
        
        **Status:** ✅ Operational
        
        **Features:**
        - Real-time physics simulation
        - Research-accurate equations
        - 3D visualization
        - Live metrics dashboard
        - Data export capabilities
        
        **Runtime:** {sim_state['time']:.1f}s
        **Samples:** {sim_state['n_samples']:,}
        **Update Counter:** {st.session_state.update_counter:,}
        """)
        
        st.markdown("""
        **License:** MIT  
        [🔗 GitHub Repository](https://github.com/Pranay1004/random-positioning-machine-iist)
        """)

# Footer
st.divider()
st.markdown("""
---
**RPM Digital Twin Simulator** © 2026 Department of Aerospace Engineering, IIST  
**License:** MIT | **Version:** 3.1.1 | Based on: *Yotov et al.* research

*Built with Streamlit · Powered by NumPy & Plotly · Real-time Physics Engine*
""")

# Auto-refresh display if running
if st.session_state.running:
    time.sleep(0.1)  # Small delay for smooth animation
    st.rerun()