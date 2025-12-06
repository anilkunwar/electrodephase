import os
import streamlit as st
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from io import BytesIO
import zipfile
import hashlib
import time

# =====================================================
# LiFePO4 Phase-Field Simulator — FINAL: Regular Solution + Elasticity + Correct Butler-Volmer
# =====================================================

st.set_page_config(page_title="LiFePO4 Phase-Field Simulator (Final Correct Version)", layout="centered")
st.title("🔋 LiFePO4 Phase-Field Simulator — Final Correct Version")

# ------------------- Session state -------------------
for key in ['cached_frames', 'cached_steps', 'parameters_hash']:
    if key not in st.session_state:
        st.session_state[key] = None

# ------------------- Sidebar -------------------
st.sidebar.header("Simulation Parameters")
grid_size = st.sidebar.slider("Grid size", 128, 1024, 512, step=128)
total_steps = st.sidebar.number_input("Total time steps", 1000, 150000, 25000, step=1000)
Omega_RT = st.sidebar.slider("Ω / RT (phase separation strength)", 8.0, 20.0, 13.0, step=0.5)
kappa = st.sidebar.slider("Gradient coefficient κ", 0.5, 4.0, 1.0, step=0.1)
M_slow = st.sidebar.number_input("Slow direction mobility", 0.1, 10.0, 1.0, step=0.1)
M_fast_ratio = st.sidebar.slider("Anisotropy ratio", 1000, 100000, 40000, step=5000)
fast_dir = st.sidebar.selectbox("Fast diffusion direction", ["Horizontal (x)", "Vertical (y)"], index=0)
save_every = st.sidebar.number_input("Save frame every N steps", 50, 2000, 500, step=50)

use_elasticity = st.sidebar.checkbox("Enable mean-field elasticity", value=True)
elastic_omega = 0.0
if use_elasticity:
    elastic_omega = st.sidebar.slider("Elastic strength Ω_el / RT", 0.0, 100.0, 40.0, step=5.0)

bc_mode = st.sidebar.radio("Boundary condition", ["Butler-Volmer (potentiostatic)", "Constant flux (galvanostatic approx)"])
flux_rows = st.sidebar.slider("Number of boundary rows for flux", 1, 50, 15, step=1,
                             help="Number of rows at bottom where flux is applied")

if bc_mode == "Butler-Volmer (potentiostatic)":
    eta_RT = st.sidebar.slider("Overpotential η / RT  (positive = charging)", -15.0, 15.0, 3.0, step=0.5)
    k_r = st.sidebar.slider("Reaction rate constant k_r", 1.0, 500.0, 80.0, step=10.0,
                            help="Low = reaction-limited, High = diffusion-limited")
    flux_rate_param = k_r * np.sinh(eta_RT * 0.5)
else:
    constant_flux = st.sidebar.slider("Constant flux rate", 0.0001, 0.02, 0.0025, format="%.6f")
    flux_rate_param = constant_flux

if st.sidebar.button("Clear Cache"):
    for key in ['cached_frames', 'cached_steps', 'parameters_hash']:
        st.session_state[key] = None
    st.rerun()

run_sim = st.button("Run Simulation", type="primary")

# ------------------- Initial condition -------------------
@st.cache_data
def make_ic(nx, ny):
    c = np.full((nx, ny), 0.02, dtype=np.float64)
    radius = int(0.08 * min(nx, ny))
    Y, X = np.ogrid[0:nx, 0:ny]
    mask = (X - nx//2)**2 + (Y - ny//2)**2 < radius**2
    c[mask] = 0.98
    return c

# ------------------- Numba kernel -------------------
_STABILITY_C = 0.22

@njit(cache=True)
def safe_log(x):
    return np.log(np.clip(x, 1e-12, 1.0 - 1e-12))

@njit(cache=True)
def update(c, dt, Omega_RT, kappa, M_x, M_y, flux_rate, elastic_omega, flux_rows):
    nx, ny = c.shape
    mu = np.empty_like(c)
    c_new = np.empty_like(c)
    inv_dx2 = 1.0
    c_mean = np.mean(c)

    # Chemical potential
    for i in range(nx):
        im = (i - 1) % nx
        ip = (i + 1) % nx
        for j in range(ny):
            jm = (j - 1) % ny
            jp = (j + 1) % ny

            ci = c[i,j]
            mu_hom = Omega_RT * (1.0 - 2.0 * ci) + safe_log(ci) - safe_log(1.0 - ci)
            lap_c = (c[im,j] + c[ip,j] + c[i,jm] + c[i,jp] - 4.0*ci) * inv_dx2
            mu[i,j] = mu_hom - kappa * lap_c + elastic_omega * (ci - c_mean)

    # Divergence
    for i in range(nx):
        im = (i - 1) % nx
        ip = (i + 1) % nx
        for j in range(ny):
            jm = (j - 1) % ny
            jp = (j + 1) % ny

            lap_mu_x = M_x * (mu[im,j] + mu[ip,j] - 2.0*mu[i,j]) * inv_dx2
            lap_mu_y = M_y * (mu[i,jm] + mu[i,jp] - 2.0*mu[i,j]) * inv_dx2

            c_new[i,j] = c[i,j] + dt * (lap_mu_x + lap_mu_y)

    # Apply flux to bottom boundary rows
    for i in range(min(flux_rows, nx)):
        for j in range(ny):
            c_new[i,j] += flux_rate * dt
    
    c_new = np.clip(c_new, 0.0, 1.0)
    return c_new

@njit(cache=True)
def get_dt(M_eff, kappa):
    return _STABILITY_C / (M_eff * kappa + 1e-30)

# ------------------- Main -------------------
if run_sim:
    # Mobilities
    if fast_dir == "Horizontal (x)":
        M_x = M_fast_ratio * M_slow
        M_y = M_slow
    else:
        M_x = M_slow
        M_y = M_fast_ratio * M_slow

    # Create parameters hash for caching
    current_params = {
        'grid_size': grid_size, 'total_steps': total_steps, 'Omega_RT': Omega_RT,
        'kappa': kappa, 'M_slow': M_slow, 'M_fast_ratio': M_fast_ratio,
        'fast_dir': fast_dir, 'save_every': save_every, 'bc_mode': bc_mode,
        'use_elasticity': use_elasticity, 'elastic_omega': elastic_omega,
        'flux_rows': flux_rows
    }
    
    # Add BC-specific parameters
    if bc_mode == "Butler-Volmer (potentiostatic)":
        current_params['eta_RT'] = eta_RT
        current_params['k_r'] = k_r
        flux_rate = flux_rate_param
    else:
        current_params['constant_flux'] = constant_flux
        flux_rate = flux_rate_param
    
    current_hash = hashlib.md5(str(sorted(current_params.items())).encode()).hexdigest()

    if st.session_state.parameters_hash == current_hash and st.session_state.cached_frames is not None:
        st.success("Cache hit – reusing results")
        frames = st.session_state.cached_frames
        steps = st.session_state.cached_steps
    else:
        c = make_ic(grid_size, grid_size)
        M_eff = max(M_x, M_y)
        dt = get_dt(M_eff, kappa)
        
        st.info(f"""
        Simulation parameters:
        - dt = {dt:.2e}
        - Ω/RT = {Omega_RT:.1f}
        - Flux rate = {flux_rate:.5f}
        - Elastic strength = {elastic_omega:.1f}
        - Fast/slow mobility ratio = {M_fast_ratio}
        """)

        frames = []
        steps = []
        progress = st.progress(0.0)
        status = st.empty()

        for step in range(1, total_steps + 1):
            c = update(c, dt, Omega_RT, kappa, M_x, M_y, flux_rate, elastic_omega, flux_rows)
            
            if step % save_every == 0 or step == total_steps:
                frames.append(c.astype(np.float32))
                steps.append(step)
                progress.progress(step / total_steps)
                if step % (save_every * 5) == 0:
                    status.text(f"Step {step:,} │ mean c = {c.mean():.4f} │ t ≈ {step*dt:.3f}")

        st.session_state.cached_frames = frames
        st.session_state.cached_steps = steps
        st.session_state.parameters_hash = current_hash
        st.balloons()

    # Visualization
    if frames:
        fig = go.Figure(
            frames=[go.Frame(
                data=go.Heatmap(
                    z=frame, 
                    zmin=0, 
                    zmax=1, 
                    colorscale='RdBu_r',
                    showscale=True
                ), 
                name=f"Step {s}"
            ) for frame, s in zip(frames, steps)]
        )
        
        fig.add_trace(go.Heatmap(
            z=frames[0], 
            zmin=0, 
            zmax=1, 
            colorscale='RdBu_r',
            colorbar=dict(title="Li concentration")
        ))
        
        fig.update_layout(
            title=f"LiₓFePO₄ – {bc_mode}",
            height=750,
            xaxis_title="y position",
            yaxis_title="x position",
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(label="Play", method="animate", args=[None, {"frame": {"duration": 80, "redraw": True}}]),
                    dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])
                ]
            )]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show final statistics
        final_frame = frames[-1]
        st.metric("Final mean concentration", f"{final_frame.mean():.4f}")
        st.metric("Min concentration", f"{final_frame.min():.4f}")
        st.metric("Max concentration", f"{final_frame.max():.4f}")
        
else:
    if st.session_state.cached_frames is not None:
        st.info("Cached result available – click 'Run Simulation' to view animation")

st.caption("Butler-Volmer kinetics with regular solution free energy. Low k_r → homogeneous filling, high k_r → sharp stripes.")
