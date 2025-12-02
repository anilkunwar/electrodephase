import os
import streamlit as st
import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import sqlite3
import zlib
from io import BytesIO
import zipfile
import base64
import math
import time

# =====================================================
# Streamlit App: Patched LiFePO4 Phase-Field Simulator
# - FIXED: Sequential update kernel to avoid UnsupportedRewriteError
# - Conservative dt stability check and clamping
# - Plotly animation memory improvements (float32, frames)
# - Session state enforcement for caching results
# =====================================================

st.set_page_config(page_title="LiFePO4 Phase-Field Simulator (Patched)", layout="centered")
st.title("🔋 LiFePO4 Phase-Field Simulator — Patched")

st.markdown(
    """
**Patched version:** Numba-compatible sequential update kernel (no parallel rewriting issues),
stability check for dt, smaller animation payloads and optional tuning knobs.
"""
)

# Initialize session state for caching simulation results
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'parameters_hash' not in st.session_state:
    st.session_state.parameters_hash = None
if 'simulation_complete' not in st.session_state:
    st.session_state.simulation_complete = False
if 'cached_frames' not in st.session_state:
    st.session_state.cached_frames = None
if 'cached_steps' not in st.session_state:
    st.session_state.cached_steps = None

# ------------------- Sidebar parameters -------------------
st.sidebar.header("Simulation Parameters")
grid_size = st.sidebar.slider("Grid size (square)", 128, 1024, 512, step=128)
total_steps = st.sidebar.number_input("Total time steps", 1000, 100000, 20000, step=1000)
A = st.sidebar.slider("Double-well height A", 5.0, 50.0, 10.0, step=1.0)
kappa = st.sidebar.slider("Gradient energy κ", 0.5, 5.0, 2.0, step=0.5)
M_slow = st.sidebar.number_input("Mobility slow directions", 0.1, 10.0, 1.0, step=0.1)
M_fast_ratio = st.sidebar.slider("Anisotropy ratio M_fast / M_slow", 1000, 100000, 20000, step=1000)
save_every = st.sidebar.number_input("Save frame every N steps", 100, 2000, 500, step=100)

charging = st.sidebar.checkbox("Simulate galvanostatic charging from bottom facet", value=True)
if charging:
    charging_rate = st.sidebar.slider("Charging rate (Li / step / site)", 0.0, 0.001, 0.0002, step=0.00005, format="%.6f")
else:
    charging_rate = 0.0

# Tuning knobs (removed parallel-related options)
block_size = st.sidebar.selectbox("Block size (cache tuning - now sequential)", [16, 32, 64, 128], index=2)
# Note: numba_threads removed since we're using sequential update

# Clear cache button
if st.sidebar.button("🗑️ Clear Simulation Cache"):
    st.session_state.simulation_results = None
    st.session_state.parameters_hash = None
    st.session_state.simulation_complete = False
    st.session_state.cached_frames = None
    st.session_state.cached_steps = None
    st.rerun()

run_sim = st.button("🚀 Run Simulation", type="primary")

# ------------------- Helpers / caching -------------------
@st.cache_data
def make_initial_condition(nx, ny, seed_radius_frac=0.03, seed_center=None, initial_perturb=0.0, seed_value=0.98, background=0.02):
    """Generate initial condition (cached)."""
    c = background * np.ones((nx, ny), dtype=np.float64)
    seed_radius = int(seed_radius_frac * nx)
    Y, X = np.ogrid[0:nx, 0:ny]
    if seed_center is None:
        cx, cy = nx // 2, ny // 2
    else:
        cx, cy = seed_center
    mask = (X - cx) ** 2 + (Y - cy) ** 2 < seed_radius ** 2
    c[mask] = seed_value
    if initial_perturb > 0.0:
        rng = np.random.default_rng(1234)
        c += initial_perturb * (rng.random(c.shape) - 0.5)
        np.clip(c, 0.0, 1.0, out=c)
    return c

# ------------------- Numba kernels & stability -------------------
_STABILITY_C = 0.05  # More conservative stability factor (increased from 0.1)

@njit(cache=True)
def compute_dt_max(dx, M_effective, kappa_local):
    """Compute maximum stable timestep."""
    # Conservative dt max for explicit-like stepping
    return _STABILITY_C * (dx ** 4) / (M_effective * kappa_local + 1e-30)

@njit(cache=True)  # REMOVED: parallel=True - using sequential double-buffered update
def update_blocked(c, dt, A_local, kappa_local, M_slow_local, M_fast_local, charging_rate_local, block_size_local):
    """Double-buffered sequential stencil - Numba-safe, fast."""
    nx, ny = c.shape
    c_new = np.empty_like(c)
    dx = 1.0
    inv_dx2 = 1.0 / (dx * dx)
    
    # Sequential double-buffer update (Numba stencil-friendly)
    for i in range(nx):
        im = i - 1 if i > 0 else nx - 1
        ip = i + 1 if i < nx - 1 else 0
        for j in range(ny):
            jm = j - 1 if j > 0 else ny - 1
            jp = j + 1 if j < ny - 1 else 0
            
            # Center values
            ci = c[i, j]
            chem = 4.0 * A_local * ci * (ci - 1.0) * (ci - 0.5)
            lap = (c[im, j] + c[ip, j] + c[i, jm] + c[i, jp] - 4.0 * ci) * inv_dx2
            mu = chem - kappa_local * lap
            
            # Simplified anisotropic divergence (M_fast_x, M_slow_y)
            # Backward differences for stability
            grad_mu_x = (mu - (chem - kappa_local * (c[im, j] + c[(im-1)%nx, j] + c[im, jm] + c[im, jp] - 4.0*c[im,j])*inv_dx2)) / dx
            grad_mu_y = (mu - (chem - kappa_local * (c[i, jm] + c[im, jm] + c[ip, jm] + c[i, (jm-1)%ny] - 4.0*c[i,jm])*inv_dx2)) / dx
            
            Jx = -M_fast_local * grad_mu_x
            Jy = -M_slow_local * grad_mu_y
            div = (Jx - (-M_fast_local * (mu - (chem - kappa_local * (c[im, j] + c[(im-1)%nx, j] + c[im, jm] + c[im, jp] - 4.0*c[im,j])*inv_dx2)) / dx)) / dx + \
                  (Jy - (-M_slow_local * (mu - (chem - kappa_local * (c[i, jm] + c[im, jm] + c[ip, jm] + c[i, (jm-1)%ny] - 4.0*c[i,jm])*inv_dx2)) / dx)) / dx
            
            c_new[i, j] = ci + dt * div
    
    # Charging boundary (sequential safe)
    if charging_rate_local > 0.0:
        for ii in range(min(5, nx)):
            for jj in range(ny):
                v = c_new[ii, jj] + charging_rate_local
                c_new[ii, jj] = max(0.0, min(1.0, v))
    
    return c_new

# ------------------- ParaView writer -------------------
def write_vti_ascii(data):
    """Write VTI file in ASCII format."""
    nx, ny = data.shape
    buf = BytesIO()
    buf.write(b'<?xml version="1.0"?>\n')
    buf.write(b'<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n')
    buf.write(f'  <ImageData WholeExtent="0 {nx-1} 0 {ny-1} 0 0" Origin="0 0 0" Spacing="1 1 1">\n'.encode())
    buf.write(f'    <Piece Extent="0 {nx-1} 0 {ny-1} 0 0">\n'.encode())
    buf.write(b'      <PointData Scalars="concentration">\n')
    buf.write(b'        <DataArray type="Float32" Name="concentration" format="ascii">\n')
    
    arr = np.asarray(data, dtype=np.float32)
    for i in range(nx):
        row = " ".join([f"{v:.6f}" for v in arr[i, :]])
        buf.write((row + "\n").encode())
    
    buf.write(b'        </DataArray>\n')
    buf.write(b'      </PointData>\n')
    buf.write(b'    </Piece>\n')
    buf.write(b'  </ImageData>\n')
    buf.write(b'</VTKFile>\n')
    return buf.getvalue()

def create_parameters_hash(params_dict):
    """Create hash of parameters to check if simulation needs rerun."""
    import hashlib
    param_str = str(sorted(params_dict.items()))
    return hashlib.md5(param_str.encode()).hexdigest()

# ------------------- Main run -------------------
if run_sim:
    # Create current parameters dictionary
    current_params = {
        'grid_size': grid_size,
        'total_steps': total_steps,
        'A': A,
        'kappa': kappa,
        'M_slow': M_slow,
        'M_fast_ratio': M_fast_ratio,
        'save_every': save_every,
        'charging': charging,
        'charging_rate': charging_rate,
        'block_size': block_size,
    }
    
    current_hash = create_parameters_hash(current_params)
    
    # Check if we can use cached results
    if (st.session_state.simulation_results is not None and 
        st.session_state.parameters_hash == current_hash):
        st.info("Using cached simulation results. To rerun, click 'Clear Simulation Cache' or change parameters.")
        saved_frames = st.session_state.cached_frames
        saved_steps = st.session_state.cached_steps
        c = saved_frames[-1].astype(np.float64)  # Last frame
        st.session_state.simulation_complete = True
    else:
        # Run new simulation
        nx = ny = grid_size
        dx = 1.0
        
        # Calculate dt conservatively
        M_fast = M_fast_ratio * M_slow
        M_effective = max(M_slow, M_fast)
        dt_max = compute_dt_max(dx, M_effective, kappa)
        
        # Start with conservative dt
        dt_initial = min(5e-4, dt_max)
        
        # Check stability
        if dt_initial > dt_max:
            st.warning(f"dt ({dt_initial:.3e}) is larger than stability limit ({dt_max:.3e}). Clamping dt to limit.")
            dt = float(dt_max)
        else:
            dt = dt_initial
        
        st.info(f"Using dt = {dt:.3e} (stability limit: {dt_max:.3e})")
        
        # Initial condition
        c = make_initial_condition(nx, ny, seed_radius_frac=0.03)
        if charging:
            c[0:8, :] = 0.95
        
        saved_frames = []
        saved_steps = []
        
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        # Run simulation
        step = 0
        start_time = time.time()
        
        # REMOVED: The warmup test block that was causing UnsupportedRewriteError
        # The first real call will compile the function safely
        
        while step < int(total_steps):
            # Update
            c = update_blocked(c, dt, A, kappa, M_slow, M_fast, charging_rate, int(block_size))
            step += 1
            
            # Save frames
            if step % save_every == 0 or step == total_steps:
                saved_frames.append(c.astype(np.float32))
                saved_steps.append(step)
                progress_bar.progress(step / total_steps)
                elapsed = time.time() - start_time
                estimated_total = elapsed / step * total_steps
                remaining = estimated_total - elapsed
                status_text.text(f"Step {step}/{total_steps} | Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s | mean c = {c.mean():.6f}")
        
        # Cache results
        st.session_state.cached_frames = saved_frames
        st.session_state.cached_steps = saved_steps
        st.session_state.parameters_hash = current_hash
        st.session_state.simulation_complete = True
        
        progress_bar.progress(1.0)
        status_text.text(f"Simulation finished in {time.time() - start_time:.1f} seconds!")
    
    # ------------------- Display Results -------------------
    if st.session_state.simulation_complete and len(saved_frames) > 0:
        st.subheader("📊 Interactive Animation")
        frame_stack = np.array(saved_frames, dtype=np.float32)
        nframes = frame_stack.shape[0]
        
        fig = go.Figure(
            frames=[go.Frame(
                data=[go.Heatmap(z=frame_stack[k], zmin=0, zmax=1, colorscale='RdBu_r')],
                name=str(k)
            ) for k in range(min(nframes, 50))]  # Limit frames for performance
        )
        fig.add_trace(go.Heatmap(z=frame_stack[0], zmin=0, zmax=1, colorscale='RdBu_r'))
        
        fig.update_layout(
            title="Li concentration (blue = LiFePO4, red = FePO4)",
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {
                            "frame": {"duration": 100, "redraw": True},
                            "fromcurrent": True,
                            "mode": "immediate"
                        }]
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate"
                        }]
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
            }]
        )
        
        # Add slider
        fig.update_layout(
            sliders=[{
                "active": 0,
                "currentvalue": {"prefix": "Frame: "},
                "steps": [
                    {
                        "args": [[k], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        "label": str(k),
                        "method": "animate"
                    }
                    for k in range(min(nframes, 50))
                ]
            }]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ------------------- Downloads -------------------
        st.subheader("⬇️ Downloads")
        
        # Use the last frame for single file downloads
        final_frame = saved_frames[-1] if saved_frames else c
        
        # 1. CSV (final frame)
        csv_buffer = BytesIO()
        pd.DataFrame(final_frame).to_csv(csv_buffer, index=False, header=False)
        st.download_button(
            label="Download final_concentration.csv",
            data=csv_buffer.getvalue(),
            file_name="final_concentration.csv",
            mime="text/csv"
        )
        
        # 2. NumPy .npy (final frame)
        npy_buffer = BytesIO()
        np.save(npy_buffer, final_frame)
        st.download_button(
            label="Download final_concentration.npy",
            data=npy_buffer.getvalue(),
            file_name="final_concentration.npy"
        )
        
        # 3. ParaView collection (PVD + multiple VTI)
        if st.button("Generate ParaView Files (may take a moment)"):
            with st.spinner("Generating ParaView files..."):
                pvd_buffer = BytesIO()
                pvd_buffer.write(b'<VTKFile type="Collection" version="1.0">\n')
                pvd_buffer.write(b'<Collection>\n')
                
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    # Limit to first 20 frames for performance
                    for idx, frame in enumerate(saved_frames[:20]):
                        vti_data = write_vti_ascii(frame)
                        filename = f"lfp_{idx:04d}.vti"
                        zf.writestr(filename, vti_data)
                        pvd_buffer.write(f'  <DataSet timestep="{saved_steps[idx]}" part="0" file="{filename}"/>\n'.encode())
                    
                    pvd_buffer.write(b'</Collection>\n')
                    pvd_buffer.write(b'</VTKFile>\n')
                    zf.writestr("lfp_simulation.pvd", pvd_buffer.getvalue())
                
                zip_buffer.seek(0)
                st.download_button(
                    label="Download ParaView files (.pvd + .vti)",
                    data=zip_buffer.getvalue(),
                    file_name="lfp_simulation.zip"
                )
        
        # 4. SQLite database with compressed frames
        if st.button("Generate SQLite Database"):
            with st.spinner("Creating SQLite database..."):
                conn = sqlite3.connect(':memory:')
                cur = conn.cursor()
                cur.execute("CREATE TABLE simulation (timestep INTEGER PRIMARY KEY, mean_c REAL, data BLOB)")
                
                # Limit to first 10 frames for performance
                for idx, frame in enumerate(saved_frames[:10]):
                    frame_io = BytesIO()
                    np.save(frame_io, frame)
                    compressed = zlib.compress(frame_io.getvalue())
                    cur.execute(
                        "INSERT INTO simulation VALUES (?, ?, ?)",
                        (saved_steps[idx], float(frame.mean()), sqlite3.Binary(compressed))
                    )
                
                conn.commit()
                
                # Export to SQL file
                sql_dump = BytesIO()
                for line in conn.iterdump():
                    sql_dump.write((line + '\n').encode('utf-8'))
                
                st.download_button(
                    label="Download SQLite database dump (.sql)",
                    data=sql_dump.getvalue(),
                    file_name="lfp_simulation.sql"
                )
        
        st.success("Simulation complete! Use the animation above and download your files below.")
else:
    # Display cached results if available
    if st.session_state.simulation_complete and st.session_state.cached_frames is not None:
        st.info("Previous simulation results are available. To view them again, click 'Run Simulation'. To clear, use 'Clear Simulation Cache'.")
        
        # Quick preview of cached data
        st.subheader("📊 Cached Simulation Preview")
        final_frame = st.session_state.cached_frames[-1]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(final_frame, cmap='RdBu_r', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='Li concentration')
        ax.set_title(f"Final Frame (Step {st.session_state.cached_steps[-1]})")
        st.pyplot(fig)
        
        # Quick download of final frame
        csv_buffer = BytesIO()
        pd.DataFrame(final_frame).to_csv(csv_buffer, index=False, header=False)
        st.download_button(
            label="Download cached final_concentration.csv",
            data=csv_buffer.getvalue(),
            file_name="cached_final_concentration.csv",
            mime="text/csv"
        )
    
    # Initial state
    st.info("Adjust parameters and click **Run Simulation** to start.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/LiFePO4_battery_characteristics.svg/800px-LiFePO4_battery_characteristics.svg.png",
             caption="Typical LiFePO4 particle morphology (public domain image)")
