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

# =====================================================
# Streamlit App: Patched LiFePO4 Phase-Field Simulator
# - Numba-friendly blocked + parallel update kernel
# - Conservative dt stability check and clamping
# - Plotly animation memory improvements (float32, frames)
# - st.experimental_memo used for initial-condition creation
# =====================================================

st.set_page_config(page_title="LiFePO4 Phase-Field Simulator (Patched)", layout="centered")
st.title("🔋 LiFePO4 Phase-Field Simulator — Patched")

st.markdown(
    """
**Patched version:** Numba-compatible blocked update kernel (no np.roll),
stability check for dt, smaller animation payloads and optional tuning knobs.
"""
)

# ------------------- Sidebar parameters -------------------
st.sidebar.header("Simulation Parameters")
grid_size = st.sidebar.slider("Grid size (square)", 128, 1024, 512, step=128)
total_steps = st.sidebar.number_input("Total time steps", 1000, 100000, 20000, step=1000)
A = st.sidebar.slider("Double-well height A", 5.0, 50.0, 10.0, step=1.0)  # sharper interfaces with higher A
kappa = st.sidebar.slider("Gradient energy κ", 0.5, 5.0, 2.0, step=0.5)
M_slow = st.sidebar.number_input("Mobility slow directions", 0.1, 10.0, 1.0, step=0.1)
M_fast_ratio = st.sidebar.slider("Anisotropy ratio M_fast / M_slow", 1000, 100000, 20000, step=1000)
save_every = st.sidebar.number_input("Save frame every N steps", 100, 2000, 500, step=100)

charging = st.sidebar.checkbox("Simulate galvanostatic charging from bottom facet", value=True)
charging_rate = st.sidebar.slider("Charging rate (Li / step / site)", 0.0, 0.001, 0.0002, step=0.00005, format="%.6f") if charging else 0.0

# Tuning knobs
block_size = st.sidebar.selectbox("Block size (cache tuning)", [16, 32, 64, 128], index=2)
numba_threads = st.sidebar.number_input("NUMBA_NUM_THREADS (env)", 1, os.cpu_count() or 1, os.cpu_count() or 1, step=1)

run_sim = st.button("🚀 Run Simulation", type="primary")

# Set number of threads for numba (environment variable must be set before Numba uses it)
os.environ["NUMBA_NUM_THREADS"] = str(numba_threads)

# ------------------- Helpers / caching -------------------
@st.experimental_memo
def make_initial_condition(nx, ny, seed_radius_frac=0.03, seed_center=None, initial_perturb=0.0, seed_value=0.98, background=0.02):
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
_STABILITY_C = 0.2

@njit
def compute_dt_max(dx, M_effective, kappa_local):
    # heuristic conservative dt max for explicit-like stepping
    return _STABILITY_C * (dx ** 4) / (M_effective * kappa_local + 1e-30)

@njit(parallel=True)
def update_blocked(c, dt, A_local, kappa_local, M_slow_local, M_fast_local, charging_rate_local, block_size_local):
    nx, ny = c.shape
    c_new = np.empty_like(c)

    # Loop over blocks in i-direction in parallel
    for bi in prange(0, nx, block_size_local):
        i_max = bi + block_size_local if bi + block_size_local <= nx else nx
        for bj in range(0, ny, block_size_local):
            j_max = bj + block_size_local if bj + block_size_local <= ny else ny

            # inner block loops: keep memory accesses local
            for i in range(bi, i_max):
                im = (i - 1) % nx
                ip = (i + 1) % nx
                for j in range(bj, j_max):
                    jm = (j - 1) % ny
                    jp = (j + 1) % ny

                    # chemical potential at center
                    ci = c[i, j]
                    chem = 4.0 * A_local * ci * (ci - 1.0) * (ci - 0.5)

                    # Laplacian (5-point)
                    lap = (c[im, j] + c[ip, j] + c[i, jm] + c[i, jp] - 4.0 * ci)

                    mu = chem - kappa_local * lap

                    # compute neighbor mu values (explicit neighbor evaluation)
                    cip = c[ip, j]
                    cim = c[im, j]
                    cjp = c[i, jp]
                    cjm = c[i, jm]

                    mu_ip = 4.0 * A_local * cip * (cip - 1.0) * (cip - 0.5) - kappa_local * (
                        c[(ip - 1) % nx, j] + c[(ip + 1) % nx, j] + c[ip, (j - 1) % ny] + c[ip, (j + 1) % ny] - 4.0 * cip)
                    mu_im = 4.0 * A_local * cim * (cim - 1.0) * (cim - 0.5) - kappa_local * (
                        c[(im - 1) % nx, j] + c[(im + 1) % nx, j] + c[im, (j - 1) % ny] + c[im, (j + 1) % ny] - 4.0 * cim)
                    mu_jp = 4.0 * A_local * cjp * (cjp - 1.0) * (cjp - 0.5) - kappa_local * (
                        c[(i - 1) % nx, jp] + c[(i + 1) % nx, jp] + c[i, (jp - 1) % ny] + c[i, (jp + 1) % ny] - 4.0 * cjp)
                    mu_jm = 4.0 * A_local * cjm * (cjm - 1.0) * (cjm - 0.5) - kappa_local * (
                        c[(i - 1) % nx, jm] + c[(i + 1) % nx, jm] + c[i, (jm - 1) % ny] + c[i, (jm + 1) % ny] - 4.0 * cjm)

                    grad_mu_x = (mu_ip - mu_im) * 0.5
                    grad_mu_y = (mu_jp - mu_jm) * 0.5

                    # anisotropic fluxes
                    Jx = -M_fast_local * grad_mu_x   # vertical / axis 0
                    Jy = -M_slow_local * grad_mu_y   # horizontal / axis 1

                    # central difference of fluxes (approximate divergence)
                    # divergence approximated by symmetric difference of neighbor grad_mu
                    div_x = ( (-M_fast_local * ((mu_ip - mu) * 0.5)) - (-M_fast_local * ((mu - mu_im) * 0.5)) ) * 0.5
                    div_y = ( (-M_slow_local * ((mu_jp - mu) * 0.5)) - (-M_slow_local * ((mu - mu_jm) * 0.5)) ) * 0.5

                    div = div_x + div_y

                    c_new[i, j] = ci + dt * div

    # charging boundary in a Numba-safe loop (explicit clamp)
    if charging_rate_local > 0.0:
        lim = 5 if 5 <= c_new.shape[0] else c_new.shape[0]
        for ii in range(lim):
            for jj in range(c_new.shape[1]):
                v = c_new[ii, jj] + charging_rate_local
                # clamp 0..1
                if v > 1.0:
                    v = 1.0
                elif v < 0.0:
                    v = 0.0
                c_new[ii, jj] = v

    return c_new

# ------------------- ParaView writer (fixed bytes formatting) -------------------
def write_vti_ascii(data):
    nx, ny = data.shape
    buf = BytesIO()
    buf.write(b'<?xml version="1.0"?>\n')
    buf.write(b'<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n')
    buf.write(f'  <ImageData WholeExtent="0 {nx-1} 0 {ny-1} 0 0" Origin="0 0 0" Spacing="1 1 1">\n'.encode())
    buf.write(f'    <Piece Extent="0 {nx-1} 0 {ny-1} 0 0">\n'.encode())
    buf.write(b'      <PointData Scalars="concentration">\n')
    buf.write(b'        <DataArray type="Float32" Name="concentration" format="ascii">\n')
    # write flattened rows
    # ensure data is float32 and row-major
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

# ------------------- Main run -------------------
if run_sim:
    nx = ny = grid_size
    dx = 1.0  # spatial step (grid units). Change if physical spacing differs.

    # choose dt conservatively and clamp
    dt = 5e-4 if grid_size >= 512 else 8e-4
    M_fast = M_fast_ratio * M_slow
    M_effective = max(M_slow, M_fast)
    dt_max = compute_dt_max(dx, M_effective, kappa)
    if dt > dt_max:
        st.warning(f"dt ({dt:.3e}) is larger than stability limit ({dt_max:.3e}). Clamping dt to limit.")
        dt = float(dt_max)

    # initial condition (cached)
    c = make_initial_condition(nx, ny, seed_radius_frac=0.03)
    if charging:
        c[0:8, :] = 0.95

    saved_frames = []
    saved_steps = []

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    # main time-stepping loop, save every 'save_every' steps
    step = 0
    next_save = 0
    while step < int(total_steps):
        # perform one update
        c = update_blocked(c, dt, A, kappa, M_slow, M_fast, charging_rate, int(block_size))

        step += 1

        # occasionally enforce charging (safety)
        if charging:
            lim = 5 if 5 <= c.shape[0] else c.shape[0]
            for ii in range(lim):
                for jj in range(c.shape[1]):
                    v = c[ii, jj] + charging_rate
                    if v > 1.0:
                        v = 1.0
                    c[ii, jj] = v

        # save frames
        if step % save_every == 0 or step == total_steps:
            saved_frames.append(c.astype(np.float32))  # downcast for storage/transfer
            saved_steps.append(step)
            progress_bar.progress(step / total_steps)
            status_text.text(f"Step {step}/{total_steps} | mean c = {c.mean():.6f}")

    # ensure final saved
    if len(saved_frames) == 0 or saved_steps[-1] != total_steps:
        saved_frames.append(c.astype(np.float32))
        saved_steps.append(int(total_steps))

    progress_bar.progress(1.0)
    status_text.text("Simulation finished!")

    # ------------------- Interactive animation with Plotly (frames) -------------------
    st.subheader("📊 Interactive Animation")
    frame_stack = np.array(saved_frames, dtype=np.float32)
    nframes = frame_stack.shape[0]

    fig = go.Figure(
        frames=[go.Frame(data=[go.Heatmap(z=frame_stack[k], zmin=0, zmax=1, colorscale='RdBu_r')], name=str(k)) for k in range(nframes)]
    )
    fig.add_trace(go.Heatmap(z=frame_stack[0], zmin=0, zmax=1, colorscale='RdBu_r'))
    fig.update_layout(
        title="Li concentration (blue = LiFePO4, red = FePO4)",
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]},
                {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]}
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
        }]
    )
    fig.update_layout(coloraxis_showscale=True)
    st.plotly_chart(fig, use_container_width=True)

    # ------------------- Downloads -------------------
    st.subheader("⬇️ Downloads")

    # 1. CSV (final frame)
    csv_buffer = BytesIO()
    pd.DataFrame(c).to_csv(csv_buffer, index=False, header=False)
    st.download_button("Download final_concentration.csv", csv_buffer.getvalue(), "final_concentration.csv", "text/csv")

    # 2. NumPy .npy (final frame)
    npy_buffer = BytesIO()
    np.save(npy_buffer, c)
    st.download_button("Download final_concentration.npy", npy_buffer.getvalue(), "final_concentration.npy")

    # 3. ParaView collection (PVD + multiple VTI, ascii format for no extra deps)
    pvd_buffer = BytesIO()
    pvd_buffer.write(b'<VTKFile type="Collection" version="1.0">\n')
    pvd_buffer.write(b'<Collection>\n')

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, frame in enumerate(saved_frames):
            vti_data = write_vti_ascii(frame)
            filename = f"lfp_{idx:04d}.vti"
            zf.writestr(filename, vti_data)
            pvd_buffer.write(f'  <DataSet timestep="{saved_steps[idx]}" part="0" file="{filename}"/>\n'.encode())

        pvd_buffer.write(b'</Collection>\n')
        pvd_buffer.write(b'</VTKFile>\n')
        zf.writestr("lfp_simulation.pvd", pvd_buffer.getvalue())

    zip_buffer.seek(0)
    st.download_button("Download ParaView files (.pvd + .vti)", zip_buffer.getvalue(), "lfp_simulation.zip")

    # 4. SQLite database with compressed frames
    conn = sqlite3.connect(':memory:')
    cur = conn.cursor()
    cur.execute("CREATE TABLE simulation (timestep INTEGER PRIMARY KEY, mean_c REAL, data BLOB)")

    for idx, frame in enumerate(saved_frames):
        frame_io = BytesIO()
        np.save(frame_io, frame)
        compressed = zlib.compress(frame_io.getvalue())
        cur.execute("INSERT INTO simulation VALUES (?, ?, ?)", (saved_steps[idx], float(frame.mean()), sqlite3.Binary(compressed)))

    conn.commit()

    sql_dump = BytesIO()
    for line in conn.iterdump():
        sql_dump.write((line + '\n').encode('utf-8'))

    st.download_button("Download SQLite database dump (.sql)", sql_dump.getvalue(), "lfp_simulation.sql")

    st.success("Simulation complete! Use the animation above and download your files below.")

else:
    st.info("Adjust parameters and click **Run Simulation** to start.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/LiFePO4_battery_characteristics.svg/800px-LiFePO4_battery_characteristics.svg.png", caption="Typical LiFePO4 particle morphology (public domain image)")
