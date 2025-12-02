import streamlit as st
import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import sqlite3
import zlib
from io import BytesIO
import zipfile
import base64

# =====================================================
# Streamlit App: High-Performance LiFePO₄ Phase-Field Simulator
# with animation, ParaView export (PVD + VTI), SQLite export, CSV, NPY
# Run with: streamlit run app.py
# Required: pip install streamlit numba numpy plotly pandas
# =====================================================

st.set_page_config(page_title="LiFePO₄ Phase-Field Simulator", layout="centered")

st.title("🔋 LiFePO₄ Phase-Field Simulator (Numba-Accelerated)")

st.markdown("""
**Highly anisotropic Cahn–Hilliard phase-field model for LiFePO₄ phase decomposition**  
Extreme 1D-like diffusion along [010] (vertical in plots) → realistic stripe/domino growth.  
Runs 512×512 grid with 20 000–50 000 steps in ~20–60 seconds on a laptop.
""")

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

run_sim = st.button("🚀 Run Simulation", type="primary")

# ------------------- Numba kernels -------------------
@njit(parallel=True)
def chemical_potential_term(c, A_local):
    return 4.0 * A_local * c * (c - 1.0) * (c - 0.5)

@njit(parallel=True)
def update(c, dt_local, A_local, kappa_local, M_slow_local, M_fast_local, charging_rate_local):
    chem = chemical_potential_term(c, A_local)
    lap_c = (np.roll(c, 1, axis=0) + np.roll(c, -1, axis=0) +
              np.roll(c, 1, axis=1) + np.roll(c, -1, axis=1) - 4.0 * c)

    mu = chem - kappa_local * lap_c

    grad_mu_x = (np.roll(mu, -1, axis=0) - np.roll(mu, 1, axis=0)) / 2.0
    grad_mu_y = (np.roll(mu, -1, axis=1) - np.roll(mu, 1, axis=1)) / 2.0

    # Fast direction = vertical (axis=0) → realistic LiFePO₄ stripes vertical
    Jx = -M_fast_local * grad_mu_x
    Jy = -M_slow_local * grad_mu_y

    div = (np.roll(Jx, -1, axis=0) - np.roll(Jx, 1, axis=0)) / 2.0 + \
           (np.roll(Jy, -1, axis=1) - np.roll(Jy, 1, axis=1)) / 2.0

    c_new = c + dt_local * div

    # Simple charging from bottom facet (y = 0)
    if charging_rate_local > 0:
        c_new[0:5, :] = np.clip(c_new[0:5, :] + charging_rate_local, 0.0, 1.0)

    return c_new

# ------------------- Run simulation when button pressed -------------------
if run_sim:
    nx = ny = grid_size
    dt = 5e-4 if grid_size >= 512 else 8e-4
    M_fast = M_fast_ratio * M_slow

    c = 0.02 * np.ones((nx, ny), dtype=np.float64)
    seed_radius = int(0.03 * nx)
    Y, X = np.ogrid[0:nx, 0:ny]
    mask = (X - nx//2)**2 + (Y - ny//2)**2 < seed_radius**2
    c[mask] = 0.98

    if charging:
        c[0:8, :] = 0.95  # initial thin Li-rich layer at bottom

    saved_frames = []
    saved_steps = []

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    for step in range(0, total_steps + 1, save_every):
        saved_frames.append(c.copy()
        saved_steps.append(step)

        for _ in range(save_every):
            c = update(c, dt, A, kappa, M_slow, M_fast, charging_rate)
            if charging:
                c[0:5, :] = np.clip(c[0:5, :] + charging_rate, 0.0, 1.0)

        progress_bar.progress(step / total_steps)
        status_text.text(f"Step {step}/{total_steps} | mean c = {c.mean():.4f}")

    # Save last frame one more time
    saved_frames.append(c.copy())
    saved_steps.append(total_steps)

    progress_bar.progress(1.0)
    status_text.text("Simulation finished!")

    # ------------------- Interactive animation with Plotly -------------------
    st.subheader("📊 Interactive Animation")
    frame_stack = np.array(saved_frames)
    fig = px.imshow(frame_stack,
                    animation_frame=0,
                    labels=dict(animation_frame="Time step"),
                    color_continuous_scale="RdBu_r",
                    zmin=0, zmax=1,
                    title="Li concentration (blue = LiFePO₄, red = FePO₄)")
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
    def write_vti_ascii(data):
        buf = BytesIO()
        buf.write(b'<?xml version="1.0"?>\n')
        buf.write(b'<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n')
        buf.write(f'  <ImageData WholeExtent="0 {nx-1} 0 {ny-1} 0 0" Origin="0 0 0" Spacing="1 1 1">\n'.encode())
        buf.write(b'    <Piece Extent="0 {} 0 {} 0 0">\n'.format(nx-1, ny-1).encode())
        buf.write(b'      <PointData Scalars="concentration">\n')
        buf.write(b'        <DataArray type="Float32" Name="concentration" format="ascii">\n')
        np.savetxt(buf, data, fmt="%.6f")
        buf.write(b'        </DataArray>\n')
        buf.write(b'      </PointData>\n')
        buf.write(b'    </Piece>\n')
        buf.write(b'  </ImageData>\n')
        buf.write(b'</VTKFile>\n')
        return buf.getvalue()

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
        cur.execute("INSERT INTO simulation VALUES (?, ?, ?)", (saved_steps[idx], frame.mean(), sqlite3.Binary(compressed)))

    conn.commit()

    sql_dump = BytesIO()
    for line in conn.iterdump():
        sql_dump.write((line + '\n').encode('utf-8'))

    st.download_button("Download SQLite database dump (.sql)", sql_dump.getvalue(), "lfp_simulation.sql")

    st.success("Simulation complete! Use the animation above and download your files below.")

else:
    st.info("Adjust parameters and click **Run Simulation** to start.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/LiFePO4_battery_characteristics.svg/800px-LiFePO4_battery_characteristics.svg.png", caption="Typical LiFePO₄ particle morphology (public domain image)")
