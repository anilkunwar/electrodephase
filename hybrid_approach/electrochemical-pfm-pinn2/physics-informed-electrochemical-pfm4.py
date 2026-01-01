# app.py - Main Streamlit Application (FULL EXPANDED VERSION with PHYSICAL GEOMETRY)
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib import rcParams
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
import json
import time
from pathlib import Path
import base64
from io import BytesIO
import sys
import warnings
warnings.filterwarnings('ignore')

# Publication-quality matplotlib settings
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Computer Modern Roman'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.6,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
    'patch.linewidth': 0.8,
})

# Set page configuration FIRST
st.set_page_config(
    page_title="LiFePO₄ Hybrid FDM-PINN Assimilation",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1E3A8A;
    text-align: center;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 1.2rem;
    color: #4B5563;
    text-align: center;
    margin-bottom: 2rem;
}
.card {
    background-color: #F3F4F6;
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    border-left: 4px solid #3B82F6;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
}
.stProgress > div > div > div > div {
    background-color: #10B981;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# 1. ORIGINAL FDM SIMULATION (MODIFIED FOR PHYSICAL GEOMETRY)
# =====================================================
@st.cache_resource
class PhysicalScalesWithElectrostaticsAndC_Rate:
    # (Same as original — no change needed)
    R = 8.314462618
    F = 96485.33212
    k_B = 1.380649e-23
    ε0 = 8.854187817e-12

    def __init__(self, c_rate=1.0):
        self.T = 298.15
        self.c_alpha = 0.03
        self.c_beta = 0.97
        self.V_m = 3.0e-5
        self.D_b = 1.0e-14
        self.ε_r = 15.0
        self.ε = self.ε_r * self.ε0
        self.z = 1.0
        self.Ω = 55e3
        self.k0_bv = 1.0e-6
        self.k0_mhc = 5.0e-7
        self.alpha = 0.5
        self.set_c_rate_parameters(c_rate)
        self.set_scales()
        self.c_ref = 0.5
        self.λ_D = self.calculate_debye_length()

    def set_c_rate_parameters(self, c_rate):
        self.c_rate = c_rate
        if c_rate <= 1.0:
            self.c_rate_factor = 1.0
            self.eta_scale = 0.01
        else:
            self.c_rate_factor = 1.0 + 0.5 * np.log10(c_rate)
            self.eta_scale = 0.01 * c_rate**0.5
        self.kappa_factor = 1.0 / (1.0 + 0.2 * np.log10(max(1.0, c_rate)))
        self.D_factor = 1.0 / (1.0 + 0.1 * c_rate**0.5)

    def set_scales(self):
        self.L0 = 1.0e-8  # 10 nm (reference length)
        self.E0 = self.Ω / self.V_m
        self.t0 = (self.L0**2) / self.D_b
        self.M0 = self.D_b / (self.E0 * self.t0)
        self.φ0 = self.R * self.T / self.F

    def calculate_debye_length(self):
        c_ref_moles_per_m3 = self.c_ref * (1/self.V_m)
        λ_D = np.sqrt(self.ε * self.R * self.T / (self.F**2 * c_ref_moles_per_m3))
        return λ_D

    def dimensionless_to_physical(self, W_dim, κ_dim, M_dim, dt_dim):
        W_phys = W_dim * self.E0
        κ_phys = κ_dim * self.E0 * self.L0**2
        M_phys = M_dim * self.M0
        dt_phys = dt_dim * self.t0
        return W_phys, κ_phys, M_phys, dt_phys

# Numba fallback (same as original — omitted for brevity but included in full code)
try:
    from numba import njit, prange
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False
    st.warning("⚠️ Numba not installed. Running in pure NumPy mode (slower).")

if USE_NUMBA:
    # (All @njit functions as in original — no change)
    @njit(fastmath=True, cache=True)
    def double_well_energy(c, A, B, C):
        return A * c**2 + B * c**3 + C * c**4

    @njit(fastmath=True, cache=True)
    def chemical_potential(c, A, B, C):
        return 2.0 * A * c + 3.0 * B * c**2 + 4.0 * C * c**3

    @njit(fastmath=True)
    def butler_volmer_flux(eta, c_surf, alpha, k0, F, R, T):
        k_f = k0 * np.exp(-alpha * F * eta / (R * T))
        k_b = k0 * np.exp((1 - alpha) * F * eta / (R * T))
        flux = k_f * (1.0 - c_surf) - k_b * c_surf
        return flux

    @njit(fastmath=True)
    def marcus_hush_chidsey_flux(eta, c_surf, k0, F, R, T):
        eta_dim = F * eta / (R * T)
        flux = k0 * np.tanh(eta_dim / 2.0) * (1.0 - c_surf)
        return flux

    @njit(fastmath=True, parallel=True)
    def compute_laplacian(field, dx):
        nx, ny = field.shape
        lap = np.zeros_like(field)
        for i in prange(nx):
            for j in prange(ny):
                im1 = (i - 1) % nx
                ip1 = (i + 1) % nx
                jm1 = (j - 1) % ny
                jp1 = (j + 1) % ny
                lap[i, j] = (field[ip1, j] + field[im1, j] +
                            field[i, jp1] + field[i, jm1] -
                            4.0 * field[i, j]) / (dx * dx)
        return lap

    @njit(fastmath=True, parallel=True)
    def compute_gradient_x(field, dx):
        nx, ny = field.shape
        grad_x = np.zeros_like(field)
        for i in prange(nx):
            for j in prange(ny):
                ip1 = (i + 1) % nx
                im1 = (i - 1) % nx
                grad_x[i, j] = (field[ip1, j] - field[im1, j]) / (2.0 * dx)
        return grad_x

    @njit(fastmath=True, parallel=True)
    def compute_gradient_y(field, dx):
        nx, ny = field.shape
        grad_y = np.zeros_like(field)
        for i in prange(nx):
            for j in prange(ny):
                jp1 = (j + 1) % ny
                jm1 = (j - 1) % ny
                grad_y[i, j] = (field[i, jp1] - field[i, jm1]) / (2.0 * dx)
        return grad_y

    @njit(fastmath=True, parallel=True)
    def solve_poisson_periodic(phi, c, c_ref, dx, epsilon, F_const, max_iter=100, tol=1e-6):
        nx, ny = c.shape
        phi_new = phi.copy()
        kappa = F_const / epsilon * dx**2
        for it in range(max_iter):
            phi_old = phi_new.copy()
            for i in prange(nx):
                for j in prange(ny):
                    ip1 = (i + 1) % nx
                    im1 = (i - 1) % nx
                    jp1 = (j + 1) % ny
                    jm1 = (j - 1) % ny
                    phi_new[i, j] = 0.25 * (phi_old[ip1, j] + phi_old[im1, j] +
                                            phi_old[i, jp1] + phi_old[i, jm1] +
                                            kappa * (c[i, j] - c_ref))
            max_diff = 0.0
            for i in prange(nx):
                for j in prange(ny):
                    diff = abs(phi_new[i, j] - phi_old[i, j])
                    if diff > max_diff:
                        max_diff = diff
            if max_diff < tol:
                break
        return phi_new

    @njit(fastmath=True, parallel=True)
    def update_concentration(c, phi, dt, dx, kappa, M, D, A, B, C,
                            z, F, R, T, kinetics_type, k0, alpha, eta):
        nx, ny = c.shape
        lap_c = compute_laplacian(c, dx)
        mu_chem = chemical_potential(c, A, B, C) - kappa * lap_c
        mu_total = mu_chem + z * F * phi

        mu_grad_x = compute_gradient_x(mu_total, dx)
        mu_grad_y = compute_gradient_y(mu_total, dx)
        phi_grad_x = compute_gradient_x(phi, dx)
        phi_grad_y = compute_gradient_y(phi, dx)

        c_safe = np.maximum(1e-6, c)
        D_eff = M * R * T / c_safe

        flux_diff_x = -M * mu_grad_x
        flux_diff_y = -M * mu_grad_y

        if kinetics_type == 0:  # PNP
            flux_mig_x = -(D_eff * z * F * c / (R * T)) * phi_grad_x
            flux_mig_y = -(D_eff * z * F * c / (R * T)) * phi_grad_y
            flux_x = flux_diff_x + flux_mig_x
            flux_y = flux_diff_y + flux_mig_y
        else:  # BV or MHC
            flux_x = flux_diff_x
            flux_y = flux_diff_y

        div_flux = np.zeros_like(c)
        for i in prange(nx):
            for j in prange(ny):
                ip1 = (i + 1) % nx
                im1 = (i - 1) % nx
                jp1 = (j + 1) % ny
                jm1 = (j - 1) % ny
                div_x = (flux_x[ip1, j] - flux_x[im1, j]) / (2.0 * dx)
                div_y = (flux_y[i, jp1] - flux_y[i, jm1]) / (2.0 * dx)
                div_flux[i, j] = div_x + div_y

        c_new = c - dt * div_flux

        if kinetics_type != 0:
            for j in prange(ny):
                c_surf = c_new[0, j]
                if kinetics_type == 1:
                    flux = butler_volmer_flux(eta, c_surf, alpha, k0, F, R, T)
                elif kinetics_type == 2:
                    flux = marcus_hush_chidsey_flux(eta, c_surf, k0, F, R, T)
                c_new[0, j] += dt * flux / dx

        c_new = np.minimum(1.0, np.maximum(0.0, c_new))
        return c_new

    @njit(fastmath=True, parallel=True)
    def compute_electric_field(phi, dx):
        nx, ny = phi.shape
        Ex = np.zeros_like(phi)
        Ey = np.zeros_like(phi)
        for i in prange(nx):
            for j in prange(ny):
                ip1 = (i + 1) % nx
                im1 = (i - 1) % nx
                jp1 = (j + 1) % ny
                jm1 = (j - 1) % ny
                Ex[i, j] = -(phi[ip1, j] - phi[im1, j]) / (2.0 * dx)
                Ey[i, j] = -(phi[i, jp1] - phi[i, jm1]) / (2.0 * dx)
        return Ex, Ey

else:
    # Pure NumPy fallback (same as original — omitted for brevity)
    def double_well_energy(c, A, B, C):
        return A * c**2 + B * c**3 + C * c**4

    def chemical_potential(c, A, B, C):
        return 2.0 * A * c + 3.0 * B * c**2 + 4.0 * C * c**3

    def butler_volmer_flux(eta, c_surf, alpha, k0, F, R, T):
        k_f = k0 * np.exp(-alpha * F * eta / (R * T))
        k_b = k0 * np.exp((1 - alpha) * F * eta / (R * T))
        flux = k_f * (1.0 - c_surf) - k_b * c_surf
        return flux

    def marcus_hush_chidsey_flux(eta, c_surf, k0, F, R, T):
        eta_dim = F * eta / (R * T)
        flux = k0 * np.tanh(eta_dim / 2.0) * (1.0 - c_surf)
        return flux

    def compute_laplacian(field, dx):
        nx, ny = field.shape
        lap = np.zeros_like(field)
        for i in range(nx):
            for j in range(ny):
                im1 = (i - 1) % nx
                ip1 = (i + 1) % nx
                jm1 = (j - 1) % ny
                jp1 = (j + 1) % ny
                lap[i, j] = (field[ip1, j] + field[im1, j] +
                            field[i, jp1] + field[i, jm1] -
                            4.0 * field[i, j]) / (dx * dx)
        return lap

    def compute_gradient_x(field, dx):
        nx, ny = field.shape
        grad_x = np.zeros_like(field)
        for i in range(nx):
            for j in range(ny):
                ip1 = (i + 1) % nx
                im1 = (i - 1) % nx
                grad_x[i, j] = (field[ip1, j] - field[im1, j]) / (2.0 * dx)
        return grad_x

    def compute_gradient_y(field, dx):
        nx, ny = field.shape
        grad_y = np.zeros_like(field)
        for i in range(nx):
            for j in range(ny):
                jp1 = (j + 1) % ny
                jm1 = (j - 1) % ny
                grad_y[i, j] = (field[i, jp1] - field[i, jm1]) / (2.0 * dx)
        return grad_y

    def solve_poisson_periodic(phi, c, c_ref, dx, epsilon, F_const, max_iter=100, tol=1e-6):
        nx, ny = c.shape
        phi_new = phi.copy()
        kappa = F_const / epsilon * dx**2
        for it in range(max_iter):
            phi_old = phi_new.copy()
            for i in range(nx):
                for j in range(ny):
                    ip1 = (i + 1) % nx
                    im1 = (i - 1) % nx
                    jp1 = (j + 1) % ny
                    jm1 = (j - 1) % ny
                    phi_new[i, j] = 0.25 * (phi_old[ip1, j] + phi_old[im1, j] +
                                            phi_old[i, jp1] + phi_old[i, jm1] +
                                            kappa * (c[i, j] - c_ref))
            max_diff = np.max(np.abs(phi_new - phi_old))
            if max_diff < tol:
                break
        return phi_new

    def update_concentration(c, phi, dt, dx, kappa, M, D, A, B, C,
                            z, F, R, T, kinetics_type, k0, alpha, eta):
        nx, ny = c.shape
        lap_c = compute_laplacian(c, dx)
        mu_chem = chemical_potential(c, A, B, C) - kappa * lap_c
        mu_total = mu_chem + z * F * phi

        mu_grad_x = compute_gradient_x(mu_total, dx)
        mu_grad_y = compute_gradient_y(mu_total, dx)
        phi_grad_x = compute_gradient_x(phi, dx)
        phi_grad_y = compute_gradient_y(phi, dx)

        c_safe = np.maximum(1e-6, c)
        D_eff = M * R * T / c_safe

        flux_diff_x = -M * mu_grad_x
        flux_diff_y = -M * mu_grad_y

        if kinetics_type == 0:  # PNP
            flux_mig_x = -(D_eff * z * F * c / (R * T)) * phi_grad_x
            flux_migr_y = -(D_eff * z * F * c / (R * T)) * phi_grad_y
            flux_x = flux_diff_x + flux_mig_x
            flux_y = flux_diff_y + flux_mig_y
        else:  # BV or MHC
            flux_x = flux_diff_x
            flux_y = flux_diff_y

        div_flux = np.zeros_like(c)
        for i in range(nx):
            for j in range(ny):
                ip1 = (i + 1) % nx
                im1 = (i - 1) % nx
                jp1 = (j + 1) % ny
                jm1 = (j - 1) % ny
                div_x = (flux_x[ip1, j] - flux_x[im1, j]) / (2.0 * dx)
                div_y = (flux_y[i, jp1] - flux_y[i, jm1]) / (2.0 * dx)
                div_flux[i, j] = div_x + div_y

        c_new = c - dt * div_flux

        if kinetics_type != 0:
            for j in range(ny):
                c_surf = c_new[0, j]
                if kinetics_type == 1:
                    flux = butler_volmer_flux(eta, c_surf, alpha, k0, F, R, T)
                elif kinetics_type == 2:
                    flux = marcus_hush_chidsey_flux(eta, c_surf, k0, F, R, T)
                c_new[0, j] += dt * flux / dx

        c_new = np.minimum(1.0, np.maximum(0.0, c_new))
        return c_new

    def compute_electric_field(phi, dx):
        nx, ny = phi.shape
        Ex = np.zeros_like(phi)
        Ey = np.zeros_like(phi)
        for i in range(nx):
            for j in range(ny):
                ip1 = (i + 1) % nx
                im1 = (i - 1) % nx
                jp1 = (j + 1) % ny
                jm1 = (j - 1) % ny
                Ex[i, j] = -(phi[ip1, j] - phi[im1, j]) / (2.0 * dx)
                Ey[i, j] = -(phi[i, jp1] - phi[i, jm1]) / (2.0 * dx)
        return Ex, Ey

@st.cache_resource
class ElectrochemicalPhaseFieldSimulation:
    """MODIFIED to accept physical domain size Lx, Ly (meters)"""

    def __init__(self, nx=128, ny=128, Lx=200e-9, Ly=200e-9, dt=0.01, c_rate=1.0):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx  # meters
        self.Ly = Ly  # meters
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.dt = dt
        self.c_rate = c_rate

        self.scales = PhysicalScalesWithElectrostaticsAndC_Rate(c_rate=c_rate)

        self.W_dim = 1.0
        self.A = self.W_dim
        self.B = -2.0 * self.W_dim
        self.C = self.W_dim
        self.kappa_dim = 2.0 * self.scales.kappa_factor
        self.M_dim = 1.0 * self.scales.D_factor

        self.update_physical_parameters()

        self.kinetics_type = 0
        self.eta_left = 0.0

        self.c = np.zeros((nx, ny))
        self.phi = np.zeros((nx, ny))
        self.Ex = np.zeros((nx, ny))
        self.Ey = np.zeros((nx, ny))

        self.time_dim = 0.0
        self.time_phys = 0.0
        self.step = 0

        self.history = {
            'time_phys': [],
            'mean_c': [],
            'std_c': [],
            'mean_phi': [],
            'voltage': [],
            'phase_FePO4': [],
            'phase_LiFePO4': [],
            'total_charge': []
        }
        self.initialize_random()

    def update_physical_parameters(self):
        self.W_phys, self.kappa_phys, self.M_phys, self.dt_phys = \
            self.scales.dimensionless_to_physical(
                self.W_dim, self.kappa_dim, self.M_dim, self.dt
            )
        self.A = self.W_dim
        self.B = -2.0 * self.W_dim
        self.C = self.W_dim

    def set_parameters(self, W_dim=None, kappa_dim=None, M_dim=None, dt_dim=None,
                       c_rate=None, kinetics_type=None):
        if W_dim is not None:
            self.W_dim = W_dim
        if kappa_dim is not None:
            self.kappa_dim = kappa_dim
        if M_dim is not None:
            self.M_dim = M_dim
        if dt_dim is not None:
            self.dt = dt_dim
        if c_rate is not None:
            self.c_rate = c_rate
            self.scales.set_c_rate_parameters(c_rate)
            self.kappa_dim = 2.0 * self.scales.kappa_factor
            self.M_dim = 1.0 * self.scales.D_factor
        if kinetics_type is not None:
            self.kinetics_type = kinetics_type
        self.update_physical_parameters()

    def initialize_random(self, c0=0.5, noise_amplitude=0.05, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.c = c0 + noise_amplitude * (2.0 * np.random.random((self.nx, self.ny)) - 1.0)
        self.c = np.clip(self.c, 0, 1)
        self.phi = np.zeros_like(self.c)
        self.eta_left = 0.0
        self.time_dim = 0.0
        self.time_phys = 0.0
        self.step = 0
        self.clear_history()

    def initialize_lithiation(self, noise_amplitude=0.05, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.c = self.scales.c_alpha * np.ones((self.nx, self.ny))
        self.c += noise_amplitude * (2.0 * np.random.random((self.nx, self.ny)) - 1.0)
        self.c = np.clip(self.c, 0, 1)
        self.phi = np.zeros_like(self.c)
        for i in range(self.nx):
            self.phi[i, :] = -0.2 * np.exp(-i / (self.nx * 0.2))
        self.eta_left = self.scales.eta_scale
        self.time_dim = 0.0
        self.time_phys = 0.0
        self.step = 0
        self.clear_history()

    def initialize_delithiation(self, noise_amplitude=0.05, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.c = self.scales.c_beta * np.ones((self.nx, self.ny))
        self.c += noise_amplitude * (2.0 * np.random.random((self.nx, self.ny)) - 1.0)
        self.c = np.clip(self.c, 0, 1)
        self.phi = np.zeros_like(self.c)
        for i in range(self.nx):
            self.phi[i, :] = 0.2 * np.exp(-i / (self.nx * 0.2))
        self.eta_left = -self.scales.eta_scale
        self.time_dim = 0.0
        self.time_phys = 0.0
        self.step = 0
        self.clear_history()

    def clear_history(self):
        self.history = {k: [] for k in self.history}

    def update_history(self):
        self.history['time_phys'].append(self.time_phys)
        self.history['mean_c'].append(np.mean(self.c))
        self.history['std_c'].append(np.std(self.c))
        self.history['mean_phi'].append(np.mean(self.phi))
        voltage = np.mean(self.phi[-1, :]) - np.mean(self.phi[0, :])
        self.history['voltage'].append(voltage)
        threshold = 0.5
        self.history['phase_FePO4'].append(np.sum(self.c < threshold) / (self.nx * self.ny))
        self.history['phase_LiFePO4'].append(np.sum(self.c >= threshold) / (self.nx * self.ny))
        self.history['total_charge'].append(np.sum(self.c - 0.5))

    def run_step(self):
        c_ref = np.mean(self.c)
        self.phi = solve_poisson_periodic(
            self.phi, self.c, c_ref, self.dx,
            self.scales.ε, self.scales.F,
            max_iter=50, tol=1e-4
        )
        self.Ex, self.Ey = compute_electric_field(self.phi, self.dx)

        if self.kinetics_type == 0:
            k0 = 0.0; alpha = 0.0; eta = 0.0
        else:
            eta = self.eta_left
            alpha = self.scales.alpha
            k0 = self.scales.k0_bv if self.kinetics_type == 1 else self.scales.k0_mhc
            k0 *= self.scales.c_rate_factor

        self.c = update_concentration(
            self.c, self.phi, self.dt, self.dx,
            self.kappa_dim, self.M_dim, self.scales.D_b * self.scales.D_factor,
            self.A, self.B, self.C,
            self.scales.z, self.scales.F, self.scales.R, self.scales.T,
            self.kinetics_type, k0, alpha, eta
        )
        self.c = np.clip(self.c, 0, 1)
        self.time_dim += self.dt
        self.time_phys += self.dt_phys
        self.step += 1
        self.update_history()

    def run_steps(self, n_steps):
        for _ in range(n_steps):
            self.run_step()

    def run_until(self, target_time_phys):
        steps_needed = max(1, int((target_time_phys - self.time_phys) / self.dt_phys))
        self.run_steps(steps_needed)

    def get_statistics(self):
        return {
            'time_phys': self.time_phys,
            'step': self.step,
            'mean_c': np.mean(self.c),
            'std_c': np.std(self.c),
            'voltage': np.mean(self.phi[-1, :]) - np.mean(self.phi[0, :]),
            'phase_FePO4': np.sum(self.c < 0.5) / (self.nx * self.ny),
            'phase_LiFePO4': np.sum(self.c >= 0.5) / (self.nx * self.ny),
            'domain_width_nm': self.Lx * 1e9,
            'domain_height_nm': self.Ly * 1e9,
            'pixel_size_nm': self.dx * 1e9,
            'c_rate': self.c_rate,
            'kinetics_type': self.kinetics_type
        }

# =====================================================
# 2. OBSERVATION GENERATOR — MODIFIED FOR PHYSICAL COORDS
# =====================================================
@st.cache_resource
class SyntheticObservationGenerator:
    def __init__(self):
        self.observation_types = {
            'microscopy': {'coverage': 0.1, 'noise_std': 0.05, 'pattern': 'random'},
            'xrd_mapping': {'coverage': 0.05, 'noise_std': 0.03, 'pattern': 'grid'},
            'tomography': {'coverage': 0.15, 'noise_std': 0.04, 'pattern': 'lines'},
            'afm': {'coverage': 0.08, 'noise_std': 0.06, 'pattern': 'random'}
        }

    def generate_sparse_observations(
        self,
        c_field: np.ndarray,
        dx: float,
        L0: float,
        observation_type: str = 'microscopy',
        measurement_time: float = 0.0,
        seed: int = None,
        custom_coverage: float = None,
        custom_noise: float = None
    ) -> Dict:
        if seed is not None:
            np.random.seed(seed)
        nx, ny = c_field.shape
        config = self.observation_types[observation_type].copy()
        if custom_coverage is not None:
            config['coverage'] = custom_coverage
        if custom_noise is not None:
            config['noise_std'] = custom_noise

        if config['pattern'] == 'random':
            n_obs = max(1, int(config['coverage'] * nx * ny))
            obs_indices = np.random.choice(nx * ny, n_obs, replace=False)
            obs_i = obs_indices // ny
            obs_j = obs_indices % ny
        elif config['pattern'] == 'grid':
            step = max(1, int(1 / np.sqrt(config['coverage'])))
            obs_i, obs_j = np.meshgrid(
                np.arange(0, nx, step),
                np.arange(0, ny, step),
                indexing='ij'
            )
            obs_i = obs_i.flatten()
            obs_j = obs_j.flatten()
        elif config['pattern'] == 'lines':
            n_lines = max(1, int(np.sqrt(config['coverage'] * nx * ny / ny)))
            line_indices = np.linspace(0, nx-1, n_lines).astype(int)
            obs_i = np.repeat(line_indices, ny)
            obs_j = np.tile(np.arange(ny), n_lines)

        if len(obs_i) == 0:
            obs_i = np.array([nx//2])
            obs_j = np.array([ny//2])

        c_obs = c_field[obs_i, obs_j]
        noise = np.random.normal(0, config['noise_std'], c_obs.shape)
        c_obs_noisy = np.clip(c_obs + noise, 0, 1)

        # NOW: physical coordinates (meters)
        x_phys = obs_i * dx  # dx is in meters
        y_phys = obs_j * dx

        return {
            'time_phys': measurement_time,
            'x_coords': x_phys,
            'y_coords': y_phys,
            'c_obs': c_obs_noisy,
            'x_idx': obs_i,
            'y_idx': obs_j,
            'noise_std': config['noise_std'],
            'coverage': len(obs_i) / (nx * ny),
            'observation_type': observation_type,
            'full_field': c_field
        }

# =====================================================
# 3. PINN & ASSIMILATION — UNCHANGED (already uses physical Lx, Ly)
# =====================================================
class LiFePO4AssimilationPINN(nn.Module):
    def __init__(self, Lx: float, Ly: float, hidden_dims: List[int] = [64, 64, 64]):
        super().__init__()
        self.Lx = Lx
        self.Ly = Ly
        self.x_scale = 1.0 / Lx if Lx > 0 else 1.0
        self.y_scale = 1.0 / Ly if Ly > 0 else 1.0
        layers = []
        input_dim = 2
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.Tanh())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_norm = x * self.x_scale
        y_norm = y * self.y_scale
        inputs = torch.stack([x_norm, y_norm], dim=-1)
        return self.net(inputs).squeeze(-1)

    def compute_chemical_potential(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        A: float,
        B: float,
        C: float,
        kappa: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x.requires_grad_(True)
        y.requires_grad_(True)
        c = self.forward(x, y)
        grad_c = torch.autograd.grad(
            c, [x, y],
            grad_outputs=torch.ones_like(c),
            create_graph=True,
            retain_graph=True
        )
        grad_c_x, grad_c_y = grad_c
        grad_c_x_x = torch.autograd.grad(
            grad_c_x, x,
            grad_outputs=torch.ones_like(grad_c_x),
            create_graph=True,
            retain_graph=True
        )[0]
        grad_c_y_y = torch.autograd.grad(
            grad_c_y, y,
            grad_outputs=torch.ones_like(grad_c_y),
            create_graph=True
        )[0]
        laplacian_c = grad_c_x_x + grad_c_y_y
        mu_chem = (
            2.0 * A * c +
            3.0 * B * c.pow(2) +
            4.0 * C * c.pow(3) -
            kappa * laplacian_c
        )
        return mu_chem, c

class PINNAssimilationTrainer:
    def __init__(self, pinn: nn.Module, device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pinn = pinn.to(device)
        self.device = device
        self.loss_history = []

    def train(
        self,
        obs_data: Dict,
        phi_field: np.ndarray,
        sim_params: Dict,
        n_epochs: int = 500,
        lr: float = 1e-3,
        data_weight: float = 1.0,
        physics_weight: float = 0.1,
        reg_weight: float = 0.01,
        batch_size: int = 256,
        progress_callback = None
    ) -> Dict:
        x_obs = torch.tensor(obs_data['x_coords'], dtype=torch.float32).to(self.device)
        y_obs = torch.tensor(obs_data['y_coords'], dtype=torch.float32).to(self.device)
        c_obs = torch.tensor(obs_data['c_obs'], dtype=torch.float32).to(self.device)
        dataset = TensorDataset(x_obs, y_obs, c_obs)
        dataloader = DataLoader(dataset, batch_size=min(batch_size, len(x_obs)), shuffle=True)
        optimizer = optim.Adam(self.pinn.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5, verbose=False)
        start_time = time.time()
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            epoch_data_loss = 0.0
            epoch_physics_loss = 0.0
            for x_batch, y_batch, c_batch in dataloader:
                optimizer.zero_grad()
                c_pred = self.pinn(x_batch, y_batch)
                loss_data = torch.mean((c_pred - c_batch).pow(2))
                loss_physics = self._compute_physics_loss(
                    x_batch, y_batch, phi_field, sim_params
                )
                l2_reg = torch.tensor(0.0).to(self.device)
                for param in self.pinn.parameters():
                    l2_reg += torch.norm(param)
                loss = (
                    data_weight * loss_data +
                    physics_weight * loss_physics +
                    reg_weight * l2_reg
                )
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(x_batch)
                epoch_data_loss += loss_data.item() * len(x_batch)
                epoch_physics_loss += loss_physics.item() * len(x_batch)
            epoch_loss /= len(x_obs)
            epoch_data_loss /= len(x_obs)
            epoch_physics_loss /= len(x_obs)
            self.loss_history.append({
                'epoch': epoch,
                'total_loss': epoch_loss,
                'data_loss': epoch_data_loss,
                'physics_loss': epoch_physics_loss
            })
            scheduler.step(epoch_loss)
            if progress_callback is not None:
                progress_callback(epoch + 1, n_epochs, epoch_loss)
        training_time = time.time() - start_time
        return {
            'training_time': training_time,
            'final_loss': self.loss_history[-1]['total_loss'] if self.loss_history else 0.0,
            'loss_history': self.loss_history,
            'num_parameters': sum(p.numel() for p in self.pinn.parameters()),
            'num_observations': len(x_obs)
        }

    def _compute_physics_loss(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        phi_field: np.ndarray,
        sim_params: Dict
    ) -> torch.Tensor:
        mu_chem, _ = self.pinn.compute_chemical_potential(
            x_batch, y_batch,
            sim_params['A'], sim_params['B'], sim_params['C'],
            sim_params['kappa']
        )
        nx, ny = phi_field.shape
        x_idx = torch.clamp((x_batch / sim_params['dx']).long(), 0, nx-1)
        y_idx = torch.clamp((y_batch / sim_params['dx']).long(), 0, ny-1)
        phi_batch = torch.tensor(phi_field[x_idx.cpu(), y_idx.cpu()],
                                 dtype=torch.float32).to(self.device)
        mu_total = mu_chem + sim_params['z'] * sim_params['F'] * phi_batch
        return torch.var(mu_total)

    def reconstruct_full_field(
        self,
        nx: int,
        ny: int,
        dx: float,
        L0: float
    ) -> np.ndarray:
        self.pinn.eval()
        Lx = nx * dx
        Ly = ny * dx
        x_grid = np.linspace(0, Lx, nx)
        y_grid = np.linspace(0, Ly, ny)
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        with torch.no_grad():
            X_tensor = torch.tensor(X.flatten(), dtype=torch.float32).to(self.device)
            Y_tensor = torch.tensor(Y.flatten(), dtype=torch.float32).to(self.device)
            c_pred = self.pinn(X_tensor, Y_tensor)
            c_field = c_pred.cpu().numpy().reshape(nx, ny)
        return c_field

class HybridFDMPINNAssimilation:
    def __init__(self):
        self.sim = None
        self.obs_gen = SyntheticObservationGenerator()
        self.assimilation_history = []
        self.pinn_reconstructions = []
        self.correction_magnitudes = []
        self.current_cycle = 0

    def initialize_simulation(
        self,
        nx: int = 128,
        ny: int = 128,
        Lx: float = 200e-9,
        Ly: float = 200e-9,
        dt: float = 0.01,
        c_rate: float = 1.0
    ):
        self.sim = ElectrochemicalPhaseFieldSimulation(
            nx=nx, ny=ny, Lx=Lx, Ly=Ly, dt=dt, c_rate=c_rate
        )

    def run_assimilation_cycle(
        self,
        t_obs: float,
        observation_type: str = 'microscopy',
        coverage: float = None,
        noise_std: float = None,
        pinn_hidden_dims: List[int] = [32, 32, 32],
        n_epochs: int = 300,
        physics_weight: float = 0.2,
        damping_factor: float = 0.7,
        progress_callback = None
    ) -> Dict:
        if self.sim is None:
            raise ValueError("Simulation not initialized.")
        st.info(f"🔁 Starting assimilation cycle at t = {t_obs:.2e} s")
        self.sim.run_until(t_obs)
        c_fdm_before = self.sim.c.copy()
        phi_fdm = self.sim.phi.copy()
        obs_data = self.obs_gen.generate_sparse_observations(
            c_fdm_before,
            self.sim.dx,  # physical dx in meters
            1.0,
            observation_type=observation_type,
            measurement_time=t_obs,
            custom_coverage=coverage,
            custom_noise=noise_std
        )
        Lx = self.sim.Lx
        Ly = self.sim.Ly
        pinn = LiFePO4AssimilationPINN(Lx, Ly, pinn_hidden_dims)
        trainer = PINNAssimilationTrainer(pinn)
        sim_params = {
            'A': self.sim.A,
            'B': self.sim.B,
            'C': self.sim.C,
            'kappa': self.sim.kappa_dim,
            'z': self.sim.scales.z,
            'F': self.sim.scales.F,
            'dx': self.sim.dx,
            'nx': self.sim.nx,
            'ny': self.sim.ny
        }
        training_stats = trainer.train(
            obs_data,
            phi_fdm,
            sim_params,
            n_epochs=n_epochs,
            physics_weight=physics_weight,
            progress_callback=progress_callback
        )
        c_pinn = trainer.reconstruct_full_field(self.sim.nx, self.sim.ny, self.sim.dx, 1.0)
        correction = c_pinn - c_fdm_before
        correction_norm = np.linalg.norm(correction) / np.sqrt(correction.size)
        self.sim.c = c_fdm_before + damping_factor * correction
        cycle_result = {
            'cycle': self.current_cycle,
            'time_phys': t_obs,
            'c_fdm_before': c_fdm_before,
            'c_pinn': c_pinn,
            'correction': correction,
            'correction_norm': correction_norm,
            'observation_data': obs_data,
            'training_stats': training_stats,
            'phi_field': phi_fdm,
            'mean_c_before': np.mean(c_fdm_before),
            'mean_c_after': np.mean(self.sim.c),
            'pinn_params': {
                'hidden_dims': pinn_hidden_dims,
                'n_epochs': n_epochs,
                'physics_weight': physics_weight
            }
        }
        self.assimilation_history.append(cycle_result)
        self.pinn_reconstructions.append(c_pinn)
        self.correction_magnitudes.append(correction_norm)
        self.current_cycle += 1
        return cycle_result

    def run_sequential_assimilation(
        self,
        observation_schedule: List[float],
        observation_type: str = 'microscopy',
        coverage: float = None,
        noise_std: float = None,
        pinn_config: Dict = None,
        damping_factor: float = 0.7,
        progress_container = None
    ) -> List[Dict]:
        results = []
        default_pinn_config = {
            'hidden_dims': [32, 32, 32],
            'n_epochs': 300,
            'physics_weight': 0.2
        }
        if pinn_config:
            default_pinn_config.update(pinn_config)
        for i, t_obs in enumerate(observation_schedule):
            if progress_container:
                progress_container.text(f"Cycle {i+1}/{len(observation_schedule)}: t = {t_obs:.0f} s")
            def update_progress(epoch, total_epochs, loss):
                if progress_container:
                    progress_container.text(f"Cycle {i+1}/{len(observation_schedule)}: Epoch {epoch}/{total_epochs}, Loss: {loss:.2e}")
            result = self.run_assimilation_cycle(
                t_obs,
                observation_type=observation_type,
                coverage=coverage,
                noise_std=noise_std,
                pinn_hidden_dims=default_pinn_config['hidden_dims'],
                n_epochs=default_pinn_config['n_epochs'],
                physics_weight=default_pinn_config['physics_weight'],
                damping_factor=damping_factor,
                progress_callback=update_progress
            )
            results.append(result)
        if progress_container:
            progress_container.text("✅ Assimilation complete!")
        return results

# =====================================================
# 4. ENHANCED VISUALIZATION
# =====================================================
def plot_concentration_field_plotly(field: np.ndarray, Lx: float, Ly: float, title: str = "Concentration Field"):
    unit = "m"
    x_max = Lx
    y_max = Ly
    if Lx < 1e-6:
        x_max = Lx * 1e9
        y_max = Ly * 1e9
        unit = "nm"
    x = np.linspace(0, x_max, field.shape[0])
    y = np.linspace(0, y_max, field.shape[1])
    fig = go.Figure(data=go.Heatmap(
        x=x, y=y, z=field.T,
        colorscale='RdYlBu', zmin=0, zmax=1,
        colorbar=dict(title="x in LiₓFePO₄")
    ))
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(title=f"x ({unit})"),
        yaxis=dict(title=f"y ({unit})"),
        width=500, height=450
    )
    return fig

def plot_concentration_matplotlib(field: np.ndarray, Lx: float, Ly: float, title: str = "Concentration Field"):
    unit = "m"
    x_max = Lx
    y_max = Ly
    if Lx < 1e-6:
        x_max = Lx * 1e9
        y_max = Ly * 1e9
        unit = "nm"
    x = np.linspace(0, x_max, field.shape[0])
    y = np.linspace(0, y_max, field.shape[1])
    X, Y = np.meshgrid(x, y, indexing='ij')
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.contourf(X, Y, field, levels=50, cmap='RdYlBu')
    ax.set_xlabel(f'x ({unit})')
    ax.set_ylabel(f'y ({unit})')
    ax.set_aspect('equal')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('x in Li$_x$FePO$_4$', rotation=270, labelpad=15)
    ax.set_title(title)
    return fig

def plot_assimilation_cycle(cycle_result: Dict, Lx: float, Ly: float):
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'FDM before assimilation',
            'PINN reconstruction',
            'Correction field',
            'Observations overlay',
            'Training loss history',
            'Electric potential'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    fig.add_trace(go.Heatmap(z=cycle_result['c_fdm_before'].T, colorscale='RdYlBu', zmin=0, zmax=1), row=1, col=1)
    fig.add_trace(go.Heatmap(z=cycle_result['c_pinn'].T, colorscale='RdYlBu', zmin=0, zmax=1), row=1, col=2)
    corr = cycle_result['correction']
    vmax = max(abs(corr.min()), abs(corr.max()))
    fig.add_trace(go.Heatmap(z=corr.T, colorscale='RdBu_r', zmin=-vmax, zmax=vmax), row=1, col=3)
    loss_hist = cycle_result['training_stats']['loss_history']
    epochs = [l['epoch'] for l in loss_hist]
    total_loss = [l['total_loss'] for l in loss_hist]
    fig.add_trace(go.Scatter(x=epochs, y=total_loss, mode='lines', name='Loss'), row=2, col=2)
    fig.add_trace(go.Heatmap(z=cycle_result['phi_field'].T, colorscale='RdBu_r'), row=2, col=3)
    obs = cycle_result['observation_data']
    unit = "m"
    x_obs = obs['x_coords']
    y_obs = obs['y_coords']
    if Lx < 1e-6:
        x_obs = x_obs * 1e9
        y_obs = y_obs * 1e9
        unit = "nm"
    fig.add_trace(go.Scatter(x=x_obs, y=y_obs, mode='markers',
        marker=dict(size=5, color=obs['c_obs'], colorscale='RdYlBu', cmin=0, cmax=1)), row=2, col=1)
    nx, ny = cycle_result['c_fdm_before'].shape
    if Lx < 1e-6:
        x_bg = np.linspace(0, Lx*1e9, nx)
        y_bg = np.linspace(0, Ly*1e9, ny)
    else:
        x_bg = np.linspace(0, Lx, nx)
        y_bg = np.linspace(0, Ly, ny)
    X_bg, Y_bg = np.meshgrid(x_bg, y_bg, indexing='ij')
    fig.add_trace(go.Heatmap(x=x_bg, y=y_bg, z=cycle_result['c_fdm_before'].T,
        colorscale='gray', opacity=0.3, showscale=False), row=2, col=1)
    fig.update_xaxes(title_text=f"x ({unit})", row=2, col=1)
    fig.update_yaxes(title_text=f"y ({unit})", row=2, col=1)
    fig.update_layout(height=800, title_text=f"Assimilation Cycle at t={cycle_result['time_phys']:.2e}s", title_x=0.5)
    return fig

# =====================================================
# 5. MAIN STREAMLIT APP — FULLY EXPANDED
# =====================================================
def main():
    st.markdown('<h1 class="main-header">🔄 LiFePO₄ Hybrid FDM-PINN Data Assimilation</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time correction with physical geometry and kinetics selection</p>', unsafe_allow_html=True)

    if 'hybrid_system' not in st.session_state:
        st.session_state.hybrid_system = HybridFDMPINNAssimilation()
    if 'sim_initialized' not in st.session_state:
        st.session_state.sim_initialized = False
    if 'assimilation_results' not in st.session_state:
        st.session_state.assimilation_results = []

    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/battery--v1.png", width=80)
        with st.expander("⚙️ Simulation Setup", expanded=True):
            grid_size = st.selectbox("Grid Resolution", ["64×64 (Fast)", "128×128 (Balanced)", "256×256 (Detailed)"], index=1)
            if grid_size == "64×64 (Fast)": nx, ny = 64, 64
            elif grid_size == "128×128 (Balanced)": nx, ny = 128, 128
            else: nx, ny = 256, 256

            # 🔥 NEW: Physical domain size sliders (nm)
            domain_nm = st.slider("Domain Size (nm)", 50, 1000, 200, 10)
            Lx = Ly = domain_nm * 1e-9  # convert to meters

            c_rate = st.slider("C-Rate", 0.1, 10.0, 1.0, 0.1)
            dt = st.slider("Time Step (s)", 1e-5, 0.1, 1e-3, 1e-5, format="%.1e")

            # Kinetics selector (from your original)
            kinetics_choice = st.selectbox(
                "Electrochemical Kinetics",
                [
                    "Poisson–Nernst–Planck (PNP) — Bulk migration",
                    "Butler–Volmer (BV) — Surface reaction",
                    "Marcus–Hush–Chidsey (MHC) — Quantum kinetics"
                ],
                index=0,
                help=(
                    "• **PNP**: Full bulk ion transport with electric field coupling.\n"
                    "• **BV**: Classical electrode kinetics; applies flux at left boundary.\n"
                    "• **MHC**: Non-adiabatic electron transfer; more accurate at high overpotentials."
                )
            )
            kinetics_type = 0
            if "BV" in kinetics_choice: kinetics_type = 1
            elif "MHC" in kinetics_choice: kinetics_type = 2

            init_type = st.selectbox("Initialization", ["Random", "Lithiation (Charge)", "Delithiation (Discharge)"], index=1)
            if kinetics_type != 0 and init_type == "Random":
                st.warning("⚠️ Use Lithiation/Delithiation with BV/MHC for physical consistency.")

            if st.button("🔄 Initialize Simulation", use_container_width=True):
                with st.spinner("Initializing simulation..."):
                    st.session_state.hybrid_system.initialize_simulation(
                        nx=nx, ny=ny, Lx=Lx, Ly=Ly, dt=dt, c_rate=c_rate
                    )
                    sim = st.session_state.hybrid_system.sim
                    sim.set_parameters(kinetics_type=kinetics_type)
                    if init_type == "Random":
                        sim.initialize_random()
                    elif init_type == "Lithiation (Charge)":
                        sim.initialize_lithiation()
                    else:
                        sim.initialize_delithiation()
                    st.session_state.sim_initialized = True
                    st.session_state.assimilation_results = []
                    st.success("✅ Simulation initialized!")
                    st.rerun()

        # Assimilation settings (same as your original)
        with st.expander("🔬 Assimilation Settings", expanded=True):
            observation_type = st.selectbox("Observation Type", ["microscopy", "xrd_mapping", "tomography", "afm"], index=0)
            coverage = st.slider("Observation Coverage", 0.01, 0.3, 0.1, 0.01)
            noise_std = st.slider("Measurement Noise", 0.0, 0.1, 0.05, 0.01)
            pinn_hidden_dims = st.text_input("Hidden Layer Sizes", "32,32,32")
            try:
                hidden_dims = [int(x.strip()) for x in pinn_hidden_dims.split(",")]
            except:
                hidden_dims = [32, 32, 32]
            n_epochs = st.slider("Training Epochs", 100, 1000, 300, 50)
            physics_weight = st.slider("Physics Weight", 0.0, 1.0, 0.2, 0.05)
            damping_factor = st.slider("Damping Factor", 0.1, 1.0, 0.7, 0.1)

        # Schedule (same)
        with st.expander("⏱️ Assimilation Schedule", expanded=True):
            schedule_input = st.text_area("Observation Times (s)", "1000, 5000, 10000")
            try:
                observation_schedule = [float(x.strip()) for x in schedule_input.split(",")]
            except:
                observation_schedule = [1000.0, 5000.0, 10000.0]

    # Status panel
    if st.session_state.sim_initialized:
        sim = st.session_state.hybrid_system.sim
        stats = sim.get_statistics()
        st.metric("Domain", f"{stats['domain_width_nm']:.0f} × {stats['domain_height_nm']:.0f} nm")
        st.metric("Kinetics", ["PNP", "BV", "MHC"][stats['kinetics_type']])
        st.metric("Time", f"{stats['time_phys']:.2e} s")
        st.metric("Cycles", len(st.session_state.assimilation_results))
    else:
        st.warning("Simulation not initialized")

    if not st.session_state.sim_initialized:
        st.info("Initialize simulation from sidebar to proceed.")
        return

    sim = st.session_state.hybrid_system.sim
    tab1, tab2, tab3 = st.tabs(["📈 Simulation", "🔬 Assimilation", "📊 Results"])

    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            steps = st.number_input("Steps to run", 1, 1000, 100)
            if st.button("▶️ Run Steps"):
                with st.spinner(f"Running {steps} steps..."):
                    sim.run_steps(steps)
                st.rerun()

            # Plotly with physical axes
            fig = plot_concentration_field_plotly(
                sim.c, sim.Lx, sim.Ly,
                title=f"LiₓFePO₄ (t={sim.time_phys:.2e}s, x={np.mean(sim.c):.3f})"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Matplotlib toggle
            if st.checkbox("Show Publication-Quality Plot (Matplotlib)"):
                fig_mpl = plot_concentration_matplotlib(sim.c, sim.Lx, sim.Ly)
                st.pyplot(fig_mpl)
                plt.close(fig_mpl)

        with col2:
            stats = sim.get_statistics()
            st.metric("Pixel Size", f"{stats['pixel_size_nm']:.1f} nm")
            st.metric("Mean x", f"{stats['mean_c']:.3f}")
            st.metric("FePO₄", f"{stats['phase_FePO4']:.1%}")

    with tab2:
        if st.button("🔁 Run Single Cycle"):
            t_obs = sim.time_phys + 1000.0
            progress_container = st.empty()
            def update_progress(epoch, total_epochs, loss):
                progress_container.text(f"Epoch {epoch}/{total_epochs}, Loss: {loss:.2e}")
            result = st.session_state.hybrid_system.run_assimilation_cycle(
                t_obs,
                observation_type=observation_type,
                coverage=coverage,
                noise_std=noise_std,
                pinn_hidden_dims=hidden_dims,
                n_epochs=n_epochs,
                physics_weight=physics_weight,
                damping_factor=damping_factor,
                progress_callback=update_progress
            )
            st.session_state.assimilation_results.append(result)
            st.success("✅ Assimilation cycle complete!")
            st.rerun()

        if st.session_state.assimilation_results:
            cycle = st.session_state.assimilation_results[-1]
            fig_cycle = plot_assimilation_cycle(cycle, sim.Lx, sim.Ly)
            st.plotly_chart(fig_cycle, use_container_width=True)

    with tab3:
        if st.session_state.assimilation_results:
            stats = {
                'total_cycles': len(st.session_state.assimilation_results),
                'correction_magnitudes': [r['correction_norm'] for r in st.session_state.assimilation_results],
                'accuracy_improvements': []
            }
            for r in st.session_state.assimilation_results:
                obs = r['observation_data']
                fdm_rmse = np.sqrt(np.mean((r['c_fdm_before'][obs['x_idx'], obs['y_idx']] - obs['c_obs'])**2))
                pinn_rmse = np.sqrt(np.mean((r['c_pinn'][obs['x_idx'], obs['y_idx']] - obs['c_obs'])**2))
                stats['accuracy_improvements'].append(fdm_rmse / pinn_rmse if pinn_rmse > 0 else 0)
            st.metric("Avg. RMSE Improvement", f"{np.mean(stats['accuracy_improvements']):.1f}x")
        else:
            st.info("Run assimilation cycles to see results.")

if __name__ == "__main__":
    main()
