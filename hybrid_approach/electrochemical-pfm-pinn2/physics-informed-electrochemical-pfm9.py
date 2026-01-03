# app.py - Main Streamlit Application (FULL VERSION WITH NANOSCALE CHARACTERIZATION)
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
import json
import time
from pathlib import Path
import base64
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import warnings
warnings.filterwarnings('ignore')
# Set page configuration FIRST
st.set_page_config(
page_title="LiFePO₄ Hybrid FDM-PINN Assimilation with Nanoscale Characterization",
page_icon="🔬",
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
.technique-card {
background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
color: white;
padding: 1rem;
border-radius: 10px;
margin: 0.5rem;
}
.graph-card {
background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
color: white;
padding: 1rem;
border-radius: 10px;
margin: 0.5rem;
}
</style>
""", unsafe_allow_html=True)
# =====================================================
# 1. ORIGINAL FDM SIMULATION (Full Implementation)
# =====================================================
@st.cache_resource
class PhysicalScalesWithElectrostaticsAndC_Rate:
    """Physical scales with electrostatics and C-rate for LiFePO₄"""
    # Fundamental constants
    R = 8.314462618  # J/(mol·K)
    F = 96485.33212  # C/mol
    k_B = 1.380649e-23  # J/K
    ε0 = 8.854187817e-12  # F/m - Vacuum permittivity

    def __init__(self, c_rate=1.0):
        # Material properties
        self.T = 298.15  # K - Temperature
        # LiFePO₄ phase compositions
        self.c_alpha = 0.03  # FePO₄ phase
        self.c_beta = 0.97  # LiFePO₄ phase
        # Molar volume
        self.V_m = 3.0e-5  # m³/mol
        # Diffusion coefficient
        self.D_b = 1.0e-14  # m²/s - Fast diffusion along b-axis
        # Electrostatic properties
        self.ε_r = 15.0  # Relative permittivity of LiFePO₄
        self.ε = self.ε_r * self.ε0  # Absolute permittivity (F/m)
        # Charge properties
        self.z = 1.0  # Li⁺ charge number
        self.ρ0 = 1.0e6  # Reference charge density (C/m³)
        # Regular solution parameter for LiFePO₄
        self.Ω = 55e3  # J/mol
        # Kinetics parameters
        self.k0_bv = 1.0e-6  # BV rate constant (m/s)
        self.k0_mhc = 5.0e-7  # MHC rate constant (m/s)
        self.alpha = 0.5  # BV symmetry factor
        # Set C-rate parameters
        self.set_c_rate_parameters(c_rate)
        # Set characteristic scales
        self.set_scales()
        # Calculate Debye length
        self.c_ref = 0.5  # Reference concentration
        self.λ_D = self.calculate_debye_length()

    def set_c_rate_parameters(self, c_rate):
        """Set C-rate dependent parameters"""
        self.c_rate = c_rate
        # C-rate scaling factor (1.0 for 1C)
        if c_rate <= 1.0:
            self.c_rate_factor = 1.0
            self.eta_scale = 0.01  # Small overpotential for slow rates
        else:
            self.c_rate_factor = 1.0 + 0.5 * np.log10(c_rate)
            self.eta_scale = 0.01 * c_rate**0.5  # Larger overpotential
        # Rate-dependent interface sharpness
        self.kappa_factor = 1.0 / (1.0 + 0.2 * np.log10(max(1.0, c_rate)))
        # Rate-dependent mobility (effective diffusion)
        self.D_factor = 1.0 / (1.0 + 0.1 * c_rate**0.5)

    def set_scales(self):
        """Set characteristic scales"""
        # Length scale: 10 nm domain
        self.L0 = 1.0e-8  # 10 nm
        # Energy density scale from regular solution
        self.E0 = self.Ω / self.V_m  # J/m³
        # Time scale from diffusion
        self.t0 = (self.L0**2) / self.D_b  # s
        # Mobility scale
        self.M0 = self.D_b / (self.E0 * self.t0)  # m⁵/(J·s)
        # Electric potential scale (thermal voltage)
        self.φ0 = self.R * self.T / self.F  # ~0.0257 V at 298K

    def calculate_debye_length(self):
        """Calculate Debye screening length"""
        c_ref_moles_per_m3 = self.c_ref * (1/self.V_m)  # mol/m³
        λ_D = np.sqrt(self.ε * self.R * self.T / (self.F**2 * c_ref_moles_per_m3))
        return λ_D

    def dimensionless_to_physical(self, W_dim, κ_dim, M_dim, dt_dim):
        """Convert dimensionless to physical"""
        W_phys = W_dim * self.E0
        κ_phys = κ_dim * self.E0 * self.L0**2
        M_phys = M_dim * self.M0
        dt_phys = dt_dim * self.t0
        return W_phys, κ_phys, M_phys, dt_phys

# Numba-accelerated functions (with fallback to numpy if numba not available)
try:
    from numba import njit, prange
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False
    st.warning("⚠️ Numba not installed. Running in pure NumPy mode (slower).")

if USE_NUMBA:
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
        # Laplacian of concentration
        lap_c = compute_laplacian(c, dx)
        # Chemical potential
        mu_chem = chemical_potential(c, A, B, C) - kappa * lap_c
        # Electrostatic contribution
        mu_total = mu_chem + z * F * phi
        # Gradients
        mu_grad_x = compute_gradient_x(mu_total, dx)
        mu_grad_y = compute_gradient_y(mu_total, dx)
        phi_grad_x = compute_gradient_x(phi, dx)
        phi_grad_y = compute_gradient_y(phi, dx)
        c_safe = np.maximum(1e-6, c)
        D_eff = M * R * T / c_safe
        # Diffusive flux
        flux_diff_x = -M * mu_grad_x
        flux_diff_y = -M * mu_grad_y
        if kinetics_type == 0:  # PNP
            # Add migration for PNP
            flux_mig_x = -(D_eff * z * F * c / (R * T)) * phi_grad_x
            flux_mig_y = -(D_eff * z * F * c / (R * T)) * phi_grad_y
            flux_x = flux_diff_x + flux_mig_x
            flux_y = flux_diff_y + flux_mig_y
        else:  # BV or MHC - no migration term
            flux_x = flux_diff_x
            flux_y = flux_diff_y
        # Divergence
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
        # Apply boundary kinetics for BV/MHC at left boundary
        if kinetics_type != 0:  # Not PNP
            for j in prange(ny):
                c_surf = c_new[0, j]
                if kinetics_type == 1:  # BV
                    flux = butler_volmer_flux(eta, c_surf, alpha, k0, F, R, T)
                elif kinetics_type == 2:  # MHC
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
    # Pure NumPy fallback implementations
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
        # Laplacian of concentration
        lap_c = compute_laplacian(c, dx)
        # Chemical potential
        mu_chem = chemical_potential(c, A, B, C) - kappa * lap_c
        # Electrostatic contribution
        mu_total = mu_chem + z * F * phi
        # Gradients
        mu_grad_x = compute_gradient_x(mu_total, dx)
        mu_grad_y = compute_gradient_y(mu_total, dx)
        phi_grad_x = compute_gradient_x(phi, dx)
        phi_grad_y = compute_gradient_y(phi, dx)
        c_safe = np.maximum(1e-6, c)
        D_eff = M * R * T / c_safe
        # Diffusive flux
        flux_diff_x = -M * mu_grad_x
        flux_diff_y = -M * mu_grad_y
        if kinetics_type == 0:  # PNP
            # Add migration for PNP
            flux_mig_x = -(D_eff * z * F * c / (R * T)) * phi_grad_x
            flux_mig_y = -(D_eff * z * F * c / (R * T)) * phi_grad_y
            flux_x = flux_diff_x + flux_mig_x
            flux_y = flux_diff_y + flux_mig_y
        else:  # BV or MHC - no migration term
            flux_x = flux_diff_x
            flux_y = flux_diff_y
        # Divergence
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
        # Apply boundary kinetics for BV/MHC at left boundary
        if kinetics_type != 0:  # Not PNP
            for j in range(ny):
                c_surf = c_new[0, j]
                if kinetics_type == 1:  # BV
                    flux = butler_volmer_flux(eta, c_surf, alpha, k0, F, R, T)
                elif kinetics_type == 2:  # MHC
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
    """Phase field simulation with electrostatics for LiFePO₄"""
    def __init__(self, nx=128, ny=128, dx=1.0, dt=0.01, c_rate=1.0):
        # Simulation grid
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dt = dt
        self.c_rate = c_rate
        # Physical scales with electrostatics and C-rate
        self.scales = PhysicalScalesWithElectrostaticsAndC_Rate(c_rate=c_rate)
        # Dimensionless parameters
        self.W_dim = 1.0
        self.A = self.W_dim
        self.B = -2.0 * self.W_dim
        self.C = self.W_dim
        self.kappa_dim = 2.0 * self.scales.kappa_factor
        self.M_dim = 1.0 * self.scales.D_factor
        # Update physical parameters
        self.update_physical_parameters()
        # Kinetics (0=PNP, 1=BV, 2=MHC)
        self.kinetics_type = 0
        self.eta_left = 0.0
        # Fields
        self.c = np.zeros((nx, ny))  # Concentration
        self.phi = np.zeros((nx, ny))  # Electric potential
        self.Ex = np.zeros((nx, ny))  # Electric field x-component
        self.Ey = np.zeros((nx, ny))  # Electric field y-component
        # Time tracking
        self.time_dim = 0.0
        self.time_phys = 0.0
        self.step = 0
        # History tracking
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
        # Initialize
        self.initialize_random()

    def update_physical_parameters(self):
        """Update physical parameters"""
        self.W_phys, self.kappa_phys, self.M_phys, self.dt_phys = \
            self.scales.dimensionless_to_physical(
                self.W_dim, self.kappa_dim, self.M_dim, self.dt
            )
        # Update double-well coefficients
        self.A = self.W_dim
        self.B = -2.0 * self.W_dim
        self.C = self.W_dim

    def set_parameters(self, W_dim=None, kappa_dim=None, M_dim=None, dt_dim=None,
                       c_rate=None, kinetics_type=None):
        """Set dimensionless parameters"""
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
        """Initialize with random fluctuations"""
        if seed is not None:
            np.random.seed(seed)
        self.c = c0 + noise_amplitude * (2.0 * np.random.random((self.nx, self.ny)) - 1.0)
        self.phi = np.zeros_like(self.c)  # Start with zero potential
        self.eta_left = 0.0
        self.time_dim = 0.0
        self.time_phys = 0.0
        self.step = 0
        self.clear_history()

    def initialize_lithiation(self, noise_amplitude=0.05, seed=None):
        """Initialize for lithiation: Uniform FePO₄ with only random noise"""
        if seed is not None:
            np.random.seed(seed)
        # Start with uniform FePO₄ (c_alpha)
        self.c = self.scales.c_alpha * np.ones((self.nx, self.ny))
        # Add only random noise - NO pre-existing seeds
        self.c += noise_amplitude * (2.0 * np.random.random((self.nx, self.ny)) - 1.0)
        self.c = np.minimum(1.0, np.maximum(0.0, self.c))
        # Apply electric potential gradient (negative for lithiation)
        # Stronger gradient at left boundary to drive insertion
        self.phi = np.zeros_like(self.c)
        for i in range(self.nx):
            # Exponential decay from left boundary
            self.phi[i, :] = -0.2 * np.exp(-i / (self.nx * 0.2))
        self.eta_left = self.scales.eta_scale
        self.time_dim = 0.0
        self.time_phys = 0.0
        self.step = 0
        self.clear_history()

    def initialize_delithiation(self, noise_amplitude=0.05, seed=None):
        """Initialize for delithiation: Uniform LiFePO₄ with only random noise"""
        if seed is not None:
            np.random.seed(seed)
        # Start with uniform LiFePO₄ (c_beta)
        self.c = self.scales.c_beta * np.ones((self.nx, self.ny))
        # Add only random noise - NO pre-existing seeds
        self.c += noise_amplitude * (2.0 * np.random.random((self.nx, self.ny)) - 1.0)
        self.c = np.minimum(1.0, np.maximum(0.0, self.c))
        # Apply electric potential gradient (positive for delithiation)
        # Stronger gradient at left boundary to drive extraction
        self.phi = np.zeros_like(self.c)
        for i in range(self.nx):
            # Exponential decay from left boundary
            self.phi[i, :] = 0.2 * np.exp(-i / (self.nx * 0.2))
        self.eta_left = -self.scales.eta_scale
        self.time_dim = 0.0
        self.time_phys = 0.0
        self.step = 0
        self.clear_history()

    def clear_history(self):
        """Clear history"""
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
        self.update_history()

    def update_history(self):
        """Update history statistics"""
        self.history['time_phys'].append(self.time_phys)
        self.history['mean_c'].append(np.mean(self.c))
        self.history['std_c'].append(np.std(self.c))
        self.history['mean_phi'].append(np.mean(self.phi))
        # Calculate voltage from potential difference
        voltage = np.mean(self.phi[-1, :]) - np.mean(self.phi[0, :])
        self.history['voltage'].append(voltage)
        # Phase fractions
        threshold = 0.5
        self.history['phase_FePO4'].append(np.sum(self.c < threshold) / (self.nx * self.ny))
        self.history['phase_LiFePO4'].append(np.sum(self.c >= threshold) / (self.nx * self.ny))
        # Total charge (simplified)
        total_charge = np.sum(self.c - 0.5)  # Relative to neutral
        self.history['total_charge'].append(total_charge)

    def run_step(self):
        """Run one time step with electrostatics"""
        # Reference concentration (for Poisson equation)
        c_ref = np.mean(self.c)
        # Solve Poisson equation for electric potential
        self.phi = solve_poisson_periodic(
            self.phi, self.c, c_ref, self.dx,
            self.scales.ε, self.scales.F,
            max_iter=50, tol=1e-4
        )
        # Compute electric field
        self.Ex, self.Ey = compute_electric_field(self.phi, self.dx)
        # Determine kinetics parameters
        if self.kinetics_type == 0:  # PNP
            k0 = 0.0  # Dummy
            alpha = 0.0
            eta = 0.0
        else:
            eta = self.eta_left
            alpha = self.scales.alpha
            if self.kinetics_type == 1:  # BV
                k0 = self.scales.k0_bv * self.scales.c_rate_factor
            else:  # MHC
                k0 = self.scales.k0_mhc * self.scales.c_rate_factor
        # Update concentration
        self.c = update_concentration(
            self.c, self.phi, self.dt, self.dx,
            self.kappa_dim, self.M_dim, self.scales.D_b * self.scales.D_factor,
            self.A, self.B, self.C,
            self.scales.z, self.scales.F, self.scales.R, self.scales.T,
            self.kinetics_type, k0, alpha, eta
        )
        # Update time
        self.time_dim += self.dt
        self.time_phys += self.dt_phys
        self.step += 1
        # Update history
        self.update_history()

    def run_steps(self, n_steps):
        """Run multiple time steps"""
        for _ in range(n_steps):
            self.run_step()

    def run_until(self, target_time_phys):
        """Run until reaching target physical time"""
        steps_needed = max(1, int((target_time_phys - self.time_phys) / self.dt_phys))
        self.run_steps(steps_needed)

    def compute_free_energy_density(self):
        """Compute free energy density"""
        energy = double_well_energy(self.c, self.A, self.B, self.C)
        # Add gradient energy
        grad_x = compute_gradient_x(self.c, self.dx)
        grad_y = compute_gradient_y(self.c, self.dx)
        grad_sq = grad_x**2 + grad_y**2
        energy += 0.5 * self.kappa_dim * grad_sq
        # Add electrostatic energy
        energy += 0.5 * self.scales.ε * self.phi**2
        return energy

    def compute_electrochemical_potential(self):
        """Compute total electrochemical potential"""
        lap_c = compute_laplacian(self.c, self.dx)
        mu_chem = chemical_potential(self.c, self.A, self.B, self.C) - self.kappa_dim * lap_c
        mu_el = self.scales.z * self.scales.F * self.phi
        return mu_chem + mu_el

    def get_statistics(self):
        """Get comprehensive statistics"""
        stats = {
            'time_phys': self.time_phys,
            'step': self.step,
            'mean_c': np.mean(self.c),
            'std_c': np.std(self.c),
            'x_Li': np.mean(self.c),
            'mean_phi': np.mean(self.phi),
            'max_phi': np.max(self.phi),
            'min_phi': np.min(self.phi),
            'mean_E': np.mean(np.sqrt(self.Ex**2 + self.Ey**2)),
            'voltage': np.mean(self.phi[-1, :]) - np.mean(self.phi[0, :]),
            'phase_FePO4': np.sum(self.c < 0.5) / (self.nx * self.ny),
            'phase_LiFePO4': np.sum(self.c >= 0.5) / (self.nx * self.ny),
            'domain_size_nm': self.nx * self.dx * self.scales.L0 * 1e9,
            'interface_width_nm': np.sqrt(self.kappa_phys / self.W_phys) * 1e9,
            'debye_length_nm': self.scales.λ_D * 1e9,
            'W_dim': self.W_dim,
            'kappa_dim': self.kappa_dim,
            'M_dim': self.M_dim,
            'W_phys': self.W_phys,
            'kappa_phys': self.kappa_phys,
            'M_phys': self.M_phys,
            'dt_phys': self.dt_phys,
            'c_rate': self.c_rate,
            'kinetics_type': self.kinetics_type
        }
        return stats

# =====================================================
# 2. ENHANCED NANOSCALE CHARACTERIZATION SIMULATOR
# =====================================================
@st.cache_resource
class NanoscaleCharacterizationSimulator:
    """Generate realistic synthetic observations for different nanoscale characterization techniques"""
    def __init__(self):
        # Enhanced characteristics for each technique
        self.technique_characteristics = {
            'xrd_mapping': {
                'description': 'X-ray Diffraction Mapping',
                'coverage': 0.05,
                'noise_std': 0.03,
                'pattern': 'grid',
                'resolution_nm': 50,  # Spatial resolution in nm
                'sampling_pattern': 'uniform_grid',
                'depth_sensitivity': 'bulk',  # Penetrates entire thickness
                'measurement_time': 300,  # Seconds per measurement
                'probe_size_nm': 100,  # X-ray beam size
                'intensity_variation': 0.1,
                'background_level': 0.05,
                'sensitivity_to_crystal_structure': True,
                'characteristic_features': [
                    'phase_specific_peaks',
                    'crystal_orientation_dependence',
                    'lattice_strain_sensitivity',
                    'composition_mapping'
                ]
            },
            'sem_microscopy': {
                'description': 'Scanning Electron Microscopy',
                'coverage': 0.15,
                'noise_std': 0.08,
                'pattern': 'raster_scan',
                'resolution_nm': 10,
                'sampling_pattern': 'raster_lines',
                'depth_sensitivity': 'surface',  # Top 1-10 nm
                'measurement_time': 60,
                'probe_size_nm': 5,
                'intensity_variation': 0.15,
                'background_level': 0.08,
                'sensitivity_to_topography': True,
                'characteristic_features': [
                    'surface_morphology',
                    'edge_enhancement',
                    'shadow_effects',
                    'charging_artifacts'
                ]
            },
            'tem_microscopy': {
                'description': 'Transmission Electron Microscopy',
                'coverage': 0.02,
                'noise_std': 0.05,
                'pattern': 'thin_foil',
                'resolution_nm': 0.2,
                'sampling_pattern': 'high_res_grid',
                'depth_sensitivity': 'projection',  # Through entire thickness
                'measurement_time': 600,
                'probe_size_nm': 1,
                'intensity_variation': 0.08,
                'background_level': 0.03,
                'sensitivity_to_lattice': True,
                'characteristic_features': [
                    'atomic_resolution',
                    'diffraction_contrast',
                    'phase_contrast',
                    'beam_damage_effects'
                ]
            },
            'afm': {
                'description': 'Atomic Force Microscopy',
                'coverage': 0.08,
                'noise_std': 0.06,
                'pattern': 'topography_scan',
                'resolution_nm': 1,
                'sampling_pattern': 'contour_lines',
                'depth_sensitivity': 'surface_topography',
                'measurement_time': 120,
                'probe_size_nm': 10,
                'intensity_variation': 0.12,
                'background_level': 0.04,
                'sensitivity_to_height': True,
                'characteristic_features': [
                    'topography_mapping',
                    'phase_imaging',
                    'force_curves',
                    'tip_convolution_artifacts'
                ]
            },
            'xray_tomography': {
                'description': 'X-ray Computed Tomography',
                'coverage': 0.25,
                'noise_std': 0.07,
                'pattern': 'volume_projection',
                'resolution_nm': 100,
                'sampling_pattern': 'sinogram_projection',
                'depth_sensitivity': 'volumetric',
                'measurement_time': 1800,
                'probe_size_nm': 1000,
                'intensity_variation': 0.05,
                'background_level': 0.06,
                'sensitivity_to_density': True,
                'characteristic_features': [
                    '3d_volume_reconstruction',
                    'absorption_contrast',
                    'phase_retrieval',
                    'reconstruction_artifacts'
                ]
            },
            'neutron_diffraction': {
                'description': 'Neutron Powder Diffraction',
                'coverage': 0.01,
                'noise_std': 0.02,
                'pattern': 'powder_average',
                'resolution_nm': 1000,  # Bulk average
                'sampling_pattern': 'ensemble_average',
                'depth_sensitivity': 'bulk_penetrating',
                'measurement_time': 3600,
                'probe_size_nm': 5000,
                'intensity_variation': 0.03,
                'background_level': 0.02,
                'sensitivity_to_light_elements': True,
                'characteristic_features': [
                    'isotope_sensitivity',
                    'magnetic_structure',
                    'hydrogen_detection',
                    'bulk_average'
                ]
            }
        }

    def generate_xrd_pattern(self, c_field: np.ndarray, lattice_parameter: float = 10.3) -> np.ndarray:
        """
        Simulate XRD-specific patterns including:
        - Phase-dependent peak positions (FePO4 vs LiFePO4 have different lattice parameters)
        - Preferred orientation effects
        - Peak broadening from strain/crystallite size
        """
        nx, ny = c_field.shape
        # Simulate different crystallographic phases
        # FePO4: smaller lattice parameter (~10.0 Å)
        # LiFePO4: larger lattice parameter (~10.3 Å)
        lattice_param_field = 10.0 + 0.3 * c_field  # Scale with Li content
        # Create synthetic diffraction intensity
        intensity = np.zeros_like(c_field)
        # Peak at 2θ ~ 25° for (200) reflection
        # Simulate peak shifting with composition
        d_spacing = lattice_param_field / 2.0  # Simplified for (200) reflection
        # Add preferred orientation effects (texture)
        grad_x = compute_gradient_x(c_field, 1.0)
        grad_y = compute_gradient_y(c_field, 1.0)
        texture_factor = 0.5 + 0.5 * np.cos(4 * np.arctan2(grad_y, grad_x))  # 4-fold symmetry for orthorhombic
        # Peak broadening from crystallite size (Scherrer equation)
        # Smaller crystallites in phase boundaries
        crystallite_size = 50 + 150 * (1 - np.abs(grad_x + grad_y))
        peak_width = 0.9 + 0.1 / (crystallite_size + 1e-6)
        # XRD intensity model
        intensity = texture_factor * np.exp(-((d_spacing - 5.15)**2) / (2 * peak_width**2))
        return intensity / (intensity.max() + 1e-10)

    def generate_microscopy_image(self, c_field: np.ndarray, technique: str = 'sem') -> np.ndarray:
        """
        Simulate microscopy images with realistic artifacts
        """
        nx, ny = c_field.shape
        if technique == 'sem':
            # SEM: surface topography and composition contrast
            # Simulate edge enhancement
            grad_x = compute_gradient_x(c_field, 1.0)
            grad_y = compute_gradient_y(c_field, 1.0)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            # Edge enhancement (typical in SEM)
            edge_enhancement = 1.0 + 2.0 * grad_mag
            # Charging artifacts near insulating regions
            # FePO4 is more insulating than LiFePO4
            charging = np.where(c_field < 0.5, 1.2, 1.0)
            # Shadow effects (directional illumination)
            shadow = 1.0 - 0.3 * np.sin(2*np.pi*np.arange(nx)[:, None]/nx) * np.sin(2*np.pi*np.arange(ny)/ny)
            image = c_field * edge_enhancement * charging * shadow
        elif technique == 'tem':
            # TEM: phase contrast and diffraction contrast
            # Phase contrast (mass-thickness)
            phase_contrast = np.sin(10 * c_field)
            # Diffraction contrast - sensitive to crystal orientation
            orientation_map = np.mod(np.arange(nx)[:, None] * 0.1 + np.arange(ny) * 0.1, 1.0)
            diffraction_contrast = 0.5 + 0.5 * np.cos(2*np.pi*orientation_map)
            # Beam damage simulation (localized)
            beam_damage = 1.0 - 0.1 * np.exp(-((nx//2 - np.arange(nx)[:, None])**2 + (ny//2 - np.arange(ny))**2) / (2*50**2))
            image = c_field * diffraction_contrast * beam_damage + 0.3 * phase_contrast
        else:
            # Default: simple contrast
            image = c_field
        return image / (image.max() + 1e-10)

    def generate_afm_topography(self, c_field: np.ndarray) -> np.ndarray:
        """
        Simulate AFM topography with realistic tip convolution effects
        """
        nx, ny = c_field.shape
        # Phase separation creates height variations
        # LiFePO4 and FePO4 have different molar volumes
        height = 0.5 + 0.1 * (c_field - 0.5)  # Height difference between phases
        # Add surface roughness
        roughness = 0.02 * np.random.randn(nx, ny)
        # Tip convolution: real AFM tips have finite size
        tip_radius = 10  # pixels
        x = np.arange(-tip_radius, tip_radius+1)
        y = np.arange(-tip_radius, tip_radius+1)
        X, Y = np.meshgrid(x, y)
        kernel = np.exp(-(X**2 + Y**2) / (2*(tip_radius/3)**2))
        kernel = kernel / kernel.sum()
        # Convolve with tip shape
        from scipy.signal import convolve2d
        topography = convolve2d(height + roughness, kernel, mode='same')
        # Add scan line artifacts (common in AFM)
        scan_lines = 1.0 + 0.05 * np.sin(2*np.pi*np.arange(nx)[:, None]/10)
        return topography * scan_lines

    def generate_tomography_sinogram(self, c_field_3d: np.ndarray, n_angles: int = 180) -> np.ndarray:
        """
        Simulate X-ray tomography projections (sinograms)
        """
        nz, nx, ny = c_field_3d.shape
        # Create sinogram by projecting at different angles
        sinogram = np.zeros((n_angles, nx))
        angles = np.linspace(0, np.pi, n_angles, endpoint=False)
        for i, angle in enumerate(angles):
            # Simple projection (Radon transform approximation)
            rotated = self.rotate_slice(c_field_3d[nz//2], angle)
            sinogram[i, :] = rotated.sum(axis=1)
        return sinogram

    def rotate_slice(self, slice_2d: np.ndarray, angle: float) -> np.ndarray:
        """Rotate 2D slice by angle (simplified)"""
        from scipy.ndimage import rotate
        return rotate(slice_2d, angle * 180/np.pi, reshape=False, order=1)

    def generate_sparse_observations(
        self,
        c_field: np.ndarray,
        dx: float,
        L0: float,
        observation_type: str = 'xrd_mapping',
        measurement_time: float = 0.0,
        seed: int = None,
        custom_coverage: float = None,
        custom_noise: float = None,
        include_artifacts: bool = True
    ) -> Dict:
        """
        Generate realistic sparse observations for specific characterization technique
        """
        if seed is not None:
            np.random.seed(seed)
        nx, ny = c_field.shape
        config = self.technique_characteristics[observation_type].copy()
        # Override with custom values if provided
        if custom_coverage is not None:
            config['coverage'] = custom_coverage
        if custom_noise is not None:
            config['noise_std'] = custom_noise
        # Technique-specific pattern generation
        if config['pattern'] == 'uniform_grid':
            # XRD: uniform grid with beam size consideration
            beam_size_pixels = max(1, int(config['probe_size_nm'] / (dx * L0 * 1e9)))
            step = max(beam_size_pixels, int(1 / np.sqrt(config['coverage'])))
            obs_i, obs_j = np.meshgrid(
                np.arange(beam_size_pixels//2, nx, step),
                np.arange(beam_size_pixels//2, ny, step),
                indexing='ij'
            )
            obs_i = obs_i.flatten()
            obs_j = obs_j.flatten()
        elif config['pattern'] == 'raster_lines':
            # SEM: raster scanning pattern
            n_lines = max(1, int(np.sqrt(config['coverage'] * nx * ny / ny)))
            line_indices = np.linspace(0, nx-1, n_lines).astype(int)
            obs_i = np.repeat(line_indices, ny)
            obs_j = np.tile(np.arange(ny), n_lines)
        elif config['pattern'] == 'high_res_grid':
            # TEM: high-resolution but sparse grid
            step = max(5, int(1 / np.sqrt(config['coverage'] * 4)))  # Very sparse for TEM
            obs_i, obs_j = np.meshgrid(
                np.arange(0, nx, step),
                np.arange(0, ny, step),
                indexing='ij'
            )
            obs_i = obs_i.flatten()
            obs_j = obs_j.flatten()
        elif config['pattern'] == 'contour_lines':
            # AFM: contour following pattern
            n_contours = max(2, int(np.sqrt(config['coverage'] * nx * ny / nx)))
            obs_i = []
            obs_j = []
            for contour_level in np.linspace(0, 1, n_contours):
                # Find contour points (simplified)
                mask = np.abs(c_field - contour_level) < 0.05
                indices = np.where(mask)
                if len(indices[0]) > 0:
                    # Sample points along contour
                    n_samples = min(100, len(indices[0]))
                    sample_idx = np.random.choice(len(indices[0]), n_samples, replace=False)
                    obs_i.extend(indices[0][sample_idx])
                    obs_j.extend(indices[1][sample_idx])
            obs_i = np.array(obs_i)
            obs_j = np.array(obs_j)
        elif config['pattern'] == 'sinogram_projection':
            # Tomography: projection lines
            n_projections = max(3, int(np.sqrt(config['coverage'] * nx)))
            obs_i = []
            obs_j = []
            for angle in np.linspace(0, np.pi, n_projections, endpoint=False):
                # Projection line through center
                center_x, center_y = nx//2, ny//2
                length = min(nx, ny)
                for r in np.linspace(-length//2, length//2, ny//2):
                    i = int(center_x + r * np.cos(angle))
                    j = int(center_y + r * np.sin(angle))
                    if 0 <= i < nx and 0 <= j < ny:
                        obs_i.append(i)
                        obs_j.append(j)
            obs_i = np.array(obs_i)
            obs_j = np.array(obs_j)
        else:
            # Default: random sampling
            n_obs = max(1, int(config['coverage'] * nx * ny))
            obs_indices = np.random.choice(nx * ny, n_obs, replace=False)
            obs_i = obs_indices // ny
            obs_j = obs_indices % ny
        # Ensure we have at least some observations
        if len(obs_i) == 0:
            obs_i = np.array([nx//2])
            obs_j = np.array([ny//2])
        # Technique-specific signal generation
        if observation_type == 'xrd_mapping':
            c_obs = self.generate_xrd_pattern(c_field)[obs_i, obs_j]
        elif observation_type in ['sem_microscopy', 'tem_microscopy']:
            technique = 'sem' if 'sem' in observation_type else 'tem'
            c_obs = self.generate_microscopy_image(c_field, technique)[obs_i, obs_j]
        elif observation_type == 'afm':
            c_obs = self.generate_afm_topography(c_field)[obs_i, obs_j]
        else:
            c_obs = c_field[obs_i, obs_j]
        # Add technique-specific noise
        if include_artifacts:
            if observation_type == 'xrd_mapping':
                # XRD: Poisson noise (counting statistics)
                noise = np.random.poisson(c_obs * 100) / 100 - c_obs
                noise += config['noise_std'] * np.random.randn(*c_obs.shape)
            elif observation_type == 'sem_microscopy':
                # SEM: shot noise and line noise
                shot_noise = np.sqrt(np.abs(c_obs)) * np.random.randn(*c_obs.shape) * 0.1
                line_noise = 0.05 * np.sin(2*np.pi*obs_i/10)  # Scan line artifacts
                noise = shot_noise + line_noise
            elif observation_type == 'tem_microscopy':
                # TEM: lower noise but beam damage effects
                noise = config['noise_std'] * np.random.randn(*c_obs.shape)
                # Beam damage increases with time
                beam_damage = 0.1 * (1 - np.exp(-measurement_time/600))
                noise *= (1 + beam_damage)
            elif observation_type == 'afm':
                # AFM: correlated noise (drift)
                noise = config['noise_std'] * np.random.randn(*c_obs.shape)
                # Add slow drift
                drift = 0.02 * np.sin(2*np.pi*measurement_time/300) * (obs_i / nx)
                noise += drift
            elif observation_type == 'xray_tomography':
                # Tomography: structured noise
                noise = config['noise_std'] * np.random.randn(*c_obs.shape)
                # Ring artifacts common in tomography
                ring_radius = np.sqrt((obs_i - nx//2)**2 + (obs_j - ny//2)**2)
                ring_artifacts = 0.03 * np.sin(2*np.pi*ring_radius/20)
                noise += ring_artifacts
            else:
                noise = config['noise_std'] * np.random.randn(*c_obs.shape)
        else:
            noise = config['noise_std'] * np.random.randn(*c_obs.shape)
        c_obs_noisy = np.clip(c_obs + noise, 0, 1)
        # Convert to physical coordinates
        x_phys = obs_i * dx * L0
        y_phys = obs_j * dx * L0
        # Add measurement metadata
        metadata = {
            'technique': observation_type,
            'description': config['description'],
            'resolution_nm': config['resolution_nm'],
            'probe_size_nm': config['probe_size_nm'],
            'measurement_time_s': measurement_time,
            'depth_sensitivity': config['depth_sensitivity'],
            'characteristic_features': config['characteristic_features'],
            'sensitivity_to': ['composition'] + (['crystal_structure'] if config.get('sensitivity_to_crystal_structure', False) else []) +
            (['topography'] if config.get('sensitivity_to_topography', False) else []) +
            (['density'] if config.get('sensitivity_to_density', False) else []),
            'artifacts_included': include_artifacts,
            'spatial_sampling_pattern': config['sampling_pattern'],
            'signal_type': 'direct_composition' if observation_type not in ['xrd_mapping', 'afm'] else 'derived_signal'
        }
        return {
            'time_phys': measurement_time,
            'x_coords': x_phys,
            'y_coords': y_phys,
            'c_obs': c_obs_noisy,
            'c_true': c_field[obs_i, obs_j],  # Ground truth for validation
            'x_idx': obs_i,
            'y_idx': obs_j,
            'noise_std': config['noise_std'],
            'coverage': len(obs_i) / (nx * ny),
            'observation_type': observation_type,
            'technique_metadata': metadata,
            'full_field': c_field,  # For validation only
            'measurement_quality_score': self.calculate_quality_score(config, len(obs_i), np.std(noise))
        }

    def calculate_quality_score(self, config: Dict, n_points: int, noise_level: float) -> float:
        """Calculate a quality score for the measurement"""
        # Factors: coverage, resolution, noise, measurement time
        coverage_score = min(1.0, n_points / 1000)  # Normalized by expected points
        resolution_score = 1.0 / (1.0 + config['resolution_nm'] / 100)  # Better resolution -> higher score
        noise_score = 1.0 / (1.0 + 10 * noise_level)
        time_score = 1.0 / (1.0 + config['measurement_time'] / 3600)  # Shorter time better
        weights = {'coverage': 0.3, 'resolution': 0.3, 'noise': 0.2, 'time': 0.2}
        score = (weights['coverage'] * coverage_score +
                 weights['resolution'] * resolution_score +
                 weights['noise'] * noise_score +
                 weights['time'] * time_score)
        return min(1.0, max(0.0, score))

    def generate_composite_observations(self, c_field: np.ndarray, dx: float, L0: float,
                                        techniques: List[str] = None) -> List[Dict]:
        """Generate observations from multiple techniques"""
        if techniques is None:
            techniques = ['xrd_mapping', 'sem_microscopy', 'afm']
        composite_data = []
        for technique in techniques:
            obs_data = self.generate_sparse_observations(
                c_field, dx, L0, technique, include_artifacts=True
            )
            composite_data.append(obs_data)
        return composite_data

    def get_technique_comparison(self) -> pd.DataFrame:
        """Get comparison table of all techniques"""
        data = []
        for tech_name, config in self.technique_characteristics.items():
            row = {
                'Technique': config['description'],
                'Spatial Resolution': f"{config['resolution_nm']} nm",
                'Coverage': f"{config['coverage']*100:.1f}%",
                'Noise Level': f"{config['noise_std']*100:.1f}%",
                'Measurement Time': f"{config['measurement_time']}s",
                'Depth Sensitivity': config['depth_sensitivity'],
                'Key Features': ', '.join(config['characteristic_features'][:2])
            }
            data.append(row)
        return pd.DataFrame(data)

# =====================================================
# 3. SYNTHETIC GRAPH GENERATOR
# =====================================================
@st.cache_resource
class SyntheticGraphGenerator:
    """
    Generate synthetic graphs representing relationships between measurement points
    based on nanoscale characterization techniques
    """
    def __init__(self):
        self.graph_types = {
            'spatial_knn': {
                'description': 'k-Nearest Neighbors in physical space',
                'edge_weighting': 'distance_inverse',
                'max_neighbors': 8,
                'use_technique_specific': False
            },
            'phase_similarity': {
                'description': 'Connect points with similar phase composition',
                'edge_weighting': 'composition_similarity',
                'similarity_threshold': 0.1,
                'use_technique_specific': True
            },
            'diffraction_correlation': {
                'description': 'Connect XRD points with similar diffraction patterns',
                'edge_weighting': 'peak_correlation',
                'correlation_threshold': 0.7,
                'use_technique_specific': True,
                'technique': 'xrd_mapping'
            },
            'topography_continuity': {
                'description': 'Connect AFM points along surface contours',
                'edge_weighting': 'height_continuity',
                'gradient_threshold': 0.05,
                'use_technique_specific': True,
                'technique': 'afm'
            },
            'multi_technique_fusion': {
                'description': 'Fuse connections from multiple techniques',
                'edge_weighting': 'confidence_weighted',
                'fusion_method': 'weighted_average',
                'use_technique_specific': True
            }
        }

    def build_spatial_graph(self, obs_data: Dict, graph_type: str = 'spatial_knn') -> Dict:
        """
        Build a graph representation of the observations
        Returns: nodes, edges, and adjacency information
        """
        n_points = len(obs_data['x_coords'])
        technique = obs_data['observation_type']
        # Node features
        nodes = {
            'positions': np.column_stack([obs_data['x_coords'], obs_data['y_coords']]),
            'concentrations': obs_data['c_obs'],
            'indices': np.column_stack([obs_data['x_idx'], obs_data['y_idx']]),
            'technique': [technique] * n_points,
            'measurement_quality': np.ones(n_points) * obs_data.get('measurement_quality_score', 0.8)
        }
        # Build edges based on graph type
        if graph_type == 'spatial_knn':
            edges = self._build_knn_graph(nodes['positions'],
                                          self.graph_types[graph_type]['max_neighbors'])
        elif graph_type == 'phase_similarity':
            edges = self._build_similarity_graph(nodes['concentrations'],
                                                 nodes['positions'],
                                                 self.graph_types[graph_type]['similarity_threshold'])
        elif graph_type == 'diffraction_correlation' and technique == 'xrd_mapping':
            # Simulate diffraction pattern correlations
            simulated_patterns = self._simulate_diffraction_patterns(nodes['indices'],
                                                                     obs_data['full_field'])
            edges = self._build_correlation_graph(simulated_patterns,
                                                  self.graph_types[graph_type]['correlation_threshold'])
        elif graph_type == 'topography_continuity' and technique == 'afm':
            # Connect along topographic contours
            edges = self._build_contour_graph(nodes['positions'],
                                              nodes['concentrations'])
        elif graph_type == 'multi_technique_fusion':
            # For composite observations
            edges = self._build_fusion_graph(nodes)
        else:
            # Default: spatial KNN
            edges = self._build_knn_graph(nodes['positions'], 6)
        # Calculate edge weights
        edge_weights = self._calculate_edge_weights(edges, nodes, graph_type, technique)
        graph = {
            'nodes': nodes,
            'edges': edges,
            'edge_weights': edge_weights,
            'graph_type': graph_type,
            'technique': technique,
            'num_nodes': n_points,
            'num_edges': len(edges),
            'average_degree': len(edges) / n_points if n_points > 0 else 0
        }
        return graph

    def _build_knn_graph(self, positions: np.ndarray, k: int) -> np.ndarray:
        """Build k-nearest neighbor graph"""
        try:
            from sklearn.neighbors import kneighbors_graph
            n_points = len(positions)
            k = min(k, n_points - 1)
            if n_points <= 1:
                return np.array([], dtype=int).reshape(0, 2)
            # Use Euclidean distance
            graph = kneighbors_graph(positions, k, mode='connectivity', include_self=False)
            # Convert to edge list
            edges = np.array(graph.nonzero()).T
            return edges
        except ImportError:
            # Fallback if sklearn not available
            n_points = len(positions)
            edges = []
            for i in range(n_points):
                # Simple KNN without sklearn
                distances = np.linalg.norm(positions - positions[i], axis=1)
                # Exclude self
                distances[i] = np.inf
                # Get k nearest
                nearest = np.argsort(distances)[:k]
                for j in nearest:
                    edges.append([i, j])
            return np.array(edges) if edges else np.array([], dtype=int).reshape(0, 2)

    def _build_similarity_graph(self, concentrations: np.ndarray, positions: np.ndarray,
                                threshold: float) -> np.ndarray:
        """Connect points with similar concentrations"""
        n_points = len(concentrations)
        edges = []
        for i in range(n_points):
            for j in range(i+1, n_points):
                # Similarity based on concentration and proximity
                conc_sim = 1.0 - abs(concentrations[i] - concentrations[j])
                dist = np.linalg.norm(positions[i] - positions[j])
                spatial_weight = np.exp(-dist / (np.mean(np.linalg.norm(positions, axis=1)) + 1e-10))
                if conc_sim > threshold and spatial_weight > 0.3:
                    edges.append([i, j])
        return np.array(edges) if edges else np.array([], dtype=int).reshape(0, 2)

    def _simulate_diffraction_patterns(self, indices: np.ndarray, c_field: np.ndarray) -> np.ndarray:
        """Simulate XRD patterns for each point"""
        n_points = len(indices)
        patterns = []
        for idx in range(n_points):
            i, j = indices[idx]
            # Extract local region
            i_min = max(0, i-5)
            i_max = min(c_field.shape[0], i+6)
            j_min = max(0, j-5)
            j_max = min(c_field.shape[1], j+6)
            local_field = c_field[i_min:i_max, j_min:j_max]
            # Simple diffraction pattern simulation
            if local_field.size > 0:
                # Create synthetic diffraction peaks
                mean_conc = local_field.mean()
                # Simulate two main peaks that shift with composition
                peak1_pos = 20 + 5 * mean_conc  # Shifts with Li content
                peak2_pos = 35 + 3 * mean_conc
                # Create pattern with Gaussian peaks
                x = np.linspace(10, 50, 100)
                pattern = (np.exp(-(x - peak1_pos)**2 / 2) +
                           0.5 * np.exp(-(x - peak2_pos)**2 / 1.5))
                patterns.append(pattern)
            else:
                patterns.append(np.zeros(100))
        return np.array(patterns)

    def _build_correlation_graph(self, patterns: np.ndarray, threshold: float) -> np.ndarray:
        """Connect points with correlated diffraction patterns"""
        from scipy.stats import pearsonr
        n_points = len(patterns)
        edges = []
        # Calculate correlation matrix
        corr_matrix = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(i+1, n_points):
                corr, _ = pearsonr(patterns[i], patterns[j])
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
        # Connect highly correlated points
        for i in range(n_points):
            for j in range(i+1, n_points):
                if corr_matrix[i, j] > threshold:
                    edges.append([i, j])
        return np.array(edges) if edges else np.array([], dtype=int).reshape(0, 2)

    def _build_contour_graph(self, positions: np.ndarray, concentrations: np.ndarray) -> np.ndarray:
        """Connect points along topographic contours (for AFM)"""
        n_points = len(positions)
        edges = []
        # Connect points along contours of similar height
        for i in range(n_points):
            current_conc = concentrations[i]
            # Find points with similar concentration
            similar_idx = np.where(np.abs(concentrations - current_conc) < 0.05)[0]
            for j in similar_idx:
                if i != j:
                    # Also check spatial proximity
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < np.mean(np.linalg.norm(positions, axis=1)) * 0.5:
                        edges.append([i, j])
        return np.array(edges) if edges else np.array([], dtype=int).reshape(0, 2)

    def _build_fusion_graph(self, nodes: Dict) -> np.ndarray:
        """Fuse multiple graph types"""
        # This is a simplified version - in practice would combine multiple graphs
        positions = nodes['positions']
        concentrations = nodes['concentrations']
        # Combine KNN and similarity graphs
        knn_edges = self._build_knn_graph(positions, 6)
        sim_edges = self._build_similarity_graph(concentrations, positions, 0.1)
        # Merge edges
        all_edges = set()
        for edge in knn_edges:
            all_edges.add(tuple(sorted(edge)))
        for edge in sim_edges:
            all_edges.add(tuple(sorted(edge)))
        return np.array(list(all_edges)) if all_edges else np.array([], dtype=int).reshape(0, 2)

    def _calculate_edge_weights(self, edges: np.ndarray, nodes: Dict,
                                graph_type: str, technique: str) -> np.ndarray:
        """Calculate weights for edges based on graph type and technique"""
        if len(edges) == 0:
            return np.array([])
        weights = []
        positions = nodes['positions']
        concentrations = nodes['concentrations']
        for edge in edges:
            i, j = edge
            if graph_type == 'spatial_knn':
                # Weight inversely proportional to distance
                dist = np.linalg.norm(positions[i] - positions[j])
                weight = 1.0 / (1.0 + dist)
            elif graph_type == 'phase_similarity':
                # Weight by composition similarity
                conc_diff = abs(concentrations[i] - concentrations[j])
                weight = 1.0 - conc_diff
            elif graph_type == 'diffraction_correlation':
                # For XRD: weight by both correlation and spatial proximity
                dist = np.linalg.norm(positions[i] - positions[j])
                conc_sim = 1.0 - abs(concentrations[i] - concentrations[j])
                weight = 0.7 * conc_sim + 0.3 * np.exp(-dist)
            elif graph_type == 'topography_continuity':
                # For AFM: weight by height continuity
                height_diff = abs(concentrations[i] - concentrations[j])
                dist = np.linalg.norm(positions[i] - positions[j])
                weight = np.exp(-height_diff/0.05) * np.exp(-dist/1e-7)
            else:
                # Default: combination
                dist = np.linalg.norm(positions[i] - positions[j])
                conc_sim = 1.0 - abs(concentrations[i] - concentrations[j])
                weight = 0.5 * conc_sim + 0.5 * np.exp(-dist)
            weights.append(weight)
        return np.array(weights)

    def visualize_graph(self, graph: Dict, obs_data: Dict = None) -> go.Figure:
        """Visualize the synthetic graph"""
        nodes = graph['nodes']
        edges = graph['edges']
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Graph Visualization', 'Node Concentration Distribution'),
            specs=[[{'type': 'scatter'}, {'type': 'histogram'}]]
        )
        # Plot nodes
        fig.add_trace(
            go.Scatter(
                x=nodes['positions'][:, 0],
                y=nodes['positions'][:, 1],
                mode='markers',
                marker=dict(
                    size=10,
                    color=nodes['concentrations'],
                    colorscale='RdYlBu',
                    showscale=True,
                    colorbar=dict(title="Concentration")
                ),
                text=[f"Node {i}: c={nodes['concentrations'][i]:.3f}"
                      for i in range(len(nodes['concentrations']))],
                hoverinfo='text',
                name='Nodes'
            ),
            row=1, col=1
        )
        # Plot edges
        if len(edges) > 0:
            edge_x = []
            edge_y = []
            for edge in edges:
                x0, y0 = nodes['positions'][edge[0]]
                x1, y1 = nodes['positions'][edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            fig.add_trace(
                go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode='lines',
                    line=dict(width=1, color='gray'),
                    hoverinfo='none',
                    showlegend=False
                ),
                row=1, col=1
            )
        # Histogram of node concentrations
        fig.add_trace(
            go.Histogram(
                x=nodes['concentrations'],
                nbinsx=20,
                marker_color='skyblue',
                opacity=0.7,
                name='Concentration Distribution'
            ),
            row=1, col=2
        )
        fig.update_layout(
            title=f"Graph: {graph['graph_type']} | Technique: {graph['technique']} | "
                  f"Nodes: {graph['num_nodes']}, Edges: {graph['num_edges']}",
            height=500,
            showlegend=True
        )
        fig.update_xaxes(title_text="X Position (m)", row=1, col=1)
        fig.update_yaxes(title_text="Y Position (m)", row=1, col=1)
        fig.update_xaxes(title_text="Concentration", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        return fig

# =====================================================
# 4. TECHNIQUE REASONING CLASS
# =====================================================
class TechniqueReasoning:
    """
    Provides logical reasoning about why different characterization techniques
    produce distinct synthetic graph features
    """
    @staticmethod
    def explain_technique_differences():
        """Explain the distinct features of each technique in synthetic graphs"""
        reasoning = {
            'xrd_mapping': {
                'spatial_pattern': 'Grid pattern due to beam rastering',
                'signal_characteristics': 'Phase-sensitive, bulk-averaging, lattice parameter dependent',
                'graph_features': 'Edges connect points with similar crystal structure, not just spatial proximity',
                'noise_characteristics': 'Poisson statistics due to photon counting, lower noise at high intensity',
                'limitations': 'Poor spatial resolution (50-100 nm), cannot resolve individual particles',
                'advantages': 'Quantitative phase analysis, strain mapping, crystallographic orientation'
            },
            'sem_microscopy': {
                'spatial_pattern': 'Raster scan lines, sequential acquisition',
                'signal_characteristics': 'Surface-sensitive, topographic contrast, compositional contrast',
                'graph_features': 'Edges follow surface features, enhanced at boundaries',
                'noise_characteristics': 'Shot noise dominant, line noise from scanning',
                'limitations': 'Surface only, charging artifacts in insulating regions',
                'advantages': 'High spatial resolution, large field of view, fast acquisition'
            },
            'tem_microscopy': {
                'spatial_pattern': 'Very sparse high-resolution grid',
                'signal_characteristics': 'Atomic resolution, phase contrast, diffraction contrast',
                'graph_features': 'Very few but highly informative nodes, local crystallographic correlations',
                'noise_characteristics': 'Low noise but beam damage effects progressive',
                'limitations': 'Very small field of view, sample preparation difficult, beam sensitive',
                'advantages': 'Atomic resolution, direct imaging of lattice, chemical analysis possible'
            },
            'afm': {
                'spatial_pattern': 'Contour following, line scanning',
                'signal_characteristics': 'Topographic height, phase imaging, mechanical properties',
                'graph_features': 'Edges follow topographic contours, strong local connectivity',
                'noise_characteristics': 'Drift over time, tip convolution artifacts',
                'limitations': 'Slow scanning, tip wear, limited field of view',
                'advantages': 'True 3D topography, operates in various environments, no charging issues'
            },
            'xray_tomography': {
                'spatial_pattern': 'Projection-based, sinogram sampling',
                'signal_characteristics': 'Volumetric, density contrast, absorption-based',
                'graph_features': '3D connectivity, cross-sectional correlations',
                'noise_characteristics': 'Ring artifacts, reconstruction artifacts',
                'limitations': 'Lower resolution, long acquisition time, complex reconstruction',
                'advantages': 'True 3D imaging, non-destructive, bulk material analysis'
            },
            'neutron_diffraction': {
                'spatial_pattern': 'Bulk averaging, powder diffraction',
                'signal_characteristics': 'Isotope sensitive, magnetic structure, hydrogen detection',
                'graph_features': 'Minimal spatial connectivity, bulk property correlations',
                'noise_characteristics': 'Low signal, requires large samples',
                'limitations': 'Low spatial resolution, requires neutron source',
                'advantages': 'Penetrates thick samples, sensitive to light elements, magnetic information'
            }
        }
        return reasoning

    @staticmethod
    def get_technique_selection_guidance():
        """Provide guidance on selecting appropriate techniques"""
        guidance = {
            'for_phase_boundaries': ['afm', 'tem_microscopy', 'sem_microscopy'],
            'for_crystal_structure': ['xrd_mapping', 'tem_microscopy', 'neutron_diffraction'],
            'for_3d_analysis': ['xray_tomography'],
            'for_surface_analysis': ['afm', 'sem_microscopy'],
            'for_bulk_average': ['xrd_mapping', 'neutron_diffraction'],
            'for_fast_acquisition': ['sem_microscopy'],
            'for_high_resolution': ['tem_microscopy', 'afm'],
            'for_in_situ_studies': ['sem_microscopy', 'afm']
        }
        return guidance

# =====================================================
# 5. LIGHTWEIGHT PINN FOR DATA ASSIMILATION
# =====================================================
class LiFePO4AssimilationPINN(nn.Module):
    """Lightweight PINN for physics-aware interpolation"""
    def __init__(self, Lx: float, Ly: float, hidden_dims: List[int] = [64, 64, 64]):
        super().__init__()
        self.Lx = Lx
        self.Ly = Ly
        # Normalize coordinates to [0, 1]
        self.x_scale = 1.0 / Lx if Lx > 0 else 1.0
        self.y_scale = 1.0 / Ly if Ly > 0 else 1.0
        # Neural network architecture
        layers = []
        input_dim = 2  # (x, y)
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.Tanh())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())  # Output concentration ∈ [0, 1]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass: predict concentration at (x, y)"""
        # Normalize coordinates
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
        """Compute chemical potential μ = f'(c) - κ∇²c"""
        x.requires_grad_(True)
        y.requires_grad_(True)
        c = self.forward(x, y)
        # Compute gradient ∇c
        grad_c = torch.autograd.grad(
            c, [x, y],
            grad_outputs=torch.ones_like(c),
            create_graph=True,
            retain_graph=True
        )
        grad_c_x, grad_c_y = grad_c
        # Compute Laplacian ∇²c = ∂²c/∂x² + ∂²c/∂y²
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
        # Chemical potential from double-well free energy
        mu_chem = (
            2.0 * A * c +
            3.0 * B * c.pow(2) +
            4.0 * C * c.pow(3) -
            kappa * laplacian_c
        )
        return mu_chem, c

# =====================================================
# 6. PINN TRAINER FOR SINGLE TIME SLICE
# =====================================================
class PINNAssimilationTrainer:
    """Trainer for lightweight PINN at single observation time"""
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
        """
        Train PINN to assimilate sparse observations
        """
        # Prepare data
        x_obs = torch.tensor(obs_data['x_coords'], dtype=torch.float32).to(self.device)
        y_obs = torch.tensor(obs_data['y_coords'], dtype=torch.float32).to(self.device)
        c_obs = torch.tensor(obs_data['c_obs'], dtype=torch.float32).to(self.device)
        # Create data loader
        dataset = TensorDataset(x_obs, y_obs, c_obs)
        dataloader = DataLoader(dataset, batch_size=min(batch_size, len(x_obs)), shuffle=True)
        # Optimizer
        optimizer = optim.Adam(self.pinn.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5, verbose=False)
        # Training loop
        start_time = time.time()
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            epoch_data_loss = 0.0
            epoch_physics_loss = 0.0
            for x_batch, y_batch, c_batch in dataloader:
                optimizer.zero_grad()
                # Data loss: match observations
                c_pred = self.pinn(x_batch, y_batch)
                loss_data = torch.mean((c_pred - c_batch).pow(2))
                # Physics loss: interface equilibrium
                loss_physics = self._compute_physics_loss(
                    x_batch, y_batch, phi_field, sim_params
                )
                # L2 regularization
                l2_reg = torch.tensor(0.0).to(self.device)
                for param in self.pinn.parameters():
                    l2_reg += torch.norm(param)
                # Total loss
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
            # Average losses
            epoch_loss /= len(x_obs)
            epoch_data_loss /= len(x_obs)
            epoch_physics_loss /= len(x_obs)
            self.loss_history.append({
                'epoch': epoch,
                'total_loss': epoch_loss,
                'data_loss': epoch_data_loss,
                'physics_loss': epoch_physics_loss
            })
            # Learning rate scheduling
            scheduler.step(epoch_loss)
            # Call progress callback if provided
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
        """Compute physics loss enforcing equilibrium at interfaces"""
        # Simple physics loss: minimize variation in chemical potential
        mu_chem, _ = self.pinn.compute_chemical_potential(
            x_batch, y_batch,
            sim_params['A'], sim_params['B'], sim_params['C'],
            sim_params['kappa']
        )
        # Interpolate electric potential at batch points
        nx, ny = phi_field.shape
        x_idx = torch.clamp((x_batch / sim_params['L0'] / sim_params['dx']).long(), 0, nx-1)
        y_idx = torch.clamp((y_batch / sim_params['L0'] / sim_params['dx']).long(), 0, ny-1)
        phi_batch = torch.tensor(phi_field[x_idx.cpu(), y_idx.cpu()],
                                 dtype=torch.float32).to(self.device)
        # Total electrochemical potential
        mu_total = mu_chem + sim_params['z'] * sim_params['F'] * phi_batch
        # Loss: minimize variance of total potential (equilibrium condition)
        loss_physics = torch.var(mu_total)
        return loss_physics

    def reconstruct_full_field(
        self,
        nx: int,
        ny: int,
        dx: float,
        L0: float
    ) -> np.ndarray:
        """Reconstruct full concentration field from trained PINN"""
        self.pinn.eval()
        Lx = nx * dx * L0
        Ly = ny * dx * L0
        # Create evaluation grid
        x_grid = np.linspace(0, Lx, nx)
        y_grid = np.linspace(0, Ly, ny)
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        with torch.no_grad():
            X_tensor = torch.tensor(X.flatten(), dtype=torch.float32).to(self.device)
            Y_tensor = torch.tensor(Y.flatten(), dtype=torch.float32).to(self.device)
            c_pred = self.pinn(X_tensor, Y_tensor)
            c_field = c_pred.cpu().numpy().reshape(nx, ny)
        return c_field

# =====================================================
# 7. ENHANCED HYBRID FDM-PINN ASSIMILATION CONTROLLER
# =====================================================
class EnhancedHybridFDMPINNAssimilation:
    """Enhanced controller with nanoscale characterization features"""
    def __init__(self):
        self.sim = None
        self.obs_gen = NanoscaleCharacterizationSimulator()
        self.graph_gen = SyntheticGraphGenerator()
        self.technique_reasoning = TechniqueReasoning()
        self.assimilation_history = []
        self.pinn_reconstructions = []
        self.correction_magnitudes = []
        self.current_cycle = 0
        self.graphs_history = []

    def initialize_simulation(
        self,
        nx: int = 128,
        ny: int = 128,
        dx: float = 1.0,
        dt: float = 0.01,
        c_rate: float = 1.0
    ):
        """Initialize or reinitialize the FDM simulation"""
        self.sim = ElectrochemicalPhaseFieldSimulation(
            nx=nx, ny=ny, dx=dx, dt=dt, c_rate=c_rate
        )

    def run_assimilation_cycle(
        self,
        t_obs: float,
        observation_type: str = 'xrd_mapping',
        coverage: float = None,
        noise_std: float = None,
        pinn_hidden_dims: List[int] = [32, 32, 32],
        n_epochs: int = 300,
        physics_weight: float = 0.2,
        damping_factor: float = 0.7,
        graph_type: str = 'spatial_knn',
        include_artifacts: bool = True,
        progress_callback = None
    ) -> Dict:
        """
        Enhanced assimilation cycle with graph-based features
        """
        if self.sim is None:
            raise ValueError("Simulation not initialized.")
        st.info(f"🔁 Starting enhanced assimilation cycle at t = {t_obs:.2e} s")
        # Step 1: Run FDM to observation time
        with st.spinner("Running FDM simulation..."):
            self.sim.run_until(t_obs)
            c_fdm_before = self.sim.c.copy()
            phi_fdm = self.sim.phi.copy()
        # Step 2: Generate enhanced synthetic observations
        with st.spinner(f"Generating {observation_type} observations..."):
            obs_data = self.obs_gen.generate_sparse_observations(
                c_fdm_before,
                self.sim.dx,
                self.sim.scales.L0,
                observation_type=observation_type,
                measurement_time=t_obs,
                custom_coverage=coverage,
                custom_noise=noise_std,
                include_artifacts=include_artifacts
            )
        # Step 3: Generate synthetic graph
        with st.spinner("Building measurement graph..."):
            graph = self.graph_gen.build_spatial_graph(obs_data, graph_type)
            self.graphs_history.append(graph)
        # Step 4: Train PINN with graph-enhanced features
        with st.spinner("Training enhanced PINN..."):
            Lx = self.sim.nx * self.sim.dx * self.sim.scales.L0
            Ly = self.sim.ny * self.sim.dx * self.sim.scales.L0
            # Use graph information to enhance PINN
            pinn = self.create_graph_enhanced_pinn(Lx, Ly, pinn_hidden_dims, graph)
            trainer = PINNAssimilationTrainer(pinn)
            sim_params = {
                'A': self.sim.A,
                'B': self.sim.B,
                'C': self.sim.C,
                'kappa': self.sim.kappa_dim,
                'z': self.sim.scales.z,
                'F': self.sim.scales.F,
                'dx': self.sim.dx,
                'L0': self.sim.scales.L0,
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
        # Step 5: Reconstruct and correct
        with st.spinner("Reconstructing field..."):
            c_pinn = trainer.reconstruct_full_field(
                self.sim.nx, self.sim.ny,
                self.sim.dx, self.sim.scales.L0
            )
            correction = c_pinn - c_fdm_before
            correction_norm = np.linalg.norm(correction) / np.sqrt(correction.size)
            # Update with damping
            self.sim.c = c_fdm_before + damping_factor * correction
        # Enhanced cycle result
        cycle_result = {
            'cycle': self.current_cycle,
            'time_phys': t_obs,
            'c_fdm_before': c_fdm_before,
            'c_pinn': c_pinn,
            'correction': correction,
            'correction_norm': correction_norm,
            'observation_data': obs_data,
            'graph_data': graph,
            'training_stats': training_stats,
            'phi_field': phi_fdm,
            'mean_c_before': np.mean(c_fdm_before),
            'mean_c_after': np.mean(self.sim.c),
            'technique_metadata': obs_data['technique_metadata'],
            'measurement_quality': obs_data.get('measurement_quality_score', 0.8),
            'graph_statistics': {
                'num_nodes': graph['num_nodes'],
                'num_edges': graph['num_edges'],
                'average_degree': graph['average_degree'],
                'graph_type': graph_type
            }
        }
        self.assimilation_history.append(cycle_result)
        self.current_cycle += 1
        return cycle_result

    def create_graph_enhanced_pinn(self, Lx: float, Ly: float,
                                   hidden_dims: List[int], graph: Dict) -> nn.Module:
        """Create PINN with graph-enhanced features"""
        class GraphEnhancedPINN(LiFePO4AssimilationPINN):
            def __init__(self, Lx, Ly, hidden_dims, graph):
                super().__init__(Lx, Ly, hidden_dims)
                self.graph = graph
                # Graph embedding layer
                self.graph_embedding = nn.Sequential(
                    nn.Linear(2 + 1, 16),  # Position + concentration
                    nn.ReLU(),
                    nn.Linear(16, 8),
                    nn.ReLU()
                )

            def forward_with_graph(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                """Forward pass with graph context"""
                c_pred = self.forward(x, y)
                # Add graph-based refinement if nodes are nearby
                if hasattr(self, 'graph'):
                    # Find nearest graph node for each point
                    positions = torch.stack([x, y], dim=-1).cpu().numpy()
                    graph_positions = self.graph['nodes']['positions']
                    if len(graph_positions) > 0:
                        try:
                            from sklearn.neighbors import NearestNeighbors
                            knn = NearestNeighbors(n_neighbors=1)
                            knn.fit(graph_positions)
                            distances, indices = knn.kneighbors(positions)
                            # Get graph node concentrations
                            graph_concentrations = torch.tensor(
                                self.graph['nodes']['concentrations'][indices.flatten()],
                                dtype=torch.float32
                            ).to(x.device)
                            # Blend with graph information
                            blend_weight = torch.exp(-torch.tensor(distances.flatten()/1e-8)).to(x.device)
                            c_refined = (1 - blend_weight) * c_pred + blend_weight * graph_concentrations
                            return c_refined
                        except ImportError:
                            # Fallback without sklearn
                            pass
                return c_pred

        return GraphEnhancedPINN(Lx, Ly, hidden_dims, graph)

    def run_sequential_assimilation(
        self,
        observation_schedule: List[float],
        observation_type: str = 'xrd_mapping',
        coverage: float = None,
        noise_std: float = None,
        pinn_config: Dict = None,
        damping_factor: float = 0.7,
        graph_type: str = 'spatial_knn',
        progress_container = None
    ) -> List[Dict]:
        """Run sequential assimilation at multiple observation times"""
        results = []
        default_pinn_config = {
            'hidden_dims': [32, 32, 32],
            'n_epochs': 300,
            'physics_weight': 0.2
        }
        if pinn_config:
            default_pinn_config.update(pinn_config)
        progress_bar = None
        status_text = None
        if progress_container:
            progress_bar = progress_container.progress(0)
            status_text = progress_container.empty()
        for i, t_obs in enumerate(observation_schedule):
            if progress_bar and status_text:
                status_text.text(f"Cycle {i+1}/{len(observation_schedule)}: t = {t_obs:.0f} s")
                progress_bar.progress((i) / len(observation_schedule))
            def update_progress(epoch, total_epochs, loss):
                if status_text:
                    status_text.text(f"Cycle {i+1}/{len(observation_schedule)}: "
                                     f"Epoch {epoch}/{total_epochs}, Loss: {loss:.2e}")
            result = self.run_assimilation_cycle(
                t_obs,
                observation_type=observation_type,
                coverage=coverage,
                noise_std=noise_std,
                pinn_hidden_dims=default_pinn_config['hidden_dims'],
                n_epochs=default_pinn_config['n_epochs'],
                physics_weight=default_pinn_config['physics_weight'],
                damping_factor=damping_factor,
                graph_type=graph_type,
                progress_callback=update_progress
            )
            results.append(result)
        if progress_bar:
            progress_bar.progress(1.0)
            status_text.text("✅ Assimilation complete!")
        return results

    def compare_techniques(self, c_field: np.ndarray, techniques: List[str] = None):
        """Compare different characterization techniques"""
        if techniques is None:
            techniques = ['xrd_mapping', 'sem_microscopy', 'afm', 'xray_tomography']
        comparison_results = []
        dx = self.sim.dx if self.sim else 1.0
        L0 = self.sim.scales.L0 if self.sim else 1e-8
        for technique in techniques:
            obs_data = self.obs_gen.generate_sparse_observations(
                c_field, dx, L0, technique, include_artifacts=True
            )
            # Build graph for this technique
            graph = self.graph_gen.build_spatial_graph(obs_data, 'spatial_knn')
            result = {
                'technique': technique,
                'description': self.obs_gen.technique_characteristics[technique]['description'],
                'num_observations': len(obs_data['c_obs']),
                'coverage': obs_data['coverage'],
                'noise_level': obs_data['noise_std'],
                'measurement_quality': obs_data.get('measurement_quality_score', 0),
                'graph_nodes': graph['num_nodes'],
                'graph_edges': graph['num_edges'],
                'graph_connectivity': graph['average_degree'],
                'resolution_nm': self.obs_gen.technique_characteristics[technique]['resolution_nm'],
                'measurement_time': self.obs_gen.technique_characteristics[technique]['measurement_time']
            }
            comparison_results.append(result)
        return pd.DataFrame(comparison_results)

    def get_assimilation_statistics(self) -> Dict:
        """Get statistics about assimilation performance"""
        if not self.assimilation_history:
            return {}
        stats = {
            'total_cycles': len(self.assimilation_history),
            'correction_magnitudes': self.correction_magnitudes,
            'mean_correction': np.mean(self.correction_magnitudes) if self.correction_magnitudes else 0,
            'accuracy_improvements': [],
            'training_times': []
        }
        for i, cycle in enumerate(self.assimilation_history):
            obs_data = cycle['observation_data']
            c_fdm_at_obs = cycle['c_fdm_before'][obs_data['x_idx'], obs_data['y_idx']]
            c_pinn_at_obs = cycle['c_pinn'][obs_data['x_idx'], obs_data['y_idx']]
            fdm_rmse = np.sqrt(np.mean((c_fdm_at_obs - obs_data['c_obs'])**2))
            pinn_rmse = np.sqrt(np.mean((c_pinn_at_obs - obs_data['c_obs'])**2))
            stats['accuracy_improvements'].append({
                'cycle': i,
                'fdm_rmse': fdm_rmse,
                'pinn_rmse': pinn_rmse,
                'improvement_ratio': fdm_rmse / pinn_rmse if pinn_rmse > 0 else 0
            })
            stats['training_times'].append(cycle['training_stats']['training_time'])
        return stats

# =====================================================
# 8. ENHANCED VISUALIZATION FUNCTIONS
# =====================================================
def plot_concentration_field(field: np.ndarray, title: str = "Concentration Field", colorbar_label: str = "x in LiₓFePO₄"):
    """Plot concentration field using Plotly"""
    fig = go.Figure(data=go.Heatmap(
        z=field.T,
        colorscale='RdYlBu',
        zmin=0,
        zmax=1,
        colorbar=dict(title=colorbar_label)
    ))
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis=dict(title="x position"),
        yaxis=dict(title="y position"),
        width=500,
        height=450,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def plot_assimilation_cycle(cycle_result: Dict):
    """Create comprehensive visualization of assimilation cycle"""
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
    # Plot 1: FDM field before assimilation
    fig.add_trace(
        go.Heatmap(
            z=cycle_result['c_fdm_before'].T,
            colorscale='RdYlBu',
            zmin=0,
            zmax=1,
            colorbar=dict(x=0.3, y=0.85, len=0.3),
            showscale=True
        ),
        row=1, col=1
    )
    # Plot 2: PINN reconstruction
    fig.add_trace(
        go.Heatmap(
            z=cycle_result['c_pinn'].T,
            colorscale='RdYlBu',
            zmin=0,
            zmax=1,
            colorbar=dict(x=0.63, y=0.85, len=0.3),
            showscale=True
        ),
        row=1, col=2
    )
    # Plot 3: Correction field
    correction = cycle_result['correction']
    vmax = max(abs(correction.min()), abs(correction.max()))
    fig.add_trace(
        go.Heatmap(
            z=correction.T,
            colorscale='RdBu_r',
            zmin=-vmax,
            zmax=vmax,
            colorbar=dict(x=0.96, y=0.85, len=0.3),
            showscale=True
        ),
        row=1, col=3
    )
    # Plot 4: Observations overlay
    obs_data = cycle_result['observation_data']
    fig.add_trace(
        go.Scatter(
            x=obs_data['x_idx'],
            y=obs_data['y_idx'],
            mode='markers',
            marker=dict(
                size=5,
                color=obs_data['c_obs'],
                colorscale='RdYlBu',
                cmin=0,
                cmax=1,
                line=dict(width=0.5, color='black')
            ),
            showlegend=False
        ),
        row=2, col=1
    )
    # Add background field
    fig.add_trace(
        go.Heatmap(
            z=cycle_result['c_fdm_before'].T,
            colorscale='gray',
            zmin=0,
            zmax=1,
            opacity=0.3,
            showscale=False
        ),
        row=2, col=1
    )
    # Plot 5: Training loss history
    loss_history = cycle_result['training_stats']['loss_history']
    epochs = [l['epoch'] for l in loss_history]
    total_loss = [l['total_loss'] for l in loss_history]
    data_loss = [l['data_loss'] for l in loss_history]
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=total_loss,
            mode='lines',
            name='Total Loss',
            line=dict(color='blue', width=2)
        ),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=data_loss,
            mode='lines',
            name='Data Loss',
            line=dict(color='red', width=2, dash='dash')
        ),
        row=2, col=2
    )
    fig.update_xaxes(title_text="Epoch", row=2, col=2)
    fig.update_yaxes(title_text="Loss", type="log", row=2, col=2)
    # Plot 6: Electric potential
    fig.add_trace(
        go.Heatmap(
            z=cycle_result['phi_field'].T,
            colorscale='RdBu_r',
            colorbar=dict(x=0.96, y=0.35, len=0.3),
            showscale=True
        ),
        row=2, col=3
    )
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(x=0.65, y=0.35),
        title_text=f"Assimilation Cycle {cycle_result['cycle']} at t = {cycle_result['time_phys']:.2e} s",
        title_x=0.5
    )
    return fig

def plot_assimilation_summary(assimilation_history: List[Dict]):
    """Create summary plots of assimilation performance"""
    if not assimilation_history:
        return None
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Correction magnitude over time',
            'Lithium content evolution',
            'Training time per cycle',
            'Observation statistics'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    times = [cycle['time_phys'] for cycle in assimilation_history]
    correction_norms = [cycle['correction_norm'] for cycle in assimilation_history]
    mean_c_before = [cycle['mean_c_before'] for cycle in assimilation_history]
    mean_c_after = [cycle['mean_c_after'] for cycle in assimilation_history]
    # Plot 1: Correction magnitude
    fig.add_trace(
        go.Scatter(
            x=times,
            y=correction_norms,
            mode='lines+markers',
            line=dict(color='blue', width=2),
            marker=dict(size=8),
            name='Correction Norm'
        ),
        row=1, col=1
    )
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Correction Norm", type="log", row=1, col=1)
    # Plot 2: Mean concentration evolution
    fig.add_trace(
        go.Scatter(
            x=times,
            y=mean_c_before,
            mode='lines+markers',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=8),
            name='FDM before'
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=mean_c_after,
            mode='lines+markers',
            line=dict(color='green', width=2),
            marker=dict(size=8),
            name='After assimilation'
        ),
        row=1, col=2
    )
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Mean x in LiₓFePO₄", row=1, col=2)
    # Plot 3: Training time
    training_times = [cycle['training_stats']['training_time']
                      for cycle in assimilation_history]
    fig.add_trace(
        go.Bar(
            x=list(range(len(training_times))),
            y=training_times,
            marker_color='purple',
            name='Training Time'
        ),
        row=2, col=1
    )
    fig.update_xaxes(title_text="Assimilation Cycle", row=2, col=1)
    fig.update_yaxes(title_text="Training Time (s)", row=2, col=1)
    # Plot 4: Observation statistics
    num_obs = [len(cycle['observation_data']['c_obs'])
               for cycle in assimilation_history]
    coverage = [cycle['observation_data']['coverage'] * 100
                for cycle in assimilation_history]
    fig.add_trace(
        go.Bar(
            x=list(range(len(num_obs))),
            y=num_obs,
            name='Number of Observations',
            marker_color='orange',
            yaxis='y'
        ),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(coverage))),
            y=coverage,
            mode='lines+markers',
            name='Coverage (%)',
            line=dict(color='red', width=2),
            marker=dict(size=8),
            yaxis='y2'
        ),
        row=2, col=2
    )
    fig.update_xaxes(title_text="Assimilation Cycle", row=2, col=2)
    fig.update_yaxes(title_text="Number of Observations", row=2, col=2)
    fig.update_yaxes(title_text="Coverage (%)", row=2, col=2, secondary_y=True)
    fig.update_layout(
        height=700,
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        title_text="Sequential Assimilation Performance Summary",
        title_x=0.5
    )
    return fig

def plot_technique_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """Plot comparison of different characterization techniques"""
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Coverage vs Resolution',
            'Measurement Quality',
            'Observation Count',
            'Noise Levels',
            'Graph Connectivity',
            'Measurement Time'
        ),
        specs=[[{'type': 'scatter'}, {'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]]
    )
    techniques = comparison_df['technique'].tolist()
    # Plot 1: Coverage vs Resolution
    fig.add_trace(
        go.Scatter(
            x=comparison_df['resolution_nm'],
            y=comparison_df['coverage'] * 100,
            mode='markers+text',
            marker=dict(size=comparison_df['measurement_quality'] * 20 + 10,
                        color=comparison_df['measurement_quality'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Quality")),
            text=comparison_df['description'],
            textposition="top center",
            name='Techniques'
        ),
        row=1, col=1
    )
    fig.update_xaxes(title_text="Resolution (nm)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Coverage (%)", row=1, col=1)
    # Plot 2: Measurement Quality
    fig.add_trace(
        go.Bar(
            x=techniques,
            y=comparison_df['measurement_quality'],
            marker_color='lightblue',
            name='Quality Score'
        ),
        row=1, col=2
    )
    fig.update_yaxes(title_text="Quality Score", range=[0, 1], row=1, col=2)
    # Plot 3: Observation Count
    fig.add_trace(
        go.Bar(
            x=techniques,
            y=comparison_df['num_observations'],
            marker_color='salmon',
            name='Observations'
        ),
        row=1, col=3
    )
    fig.update_yaxes(title_text="Number of Observations", row=1, col=3)
    # Plot 4: Noise Levels
    fig.add_trace(
        go.Bar(
            x=techniques,
            y=comparison_df['noise_level'] * 100,
            marker_color='lightgreen',
            name='Noise (%)'
        ),
        row=2, col=1
    )
    fig.update_yaxes(title_text="Noise Level (%)", row=2, col=1)
    # Plot 5: Graph Connectivity
    fig.add_trace(
        go.Bar(
            x=techniques,
            y=comparison_df['graph_connectivity'],
            marker_color='gold',
            name='Avg Degree'
        ),
        row=2, col=2
    )
    fig.update_yaxes(title_text="Average Graph Degree", row=2, col=2)
    # Plot 6: Measurement Time
    fig.add_trace(
        go.Bar(
            x=techniques,
            y=comparison_df['measurement_time'],
            marker_color='violet',
            name='Time (s)'
        ),
        row=2, col=3
    )
    fig.update_yaxes(title_text="Measurement Time (s)", type="log", row=2, col=3)
    fig.update_layout(
        height=700,
        showlegend=False,
        title_text="Characterization Technique Comparison",
        title_x=0.5
    )
    return fig

def plot_technique_specific_artifacts(obs_data: Dict) -> go.Figure:
    """Visualize technique-specific artifacts and features"""
    technique = obs_data['observation_type']
    metadata = obs_data['technique_metadata']
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'{metadata["description"]} Observations',
            'Spatial Sampling Pattern',
            'Signal Characteristics',
            'Technique Metadata'
        ),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'bar'}, {'type': 'table'}]]
    )
    # Plot 1: Observations
    fig.add_trace(
        go.Scatter(
            x=obs_data['x_coords'],
            y=obs_data['y_coords'],
            mode='markers',
            marker=dict(
                size=8,
                color=obs_data['c_obs'],
                colorscale='RdYlBu',
                showscale=True,
                colorbar=dict(title="Signal")
            ),
            text=[f"c={val:.3f}" for val in obs_data['c_obs']],
            hoverinfo='text',
            name='Observations'
        ),
        row=1, col=1
    )
    # Plot 2: Sampling pattern
    fig.add_trace(
        go.Scatter(
            x=obs_data['x_idx'],
            y=obs_data['y_idx'],
            mode='markers',
            marker=dict(size=5, color='blue'),
            name=f'Pattern: {metadata["spatial_sampling_pattern"]}'
        ),
        row=1, col=2
    )
    # Plot 3: Signal characteristics
    fig.add_trace(
        go.Histogram(
            x=obs_data['c_obs'],
            nbinsx=20,
            marker_color='green',
            opacity=0.7,
            name='Signal Distribution'
        ),
        row=2, col=1
    )
    # Plot 4: Metadata table
    table_data = []
    for key, value in metadata.items():
        if isinstance(value, list):
            value = ', '.join(str(v) for v in value[:3]) + ('...' if len(value) > 3 else '')
        table_data.append([key, str(value)])
    fig.add_trace(
        go.Table(
            header=dict(values=['Parameter', 'Value'],
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[[row[0] for row in table_data],
                             [row[1] for row in table_data]],
                        fill_color='lavender',
                        align='left'),
            columnwidth=[0.3, 0.7]
        ),
        row=2, col=2
    )
    fig.update_layout(
        height=700,
        title_text=f"Technique Analysis: {technique}",
        title_x=0.5
    )
    fig.update_xaxes(title_text="X Position (m)", row=1, col=1)
    fig.update_yaxes(title_text="Y Position (m)", row=1, col=1)
    fig.update_xaxes(title_text="X Index", row=1, col=2)
    fig.update_yaxes(title_text="Y Index", row=1, col=2)
    fig.update_xaxes(title_text="Signal Value", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    return fig

# =====================================================
# 9. ENHANCED STREAMLIT APPLICATION
# =====================================================
def main():
    """Enhanced Streamlit app with nanoscale characterization features"""
    # Title and header
    st.markdown('<h1 class="main-header">🔬 LiFePO₄ Hybrid FDM-PINN Assimilation with Nanoscale Characterization</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time correction of phase field simulations with realistic experimental observations from XRD, SEM, TEM, AFM, and Tomography</p>', unsafe_allow_html=True)
    # Initialize session state
    if 'enhanced_system' not in st.session_state:
        st.session_state.enhanced_system = EnhancedHybridFDMPINNAssimilation()
    if 'sim_initialized' not in st.session_state:
        st.session_state.sim_initialized = False
    if 'assimilation_results' not in st.session_state:
        st.session_state.assimilation_results = []
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = None
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/microscope.png", width=80)
        st.markdown("### 🎛️ Control Panel")
        # Simulation setup
        with st.expander("⚙️ Simulation Setup", expanded=True):
            grid_size = st.selectbox(
                "Grid Resolution",
                ["64×64 (Fast)", "128×128 (Balanced)", "256×256 (Detailed)"],
                index=1
            )
            if grid_size == "64×64 (Fast)":
                nx, ny = 64, 64
            elif grid_size == "128×128 (Balanced)":
                nx, ny = 128, 128
            else:
                nx, ny = 256, 256
            c_rate = st.slider("C-Rate", 0.1, 10.0, 1.0, 0.1)
            dt = st.slider("Time Step (Δt)", 0.001, 0.1, 0.01, 0.001)
            # Kinetics selector
            kinetics_choice = st.selectbox(
                "Electrochemical Kinetics",
                [
                    "Poisson–Nernst–Planck (PNP) — Bulk migration",
                    "Butler–Volmer (BV) — Surface reaction",
                    "Marcus–Hush–Chidsey (MHC) — Quantum kinetics"
                ],
                index=0,
                help=(
                    "• **PNP**: Full bulk ion transport with electric field coupling (default).\n"
                    "• **BV**: Classical electrode kinetics; applies flux at left boundary.\n"
                    "• **MHC**: Non-adiabatic electron transfer; more accurate at high overpotentials."
                )
            )
            kinetics_type = 0
            if "BV" in kinetics_choice:
                kinetics_type = 1
            elif "MHC" in kinetics_choice:
                kinetics_type = 2
            init_type = st.selectbox(
                "Initialization",
                ["Random", "Lithiation (Charge)", "Delithiation (Discharge)"],
                index=0
            )
            if st.button("🔄 Initialize Simulation", use_container_width=True):
                with st.spinner("Initializing simulation..."):
                    st.session_state.enhanced_system.initialize_simulation(
                        nx=nx, ny=ny, dx=1.0, dt=dt, c_rate=c_rate
                    )
                    sim = st.session_state.enhanced_system.sim
                    # Set kinetics BEFORE initialization
                    sim.set_parameters(kinetics_type=kinetics_type)
                    if init_type == "Random":
                        sim.initialize_random()
                    elif init_type == "Lithiation (Charge)":
                        sim.initialize_lithiation()
                    else:
                        sim.initialize_delithiation()
                    st.session_state.sim_initialized = True
                    st.session_state.assimilation_results = []
                    st.session_state.comparison_results = None
                    st.success("✅ Simulation initialized!")
                    st.rerun()
        # Enhanced characterization configuration
        with st.expander("🔬 Nanoscale Characterization", expanded=True):
            technique = st.selectbox(
                "Measurement Technique",
                list(st.session_state.enhanced_system.obs_gen.technique_characteristics.keys()),
                format_func=lambda x: st.session_state.enhanced_system.obs_gen.technique_characteristics[x]['description'],
                index=0
            )
            # Show technique details
            tech_config = st.session_state.enhanced_system.obs_gen.technique_characteristics[technique]
            st.caption(f"**Resolution:** {tech_config['resolution_nm']} nm")
            st.caption(f"**Coverage:** {tech_config['coverage']*100:.1f}%")
            st.caption(f"**Measurement Time:** {tech_config['measurement_time']}s")
            st.caption(f"**Key Features:** {', '.join(tech_config['characteristic_features'][:2])}")
            coverage = st.slider(
                "Observation Coverage",
                0.01, 0.3, tech_config['coverage'], 0.01,
                help="Fraction of grid points with measurements"
            )
            noise_std = st.slider(
                "Measurement Noise",
                0.0, 0.2, tech_config['noise_std'], 0.01,
                help="Standard deviation of Gaussian noise"
            )
            # Graph type selection
            graph_type = st.selectbox(
                "Graph Construction Method",
                list(st.session_state.enhanced_system.graph_gen.graph_types.keys()),
                format_func=lambda x: st.session_state.enhanced_system.graph_gen.graph_types[x]['description'],
                index=0
            )
            include_artifacts = st.checkbox("Include Technique-Specific Artifacts", value=True)
        # PINN settings
        with st.expander("🧠 PINN Configuration", expanded=True):
            pinn_hidden_dims = st.text_input(
                "Hidden Layer Sizes",
                "32,32,32",
                help="Comma-separated list of hidden layer sizes"
            )
            try:
                hidden_dims = [int(x.strip()) for x in pinn_hidden_dims.split(",")]
            except:
                hidden_dims = [32, 32, 32]
            n_epochs = st.slider("Training Epochs", 100, 1000, 300, 50)
            physics_weight = st.slider("Physics Weight", 0.0, 1.0, 0.2, 0.05)
            damping_factor = st.slider("Damping Factor", 0.1, 1.0, 0.7, 0.1)
        # Assimilation schedule
        with st.expander("⏱️ Assimilation Schedule", expanded=True):
            schedule_type = st.radio(
                "Schedule Type",
                ["Manual Entry", "Automatic (Linear)", "Automatic (Logarithmic)"],
                index=0
            )
            if schedule_type == "Manual Entry":
                schedule_input = st.text_area(
                    "Observation Times (s)",
                    "1000, 5000, 10000, 20000, 50000",
                    help="Comma-separated list of observation times in seconds"
                )
                try:
                    observation_schedule = [float(x.strip()) for x in schedule_input.split(",")]
                except:
                    observation_schedule = [1000.0, 5000.0, 10000.0, 20000.0, 50000.0]
            elif schedule_type == "Automatic (Linear)":
                num_cycles = st.slider("Number of Cycles", 3, 10, 5)
                total_time = st.number_input("Total Time (s)", 1000.0, 1e6, 50000.0)
                observation_schedule = np.linspace(1000, total_time, num_cycles).tolist()
            else:  # Logarithmic
                num_cycles = st.slider("Number of Cycles", 3, 10, 5)
                total_time = st.number_input("Total Time (s)", 1000.0, 1e6, 50000.0)
                observation_schedule = np.logspace(np.log10(1000), np.log10(total_time), num_cycles).tolist()
            st.markdown("**Scheduled Observations:**")
            for i, t in enumerate(observation_schedule):
                st.caption(f"Cycle {i+1}: t = {t:.0f} s")
        # Multi-technique comparison
        with st.expander("📊 Technique Comparison", expanded=False):
            compare_techniques = st.multiselect(
                "Select Techniques to Compare",
                list(st.session_state.enhanced_system.obs_gen.technique_characteristics.keys()),
                default=['xrd_mapping', 'sem_microscopy', 'afm'],
                format_func=lambda x: st.session_state.enhanced_system.obs_gen.technique_characteristics[x]['description']
            )
            if st.button("Run Technique Comparison", use_container_width=True):
                with st.spinner("Comparing techniques..."):
                    if st.session_state.sim_initialized:
                        sim = st.session_state.enhanced_system.sim
                        comparison_df = st.session_state.enhanced_system.compare_techniques(
                            sim.c, compare_techniques
                        )
                        st.session_state.comparison_results = comparison_df
                        st.rerun()
        # Run assimilation
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            run_single = st.button("🔁 Run Single Cycle", use_container_width=True)
        with col2:
            run_sequential = st.button("🔄 Run Sequential", use_container_width=True, type="primary")
        # Clear results
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.assimilation_results = []
            st.session_state.enhanced_system.current_cycle = 0
            st.session_state.enhanced_system.assimilation_history = []
            st.session_state.enhanced_system.graphs_history = []
            st.session_state.comparison_results = None
            st.success("Results cleared!")
            st.rerun()
    # Display current status
    st.divider()
    st.markdown("### 📊 Current Status")
    if st.session_state.sim_initialized:
        sim = st.session_state.enhanced_system.sim
        stats = sim.get_statistics()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Simulation Time", f"{stats['time_phys']:.2e} s")
            st.metric("Lithium Content", f"{stats['mean_c']:.3f}")
        with col2:
            st.metric("C-Rate", f"{stats['c_rate']}C")
            st.metric("Kinetics", ["PNP", "BV", "MHC"][stats['kinetics_type']])
        with col3:
            st.metric("Phase FePO₄", f"{stats['phase_FePO4']:.1%}")
            st.metric("Phase LiFePO₄", f"{stats['phase_LiFePO4']:.1%}")
        with col4:
            st.metric("Assimilation Cycles", len(st.session_state.assimilation_results))
            st.metric("Domain Size", f"{stats['domain_size_nm']:.0f} nm")
    else:
        st.warning("Simulation not initialized")
    # Main content area
    if not st.session_state.sim_initialized:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class="card">
            <h3>🚀 Welcome to Enhanced FDM-PINN Assimilation with Nanoscale Characterization</h3>
            <p>This system combines:</p>
            <ul>
            <li><strong>FDM Simulation:</strong> Physics-based phase field model</li>
            <li><strong>PINN Correction:</strong> Physics-informed neural networks</li>
            <li><strong>Realistic Experimental Data:</strong> XRD, SEM, TEM, AFM, Tomography</li>
            <li><strong>Graph-Based Analysis:</strong> Synthetic graphs representing measurement relationships</li>
            </ul>
            <p><strong>To get started:</strong></p>
            <ol>
            <li>Configure simulation settings in the sidebar</li>
            <li>Click "Initialize Simulation"</li>
            <li>Select characterization technique and parameters</li>
            <li>Run assimilation cycles</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
        # Quick start buttons
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Quick Start: XRD Characterization", use_container_width=True):
                st.session_state.enhanced_system.initialize_simulation()
                sim = st.session_state.enhanced_system.sim
                sim.set_parameters(kinetics_type=0)  # Default to PNP
                sim.initialize_lithiation()
                st.session_state.sim_initialized = True
                st.rerun()
        with col_b:
            if st.button("Demo Mode: Multi-Technique", use_container_width=True):
                # Run demo
                st.session_state.enhanced_system.initialize_simulation(nx=64, ny=64)
                sim = st.session_state.enhanced_system.sim
                sim.set_parameters(kinetics_type=0)
                sim.initialize_lithiation()
                st.session_state.sim_initialized = True
                # Run quick assimilation
                progress_container = st.empty()
                results = st.session_state.enhanced_system.run_sequential_assimilation(
                    [1000.0, 5000.0, 10000.0],
                    observation_type='xrd_mapping',
                    coverage=0.1,
                    progress_container=progress_container
                )
                st.session_state.assimilation_results = results
                st.rerun()
        return
    # Main simulation interface
    sim = st.session_state.enhanced_system.sim
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Simulation View",
        "🔬 Assimilation",
        "📊 Results",
        "🔍 Technique Analysis",
        "🧠 Graph Features"
    ])
    with tab1:
        # Simulation controls and visualization
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Current Simulation State")
            # Run simulation controls
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                steps = st.number_input("Steps to run", 1, 1000, 100)
                if st.button("▶️ Run Steps", use_container_width=True):
                    with st.spinner(f"Running {steps} steps..."):
                        sim.run_steps(steps)
                    st.rerun()
            with col_b:
                target_time = st.number_input("Target time (s)",
                                              float(sim.time_phys),
                                              1e6,
                                              float(sim.time_phys * 2))
                if st.button("⏱️ Run to Time", use_container_width=True):
                    with st.spinner(f"Running to t = {target_time:.0f} s..."):
                        sim.run_until(target_time)
                    st.rerun()
            with col_c:
                if st.button("🔄 Reset Simulation", use_container_width=True):
                    sim.initialize_random()
                    st.rerun()
            # Plot current concentration field
            fig = plot_concentration_field(
                sim.c,
                title=f"LiₓFePO₄ Concentration (t = {sim.time_phys:.2e} s, x = {np.mean(sim.c):.3f})"
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Simulation Statistics")
            stats = sim.get_statistics()
            # Key metrics
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Time", f"{stats['time_phys']:.2e} s")
                st.metric("Steps", stats['step'])
                st.metric("Mean x", f"{stats['mean_c']:.3f}")
                st.metric("Std x", f"{stats['std_c']:.3f}")
            with metric_col2:
                st.metric("FePO₄", f"{stats['phase_FePO4']:.1%}")
                st.metric("LiFePO₄", f"{stats['phase_LiFePO4']:.1%}")
                st.metric("Voltage", f"{stats['voltage']:.3f} V")
                st.metric("Interface Width", f"{stats['interface_width_nm']:.1f} nm")
                st.metric("Kinetics", ["PNP", "BV", "MHC"][stats['kinetics_type']])
            # Phase distribution plot
            fig_pie = go.Figure(data=[go.Pie(
                labels=['FePO₄-rich', 'LiFePO₄-rich'],
                values=[stats['phase_FePO4'], stats['phase_LiFePO4']],
                hole=0.3,
                marker_colors=['#FF6B6B', '#4ECDC4']
            )])
            fig_pie.update_layout(
                title="Phase Fractions",
                height=250,
                margin=dict(t=50, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            # Electric field visualization
            if st.checkbox("Show Electric Field", False):
                E_mag = np.sqrt(sim.Ex**2 + sim.Ey**2)
                fig_E = plot_concentration_field(
                    E_mag,
                    title="Electric Field Magnitude",
                    colorbar_label="|E| (V/m)"
                )
                st.plotly_chart(fig_E, use_container_width=True)
    with tab2:
        # Assimilation interface
        st.subheader("Enhanced FDM-PINN Assimilation")
        # Run assimilation based on button clicks
        if run_single:
            if 'observation_schedule' in locals() or 'observation_schedule' in globals():
                t_obs = observation_schedule[min(st.session_state.enhanced_system.current_cycle,
                                                len(observation_schedule)-1)]
            else:
                # Fallback: use last scheduled time or default
                t_obs = sim.time_phys + 1000.0
            progress_container = st.empty()
            with progress_container:
                st.info(f"Running single assimilation cycle at t = {t_obs:.0f} s")
                progress_bar = st.progress(0)
                status_text = st.empty()
                def update_progress(epoch, total_epochs, loss):
                    progress_bar.progress(epoch / total_epochs)
                    status_text.text(f"Epoch {epoch}/{total_epochs}, Loss: {loss:.2e}")
                result = st.session_state.enhanced_system.run_assimilation_cycle(
                    t_obs,
                    observation_type=technique,
                    coverage=coverage,
                    noise_std=noise_std,
                    pinn_hidden_dims=hidden_dims,
                    n_epochs=n_epochs,
                    physics_weight=physics_weight,
                    damping_factor=damping_factor,
                    graph_type=graph_type,
                    include_artifacts=include_artifacts,
                    progress_callback=update_progress
                )
                st.session_state.assimilation_results.append(result)
                progress_container.empty()
                st.success(f"✅ Assimilation cycle {len(st.session_state.assimilation_results)} complete!")
                st.rerun()
        if run_sequential:
            # Ensure observation_schedule is defined
            if 'observation_schedule' not in locals():
                observation_schedule = [1000.0, 5000.0, 10000.0, 20000.0, 50000.0]
            progress_container = st.empty()
            results = st.session_state.enhanced_system.run_sequential_assimilation(
                observation_schedule,
                observation_type=technique,
                coverage=coverage,
                noise_std=noise_std,
                pinn_config={
                    'hidden_dims': hidden_dims,
                    'n_epochs': n_epochs,
                    'physics_weight': physics_weight
                },
                damping_factor=damping_factor,
                graph_type=graph_type,
                progress_container=progress_container
            )
            st.session_state.assimilation_results = results
            st.success(f"✅ Sequential assimilation complete! {len(results)} cycles run.")
            st.rerun()
        # Display assimilation results
        if st.session_state.assimilation_results:
            st.subheader("Assimilation Results")
            # Select cycle to view
            cycle_to_view = st.selectbox(
                "Select Cycle to View",
                range(len(st.session_state.assimilation_results)),
                format_func=lambda x: f"Cycle {x+1} at t={st.session_state.assimilation_results[x]['time_phys']:.0f}s"
            )
            if cycle_to_view < len(st.session_state.assimilation_results):
                result = st.session_state.assimilation_results[cycle_to_view]
                # Display cycle visualization
                fig_cycle = plot_assimilation_cycle(result)
                st.plotly_chart(fig_cycle, use_container_width=True)
                # Cycle statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Correction Norm", f"{result['correction_norm']:.2e}")
                with col2:
                    st.metric("Observations", len(result['observation_data']['c_obs']))
                with col3:
                    st.metric("Coverage", f"{result['observation_data']['coverage']:.1%}")
                with col4:
                    st.metric("Training Time", f"{result['training_stats']['training_time']:.1f}s")
                st.metric("Graph Nodes", result['graph_statistics']['num_nodes'])
                # Show observation points
                with st.expander("📋 Observation Details"):
                    obs_df = pd.DataFrame({
                        'x (m)': result['observation_data']['x_coords'],
                        'y (m)': result['observation_data']['y_coords'],
                        'Measured x': result['observation_data']['c_obs'],
                        'FDM Prediction': result['c_fdm_before'][
                            result['observation_data']['x_idx'],
                            result['observation_data']['y_idx']
                        ],
                        'PINN Prediction': result['c_pinn'][
                            result['observation_data']['x_idx'],
                            result['observation_data']['y_idx']
                        ]
                    })
                    st.dataframe(obs_df.head(20), use_container_width=True)
                    # Download button
                    csv = obs_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Observation Data",
                        data=csv,
                        file_name=f"observations_cycle_{cycle_to_view+1}.csv",
                        mime="text/csv"
                    )
            else:
                st.error("Selected cycle index out of range.")
        else:
            # Show assimilation instructions
            st.info("👈 Configure assimilation settings and run cycles from the sidebar")
        # Technique comparison preview
        st.subheader("Available Characterization Techniques")
        technique_df = st.session_state.enhanced_system.obs_gen.get_technique_comparison()
        st.dataframe(technique_df, use_container_width=True)
    with tab3:
        # Results and analysis
        st.subheader("Assimilation Performance Analysis")
        if st.session_state.assimilation_results:
            # Summary statistics
            stats = st.session_state.enhanced_system.get_assimilation_statistics()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Cycles", stats['total_cycles'])
            with col2:
                st.metric("Avg Correction", f"{stats['mean_correction']:.2e}")
            with col3:
                if stats['accuracy_improvements']:
                    avg_improvement = np.mean([a['improvement_ratio']
                                               for a in stats['accuracy_improvements']])
                    st.metric("Avg RMSE Improvement", f"{avg_improvement:.1f}x")
            with col4:
                if stats['training_times']:
                    st.metric("Total Training Time", f"{np.sum(stats['training_times']):.1f}s")
            # Summary plot
            fig_summary = plot_assimilation_summary(st.session_state.assimilation_results)
            if fig_summary:
                st.plotly_chart(fig_summary, use_container_width=True)
            # Detailed analysis
            with st.expander("📈 Detailed Analysis"):
                if stats['accuracy_improvements']:
                    accuracy_df = pd.DataFrame(stats['accuracy_improvements'])
                    fig_accuracy = go.Figure()
                    fig_accuracy.add_trace(go.Scatter(
                        x=accuracy_df['cycle'],
                        y=accuracy_df['fdm_rmse'],
                        mode='lines+markers',
                        name='FDM RMSE',
                        line=dict(color='red', width=2)
                    ))
                    fig_accuracy.add_trace(go.Scatter(
                        x=accuracy_df['cycle'],
                        y=accuracy_df['pinn_rmse'],
                        mode='lines+markers',
                        name='PINN RMSE',
                        line=dict(color='blue', width=2)
                    ))
                    fig_accuracy.update_layout(
                        title="RMSE Comparison",
                        xaxis_title="Assimilation Cycle",
                        yaxis_title="RMSE",
                        height=400
                    )
                    st.plotly_chart(fig_accuracy, use_container_width=True)
                    # Show data table
                    st.dataframe(accuracy_df, use_container_width=True)
            # Export results
            with st.expander("💾 Export Results"):
                if st.button("Export All Results to CSV"):
                    # Compile all results into a DataFrame
                    all_data = []
                    for i, result in enumerate(st.session_state.assimilation_results):
                        cycle_data = {
                            'cycle': i+1,
                            'time_phys': result['time_phys'],
                            'correction_norm': result['correction_norm'],
                            'mean_c_before': result['mean_c_before'],
                            'mean_c_after': result['mean_c_after'],
                            'num_observations': len(result['observation_data']['c_obs']),
                            'coverage': result['observation_data']['coverage'],
                            'noise_std': result['observation_data']['noise_std'],
                            'training_time': result['training_stats']['training_time'],
                            'final_loss': result['training_stats']['final_loss'],
                            'technique': result['observation_data']['observation_type'],
                            'graph_nodes': result['graph_statistics']['num_nodes'],
                            'graph_edges': result['graph_statistics']['num_edges']
                        }
                        all_data.append(cycle_data)
                    df = pd.DataFrame(all_data)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Summary CSV",
                        data=csv,
                        file_name="assimilation_summary.csv",
                        mime="text/csv"
                    )
        else:
            st.info("Run assimilation cycles to see performance analysis here.")
    with tab4:
        # Technique analysis
        st.subheader("Characterization Technique Analysis")
        if st.session_state.comparison_results is not None:
            comparison_df = st.session_state.comparison_results
            fig_compare = plot_technique_comparison(comparison_df)
            st.plotly_chart(fig_compare, use_container_width=True)
            # Show reasoning
            with st.expander("Technique Reasoning", expanded=False):
                reasoning = TechniqueReasoning.explain_technique_differences()
                selected_tech = st.selectbox(
                    "Select technique for detailed reasoning",
                    list(reasoning.keys())
                )
                if selected_tech in reasoning:
                    tech_reason = reasoning[selected_tech]
                    cols = st.columns(2)
                    with cols[0]:
                        st.markdown("**Spatial Pattern:**")
                        st.info(tech_reason['spatial_pattern'])
                        st.markdown("**Signal Characteristics:**")
                        st.info(tech_reason['signal_characteristics'])
                        st.markdown("**Graph Features:**")
                        st.info(tech_reason['graph_features'])
                    with cols[1]:
                        st.markdown("**Noise Characteristics:**")
                        st.warning(tech_reason['noise_characteristics'])
                        st.markdown("**Limitations:**")
                        st.error(tech_reason['limitations'])
                        st.markdown("**Advantages:**")
                        st.success(tech_reason['advantages'])
        else:
            st.info("Run technique comparison from the sidebar to see analysis here.")
        # Show technique guidance
        with st.expander("Technique Selection Guidance", expanded=True):
            guidance = TechniqueReasoning.get_technique_selection_guidance()
            for purpose, techniques in guidance.items():
                st.markdown(f"**For {purpose.replace('_', ' ').title()}:**")
                tech_descriptions = []
                for tech in techniques:
                    if tech in st.session_state.enhanced_system.obs_gen.technique_characteristics:
                        desc = st.session_state.enhanced_system.obs_gen.technique_characteristics[tech]['description']
                        tech_descriptions.append(f"• {desc}")
                if tech_descriptions:
                    st.markdown("\n".join(tech_descriptions[:3]))
                st.markdown("---")
            # Show available techniques
            st.subheader("Available Techniques")
            technique_df = st.session_state.enhanced_system.obs_gen.get_technique_comparison()
            st.dataframe(technique_df, use_container_width=True)
    with tab5:
        # Graph feature analysis
        st.subheader("Graph Feature Analysis")
        if st.session_state.get('assimilation_results'):
            # Select cycle to analyze graph
            cycle_idx = st.selectbox(
                "Select assimilation cycle",
                range(len(st.session_state.assimilation_results)),
                format_func=lambda x: f"Cycle {x+1}"
            )
            if cycle_idx < len(st.session_state.assimilation_results):
                result = st.session_state.assimilation_results[cycle_idx]
                if 'graph_data' in result:
                    # Visualize graph
                    fig_graph = st.session_state.enhanced_system.graph_gen.visualize_graph(
                        result['graph_data'], result['observation_data']
                    )
                    st.plotly_chart(fig_graph, use_container_width=True)
                    # Graph statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Nodes", result['graph_statistics']['num_nodes'])
                    with col2:
                        st.metric("Edges", result['graph_statistics']['num_edges'])
                    with col3:
                        st.metric("Avg Degree", f"{result['graph_statistics']['average_degree']:.2f}")
                    with col4:
                        st.metric("Graph Type", result['graph_statistics']['graph_type'])
                    # Show technique-specific artifacts
                    st.subheader("Technique-Specific Features")
                    fig_artifacts = plot_technique_specific_artifacts(result['observation_data'])
                    st.plotly_chart(fig_artifacts, use_container_width=True)
                    # Edge weight distribution
                    if len(result['graph_data']['edge_weights']) > 0:
                        fig_weights = go.Figure(data=[go.Histogram(
                            x=result['graph_data']['edge_weights'],
                            nbinsx=20,
                            marker_color='purple'
                        )])
                        fig_weights.update_layout(
                            title="Edge Weight Distribution",
                            xaxis_title="Weight",
                            yaxis_title="Count",
                            height=300
                        )
                        st.plotly_chart(fig_weights, use_container_width=True)
                    # Graph type explanation
                    with st.expander("📝 Graph Type Explanation", expanded=False):
                        graph_types = st.session_state.enhanced_system.graph_gen.graph_types
                        selected_type = result['graph_statistics']['graph_type']
                        if selected_type in graph_types:
                            st.markdown(f"**{graph_types[selected_type]['description']}**")
                            st.markdown(f"**Edge Weighting:** {graph_types[selected_type]['edge_weighting']}")
                            st.markdown(f"**Use Technique Specific:** {graph_types[selected_type]['use_technique_specific']}")
                            if selected_type == 'diffraction_correlation':
                                st.markdown("*Specifically designed for XRD mapping data*")
                            elif selected_type == 'topography_continuity':
                                st.markdown("*Specifically designed for AFM topography data*")
            else:
                st.info("Run assimilation cycles to see graph analysis here.")
        # Show available graph types
        st.subheader("Available Graph Construction Methods")
        graph_types_df = pd.DataFrame([
            {
                'Type': name,
                'Description': config['description'],
                'Edge Weighting': config['edge_weighting'],
                'Technique Specific': config['use_technique_specific']
            }
            for name, config in st.session_state.enhanced_system.graph_gen.graph_types.items()
        ])
        st.dataframe(graph_types_df, use_container_width=True)

# Run the app
if __name__ == "__main__":
    main()
