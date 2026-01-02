# app.py - Main Streamlit Application (FULL VERSION WITH KINETICS SELECTION)
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
# 1. ORIGINAL FDM SIMULATION (Full Implementation)
# =====================================================
@st.cache_resource
class PhysicalScalesWithElectrostaticsAndC_Rate:
    """Physical scales with electrostatics and C-rate for LiFePO₄"""
    # Fundamental constants
    R = 8.314462618 # J/(mol·K)
    F = 96485.33212 # C/mol
    k_B = 1.380649e-23 # J/K
    ε0 = 8.854187817e-12 # F/m - Vacuum permittivity
    def __init__(self, c_rate=1.0):
        # Material properties
        self.T = 298.15 # K - Temperature
        # LiFePO₄ phase compositions
        self.c_alpha = 0.03 # FePO₄ phase
        self.c_beta = 0.97 # LiFePO₄ phase
        # Molar volume
        self.V_m = 3.0e-5 # m³/mol
        # Diffusion coefficient
        self.D_b = 1.0e-14 # m²/s - Fast diffusion along b-axis
        # Electrostatic properties
        self.ε_r = 15.0 # Relative permittivity of LiFePO₄
        self.ε = self.ε_r * self.ε0 # Absolute permittivity (F/m)
        # Charge properties
        self.z = 1.0 # Li⁺ charge number
        self.ρ0 = 1.0e6 # Reference charge density (C/m³)
        # Regular solution parameter for LiFePO₄
        self.Ω = 55e3 # J/mol
        # Kinetics parameters
        self.k0_bv = 1.0e-6 # BV rate constant (m/s)
        self.k0_mhc = 5.0e-7 # MHC rate constant (m/s)
        self.alpha = 0.5 # BV symmetry factor
       
        # Set C-rate parameters
        self.set_c_rate_parameters(c_rate)
        # Set characteristic scales
        self.set_scales()
        # Calculate Debye length
        self.c_ref = 0.5 # Reference concentration
        self.λ_D = self.calculate_debye_length()
    def set_c_rate_parameters(self, c_rate):
        """Set C-rate dependent parameters"""
        self.c_rate = c_rate
        # C-rate scaling factor (1.0 for 1C)
        if c_rate <= 1.0:
            self.c_rate_factor = 1.0
            self.eta_scale = 0.01 # Small overpotential for slow rates
        else:
            self.c_rate_factor = 1.0 + 0.5 * np.log10(c_rate)
            self.eta_scale = 0.01 * c_rate**0.5 # Larger overpotential
       
        # Rate-dependent interface sharpness
        self.kappa_factor = 1.0 / (1.0 + 0.2 * np.log10(max(1.0, c_rate)))
        # Rate-dependent mobility (effective diffusion)
        self.D_factor = 1.0 / (1.0 + 0.1 * c_rate**0.5)
    def set_scales(self):
        """Set characteristic scales"""
        # Length scale: 10 nm domain
        self.L0 = 1.0e-8 # 10 nm
        # Energy density scale from regular solution
        self.E0 = self.Ω / self.V_m # J/m³
        # Time scale from diffusion
        self.t0 = (self.L0**2) / self.D_b # s
        # Mobility scale
        self.M0 = self.D_b / (self.E0 * self.t0) # m⁵/(J·s)
        # Electric potential scale (thermal voltage)
        self.φ0 = self.R * self.T / self.F # ~0.0257 V at 298K
    def calculate_debye_length(self):
        """Calculate Debye screening length"""
        c_ref_moles_per_m3 = self.c_ref * (1/self.V_m) # mol/m³
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
        if kinetics_type == 0: # PNP
            # Add migration for PNP
            flux_mig_x = -(D_eff * z * F * c / (R * T)) * phi_grad_x
            flux_mig_y = -(D_eff * z * F * c / (R * T)) * phi_grad_y
            flux_x = flux_diff_x + flux_mig_x
            flux_y = flux_diff_y + flux_mig_y
        else: # BV or MHC - no migration term
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
        if kinetics_type != 0: # Not PNP
            for j in prange(ny):
                c_surf = c_new[0, j]
                if kinetics_type == 1: # BV
                    flux = butler_volmer_flux(eta, c_surf, alpha, k0, F, R, T)
                elif kinetics_type == 2: # MHC
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
        if kinetics_type == 0: # PNP
            # Add migration for PNP
            flux_mig_x = -(D_eff * z * F * c / (R * T)) * phi_grad_x
            flux_mig_y = -(D_eff * z * F * c / (R * T)) * phi_grad_y
            flux_x = flux_diff_x + flux_mig_x
            flux_y = flux_diff_y + flux_mig_y
        else: # BV or MHC - no migration term
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
        if kinetics_type != 0: # Not PNP
            for j in range(ny):
                c_surf = c_new[0, j]
                if kinetics_type == 1: # BV
                    flux = butler_volmer_flux(eta, c_surf, alpha, k0, F, R, T)
                elif kinetics_type == 2: # MHC
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
        self.c = np.zeros((nx, ny)) # Concentration
        self.phi = np.zeros((nx, ny)) # Electric potential
        self.Ex = np.zeros((nx, ny)) # Electric field x-component
        self.Ey = np.zeros((nx, ny)) # Electric field y-component
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
        self.phi = np.zeros_like(self.c) # Start with zero potential
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
        total_charge = np.sum(self.c - 0.5) # Relative to neutral
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
        if self.kinetics_type == 0: # PNP
            k0 = 0.0 # Dummy
            alpha = 0.0
            eta = 0.0
        else:
            eta = self.eta_left
            alpha = self.scales.alpha
            if self.kinetics_type == 1: # BV
                k0 = self.scales.k0_bv * self.scales.c_rate_factor
            else: # MHC
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
# 2. SYNTHETIC OBSERVATION GENERATOR (ENHANCED)
# =====================================================
@st.cache_resource
class SyntheticObservationGenerator:
    """Generate synthetic experimental observations from FDM simulation with technique-specific aggregation"""
    def __init__(self):
        self.observation_types = {
            'microscopy': {
                'coverage': 0.1, 
                'noise_std': 0.05, 
                'pattern': 'random', 
                'aggregation': 'none'  # Pointwise, high local variability
            },
            'xrd_mapping': {
                'coverage': 0.05, 
                'noise_std': 0.03, 
                'pattern': 'grid', 
                'aggregation': 'local',  # Average over beam spot for phase fraction
                'kernel_size': 5
            },
            'tomography': {
                'coverage': 0.15, 
                'noise_std': 0.04, 
                'pattern': 'lines', 
                'aggregation': 'line'  # Average along projection lines
            },
            'afm': {
                'coverage': 0.08, 
                'noise_std': 0.06, 
                'pattern': 'random', 
                'aggregation': 'none'  # Pointwise surface measurements
            }
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
        custom_noise: float = None,
        custom_aggregation: str = None,
        custom_kernel_size: int = None
    ) -> Dict:
        """
        Generate synthetic sparse observations from concentration field, with aggregation for realism.
        - 'none': Direct point values.
        - 'local': Average over local kernel (e.g., XRD beam spot).
        - 'line': Average per line (e.g., tomography projections).
        """
        if seed is not None:
            np.random.seed(seed)
        nx, ny = c_field.shape
        config = self.observation_types[observation_type].copy()
        # Override with custom values if provided
        if custom_coverage is not None:
            config['coverage'] = custom_coverage
        if custom_noise is not None:
            config['noise_std'] = custom_noise
        if custom_aggregation is not None:
            config['aggregation'] = custom_aggregation
        if custom_kernel_size is not None:
            config['kernel_size'] = custom_kernel_size
        # Generate observation locations based on pattern
        if config['pattern'] == 'random':
            n_obs = max(1, int(config['coverage'] * nx * ny))
            obs_indices = np.random.choice(nx * ny, n_obs, replace=False)
            obs_i = obs_indices // ny
            obs_j = obs_indices % ny
        elif config['pattern'] == 'grid':
            # Regular grid sampling
            step = max(1, int(1 / np.sqrt(config['coverage'])))
            obs_i, obs_j = np.meshgrid(
                np.arange(0, nx, step),
                np.arange(0, ny, step),
                indexing='ij'
            )
            obs_i = obs_i.flatten()
            obs_j = obs_j.flatten()
        elif config['pattern'] == 'lines':
            # Line scans (common in AFM/SEM/tomography)
            n_lines = max(1, int(np.sqrt(config['coverage'] * nx * ny / ny)))
            line_indices = np.linspace(0, nx-1, n_lines).astype(int)
            obs_i = np.repeat(line_indices, ny)
            obs_j = np.tile(np.arange(ny), n_lines)
        # Ensure we have at least some observations
        if len(obs_i) == 0:
            obs_i = np.array([nx//2])
            obs_j = np.array([ny//2])
        # Compute c_obs with aggregation
        c_obs = np.zeros(len(obs_i))
        agg_type = config.get('aggregation', 'none')
        if agg_type == 'none':
            c_obs = c_field[obs_i, obs_j]
        elif agg_type == 'local':
            ks = config.get('kernel_size', 3)
            half = ks // 2
            for k in range(len(obs_i)):
                i, j = obs_i[k], obs_j[k]
                i_min = max(0, i - half)
                i_max = min(nx, i + half + 1)
                j_min = max(0, j - half)
                j_max = min(ny, j + half + 1)
                c_obs[k] = np.mean(c_field[i_min:i_max, j_min:j_max])
        elif agg_type == 'line':
            # Assume horizontal lines (grouped by obs_i)
            unique_i = np.unique(obs_i)
            for ui in unique_i:
                mask = (obs_i == ui)
                # Average along the entire row for projection simulation
                line_mean = np.mean(c_field[ui, :])
                c_obs[mask] = line_mean
        # Add noise to observations
        noise = np.random.normal(0, config['noise_std'], c_obs.shape)
        c_obs_noisy = np.clip(c_obs + noise, 0, 1)
        # Convert to physical coordinates
        x_phys = obs_i * dx * L0
        y_phys = obs_j * dx * L0
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
            'aggregation_type': agg_type,  # Added for tracking
            'full_field': c_field  # For validation only
        }
# =====================================================
# 3. LIGHTWEIGHT PINN FOR DATA ASSIMILATION
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
        input_dim = 2 # (x, y)
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.Tanh())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid()) # Output concentration ∈ [0, 1]
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
# 4. PINN TRAINER FOR SINGLE TIME SLICE
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
# 5. HYBRID FDM-PINN ASSIMILATION CONTROLLER
# =====================================================
class HybridFDMPINNAssimilation:
    """Main controller for hybrid FDM-PINN data assimilation"""
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
        observation_type: str = 'microscopy',
        coverage: float = None,
        noise_std: float = None,
        pinn_hidden_dims: List[int] = [32, 32, 32],
        n_epochs: int = 300,
        physics_weight: float = 0.2,
        damping_factor: float = 0.7,
        progress_callback = None
    ) -> Dict:
        """
        Run one assimilation cycle at observation time t_obs
        """
        if self.sim is None:
            raise ValueError("Simulation not initialized. Call initialize_simulation() first.")
        st.info(f"🔁 Starting assimilation cycle at t = {t_obs:.2e} s")
        # Step 1: Run FDM to observation time
        with st.spinner("Running FDM simulation..."):
            self.sim.run_until(t_obs)
            c_fdm_before = self.sim.c.copy()
            phi_fdm = self.sim.phi.copy()
        # Step 2: Generate sparse observations
        with st.spinner("Generating synthetic observations..."):
            obs_data = self.obs_gen.generate_sparse_observations(
                c_fdm_before,
                self.sim.dx,
                self.sim.scales.L0,
                observation_type=observation_type,
                measurement_time=t_obs,
                custom_coverage=coverage,
                custom_noise=noise_std
            )
        # Step 3: Initialize and train PINN
        with st.spinner("Training PINN for physics-aware reconstruction..."):
            Lx = self.sim.nx * self.sim.dx * self.sim.scales.L0
            Ly = self.sim.ny * self.sim.dx * self.sim.scales.L0
            pinn = LiFePO4AssimilationPINN(Lx, Ly, pinn_hidden_dims)
            trainer = PINNAssimilationTrainer(pinn)
            # Extract simulation parameters for PINN
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
        # Step 4: Reconstruct full field
        with st.spinner("Reconstructing full concentration field..."):
            c_pinn = trainer.reconstruct_full_field(
                self.sim.nx, self.sim.ny,
                self.sim.dx, self.sim.scales.L0
            )
        # Step 5: Compute correction and update FDM
        with st.spinner("Updating FDM state..."):
            correction = c_pinn - c_fdm_before
            correction_norm = np.linalg.norm(correction) / np.sqrt(correction.size)
            # Update FDM state with damping
            self.sim.c = c_fdm_before + damping_factor * correction
            # Store assimilation results
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
                progress_callback=update_progress
            )
            results.append(result)
        if progress_bar:
            progress_bar.progress(1.0)
            status_text.text("✅ Assimilation complete!")
        return results
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
# 6. VISUALIZATION FUNCTIONS
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
# =====================================================
# 7. STREAMLIT APPLICATION (FULLY EXPANDED)
# =====================================================
def main():
    """Main Streamlit application"""
    # Title and header
    st.markdown('<h1 class="main-header">🔄 LiFePO₄ Hybrid FDM-PINN Data Assimilation</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time correction of phase field simulations with sparse experimental observations</p>', unsafe_allow_html=True)
    # Initialize session state
    if 'hybrid_system' not in st.session_state:
        st.session_state.hybrid_system = HybridFDMPINNAssimilation()
    if 'sim_initialized' not in st.session_state:
        st.session_state.sim_initialized = False
    if 'assimilation_results' not in st.session_state:
        st.session_state.assimilation_results = []
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/battery--v1.png", width=80)
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
            # 🔥 NEW: Kinetics selector
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
            # 🔥 NEW: Warning for non-PNP + Random
            if kinetics_type != 0 and init_type == "Random":
                st.warning(
                    "⚠️ **Warning**: Non-PNP kinetics (BV/MHC) work best with **Lithiation/Delithiation** "
                    "initialization, which sets consistent boundary overpotentials. "
                    "Random initialization may cause unphysical fluxes."
                )
            if st.button("🔄 Initialize Simulation", use_container_width=True):
                with st.spinner("Initializing simulation..."):
                    st.session_state.hybrid_system.initialize_simulation(
                        nx=nx, ny=ny, dx=1.0, dt=dt, c_rate=c_rate
                    )
                    sim = st.session_state.hybrid_system.sim
                    # 🔥 NEW: Set kinetics BEFORE initialization
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
        # Assimilation configuration
        with st.expander("🔬 Assimilation Settings", expanded=True):
            observation_type = st.selectbox(
                "Observation Type",
                ["microscopy", "xrd_mapping", "tomography", "afm"],
                index=0,
                help="Type of measurement instrument"
            )
            coverage = st.slider(
                "Observation Coverage",
                0.01, 0.3, 0.1, 0.01,
                help="Fraction of grid points with measurements"
            )
            noise_std = st.slider(
                "Measurement Noise",
                0.0, 0.1, 0.05, 0.01,
                help="Standard deviation of Gaussian noise"
            )
            # PINN settings
            st.markdown("**🧠 PINN Configuration**")
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
            else: # Logarithmic
                num_cycles = st.slider("Number of Cycles", 3, 10, 5)
                total_time = st.number_input("Total Time (s)", 1000.0, 1e6, 50000.0)
                observation_schedule = np.logspace(np.log10(1000), np.log10(total_time), num_cycles).tolist()
            st.markdown("**Scheduled Observations:**")
            for i, t in enumerate(observation_schedule):
                st.caption(f"Cycle {i+1}: t = {t:.0f} s")
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
            st.session_state.hybrid_system.current_cycle = 0
            st.session_state.hybrid_system.assimilation_history = []
            st.success("Results cleared!")
            st.rerun()
    # Display current status
    st.divider()
    st.markdown("### 📊 Current Status")
    if st.session_state.sim_initialized:
        sim = st.session_state.hybrid_system.sim
        stats = sim.get_statistics()
        st.metric("Simulation Time", f"{stats['time_phys']:.2e} s")
        st.metric("Lithium Content", f"{stats['mean_c']:.3f}")
        st.metric("C-Rate", f"{stats['c_rate']}C")
        st.metric("Kinetics", ["PNP", "BV", "MHC"][stats['kinetics_type']])
        st.metric("Assimilation Cycles", len(st.session_state.assimilation_results))
    else:
        st.warning("Simulation not initialized")
    # Main content area
    if not st.session_state.sim_initialized:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class="card">
            <h3>🚀 Welcome to Hybrid FDM-PINN Assimilation</h3>
            <p>This system combines:</p>
            <ul>
            <li><strong>FDM Simulation:</strong> Physics-based phase field model</li>
            <li><strong>PINN Correction:</strong> Physics-informed neural networks</li>
            <li><strong>Real Data:</strong> Sparse experimental observations</li>
            <li><strong>Kinetics Models:</strong> PNP, BV, or MHC selectable</li>
            </ul>
            <p><strong>To get started:</strong></p>
            <ol>
            <li>Configure simulation settings in the sidebar</li>
            <li>Click "Initialize Simulation"</li>
            <li>Set up assimilation parameters</li>
            <li>Run assimilation cycles</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
        # Quick start buttons
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Quick Start: Default Setup", use_container_width=True):
                st.session_state.hybrid_system.initialize_simulation()
                sim = st.session_state.hybrid_system.sim
                sim.set_parameters(kinetics_type=0) # Default to PNP
                sim.initialize_lithiation()
                st.session_state.sim_initialized = True
                st.rerun()
        with col_b:
            if st.button("Demo Mode", use_container_width=True):
                # Run demo
                st.session_state.hybrid_system.initialize_simulation(nx=64, ny=64)
                sim = st.session_state.hybrid_system.sim
                sim.set_parameters(kinetics_type=0) # Demo uses PNP
                sim.initialize_lithiation()
                st.session_state.sim_initialized = True
                # Run quick assimilation
                progress_container = st.empty()
                results = st.session_state.hybrid_system.run_sequential_assimilation(
                    [1000.0, 5000.0, 10000.0],
                    observation_type='microscopy',
                    coverage=0.1,
                    progress_container=progress_container
                )
                st.session_state.assimilation_results = results
                st.rerun()
        return
    # Main simulation interface
    sim = st.session_state.hybrid_system.sim
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Simulation View",
        "🔬 Assimilation",
        "📊 Results",
        "📚 Documentation"
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
                st.metric("Domain", f"{stats['domain_size_nm']:.0f} nm")
                # 🔥 NEW: Kinetics in metrics panel
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
        st.subheader("Hybrid FDM-PINN Assimilation")
        # Run assimilation based on button clicks
        if run_single:
            if 'observation_schedule' in locals() or 'observation_schedule' in globals():
                t_obs = observation_schedule[min(st.session_state.hybrid_system.current_cycle,
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
                progress_container.empty()
                st.success(f"✅ Assimilation cycle {len(st.session_state.assimilation_results)} complete!")
                st.rerun()
        if run_sequential:
            # Ensure observation_schedule is defined
            if 'observation_schedule' not in locals():
                observation_schedule = [1000.0, 5000.0, 10000.0, 20000.0, 50000.0]
            progress_container = st.empty()
            results = st.session_state.hybrid_system.run_sequential_assimilation(
                observation_schedule,
                observation_type=observation_type,
                coverage=coverage,
                noise_std=noise_std,
                pinn_config={
                    'hidden_dims': hidden_dims,
                    'n_epochs': n_epochs,
                    'physics_weight': physics_weight
                },
                damping_factor=damping_factor,
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
                    # 🔥 NEW: Show kinetics type used
                    st.metric("Kinetics", ["PNP", "BV", "MHC"][sim.kinetics_type])
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
            # Show example of what assimilation does
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                <div class="card">
                <h4>🎯 What Assimilation Does</h4>
                <p>1. <strong>Run FDM</strong> to observation time</p>
                <p>2. <strong>Generate synthetic data</strong> mimicking real experiments</p>
                <p>3. <strong>Train lightweight PINN</strong> to reconstruct full field</p>
                <p>4. <strong>Correct FDM</strong> using PINN reconstruction</p>
                <p>5. <strong>Continue simulation</strong> from corrected state</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown("""
                <div class="card">
                <h4>📊 Expected Benefits</h4>
                <p>✅ <strong>3-5× accuracy improvement</strong> vs pure FDM</p>
                <p>✅ <strong>Handles sparse data</strong> (5-20% coverage)</p>
                <p>✅ <strong>Maintains physics</strong> via PDE constraints</p>
                <p>✅ <strong>Real-time calibration</strong> with experimental data</p>
                <p>✅ <strong>Adapts to degradation</strong> over battery lifetime</p>
                </div>
                """, unsafe_allow_html=True)
    with tab3:
        # Results and analysis
        st.subheader("Assimilation Performance Analysis")
        if st.session_state.assimilation_results:
            # Summary statistics
            stats = st.session_state.hybrid_system.get_assimilation_statistics()
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
                    # 🔥 NEW: Consistent kinetics info
                    st.metric("Kinetics", ["PNP", "BV", "MHC"][sim.kinetics_type])
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
                            'kinetics': ["PNP", "BV", "MHC"][sim.kinetics_type]
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
        # Documentation
        st.subheader("📚 Documentation & Theory")
        col1, col2 = st.columns([2, 1])
        with col1:
            with st.expander("🧠 How It Works", expanded=True):
                st.markdown("""
                ### Hybrid FDM-PINN Assimilation Workflow
                1. **FDM Forward Simulation**
                    - Physics-based Cahn-Hilliard with electrostatics
                    - Models LiFePO₄ phase separation under electric fields
                    - Includes C-rate effects and **selectable kinetics models (PNP/BV/MHC)**
                2. **Sparse Observation Generation**
                    - Mimics real experimental data (microscopy, XRD, etc.)
                    - 5-20% spatial coverage typical of real measurements
                    - Adds realistic measurement noise
                3. **Lightweight PINN Correction**
                    - Small neural network trained only at observation times
                    - Interpolates sparse data with physics constraints
                    - Enforces chemical potential equilibrium at interfaces
                4. **Sequential State Correction**
                    - PINN reconstruction corrects FDM state
                    - Damped correction for stability
                    - Continue FDM simulation from corrected state
                5. **Iterative Improvement**
                    - Repeat at each observation time
                    - System learns from accumulating data
                    - Adaptive to changing conditions
                """)
            with st.expander("⚡ Governing Equations", expanded=False):
                st.markdown("""
                ### Extended Cahn-Hilliard with Electrostatics
                **Free Energy Density:**""")
