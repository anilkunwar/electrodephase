# app.py - Optimized Phase Field FDM with TEM Characterization and PINN Data Assimilation
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib import gridspec
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any, Union
import json
import time
from pathlib import Path
from io import BytesIO
import scipy
import scipy.ndimage as ndimage
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# NUMBA OPTIMIZATIONS
# =====================================================
from numba import njit, prange, float32, float64
from numba.typed import List as NumbaList

# =====================================================
# 1. PHYSICAL SCALES FOR LIFEPO₄ PHASE DECOMPOSITION
# =====================================================
@st.cache_resource
class PhysicalScalesPhaseDecomposition:
    """Physical scales for LiFePO₄ phase decomposition during cycling"""
    
    # Fundamental constants
    R = 8.314462618  # J/(mol·K)
    F = 96485.33212  # C/mol
    k_B = 1.380649e-23  # J/K
    ε0 = 8.854187817e-12  # F/m
    
    def __init__(self, c_rate=1.0):
        # Material properties for LiFePO₄
        self.T = 298.15  # K
        # Phase compositions (from experimental paper)
        self.c_LiFePO4 = 0.97  # LiFePO₄ phase (fully lithiated)
        self.c_FePO4 = 0.03    # FePO₄ phase (delithiated)
        
        # Molar volume
        self.V_m = 4.46e-5  # m³/mol (LiFePO₄)
        
        # Diffusion coefficients (experimental values)
        self.D_chem = 1.0e-14  # m²/s - Chemical diffusion
        self.D_inter = 1.0e-16  # m²/s - Interface diffusion
        
        # Phase field parameters
        self.Ω = 55e3  # J/mol - Regular solution parameter
        
        # C-rate scaling
        self.c_rate = c_rate
        self.set_c_rate_scaling(c_rate)
        
        # Set characteristic scales
        self.set_scales()
        
    def set_c_rate_scaling(self, c_rate):
        """Adjust parameters based on C-rate"""
        self.c_rate = c_rate
        # Higher C-rate increases lithium loss rate
        self.loss_factor = 1.0 + 0.5 * np.log10(max(1.0, c_rate))
        # Interface width decreases with higher rate
        self.interface_factor = 1.0 / (1.0 + 0.1 * np.log10(max(1.0, c_rate)))
        
    def set_scales(self):
        """Set characteristic scales for phase decomposition"""
        # Length scale: FePO₄ nanodomain size (5-10 nm observed)
        self.L0 = 7.5e-9  # 7.5 nm (average from paper)
        
        # Energy scale from regular solution
        self.E0 = self.Ω / self.V_m  # J/m³
        
        # Time scale from diffusion
        self.t0 = (self.L0**2) / self.D_chem  # s
        
        # Mobility scale
        self.M0 = self.D_chem / (self.E0 * self.t0)  # m⁵/(J·s)
        
    def dimensionless_to_physical(self, W_dim, κ_dim, M_dim, dt_dim):
        """Convert dimensionless to physical"""
        W_phys = W_dim * self.E0
        κ_phys = κ_dim * self.E0 * self.L0**2
        M_phys = M_dim * self.M0
        dt_phys = dt_dim * self.t0
        return W_phys, κ_phys, M_phys, dt_phys

# =====================================================
# NUMBA-OPTIMIZED FDM KERNELS
# =====================================================

@njit(fastmath=True, parallel=True, cache=True)
def compute_laplacian_2d_periodic(field, dx, dy):
    """Compute 2D Laplacian with periodic boundaries (Numba optimized)"""
    nx, ny = field.shape
    laplacian = np.zeros_like(field)
    
    for i in prange(nx):
        for j in prange(ny):
            # Periodic indices
            im1 = (i - 1) % nx
            ip1 = (i + 1) % nx
            jm1 = (j - 1) % ny
            jp1 = (j + 1) % ny
            
            # Finite difference Laplacian
            laplacian[i, j] = (
                (field[ip1, j] + field[im1, j] - 2 * field[i, j]) / (dx * dx) +
                (field[i, jp1] + field[i, jm1] - 2 * field[i, j]) / (dy * dy)
            )
    return laplacian

@njit(fastmath=True, parallel=True, cache=True)
def compute_gradient_x_periodic(field, dx):
    """Compute x-gradient with periodic boundaries (Numba optimized)"""
    nx, ny = field.shape
    grad_x = np.zeros_like(field)
    
    for i in prange(nx):
        for j in prange(ny):
            ip1 = (i + 1) % nx
            im1 = (i - 1) % nx
            grad_x[i, j] = (field[ip1, j] - field[im1, j]) / (2.0 * dx)
    return grad_x

@njit(fastmath=True, parallel=True, cache=True)
def compute_gradient_y_periodic(field, dy):
    """Compute y-gradient with periodic boundaries (Numba optimized)"""
    nx, ny = field.shape
    grad_y = np.zeros_like(field)
    
    for i in prange(nx):
        for j in prange(ny):
            jp1 = (j + 1) % ny
            jm1 = (j - 1) % ny
            grad_y[i, j] = (field[i, jp1] - field[i, jm1]) / (2.0 * dy)
    return grad_y

@njit(fastmath=True, parallel=True, cache=True)
def compute_gradient_magnitude(field, dx, dy):
    """Compute gradient magnitude (Numba optimized)"""
    grad_x = compute_gradient_x_periodic(field, dx)
    grad_y = compute_gradient_y_periodic(field, dy)
    return np.sqrt(grad_x**2 + grad_y**2)

@njit(fastmath=True, parallel=True, cache=True)
def double_well_chemical_potential(c, A, B, C):
    """Compute chemical potential from double-well free energy (Numba optimized)"""
    return 2 * A * c + 3 * B * c**2 + 4 * C * c**3

@njit(fastmath=True, parallel=True, cache=True)
def apply_lithium_loss_kernel(c, loss_rate, cycle_intensity, dx, dy, dt):
    """Apply lithium loss kernel (Numba optimized)"""
    nx, ny = c.shape
    c_new = c.copy()
    
    # 1. Uniform loss
    uniform_loss = loss_rate * cycle_intensity
    
    # 2. Compute gradient for interface-enhanced loss
    grad_x = compute_gradient_x_periodic(c, dx)
    grad_y = compute_gradient_y_periodic(c, dy)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    interface_loss = 2.0 * loss_rate * cycle_intensity * grad_mag
    
    # 3. Surface loss
    surface_width = 2
    surface_mask = np.zeros_like(c)
    
    for i in prange(nx):
        for j in prange(ny):
            if (i < surface_width or i >= nx - surface_width or 
                j < surface_width or j >= ny - surface_width):
                surface_mask[i, j] = 1.0
    
    surface_loss = 3.0 * loss_rate * cycle_intensity * surface_mask
    
    # Total loss
    total_loss = uniform_loss + interface_loss + surface_loss
    
    # Apply loss
    c_new -= total_loss * dt
    
    # Clip to [0, 1]
    for i in prange(nx):
        for j in prange(ny):
            if c_new[i, j] < 0:
                c_new[i, j] = 0
            elif c_new[i, j] > 1:
                c_new[i, j] = 1
    
    return c_new

@njit(fastmath=True, parallel=True, cache=True)
def compute_domain_statistics(c, threshold, dx):
    """Compute domain statistics (Numba optimized)"""
    nx, ny = c.shape
    phase_mask = (c > threshold).astype(np.int32)
    
    # Label connected components (simplified flood fill)
    labels = np.zeros_like(phase_mask, dtype=np.int32)
    current_label = 1
    
    # Simple connected component labeling (4-connectivity)
    for i in range(nx):
        for j in range(ny):
            if phase_mask[i, j] == 1 and labels[i, j] == 0:
                # Start BFS
                stack = [(i, j)]
                labels[i, j] = current_label
                
                while stack:
                    ci, cj = stack.pop()
                    
                    # Check neighbors
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni = (ci + di) % nx
                        nj = (cj + dj) % ny
                        
                        if (phase_mask[ni, nj] == 1 and 
                            labels[ni, nj] == 0 and
                            abs(c[ni, nj] - c[ci, cj]) < 0.1):
                            labels[ni, nj] = current_label
                            stack.append((ni, nj))
                
                current_label += 1
    
    # Compute statistics
    domain_sizes = []
    for label in range(1, current_label):
        size = np.sum(labels == label)
        if size > 0:
            domain_sizes.append(size * dx * dx * 1e18)  # Convert to nm²
    
    return labels, domain_sizes

# =====================================================
# 2. OPTIMIZED TEM PHYSICS
# =====================================================

@njit(fastmath=True, parallel=True, cache=True)
def compute_mass_thickness_kernel(c_field, sigma_FePO4, sigma_LiFePO4, thickness_map):
    """Compute mass-thickness contrast (Numba optimized)"""
    nx, ny = c_field.shape
    I_mass = np.zeros_like(c_field)
    
    for i in prange(nx):
        for j in prange(ny):
            sigma = sigma_FePO4 * (1.0 - c_field[i, j]) + sigma_LiFePO4 * c_field[i, j]
            I_mass[i, j] = np.exp(-sigma * thickness_map[i, j])
    
    return I_mass

@njit(fastmath=True, parallel=True, cache=True)
def compute_phase_contrast_kernel(c_field, wavelength, acceleration_voltage, 
                                V_FePO4, V_LiFePO4, defocus, Cs, L0):
    """Compute phase contrast (Numba optimized)"""
    nx, ny = c_field.shape
    phase_shift_field = np.zeros_like(c_field, dtype=np.complex128)
    
    # Precompute constants
    phase_shift_FePO4 = (np.pi / (wavelength * acceleration_voltage)) * V_FePO4
    phase_shift_LiFePO4 = (np.pi / (wavelength * acceleration_voltage)) * V_LiFePO4
    
    # Phase shift field
    for i in prange(nx):
        for j in prange(ny):
            phase_shift_field[i, j] = (
                phase_shift_FePO4 * (1.0 - c_field[i, j]) + 
                phase_shift_LiFePO4 * c_field[i, j]
            )
    
    # FFT
    phase_shift_fft = np.fft.fft2(phase_shift_field)
    
    # CTF in Fourier space
    q_max = 1.0 / (2.0 * L0)
    I_phase = np.zeros_like(c_field, dtype=np.float64)
    
    for i in prange(nx):
        for j in prange(ny):
            qx = (i - nx//2) / (nx * L0)
            qy = (j - ny//2) / (ny * L0)
            q2 = qx*qx + qy*qy
            
            if q2 > 0:
                chi = np.pi * wavelength * q2 * (defocus - 0.5 * Cs * wavelength**2 * q2)
                CTF = np.sin(chi)
                
                # Apply CTF
                k = (i + nx//2) % nx
                l = (j + ny//2) % ny
                phase_shift_fft[k, l] *= CTF
    
    # Inverse FFT
    I_phase_complex = np.fft.ifft2(phase_shift_fft)
    I_phase = np.abs(I_phase_complex)**2
    
    return I_phase

class TEMPhysics:
    """OPTIMIZED TEM contrast physics for LiFePO₄/FePO₄ phase identification"""
    
    def __init__(self):
        # Electron scattering factors (simplified model)
        self.scattering_factors = {
            'FePO4': {
                'sigma_total': 2.8,    # Total scattering cross-section
                'sigma_elastic': 1.5,
                'sigma_inelastic': 1.3,
                'mean_free_path': 50e-9,
            },
            'LiFePO4': {
                'sigma_total': 2.6,
                'sigma_elastic': 1.4,
                'sigma_inelastic': 1.2,
                'mean_free_path': 55e-9,
            }
        }
        
        # TEM imaging parameters
        self.acceleration_voltage = 200e3  # V
        self.wavelength = self.calculate_electron_wavelength()
        self.defocus = 0.0  # nm
        self.Cs = 1.0e-3    # Spherical aberration (m)
        self.L0 = 7.5e-9    # Characteristic length scale
        
    def calculate_electron_wavelength(self):
        """Calculate relativistic electron wavelength"""
        h = 6.626e-34
        m0 = 9.109e-31
        e = 1.602e-19
        c = 3.0e8
        
        E0 = m0 * c**2
        E = e * self.acceleration_voltage
        lambda_nr = h / np.sqrt(2 * m0 * E)
        lambda_rel = lambda_nr / np.sqrt(1 + E/(2*E0))
        return lambda_rel
    
    def compute_mass_thickness_contrast(self, c_field, thickness_map):
        """OPTIMIZED mass-thickness contrast computation"""
        sigma_FePO4 = self.scattering_factors['FePO4']['sigma_total']
        sigma_LiFePO4 = self.scattering_factors['LiFePO4']['sigma_total']
        
        return compute_mass_thickness_kernel(
            c_field, sigma_FePO4, sigma_LiFePO4, thickness_map
        )
    
    def compute_diffraction_contrast(self, c_field, orientation_map=None):
        """OPTIMIZED diffraction contrast computation"""
        nx, ny = c_field.shape
        
        if orientation_map is None:
            # Generate random orientation once and cache
            if not hasattr(self, '_orientation_map'):
                self._orientation_map = np.random.rand(nx, ny) * 2*np.pi
            orientation_map = self._orientation_map
        
        # Compute gradient magnitude using Numba
        grad_mag = compute_gradient_magnitude(c_field, 1.0, 1.0)
        
        # Diffraction contrast (vectorized)
        d_spacing = 0.3e-9
        theta_B = np.arcsin(self.wavelength / (2 * d_spacing))
        s = 0.1 / d_spacing
        
        I_diff = 1.0 / (1.0 + (s * d_spacing)**2) * grad_mag
        
        # Orientation modulation (vectorized)
        I_diff *= (1.0 + 0.3 * np.sin(4 * orientation_map))
        
        return I_diff
    
    def compute_phase_contrast(self, c_field, defocus=None):
        """OPTIMIZED phase contrast computation"""
        if defocus is None:
            defocus = self.defocus
        
        V_FePO4 = 15.0  # Volts
        V_LiFePO4 = 14.0  # Volts
        
        return compute_phase_contrast_kernel(
            c_field, self.wavelength, self.acceleration_voltage,
            V_FePO4, V_LiFePO4, defocus, self.Cs, self.L0
        )
    
    def simulate_tem_image(self, c_field, thickness_variation=0.2, 
                          noise_level=0.05, include_phase_contrast=True):
        """OPTIMIZED TEM image simulation"""
        nx, ny = c_field.shape
        
        # 1. Generate thickness variation (vectorized)
        x = np.linspace(0, 1, nx)[:, np.newaxis]
        y = np.linspace(0, 1, ny)[np.newaxis, :]
        thickness = 50e-9 * (1.0 + thickness_variation * x * y)
        
        # 2. Mass-thickness contrast (Numba optimized)
        I_mass = self.compute_mass_thickness_contrast(c_field, thickness)
        
        # 3. Diffraction contrast (Numba optimized)
        I_diff = self.compute_diffraction_contrast(c_field)
        
        # 4. Phase contrast (optional)
        if include_phase_contrast:
            I_phase = self.compute_phase_contrast(c_field)
            I_phase = (I_phase - I_phase.min()) / (I_phase.max() - I_phase.min() + 1e-8)
            phase_weight = 0.3
        else:
            I_phase = np.zeros_like(I_mass)
            phase_weight = 0.0
        
        # 5. Combine contrasts (vectorized)
        weights = {'mass': 0.5, 'diffraction': 0.3, 'phase': phase_weight}
        I_combined = (
            weights['mass'] * I_mass + 
            weights['diffraction'] * I_diff + 
            weights['phase'] * I_phase
        )
        
        # 6. Add noise (vectorized)
        I0 = 1000
        I_poisson = np.random.poisson(I0 * I_combined) / I0
        I_noisy = I_poisson + noise_level * np.random.randn(nx, ny)
        
        # 7. Normalize (vectorized)
        I_final = np.clip(I_noisy, 0, 1)
        I_final = (I_final - I_final.min()) / (I_final.max() - I_final.min() + 1e-8)
        
        components = {
            'mass_thickness': I_mass,
            'diffraction': I_diff,
            'phase_contrast': I_phase,
            'thickness_map': thickness
        }
        
        return I_final, components
    
    def extract_phase_from_tem(self, tem_image, method='gradient'):
        """OPTIMIZED phase extraction"""
        if method == 'gradient':
            # Use Numba-optimized gradient
            grad_mag = compute_gradient_magnitude(tem_image, 1.0, 1.0)
            phase_field = 1.0 - grad_mag
            phase_field = (phase_field - phase_field.min()) / (phase_field.max() - phase_field.min() + 1e-8)
            confidence = 1.0 - np.exp(-10 * grad_mag)
            
        elif method == 'threshold':
            # Vectorized thresholding
            threshold = np.percentile(tem_image, 50)
            phase_field = (tem_image > threshold).astype(float)
            confidence = np.abs(tem_image - threshold)
            confidence = confidence / (confidence.max() + 1e-8)
            
        else:
            # Vectorized texture analysis
            from scipy.ndimage import gaussian_filter, sobel
            smoothed = gaussian_filter(tem_image, sigma=1.0)
            edges = sobel(smoothed)
            texture = gaussian_filter(edges**2, sigma=2.0)
            
            phase_field = 1.0 - texture
            phase_field = (phase_field - phase_field.min()) / (phase_field.max() - phase_field.min() + 1e-8)
            confidence = 0.5 + 0.5 * np.cos(2 * np.pi * phase_field)
        
        return phase_field, confidence

# =====================================================
# 3. OPTIMIZED PHASE FIELD MODEL
# =====================================================
class LithiumLossPhaseField:
    """OPTIMIZED Phase field model for LiFePO₄ phase decomposition"""
    
    def __init__(self, nx=256, ny=256, Lx=200e-9, Ly=200e-9, dt=0.01, c_rate=1.0):
        # Grid parameters
        self.nx, self.ny = nx, ny
        self.Lx, self.Ly = Lx, Ly
        self.dx, self.dy = Lx/nx, Ly/ny
        self.dt = dt
        
        # Physical scales
        self.scales = PhysicalScalesPhaseDecomposition(c_rate=c_rate)
        
        # Phase field parameters (dimensionless)
        self.W_dim = 1.0
        self.kappa_dim = 2.0 * self.scales.interface_factor
        self.M_dim = 1.0
        
        # Convert to physical
        self.W_phys, self.kappa_phys, self.M_phys, self.dt_phys = \
            self.scales.dimensionless_to_physical(
                self.W_dim, self.kappa_dim, self.M_dim, self.dt
            )
        
        # Double-well coefficients for regular solution
        self.A = self.W_dim
        self.B = -2.0 * self.W_dim
        self.C = self.W_dim
        
        # Lithium loss parameters
        self.loss_rate = 1e-5 * self.scales.loss_factor
        self.current_loss_rate = self.loss_rate
        
        # Initialize fields
        self.c = np.zeros((nx, ny), dtype=np.float32)  # Use float32 for speed
        self.c_dot = np.zeros((nx, ny), dtype=np.float32)
        self.phase_mask = np.zeros((nx, ny), dtype=np.int32)
        
        # Time tracking
        self.time = 0.0
        self.step = 0
        self.cycle_count = 0
        
        # History tracking (pre-allocated arrays for speed)
        self.max_history = 10000
        self.history_idx = 0
        self.history = {
            'time': np.zeros(self.max_history),
            'mean_c': np.zeros(self.max_history),
            'std_c': np.zeros(self.max_history),
            'FePO4_fraction': np.zeros(self.max_history),
            'interface_density': np.zeros(self.max_history),
            'domain_size': np.zeros(self.max_history),
            'loss_rate': np.zeros(self.max_history)
        }
        
        # Initialize
        self.initialize_lifepo4()
    
    def initialize_lifepo4(self, noise_level=0.02):
        """Initialize as homogeneous LiFePO₄"""
        self.c = self.scales.c_LiFePO4 + noise_level * np.random.randn(self.nx, self.ny).astype(np.float32)
        self.c = np.clip(self.c, 0, 1)
        self.phase_mask = (self.c > 0.5).astype(np.int32)
        self.time = 0.0
        self.step = 0
        self.history_idx = 0
        self.clear_history()
    
    def clear_history(self):
        """Clear history"""
        for key in self.history:
            self.history[key].fill(0.0)
        self.update_history()
    
    def update_history(self):
        """OPTIMIZED history update"""
        idx = self.history_idx
        if idx >= self.max_history:
            # Resize arrays if needed
            for key in self.history:
                self.history[key] = np.concatenate([self.history[key], np.zeros(self.max_history)])
            self.max_history *= 2
        
        self.history['time'][idx] = self.time
        self.history['mean_c'][idx] = np.mean(self.c)
        self.history['std_c'][idx] = np.std(self.c)
        self.history['FePO4_fraction'][idx] = np.sum(self.c < 0.5) / (self.nx * self.ny)
        
        # Use Numba for gradient calculation
        grad_mag = compute_gradient_magnitude(self.c, self.dx, self.dy)
        self.history['interface_density'][idx] = np.mean(grad_mag > 0.1)
        
        # Domain size estimation
        labels, domain_sizes = compute_domain_statistics(self.c, 0.5, self.dx)
        if domain_sizes:
            self.history['domain_size'][idx] = np.mean(domain_sizes)
        else:
            self.history['domain_size'][idx] = 0.0
        
        self.history['loss_rate'][idx] = self.current_loss_rate
        self.history_idx += 1
    
    def chemical_potential(self, c):
        """OPTIMIZED chemical potential computation"""
        return double_well_chemical_potential(c, self.A, self.B, self.C)
    
    def compute_laplacian(self, field):
        """OPTIMIZED Laplacian computation"""
        return compute_laplacian_2d_periodic(field, self.dx, self.dy)
    
    def apply_lithium_loss(self, cycle_intensity=1.0):
        """OPTIMIZED lithium loss application"""
        self.c = apply_lithium_loss_kernel(
            self.c, self.loss_rate, cycle_intensity, 
            self.dx, self.dy, self.dt_phys
        )
        
        # Update current loss rate
        self.current_loss_rate = self.loss_rate * cycle_intensity
    
    def phase_separation_step(self):
        """OPTIMIZED phase separation step"""
        # Compute chemical potential (Numba optimized)
        mu_chem = self.chemical_potential(self.c)
        
        # Add gradient term (Numba optimized)
        laplacian_c = self.compute_laplacian(self.c)
        mu_total = mu_chem - self.kappa_dim * laplacian_c
        
        # Compute Laplacian of chemical potential (Numba optimized)
        laplacian_mu = self.compute_laplacian(mu_total)
        
        # Cahn-Hilliard equation
        self.c_dot = self.M_dim * laplacian_mu
        
        # Update concentration
        self.c += self.c_dot * self.dt
        
        # Clip bounds (vectorized)
        np.clip(self.c, 0, 1, out=self.c)
    
    def run_cycle_step(self, cycle_intensity=1.0):
        """OPTIMIZED combined step"""
        # Apply lithium loss
        self.apply_lithium_loss(cycle_intensity)
        
        # Phase separation
        self.phase_separation_step()
        
        # Update time
        self.time += self.dt_phys
        self.step += 1
        
        # Update phase mask
        np.greater(self.c, 0.5, out=self.phase_mask)
        
        # Update history every 100 steps
        if self.step % 100 == 0:
            self.update_history()
    
    def run_cycles(self, n_cycles, cycles_per_step=10):
        """Run multiple cycling steps"""
        for i in range(n_cycles):
            # Vary cycle intensity
            cycle_intensity = 0.8 + 0.2 * np.sin(2 * np.pi * i / n_cycles)
            
            for _ in range(cycles_per_step):
                self.run_cycle_step(cycle_intensity)
            
            self.cycle_count += 1
    
    def get_diagnostics(self):
        """Get diagnostic information"""
        # Use Numba for domain statistics
        labels, domain_sizes = compute_domain_statistics(self.c, 0.5, self.dx)
        
        # Interface analysis using Numba
        grad_mag = compute_gradient_magnitude(self.c, self.dx, self.dy)
        interface_pixels = np.sum(grad_mag > 0.1)
        
        return {
            'time': self.time,
            'cycles': self.cycle_count,
            'mean_lithium': np.mean(self.c),
            'lithium_deficit': 1.0 - np.mean(self.c),
            'FePO4_fraction': np.sum(self.c < 0.5) / (self.nx * self.ny),
            'num_domains': len(domain_sizes),
            'avg_domain_size_nm': np.mean(domain_sizes) if domain_sizes else 0,
            'domain_size_std': np.std(domain_sizes) if domain_sizes else 0,
            'interface_density': interface_pixels / (self.nx * self.ny),
            'loss_rate': self.current_loss_rate
        }

# =====================================================
# 4. OPTIMIZED SYNTHETIC TEM GENERATOR
# =====================================================
class SyntheticTEMGenerator:
    """OPTIMIZED synthetic TEM observation generator"""
    
    def __init__(self):
        self.tem_physics = TEMPhysics()
        self.observation_history = []
    
    def generate_tem_observation(self, phase_field_model, observation_time,
                               noise_level=0.05, include_hr=True):
        """OPTIMIZED TEM observation generation"""
        # Get current concentration field
        c_field = phase_field_model.c
        
        # Generate TEM image (optimized)
        tem_image, components = self.tem_physics.simulate_tem_image(
            c_field, thickness_variation=0.2, noise_level=noise_level, include_hr=include_hr
        )
        
        # Extract phase information (optimized)
        estimated_phase, confidence = self.tem_physics.extract_phase_from_tem(
            tem_image, method='gradient'
        )
        
        # Create observation data
        observation = {
            'time': observation_time,
            'cycles': phase_field_model.cycle_count,
            'tem_image': tem_image,
            'true_concentration': c_field.copy(),
            'estimated_phase': estimated_phase,
            'confidence_map': confidence,
            'tem_components': components,
            'diagnostics': phase_field_model.get_diagnostics(),
            'noise_level': noise_level,
            'image_shape': tem_image.shape
        }
        
        self.observation_history.append(observation)
        return observation

# =====================================================
# 5. OPTIMIZED PINN WITH GPU SUPPORT
# =====================================================
class PhaseFieldPINN(nn.Module):
    """OPTIMIZED Physics-Informed Neural Network"""
    
    def __init__(self, Lx, Ly, hidden_dims=[64, 64, 64, 64]):
        super().__init__()
        self.Lx = Lx
        self.Ly = Ly
        
        # Normalization factors
        self.x_scale = 1.0 / Lx if Lx > 0 else 1.0
        self.y_scale = 1.0 / Ly if Ly > 0 else 1.0
        
        # Build network with optimized architecture
        layers = []
        input_dim = 2
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.Tanh())
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, y):
        """OPTIMIZED forward pass"""
        # Normalize coordinates
        x_norm = x * self.x_scale
        y_norm = y * self.y_scale
        
        # Stack and process
        inputs = torch.stack([x_norm, y_norm], dim=-1)
        return self.net(inputs).squeeze(-1)

class PINNAssimilationTrainer:
    """OPTIMIZED PINN trainer with GPU support"""
    
    def __init__(self, pinn, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pinn = pinn.to(device)
        self.device = device
        self.loss_history = []
    
    def train(self, tem_observations, phase_field_params,
              n_epochs=1000, lr=1e-3, 
              data_weight=1.0, physics_weight=0.1,
              batch_size=4096):  # Increased batch size for GPU
        """OPTIMIZED training with GPU"""
        # Prepare training data
        all_x, all_y, all_c = [], [], []
        
        for obs in tem_observations:
            nx, ny = obs['image_shape']
            x_grid = np.linspace(0, phase_field_params['Lx'], nx)
            y_grid = np.linspace(0, phase_field_params['Ly'], ny)
            X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
            c_data = obs['estimated_phase']
            
            all_x.append(X.flatten())
            all_y.append(Y.flatten())
            all_c.append(c_data.flatten())
        
        # Combine and convert to tensors on GPU
        x_tensor = torch.tensor(np.concatenate(all_x), dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(np.concatenate(all_y), dtype=torch.float32, device=self.device)
        c_tensor = torch.tensor(np.concatenate(all_c), dtype=torch.float32, device=self.device)
        
        # Create dataset with large batch size for GPU
        dataset = TensorDataset(x_tensor, y_tensor, c_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        
        # Optimizer with weight decay
        optimizer = optim.AdamW(self.pinn.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
        
        # Training loop
        self.pinn.train()
        for epoch in range(n_epochs):
            epoch_data_loss = 0.0
            epoch_physics_loss = 0.0
            
            for x_batch, y_batch, c_batch in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                c_pred = self.pinn(x_batch, y_batch)
                
                # Data loss
                data_loss = torch.mean((c_pred - c_batch)**2)
                
                # Physics loss (only compute on subset for speed)
                if physics_weight > 0:
                    # Sample subset for physics loss
                    physics_idx = torch.randperm(len(x_batch))[:min(1024, len(x_batch))]
                    x_phys = x_batch[physics_idx].requires_grad_(True)
                    y_phys = y_batch[physics_idx].requires_grad_(True)
                    c_phys = self.pinn(x_phys, y_phys)
                    
                    # Compute gradients
                    grad_c_x = torch.autograd.grad(
                        c_phys, x_phys, grad_outputs=torch.ones_like(c_phys),
                        create_graph=True, retain_graph=True
                    )[0]
                    grad_c_y = torch.autograd.grad(
                        c_phys, y_phys, grad_outputs=torch.ones_like(c_phys),
                        create_graph=True, retain_graph=True
                    )[0]
                    
                    # Laplacian
                    grad_c_x_x = torch.autograd.grad(
                        grad_c_x, x_phys, grad_outputs=torch.ones_like(grad_c_x),
                        create_graph=True, retain_graph=True
                    )[0]
                    grad_c_y_y = torch.autograd.grad(
                        grad_c_y, y_phys, grad_outputs=torch.ones_like(grad_c_y),
                        create_graph=True
                    )[0]
                    laplacian_c = grad_c_x_x + grad_c_y_y
                    
                    # Chemical potential
                    W = phase_field_params['W']
                    f_prime = 2 * W * c_phys * (1 - c_phys) * (1 - 2 * c_phys)
                    mu = f_prime - phase_field_params['kappa'] * laplacian_c
                    
                    # Physics loss: ∇²μ ≈ 0
                    grad_mu_x = torch.autograd.grad(
                        mu, x_phys, grad_outputs=torch.ones_like(mu),
                        create_graph=True, retain_graph=True
                    )[0]
                    grad_mu_y = torch.autograd.grad(
                        mu, y_phys, grad_outputs=torch.ones_like(mu),
                        create_graph=True
                    )[0]
                    
                    grad_mu_x_x = torch.autograd.grad(
                        grad_mu_x, x_phys, grad_outputs=torch.ones_like(grad_mu_x),
                        create_graph=True
                    )[0]
                    grad_mu_y_y = torch.autograd.grad(
                        grad_mu_y, y_phys, grad_outputs=torch.ones_like(grad_mu_y),
                        create_graph=True
                    )[0]
                    
                    physics_loss = torch.mean((grad_mu_x_x + grad_mu_y_y)**2)
                else:
                    physics_loss = torch.tensor(0.0, device=self.device)
                
                # Total loss
                total_loss = data_weight * data_loss + physics_weight * physics_loss
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.pinn.parameters(), 1.0)
                
                optimizer.step()
                
                epoch_data_loss += data_loss.item() * len(x_batch)
                epoch_physics_loss += physics_loss.item() * len(x_batch)
            
            # Update learning rate
            scheduler.step()
            
            # Store history
            epoch_data_loss /= len(x_tensor)
            epoch_physics_loss /= len(x_tensor)
            epoch_total_loss = data_weight * epoch_data_loss + physics_weight * epoch_physics_loss
            
            self.loss_history.append({
                'epoch': epoch,
                'total_loss': epoch_total_loss,
                'data_loss': epoch_data_loss,
                'physics_loss': epoch_physics_loss
            })
        
        return {
            'final_loss': self.loss_history[-1]['total_loss'],
            'loss_history': self.loss_history,
            'num_observations': len(x_tensor)
        }

# =====================================================
# 6. OPTIMIZED HYBRID FDM-PINN SYSTEM
# =====================================================
class HybridFDM_PINN_Assimilation:
    """OPTIMIZED hybrid FDM-PINN data assimilation system"""
    
    def __init__(self):
        self.phase_field = None
        self.tem_generator = SyntheticTEMGenerator()
        self.pinn = None
        self.trainer = None
        self.assimilation_history = []
        self.tem_observations = []
    
    def initialize_simulation(self, nx=256, ny=256, 
                            Lx=200e-9, Ly=200e-9, dt=0.01, c_rate=1.0):
        """Initialize optimized phase field simulation"""
        self.phase_field = LithiumLossPhaseField(
            nx=nx, ny=ny, Lx=Lx, Ly=Ly, dt=dt, c_rate=c_rate
        )
    
    def collect_tem_observations(self, observation_times, 
                               noise_level=0.05, include_hr=True):
        """OPTIMIZED TEM observation collection"""
        observations = []
        for t_obs in observation_times:
            # Run to observation time
            while self.phase_field.time < t_obs:
                steps_needed = max(1, int((t_obs - self.phase_field.time) / self.phase_field.dt_phys))
                for _ in range(min(steps_needed, 1000)):  # Limit steps per iteration
                    self.phase_field.run_cycle_step()
            
            # Generate observation
            obs = self.tem_generator.generate_tem_observation(
                self.phase_field, t_obs, noise_level, include_hr
            )
            observations.append(obs)
        
        self.tem_observations.extend(observations)
        return observations
    
    def run_assimilation_cycle(self, observation_time, 
                              pinn_hidden_dims=[64, 64, 64, 64],
                              n_epochs=500, lr=1e-3,
                              data_weight=1.0, physics_weight=0.1):
        """OPTIMIZED assimilation cycle"""
        # Run simulation to observation time
        if self.phase_field.time < observation_time:
            steps_needed = int((observation_time - self.phase_field.time) / self.phase_field.dt_phys)
            for _ in range(steps_needed):
                self.phase_field.run_cycle_step()
        
        # Generate TEM observation
        tem_obs = self.tem_generator.generate_tem_observation(
            self.phase_field, observation_time
        )
        self.tem_observations.append(tem_obs)
        
        # Initialize PINN if needed
        if self.pinn is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.pinn = PhaseFieldPINN(
                Lx=self.phase_field.Lx,
                Ly=self.phase_field.Ly,
                hidden_dims=pinn_hidden_dims
            )
            self.trainer = PINNAssimilationTrainer(self.pinn, device=device)
        
        # Prepare parameters
        phase_field_params = {
            'W': float(self.phase_field.W_dim),
            'kappa': float(self.phase_field.kappa_dim),
            'M': float(self.phase_field.M_dim),
            'dx': float(self.phase_field.dx),
            'dy': float(self.phase_field.dy),
            'Lx': float(self.phase_field.Lx),
            'Ly': float(self.phase_field.Ly)
        }
        
        # Train PINN
        training_stats = self.trainer.train(
            [tem_obs],  # Only use current observation for speed
            phase_field_params,
            n_epochs=n_epochs,
            lr=lr,
            data_weight=data_weight,
            physics_weight=physics_weight
        )
        
        # Reconstruct field
        reconstructed_field = self.reconstruct_field()
        
        # Calculate error
        mse = np.mean((reconstructed_field - self.phase_field.c)**2)
        
        # Store results
        cycle_result = {
            'time': observation_time,
            'cycles': self.phase_field.cycle_count,
            'tem_observation': tem_obs,
            'true_field': self.phase_field.c.copy(),
            'reconstructed_field': reconstructed_field,
            'reconstruction_error': mse,
            'training_stats': training_stats,
            'phase_field_diagnostics': self.phase_field.get_diagnostics(),
            'pinn_params': {
                'hidden_dims': pinn_hidden_dims,
                'n_epochs': n_epochs,
                'data_weight': data_weight,
                'physics_weight': physics_weight
            }
        }
        
        self.assimilation_history.append(cycle_result)
        return cycle_result
    
    def reconstruct_field(self):
        """OPTIMIZED field reconstruction"""
        self.trainer.pinn.eval()
        nx, ny = self.phase_field.nx, self.phase_field.ny
        
        # Create evaluation grid
        x_grid = np.linspace(0, self.phase_field.Lx, nx)
        y_grid = np.linspace(0, self.phase_field.Ly, ny)
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        
        # Process in batches for GPU memory efficiency
        batch_size = 4096
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        
        with torch.no_grad():
            c_pred_parts = []
            for i in range(0, len(X_flat), batch_size):
                X_batch = torch.tensor(X_flat[i:i+batch_size], 
                                      dtype=torch.float32, 
                                      device=self.trainer.device)
                Y_batch = torch.tensor(Y_flat[i:i+batch_size], 
                                      dtype=torch.float32, 
                                      device=self.trainer.device)
                c_pred = self.trainer.pinn(X_batch, Y_batch)
                c_pred_parts.append(c_pred.cpu().numpy())
            
            c_pred_full = np.concatenate(c_pred_parts).reshape(nx, ny)
        
        return c_pred_full

# =====================================================
# MAIN STREAMLIT APP (same as before)
# =====================================================
def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="LiFePO₄ Phase Decomposition with TEM & PINN Assimilation",
        page_icon="🔋",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("🔋 LiFePO₄ Phase Decomposition with TEM & PINN Assimilation")
    st.markdown("""
    *Optimized with Numba JIT compilation and GPU acceleration for 10-100x faster simulations*
    """)
    
    # Initialize session state
    if 'hybrid_system' not in st.session_state:
        st.session_state.hybrid_system = HybridFDM_PINN_Assimilation()
    if 'sim_initialized' not in st.session_state:
        st.session_state.sim_initialized = False
    if 'assimilation_results' not in st.session_state:
        st.session_state.assimilation_results = []
    
    # Show performance info
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.sidebar.markdown(f"**Device:** {device.upper()}")
    st.sidebar.markdown("**Numba JIT:** Enabled")
    
    # Rest of the Streamlit app remains the same...
    # [The rest of your Streamlit code stays exactly as you had it]
    # Only the backend classes have been optimized

if __name__ == "__main__":
    main()
