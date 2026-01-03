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
from numba import njit, prange, float32, float64, complex128, void, types
from numba.experimental import jitclass
import numba
import concurrent.futures
from functools import lru_cache
warnings.filterwarnings('ignore')

# =====================================================
# Numba Configuration and Performance Monitoring
# =====================================================
class PerformanceMonitor:
    """Monitor and optimize performance"""
    
    def __init__(self):
        self.timers = {}
        self.memory_usage = {}
        self.history = []
        
    def start_timer(self, name):
        self.timers[name] = time.perf_counter()
    
    def stop_timer(self, name):
        if name in self.timers:
            elapsed = time.perf_counter() - self.timers[name]
            self.history.append((name, elapsed, time.time()))
            return elapsed
        return 0.0
    
    def get_performance_stats(self):
        """Get comprehensive performance statistics"""
        if not self.history:
            return {}
        
        df = pd.DataFrame(self.history, columns=['operation', 'time', 'timestamp'])
        stats = df.groupby('operation').agg({
            'time': ['mean', 'std', 'min', 'max', 'count']
        }).round(6)
        
        return {
            'summary': stats,
            'total_operations': len(df),
            'total_time': df['time'].sum(),
            'avg_time_per_op': df['time'].mean(),
            'slowest_operations': df.nlargest(5, 'time').to_dict('records')
        }

# Global performance monitor
perf_mon = PerformanceMonitor()

# =====================================================
# Numba-Optimized Core Functions
# =====================================================

@njit(fastmath=True, parallel=True, cache=True, nogil=True)
def compute_laplacian_2d_periodic(field, dx, dy):
    """Compute 2D Laplacian with periodic boundaries"""
    nx, ny = field.shape
    laplacian = np.zeros_like(field)
    
    dx2 = dx * dx
    dy2 = dy * dy
    
    for i in prange(nx):
        im1 = (i - 1) % nx
        ip1 = (i + 1) % nx
        
        for j in prange(ny):
            jm1 = (j - 1) % ny
            jp1 = (j + 1) % ny
            
            laplacian[i, j] = (
                (field[ip1, j] + field[im1, j] - 2.0 * field[i, j]) / dx2 +
                (field[i, jp1] + field[i, jm1] - 2.0 * field[i, j]) / dy2
            )
    
    return laplacian

@njit(fastmath=True, parallel=True, cache=True, nogil=True)
def compute_gradient_2d_periodic(field, dx, dy):
    """Compute gradient in x and y directions with periodic boundaries"""
    nx, ny = field.shape
    grad_x = np.zeros_like(field)
    grad_y = np.zeros_like(field)
    
    dx2 = 2.0 * dx
    dy2 = 2.0 * dy
    
    for i in prange(nx):
        im1 = (i - 1) % nx
        ip1 = (i + 1) % nx
        
        for j in prange(ny):
            jm1 = (j - 1) % ny
            jp1 = (j + 1) % ny
            
            grad_x[i, j] = (field[ip1, j] - field[im1, j]) / dx2
            grad_y[i, j] = (field[i, jp1] - field[i, jm1]) / dy2
    
    return grad_x, grad_y

@njit(fastmath=True, parallel=True, cache=True, nogil=True)
def compute_laplacian_2d_neumann(field, dx, dy):
    """Compute 2D Laplacian with Neumann boundary conditions (zero flux)"""
    nx, ny = field.shape
    laplacian = np.zeros_like(field)
    
    dx2 = dx * dx
    dy2 = dy * dy
    
    for i in prange(nx):
        for j in prange(ny):
            # x-direction
            if i == 0:
                # Left boundary: du/dx = 0
                lap_x = (2.0 * field[1, j] - 2.0 * field[0, j]) / dx2
            elif i == nx - 1:
                # Right boundary: du/dx = 0
                lap_x = (2.0 * field[nx-2, j] - 2.0 * field[nx-1, j]) / dx2
            else:
                lap_x = (field[i+1, j] + field[i-1, j] - 2.0 * field[i, j]) / dx2
            
            # y-direction
            if j == 0:
                # Bottom boundary: du/dy = 0
                lap_y = (2.0 * field[i, 1] - 2.0 * field[i, 0]) / dy2
            elif j == ny - 1:
                # Top boundary: du/dy = 0
                lap_y = (2.0 * field[i, ny-2] - 2.0 * field[i, ny-1]) / dy2
            else:
                lap_y = (field[i, j+1] + field[i, j-1] - 2.0 * field[i, j]) / dy2
            
            laplacian[i, j] = lap_x + lap_y
    
    return laplacian

@njit(fastmath=True, parallel=True, cache=True, nogil=True)
def chemical_potential_double_well(c, A, B, C):
    """Compute chemical potential from double-well free energy"""
    nx, ny = c.shape
    mu = np.zeros_like(c)
    
    for i in prange(nx):
        for j in prange(ny):
            c_val = c[i, j]
            mu[i, j] = 2.0 * A * c_val + 3.0 * B * c_val * c_val + 4.0 * C * c_val * c_val * c_val
    
    return mu

@njit(fastmath=True, parallel=True, cache=True, nogil=True)
def evolve_cahn_hilliard(c, dt, dx, dy, A, B, C, kappa, M):
    """Single step of Cahn-Hilliard equation with periodic boundaries"""
    # Compute chemical potential
    mu_chem = chemical_potential_double_well(c, A, B, C)
    
    # Add gradient energy term
    lap_c = compute_laplacian_2d_periodic(c, dx, dy)
    mu_total = mu_chem - kappa * lap_c
    
    # Compute Laplacian of total chemical potential
    lap_mu = compute_laplacian_2d_periodic(mu_total, dx, dy)
    
    # Update concentration
    c_new = c + M * dt * lap_mu
    
    # Apply bounds
    for i in prange(c_new.shape[0]):
        for j in prange(c_new.shape[1]):
            if c_new[i, j] < 0.0:
                c_new[i, j] = 0.0
            elif c_new[i, j] > 1.0:
                c_new[i, j] = 1.0
    
    return c_new

@njit(fastmath=True, parallel=True, cache=True, nogil=True)
def compute_domain_statistics(c, threshold, dx, dy):
    """Compute connected domain statistics with flood fill algorithm"""
    nx, ny = c.shape
    visited = np.zeros((nx, ny), dtype=np.int32)
    labels = np.zeros((nx, ny), dtype=np.int32)
    domain_sizes = []
    domain_areas = []
    domain_perimeters = []
    
    current_label = 1
    
    # 4-connectivity neighbors
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    for i in range(nx):
        for j in range(ny):
            if c[i, j] > threshold and visited[i, j] == 0:
                # Start BFS
                stack = [(i, j)]
                visited[i, j] = 1
                labels[i, j] = current_label
                size = 1
                perimeter = 0
                
                while stack:
                    ci, cj = stack.pop()
                    
                    # Count perimeter
                    for di, dj in directions:
                        ni = (ci + di) % nx
                        nj = (cj + dj) % ny
                        
                        # Check if neighbor is in different phase
                        if c[ni, nj] <= threshold:
                            perimeter += 1
                    
                    # Add neighbors
                    for di, dj in directions:
                        ni = (ci + di) % nx
                        nj = (cj + dj) % ny
                        
                        if (c[ni, nj] > threshold and 
                            visited[ni, nj] == 0 and
                            abs(c[ni, nj] - c[ci, cj]) < 0.1):  # Similar concentration
                            visited[ni, nj] = 1
                            labels[ni, nj] = current_label
                            stack.append((ni, nj))
                            size += 1
                
                if size > 0:
                    domain_sizes.append(size)
                    domain_areas.append(size * dx * dy * 1e18)  # Convert to nm²
                    domain_perimeters.append(perimeter)
                    current_label += 1
    
    # Calculate statistics
    if domain_sizes:
        avg_size = np.mean(domain_sizes)
        std_size = np.std(domain_sizes)
        avg_area = np.mean(domain_areas)
        avg_perimeter = np.mean(domain_perimeters)
        circularity = 4 * np.pi * avg_area / (avg_perimeter * avg_perimeter) if avg_perimeter > 0 else 0
    else:
        avg_size = 0.0
        std_size = 0.0
        avg_area = 0.0
        avg_perimeter = 0.0
        circularity = 0.0
    
    return {
        'labels': labels,
        'num_domains': len(domain_sizes),
        'avg_size': avg_size,
        'std_size': std_size,
        'avg_area_nm2': avg_area,
        'avg_perimeter': avg_perimeter,
        'circularity': circularity,
        'size_distribution': domain_sizes
    }

@njit(fastmath=True, parallel=True, cache=True, nogil=True)
def compute_interface_properties(c, dx, dy, threshold=0.5):
    """Compute interface properties including width and curvature"""
    nx, ny = c.shape
    
    # Compute gradients
    grad_x, grad_y = compute_gradient_2d_periodic(c, dx, dy)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Interface width estimation
    interface_mask = (grad_mag > np.percentile(grad_mag, 90)).astype(np.float32)
    
    # Compute curvature
    curvature = np.zeros_like(c)
    
    for i in prange(1, nx-1):
        for j in prange(1, ny-1):
            if interface_mask[i, j] > 0.5:
                # Compute second derivatives
                cxx = (c[i+1, j] - 2*c[i, j] + c[i-1, j]) / (dx*dx)
                cyy = (c[i, j+1] - 2*c[i, j] + c[i, j-1]) / (dy*dy)
                cxy = (c[i+1, j+1] - c[i+1, j-1] - c[i-1, j+1] + c[i-1, j-1]) / (4*dx*dy)
                
                denom = (1.0 + grad_x[i, j]**2 + grad_y[i, j]**2)**1.5
                if denom > 1e-8:
                    curvature[i, j] = (cxx*(1 + grad_y[i, j]**2) + 
                                      cyy*(1 + grad_x[i, j]**2) - 
                                      2*grad_x[i, j]*grad_y[i, j]*cxy) / denom
    
    # Statistics
    interface_pixels = np.sum(interface_mask)
    total_pixels = nx * ny
    interface_density = interface_pixels / total_pixels
    
    # Average gradient magnitude at interface
    avg_grad_mag = np.sum(grad_mag * interface_mask) / interface_pixels if interface_pixels > 0 else 0.0
    
    # Average curvature
    interface_curvature = curvature[interface_mask > 0.5]
    avg_curvature = np.mean(interface_curvature) if len(interface_curvature) > 0 else 0.0
    
    return {
        'gradient_magnitude': grad_mag,
        'interface_mask': interface_mask,
        'curvature': curvature,
        'interface_density': interface_density,
        'avg_gradient_magnitude': avg_grad_mag,
        'avg_curvature': avg_curvature,
        'interface_pixels': interface_pixels
    }

@njit(fastmath=True, parallel=True, cache=True, nogil=True)
def compute_tem_mass_thickness(c, thickness, sigma_FePO4, sigma_LiFePO4):
    """Compute mass-thickness contrast for TEM"""
    nx, ny = c.shape
    contrast = np.zeros_like(c)
    
    for i in prange(nx):
        for j in prange(ny):
            # Linear interpolation of scattering cross-section
            sigma = sigma_FePO4 * (1.0 - c[i, j]) + sigma_LiFePO4 * c[i, j]
            # Exponential attenuation
            contrast[i, j] = np.exp(-sigma * thickness[i, j])
    
    return contrast

@njit(fastmath=True, parallel=True, cache=True, nogil=True)
def compute_tem_diffraction(c, wavelength, d_spacing):
    """Compute diffraction contrast for TEM"""
    nx, ny = c.shape
    
    # Compute gradient magnitude (phase boundaries)
    grad_x, grad_y = compute_gradient_2d_periodic(c, 1.0, 1.0)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Bragg condition
    theta_B = np.arcsin(wavelength / (2.0 * d_spacing))
    s = 0.1 / d_spacing  # Deviation parameter
    
    # Diffraction contrast intensity
    contrast = np.zeros_like(c)
    for i in prange(nx):
        for j in prange(ny):
            contrast[i, j] = 1.0 / (1.0 + (s * d_spacing)**2) * grad_mag[i, j]
    
    return contrast

@njit(fastmath=True, parallel=True, cache=True, nogil=True)
def compute_fft_2d_real(field):
    """Compute 2D FFT of real-valued field"""
    return np.fft.fft2(field)

@njit(fastmath=True, parallel=True, cache=True, nogil=True)
def compute_phase_contrast(c, wavelength, voltage, defocus, Cs, thickness):
    """Compute phase contrast for HRTEM"""
    nx, ny = c.shape
    
    # Phase shift from mean inner potential
    # Values for LiFePO4 and FePO4 (approximate)
    V_FePO4 = 15.0  # Volts
    V_LiFePO4 = 14.0  # Volts
    
    # Phase shift field
    phase_shift = np.zeros((nx, ny), dtype=np.complex128)
    
    # Precompute constant
    const = np.pi / (wavelength * voltage)
    
    for i in prange(nx):
        for j in prange(ny):
            # Interpolate potential based on concentration
            V = V_FePO4 * (1.0 - c[i, j]) + V_LiFePO4 * c[i, j]
            phase_shift[i, j] = const * V * thickness[i, j]
    
    # FFT
    phase_shift_fft = np.fft.fft2(phase_shift)
    
    # Contrast Transfer Function
    contrast = np.zeros((nx, ny), dtype=np.complex128)
    qx_max = 1.0 / (2.0 * (nx * 1e-10))  # Assuming 0.1 nm pixel size
    qy_max = 1.0 / (2.0 * (ny * 1e-10))
    
    for i in prange(nx):
        for j in prange(ny):
            # Spatial frequencies
            qx = (i - nx//2) * qx_max / nx
            qy = (j - ny//2) * qy_max / ny
            q2 = qx*qx + qy*qy
            
            if q2 > 0:
                # CTF parameters
                chi = np.pi * wavelength * q2 * (defocus - 0.5 * Cs * wavelength**2 * q2)
                CTF = np.sin(chi)
                
                # Apply CTF
                idx = (i + nx//2) % nx
                idy = (j + ny//2) % ny
                contrast[idx, idy] = phase_shift_fft[idx, idy] * CTF
    
    # Inverse FFT
    image_complex = np.fft.ifft2(contrast)
    image_intensity = np.abs(image_complex)**2
    
    return image_intensity

# =====================================================
# Enhanced Physical Scales with Validation
# =====================================================
@st.cache_resource
class PhysicalScalesPhaseDecomposition:
    """Enhanced physical scales for LiFePO₄ with validation"""
    
    # Fundamental constants
    R = 8.314462618  # J/(mol·K)
    F = 96485.33212  # C/mol
    k_B = 1.380649e-23  # J/K
    ε0 = 8.854187817e-12  # F/m
    NA = 6.02214076e23  # Avogadro's number
    
    def __init__(self, c_rate=1.0, temperature=298.15, pressure=1.0e5):
        # Material properties for LiFePO₄
        self.T = temperature  # K
        self.P = pressure  # Pa
        
        # Phase compositions (from experimental paper)
        self.c_LiFePO4 = 0.97  # LiFePO₄ phase (fully lithiated)
        self.c_FePO4 = 0.03    # FePO₄ phase (delithiated)
        
        # Molar volume (LiFePO₄)
        self.V_m = 4.46e-5  # m³/mol
        
        # Density
        self.rho = self.calculate_density()
        
        # Diffusion coefficients (temperature dependent)
        self.D0_chem = 1.0e-14  # Pre-exponential factor
        self.Ea_chem = 0.3  # Activation energy in eV
        self.D_chem = self.calculate_diffusion_coefficient()
        
        # Phase field parameters
        self.Ω = 55e3  # J/mol - Regular solution parameter
        self.gamma = 0.1  # J/m² - Surface energy
        
        # C-rate scaling
        self.c_rate = c_rate
        self.set_c_rate_scaling(c_rate)
        
        # Set characteristic scales
        self.set_scales()
        
        # Validate parameters
        self.validate_parameters()
        
    def calculate_density(self):
        """Calculate density from molar volume"""
        # Molar mass of LiFePO₄ (g/mol)
        M_LiFePO4 = 157.76  # g/mol
        return M_LiFePO4 / (self.V_m * 1e6)  # kg/m³
    
    def calculate_diffusion_coefficient(self):
        """Calculate temperature-dependent diffusion coefficient"""
        # Arrhenius equation
        k_B_eV = 8.617333262145e-5  # eV/K
        return self.D0_chem * np.exp(-self.Ea_chem / (k_B_eV * self.T))
    
    def set_c_rate_scaling(self, c_rate):
        """Adjust parameters based on C-rate"""
        self.c_rate = c_rate
        # Higher C-rate increases lithium loss rate
        self.loss_factor = 1.0 + 0.5 * np.log10(max(0.1, c_rate))
        # Interface width decreases with higher rate
        self.interface_factor = 1.0 / (1.0 + 0.1 * np.log10(max(0.1, c_rate)))
        # Mobility scaling
        self.mobility_factor = 1.0 / (1.0 + 0.2 * np.log10(max(0.1, c_rate)))
        
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
        
        # Interface width scale
        self.l_int = np.sqrt(self.E0 * self.L0**2 / self.gamma)  # m
        
        # Characteristic velocity
        self.v0 = self.L0 / self.t0  # m/s
        
        print(f"Physical scales initialized:")
        print(f"  Length scale: {self.L0:.2e} m ({self.L0*1e9:.1f} nm)")
        print(f"  Time scale: {self.t0:.2e} s")
        print(f"  Interface width: {self.l_int:.2e} m ({self.l_int*1e9:.1f} nm)")
        print(f"  Characteristic velocity: {self.v0:.2e} m/s")
        
    def validate_parameters(self):
        """Validate physical parameters for consistency"""
        warnings = []
        
        # Check diffusion coefficient
        if self.D_chem < 1e-20 or self.D_chem > 1e-10:
            warnings.append(f"Diffusion coefficient {self.D_chem:.2e} m²/s is outside typical range")
        
        # Check interface width
        if self.l_int < 1e-10 or self.l_int > 1e-7:
            warnings.append(f"Interface width {self.l_int:.2e} m may be unrealistic")
        
        # Check time scale
        if self.t0 < 1e-12 or self.t0 > 1e6:
            warnings.append(f"Time scale {self.t0:.2e} s may be unrealistic")
        
        if warnings:
            print("Parameter validation warnings:")
            for w in warnings:
                print(f"  - {w}")
        
    def dimensionless_to_physical(self, W_dim, κ_dim, M_dim, dt_dim):
        """Convert dimensionless to physical"""
        W_phys = W_dim * self.E0
        κ_phys = κ_dim * self.E0 * self.L0**2
        M_phys = M_dim * self.M0 * self.mobility_factor
        dt_phys = dt_dim * self.t0
        return W_phys, κ_phys, M_phys, dt_phys
    
    def physical_to_dimensionless(self, W_phys, κ_phys, M_phys, dt_phys):
        """Convert physical to dimensionless"""
        W_dim = W_phys / self.E0
        κ_dim = κ_phys / (self.E0 * self.L0**2)
        M_dim = M_phys / (self.M0 * self.mobility_factor)
        dt_dim = dt_phys / self.t0
        return W_dim, κ_dim, M_dim, dt_dim

# =====================================================
# Enhanced TEM Physics with Realistic Parameters
# =====================================================
class EnhancedTEMPhysics:
    """Enhanced TEM physics with realistic parameters and validation"""
    
    def __init__(self, acceleration_voltage=200e3, Cs=1.0e-3, defocus=0.0):
        # TEM parameters
        self.acceleration_voltage = acceleration_voltage  # V
        self.Cs = Cs  # Spherical aberration (m)
        self.defocus = defocus  # m
        
        # Calculate electron wavelength
        self.wavelength = self.calculate_electron_wavelength()
        
        # Material properties for LiFePO₄ and FePO₄
        self.scattering_factors = self.get_scattering_factors()
        
        # Imaging parameters
        self.dose = 1e4  # e⁻/Å²
        self.detector_gain = 1.0
        self.detector_noise = 5.0  # RMS electrons
        
        # Cache for frequently used data
        self._cache = {}
        
    def calculate_electron_wavelength(self):
        """Calculate relativistic electron wavelength with high precision"""
        h = 6.62607015e-34  # Planck constant (J·s)
        m0 = 9.1093837015e-31  # Electron rest mass (kg)
        e = 1.602176634e-19  # Elementary charge (C)
        c = 299792458  # Speed of light (m/s)
        
        # Relativistic kinetic energy
        E_kin = e * self.acceleration_voltage
        E0 = m0 * c**2  # Rest energy
        
        # Total energy
        E_total = E_kin + E0
        
        # Relativistic momentum
        p = np.sqrt(E_total**2 - E0**2) / c
        
        # De Broglie wavelength
        wavelength = h / p
        
        return wavelength
    
    def get_scattering_factors(self):
        """Get scattering factors from experimental data or approximations"""
        # Based on electron scattering factors for LiFePO₄ and FePO₄
        return {
            'FePO4': {
                'sigma_total': 2.8e-28,  # m²/atom
                'sigma_elastic': 1.5e-28,
                'sigma_inelastic': 1.3e-28,
                'mean_free_path': 50e-9,
                'density': 3.6e3,  # kg/m³
                'mean_inner_potential': 15.0,  # V
            },
            'LiFePO4': {
                'sigma_total': 2.6e-28,
                'sigma_elastic': 1.4e-28,
                'sigma_inelastic': 1.2e-28,
                'mean_free_path': 55e-9,
                'density': 3.5e3,
                'mean_inner_potential': 14.0,
            }
        }
    
    def generate_thickness_map(self, shape, thickness_mean=50e-9, thickness_variation=0.2):
        """Generate realistic thickness variation map"""
        nx, ny = shape
        
        # Base thickness with random variation
        thickness = thickness_mean * np.ones(shape)
        
        # Add wedge shape (common in TEM samples)
        x = np.linspace(0, 1, nx)[:, np.newaxis]
        y = np.linspace(0, 1, ny)[np.newaxis, :]
        
        # Wedge gradient
        wedge = 1.0 + thickness_variation * (x + y) / 2.0
        
        # Add random roughness
        roughness = 0.05 * np.random.randn(nx, ny)
        
        thickness = thickness * wedge * (1.0 + roughness)
        
        # Ensure positive thickness
        thickness = np.maximum(10e-9, thickness)  # Minimum 10 nm
        
        return thickness
    
    def simulate_tem_image(self, c_field, thickness_variation=0.2, 
                          noise_level=0.05, include_phase_contrast=True):
        """Generate synthetic TEM image with multiple contrast mechanisms"""
        perf_mon.start_timer('tem_simulation')
        
        nx, ny = c_field.shape
        
        # 1. Generate thickness map
        thickness_map = self.generate_thickness_map((nx, ny), thickness_variation=thickness_variation)
        
        # 2. Mass-thickness contrast (Numba optimized)
        sigma_FePO4 = self.scattering_factors['FePO4']['sigma_total']
        sigma_LiFePO4 = self.scattering_factors['LiFePO4']['sigma_total']
        I_mass = compute_tem_mass_thickness(c_field, thickness_map, sigma_FePO4, sigma_LiFePO4)
        
        # 3. Diffraction contrast (Numba optimized)
        d_spacing = 0.3e-9  # Typical d-spacing for LiFePO₄ (200)
        I_diff = compute_tem_diffraction(c_field, self.wavelength, d_spacing)
        
        # 4. Phase contrast (optional)
        if include_phase_contrast:
            I_phase = compute_phase_contrast(
                c_field, self.wavelength, self.acceleration_voltage,
                self.defocus, self.Cs, thickness_map
            )
            I_phase = (I_phase - I_phase.min()) / (I_phase.max() - I_phase.min() + 1e-8)
            phase_weight = 0.3
        else:
            I_phase = np.zeros_like(I_mass)
            phase_weight = 0.0
        
        # 5. Combine contrasts
        weights = {'mass': 0.4, 'diffraction': 0.3, 'phase': phase_weight}
        I_combined = (weights['mass'] * I_mass + 
                     weights['diffraction'] * I_diff + 
                     weights['phase'] * I_phase)
        
        # 6. Add realistic noise
        I_noisy = self.add_realistic_noise(I_combined, self.dose, self.detector_noise)
        
        # 7. Normalize
        I_final = self.normalize_image(I_noisy)
        
        components = {
            'mass_thickness': I_mass,
            'diffraction': I_diff,
            'phase_contrast': I_phase,
            'thickness_map': thickness_map,
            'combined': I_combined
        }
        
        perf_mon.stop_timer('tem_simulation')
        return I_final, components
    
    def add_realistic_noise(self, image, dose, detector_noise):
        """Add realistic TEM noise including shot noise and detector noise"""
        # Convert to electron counts
        I0 = dose * 1e20  # Convert from e⁻/Å² to e⁻/m²
        
        # Shot noise (Poisson)
        I_counts = np.random.poisson(I0 * image)
        
        # Detector noise (Gaussian)
        I_detector = I_counts + detector_noise * np.random.randn(*image.shape)
        
        # Convert back to intensity
        I_noisy = I_detector / I0
        
        return I_noisy
    
    def normalize_image(self, image):
        """Normalize image with histogram equalization"""
        # Clip outliers
        p_low, p_high = np.percentile(image, [1, 99])
        image_clipped = np.clip(image, p_low, p_high)
        
        # Normalize to [0, 1]
        image_norm = (image_clipped - image_clipped.min()) / (image_clipped.max() - image_clipped.min() + 1e-8)
        
        return image_norm
    
    def extract_phase_information(self, tem_image, method='advanced'):
        """Extract phase information from TEM image using advanced methods"""
        if method == 'gradient_based':
            # Use gradient information
            grad_x, grad_y = compute_gradient_2d_periodic(tem_image, 1.0, 1.0)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # Phase field from inverse gradient
            phase_field = 1.0 - grad_mag / (grad_mag.max() + 1e-8)
            
            # Confidence based on gradient magnitude
            confidence = 1.0 - np.exp(-10 * grad_mag)
            
        elif method == 'ml_enhanced':
            # ML-like approach with texture analysis
            from scipy.ndimage import gaussian_filter, laplace
            
            # Multi-scale analysis
            scales = [1.0, 2.0, 4.0]
            features = []
            
            for sigma in scales:
                smoothed = gaussian_filter(tem_image, sigma=sigma)
                features.append(smoothed)
                features.append(laplace(smoothed))
                features.append(gaussian_filter(np.gradient(smoothed)[0]**2, sigma=sigma*2))
            
            # Combine features
            combined = np.mean(features, axis=0)
            
            # Normalize
            phase_field = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
            confidence = 0.5 + 0.3 * np.cos(2 * np.pi * phase_field)
            
        else:  # Adaptive threshold
            # Otsu-like adaptive threshold
            hist, bins = np.histogram(tem_image.flatten(), bins=256)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            # Find optimal threshold
            total = hist.sum()
            sum_total = np.dot(bin_centers, hist)
            
            sum_back = 0
            w_back = 0
            max_var = 0
            threshold = 0
            
            for i in range(len(hist)):
                w_back += hist[i]
                if w_back == 0:
                    continue
                
                w_fore = total - w_back
                if w_fore == 0:
                    break
                
                sum_back += bin_centers[i] * hist[i]
                mean_back = sum_back / w_back
                mean_fore = (sum_total - sum_back) / w_fore
                
                # Between-class variance
                var_between = w_back * w_fore * (mean_back - mean_fore)**2
                
                if var_between > max_var:
                    max_var = var_between
                    threshold = bin_centers[i]
            
            phase_field = (tem_image > threshold).astype(float)
            confidence = np.abs(tem_image - threshold)
            confidence = confidence / (confidence.max() + 1e-8)
        
        return phase_field, confidence

# =====================================================
# Enhanced Phase Field Model with Multiple Features
# =====================================================
class EnhancedLithiumLossPhaseField:
    """Enhanced phase field model with multiple lithium loss mechanisms"""
    
    def __init__(self, nx=256, ny=256, Lx=200e-9, Ly=200e-9, 
                 dt=0.01, c_rate=1.0, boundary_condition='periodic'):
        # Grid parameters
        self.nx, self.ny = nx, ny
        self.Lx, self.Ly = Lx, Ly
        self.dx, self.dy = Lx/nx, Ly/ny
        self.dt = dt
        
        # Boundary condition
        self.boundary_condition = boundary_condition  # 'periodic' or 'neumann'
        
        # Physical scales
        self.scales = PhysicalScalesPhaseDecomposition(c_rate=c_rate)
        
        # Phase field parameters (dimensionless)
        self.W_dim = 1.0  # Double-well height
        self.kappa_dim = 2.0 * self.scales.interface_factor  # Gradient energy
        self.M_dim = 1.0  # Mobility
        
        # Convert to physical
        self.W_phys, self.kappa_phys, self.M_phys, self.dt_phys = \
            self.scales.dimensionless_to_physical(
                self.W_dim, self.kappa_dim, self.M_dim, self.dt
            )
        
        # Double-well coefficients
        self.A = self.W_dim
        self.B = -2.0 * self.W_dim
        self.C = self.W_dim
        
        # Lithium loss parameters
        self.loss_rate_base = 1e-5 * self.scales.loss_factor
        self.current_loss_rate = self.loss_rate_base
        
        # Multiple loss mechanisms
        self.loss_mechanisms = {
            'uniform': True,      # Uniform SEI formation
            'interface': True,    # Interface-enhanced loss
            'surface': True,      # Surface loss
            'stress': False,      # Stress-induced loss (future)
            'diffusion': True     # Diffusion-limited loss
        }
        
        # Initialize fields
        self.c = np.zeros((nx, ny), dtype=np.float32)  # Lithium concentration
        self.c_dot = np.zeros((nx, ny), dtype=np.float32)
        self.phase_mask = np.zeros((nx, ny), dtype=np.int32)
        
        # Additional fields for analysis
        self.chemical_potential_field = np.zeros((nx, ny), dtype=np.float32)
        self.gradient_energy = np.zeros((nx, ny), dtype=np.float32)
        
        # Time tracking
        self.time = 0.0
        self.step = 0
        self.cycle_count = 0
        
        # History tracking with pre-allocation
        self.max_history = 100000
        self.history_idx = 0
        self.history = {
            'time': np.zeros(self.max_history),
            'mean_c': np.zeros(self.max_history),
            'std_c': np.zeros(self.max_history),
            'FePO4_fraction': np.zeros(self.max_history),
            'interface_density': np.zeros(self.max_history),
            'domain_size_nm': np.zeros(self.max_history),
            'loss_rate': np.zeros(self.max_history),
            'chemical_potential_mean': np.zeros(self.max_history),
            'gradient_energy_mean': np.zeros(self.max_history)
        }
        
        # Initialize
        self.initialize_lifepo4()
        
        # Performance optimization
        self._cache_valid = False
        self._cached_gradient = None
        self._cached_laplacian = None
        
    def initialize_lifepo4(self, noise_level=0.02, pattern=None):
        """Initialize with various patterns"""
        if pattern == 'homogeneous':
            self.c = self.scales.c_LiFePO4 * np.ones((self.nx, self.ny))
            
        elif pattern == 'gradient':
            # Linear gradient
            x = np.linspace(0, 1, self.nx)[:, np.newaxis]
            self.c = self.scales.c_LiFePO4 * (0.8 + 0.2 * x)
            
        elif pattern == 'circular_seed':
            # Circular FePO₄ seed in center
            self.c = self.scales.c_LiFePO4 * np.ones((self.nx, self.ny))
            center_x, center_y = self.nx//2, self.ny//2
            radius = min(self.nx, self.ny) // 4
            
            for i in range(self.nx):
                for j in range(self.ny):
                    dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                    if dist < radius:
                        self.c[i, j] = self.scales.c_FePO4
        
        else:  # Random
            self.c = self.scales.c_LiFePO4 * np.ones((self.nx, self.ny))
            self.c += noise_level * np.random.randn(self.nx, self.ny)
        
        # Clip and update
        self.c = np.clip(self.c, 0, 1).astype(np.float32)
        self.phase_mask = (self.c > 0.5).astype(np.int32)
        self.time = 0.0
        self.step = 0
        self.history_idx = 0
        self.clear_history()
        self._cache_valid = False
    
    def clear_history(self):
        """Clear history arrays"""
        for key in self.history:
            self.history[key].fill(0.0)
        self.history_idx = 0
        self.update_history()
    
    def update_history(self):
        """Update history with comprehensive statistics"""
        idx = self.history_idx
        if idx >= self.max_history:
            # Double array size
            new_size = self.max_history * 2
            for key in self.history:
                self.history[key] = np.concatenate([self.history[key], np.zeros(new_size - self.max_history)])
            self.max_history = new_size
        
        # Basic statistics
        self.history['time'][idx] = self.time
        self.history['mean_c'][idx] = np.mean(self.c)
        self.history['std_c'][idx] = np.std(self.c)
        self.history['FePO4_fraction'][idx] = np.sum(self.c < 0.5) / (self.nx * self.ny)
        self.history['loss_rate'][idx] = self.current_loss_rate
        
        # Compute interface properties
        interface_props = compute_interface_properties(self.c, self.dx, self.dy)
        self.history['interface_density'][idx] = interface_props['interface_density']
        
        # Compute domain statistics
        domain_stats = compute_domain_statistics(self.c, 0.5, self.dx, self.dy)
        self.history['domain_size_nm'][idx] = np.sqrt(domain_stats['avg_area_nm2']) if domain_stats['avg_area_nm2'] > 0 else 0.0
        
        # Chemical potential and gradient energy
        mu_chem = chemical_potential_double_well(self.c, self.A, self.B, self.C)
        grad_x, grad_y = compute_gradient_2d_periodic(self.c, self.dx, self.dy)
        grad_energy = 0.5 * self.kappa_dim * (grad_x**2 + grad_y**2)
        
        self.history['chemical_potential_mean'][idx] = np.mean(mu_chem)
        self.history['gradient_energy_mean'][idx] = np.mean(grad_energy)
        
        self.history_idx += 1
    
    def compute_laplacian(self, field):
        """Compute Laplacian with specified boundary conditions"""
        if self.boundary_condition == 'periodic':
            return compute_laplacian_2d_periodic(field, self.dx, self.dy)
        else:  # Neumann
            return compute_laplacian_2d_neumann(field, self.dx, self.dy)
    
    def apply_lithium_loss_advanced(self, cycle_intensity=1.0):
        """Apply multiple lithium loss mechanisms"""
        total_loss = np.zeros_like(self.c)
        
        # 1. Uniform loss (SEI formation)
        if self.loss_mechanisms['uniform']:
            uniform_loss = self.loss_rate_base * cycle_intensity
            total_loss += uniform_loss
        
        # 2. Interface-enhanced loss
        if self.loss_mechanisms['interface']:
            grad_x, grad_y = compute_gradient_2d_periodic(self.c, self.dx, self.dy)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            interface_loss = 2.0 * self.loss_rate_base * cycle_intensity * grad_mag
            total_loss += interface_loss
        
        # 3. Surface loss
        if self.loss_mechanisms['surface']:
            surface_width = 3
            surface_mask = np.zeros_like(self.c)
            surface_mask[:surface_width, :] = 1
            surface_mask[-surface_width:, :] = 1
            surface_mask[:, :surface_width] = 1
            surface_mask[:, -surface_width:] = 1
            surface_loss = 3.0 * self.loss_rate_base * cycle_intensity * surface_mask
            total_loss += surface_loss
        
        # 4. Diffusion-limited loss
        if self.loss_mechanisms['diffusion']:
            # Loss proportional to concentration gradient
            lap_c = self.compute_laplacian(self.c)
            diffusion_loss = 0.1 * self.loss_rate_base * cycle_intensity * np.abs(lap_c)
            total_loss += diffusion_loss
        
        # Apply loss
        self.c -= total_loss * self.dt_phys
        
        # Ensure bounds
        np.clip(self.c, 0.0, 1.0, out=self.c)
        
        # Update current loss rate
        self.current_loss_rate = np.mean(total_loss)
        
        # Invalidate cache
        self._cache_valid = False
    
    def phase_separation_step_optimized(self):
        """Optimized phase separation step"""
        # Use Numba-optimized function
        self.c = evolve_cahn_hilliard(
            self.c, self.dt, self.dx, self.dy,
            self.A, self.B, self.C, self.kappa_dim, self.M_dim
        )
        
        # Update phase mask
        np.greater(self.c, 0.5, out=self.phase_mask)
        
        # Invalidate cache
        self._cache_valid = False
    
    def run_cycle_step(self, cycle_intensity=1.0):
        """Run one complete cycle step"""
        perf_mon.start_timer('cycle_step')
        
        # Apply lithium loss
        self.apply_lithium_loss_advanced(cycle_intensity)
        
        # Phase separation
        self.phase_separation_step_optimized()
        
        # Update time
        self.time += self.dt_phys
        self.step += 1
        
        # Update history every 100 steps
        if self.step % 100 == 0:
            self.update_history()
        
        perf_mon.stop_timer('cycle_step')
    
    def run_cycles_batch(self, n_cycles, cycles_per_step=10, progress_callback=None):
        """Run multiple cycles with progress reporting"""
        for i in range(n_cycles):
            # Vary cycle intensity
            cycle_intensity = 0.8 + 0.2 * np.sin(2 * np.pi * i / n_cycles)
            
            for _ in range(cycles_per_step):
                self.run_cycle_step(cycle_intensity)
            
            self.cycle_count += 1
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(i + 1, n_cycles)
    
    def get_comprehensive_diagnostics(self):
        """Get comprehensive diagnostic information"""
        # Domain statistics
        domain_stats = compute_domain_statistics(self.c, 0.5, self.dx, self.dy)
        
        # Interface properties
        interface_props = compute_interface_properties(self.c, self.dx, self.dy)
        
        # Chemical potential
        mu_chem = chemical_potential_double_well(self.c, self.A, self.B, self.C)
        
        # Free energy components
        grad_x, grad_y = compute_gradient_2d_periodic(self.c, self.dx, self.dy)
        bulk_energy = self.A * self.c**2 + self.B * self.c**3 + self.C * self.c**4
        grad_energy = 0.5 * self.kappa_dim * (grad_x**2 + grad_y**2)
        total_energy = np.mean(bulk_energy + grad_energy)
        
        diagnostics = {
            # Time and cycles
            'time': self.time,
            'step': self.step,
            'cycles': self.cycle_count,
            
            # Concentration statistics
            'mean_lithium': np.mean(self.c),
            'std_lithium': np.std(self.c),
            'lithium_deficit': 1.0 - np.mean(self.c),
            'min_lithium': np.min(self.c),
            'max_lithium': np.max(self.c),
            
            # Phase fractions
            'FePO4_fraction': np.sum(self.c < 0.5) / (self.nx * self.ny),
            'LiFePO4_fraction': np.sum(self.c >= 0.5) / (self.nx * self.ny),
            'mixed_phase_fraction': np.sum((self.c > 0.3) & (self.c < 0.7)) / (self.nx * self.ny),
            
            # Domain statistics
            'num_domains': domain_stats['num_domains'],
            'avg_domain_size_nm': np.sqrt(domain_stats['avg_area_nm2']) if domain_stats['avg_area_nm2'] > 0 else 0,
            'domain_size_std_nm': domain_stats['std_size'] * self.dx * 1e9,
            'domain_circularity': domain_stats['circularity'],
            
            # Interface properties
            'interface_density': interface_props['interface_density'],
            'avg_gradient_magnitude': interface_props['avg_gradient_magnitude'],
            'avg_curvature': interface_props['avg_curvature'],
            'interface_pixels': interface_props['interface_pixels'],
            
            # Energy statistics
            'bulk_energy_mean': np.mean(bulk_energy),
            'gradient_energy_mean': np.mean(grad_energy),
            'total_energy': total_energy,
            'chemical_potential_mean': np.mean(mu_chem),
            'chemical_potential_std': np.std(mu_chem),
            
            # Loss statistics
            'loss_rate': self.current_loss_rate,
            'total_lithium_lost': (self.scales.c_LiFePO4 - np.mean(self.c)) * self.nx * self.ny,
            
            # Simulation parameters
            'grid_size': f"{self.nx}x{self.ny}",
            'domain_size_nm': f"{self.Lx*1e9:.1f}x{self.Ly*1e9:.1f}",
            'time_step_phys': self.dt_phys,
            'c_rate': self.scales.c_rate,
            'boundary_condition': self.boundary_condition
        }
        
        return diagnostics
    
    def save_state(self, filename):
        """Save current state to file"""
        state = {
            'c': self.c,
            'time': self.time,
            'step': self.step,
            'cycle_count': self.cycle_count,
            'history_idx': self.history_idx,
            'history': {k: v[:self.history_idx] for k, v in self.history.items()},
            'parameters': {
                'nx': self.nx, 'ny': self.ny,
                'Lx': self.Lx, 'Ly': self.Ly,
                'dx': self.dx, 'dy': self.dy,
                'dt': self.dt,
                'W_dim': self.W_dim,
                'kappa_dim': self.kappa_dim,
                'M_dim': self.M_dim,
                'c_rate': self.scales.c_rate
            }
        }
        np.savez_compressed(filename, **state)
    
    def load_state(self, filename):
        """Load state from file"""
        data = np.load(filename, allow_pickle=True)
        
        self.c = data['c']
        self.time = data['time']
        self.step = data['step']
        self.cycle_count = data['cycle_count']
        self.history_idx = data['history_idx']
        
        # Load history
        for key in self.history:
            if key in data['history'].item():
                loaded_len = len(data['history'].item()[key])
                if loaded_len <= self.max_history:
                    self.history[key][:loaded_len] = data['history'].item()[key]
                else:
                    self.history[key] = data['history'].item()[key]
                    self.max_history = loaded_len
        
        # Update derived fields
        self.phase_mask = (self.c > 0.5).astype(np.int32)
        self._cache_valid = False

# =====================================================
# Enhanced PINN with Advanced Architecture
# =====================================================
class EnhancedPhaseFieldPINN(nn.Module):
    """Enhanced PINN with residual connections and adaptive activation"""
    
    def __init__(self, Lx, Ly, hidden_dims=[128, 128, 128, 128], 
                 activation='swish', residual_connections=True):
        super().__init__()
        self.Lx = Lx
        self.Ly = Ly
        self.residual_connections = residual_connections
        
        # Normalization
        self.x_scale = 2.0 / Lx if Lx > 0 else 1.0
        self.y_scale = 2.0 / Ly if Ly > 0 else 1.0
        
        # Activation function
        if activation == 'swish':
            self.activation = nn.SiLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'mish':
            self.activation = nn.Mish()
        else:
            self.activation = nn.Tanh()
        
        # Build network with optional residual connections
        layers = []
        input_dim = 2
        prev_dim = input_dim
        
        for i, h_dim in enumerate(hidden_dims):
            # Linear layer
            linear = nn.Linear(prev_dim, h_dim)
            
            # Weight initialization
            nn.init.xavier_uniform_(linear.weight, gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(linear.bias)
            
            layers.append(linear)
            layers.append(self.activation)
            
            # Add residual connection if enabled and dimensions match
            if self.residual_connections and i > 0 and prev_dim == h_dim:
                layers.append(nn.Linear(h_dim, h_dim))
                layers.append(self.activation)
            
            prev_dim = h_dim
        
        # Output layer
        output_layer = nn.Linear(prev_dim, 1)
        nn.init.xavier_uniform_(output_layer.weight, gain=0.5)
        nn.init.zeros_(output_layer.bias)
        
        layers.append(output_layer)
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
        
        # Physics parameters (trainable)
        self.W = nn.Parameter(torch.tensor(1.0))
        self.kappa = nn.Parameter(torch.tensor(1.0))
        self.M = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x, y):
        """Forward pass with coordinate normalization"""
        # Normalize coordinates to [-1, 1]
        x_norm = (x * self.x_scale) - 1.0
        y_norm = (y * self.y_scale) - 1.0
        
        inputs = torch.stack([x_norm, y_norm], dim=-1)
        return self.net(inputs).squeeze(-1)
    
    def compute_physics_loss(self, x, y, c_pred, dx, dy, 
                           include_gradients=True, include_laplacian=True):
        """Compute comprehensive physics loss"""
        # Enable gradient computation
        x.requires_grad_(True)
        y.requires_grad_(True)
        
        # First derivatives
        grad_c = torch.autograd.grad(
            c_pred, [x, y],
            grad_outputs=torch.ones_like(c_pred),
            create_graph=True,
            retain_graph=True
        )
        grad_c_x, grad_c_y = grad_c
        
        # Gradient energy loss
        grad_loss = torch.mean(grad_c_x**2 + grad_c_y**2) if include_gradients else 0.0
        
        if include_laplacian:
            # Second derivatives for Laplacian
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
            
            # Chemical potential
            f_prime = 2 * self.W * c_pred * (1 - c_pred) * (1 - 2 * c_pred)
            mu = f_prime - self.kappa * laplacian_c
            
            # Laplacian of chemical potential
            grad_mu_x = torch.autograd.grad(
                mu, x,
                grad_outputs=torch.ones_like(mu),
                create_graph=True,
                retain_graph=True
            )[0]
            grad_mu_y = torch.autograd.grad(
                mu, y,
                grad_outputs=torch.ones_like(mu),
                create_graph=True
            )[0]
            
            grad_mu_x_x = torch.autograd.grad(
                grad_mu_x, x,
                grad_outputs=torch.ones_like(grad_mu_x),
                create_graph=True
            )[0]
            grad_mu_y_y = torch.autograd.grad(
                grad_mu_y, y,
                grad_outputs=torch.ones_like(grad_mu_y),
                create_graph=True
            )[0]
            
            laplacian_mu = grad_mu_x_x + grad_mu_y_y
            
            # Cahn-Hilliard equilibrium loss
            ch_loss = torch.mean(laplacian_mu**2)
        else:
            ch_loss = 0.0
        
        # Regularization losses
        reg_loss = 0.001 * (self.W**2 + self.kappa**2 + self.M**2)
        
        # Bounds penalty
        bounds_loss = torch.mean(torch.relu(c_pred - 1.0)**2 + torch.relu(-c_pred)**2)
        
        total_physics_loss = ch_loss + 0.1 * grad_loss + reg_loss + 0.01 * bounds_loss
        
        physics_stats = {
            'ch_loss': ch_loss.item() if isinstance(ch_loss, torch.Tensor) else ch_loss,
            'grad_loss': grad_loss.item() if isinstance(grad_loss, torch.Tensor) else grad_loss,
            'reg_loss': reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss,
            'bounds_loss': bounds_loss.item() if isinstance(bounds_loss, torch.Tensor) else bounds_loss
        }
        
        return total_physics_loss, physics_stats

class EnhancedPINNTrainer:
    """Enhanced trainer with advanced optimization techniques"""
    
    def __init__(self, pinn, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.pinn = pinn.to(device)
        self.loss_history = []
        self.best_loss = float('inf')
        self.best_model_state = None
        
    def train(self, tem_observations, phase_field_params,
              n_epochs=1000, lr=1e-3, batch_size=4096,
              data_weight=1.0, physics_weight=0.1,
              scheduler_type='cosine', warmup_epochs=50):
        """Enhanced training with advanced features"""
        # Prepare training data
        train_data = self.prepare_training_data(tem_observations, phase_field_params)
        
        # Create dataloader
        dataset = TensorDataset(*train_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, 
                               shuffle=True, pin_memory=True, num_workers=2)
        
        # Optimizer with weight decay
        optimizer = optim.AdamW(self.pinn.parameters(), lr=lr, 
                               weight_decay=1e-4, betas=(0.9, 0.999))
        
        # Learning rate scheduler
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=0.5)
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
        
        # Warmup scheduler
        if warmup_epochs > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, 
                                                          start_factor=0.01,
                                                          total_iters=warmup_epochs)
            scheduler = optim.lr_scheduler.SequentialLR(optimizer, 
                                                       [warmup_scheduler, scheduler],
                                                       milestones=[warmup_epochs])
        
        # Training loop
        self.pinn.train()
        for epoch in range(n_epochs):
            epoch_data_loss = 0.0
            epoch_physics_loss = 0.0
            epoch_total_loss = 0.0
            
            for batch in dataloader:
                x_batch, y_batch, c_batch = [b.to(self.device) for b in batch]
                
                optimizer.zero_grad()
                
                # Forward pass
                c_pred = self.pinn(x_batch, y_batch)
                
                # Data loss
                data_loss = torch.mean((c_pred - c_batch)**2)
                
                # Physics loss
                physics_loss, physics_stats = self.pinn.compute_physics_loss(
                    x_batch, y_batch, c_pred,
                    phase_field_params['dx'],
                    phase_field_params['dy']
                )
                
                # Total loss
                total_loss = data_weight * data_loss + physics_weight * physics_loss
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.pinn.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Accumulate losses
                epoch_data_loss += data_loss.item() * len(x_batch)
                epoch_physics_loss += physics_loss.item() * len(x_batch)
                epoch_total_loss += total_loss.item() * len(x_batch)
            
            # Average losses
            n_samples = len(train_data[0])
            epoch_data_loss /= n_samples
            epoch_physics_loss /= n_samples
            epoch_total_loss /= n_samples
            
            # Update learning rate
            if scheduler_type == 'plateau':
                scheduler.step(epoch_total_loss)
            else:
                scheduler.step()
            
            # Store history
            self.loss_history.append({
                'epoch': epoch,
                'total_loss': epoch_total_loss,
                'data_loss': epoch_data_loss,
                'physics_loss': epoch_physics_loss,
                'learning_rate': optimizer.param_groups[0]['lr'],
                **physics_stats
            })
            
            # Save best model
            if epoch_total_loss < self.best_loss:
                self.best_loss = epoch_total_loss
                self.best_model_state = self.pinn.state_dict().copy()
            
            # Print progress
            if epoch % 100 == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}: Total Loss = {epoch_total_loss:.2e}, "
                      f"Data Loss = {epoch_data_loss:.2e}, "
                      f"Physics Loss = {epoch_physics_loss:.2e}, "
                      f"LR = {lr:.2e}")
        
        # Load best model
        if self.best_model_state is not None:
            self.pinn.load_state_dict(self.best_model_state)
        
        return {
            'final_loss': self.loss_history[-1]['total_loss'],
            'best_loss': self.best_loss,
            'loss_history': self.loss_history,
            'num_samples': n_samples,
            'model_parameters': sum(p.numel() for p in self.pinn.parameters())
        }
    
    def prepare_training_data(self, tem_observations, phase_field_params):
        """Prepare training data with data augmentation"""
        all_x, all_y, all_c = [], [], []
        
        for obs in tem_observations:
            nx, ny = obs['image_shape']
            
            # Create coordinate grid
            x_grid = np.linspace(0, phase_field_params['Lx'], nx)
            y_grid = np.linspace(0, phase_field_params['Ly'], ny)
            X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
            
            # Use estimated phase
            c_data = obs['estimated_phase']
            
            # Data augmentation: random rotations and flips
            for aug in range(4):  # 4 rotations
                if aug > 0:
                    X_rot = np.rot90(X, k=aug)
                    Y_rot = np.rot90(Y, k=aug)
                    c_rot = np.rot90(c_data, k=aug)
                else:
                    X_rot, Y_rot, c_rot = X, Y, c_data
                
                # Random flip
                if np.random.random() > 0.5:
                    X_rot = np.flipud(X_rot)
                    c_rot = np.flipud(c_rot)
                
                if np.random.random() > 0.5:
                    Y_rot = np.fliplr(Y_rot)
                    c_rot = np.fliplr(c_rot)
                
                all_x.append(X_rot.flatten())
                all_y.append(Y_rot.flatten())
                all_c.append(c_rot.flatten())
        
        # Combine and convert to tensors
        x_tensor = torch.tensor(np.concatenate(all_x), dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(np.concatenate(all_y), dtype=torch.float32, device=self.device)
        c_tensor = torch.tensor(np.concatenate(all_c), dtype=torch.float32, device=self.device)
        
        return x_tensor, y_tensor, c_tensor
    
    def reconstruct_field(self, nx, ny, Lx, Ly, batch_size=8192):
        """Reconstruct full field with memory-efficient batching"""
        self.pinn.eval()
        
        # Create evaluation grid
        x_grid = np.linspace(0, Lx, nx)
        y_grid = np.linspace(0, Ly, ny)
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        
        c_pred_parts = []
        
        with torch.no_grad():
            for i in range(0, len(X_flat), batch_size):
                X_batch = torch.tensor(X_flat[i:i+batch_size], 
                                      dtype=torch.float32, 
                                      device=self.device)
                Y_batch = torch.tensor(Y_flat[i:i+batch_size], 
                                      dtype=torch.float32, 
                                      device=self.device)
                
                c_pred = self.pinn(X_batch, Y_batch)
                c_pred_parts.append(c_pred.cpu().numpy())
        
        c_pred_full = np.concatenate(c_pred_parts).reshape(nx, ny)
        
        return c_pred_full

# =====================================================
# Enhanced Hybrid FDM-PINN Assimilation System
# =====================================================
class EnhancedHybridFDM_PINN_Assimilation:
    """Enhanced hybrid system with multiple assimilation strategies"""
    
    def __init__(self):
        self.phase_field = None
        self.tem_physics = EnhancedTEMPhysics()
        self.pinn = None
        self.trainer = None
        
        # Assimilation state
        self.assimilation_history = []
        self.tem_observations = []
        self.reconstruction_errors = []
        
        # Configuration
        self.config = {
            'assimilation_strategy': 'sequential',  # 'sequential', 'ensemble', 'adaptive'
            'update_frequency': 10,  # Steps between assimilations
            'pinn_update_frequency': 5,  # Assimilation cycles between PINN updates
            'use_transfer_learning': True,
            'ensemble_size': 3,
            'adaptive_weighting': True
        }
        
        # Performance tracking
        self.performance_stats = {
            'total_assimilation_time': 0.0,
            'avg_assimilation_time': 0.0,
            'num_assimilations': 0,
            'reconstruction_mse_history': [],
            'pinn_training_history': []
        }
    
    def initialize_simulation(self, nx=256, ny=256, 
                            Lx=200e-9, Ly=200e-9, dt=0.01, 
                            c_rate=1.0, boundary_condition='periodic',
                            initialization_pattern='random'):
        """Initialize enhanced simulation"""
        self.phase_field = EnhancedLithiumLossPhaseField(
            nx=nx, ny=ny, Lx=Lx, Ly=Ly, dt=dt, c_rate=c_rate,
            boundary_condition=boundary_condition
        )
        
        # Apply initialization pattern
        self.phase_field.initialize_lifepo4(pattern=initialization_pattern)
        
        # Reset assimilation state
        self.assimilation_history = []
        self.tem_observations = []
        self.reconstruction_errors = []
        
        print(f"Enhanced simulation initialized:")
        print(f"  Grid: {nx}x{ny}")
        print(f"  Domain: {Lx*1e9:.1f}x{Ly*1e9:.1f} nm")
        print(f"  Boundary condition: {boundary_condition}")
        print(f"  C-rate: {c_rate}")
    
    def collect_tem_observations_batch(self, observation_times, 
                                      noise_level=0.05, include_hr=True,
                                      parallel=False):
        """Collect TEM observations with optional parallel processing"""
        observations = []
        
        if parallel and len(observation_times) > 1:
            # Parallel processing for multiple observations
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for t_obs in observation_times:
                    future = executor.submit(
                        self._generate_single_observation,
                        t_obs, noise_level, include_hr
                    )
                    futures.append(future)
                
                for future in concurrent.futures.as_completed(futures):
                    observations.append(future.result())
        else:
            # Sequential processing
            for t_obs in observation_times:
                obs = self._generate_single_observation(t_obs, noise_level, include_hr)
                observations.append(obs)
        
        self.tem_observations.extend(observations)
        return observations
    
    def _generate_single_observation(self, observation_time, noise_level, include_hr):
        """Generate single TEM observation"""
        # Run simulation to observation time
        if self.phase_field.time < observation_time:
            steps_needed = int((observation_time - self.phase_field.time) / self.phase_field.dt_phys)
            for _ in range(steps_needed):
                self.phase_field.run_cycle_step()
        
        # Generate TEM image
        tem_image, components = self.tem_physics.simulate_tem_image(
            self.phase_field.c,
            noise_level=noise_level,
            include_phase_contrast=include_hr
        )
        
        # Extract phase information
        estimated_phase, confidence = self.tem_physics.extract_phase_information(
            tem_image, method='ml_enhanced'
        )
        
        # Create observation
        observation = {
            'time': observation_time,
            'cycles': self.phase_field.cycle_count,
            'tem_image': tem_image,
            'true_concentration': self.phase_field.c.copy(),
            'estimated_phase': estimated_phase,
            'confidence_map': confidence,
            'tem_components': components,
            'diagnostics': self.phase_field.get_comprehensive_diagnostics(),
            'noise_level': noise_level,
            'image_shape': tem_image.shape
        }
        
        return observation
    
    def run_assimilation_cycle_enhanced(self, observation_time, 
                                      pinn_params=None,
                                      training_params=None):
        """Run enhanced assimilation cycle"""
        perf_mon.start_timer('assimilation_cycle')
        
        # Default parameters
        if pinn_params is None:
            pinn_params = {
                'hidden_dims': [128, 128, 128, 128],
                'activation': 'swish',
                'residual_connections': True
            }
        
        if training_params is None:
            training_params = {
                'n_epochs': 500,
                'lr': 1e-3,
                'batch_size': 4096,
                'data_weight': 1.0,
                'physics_weight': 0.1,
                'scheduler_type': 'cosine',
                'warmup_epochs': 50
            }
        
        # 1. Generate TEM observation
        tem_obs = self._generate_single_observation(observation_time, 0.05, True)
        self.tem_observations.append(tem_obs)
        
        # 2. Initialize or update PINN
        if self.pinn is None:
            self._initialize_pinn(pinn_params)
        
        # 3. Prepare phase field parameters
        phase_field_params = {
            'dx': self.phase_field.dx,
            'dy': self.phase_field.dy,
            'Lx': self.phase_field.Lx,
            'Ly': self.phase_field.Ly,
            'W': float(self.phase_field.W_dim),
            'kappa': float(self.phase_field.kappa_dim),
            'M': float(self.phase_field.M_dim)
        }
        
        # 4. Train PINN
        training_stats = self.trainer.train(
            [tem_obs],
            phase_field_params,
            **training_params
        )
        
        # 5. Reconstruct field
        reconstructed_field = self.trainer.reconstruct_field(
            self.phase_field.nx,
            self.phase_field.ny,
            self.phase_field.Lx,
            self.phase_field.Ly
        )
        
        # 6. Calculate metrics
        metrics = self._calculate_assimilation_metrics(
            reconstructed_field, 
            self.phase_field.c,
            tem_obs['estimated_phase']
        )
        
        # 7. Update phase field (data assimilation)
        if self.config['adaptive_weighting']:
            alpha = self._calculate_assimilation_weight(metrics['mse'])
            self._update_phase_field_adaptive(reconstructed_field, alpha)
        else:
            self._update_phase_field_simple(reconstructed_field)
        
        # 8. Store results
        cycle_result = self._create_assimilation_result(
            observation_time, tem_obs, reconstructed_field, 
            metrics, training_stats, pinn_params, training_params
        )
        
        self.assimilation_history.append(cycle_result)
        self.reconstruction_errors.append(metrics['mse'])
        
        # Update performance statistics
        self._update_performance_stats(perf_mon.stop_timer('assimilation_cycle'))
        
        return cycle_result
    
    def _initialize_pinn(self, pinn_params):
        """Initialize PINN with transfer learning if available"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.pinn = EnhancedPhaseFieldPINN(
            Lx=self.phase_field.Lx,
            Ly=self.phase_field.Ly,
            **pinn_params
        )
        
        # Apply transfer learning if previous model exists
        if hasattr(self, 'previous_pinn_state') and self.config['use_transfer_learning']:
            self.pinn.load_state_dict(self.previous_pinn_state)
            print("Applied transfer learning from previous model")
        
        self.trainer = EnhancedPINNTrainer(self.pinn, device=device)
    
    def _calculate_assimilation_metrics(self, reconstructed, true, estimated):
        """Calculate comprehensive assimilation metrics"""
        # Basic MSE
        mse = np.mean((reconstructed - true)**2)
        
        # Structural similarity
        from skimage.metrics import structural_similarity as ssim
        ssim_val = ssim(reconstructed, true, data_range=1.0)
        
        # Correlation coefficient
        correlation = np.corrcoef(reconstructed.flatten(), true.flatten())[0, 1]
        
        # Phase boundary accuracy
        grad_rec = np.gradient(reconstructed)
        grad_true = np.gradient(true)
        grad_mag_rec = np.sqrt(grad_rec[0]**2 + grad_rec[1]**2)
        grad_mag_true = np.sqrt(grad_true[0]**2 + grad_true[1]**2)
        
        # Threshold gradients to get boundaries
        boundary_rec = (grad_mag_rec > np.percentile(grad_mag_rec, 90))
        boundary_true = (grad_mag_true > np.percentile(grad_mag_true, 90))
        
        # Boundary accuracy
        boundary_accuracy = np.sum(boundary_rec & boundary_true) / np.sum(boundary_true)
        
        # Domain statistics comparison
        stats_rec = compute_domain_statistics(reconstructed, 0.5, 
                                            self.phase_field.dx, self.phase_field.dy)
        stats_true = compute_domain_statistics(true, 0.5, 
                                             self.phase_field.dx, self.phase_field.dy)
        
        domain_size_error = abs(stats_rec['avg_area_nm2'] - stats_true['avg_area_nm2']) / stats_true['avg_area_nm2'] if stats_true['avg_area_nm2'] > 0 else 0
        
        metrics = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'ssim': ssim_val,
            'correlation': correlation,
            'boundary_accuracy': boundary_accuracy,
            'domain_size_error': domain_size_error,
            'num_domains_rec': stats_rec['num_domains'],
            'num_domains_true': stats_true['num_domains'],
            'domain_count_error': abs(stats_rec['num_domains'] - stats_true['num_domains']) / max(1, stats_true['num_domains'])
        }
        
        return metrics
    
    def _calculate_assimilation_weight(self, mse):
        """Calculate adaptive assimilation weight based on reconstruction quality"""
        # Lower MSE -> higher weight for reconstruction
        alpha_max = 0.3  # Maximum assimilation strength
        alpha_min = 0.05  # Minimum assimilation strength
        mse_threshold = 0.01  # MSE above this gets minimal weight
        
        if mse < 1e-6:
            return alpha_max
        elif mse > mse_threshold:
            return alpha_min
        else:
            # Sigmoid decay
            decay_rate = 50.0
            weight = alpha_max * (1.0 - 1.0 / (1.0 + np.exp(-decay_rate * (mse_threshold/2 - mse))))
            return max(alpha_min, min(alpha_max, weight))
    
    def _update_phase_field_adaptive(self, reconstructed_field, alpha):
        """Update phase field with adaptive weighting"""
        # Blend original and reconstructed fields
        self.phase_field.c = (1.0 - alpha) * self.phase_field.c + alpha * reconstructed_field
        
        # Ensure bounds
        np.clip(self.phase_field.c, 0.0, 1.0, out=self.phase_field.c)
        
        # Update derived fields
        self.phase_field.phase_mask = (self.phase_field.c > 0.5).astype(np.int32)
        self.phase_field._cache_valid = False
    
    def _update_phase_field_simple(self, reconstructed_field, alpha=0.1):
        """Simple update with fixed weight"""
        self._update_phase_field_adaptive(reconstructed_field, alpha)
    
    def _create_assimilation_result(self, time, tem_obs, reconstructed_field,
                                   metrics, training_stats, pinn_params, training_params):
        """Create comprehensive assimilation result"""
        return {
            'time': time,
            'cycles': self.phase_field.cycle_count,
            'tem_observation': tem_obs,
            'true_field': self.phase_field.c.copy(),
            'reconstructed_field': reconstructed_field,
            'assimilation_metrics': metrics,
            'training_stats': training_stats,
            'pinn_params': pinn_params,
            'training_params': training_params,
            'phase_field_diagnostics': self.phase_field.get_comprehensive_diagnostics(),
            'assimilation_weight': self._calculate_assimilation_weight(metrics['mse'])
        }
    
    def _update_performance_stats(self, assimilation_time):
        """Update performance statistics"""
        self.performance_stats['total_assimilation_time'] += assimilation_time
        self.performance_stats['num_assimilations'] += 1
        self.performance_stats['avg_assimilation_time'] = (
            self.performance_stats['total_assimilation_time'] / 
            self.performance_stats['num_assimilations']
        )
    
    def run_sequential_assimilation(self, observation_schedule, **kwargs):
        """Run sequential assimilation"""
        results = []
        
        for i, t_obs in enumerate(observation_schedule):
            print(f"Assimilation cycle {i+1}/{len(observation_schedule)} at t = {t_obs:.0f}s")
            
            result = self.run_assimilation_cycle_enhanced(t_obs, **kwargs)
            results.append(result)
            
            # Save intermediate state
            if (i + 1) % 5 == 0:
                self.save_state(f"assimilation_checkpoint_{i+1}.npz")
        
        return results
    
    def get_assimilation_statistics(self):
        """Get comprehensive assimilation statistics"""
        if not self.assimilation_history:
            return {}
        
        errors = [cycle['assimilation_metrics']['mse'] for cycle in self.assimilation_history]
        times = [cycle['time'] for cycle in self.assimilation_history]
        
        # Training statistics
        training_epochs = []
        training_losses = []
        for cycle in self.assimilation_history:
            if 'training_stats' in cycle and 'loss_history' in cycle['training_stats']:
                training_epochs.append(len(cycle['training_stats']['loss_history']))
                training_losses.append(cycle['training_stats']['loss_history'][-1]['total_loss'])
        
        stats = {
            'num_assimilations': len(self.assimilation_history),
            'time_range': [min(times), max(times)],
            'cycle_range': [self.assimilation_history[0]['cycles'], 
                           self.assimilation_history[-1]['cycles']],
            
            # Error statistics
            'mse_mean': np.mean(errors),
            'mse_std': np.std(errors),
            'mse_min': np.min(errors),
            'mse_max': np.max(errors),
            'mse_trend': self._calculate_trend(errors),
            
            # Training statistics
            'avg_training_epochs': np.mean(training_epochs) if training_epochs else 0,
            'avg_training_loss': np.mean(training_losses) if training_losses else 0,
            
            # Performance statistics
            **self.performance_stats,
            
            # Phase evolution
            'lithium_deficit_trajectory': [
                cycle['phase_field_diagnostics']['lithium_deficit']
                for cycle in self.assimilation_history
            ],
            'domain_evolution': [
                cycle['phase_field_diagnostics']['avg_domain_size_nm']
                for cycle in self.assimilation_history
            ],
            'interface_density_evolution': [
                cycle['phase_field_diagnostics']['interface_density']
                for cycle in self.assimilation_history
            ]
        }
        
        return stats
    
    def _calculate_trend(self, values, window=5):
        """Calculate trend of values using moving average"""
        if len(values) < window:
            return 0.0
        
        # Simple linear regression on moving average
        ma = np.convolve(values, np.ones(window)/window, mode='valid')
        x = np.arange(len(ma))
        slope = np.polyfit(x, ma, 1)[0]
        
        return slope
    
    def save_state(self, filename):
        """Save current state including PINN model"""
        state = {
            'phase_field_state': {
                'c': self.phase_field.c,
                'time': self.phase_field.time,
                'step': self.phase_field.step,
                'cycle_count': self.phase_field.cycle_count,
                'history_idx': self.phase_field.history_idx,
                'history': {k: v[:self.phase_field.history_idx] 
                          for k, v in self.phase_field.history.items()}
            },
            'assimilation_history': self.assimilation_history,
            'tem_observations': self.tem_observations,
            'reconstruction_errors': self.reconstruction_errors,
            'performance_stats': self.performance_stats,
            'config': self.config
        }
        
        # Save PINN model if available
        if self.pinn is not None:
            state['pinn_state_dict'] = self.pinn.state_dict()
        
        np.savez_compressed(filename, **state)
        print(f"State saved to {filename}")
    
    def load_state(self, filename):
        """Load state from file"""
        data = np.load(filename, allow_pickle=True)
        
        # Load phase field state
        if 'phase_field_state' in data:
            pf_state = data['phase_field_state'].item()
            self.phase_field.c = pf_state['c']
            self.phase_field.time = pf_state['time']
            self.phase_field.step = pf_state['step']
            self.phase_field.cycle_count = pf_state['cycle_count']
            self.phase_field.history_idx = pf_state['history_idx']
            
            # Load history
            for key in self.phase_field.history:
                if key in pf_state['history']:
                    loaded_len = len(pf_state['history'][key])
                    if loaded_len <= self.phase_field.max_history:
                        self.phase_field.history[key][:loaded_len] = pf_state['history'][key]
                    else:
                        self.phase_field.history[key] = pf_state['history'][key]
                        self.phase_field.max_history = loaded_len
            
            # Update derived fields
            self.phase_field.phase_mask = (self.phase_field.c > 0.5).astype(np.int32)
            self.phase_field._cache_valid = False
        
        # Load assimilation data
        if 'assimilation_history' in data:
            self.assimilation_history = data['assimilation_history'].tolist()
        
        if 'tem_observations' in data:
            self.tem_observations = data['tem_observations'].tolist()
        
        if 'reconstruction_errors' in data:
            self.reconstruction_errors = data['reconstruction_errors'].tolist()
        
        if 'performance_stats' in data:
            self.performance_stats = data['performance_stats'].item()
        
        if 'config' in data:
            self.config.update(data['config'].item())
        
        # Load PINN model if available
        if 'pinn_state_dict' in data and self.pinn is not None:
            self.pinn.load_state_dict(data['pinn_state_dict'].item())
            print("Loaded PINN model from saved state")
        
        print(f"State loaded from {filename}")

# =====================================================
# Enhanced Visualization Functions
# =====================================================
def create_comprehensive_visualization(cycle_result):
    """Create comprehensive visualization of assimilation results"""
    # Create subplots with enhanced layout
    fig = make_subplots(
        rows=3, cols=4,
        subplot_titles=(
            'True Lithium Concentration',
            'PINN Reconstruction',
            'Reconstruction Error',
            'Error Distribution',
            'TEM Image',
            'Estimated Phase (TEM)',
            'Confidence Map',
            'Lithium Deficit Analysis',
            'Interface Properties',
            'Domain Statistics',
            'Training Loss',
            'Assimilation Metrics'
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
        specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}, 
                {'type': 'heatmap'}, {'type': 'histogram'}],
               [{'type': 'heatmap'}, {'type': 'heatmap'}, 
                {'type': 'heatmap'}, {'type': 'scatter'}],
               [{'type': 'heatmap'}, {'type': 'bar'}, 
                {'type': 'scatter'}, {'type': 'table'}]]
    )
    
    # 1. True concentration field
    fig.add_trace(
        go.Heatmap(
            z=cycle_result['true_field'].T,
            colorscale='RdYlBu',
            zmin=0, zmax=1,
            colorbar=dict(x=0.23, y=0.95, len=0.25, title='x in LiₓFePO₄'),
            showscale=True
        ),
        row=1, col=1
    )
    
    # 2. PINN reconstruction
    fig.add_trace(
        go.Heatmap(
            z=cycle_result['reconstructed_field'].T,
            colorscale='RdYlBu',
            zmin=0, zmax=1,
            colorbar=dict(x=0.49, y=0.95, len=0.25, title='x in LiₓFePO₄'),
            showscale=True
        ),
        row=1, col=2
    )
    
    # 3. Reconstruction error
    error = cycle_result['reconstructed_field'] - cycle_result['true_field']
    vmax = max(abs(error.min()), abs(error.max()))
    fig.add_trace(
        go.Heatmap(
            z=error.T,
            colorscale='RdBu_r',
            zmin=-vmax, zmax=vmax,
            colorbar=dict(x=0.75, y=0.95, len=0.25, title='Error'),
            showscale=True
        ),
        row=1, col=3
    )
    
    # 4. Error distribution histogram
    fig.add_trace(
        go.Histogram(
            x=error.flatten(),
            nbinsx=50,
            marker_color='blue',
            opacity=0.7,
            name='Error distribution'
        ),
        row=1, col=4
    )
    
    # 5. TEM image
    tem_obs = cycle_result['tem_observation']
    fig.add_trace(
        go.Heatmap(
            z=tem_obs['tem_image'].T,
            colorscale='gray',
            showscale=False
        ),
        row=2, col=1
    )
    
    # 6. Estimated phase from TEM
    fig.add_trace(
        go.Heatmap(
            z=tem_obs['estimated_phase'].T,
            colorscale='RdYlBu',
            zmin=0, zmax=1,
            showscale=False
        ),
        row=2, col=2
    )
    
    # 7. Confidence map
    fig.add_trace(
        go.Heatmap(
            z=tem_obs['confidence_map'].T,
            colorscale='RdYlGn',
            zmin=0, zmax=1,
            showscale=False
        ),
        row=2, col=3
    )
    
    # 8. Lithium deficit analysis
    diagnostics = cycle_result['phase_field_diagnostics']
    fig.add_trace(
        go.Scatter(
            x=[diagnostics['FePO4_fraction']],
            y=[diagnostics['lithium_deficit']],
            mode='markers',
            marker=dict(
                size=20,
                color=diagnostics['avg_domain_size_nm'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Domain Size (nm)', x=0.95, y=0.65, len=0.25)
            ),
            text=f"Domains: {diagnostics['num_domains']}<br>"
                 f"Size: {diagnostics['avg_domain_size_nm']:.1f} nm<br>"
                 f"Interface: {diagnostics['interface_density']:.3f}<br>"
                 f"Circularity: {diagnostics.get('domain_circularity', 0):.2f}",
            hoverinfo='text'
        ),
        row=2, col=4
    )
    
    # 9. Interface properties
    interface_mask = compute_interface_properties(
        cycle_result['true_field'], 1.0, 1.0
    )['interface_mask']
    fig.add_trace(
        go.Heatmap(
            z=interface_mask.T,
            colorscale='viridis',
            showscale=False
        ),
        row=3, col=1
    )
    
    # 10. Domain statistics bar chart
    domain_stats = compute_domain_statistics(
        cycle_result['true_field'], 0.5, 1.0, 1.0
    )
    fig.add_trace(
        go.Bar(
            x=['Number', 'Avg Size', 'Circularity'],
            y=[domain_stats['num_domains'], 
               np.sqrt(domain_stats['avg_area_nm2']) if domain_stats['avg_area_nm2'] > 0 else 0,
               domain_stats['circularity']],
            marker_color=['blue', 'green', 'red'],
            name='Domain Statistics'
        ),
        row=3, col=2
    )
    
    # 11. Training loss history
    if 'training_stats' in cycle_result and 'loss_history' in cycle_result['training_stats']:
        loss_history = cycle_result['training_stats']['loss_history']
        epochs = [lh['epoch'] for lh in loss_history]
        total_loss = [lh['total_loss'] for lh in loss_history]
        
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=total_loss,
                mode='lines',
                line=dict(color='purple', width=2),
                name='Training Loss'
            ),
            row=3, col=3
        )
    
    # 12. Assimilation metrics table
    metrics = cycle_result.get('assimilation_metrics', {})
    metrics_data = [
        ['Metric', 'Value'],
        ['MSE', f"{metrics.get('mse', 0):.2e}"],
        ['RMSE', f"{metrics.get('rmse', 0):.3f}"],
        ['SSIM', f"{metrics.get('ssim', 0):.3f}"],
        ['Correlation', f"{metrics.get('correlation', 0):.3f}"],
        ['Boundary Accuracy', f"{metrics.get('boundary_accuracy', 0):.3f}"]
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(values=metrics_data[0],
                       fill_color='paleturquoise',
                       align='left'),
            cells=dict(values=[row[0] for row in metrics_data[1:]] + [row[1] for row in metrics_data[1:]],
                      fill_color='lavender',
                      align='left')
        ),
        row=3, col=4
    )
    
    # Update layout
    fig.update_layout(
        height=1200,
        title_text=f"Enhanced Assimilation Analysis at t = {cycle_result['time']:.0f}s, "
                   f"Cycle {cycle_result['cycles']}",
        title_x=0.5,
        showlegend=False
    )
    
    # Update axes
    for i in range(1, 4):
        for j in range(1, 4):
            fig.update_xaxes(title_text="x (nm)", row=i, col=j)
            fig.update_yaxes(title_text="y (nm)", row=i, col=j)
    
    return fig

def create_performance_dashboard(hybrid_system):
    """Create performance dashboard"""
    stats = hybrid_system.get_assimilation_statistics()
    perf_stats = perf_mon.get_performance_stats()
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Reconstruction Error Evolution',
            'Training Loss Evolution',
            'Phase Evolution',
            'Performance Metrics',
            'Domain Size Evolution',
            'Interface Evolution'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # 1. Reconstruction error evolution
    if hybrid_system.reconstruction_errors:
        times = [cycle['time'] for cycle in hybrid_system.assimilation_history]
        fig.add_trace(
            go.Scatter(
                x=times,
                y=hybrid_system.reconstruction_errors,
                mode='lines+markers',
                line=dict(color='red', width=2),
                marker=dict(size=6),
                name='Reconstruction MSE'
            ),
            row=1, col=1
        )
        fig.update_yaxes(title_text="MSE", type="log", row=1, col=1)
    
    # 2. Training loss evolution
    training_losses = []
    for cycle in hybrid_system.assimilation_history:
        if 'training_stats' in cycle and 'loss_history' in cycle['training_stats']:
            training_losses.append(cycle['training_stats']['loss_history'][-1]['total_loss'])
    
    if training_losses:
        fig.add_trace(
            go.Scatter(
                x=list(range(len(training_losses))),
                y=training_losses,
                mode='lines+markers',
                line=dict(color='blue', width=2),
                marker=dict(size=6),
                name='Final Training Loss'
            ),
            row=1, col=2
        )
        fig.update_yaxes(title_text="Loss", type="log", row=1, col=2)
    
    # 3. Phase evolution
    if hybrid_system.assimilation_history:
        times = [cycle['time'] for cycle in hybrid_system.assimilation_history]
        lithium_deficit = [cycle['phase_field_diagnostics']['lithium_deficit'] 
                          for cycle in hybrid_system.assimilation_history]
        FePO4_fraction = [cycle['phase_field_diagnostics']['FePO4_fraction'] 
                         for cycle in hybrid_system.assimilation_history]
        
        fig.add_trace(
            go.Scatter(
                x=times,
                y=lithium_deficit,
                mode='lines',
                line=dict(color='red', width=2),
                name='Li deficit (x)'
            ),
            row=1, col=3
        )
        
        fig.add_trace(
            go.Scatter(
                x=times,
                y=FePO4_fraction,
                mode='lines',
                line=dict(color='blue', width=2),
                name='FePO₄ fraction',
                yaxis='y2'
            ),
            row=1, col=3
        )
        
        fig.update_layout(
            yaxis2=dict(
                title="FePO₄ fraction",
                overlaying='y',
                side='right'
            )
        )
    
    # 4. Performance metrics table
    if perf_stats:
        perf_data = [
            ['Operation', 'Avg Time (s)', 'Count'],
            *[[op, f"{perf_stats['summary'].loc[op, ('time', 'mean')]:.4f}", 
               f"{int(perf_stats['summary'].loc[op, ('time', 'count')])}"]
              for op in perf_stats['summary'].index[:5]]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=perf_data[0],
                           fill_color='paleturquoise',
                           align='left'),
                cells=dict(values=[row[0] for row in perf_data[1:]] + 
                          [row[1] for row in perf_data[1:]] + 
                          [row[2] for row in perf_data[1:]],
                          fill_color='lavender',
                          align='left')
            ),
            row=2, col=1
        )
    
    # 5. Domain size evolution
    if hybrid_system.assimilation_history:
        times = [cycle['time'] for cycle in hybrid_system.assimilation_history]
        domain_sizes = [cycle['phase_field_diagnostics']['avg_domain_size_nm'] 
                       for cycle in hybrid_system.assimilation_history]
        
        fig.add_trace(
            go.Scatter(
                x=times,
                y=domain_sizes,
                mode='lines+markers',
                line=dict(color='green', width=2),
                marker=dict(size=6),
                name='Avg Domain Size'
            ),
            row=2, col=2
        )
        fig.update_yaxes(title_text="Domain Size (nm)", row=2, col=2)
    
    # 6. Interface evolution
    if hybrid_system.assimilation_history:
        times = [cycle['time'] for cycle in hybrid_system.assimilation_history]
        interface_density = [cycle['phase_field_diagnostics']['interface_density'] 
                           for cycle in hybrid_system.assimilation_history]
        
        fig.add_trace(
            go.Scatter(
                x=times,
                y=interface_density,
                mode='lines+markers',
                line=dict(color='purple', width=2),
                marker=dict(size=6),
                name='Interface Density'
            ),
            row=2, col=3
        )
        fig.update_yaxes(title_text="Interface Density", row=2, col=3)
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Performance Dashboard",
        title_x=0.5,
        showlegend=True
    )
    
    # Update axes
    for i in range(1, 3):
        for j in range(1, 4):
            if not (i == 2 and j == 1):  # Skip table subplot
                fig.update_xaxes(title_text="Time (s)", row=i, col=j)
    
    return fig

# =====================================================
# Main Streamlit Application
# =====================================================
def main():
    """Enhanced Streamlit application"""
    st.set_page_config(
        page_title="Enhanced LiFePO₄ Phase Decomposition with TEM & PINN Assimilation",
        page_icon="🔋",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
        padding: 10px;
        border-radius: 10px;
        background-color: rgba(59, 130, 246, 0.1);
    }
    .section-header {
        font-size: 1.8rem;
        color: #283593;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #c5cae9;
        font-weight: 600;
    }
    .card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        color: white;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #3B82F6;
    }
    .stButton button {
        background: linear-gradient(90deg, #3B82F6, #1E3A8A);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.markdown('<h1 class="main-header">🔋 Enhanced LiFePO₄ Phase Decomposition with TEM & PINN Assimilation</h1>', unsafe_allow_html=True)
    st.markdown("""
    *Advanced phase field modeling with Numba-optimized FDM, realistic TEM characterization, 
    and enhanced Physics-Informed Neural Networks for high-performance data assimilation.*
    """)
    
    # Initialize session state
    if 'hybrid_system' not in st.session_state:
        st.session_state.hybrid_system = EnhancedHybridFDM_PINN_Assimilation()
    
    if 'sim_initialized' not in st.session_state:
        st.session_state.sim_initialized = False
    
    if 'assimilation_results' not in st.session_state:
        st.session_state.assimilation_results = []
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/electron-microscope.png", width=80)
        st.markdown("### 🎛️ Enhanced Control Panel")
        
        # Performance info
        device = 'CUDA 🚀' if torch.cuda.is_available() else 'CPU ⚡'
        st.markdown(f"**Compute Device:** {device}")
        st.markdown("**Numba JIT:** Enabled ✅")
        st.markdown("**Parallel Processing:** Available ✅")
        
        st.divider()
        
        # Simulation setup
        with st.expander("⚙️ Enhanced Simulation Setup", expanded=True):
            # Grid configuration
            col1, col2 = st.columns(2)
            with col1:
                nx = st.selectbox("Grid X", [64, 128, 256, 512], index=2)
            with col2:
                ny = st.selectbox("Grid Y", [64, 128, 256, 512], index=2)
            
            domain_nm = st.slider("Domain Size (nm)", 50, 1000, 200, 50)
            Lx = Ly = domain_nm * 1e-9
            
            # Advanced parameters
            c_rate = st.slider("C-Rate", 0.1, 20.0, 1.0, 0.1)
            dt = st.slider("Time Step (Δt)", 0.0001, 0.1, 0.01, 0.0001)
            
            boundary_condition = st.selectbox(
                "Boundary Condition",
                ["Periodic", "Neumann (Zero Flux)"],
                index=0
            )
            
            init_pattern = st.selectbox(
                "Initialization Pattern",
                ["Random", "Homogeneous", "Gradient", "Circular Seed"],
                index=0
            )
            
            init_lithium = st.slider("Initial Lithium Content", 0.8, 1.0, 0.97, 0.01)
            
            if st.button("🚀 Initialize Enhanced Simulation", use_container_width=True):
                with st.spinner("Initializing enhanced simulation..."):
                    st.session_state.hybrid_system.initialize_simulation(
                        nx=nx, ny=ny, Lx=Lx, Ly=Ly, dt=dt, c_rate=c_rate,
                        boundary_condition=boundary_condition.lower().split()[0],
                        initialization_pattern=init_pattern.lower().replace(' ', '_')
                    )
                    st.session_state.hybrid_system.phase_field.scales.c_LiFePO4 = init_lithium
                    st.session_state.sim_initialized = True
                    st.session_state.assimilation_results = []
                    st.success("✅ Enhanced simulation initialized!")
        
        # TEM observation settings
        with st.expander("🔬 Enhanced TEM Characterization", expanded=True):
            tem_noise = st.slider("TEM Noise Level", 0.0, 0.2, 0.05, 0.01)
            include_hr = st.checkbox("Include Phase Contrast (HRTEM)", True)
            acceleration_kv = st.slider("Acceleration Voltage (kV)", 80, 300, 200, 20)
            
            observation_interval = st.slider("Observation Interval (s)", 100, 10000, 1000, 100)
            num_observations = st.slider("Number of Observations", 3, 50, 10)
            
            parallel_processing = st.checkbox("Enable Parallel Processing", True)
        
        # PINN assimilation settings
        with st.expander("🧠 Enhanced PINN Assimilation", expanded=True):
            pinn_architecture = st.selectbox(
                "PINN Architecture",
                ["Standard (4x128)", "Deep (6x256)", "Wide (4x256)", "Custom"],
                index=0
            )
            
            if pinn_architecture == "Custom":
                pinn_layers = st.text_input("Custom Layers", "128,128,128,128,128")
                try:
                    hidden_dims = [int(x.strip()) for x in pinn_layers.split(",")]
                except:
                    hidden_dims = [128, 128, 128, 128]
            else:
                arch_map = {
                    "Standard (4x128)": [128, 128, 128, 128],
                    "Deep (6x256)": [256, 256, 256, 256, 256, 256],
                    "Wide (4x256)": [256, 256, 256, 256]
                }
                hidden_dims = arch_map[pinn_architecture]
            
            activation = st.selectbox(
                "Activation Function",
                ["Swish", "GELU", "Mish", "Tanh"],
                index=0
            )
            
            n_epochs = st.slider("Training Epochs", 100, 5000, 1000, 100)
            batch_size = st.slider("Batch Size", 512, 8192, 4096, 512)
            
            col_w1, col_w2 = st.columns(2)
            with col_w1:
                data_weight = st.slider("Data Weight", 0.1, 10.0, 1.0, 0.1)
            with col_w2:
                physics_weight = st.slider("Physics Weight", 0.01, 2.0, 0.1, 0.01)
        
        # Run controls
        st.divider()
        col_run1, col_run2 = st.columns(2)
        with col_run1:
            run_single = st.button("📸 Single Assimilation", use_container_width=True)
        with col_run2:
            run_sequential = st.button("🔄 Sequential Assimilation", use_container_width=True, type="primary")
        
        st.divider()
        
        # Save/Load controls
        col_save, col_load = st.columns(2)
        with col_save:
            if st.button("💾 Save State", use_container_width=True):
                filename = f"simulation_state_{int(time.time())}.npz"
                st.session_state.hybrid_system.save_state(filename)
                st.success(f"State saved to {filename}")
        
        with col_load:
            uploaded_file = st.file_uploader("Load State", type=['npz'])
            if uploaded_file is not None:
                with open("temp_state.npz", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.hybrid_system.load_state("temp_state.npz")
                st.success("State loaded successfully!")
                st.rerun()
    
    # Main content
    if not st.session_state.sim_initialized:
        # Welcome screen
        col_welcome1, col_welcome2 = st.columns([2, 1])
        
        with col_welcome1:
            st.markdown("""
            <div class="card">
            <h3>🚀 Welcome to Enhanced Phase Decomposition Analysis</h3>
            <p>This enhanced system provides:</p>
            <ul>
            <li><strong>Numba-optimized FDM</strong>: 10-100x faster phase field simulations</li>
            <li><strong>Realistic TEM Physics</strong>: Mass-thickness, diffraction, and phase contrast</li>
            <li><strong>Enhanced PINNs</strong>: Residual connections, adaptive activation, transfer learning</li>
            <li><strong>Multiple Assimilation Strategies</strong>: Sequential, ensemble, adaptive weighting</li>
            <li><strong>Comprehensive Analysis</strong>: Domain statistics, interface properties, energy tracking</li>
            <li><strong>Performance Monitoring</strong>: Real-time performance tracking and optimization</li>
            </ul>
            <p><strong>Experimental Basis:</strong> Based on neutron diffraction/TEM studies showing 5-10 nm FePO₄ domains in Li₁₋ₓFePO₄ during cycling.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_welcome2:
            st.markdown("""
            <div class="card">
            <h4>⚡ Performance Features</h4>
            <ul>
            <li>GPU acceleration for PINN training</li>
            <li>Parallel TEM image generation</li>
            <li>Memory-efficient batch processing</li>
            <li>Real-time visualization</li>
            <li>State saving/loading</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
            <h4>🎯 Quick Start</h4>
            <ol>
            <li>Configure simulation parameters</li>
            <li>Initialize enhanced simulation</li>
            <li>Set TEM observation schedule</li>
            <li>Run assimilation cycles</li>
            <li>Analyze comprehensive results</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("Quick Start: Run Enhanced Demo", use_container_width=True):
            st.session_state.hybrid_system.initialize_simulation(
                nx=256, ny=256, Lx=200e-9, Ly=200e-9, dt=0.01, c_rate=1.0
            )
            st.session_state.sim_initialized = True
            
            # Run initial cycles
            with st.spinner("Running initial simulation..."):
                for _ in range(100):
                    st.session_state.hybrid_system.phase_field.run_cycle_step()
            
            st.rerun()
        
        return
    
    # Simulation is initialized
    hybrid = st.session_state.hybrid_system
    phase_field = hybrid.phase_field
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Current State",
        "🔬 TEM Assimilation",
        "📊 Advanced Analysis",
        "⚡ Performance",
        "📚 Documentation"
    ])
    
    with tab1:
        st.markdown("### 📊 Current Phase Field State")
        
        # Top metrics row
        col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
        
        with col_metrics1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Time", f"{phase_field.time:.0f} s")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_metrics2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            diag = phase_field.get_comprehensive_diagnostics()
            st.metric("Li Deficit (x)", f"{diag['lithium_deficit']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_metrics3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("FePO₄ Domains", diag['num_domains'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_metrics4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Interface Density", f"{diag['interface_density']:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Main visualization
        col_viz1, col_viz2 = st.columns([2, 1])
        
        with col_viz1:
            # Concentration field
            fig_current = go.Figure()
            fig_current.add_trace(go.Heatmap(
                z=phase_field.c.T,
                colorscale='RdYlBu',
                zmin=0, zmax=1,
                colorbar=dict(title="x in LiₓFePO₄", len=0.8)
            ))
            fig_current.update_layout(
                title=f"Lithium Concentration (t = {phase_field.time:.0f}s, Cycle {phase_field.cycle_count})",
                xaxis_title="x position (nm)",
                yaxis_title="y position (nm)",
                height=500,
                margin=dict(l=0, r=0, t=40, b=40)
            )
            st.plotly_chart(fig_current, use_container_width=True)
        
        with col_viz2:
            # Phase fractions pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=['LiFePO₄-rich', 'FePO₄-rich', 'Mixed'],
                values=[diag['LiFePO4_fraction'], 
                       diag['FePO4_fraction'],
                       diag['mixed_phase_fraction']],
                hole=0.3,
                marker_colors=['#4ECDC4', '#FF6B6B', '#FFD166'],
                textinfo='percent+label'
            )])
            fig_pie.update_layout(
                title="Phase Fractions",
                height=250,
                margin=dict(t=50, b=20, l=20, r=20),
                showlegend=False
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Histogram
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=phase_field.c.flatten(),
                nbinsx=50,
                marker_color='blue',
                opacity=0.7,
                name='Concentration'
            ))
            fig_hist.add_vline(x=phase_field.scales.c_FePO4, 
                             line_dash="dash", line_color="red")
            fig_hist.add_vline(x=phase_field.scales.c_LiFePO4, 
                             line_dash="dash", line_color="green")
            fig_hist.update_layout(
                title="Concentration Distribution",
                xaxis_title="x in LiₓFePO₄",
                yaxis_title="Frequency",
                height=250,
                margin=dict(t=40, b=40, l=40, r=40)
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Simulation controls
        st.markdown("### 🎮 Simulation Controls")
        col_ctl1, col_ctl2, col_ctl3, col_ctl4 = st.columns(4)
        
        with col_ctl1:
            steps = st.number_input("Steps to run", 1, 10000, 100)
            if st.button("▶️ Run Steps", use_container_width=True):
                with st.spinner(f"Running {steps} steps..."):
                    for _ in range(steps):
                        phase_field.run_cycle_step()
                st.rerun()
        
        with col_ctl2:
            cycles = st.number_input("Cycles to run", 1, 1000, 10)
            if st.button("🔁 Run Cycles", use_container_width=True):
                with st.spinner(f"Running {cycles} cycles..."):
                    phase_field.run_cycles_batch(cycles, cycles_per_step=10)
                st.rerun()
        
        with col_ctl3:
            observation_time = st.number_input("Observation Time (s)", 
                                             float(phase_field.time), 
                                             float(phase_field.time + 10000),
                                             float(phase_field.time + 1000))
            if st.button("📸 Generate TEM", use_container_width=True):
                with st.spinner("Generating TEM observation..."):
                    obs = hybrid._generate_single_observation(observation_time, 0.05, True)
                    st.session_state.tem_observation = obs
                    st.success("TEM observation generated!")
                st.rerun()
        
        with col_ctl4:
            if st.button("🔄 Reset Simulation", use_container_width=True):
                phase_field.initialize_lifepo4()
                st.session_state.assimilation_results = []
                st.rerun()
    
    with tab2:
        st.markdown("### 🔬 TEM Characterization & PINN Assimilation")
        
        # Generate observation schedule
        observation_schedule = []
        if phase_field.time > 0:
            start_time = phase_field.time
            for i in range(num_observations):
                observation_schedule.append(start_time + (i+1) * observation_interval)
        
        # Run assimilation
        if run_single and observation_schedule:
            t_obs = observation_schedule[0]
            with st.spinner(f"Running enhanced assimilation at t = {t_obs:.0f}s..."):
                pinn_params = {
                    'hidden_dims': hidden_dims,
                    'activation': activation.lower(),
                    'residual_connections': True
                }
                
                training_params = {
                    'n_epochs': n_epochs,
                    'lr': 1e-3,
                    'batch_size': batch_size,
                    'data_weight': data_weight,
                    'physics_weight': physics_weight,
                    'scheduler_type': 'cosine',
                    'warmup_epochs': 50
                }
                
                result = hybrid.run_assimilation_cycle_enhanced(
                    t_obs,
                    pinn_params=pinn_params,
                    training_params=training_params
                )
                
                st.session_state.assimilation_results.append(result)
                st.success(f"✅ Assimilation complete! MSE = {result['assimilation_metrics']['mse']:.2e}")
                st.rerun()
        
        if run_sequential and observation_schedule:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            for i, t_obs in enumerate(observation_schedule):
                status_text.text(f"Cycle {i+1}/{len(observation_schedule)} at t = {t_obs:.0f}s")
                progress_bar.progress((i) / len(observation_schedule))
                
                result = hybrid.run_assimilation_cycle_enhanced(t_obs)
                results.append(result)
            
            st.session_state.assimilation_results = results
            progress_bar.progress(1.0)
            status_text.text("✅ Sequential assimilation complete!")
            st.rerun()
        
        # Display assimilation results
        if st.session_state.assimilation_results:
            st.markdown("### 📊 Enhanced Assimilation Results")
            
            # Select cycle to view
            cycle_idx = st.selectbox(
                "Select Assimilation Cycle",
                range(len(st.session_state.assimilation_results)),
                format_func=lambda x: f"Cycle {x+1} at t={st.session_state.assimilation_results[x]['time']:.0f}s"
            )
            
            if cycle_idx < len(st.session_state.assimilation_results):
                result = st.session_state.assimilation_results[cycle_idx]
                
                # Comprehensive visualization
                fig_comprehensive = create_comprehensive_visualization(result)
                st.plotly_chart(fig_comprehensive, use_container_width=True)
                
                # Metrics summary
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                metrics = result['assimilation_metrics']
                
                with col_m1:
                    st.metric("Reconstruction MSE", f"{metrics['mse']:.2e}")
                    st.metric("RMSE", f"{metrics['rmse']:.3f}")
                
                with col_m2:
                    st.metric("SSIM", f"{metrics['ssim']:.3f}")
                    st.metric("Correlation", f"{metrics['correlation']:.3f}")
                
                with col_m3:
                    st.metric("Boundary Accuracy", f"{metrics['boundary_accuracy']:.3f}")
                    st.metric("Domain Count Error", f"{metrics['domain_count_error']:.3f}")
                
                with col_m4:
                    st.metric("Assimilation Weight", f"{result['assimilation_weight']:.3f}")
                    st.metric("Training Epochs", result['training_params']['n_epochs'])
                
                # Training details expander
                with st.expander("📈 Training Details"):
                    if 'training_stats' in result and 'loss_history' in result['training_stats']:
                        loss_history = result['training_stats']['loss_history']
                        df_loss = pd.DataFrame(loss_history)
                        
                        fig_loss = go.Figure()
                        fig_loss.add_trace(go.Scatter(
                            x=df_loss['epoch'], y=df_loss['total_loss'],
                            mode='lines', name='Total Loss',
                            line=dict(color='blue', width=2)
                        ))
                        fig_loss.add_trace(go.Scatter(
                            x=df_loss['epoch'], y=df_loss['data_loss'],
                            mode='lines', name='Data Loss',
                            line=dict(color='red', width=2, dash='dash')
                        ))
                        fig_loss.add_trace(go.Scatter(
                            x=df_loss['epoch'], y=df_loss['physics_loss'],
                            mode='lines', name='Physics Loss',
                            line=dict(color='green', width=2, dash='dot')
                        ))
                        
                        fig_loss.update_layout(
                            title='PINN Training Loss History',
                            xaxis_title='Epoch',
                            yaxis_title='Loss',
                            yaxis_type='log',
                            height=400,
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_loss, use_container_width=True)
                        
                        # Training statistics table
                        st.dataframe(df_loss.tail(10))
        else:
            st.info("👈 Run assimilation cycles to see results here")
    
    with tab3:
        st.markdown("### 📊 Advanced Analysis")
        
        if st.session_state.assimilation_results:
            # Create performance dashboard
            fig_dashboard = create_performance_dashboard(hybrid)
            st.plotly_chart(fig_dashboard, use_container_width=True)
            
            # Export data
            with st.expander("💾 Export Analysis Data", expanded=False):
                # Create comprehensive dataframe
                summary_data = []
                for i, result in enumerate(st.session_state.assimilation_results):
                    diag = result['phase_field_diagnostics']
                    metrics = result['assimilation_metrics']
                    
                    summary_data.append({
                        'cycle': i+1,
                        'time_s': result['time'],
                        'cycles': result['cycles'],
                        'lithium_deficit': diag['lithium_deficit'],
                        'FePO4_fraction': diag['FePO4_fraction'],
                        'mixed_phase_fraction': diag['mixed_phase_fraction'],
                        'num_domains': diag['num_domains'],
                        'avg_domain_size_nm': diag['avg_domain_size_nm'],
                        'interface_density': diag['interface_density'],
                        'avg_curvature': diag['avg_curvature'],
                        'reconstruction_mse': metrics['mse'],
                        'reconstruction_rmse': metrics['rmse'],
                        'ssim': metrics['ssim'],
                        'correlation': metrics['correlation'],
                        'boundary_accuracy': metrics['boundary_accuracy'],
                        'assimilation_weight': result['assimilation_weight'],
                        'tem_noise_level': result['tem_observation']['noise_level']
                    })
                
                df_summary = pd.DataFrame(summary_data)
                
                # Display dataframe
                st.dataframe(df_summary, use_container_width=True, height=400)
                
                # Download options
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    csv = df_summary.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Summary CSV",
                        data=csv,
                        file_name="phase_decomposition_analysis.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col_dl2:
                    json_data = json.dumps(summary_data, indent=2)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name="phase_decomposition_analysis.json",
                        mime="application/json",
                        use_container_width=True
                    )
            
            # Advanced analysis expander
            with st.expander("🔍 Advanced Statistical Analysis", expanded=False):
                if len(st.session_state.assimilation_results) > 5:
                    # Time series analysis
                    times = [r['time'] for r in st.session_state.assimilation_results]
                    errors = [r['assimilation_metrics']['mse'] for r in st.session_state.assimilation_results]
                    
                    # Calculate trends
                    x = np.array(times)
                    y = np.log10(np.array(errors) + 1e-10)
                    
                    # Linear regression
                    slope, intercept = np.polyfit(x, y, 1)
                    trend_line = slope * x + intercept
                    
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(
                        x=times, y=errors,
                        mode='markers',
                        marker=dict(size=8, color='blue'),
                        name='MSE'
                    ))
                    fig_trend.add_trace(go.Scatter(
                        x=times, y=10**trend_line,
                        mode='lines',
                        line=dict(color='red', width=2, dash='dash'),
                        name=f'Trend (slope={slope:.2e})'
                    ))
                    
                    fig_trend.update_layout(
                        title='Error Trend Analysis',
                        xaxis_title='Time (s)',
                        yaxis_title='MSE',
                        yaxis_type='log',
                        height=400
                    )
                    
                    st.plotly_chart(fig_trend, use_container_width=True)
                    
                    # Statistical summary
                    col_stats1, col_stats2, col_stats3 = st.columns(3)
                    
                    with col_stats1:
                        st.metric("Error Mean", f"{np.mean(errors):.2e}")
                        st.metric("Error Std", f"{np.std(errors):.2e}")
                    
                    with col_stats2:
                        st.metric("Error Min", f"{np.min(errors):.2e}")
                        st.metric("Error Max", f"{np.max(errors):.2e}")
                    
                    with col_stats3:
                        st.metric("Error Trend", f"{slope:.2e}")
                        st.metric("R²", f"{np.corrcoef(x, y)[0,1]**2:.3f}")
        else:
            st.info("Run assimilation cycles to see advanced analysis")
    
    with tab4:
        st.markdown("### ⚡ Performance Dashboard")
        
        # Get performance statistics
        perf_stats = perf_mon.get_performance_stats()
        assim_stats = hybrid.get_assimilation_statistics()
        
        # Performance metrics
        col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
        
        with col_perf1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Operations", f"{perf_stats.get('total_operations', 0)}")
            st.metric("Total Time", f"{perf_stats.get('total_time', 0):.2f} s")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_perf2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg Time/Operation", f"{perf_stats.get('avg_time_per_op', 0):.4f} s")
            st.metric("Assimilations", f"{assim_stats.get('num_assimilations', 0)}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_perf3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg Assimilation Time", f"{assim_stats.get('avg_assimilation_time', 0):.2f} s")
            st.metric("Total Assimilation Time", f"{assim_stats.get('total_assimilation_time', 0):.2f} s")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_perf4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if 'mse_mean' in assim_stats:
                st.metric("Avg MSE", f"{assim_stats['mse_mean']:.2e}")
                st.metric("MSE Trend", f"{assim_stats.get('mse_trend', 0):.2e}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed performance table
        st.markdown("#### 📊 Operation Performance")
        if perf_stats and 'summary' in perf_stats:
            df_perf = perf_stats['summary']
            df_perf_display = df_perf.reset_index()
            df_perf_display.columns = ['Operation', 'Mean Time (s)', 'Std Dev', 'Min', 'Max', 'Count']
            
            st.dataframe(
                df_perf_display.style.format({
                    'Mean Time (s)': '{:.6f}',
                    'Std Dev': '{:.6f}',
                    'Min': '{:.6f}',
                    'Max': '{:.6f}'
                }),
                use_container_width=True,
                height=400
            )
        
        # Performance optimization tips
        with st.expander("💡 Performance Optimization Tips", expanded=False):
            st.markdown("""
            ### Optimization Strategies:
            
            **1. Numba JIT Optimization:**
            - All critical functions are JIT-compiled
            - Use `fastmath=True` for faster but less precise math
            - Parallel loops with `prange` for multi-core support
            
            **2. Memory Optimization:**
            - Use `float32` instead of `float64` where possible
            - Pre-allocate arrays for history tracking
            - Batch processing for GPU operations
            
            **3. GPU Acceleration:**
            - Automatic CUDA detection and usage
            - Large batch sizes for GPU efficiency
            - Gradient checkpointing for memory efficiency
            
            **4. Algorithm Optimization:**
            - Adaptive time stepping
            - Sparse matrix operations where possible
            - Caching of intermediate results
            
            **5. Parallel Processing:**
            - Multi-threading for TEM image generation
            - Async I/O operations
            - Distributed computing for ensemble methods
            """)
        
        # Clear performance data
        if st.button("🔄 Clear Performance Data", use_container_width=True):
            global perf_mon
            perf_mon = PerformanceMonitor()
            st.success("Performance data cleared!")
            st.rerun()
    
    with tab5:
        st.markdown("### 📚 Enhanced Documentation")
        
        col_doc1, col_doc2 = st.columns([2, 1])
        
        with col_doc1:
            # Theory documentation
            with st.expander("🧪 Enhanced Physics Model", expanded=True):
                st.markdown("""
                #### Enhanced LiFePO₄ Phase Decomposition Model
                
                **Physical Basis:**
                - Cahn-Hilliard equation with multiple lithium loss mechanisms
                - Temperature-dependent diffusion coefficients
                - Realistic interface energy and width
                - Multiple boundary conditions (periodic, Neumann)
                
                **Enhanced Features:**
                1. **Multiple Loss Mechanisms:**
                   - Uniform SEI formation loss
                   - Interface-enhanced loss at phase boundaries
                   - Surface loss at particle edges
                   - Diffusion-limited loss
                
                2. **Advanced Numerical Methods:**
                   - Numba JIT-compiled finite differences
                   - Adaptive time stepping
                   - Memory-efficient history tracking
                   - Parallel processing support
                
                3. **Comprehensive Diagnostics:**
                   - Domain statistics (size, shape, distribution)
                   - Interface properties (width, curvature, density)
                   - Energy components (bulk, gradient, total)
                   - Chemical potential distribution
                
                **Governing Equations:**
                ```
                ∂c/∂t = M∇²[∂f/∂c - κ∇²c] + Σ S_i
                ```
                Where S_i are the various loss mechanisms.
                """)
            
            with st.expander("🔬 Advanced TEM Physics", expanded=False):
                st.markdown("""
                #### Realistic TEM Contrast Mechanisms
                
                **1. Mass-Thickness Contrast:**
                ```
                I/I₀ = exp(-Σᵢ σᵢρᵢt)
                ```
                - σ: Element-dependent scattering cross-section
                - ρ: Density variation with Li content
                - t: Realistic thickness variation (wedge-shaped)
                
                **2. Diffraction Contrast:**
                - Based on Bragg condition with deviation parameter
                - Orientation-dependent modulation
                - Strain contrast at phase boundaries
                
                **3. Phase Contrast (HRTEM):**
                ```
                φ(x,y) = (π/λE) ∫ V(x,y,z) dz
                ```
                - Contrast Transfer Function (CTF) with defocus and spherical aberration
                - Multi-slice simulation for thick samples
                - Noise models: shot noise, detector noise
                
                **Image Processing:**
                - Multi-scale texture analysis
                - Adaptive thresholding
                - Confidence estimation
                """)
            
            with st.expander("🧠 Enhanced PINN Architecture", expanded=False):
                st.markdown("""
                #### Advanced Physics-Informed Neural Networks
                
                **Architecture Features:**
                - Residual connections for gradient flow
                - Adaptive activation functions (Swish, GELU, Mish)
                - Batch normalization layers
                - Dropout for regularization
                
                **Training Strategies:**
                1. **Transfer Learning:**
                   - Reuse trained weights from previous cycles
                   - Fine-tuning on new observations
                
                2. **Advanced Optimization:**
                   - AdamW with weight decay
                   - Cosine annealing learning rate
                   - Warmup phase for stable training
                   - Gradient clipping
                
                3. **Loss Functions:**
                   ```
                   L = w_data·L_data + w_physics·L_physics + w_reg·L_reg
                   ```
                   Where L_physics enforces Cahn-Hilliard equilibrium
                
                4. **Data Augmentation:**
                   - Random rotations and flips
                   - Noise injection
                   - Multi-scale training
                """)
        
        with col_doc2:
            # Quick reference cards
            st.markdown("""
            <div class="card">
            <h4>⚙️ Key Parameters</h4>
            <p><strong>Physical Scales:</strong></p>
            <ul>
            <li>Domain: 5-10 nm (FePO₄)</li>
            <li>Li deficit: 0-0.3 in Li₁₋ₓFePO₄</li>
            <li>Diffusion: 1e-14 m²/s (300K)</li>
            <li>C-rate factor: 0.1-20</li>
            </ul>
            <p><strong>TEM Parameters:</strong></p>
            <ul>
            <li>Acceleration: 80-300 kV</li>
            <li>Wavelength: 2.5-4.2 pm</li>
            <li>Contrast: 3 mechanisms</li>
            <li>Noise: 0-20%</li>
            </ul>
            <p><strong>PINN Parameters:</strong></p>
            <ul>
            <li>Layers: 4-8 hidden layers</li>
            <li>Neurons: 128-512 per layer</li>
            <li>Activation: Swish/GELU/Mish</li>
            <li>Training: 100-5000 epochs</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
            <h4>🚀 Performance Tips</h4>
            <ol>
            <li>Start with 256×256 grid for balance</li>
            <li>Use GPU if available for PINN training</li>
            <li>Enable parallel TEM generation</li>
            <li>Save states for later analysis</li>
            <li>Monitor performance dashboard</li>
            <li>Adjust assimilation weights based on error</li>
            <li>Use transfer learning for faster convergence</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
            <h4>📊 Output Metrics</h4>
            <ul>
            <li><strong>MSE/RMSE:</strong> Reconstruction accuracy</li>
            <li><strong>SSIM:</strong> Structural similarity</li>
            <li><strong>Correlation:</strong> Linear relationship</li>
            <li><strong>Boundary Accuracy:</strong> Interface detection</li>
            <li><strong>Domain Statistics:</strong> Size, shape, count</li>
            <li><strong>Interface Properties:</strong> Width, curvature</li>
            <li><strong>Energy Components:</strong> Bulk, gradient, total</li>
            <li><strong>Performance Metrics:</strong> Timing, memory</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Initialize performance monitor
    global perf_mon
    perf_mon = PerformanceMonitor()
    
    # Run application
    main()
