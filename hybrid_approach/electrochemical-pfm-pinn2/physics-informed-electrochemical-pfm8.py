# app.py - Phase Field FDM with TEM Characterization and PINN Data Assimilation
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
# 2. TEM PHYSICS FOR PHASE CHARACTERIZATION
# =====================================================
class TEMPhysics:
    """TEM contrast physics for LiFePO₄/FePO₄ phase identification"""
    
    def __init__(self):
        # Electron scattering factors (simplified model)
        # Based on Z-contrast: FePO₄ (Fe³⁺) vs LiFePO₄ (Fe²⁺)
        self.scattering_factors = {
            'FePO4': {
                'sigma_total': 2.8,    # Total scattering cross-section
                'sigma_elastic': 1.5,  # Elastic scattering
                'sigma_inelastic': 1.3, # Inelastic scattering
                'mean_free_path': 50e-9, # m
            },
            'LiFePO4': {
                'sigma_total': 2.6,    # Slightly lower for Fe²⁺
                'sigma_elastic': 1.4,
                'sigma_inelastic': 1.2,
                'mean_free_path': 55e-9,
            }
        }
        
        # TEM imaging parameters (typical for high-resolution TEM)
        self.acceleration_voltage = 200e3  # V
        self.wavelength = self.calculate_electron_wavelength()
        self.defocus = 0.0  # nm
        self.Cs = 1.0e-3    # Spherical aberration (m)
        
        # Phase contrast parameters
        self.phase_contrast_sigma = 0.5  # Phase contrast strength
        
    def calculate_electron_wavelength(self):
        """Calculate relativistic electron wavelength"""
        h = 6.626e-34  # Planck constant
        m0 = 9.109e-31  # Electron rest mass
        e = 1.602e-19   # Electron charge
        c = 3.0e8       # Speed of light
        
        # Relativistic correction
        E0 = m0 * c**2
        E = e * self.acceleration_voltage
        lambda_nr = h / np.sqrt(2 * m0 * E)  # Non-relativistic
        lambda_rel = lambda_nr / np.sqrt(1 + E/(2*E0))  # Relativistic
        return lambda_rel
    
    def compute_mass_thickness_contrast(self, c_field, thickness_map):
        """
        Compute mass-thickness contrast for TEM
        
        Parameters:
        -----------
        c_field : np.ndarray
            Lithium concentration field (0 = FePO₄, 1 = LiFePO₄)
        thickness_map : np.ndarray
            Sample thickness in meters
        
        Returns:
        --------
        I_mass : np.ndarray
            Mass-thickness contrast image
        """
        nx, ny = c_field.shape
        
        # Interpolate scattering cross-section based on composition
        sigma_FePO4 = self.scattering_factors['FePO4']['sigma_total']
        sigma_LiFePO4 = self.scattering_factors['LiFePO4']['sigma_total']
        
        sigma_field = sigma_FePO4 * (1 - c_field) + sigma_LiFePO4 * c_field
        
        # Mass-thickness contrast: I/I0 = exp(-σρt)
        # Simplified: I ∝ exp(-σ*t)
        I_mass = np.exp(-sigma_field * thickness_map)
        
        return I_mass
    
    def compute_diffraction_contrast(self, c_field, orientation_map=None):
        """
        Compute diffraction contrast from phase boundaries
        
        Parameters:
        -----------
        c_field : np.ndarray
            Lithium concentration field
        orientation_map : np.ndarray, optional
            Crystallographic orientation (in radians)
        
        Returns:
        --------
        I_diff : np.ndarray
            Diffraction contrast image
        """
        nx, ny = c_field.shape
        
        if orientation_map is None:
            # Generate random orientation for each grain
            orientation_map = np.random.rand(nx, ny) * 2*np.pi
        
        # Compute phase boundaries (high concentration gradient)
        grad_c = np.gradient(c_field)
        grad_mag = np.sqrt(grad_c[0]**2 + grad_c[1]**2)
        
        # Bragg condition approximation
        # Diffraction contrast strong at interfaces
        d_spacing = 0.3e-9  # Typical d-spacing for LiFePO₄ (200)
        theta_B = np.arcsin(self.wavelength / (2 * d_spacing))
        
        # Deviation from Bragg condition
        s = 0.1 / d_spacing  # Small deviation parameter
        
        # Diffraction contrast intensity (simplified)
        I_diff = 1.0 / (1.0 + (s * d_spacing)**2) * grad_mag
        
        # Add orientation-dependent modulation
        I_diff *= (1.0 + 0.3 * np.sin(4 * orientation_map))
        
        return I_diff
    
    def compute_phase_contrast(self, c_field, defocus=None):
        """
        Compute phase contrast for high-resolution TEM
        
        Parameters:
        -----------
        c_field : np.ndarray
            Lithium concentration field
        defocus : float, optional
            Defocus value in meters
        
        Returns:
        --------
        I_phase : np.ndarray
            Phase contrast image
        """
        if defocus is None:
            defocus = self.defocus
        
        # Phase shift due to electrostatic potential
        # FePO₄ and LiFePO₄ have different mean inner potentials
        V_FePO4 = 15.0  # Volts
        V_LiFePO4 = 14.0  # Volts
        
        # Phase shift: φ = (π/λE) * ∫ V dz
        lambda_rel = self.wavelength
        E = self.acceleration_voltage
        
        phase_shift_FePO4 = (np.pi / (lambda_rel * E)) * V_FePO4
        phase_shift_LiFePO4 = (np.pi / (lambda_rel * E)) * V_LiFePO4
        
        # Interpolate phase shift
        phase_shift_field = phase_shift_FePO4 * (1 - c_field) + phase_shift_LiFePO4 * c_field
        
        # Contrast transfer function (simplified)
        q_max = 1.0 / (2 * self.L0)  # Maximum spatial frequency
        qx, qy = np.meshgrid(np.linspace(-q_max, q_max, c_field.shape[1]),
                           np.linspace(-q_max, q_max, c_field.shape[0]))
        q = np.sqrt(qx**2 + qy**2)
        
        # CTF: sin(χ(q))
        chi = np.pi * lambda_rel * q**2 * (defocus - 0.5 * self.Cs * lambda_rel**2 * q**2)
        CTF = np.sin(chi)
        
        # Phase contrast image
        I_phase = np.abs(np.fft.ifft2(np.fft.fft2(phase_shift_field) * CTF))**2
        
        return I_phase
    
    def simulate_tem_image(self, c_field, thickness_variation=0.2, 
                          noise_level=0.05, include_phase_contrast=True):
        """
        Generate synthetic TEM image from concentration field
        
        Parameters:
        -----------
        c_field : np.ndarray
            Lithium concentration field (0-1)
        thickness_variation : float
            Relative thickness variation (0-1)
        noise_level : float
            Gaussian noise standard deviation
        include_phase_contrast : bool
            Whether to include phase contrast
        
        Returns:
        --------
        tem_image : np.ndarray
            Synthetic TEM image (normalized 0-1)
        components : dict
            Individual contrast components
        """
        nx, ny = c_field.shape
        
        # 1. Generate thickness variation (wedge-shaped for simplicity)
        thickness = 50e-9 * (1.0 + thickness_variation * 
                           (np.linspace(0, 1, nx)[:, np.newaxis] * 
                            np.linspace(0, 1, ny)[np.newaxis, :]))
        
        # 2. Mass-thickness contrast
        I_mass = self.compute_mass_thickness_contrast(c_field, thickness)
        
        # 3. Diffraction contrast
        I_diff = self.compute_diffraction_contrast(c_field)
        
        # 4. Phase contrast (optional, for HRTEM)
        if include_phase_contrast:
            I_phase = self.compute_phase_contrast(c_field)
            # Normalize and combine
            I_phase = (I_phase - np.min(I_phase)) / (np.max(I_phase) - np.min(I_phase))
            phase_weight = 0.3
        else:
            I_phase = np.zeros_like(I_mass)
            phase_weight = 0.0
        
        # 5. Combine contrasts with weights
        weights = {'mass': 0.5, 'diffraction': 0.3, 'phase': phase_weight}
        I_combined = (weights['mass'] * I_mass + 
                     weights['diffraction'] * I_diff + 
                     weights['phase'] * I_phase)
        
        # 6. Add shot noise (Poisson) and detector noise (Gaussian)
        I0 = 1000  # Average electron count per pixel
        I_poisson = np.random.poisson(I0 * I_combined) / I0
        I_noisy = I_poisson + noise_level * np.random.randn(nx, ny)
        
        # 7. Normalize to 0-1
        I_final = np.clip(I_noisy, 0, 1)
        I_final = (I_final - np.min(I_final)) / (np.max(I_final) - np.min(I_final) + 1e-8)
        
        components = {
            'mass_thickness': I_mass,
            'diffraction': I_diff,
            'phase_contrast': I_phase,
            'thickness_map': thickness
        }
        
        return I_final, components
    
    def extract_phase_from_tem(self, tem_image, method='gradient'):
        """
        Extract phase information from TEM image
        
        Parameters:
        -----------
        tem_image : np.ndarray
            TEM image
        method : str
            Extraction method ('gradient', 'threshold', 'ml')
        
        Returns:
        --------
        phase_field : np.ndarray
            Estimated phase field (0-1)
        confidence : np.ndarray
            Confidence map for estimation
        """
        if method == 'gradient':
            # Use image gradients to find phase boundaries
            grad = np.gradient(tem_image)
            grad_mag = np.sqrt(grad[0]**2 + grad[1]**2)
            
            # Phase field estimated from local image features
            phase_field = 1.0 - grad_mag  # Lower gradient = more uniform phase
            phase_field = (phase_field - np.min(phase_field)) / \
                         (np.max(phase_field) - np.min(phase_field))
            
            # Confidence based on gradient magnitude
            confidence = 1.0 - np.exp(-10 * grad_mag)
            
        elif method == 'threshold':
            # Simple thresholding
            threshold = np.percentile(tem_image, 50)
            phase_field = (tem_image > threshold).astype(float)
            confidence = np.abs(tem_image - threshold)
            confidence = confidence / np.max(confidence)
            
        else:  # Simple ML-like approach
            # Use texture features
            from scipy.ndimage import gaussian_filter, sobel
            smoothed = gaussian_filter(tem_image, sigma=1.0)
            edges = sobel(smoothed)
            texture = gaussian_filter(edges**2, sigma=2.0)
            
            phase_field = 1.0 - texture
            phase_field = (phase_field - np.min(phase_field)) / \
                         (np.max(phase_field) - np.min(phase_field))
            confidence = 0.5 + 0.5 * np.cos(2 * np.pi * phase_field)
        
        return phase_field, confidence

# =====================================================
# 3. PHASE FIELD MODEL FOR LITHIUM LOSS
# =====================================================
class LithiumLossPhaseField:
    """Phase field model for LiFePO₄ phase decomposition during cycling"""
    
    def __init__(self, nx=256, ny=256, Lx=200e-9, Ly=200e-9, dt=0.01, c_rate=1.0):
        # Grid parameters
        self.nx, self.ny = nx, ny
        self.Lx, self.Ly = Lx, Ly
        self.dx, self.dy = Lx/nx, Ly/ny
        self.dt = dt
        
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
        
        # Double-well coefficients for regular solution
        self.A = self.W_dim
        self.B = -2.0 * self.W_dim
        self.C = self.W_dim
        
        # Lithium loss parameters
        self.loss_rate = 1e-5 * self.scales.loss_factor  # Base loss rate
        self.current_loss_rate = self.loss_rate
        
        # Initialize fields
        self.c = np.zeros((nx, ny))  # Lithium concentration
        self.c_dot = np.zeros((nx, ny))  # Rate of change
        self.phase_mask = np.zeros((nx, ny))  # Phase identification
        
        # Time tracking
        self.time = 0.0
        self.step = 0
        self.cycle_count = 0
        
        # History for diagnostics
        self.history = {
            'time': [],
            'mean_c': [],
            'std_c': [],
            'FePO4_fraction': [],
            'interface_density': [],
            'domain_size': [],
            'loss_rate': []
        }
        
        # Initialize with homogeneous LiFePO₄
        self.initialize_lifepo4()
    
    def initialize_lifepo4(self, noise_level=0.02):
        """Initialize as homogeneous LiFePO₄ (fully lithiated)"""
        self.c = self.scales.c_LiFePO4 * np.ones((self.nx, self.ny))
        self.c += noise_level * np.random.randn(self.nx, self.ny)
        self.c = np.clip(self.c, 0, 1)
        self.phase_mask = (self.c > 0.5).astype(int)
        self.time = 0.0
        self.step = 0
        self.clear_history()
    
    def clear_history(self):
        """Clear history"""
        self.history = {
            'time': [], 'mean_c': [], 'std_c': [],
            'FePO4_fraction': [], 'interface_density': [],
            'domain_size': [], 'loss_rate': []
        }
        self.update_history()
    
    def update_history(self):
        """Update history statistics"""
        self.history['time'].append(self.time)
        self.history['mean_c'].append(np.mean(self.c))
        self.history['std_c'].append(np.std(self.c))
        
        # FePO₄ phase fraction (c < 0.5)
        self.history['FePO4_fraction'].append(np.sum(self.c < 0.5) / (self.nx * self.ny))
        
        # Interface density
        grad = np.gradient(self.c)
        grad_mag = np.sqrt(grad[0]**2 + grad[1]**2)
        self.history['interface_density'].append(np.mean(grad_mag > 0.1))
        
        # Domain size estimation
        labeled, num_features = ndimage.label(self.c > 0.5)
        if num_features > 0:
            sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
            avg_size = np.mean(sizes) * self.dx * 1e9  # Convert to nm
        else:
            avg_size = 0.0
        self.history['domain_size'].append(avg_size)
        
        self.history['loss_rate'].append(self.current_loss_rate)
    
    def chemical_potential(self, c):
        """Compute chemical potential from double-well free energy"""
        mu_chem = (2 * self.A * c + 
                   3 * self.B * c**2 + 
                   4 * self.C * c**3)
        return mu_chem
    
    def compute_laplacian(self, field):
        """Compute Laplacian with periodic boundary conditions"""
        laplacian = np.zeros_like(field)
        
        # Central differences
        laplacian[1:-1, 1:-1] = (
            (field[2:, 1:-1] + field[:-2, 1:-1] - 2*field[1:-1, 1:-1]) / self.dx**2 +
            (field[1:-1, 2:] + field[1:-1, :-2] - 2*field[1:-1, 1:-1]) / self.dy**2
        )
        
        # Periodic boundaries
        # x-boundaries
        laplacian[0, 1:-1] = (
            (field[1, 1:-1] + field[-1, 1:-1] - 2*field[0, 1:-1]) / self.dx**2 +
            (field[0, 2:] + field[0, :-2] - 2*field[0, 1:-1]) / self.dy**2
        )
        laplacian[-1, 1:-1] = (
            (field[0, 1:-1] + field[-2, 1:-1] - 2*field[-1, 1:-1]) / self.dx**2 +
            (field[-1, 2:] + field[-1, :-2] - 2*field[-1, 1:-1]) / self.dy**2
        )
        
        # y-boundaries
        laplacian[1:-1, 0] = (
            (field[2:, 0] + field[:-2, 0] - 2*field[1:-1, 0]) / self.dx**2 +
            (field[1:-1, 1] + field[1:-1, -1] - 2*field[1:-1, 0]) / self.dy**2
        )
        laplacian[1:-1, -1] = (
            (field[2:, -1] + field[:-2, -1] - 2*field[1:-1, -1]) / self.dx**2 +
            (field[1:-1, 0] + field[1:-1, -2] - 2*field[1:-1, -1]) / self.dy**2
        )
        
        # Corners
        laplacian[0, 0] = (
            (field[1, 0] + field[-1, 0] - 2*field[0, 0]) / self.dx**2 +
            (field[0, 1] + field[0, -1] - 2*field[0, 0]) / self.dy**2
        )
        laplacian[0, -1] = (
            (field[1, -1] + field[-1, -1] - 2*field[0, -1]) / self.dx**2 +
            (field[0, 0] + field[0, -2] - 2*field[0, -1]) / self.dy**2
        )
        laplacian[-1, 0] = (
            (field[0, 0] + field[-2, 0] - 2*field[-1, 0]) / self.dx**2 +
            (field[-1, 1] + field[-1, -1] - 2*field[-1, 0]) / self.dy**2
        )
        laplacian[-1, -1] = (
            (field[0, -1] + field[-2, -1] - 2*field[-1, -1]) / self.dx**2 +
            (field[-1, 0] + field[-1, -2] - 2*field[-1, -1]) / self.dy**2
        )
        
        return laplacian
    
    def apply_lithium_loss(self, cycle_intensity=1.0):
        """
        Apply lithium loss during cycling
        
        Parameters:
        -----------
        cycle_intensity : float
            Intensity of cycling (0-1)
        """
        # Lithium loss is proportional to current and occurs preferentially
        # at phase boundaries and surfaces
        
        # 1. Uniform loss (simulating SEI formation, trapped Li⁺)
        uniform_loss = self.loss_rate * cycle_intensity
        
        # 2. Interface-enhanced loss (loss at phase boundaries)
        grad = np.gradient(self.c)
        grad_mag = np.sqrt(grad[0]**2 + grad[1]**2)
        interface_loss = 2.0 * self.loss_rate * cycle_intensity * grad_mag
        
        # 3. Surface loss (edges of simulation)
        surface_mask = np.zeros_like(self.c)
        surface_width = 2
        surface_mask[:surface_width, :] = 1
        surface_mask[-surface_width:, :] = 1
        surface_mask[:, :surface_width] = 1
        surface_mask[:, -surface_width:] = 1
        surface_loss = 3.0 * self.loss_rate * cycle_intensity * surface_mask
        
        # Total loss
        total_loss = uniform_loss + interface_loss + surface_loss
        
        # Apply loss (negative flux)
        self.c -= total_loss * self.dt_phys
        
        # Ensure bounds
        self.c = np.clip(self.c, 0, 1)
        
        # Update current loss rate for history
        self.current_loss_rate = np.mean(total_loss)
    
    def phase_separation_step(self):
        """Perform one phase separation step (Cahn-Hilliard)"""
        # Compute chemical potential
        mu_chem = self.chemical_potential(self.c)
        
        # Add gradient term: -κ∇²c
        laplacian_c = self.compute_laplacian(self.c)
        mu_total = mu_chem - self.kappa_dim * laplacian_c
        
        # Compute Laplacian of chemical potential
        laplacian_mu = self.compute_laplacian(mu_total)
        
        # Cahn-Hilliard equation: ∂c/∂t = M∇²μ
        self.c_dot = self.M_dim * laplacian_mu
        
        # Update concentration
        self.c += self.c_dot * self.dt
        
        # Ensure bounds
        self.c = np.clip(self.c, 0, 1)
    
    def run_cycle_step(self, cycle_intensity=1.0):
        """Run one combined step: lithium loss + phase separation"""
        # Apply lithium loss (non-conservative)
        self.apply_lithium_loss(cycle_intensity)
        
        # Phase separation (conservative)
        self.phase_separation_step()
        
        # Update time
        self.time += self.dt_phys
        self.step += 1
        
        # Update phase mask
        self.phase_mask = (self.c > 0.5).astype(int)
        
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
        # Domain statistics
        labeled, num_domains = ndimage.label(self.c > 0.5)
        domain_sizes = []
        if num_domains > 0:
            for i in range(1, num_domains + 1):
                domain_sizes.append(np.sum(labeled == i))
        
        # Interface analysis
        grad = np.gradient(self.c)
        grad_mag = np.sqrt(grad[0]**2 + grad[1]**2)
        interface_pixels = np.sum(grad_mag > 0.1)
        
        return {
            'time': self.time,
            'cycles': self.cycle_count,
            'mean_lithium': np.mean(self.c),
            'lithium_deficit': 1.0 - np.mean(self.c),  # x in Li₁₋ₓFePO₄
            'FePO4_fraction': np.sum(self.c < 0.5) / (self.nx * self.ny),
            'num_domains': num_domains,
            'avg_domain_size_nm': np.mean(domain_sizes) * self.dx * 1e9 if domain_sizes else 0,
            'domain_size_std': np.std(domain_sizes) * self.dx * 1e9 if domain_sizes else 0,
            'interface_density': interface_pixels / (self.nx * self.ny),
            'loss_rate': self.current_loss_rate
        }

# =====================================================
# 4. SYNTHETIC TEM OBSERVATION GENERATOR
# =====================================================
class SyntheticTEMGenerator:
    """Generate synthetic TEM observations during cycling"""
    
    def __init__(self):
        self.tem_physics = TEMPhysics()
        self.observation_history = []
        
    def generate_tem_observation(self, phase_field_model, observation_time,
                               noise_level=0.05, include_hr=True):
        """
        Generate synthetic TEM observation at given time
        
        Parameters:
        -----------
        phase_field_model : LithiumLossPhaseField
            Current phase field state
        observation_time : float
            Time of observation
        noise_level : float
            TEM image noise
        include_hr : bool
            Include high-resolution phase contrast
        
        Returns:
        --------
        observation : dict
            TEM observation data
        """
        # Get current concentration field
        c_field = phase_field_model.c
        
        # Generate TEM image
        tem_image, components = self.tem_physics.simulate_tem_image(
            c_field,
            thickness_variation=0.2,
            noise_level=noise_level,
            include_phase_contrast=include_hr
        )
        
        # Extract phase information from TEM
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
        
        # Store in history
        self.observation_history.append(observation)
        
        return observation
    
    def generate_observation_series(self, phase_field_model, 
                                  observation_times, **kwargs):
        """Generate series of TEM observations"""
        observations = []
        
        for t_obs in observation_times:
            # Run simulation to observation time
            if phase_field_model.time < t_obs:
                steps_needed = int((t_obs - phase_field_model.time) / phase_field_model.dt_phys)
                for _ in range(steps_needed):
                    phase_field_model.run_cycle_step()
            
            # Generate observation
            obs = self.generate_tem_observation(phase_field_model, t_obs, **kwargs)
            observations.append(obs)
        
        return observations

# =====================================================
# 5. PINN FOR PHASE FIELD DATA ASSIMILATION
# =====================================================
class PhaseFieldPINN(nn.Module):
    """Physics-Informed Neural Network for phase field assimilation"""
    
    def __init__(self, Lx, Ly, hidden_dims=[64, 64, 64, 64]):
        super().__init__()
        self.Lx = Lx
        self.Ly = Ly
        
        # Normalization factors
        self.x_scale = 1.0 / Lx if Lx > 0 else 1.0
        self.y_scale = 1.0 / Ly if Ly > 0 else 1.0
        
        # Neural network
        layers = []
        input_dim = 2  # x, y
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.Tanh())
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Output concentration 0-1
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x, y):
        """Forward pass"""
        # Normalize coordinates
        x_norm = x * self.x_scale
        y_norm = y * self.y_scale
        
        inputs = torch.stack([x_norm, y_norm], dim=-1)
        return self.net(inputs).squeeze(-1)
    
    def compute_physics_loss(self, x, y, c_pred, 
                           W, kappa, M, dx):
        """
        Compute physics loss based on Cahn-Hilliard equation
        
        Parameters:
        -----------
        x, y : torch.Tensor
            Coordinates
        c_pred : torch.Tensor
            Predicted concentration
        W, kappa, M : float
            Phase field parameters
        dx : float
            Grid spacing
        """
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
        
        # Second derivatives (Laplacian)
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
        
        # Chemical potential: μ = f'(c) - κ∇²c
        # f(c) = Wc²(1-c)² (double-well)
        f_prime = 2 * W * c_pred * (1 - c_pred) * (1 - 2 * c_pred)
        mu = f_prime - kappa * laplacian_c
        
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
        
        laplacian_mu = torch.autograd.grad(
            grad_mu_x, x,
            grad_outputs=torch.ones_like(grad_mu_x),
            create_graph=True
        )[0] + torch.autograd.grad(
            grad_mu_y, y,
            grad_outputs=torch.ones_like(grad_mu_y),
            create_graph=True
        )[0]
        
        # Cahn-Hilliard equation: ∂c/∂t = M∇²μ
        # For steady-state or slow evolution, ∇²μ ≈ 0
        physics_loss = torch.mean(laplacian_mu**2)
        
        return physics_loss, mu, laplacian_c

class PINNAssimilationTrainer:
    """Trainer for PINN-based data assimilation"""
    
    def __init__(self, pinn, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.pinn = pinn.to(device)
        self.device = device
        self.loss_history = []
    
    def train(self, tem_observations, phase_field_params,
              n_epochs=1000, lr=1e-3, 
              data_weight=1.0, physics_weight=0.1):
        """
        Train PINN to assimilate TEM observations
        
        Parameters:
        -----------
        tem_observations : list
            List of TEM observation dictionaries
        phase_field_params : dict
            Phase field parameters {W, kappa, M, dx, Lx, Ly}
        n_epochs : int
            Number of training epochs
        lr : float
            Learning rate
        data_weight : float
            Weight for data loss
        physics_weight : float
            Weight for physics loss
        """
        # Prepare training data from TEM observations
        all_x, all_y, all_c = [], [], []
        
        for obs in tem_observations:
            nx, ny = obs['image_shape']
            
            # Create coordinate grid
            x_grid = np.linspace(0, phase_field_params['Lx'], nx)
            y_grid = np.linspace(0, phase_field_params['Ly'], ny)
            X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
            
            # Use estimated phase from TEM (or subsample for efficiency)
            c_data = obs['estimated_phase']
            
            # Flatten
            all_x.append(X.flatten())
            all_y.append(Y.flatten())
            all_c.append(c_data.flatten())
        
        # Combine all observations
        x_data = np.concatenate(all_x)
        y_data = np.concatenate(all_y)
        c_data = np.concatenate(all_c)
        
        # Convert to tensors
        x_tensor = torch.tensor(x_data, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_data, dtype=torch.float32).to(self.device)
        c_tensor = torch.tensor(c_data, dtype=torch.float32).to(self.device)
        
        # Create dataset
        dataset = TensorDataset(x_tensor, y_tensor, c_tensor)
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
        
        # Optimizer
        optimizer = optim.Adam(self.pinn.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=50, factor=0.5, verbose=False
        )
        
        # Training loop
        for epoch in range(n_epochs):
            epoch_data_loss = 0.0
            epoch_physics_loss = 0.0
            
            for x_batch, y_batch, c_batch in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                c_pred = self.pinn(x_batch, y_batch)
                
                # Data loss
                data_loss = torch.mean((c_pred - c_batch)**2)
                
                # Physics loss
                physics_loss, mu, laplacian_c = self.pinn.compute_physics_loss(
                    x_batch, y_batch, c_pred,
                    phase_field_params['W'],
                    phase_field_params['kappa'],
                    phase_field_params['M'],
                    phase_field_params['dx']
                )
                
                # Total loss
                total_loss = (data_weight * data_loss + 
                            physics_weight * physics_loss)
                
                total_loss.backward()
                optimizer.step()
                
                epoch_data_loss += data_loss.item() * len(x_batch)
                epoch_physics_loss += physics_loss.item() * len(x_batch)
            
            # Average losses
            epoch_data_loss /= len(x_tensor)
            epoch_physics_loss /= len(x_tensor)
            epoch_total_loss = (data_weight * epoch_data_loss + 
                              physics_weight * epoch_physics_loss)
            
            # Store history
            self.loss_history.append({
                'epoch': epoch,
                'total_loss': epoch_total_loss,
                'data_loss': epoch_data_loss,
                'physics_loss': epoch_physics_loss
            })
            
            # Learning rate scheduling
            scheduler.step(epoch_total_loss)
            
            # Print progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Total Loss = {epoch_total_loss:.2e}, "
                      f"Data Loss = {epoch_data_loss:.2e}, "
                      f"Physics Loss = {epoch_physics_loss:.2e}")
        
        return {
            'final_loss': self.loss_history[-1]['total_loss'],
            'loss_history': self.loss_history,
            'num_observations': len(x_tensor)
        }
    
    def reconstruct_field(self, nx, ny, Lx, Ly):
        """Reconstruct full concentration field"""
        self.pinn.eval()
        
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
# 6. HYBRID FDM-PINN ASSIMILATION SYSTEM
# =====================================================
class HybridFDM_PINN_Assimilation:
    """Main system for hybrid FDM-PINN data assimilation"""
    
    def __init__(self):
        self.phase_field = None
        self.tem_generator = SyntheticTEMGenerator()
        self.pinn = None
        self.trainer = None
        
        # Assimilation history
        self.assimilation_history = []
        self.tem_observations = []
        
    def initialize_simulation(self, nx=256, ny=256, 
                            Lx=200e-9, Ly=200e-9, dt=0.01, c_rate=1.0):
        """Initialize phase field simulation"""
        self.phase_field = LithiumLossPhaseField(
            nx=nx, ny=ny, Lx=Lx, Ly=Ly, dt=dt, c_rate=c_rate
        )
    
    def collect_tem_observations(self, observation_times, 
                               noise_level=0.05, include_hr=True):
        """Collect TEM observations during cycling"""
        observations = self.tem_generator.generate_observation_series(
            self.phase_field,
            observation_times,
            noise_level=noise_level,
            include_hr=include_hr
        )
        
        self.tem_observations.extend(observations)
        return observations
    
    def run_assimilation_cycle(self, observation_time, 
                              pinn_hidden_dims=[64, 64, 64, 64],
                              n_epochs=500, lr=1e-3,
                              data_weight=1.0, physics_weight=0.1):
        """
        Run one assimilation cycle
        
        Parameters:
        -----------
        observation_time : float
            Time for TEM observation
        pinn_hidden_dims : list
            PINN architecture
        n_epochs : int
            Training epochs
        lr : float
            Learning rate
        data_weight, physics_weight : float
            Loss weights
        """
        # 1. Run simulation to observation time
        if self.phase_field.time < observation_time:
            steps_needed = int((observation_time - self.phase_field.time) / 
                             self.phase_field.dt_phys)
            for _ in range(steps_needed):
                self.phase_field.run_cycle_step()
        
        # 2. Generate TEM observation
        tem_obs = self.tem_generator.generate_tem_observation(
            self.phase_field, observation_time
        )
        self.tem_observations.append(tem_obs)
        
        # 3. Initialize PINN if not exists
        if self.pinn is None:
            self.pinn = PhaseFieldPINN(
                Lx=self.phase_field.Lx,
                Ly=self.phase_field.Ly,
                hidden_dims=pinn_hidden_dims
            )
            self.trainer = PINNAssimilationTrainer(self.pinn)
        
        # 4. Prepare phase field parameters for PINN
        phase_field_params = {
            'W': self.phase_field.W_dim,
            'kappa': self.phase_field.kappa_dim,
            'M': self.phase_field.M_dim,
            'dx': self.phase_field.dx,
            'Lx': self.phase_field.Lx,
            'Ly': self.phase_field.Ly
        }
        
        # 5. Train PINN on TEM observations
        training_stats = self.trainer.train(
            self.tem_observations,
            phase_field_params,
            n_epochs=n_epochs,
            lr=lr,
            data_weight=data_weight,
            physics_weight=physics_weight
        )
        
        # 6. Reconstruct field
        reconstructed_field = self.trainer.reconstruct_field(
            self.phase_field.nx,
            self.phase_field.ny,
            self.phase_field.Lx,
            self.phase_field.Ly
        )
        
        # 7. Compare with true field
        true_field = self.phase_field.c
        mse = np.mean((reconstructed_field - true_field)**2)
        
        # 8. Store assimilation results
        cycle_result = {
            'time': observation_time,
            'cycles': self.phase_field.cycle_count,
            'tem_observation': tem_obs,
            'true_field': true_field.copy(),
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
    
    def run_sequential_assimilation(self, observation_schedule, **kwargs):
        """Run sequential assimilation at multiple times"""
        results = []
        
        for i, t_obs in enumerate(observation_schedule):
            print(f"Running assimilation cycle {i+1}/{len(observation_schedule)} at t = {t_obs:.0f}s")
            
            result = self.run_assimilation_cycle(t_obs, **kwargs)
            results.append(result)
        
        return results
    
    def get_assimilation_statistics(self):
        """Get assimilation performance statistics"""
        if not self.assimilation_history:
            return {}
        
        stats = {
            'num_cycles': len(self.assimilation_history),
            'errors': [cycle['reconstruction_error'] 
                      for cycle in self.assimilation_history],
            'mean_error': np.mean([cycle['reconstruction_error'] 
                                  for cycle in self.assimilation_history]),
            'lithium_deficit_trajectory': [
                cycle['phase_field_diagnostics']['lithium_deficit']
                for cycle in self.assimilation_history
            ],
            'domain_evolution': [
                cycle['phase_field_diagnostics']['avg_domain_size_nm']
                for cycle in self.assimilation_history
            ]
        }
        
        return stats

# =====================================================
# 7. VISUALIZATION FUNCTIONS
# =====================================================
def plot_phase_field_comparison(cycle_result):
    """Plot comparison between true and reconstructed phase fields"""
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'True Li Concentration',
            'PINN Reconstruction',
            'Reconstruction Error',
            'TEM Image',
            'Estimated Phase from TEM',
            'Lithium Deficit Analysis'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # True concentration field
    fig.add_trace(
        go.Heatmap(
            z=cycle_result['true_field'].T,
            colorscale='RdYlBu',
            zmin=0, zmax=1,
            colorbar=dict(x=0.3, y=0.85, len=0.3),
            showscale=True
        ),
        row=1, col=1
    )
    
    # PINN reconstruction
    fig.add_trace(
        go.Heatmap(
            z=cycle_result['reconstructed_field'].T,
            colorscale='RdYlBu',
            zmin=0, zmax=1,
            colorbar=dict(x=0.7, y=0.85, len=0.3),
            showscale=True
        ),
        row=1, col=2
    )
    
    # Reconstruction error
    error = cycle_result['reconstructed_field'] - cycle_result['true_field']
    vmax = max(abs(error.min()), abs(error.max()))
    fig.add_trace(
        go.Heatmap(
            z=error.T,
            colorscale='RdBu_r',
            zmin=-vmax, zmax=vmax,
            colorbar=dict(x=0.3, y=0.45, len=0.3),
            showscale=True
        ),
        row=1, col=3
    )
    
    # TEM image
    tem_obs = cycle_result['tem_observation']
    fig.add_trace(
        go.Heatmap(
            z=tem_obs['tem_image'].T,
            colorscale='gray',
            showscale=False
        ),
        row=2, col=1
    )
    
    # Estimated phase from TEM
    fig.add_trace(
        go.Heatmap(
            z=tem_obs['estimated_phase'].T,
            colorscale='RdYlBu',
            zmin=0, zmax=1,
            showscale=False
        ),
        row=2, col=2
    )
    
    # Lithium deficit analysis
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
                colorbar=dict(title='Domain Size (nm)', x=0.95, y=0.45, len=0.3)
            ),
            text=f"Domains: {diagnostics['num_domains']}<br>"
                 f"Size: {diagnostics['avg_domain_size_nm']:.1f} nm<br>"
                 f"Interface: {diagnostics['interface_density']:.3f}",
            hoverinfo='text'
        ),
        row=2, col=3
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text=f"Assimilation at t = {cycle_result['time']:.0f}s, "
                   f"Cycle {cycle_result['cycles']}, "
                   f"Error = {cycle_result['reconstruction_error']:.2e}",
        title_x=0.5
    )
    
    # Update axes
    fig.update_xaxes(title_text="x (nm)", row=1, col=1)
    fig.update_yaxes(title_text="y (nm)", row=1, col=1)
    fig.update_xaxes(title_text="x (nm)", row=1, col=2)
    fig.update_xaxes(title_text="x (nm)", row=1, col=3)
    
    return fig

def plot_tem_physics(tem_obs):
    """Plot TEM physics components"""
    components = tem_obs['tem_components']
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Mass-Thickness Contrast',
            'Diffraction Contrast',
            'Phase Contrast',
            'Combined TEM Image',
            'Sample Thickness Map',
            'Phase Estimation Confidence'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Mass-thickness contrast
    fig.add_trace(
        go.Heatmap(
            z=components['mass_thickness'].T,
            colorscale='gray',
            showscale=False
        ),
        row=1, col=1
    )
    
    # Diffraction contrast
    fig.add_trace(
        go.Heatmap(
            z=components['diffraction'].T,
            colorscale='gray',
            showscale=False
        ),
        row=1, col=2
    )
    
    # Phase contrast
    if 'phase_contrast' in components:
        fig.add_trace(
            go.Heatmap(
                z=components['phase_contrast'].T,
                colorscale='gray',
                showscale=False
            ),
            row=1, col=3
        )
    
    # Combined TEM image
    fig.add_trace(
        go.Heatmap(
            z=tem_obs['tem_image'].T,
            colorscale='gray',
            colorbar=dict(x=0.45, y=0.45, len=0.3),
            showscale=True
        ),
        row=2, col=1
    )
    
    # Thickness map
    fig.add_trace(
        go.Heatmap(
            z=components['thickness_map'].T * 1e9,  # Convert to nm
            colorscale='Viridis',
            colorbar=dict(title='Thickness (nm)', x=0.85, y=0.45, len=0.3),
            showscale=True
        ),
        row=2, col=2
    )
    
    # Phase estimation confidence
    fig.add_trace(
        go.Heatmap(
            z=tem_obs['confidence_map'].T,
            colorscale='RdYlGn',
            zmin=0, zmax=1,
            showscale=False
        ),
        row=2, col=3
    )
    
    fig.update_layout(
        height=700,
        title_text=f"TEM Physics Analysis (t = {tem_obs['time']:.0f}s)",
        title_x=0.5
    )
    
    return fig

def plot_degradation_trajectory(assimilation_history):
    """Plot lithium loss and phase decomposition trajectory"""
    if not assimilation_history:
        return go.Figure()
    
    times = [cycle['time'] for cycle in assimilation_history]
    lithium_deficit = [cycle['phase_field_diagnostics']['lithium_deficit'] 
                      for cycle in assimilation_history]
    FePO4_fraction = [cycle['phase_field_diagnostics']['FePO4_fraction'] 
                     for cycle in assimilation_history]
    domain_sizes = [cycle['phase_field_diagnostics']['avg_domain_size_nm'] 
                   for cycle in assimilation_history]
    errors = [cycle['reconstruction_error'] for cycle in assimilation_history]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Lithium Deficit (x in Li₁₋ₓFePO₄)',
            'FePO₄ Phase Fraction',
            'FePO₄ Domain Size Evolution',
            'PINN Reconstruction Error'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # Lithium deficit
    fig.add_trace(
        go.Scatter(
            x=times,
            y=lithium_deficit,
            mode='lines+markers',
            line=dict(color='red', width=2),
            marker=dict(size=8),
            name='Li deficit'
        ),
        row=1, col=1
    )
    
    # FePO4 fraction
    fig.add_trace(
        go.Scatter(
            x=times,
            y=FePO4_fraction,
            mode='lines+markers',
            line=dict(color='blue', width=2),
            marker=dict(size=8),
            name='FePO₄ fraction'
        ),
        row=1, col=2
    )
    
    # Domain size
    fig.add_trace(
        go.Scatter(
            x=times,
            y=domain_sizes,
            mode='lines+markers',
            line=dict(color='green', width=2),
            marker=dict(size=8),
            name='Domain size'
        ),
        row=2, col=1
    )
    
    # Reconstruction error
    fig.add_trace(
        go.Scatter(
            x=times,
            y=errors,
            mode='lines+markers',
            line=dict(color='purple', width=2),
            marker=dict(size=8),
            name='Reconstruction error'
        ),
        row=2, col=2
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    
    fig.update_yaxes(title_text="Lithium deficit", row=1, col=1)
    fig.update_yaxes(title_text="FePO₄ fraction", row=1, col=2)
    fig.update_yaxes(title_text="Domain size (nm)", row=2, col=1)
    fig.update_yaxes(title_text="MSE", type="log", row=2, col=2)
    
    fig.update_layout(
        height=700,
        title_text="Phase Decomposition Trajectory During Cycling",
        title_x=0.5,
        showlegend=True
    )
    
    return fig

# =====================================================
# 8. STREAMLIT APPLICATION
# =====================================================
def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="LiFePO₄ Phase Decomposition with TEM & PINN Assimilation",
        page_icon="🔋",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .section-header {
        font-size: 1.8rem;
        color: #283593;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #c5cae9;
    }
    .card {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #3B82F6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.markdown('<h1 class="main-header">🔋 LiFePO₄ Phase Decomposition with TEM & PINN Assimilation</h1>', unsafe_allow_html=True)
    st.markdown("""
    *Modeling lithium loss and phase separation during battery cycling using Phase Field FDM, 
    synthetic TEM characterization, and Physics-Informed Neural Networks for data assimilation.*
    """)
    
    # Initialize session state
    if 'hybrid_system' not in st.session_state:
        st.session_state.hybrid_system = HybridFDM_PINN_Assimilation()
    if 'sim_initialized' not in st.session_state:
        st.session_state.sim_initialized = False
    if 'assimilation_results' not in st.session_state:
        st.session_state.assimilation_results = []
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/electron-microscope.png", width=80)
        st.markdown("### 🎛️ Control Panel")
        
        # Simulation setup
        with st.expander("⚙️ Phase Field Simulation Setup", expanded=True):
            grid_size = st.selectbox(
                "Grid Resolution",
                ["128×128 (Fast)", "256×256 (Standard)", "512×512 (High-res)"],
                index=1
            )
            if "128" in grid_size:
                nx, ny = 128, 128
            elif "256" in grid_size:
                nx, ny = 256, 256
            else:
                nx, ny = 512, 512
            
            domain_nm = st.slider("Domain Size (nm)", 50, 500, 200, 10)
            Lx = Ly = domain_nm * 1e-9
            
            c_rate = st.slider("C-Rate", 0.1, 10.0, 1.0, 0.1)
            dt = st.slider("Time Step (Δt)", 0.001, 0.1, 0.01, 0.001)
            
            init_lithium = st.slider("Initial Lithium Content", 0.8, 1.0, 0.97, 0.01)
            
            if st.button("🔄 Initialize Simulation", use_container_width=True):
                with st.spinner("Initializing phase field simulation..."):
                    st.session_state.hybrid_system.initialize_simulation(
                        nx=nx, ny=ny, Lx=Lx, Ly=Ly, dt=dt, c_rate=c_rate
                    )
                    st.session_state.hybrid_system.phase_field.scales.c_LiFePO4 = init_lithium
                    st.session_state.hybrid_system.phase_field.initialize_lifepo4()
                    st.session_state.sim_initialized = True
                    st.session_state.assimilation_results = []
                    st.success("✅ Simulation initialized!")
        
        # TEM observation settings
        with st.expander("🔬 TEM Characterization", expanded=True):
            tem_noise = st.slider("TEM Noise Level", 0.0, 0.2, 0.05, 0.01)
            include_hr = st.checkbox("Include Phase Contrast (HRTEM)", True)
            observation_interval = st.slider("Observation Interval (s)", 100, 10000, 1000, 100)
            num_observations = st.slider("Number of Observations", 3, 20, 5)
        
        # PINN assimilation settings
        with st.expander("🧠 PINN Assimilation", expanded=True):
            pinn_layers = st.text_input("PINN Hidden Layers", "64,64,64,64")
            try:
                hidden_dims = [int(x.strip()) for x in pinn_layers.split(",")]
            except:
                hidden_dims = [64, 64, 64, 64]
            
            n_epochs = st.slider("Training Epochs", 100, 2000, 500, 50)
            data_weight = st.slider("Data Loss Weight", 0.1, 5.0, 1.0, 0.1)
            physics_weight = st.slider("Physics Loss Weight", 0.01, 1.0, 0.1, 0.01)
        
        # Run controls
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            run_single = st.button("📸 Single Assimilation", use_container_width=True)
        with col2:
            run_sequential = st.button("🔄 Sequential Assimilation", use_container_width=True, type="primary")
    
    # Main content
    if not st.session_state.sim_initialized:
        # Welcome screen
        st.markdown("""
        <div class="card">
        <h3>🚀 Welcome to Phase Decomposition Analysis</h3>
        <p>This system models the phase decomposition in LiFePO₄ during battery cycling:</p>
        <ul>
        <li><strong>Phase Field FDM</strong>: Models lithium loss and FePO₄ nanodomain formation</li>
        <li><strong>Synthetic TEM Physics</strong>: Realistic TEM image generation with mass-thickness, diffraction, and phase contrast</li>
        <li><strong>PINN Data Assimilation</strong>: Physics-informed neural networks to reconstruct phase fields from sparse TEM data</li>
        <li><strong>Cycling Analysis</strong>: Track lithium deficit (x in Li₁₋ₓFePO₄) and FePO₄ domain evolution</li>
        </ul>
        <p><strong>Experimental Basis:</strong> Based on neutron diffraction/TEM studies showing 5-10 nm FePO₄ domains in Li₁₋ₓFePO₄ during cycling.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Quick Start: Run Demo", use_container_width=True):
            st.session_state.hybrid_system.initialize_simulation(
                nx=256, ny=256, Lx=200e-9, Ly=200e-9, dt=0.01, c_rate=1.0
            )
            st.session_state.sim_initialized = True
            
            # Run some cycles
            for _ in range(50):
                st.session_state.hybrid_system.phase_field.run_cycle_step()
            
            st.rerun()
        
        return
    
    # Simulation is initialized
    hybrid = st.session_state.hybrid_system
    phase_field = hybrid.phase_field
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Current State",
        "🔬 TEM Assimilation",
        "📊 Degradation Analysis",
        "📚 Theory & Documentation"
    ])
    
    with tab1:
        st.markdown("### 📊 Current Phase Field State")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Plot current concentration field
            fig_current = go.Figure()
            fig_current.add_trace(go.Heatmap(
                z=phase_field.c.T,
                colorscale='RdYlBu',
                zmin=0, zmax=1,
                colorbar=dict(title="x in LiₓFePO₄")
            ))
            fig_current.update_layout(
                title=f"Lithium Concentration (t = {phase_field.time:.0f}s, Cycle {phase_field.cycle_count})",
                xaxis_title="x position (nm)",
                yaxis_title="y position (nm)",
                height=500
            )
            st.plotly_chart(fig_current, use_container_width=True)
        
        with col2:
            # Current statistics
            st.subheader("Current Diagnostics")
            diag = phase_field.get_diagnostics()
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Time", f"{phase_field.time:.0f} s")
                st.metric("Cycles", phase_field.cycle_count)
                st.metric("Mean Li", f"{diag['mean_lithium']:.3f}")
                st.metric("Li deficit (x)", f"{diag['lithium_deficit']:.3f}")
            
            with col_b:
                st.metric("FePO₄ fraction", f"{diag['FePO4_fraction']:.1%}")
                st.metric("FePO₄ domains", diag['num_domains'])
                st.metric("Avg domain size", f"{diag['avg_domain_size_nm']:.1f} nm")
                st.metric("Interface density", f"{diag['interface_density']:.3f}")
            
            # Phase distribution pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=['LiFePO₄-rich', 'FePO₄-rich'],
                values=[1-diag['FePO4_fraction'], diag['FePO4_fraction']],
                hole=0.3,
                marker_colors=['#4ECDC4', '#FF6B6B']
            )])
            fig_pie.update_layout(
                title="Phase Fractions",
                height=250,
                margin=dict(t=50, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Run simulation controls
        st.subheader("Simulation Controls")
        col_ctl1, col_ctl2, col_ctl3 = st.columns(3)
        with col_ctl1:
            steps = st.number_input("Steps to run", 1, 1000, 100)
            if st.button("▶️ Run Steps", use_container_width=True):
                with st.spinner(f"Running {steps} steps..."):
                    for _ in range(steps):
                        phase_field.run_cycle_step()
                st.rerun()
        
        with col_ctl2:
            cycles = st.number_input("Cycles to run", 1, 100, 10)
            if st.button("🔁 Run Cycles", use_container_width=True):
                with st.spinner(f"Running {cycles} cycles..."):
                    phase_field.run_cycles(cycles, cycles_per_step=10)
                st.rerun()
        
        with col_ctl3:
            if st.button("🔄 Reset Simulation", use_container_width=True):
                phase_field.initialize_lifepo4()
                st.session_state.assimilation_results = []
                st.rerun()
    
    with tab2:
        st.markdown("### 🔬 TEM Characterization & PINN Assimilation")
        
        # Generate TEM observation schedule
        observation_schedule = []
        if phase_field.time > 0:
            start_time = phase_field.time
            for i in range(num_observations):
                observation_schedule.append(start_time + (i+1) * observation_interval)
        
        # Run assimilation based on button clicks
        if run_single and observation_schedule:
            t_obs = observation_schedule[0]
            
            with st.spinner(f"Running assimilation at t = {t_obs:.0f}s..."):
                result = hybrid.run_assimilation_cycle(
                    t_obs,
                    pinn_hidden_dims=hidden_dims,
                    n_epochs=n_epochs,
                    data_weight=data_weight,
                    physics_weight=physics_weight
                )
                st.session_state.assimilation_results.append(result)
                st.success(f"✅ Assimilation complete! MSE = {result['reconstruction_error']:.2e}")
                st.rerun()
        
        if run_sequential and observation_schedule:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            for i, t_obs in enumerate(observation_schedule):
                status_text.text(f"Cycle {i+1}/{len(observation_schedule)} at t = {t_obs:.0f}s")
                progress_bar.progress((i) / len(observation_schedule))
                
                result = hybrid.run_assimilation_cycle(
                    t_obs,
                    pinn_hidden_dims=hidden_dims,
                    n_epochs=n_epochs,
                    data_weight=data_weight,
                    physics_weight=physics_weight
                )
                results.append(result)
            
            st.session_state.assimilation_results = results
            progress_bar.progress(1.0)
            status_text.text("✅ Sequential assimilation complete!")
            st.rerun()
        
        # Display assimilation results
        if st.session_state.assimilation_results:
            st.markdown("### 📊 Assimilation Results")
            
            # Select cycle to view
            cycle_idx = st.selectbox(
                "Select Assimilation Cycle",
                range(len(st.session_state.assimilation_results)),
                format_func=lambda x: f"Cycle {x+1} at t={st.session_state.assimilation_results[x]['time']:.0f}s"
            )
            
            if cycle_idx < len(st.session_state.assimilation_results):
                result = st.session_state.assimilation_results[cycle_idx]
                
                # Plot comparison
                fig_compare = plot_phase_field_comparison(result)
                st.plotly_chart(fig_compare, use_container_width=True)
                
                # TEM physics details
                with st.expander("🔬 TEM Physics Details"):
                    fig_tem = plot_tem_physics(result['tem_observation'])
                    st.plotly_chart(fig_tem, use_container_width=True)
                
                # Training statistics
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                with col_s1:
                    st.metric("Reconstruction MSE", f"{result['reconstruction_error']:.2e}")
                with col_s2:
                    st.metric("Training Epochs", result['pinn_params']['n_epochs'])
                with col_s3:
                    st.metric("Data Weight", result['pinn_params']['data_weight'])
                with col_s4:
                    st.metric("Physics Weight", result['pinn_params']['physics_weight'])
                
                # Training loss history
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
                        height=400
                    )
                    st.plotly_chart(fig_loss, use_container_width=True)
        else:
            st.info("👈 Run assimilation cycles to see results here")
    
    with tab3:
        st.markdown("### 📊 Phase Decomposition Analysis")
        
        if st.session_state.assimilation_results:
            # Plot degradation trajectory
            fig_traj = plot_degradation_trajectory(st.session_state.assimilation_results)
            st.plotly_chart(fig_traj, use_container_width=True)
            
            # Export data
            with st.expander("💾 Export Analysis Data"):
                # Create summary dataframe
                summary_data = []
                for i, result in enumerate(st.session_state.assimilation_results):
                    diag = result['phase_field_diagnostics']
                    summary_data.append({
                        'cycle': i+1,
                        'time_s': result['time'],
                        'cycles': result['cycles'],
                        'lithium_deficit': diag['lithium_deficit'],
                        'FePO4_fraction': diag['FePO4_fraction'],
                        'num_domains': diag['num_domains'],
                        'avg_domain_size_nm': diag['avg_domain_size_nm'],
                        'domain_size_std_nm': diag['domain_size_std'],
                        'interface_density': diag['interface_density'],
                        'reconstruction_mse': result['reconstruction_error'],
                        'tem_noise_level': result['tem_observation']['noise_level']
                    })
                
                df_summary = pd.DataFrame(summary_data)
                
                # Display dataframe
                st.dataframe(df_summary, use_container_width=True)
                
                # Download button
                csv = df_summary.to_csv(index=False)
                st.download_button(
                    label="📥 Download Summary CSV",
                    data=csv,
                    file_name="phase_decomposition_analysis.csv",
                    mime="text/csv"
                )
        else:
            st.info("Run assimilation cycles to see degradation analysis")
    
    with tab4:
        st.markdown("### 📚 Theory & Documentation")
        
        col_doc1, col_doc2 = st.columns([2, 1])
        
        with col_doc1:
            with st.expander("🧪 Phase Decomposition Physics", expanded=True):
                st.markdown("""
                #### LiFePO₄ Phase Decomposition During Cycling
                
                **Experimental Observations (from neutron diffraction/TEM):**
                - Continuous lithium loss during cycling → Li₁₋ₓFePO₄
                - Phase separation into FePO₄ nanodomains (5-10 nm) in LiFePO₄ matrix
                - Preservation of olivine skeleton during phase change
                
                **Phase Field Model:**
                ```
                ∂c/∂t = M∇²[∂f/∂c - κ∇²c] + S_loss
                ```
                Where:
                - `c`: Lithium concentration (0 = FePO₄, 1 = LiFePO₄)
                - `M`: Mobility coefficient
                - `f(c)`: Double-well free energy: f(c) = Wc²(1-c)²
                - `κ`: Gradient energy coefficient
                - `S_loss`: Lithium loss source term
                
                **Lithium Loss Mechanisms:**
                1. **Uniform loss**: SEI formation, trapped Li⁺
                2. **Interface-enhanced loss**: Higher at phase boundaries
                3. **Surface loss**: At particle edges
                """)
            
            with st.expander("🔬 TEM Characterization Physics", expanded=False):
                st.markdown("""
                #### TEM Contrast Mechanisms
                
                **1. Mass-Thickness Contrast:**
                ```
                I/I₀ = exp(-σρt)
                ```
                - `σ`: Total scattering cross-section
                - `ρ`: Density (varies with Li content)
                - `t`: Sample thickness
                - FePO₄ scatters more electrons (Fe³⁺ vs Fe²⁺)
                
                **2. Diffraction Contrast:**
                - Strong at phase boundaries (strain fields)
                - Bragg condition: λ = 2d sinθ
                - Sensitive to crystal orientation
                
                **3. Phase Contrast (HRTEM):**
                ```
                φ = (π/λE) ∫ V dz
                ```
                - `V`: Electrostatic potential (different for FePO₄/LiFePO₄)
                - `λ`: Electron wavelength
                - `E`: Acceleration voltage
                """)
            
            with st.expander("🧠 PINN Data Assimilation", expanded=False):
                st.markdown("""
                #### Physics-Informed Neural Networks
                
                **Architecture:**
                - Input: Spatial coordinates (x, y)
                - Output: Lithium concentration c(x, y)
                - Activation: Tanh (smooth)
                - Final layer: Sigmoid (bounded 0-1)
                
                **Loss Function:**
                ```
                L = w_data·L_data + w_physics·L_physics
                ```
                
                **Data Loss:**
                ```
                L_data = MSE(c_PINN, c_TEM)
                ```
                Where c_TEM is estimated from TEM image
                
                **Physics Loss:**
                ```
                L_physics = MSE(∇²μ)
                ```
                Enforces Cahn-Hilliard equilibrium: ∇²μ ≈ 0
                """)
        
        with col_doc2:
            st.markdown("""
            <div class="card">
            <h4>🔋 Key Parameters</h4>
            <p><strong>Physical Scales:</strong></p>
            <ul>
            <li>Domain size: 5-10 nm (FePO₄)</li>
            <li>Li deficit x: 0-0.3 in Li₁₋ₓFePO₄</li>
            <li>Diffusion: 1e-14 m²/s</li>
            <li>C-rate factor: 1-10</li>
            </ul>
            <p><strong>TEM Parameters:</strong></p>
            <ul>
            <li>Acceleration: 200 kV</li>
            <li>Wavelength: 2.5 pm</li>
            <li>Contrast: Mass-thickness + diffraction</li>
            <li>Noise: 5-10% typical</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
            <h4>🎯 Quick Start Guide</h4>
            <ol>
            <li>Initialize simulation with grid size</li>
            <li>Set C-rate and initial Li content</li>
            <li>Configure TEM observation schedule</li>
            <li>Run assimilation cycles</li>
            <li>Analyze degradation trajectory</li>
            <li>Export results for comparison</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
