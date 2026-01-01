# app_publication.py - Complete Enhanced Application
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib import rcParams, gridspec
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import scipy
from scipy import stats, interpolate, ndimage
import warnings
warnings.filterwarnings('ignore')
import json
import time
from datetime import datetime
from pathlib import Path
import base64
from io import BytesIO
import pickle
import zipfile
from typing import Dict, List, Tuple, Optional, Any
import itertools
from collections import defaultdict
import textwrap
import sys

# =====================================================
# 0. SETUP FOR PUBLICATION QUALITY
# =====================================================

# Set publication quality matplotlib settings
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
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

# Set page configuration
st.set_page_config(
    page_title="LiFePO₄ Phase Field & Degradation Model",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for publication-style
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    .main-header {
        font-family: 'Roboto', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: #1a237e;
        text-align: center;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3949ab;
    }
    
    .sub-header {
        font-family: 'Roboto', sans-serif;
        font-size: 1.4rem;
        font-weight: 300;
        color: #5c6bc0;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    .section-header {
        font-family: 'Roboto', sans-serif;
        font-size: 1.6rem;
        font-weight: 600;
        color: #283593;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #c5cae9;
    }
    
    .subsection-header {
        font-family: 'Roboto', sans-serif;
        font-size: 1.2rem;
        font-weight: 500;
        color: #3949ab;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    
    .paper-card {
        background: linear-gradient(135deg, #f5f7ff 0%, #ffffff 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-left: 5px solid #3949ab;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
    }
    
    .paper-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }
    
    .metric-box {
        background: linear-gradient(135deg, #3949ab 0%, #283593 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 3px 5px rgba(0, 0, 0, 0.1);
    }
    
    .equation-box {
        background-color: #f8f9fa;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
    }
    
    .warning-box {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .info-box {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .download-button {
        background: linear-gradient(135deg, #4caf50 0%, #2e7d32 100%);
        color: white;
        padding: 0.7rem 1.5rem;
        border-radius: 25px;
        border: none;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }
    
    .download-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3949ab, #5c6bc0);
    }
    
    .stSelectbox, .stSlider, .stNumberInput {
        font-family: 'Roboto', sans-serif;
    }
    
    .caption {
        font-size: 0.9rem;
        color: #666;
        font-style: italic;
        text-align: center;
        margin-top: 0.5rem;
    }
    
    .table-header {
        background-color: #3949ab;
        color: white;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# 1. ENHANCED PHYSICAL MODEL WITH DEGRADATION (CORRECTED)
# =====================================================

class EnhancedPhysicalScalesWithDegradation:
    """Enhanced physical scales with explicit degradation models"""
    
    # Fundamental constants
    R = 8.314462618  # J/(mol·K)
    F = 96485.33212  # C/mol
    k_B = 1.380649e-23  # J/K
    ε0 = 8.854187817e-12  # F/m
    e = 1.602176634e-19  # Elementary charge (C)
    Na = 6.02214076e23  # Avogadro's number
    
    def __init__(self, c_rate=1.0, temperature=298.15, degradation_mode="standard"):
        # Material properties
        self.T = temperature  # K
        self.T_ref = 298.15  # Reference temperature
        
        # LiFePO₄ phase compositions
        self.c_alpha = 0.03  # FePO₄ phase (x in Li_xFePO₄)
        self.c_beta = 0.97   # LiFePO₄ phase
        self.c_miscibility_gap = [0.05, 0.95]  # Miscibility gap boundaries
        
        # Molar volume and density
        self.V_m = 4.438e-5  # m³/mol (LiFePO₄)
        self.rho = 3600  # kg/m³ (density)
        self.M_w = 157.76e-3  # kg/mol (molar mass LiFePO₄)
        
        # Diffusion coefficients (anisotropic)
        self.D_bb = 1.0e-14  # m²/s - Fast along b-axis
        self.D_aa = 1.0e-16  # m²/s - Slow along a-axis
        self.D_cc = 1.0e-16  # m²/s - Slow along c-axis
        
        # Elastic properties (for stress calculation)
        self.E = 150e9  # Young's modulus (Pa) - LiFePO₄
        self.nu = 0.3  # Poisson's ratio
        self.G = self.E / (2 * (1 + self.nu))  # Shear modulus
        self.K = self.E / (3 * (1 - 2 * self.nu))  # Bulk modulus
        
        # Partial molar volume (for chemical expansion)
        self.omega = 3.0e-6  # m³/mol - partial molar volume of Li
        
        # Electrostatic properties
        self.ε_r = 15.0  # Relative permittivity
        self.ε = self.ε_r * self.ε0  # Absolute permittivity
        
        # Charge properties
        self.z = 1.0  # Li⁺ charge number
        self.ρ0 = 1.0e6  # Reference charge density
        
        # Thermodynamic properties
        self.Ω = 55e3  # Regular solution parameter (J/mol)
        self.delta_S = 35.0  # Entropy of mixing (J/mol·K)
        self.H_mix = 8e3  # Enthalpy of mixing (J/mol)
        
        # Kinetics parameters
        self.k0_bv = 1.0e-6  # BV rate constant (m/s)
        self.k0_mhc = 5.0e-7  # MHC rate constant (m/s)
        self.alpha = 0.5  # BV symmetry factor
        
        # SEI growth parameters
        self.k_SEI = 1.0e-12  # SEI growth rate constant (m/s)
        self.D_SEI = 1.0e-17  # Li⁺ diffusion in SEI (m²/s)
        self.L_SEI_0 = 10e-9  # Initial SEI thickness (10 nm)
        self.Ea_SEI = 50e3  # Activation energy for SEI growth (J/mol)
        
        # Mechanical degradation parameters
        self.K_IC = 1.5e6  # Fracture toughness (Pa·√m)
        self.sigma_max = 500e6  # Maximum tensile stress (Pa)
        self.gamma_surface = 1.0  # Surface energy (J/m²)
        
        # Calendar aging parameters
        self.k_calendar = 1.0e-9  # Calendar aging rate (1/s)
        
        # Set degradation mode
        self.degradation_mode = degradation_mode
        self.set_degradation_parameters(degradation_mode)
        
        # Set C-rate parameters
        self.set_c_rate_parameters(c_rate)
        
        # Set characteristic scales
        self.set_scales()
        
        # Calculate derived properties
        self.calculate_derived_properties()
        
    def set_degradation_parameters(self, mode):
        """Configure degradation parameters based on mode"""
        if mode == "accelerated":
            # Accelerated degradation for lab testing
            self.k_SEI *= 10
            self.k_calendar *= 100
            self.sigma_max *= 0.7  # Lower strength
            
        elif mode == "conservative":
            # Conservative/long-life parameters
            self.k_SEI *= 0.1
            self.k_calendar *= 0.01
            self.sigma_max *= 1.3  # Higher strength
            
        elif mode == "mechanical_focus":
            # Focus on mechanical degradation
            self.K_IC *= 0.5
            self.gamma_surface *= 0.8
            
        elif mode == "SEI_focus":
            # Focus on SEI growth
            self.k_SEI *= 5
            self.D_SEI *= 0.5
    
    def set_c_rate_parameters(self, c_rate):
        """Set C-rate dependent parameters"""
        self.c_rate = float(c_rate)
        
        # C-rate scaling factor
        if c_rate <= 0.1:
            self.c_rate_factor = 0.5
            self.eta_scale = 0.005
            self.heat_gen_factor = 0.1
        elif c_rate <= 1.0:
            self.c_rate_factor = 1.0
            self.eta_scale = 0.01
            self.heat_gen_factor = 1.0
        elif c_rate <= 5.0:
            self.c_rate_factor = 2.0
            self.eta_scale = 0.05
            self.heat_gen_factor = 5.0
        else:
            self.c_rate_factor = 5.0
            self.eta_scale = 0.1
            self.heat_gen_factor = 10.0
        
        # Rate-dependent interface sharpness
        self.kappa_factor = 1.0 / (1.0 + 0.5 * np.log10(max(0.1, c_rate)))
        
        # Rate-dependent mobility (effective diffusion)
        self.D_factor = 1.0 / (1.0 + 0.2 * c_rate**0.5)
        
        # Rate-dependent SEI growth
        self.SEI_factor = 1.0 + 0.5 * np.log10(max(1.0, c_rate))
        
    def set_scales(self):
        """Set characteristic scales for non-dimensionalization"""
        # Length scale: 100 nm domain (representative particle size)
        self.L0 = 1.0e-7  # 100 nm
        
        # Energy density scale from regular solution
        self.E0 = self.Ω / self.V_m  # J/m³
        
        # Time scale from diffusion (using fastest direction)
        self.t0 = (self.L0**2) / self.D_bb  # s
        
        # Stress scale (from chemical expansion)
        self.sigma0 = self.E * self.omega / self.V_m  # Pa
        
        # Electric potential scale (thermal voltage)
        self.phi0 = self.R * self.T / self.F  # V
        
        # Current density scale
        self.i0 = self.F * self.D_bb * self.c_beta / self.L0  # A/m²
        
        # Temperature scale
        self.T0 = self.eta_scale * self.F / (self.R * self.alpha)  # K
        
    def calculate_derived_properties(self):
        """Calculate important derived physical properties"""
        # Debye length
        c_ref_moles = 0.5 / self.V_m  # mol/m³
        self.lambda_D = np.sqrt(self.ε * self.R * self.T / 
                              (self.F**2 * c_ref_moles))
        
        # Diffusion length at 1C
        self.L_diff = np.sqrt(self.D_bb * 3600)  # Diffusion length in 1 hour
        
        # Theoretical capacity
        self.Q_theoretical = self.F / (self.M_w / 1000)  # C/g ≈ 170 mAh/g
        
        # Thermal properties
        self.Cp = 700  # Specific heat capacity (J/kg·K)
        self.k_thermal = 1.5  # Thermal conductivity (W/m·K)
        
        # Calculate Arrhenius factors
        self.arrhenius_T = np.exp(-self.Ea_SEI / self.R * 
                                 (1/self.T - 1/self.T_ref))
        
    def dimensionless_to_physical(self, params_dict):
        """Convert dimensionless parameters to physical"""
        phys_dict = {}
        
        if 'kappa' in params_dict:
            phys_dict['kappa'] = params_dict['kappa'] * self.E0 * self.L0**2
        
        if 'M' in params_dict:
            phys_dict['M'] = params_dict['M'] * self.D_bb / (self.E0 * self.t0)
        
        if 'W' in params_dict:
            phys_dict['W'] = params_dict['W'] * self.E0
        
        if 'dt' in params_dict:
            phys_dict['dt'] = params_dict['dt'] * self.t0
        
        if 'L' in params_dict:
            phys_dict['L'] = params_dict['L'] * self.L0
        
        return phys_dict

# =====================================================
# 2. ENHANCED PHASE FIELD MODEL WITH DEGRADATION (CORRECTED)
# =====================================================

class DegradationPhaseFieldModel:
    """Enhanced phase field model with explicit degradation mechanisms"""
    
    def __init__(self, nx=256, ny=256, Lx=1.0, Ly=1.0, 
                 physics_params=None, degradation_params=None):
        
        # Grid parameters
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx / (nx - 1) if nx > 1 else Lx
        self.dy = Ly / (ny - 1) if ny > 1 else Ly
        
        # Time parameters
        self.dt = 0.01
        self.time = 0.0
        self.step = 0
        
        # Initialize fields
        self.c = np.zeros((nx, ny))  # Lithium concentration
        self.phi = np.zeros((nx, ny))  # Electric potential
        self.T = 298.15 * np.ones((nx, ny))  # Temperature field
        self.stress = np.zeros((nx, ny, 3, 3))  # Stress tensor (3x3 at each point)
        self.damage = np.zeros((nx, ny))  # Damage field (0-1)
        self.SEI_thickness = np.zeros((nx, ny))  # SEI thickness
        
        # Degradation accumulators
        self.capacity_loss = 0.0
        self.power_loss = 0.0
        self.crack_length = 0.0
        self.SEI_resistance = 0.0
        
        # History tracking
        self.history = defaultdict(list)
        
        # Set physical parameters
        self.set_physics_parameters(physics_params or {})
        self.set_degradation_parameters(degradation_params or {})
        
        # Initialize
        self.initialize_fields()
        
    def set_physics_parameters(self, params):
        """Set physical parameters with defaults"""
        default_params = {
            # Thermodynamic parameters
            'A': 1.0,  # Double well coefficient
            'B': -2.0,
            'C': 1.0,
            'kappa': 0.1,  # Gradient energy coefficient
            'M': 1.0,  # Mobility
            'D': 1.0e-14,  # Diffusion coefficient
            
            # Electrostatic parameters
            'epsilon': 15.0 * 8.854e-12,
            'z': 1.0,
            'F': 96485.0,
            'R': 8.314,
            
            # Elastic parameters
            'E': 150e9,
            'nu': 0.3,
            'omega': 3.0e-6,  # Partial molar volume
            
            # Thermal parameters
            'k_thermal': 1.5,
            'Cp': 700,
            'rho': 3600,
            
            # Boundary conditions
            'c_left': 0.03,  # FePO4 at left boundary
            'c_right': 0.97,  # LiFePO4 at right boundary
            'phi_left': 0.0,
            'phi_right': 0.0,
            'eta': 0.0,  # Overpotential
            
            # Kinetics
            'kinetics_type': 'BV',  # 'PNP', 'BV', or 'MHC'
            'k0': 1.0e-6,
            'alpha': 0.5,
        }
        self.params = default_params.copy()
        self.params.update(params)
        
    def set_degradation_parameters(self, params):
        """Set degradation parameters with defaults"""
        default_params = {
            # SEI growth
            'k_SEI': 1.0e-12,
            'D_SEI': 1.0e-17,
            'L_SEI_0': 10e-9,
            'Ea_SEI': 50e3,
            
            # Mechanical degradation
            'K_IC': 1.5e6,
            'sigma_max': 500e6,
            'gamma_c': 1.0,  # Critical energy release rate
            'beta_crack': 0.1,  # Crack propagation coefficient
            
            # Calendar aging
            'k_calendar': 1.0e-9,
            'Q_calendar': 50e3,  # Activation energy
            
            # Capacity fade - CORRECTED: Removed stray % symbols
            'Q_loss_SEI': 0.1,  # capacity loss per micron SEI (0.1%)
            'Q_loss_mech': 0.05,  # capacity loss per crack (0.05%)
            
            # Resistance increase
            'R_increase_SEI': 0.1,  # Ω per micron SEI
        }
        self.degradation_params = default_params.copy()
        self.degradation_params.update(params)
        
    def initialize_fields(self, mode='lithiation', noise_level=0.05):
        """Initialize concentration field based on mode"""
        if mode == 'lithiation':
            # Start with FePO4 (low lithium)
            self.c = self.params['c_left'] * np.ones((self.nx, self.ny))
            # Add phase boundary in middle
            x = np.linspace(0, self.Lx, self.nx)
            y = np.linspace(0, self.Ly, self.ny)
            X, Y = np.meshgrid(x, y, indexing='ij')
            self.c += 0.5 * (1 + np.tanh((X - 0.5*self.Lx)/0.1)) * (
                self.params['c_right'] - self.params['c_left'])
            
        elif mode == 'delithiation':
            # Start with LiFePO4 (high lithium)
            self.c = self.params['c_right'] * np.ones((self.nx, self.ny))
            # Add phase boundary
            x = np.linspace(0, self.Lx, self.nx)
            y = np.linspace(0, self.Ly, self.ny)
            X, Y = np.meshgrid(x, y, indexing='ij')
            self.c -= 0.5 * (1 + np.tanh((X - 0.5*self.Lx)/0.1)) * (
                self.params['c_right'] - self.params['c_left'])
            
        elif mode == 'spinodal':
            # Random initialization in spinodal region
            self.c = 0.5 + noise_level * (np.random.rand(self.nx, self.ny) - 0.5)
            
        elif mode == 'two_phase':
            # Clear two-phase structure
            x = np.linspace(0, self.Lx, self.nx)
            y = np.linspace(0, self.Ly, self.ny)
            X, Y = np.meshgrid(x, y, indexing='ij')
            self.c = 0.5 + 0.4 * np.sin(4*np.pi*X/self.Lx) * np.sin(4*np.pi*Y/self.Ly)
        
        # Add random noise
        self.c += noise_level * 0.1 * (np.random.rand(self.nx, self.ny) - 0.5)
        self.c = np.clip(self.c, 0.0, 1.0)
        
        # Initialize SEI thickness
        self.SEI_thickness = self.degradation_params['L_SEI_0'] * np.ones((self.nx, self.ny))
        
        # Initialize damage field
        self.damage = 0.01 * np.random.rand(self.nx, self.ny)
        
        # Reset time
        self.time = 0.0
        self.step = 0
        self.history = defaultdict(list)
        
        # Store initial state
        self.record_history()
        
    def compute_chemical_potential(self):
        """Compute chemical potential including stress effects"""
        # Standard chemical potential from double well
        mu_chem = (2*self.params['A']*self.c + 
                   3*self.params['B']*self.c**2 + 
                   4*self.params['C']*self.c**3)
        
        # Add gradient term
        lap_c = self.laplacian(self.c)
        mu_chem -= self.params['kappa'] * lap_c
        
        # Add stress contribution (chemical expansion)
        if hasattr(self, 'stress_hydrostatic'):
            mu_chem += self.params['omega'] * self.stress_hydrostatic
        
        # Add electrostatic contribution
        mu_chem += self.params['z'] * self.params['F'] * self.phi
        
        return mu_chem
    
    def compute_stress_field(self):
        """Compute stress field from concentration gradients"""
        if self.nx <= 1 or self.ny <= 1:
            self.stress_vm = np.zeros_like(self.c)
            self.stress_hydrostatic = np.zeros_like(self.c)
            return
            
        # Strain from concentration (dilatational eigenstrain)
        epsilon_star = self.params['omega'] * (self.c - np.mean(self.c))
        
        # Compute stress using linear elasticity (plane stress)
        E = self.params['E']
        nu = self.params['nu']
        
        # For 2D plane stress
        C11 = E / (1 - nu**2)
        C12 = nu * E / (1 - nu**2)
        C44 = E / (2 * (1 + nu))
        
        # Compute strain gradients
        grad_c = np.gradient(self.c, self.dx, self.dy)
        
        # Simple approximation: stress proportional to concentration gradient
        sigma_xx = C11 * grad_c[0] * self.dx
        sigma_yy = C11 * grad_c[1] * self.dy
        sigma_xy = C44 * (grad_c[0] + grad_c[1]) * self.dx
        
        # Store stress components
        self.stress[:, :, 0, 0] = sigma_xx
        self.stress[:, :, 1, 1] = sigma_yy
        self.stress[:, :, 0, 1] = sigma_xy
        self.stress[:, :, 1, 0] = sigma_xy
        
        # Compute von Mises stress
        self.stress_vm = np.sqrt(
            0.5 * ((sigma_xx - sigma_yy)**2 + 
                   sigma_xx**2 + sigma_yy**2 + 
                   6 * sigma_xy**2)
        )
        
        # Hydrostatic stress
        self.stress_hydrostatic = (sigma_xx + sigma_yy) / 3
        
    def update_SEI_growth(self):
        """Update SEI thickness based on local conditions"""
        if self.nx <= 1 or self.ny <= 1:
            return
            
        # SEI growth rate depends on potential and temperature
        phi_surface = self.phi[0, :]  # Potential at surface
        
        # Arrhenius temperature dependence
        T_factor = np.exp(-self.degradation_params['Ea_SEI'] / 
                         (self.params['R'] * self.T[0, :]) * 
                         (1 - 298.15/self.T[0, :]))
        
        # Growth rate: higher at negative potentials (lithiation)
        growth_rate = (self.degradation_params['k_SEI'] * 
                      np.exp(-0.5 * self.params['F'] * phi_surface / 
                            (self.params['R'] * self.T[0, :])) *
                      T_factor)
        
        # Update SEI thickness (only at surface)
        self.SEI_thickness[0, :] += growth_rate * self.dt
        
        # Diffuse SEI thickness (simplified model)
        self.SEI_thickness = self.diffuse_field(self.SEI_thickness, 
                                               self.degradation_params['D_SEI'])
        
        # Update SEI resistance
        R_SEI = self.SEI_thickness * self.degradation_params['R_increase_SEI'] / 1e-6
        self.SEI_resistance = np.mean(R_SEI)
        
    def update_damage(self):
        """Update damage field based on stress and SEI"""
        if self.nx <= 1 or self.ny <= 1:
            return
            
        # Stress-based damage accumulation
        stress_ratio = self.stress_vm / self.degradation_params['sigma_max']
        stress_ratio = np.clip(stress_ratio, 0, 2)
        
        # Damage rate from stress
        damage_rate_stress = (self.degradation_params['beta_crack'] * 
                            stress_ratio**2 * self.dt)
        
        # SEI-induced damage (brittle SEI can crack)
        SEI_stress = (self.SEI_thickness * 1e9 * 0.1)  # Simplified stress from SEI
        SEI_stress_ratio = SEI_stress / self.degradation_params['sigma_max']
        
        damage_rate_SEI = (0.01 * self.degradation_params['beta_crack'] * 
                          SEI_stress_ratio**2 * self.dt)
        
        # Total damage rate
        damage_rate = damage_rate_stress + damage_rate_SEI
        
        # Update damage field
        self.damage = np.minimum(1.0, self.damage + damage_rate)
        
        # Compute crack length (simplified)
        crack_pixels = np.sum(self.damage > 0.5)
        self.crack_length = crack_pixels * self.dx
        
    def update_capacity_loss(self):
        """Update capacity loss from degradation mechanisms"""
        # SEI-related capacity loss (Li consumed in SEI)
        Q_loss_SEI = (np.mean(self.SEI_thickness) * 1e6 *  # convert to microns
                     self.degradation_params['Q_loss_SEI'])
        
        # Mechanical degradation capacity loss
        Q_loss_mech = (self.crack_length / self.Lx * 
                      self.degradation_params['Q_loss_mech'])
        
        # Calendar aging (time-dependent)
        Q_loss_calendar = (self.degradation_params['k_calendar'] * 
                          self.time * 100)  # percentage
        
        # Total capacity loss
        self.capacity_loss = min(100.0, Q_loss_SEI + Q_loss_mech + Q_loss_calendar)
        
        # Power loss (increased resistance)
        resistance_increase = (self.SEI_resistance + 
                             0.1 * self.crack_length / self.Lx)
        self.power_loss = min(100.0, resistance_increase * 10)  # simplified
        
    def laplacian(self, field):
        """Compute 5-point stencil Laplacian"""
        if self.nx <= 2 or self.ny <= 2:
            return np.zeros_like(field)
            
        laplacian = np.zeros_like(field)
        laplacian[1:-1, 1:-1] = (
            field[2:, 1:-1] + field[:-2, 1:-1] +
            field[1:-1, 2:] + field[1:-1, :-2] -
            4 * field[1:-1, 1:-1]
        ) / (self.dx * self.dx)
        return laplacian
    
    def gradient(self, field):
        """Compute gradient of field"""
        if self.nx <= 1 or self.ny <= 1:
            return np.zeros_like(field), np.zeros_like(field)
        grad_x, grad_y = np.gradient(field, self.dx, self.dy)
        return grad_x, grad_y
    
    def diffuse_field(self, field, D):
        """Diffuse field with coefficient D"""
        if self.nx <= 2 or self.ny <= 2:
            return field
        laplacian = self.laplacian(field)
        return field + D * self.dt * laplacian
    
    def step_time(self, n_steps=1):
        """Advance simulation by n_steps"""
        for _ in range(n_steps):
            # Update chemical potential
            mu = self.compute_chemical_potential()
            
            # Update concentration (simplified Cahn-Hilliard)
            grad_mu = self.gradient(mu)
            if isinstance(grad_mu, tuple):
                grad_mu_x, grad_mu_y = grad_mu
                flux_x = -self.params['M'] * grad_mu_x
                flux_y = -self.params['M'] * grad_mu_y
                
                # Compute divergence of flux
                if self.nx > 2 and self.ny > 2:
                    div_flux = np.zeros_like(self.c)
                    div_flux[1:-1, 1:-1] = (
                        (flux_x[2:, 1:-1] - flux_x[:-2, 1:-1]) / (2*self.dx) +
                        (flux_y[1:-1, 2:] - flux_y[1:-1, :-2]) / (2*self.dy)
                    )
                    self.c -= self.dt * div_flux
            
            self.c = np.clip(self.c, 0.0, 1.0)
            
            # Update electric potential (simplified Poisson)
            charge_density = self.params['F'] * (self.c - 0.5)
            self.phi = self.diffuse_field(self.phi, 1/self.params['epsilon']) - charge_density
            
            # Update stress
            self.compute_stress_field()
            
            # Update degradation mechanisms
            self.update_SEI_growth()
            self.update_damage()
            
            # Update capacity loss
            self.update_capacity_loss()
            
            # Update time
            self.time += self.dt
            self.step += 1
            
            # Record history periodically
            if self.step % 10 == 0:
                self.record_history()
    
    def record_history(self):
        """Record simulation state in history"""
        self.history['time'].append(self.time)
        self.history['mean_c'].append(np.mean(self.c))
        self.history['std_c'].append(np.std(self.c))
        self.history['phase_alpha'].append(np.mean(self.c < 0.5))
        self.history['phase_beta'].append(np.mean(self.c >= 0.5))
        self.history['capacity_loss'].append(self.capacity_loss)
        self.history['power_loss'].append(self.power_loss)
        self.history['crack_length'].append(self.crack_length)
        self.history['SEI_thickness'].append(np.mean(self.SEI_thickness))
        
        if hasattr(self, 'stress_vm'):
            self.history['max_stress'].append(np.max(self.stress_vm))
        else:
            self.history['max_stress'].append(0.0)
        
        # Record full fields less frequently
        if self.step % 100 == 0:
            self.history['c_fields'].append(self.c.copy())
            self.history['phi_fields'].append(self.phi.copy())
            if hasattr(self, 'stress_vm'):
                self.history['stress_fields'].append(self.stress_vm.copy())
            self.history['damage_fields'].append(self.damage.copy())
            self.history['SEI_fields'].append(self.SEI_thickness.copy())

# =====================================================
# 3. ENHANCED VISUALIZATION MODULE FOR PUBLICATIONS
# =====================================================

class PublicationVisualizer:
    """Production of publication-quality figures and animations"""
    
    def __init__(self, style='nature'):
        """Initialize with specific journal style"""
        self.style = style
        self.set_style(style)
        
        # Color maps for different quantities
        self.cmaps = {
            'concentration': 'viridis',
            'potential': 'RdBu_r',
            'stress': 'inferno',
            'damage': 'hot',
            'SEI': 'YlOrBr',
            'temperature': 'plasma',
            'electric_field': 'rainbow'
        }
        
        # Journal-specific formats
        self.formats = {
            'nature': {'width': 7.2, 'height': 9.0, 'dpi': 600},  # Nature single column
            'science': {'width': 7.2, 'height': 9.0, 'dpi': 600},  # Science
            'acs': {'width': 3.5, 'height': 9.0, 'dpi': 600},  # ACS single column
            'elsevier': {'width': 9.0, 'height': 11.0, 'dpi': 600},  # Elsevier full page
            'thesis': {'width': 6.0, 'height': 8.0, 'dpi': 600},  # Thesis format
        }
        
    def set_style(self, style):
        """Set matplotlib style for specific journal"""
        if style == 'nature':
            plt.rcParams.update({
                'font.size': 7,
                'axes.labelsize': 8,
                'axes.titlesize': 9,
                'legend.fontsize': 7,
                'xtick.labelsize': 7,
                'ytick.labelsize': 7,
                'figure.titlesize': 10,
            })
        elif style == 'science':
            plt.rcParams.update({
                'font.size': 8,
                'axes.labelsize': 9,
                'axes.titlesize': 10,
                'legend.fontsize': 8,
                'xtick.labelsize': 8,
                'ytick.labelsize': 8,
                'figure.titlesize': 11,
            })
        elif style == 'acs':
            plt.rcParams.update({
                'font.size': 8,
                'axes.labelsize': 9,
                'axes.titlesize': 10,
                'legend.fontsize': 8,
                'xtick.labelsize': 8,
                'ytick.labelsize': 8,
                'figure.titlesize': 11,
            })
        elif style == 'elsevier':
            plt.rcParams.update({
                'font.size': 9,
                'axes.labelsize': 10,
                'axes.titlesize': 11,
                'legend.fontsize': 9,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9,
                'figure.titlesize': 12,
            })
    
    def create_multi_panel_figure(self, model, title="LiFePO₄ Phase Field Simulation"):
        """
        Create a comprehensive multi-panel figure for publications
        """
        fig = plt.figure(figsize=(self.formats[self.style]['width'], 
                                 self.formats[self.style]['height']))
        
        # Use gridspec for complex layout
        gs = gridspec.GridSpec(3, 3, figure=fig, 
                              height_ratios=[1.2, 1, 1],
                              width_ratios=[1, 1, 1],
                              hspace=0.3, wspace=0.3)
        
        # Panel A: Concentration field with phase boundaries
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_concentration_field(model.c, ax=ax1, add_colorbar=True)
        ax1.set_title('(a) Li Concentration Field', fontweight='bold', pad=10)
        ax1.text(0.02, 0.98, f't = {model.time:.1f} s', 
                transform=ax1.transAxes, fontsize=7,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Panel B: Electric potential with field lines
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_potential_field(model.phi, model.c, ax=ax2)
        ax2.set_title('(b) Electric Potential', fontweight='bold', pad=10)
        
        # Panel C: Stress distribution
        ax3 = fig.add_subplot(gs[0, 2])
        if hasattr(model, 'stress_vm'):
            self.plot_stress_field(model.stress_vm, ax=ax3)
            ax3.set_title('(c) Von Mises Stress', fontweight='bold', pad=10)
        else:
            ax3.text(0.5, 0.5, 'Stress data\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('(c) Von Mises Stress', fontweight='bold', pad=10)
        
        # Panel D: Degradation mechanisms
        ax4 = fig.add_subplot(gs[1, :])
        if model.history and 'time' in model.history and len(model.history['time']) > 1:
            self.plot_degradation_summary(model, ax=ax4)
            ax4.set_title('(d) Degradation Evolution', fontweight='bold', pad=10)
        else:
            ax4.text(0.5, 0.5, 'Run simulation to see\ndegradation evolution', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('(d) Degradation Evolution', fontweight='bold', pad=10)
        
        # Panel E: Phase evolution
        ax5 = fig.add_subplot(gs[2, 0])
        if model.history and 'time' in model.history and len(model.history['time']) > 1:
            self.plot_phase_evolution(model.history, ax=ax5)
            ax5.set_title('(e) Phase Fractions', fontweight='bold', pad=10)
        else:
            ax5.text(0.5, 0.5, 'Run simulation to see\nphase evolution', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('(e) Phase Fractions', fontweight='bold', pad=10)
        
        # Panel F: Capacity fade
        ax6 = fig.add_subplot(gs[2, 1])
        if model.history and 'time' in model.history and len(model.history['time']) > 1:
            self.plot_capacity_fade(model.history, ax=ax6)
            ax6.set_title('(f) Capacity Fade', fontweight='bold', pad=10)
        else:
            ax6.text(0.5, 0.5, 'Run simulation to see\ncapacity fade', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('(f) Capacity Fade', fontweight='bold', pad=10)
        
        # Panel G: Damage accumulation
        ax7 = fig.add_subplot(gs[2, 2])
        self.plot_damage_field(model.damage, ax=ax7)
        ax7.set_title('(g) Damage Accumulation', fontweight='bold', pad=10)
        
        # Add figure label
        fig.text(0.02, 0.98, 'Figure 1', fontsize=12, fontweight='bold',
                verticalalignment='top')
        
        # Add caption space
        caption_text = (f'{title}. Simulation parameters: Grid = {model.nx}×{model.ny}, '
                       f'Time = {model.time:.1f} s, Steps = {model.step:,}, '
                       f'Capacity loss = {model.capacity_loss:.1f}%.')
        fig.text(0.02, 0.02, caption_text,
                fontsize=7, style='italic', wrap=True)
        
        plt.tight_layout()
        return fig
    
    def plot_concentration_field(self, c_field, ax=None, add_colorbar=True, **kwargs):
        """Plot concentration field with publication quality"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3.5))
        
        # Create contour plot with phase boundaries
        X, Y = np.meshgrid(np.arange(c_field.shape[1]), 
                          np.arange(c_field.shape[0]))
        
        # Main contour plot
        contour = ax.contourf(X, Y, c_field.T, 
                             levels=50, 
                             cmap=self.cmaps['concentration'],
                             vmin=0, vmax=1)
        
        # Add phase boundary contour if there's variation
        if np.std(c_field) > 0.01:
            ax.contour(X, Y, c_field.T, 
                      levels=[0.5], 
                      colors='white', 
                      linewidths=1.5,
                      linestyles='--')
        
        ax.set_xlabel('x position')
        ax.set_ylabel('y position')
        ax.set_aspect('equal')
        
        if add_colorbar:
            cbar = plt.colorbar(contour, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('$x$ in Li$_x$FePO$_4$', rotation=270, labelpad=15)
            cbar.ax.tick_params(labelsize=7)
        
        return ax
    
    def plot_potential_field(self, phi_field, c_field=None, ax=None, **kwargs):
        """Plot electric potential with field lines"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3.5))
        
        X, Y = np.meshgrid(np.arange(phi_field.shape[1]), 
                          np.arange(phi_field.shape[0]))
        
        # Potential contour
        contour = ax.contourf(X, Y, phi_field.T, 
                             levels=50, 
                             cmap=self.cmaps['potential'])
        
        # Electric field vectors if there's variation
        if np.std(phi_field) > 0.01:
            try:
                Ex, Ey = np.gradient(-phi_field)  # E = -∇φ
                
                # Subsample for clarity
                stride = max(1, phi_field.shape[0]//15)
                ax.quiver(X[::stride, ::stride], Y[::stride, ::stride],
                         Ex.T[::stride, ::stride], Ey.T[::stride, ::stride],
                         color='white', scale=20, width=0.003,
                         headwidth=3, headlength=4)
            except:
                pass
        
        ax.set_xlabel('x position')
        ax.set_ylabel('y position')
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(contour, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Electric Potential (V)', rotation=270, labelpad=15)
        cbar.ax.tick_params(labelsize=7)
        
        return ax
    
    def plot_stress_field(self, stress_field, ax=None, **kwargs):
        """Plot stress distribution"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3.5))
        
        X, Y = np.meshgrid(np.arange(stress_field.shape[1]), 
                          np.arange(stress_field.shape[0]))
        
        # Convert to MPa for better readability
        stress_mpa = stress_field / 1e6
        
        contour = ax.contourf(X, Y, stress_mpa.T,
                             levels=50, 
                             cmap=self.cmaps['stress'])
        
        # Add stress concentration contours if there's variation
        if np.std(stress_field) > 0.01:
            ax.contour(X, Y, stress_mpa.T, 
                      levels=[np.percentile(stress_mpa, 95)], 
                      colors='red', 
                      linewidths=1.0,
                      linestyles='-')
        
        ax.set_xlabel('x position')
        ax.set_ylabel('y position')
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(contour, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Von Mises Stress (MPa)', rotation=270, labelpad=15)
        cbar.ax.tick_params(labelsize=7)
        
        return ax
    
    def plot_damage_field(self, damage_field, ax=None, **kwargs):
        """Plot damage accumulation"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3.5))
        
        X, Y = np.meshgrid(np.arange(damage_field.shape[1]), 
                          np.arange(damage_field.shape[0]))
        
        # Damage field
        contour = ax.contourf(X, Y, damage_field.T, 
                             levels=50, 
                             cmap=self.cmaps['damage'],
                             vmin=0, vmax=1)
        
        # Crack locations (damage > 0.7)
        if np.max(damage_field) > 0.7:
            crack_mask = damage_field.T > 0.7
            ax.contour(X, Y, crack_mask.astype(float), 
                      levels=[0.5], 
                      colors='black', 
                      linewidths=1.5)
        
        ax.set_xlabel('x position')
        ax.set_ylabel('y position')
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(contour, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Damage Parameter', rotation=270, labelpad=15)
        cbar.ax.tick_params(labelsize=7)
        
        return ax
    
    def plot_degradation_summary(self, model, ax=None):
        """Plot comprehensive degradation summary"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 3))
        
        if not model.history or 'time' not in model.history:
            return ax
        
        time = np.array(model.history['time'])
        
        # Create twin axes for different scales
        ax2 = ax.twinx()
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        
        # Plot capacity fade
        if 'capacity_loss' in model.history and len(model.history['capacity_loss']) > 0:
            capacity_loss = np.array(model.history['capacity_loss'])
            line1 = ax.plot(time/3600, capacity_loss, 'b-', linewidth=2, 
                           label='Capacity Loss (%)', marker='o', markersize=3)
        
        # Plot power loss
        if 'power_loss' in model.history and len(model.history['power_loss']) > 0:
            power_loss = np.array(model.history['power_loss'])
            line2 = ax2.plot(time/3600, power_loss, 'r--', linewidth=2,
                            label='Power Loss (%)', marker='s', markersize=3)
        
        # Plot crack length
        if 'crack_length' in model.history and len(model.history['crack_length']) > 0:
            crack_length = np.array(model.history['crack_length'])
            line3 = ax3.plot(time/3600, crack_length*1e9, 'g-.', linewidth=2,
                            label='Crack Length (nm)', marker='^', markersize=3)
        
        # Plot SEI thickness
        if 'SEI_thickness' in model.history and len(model.history['SEI_thickness']) > 0:
            SEI_thickness = np.array(model.history['SEI_thickness'])
            line4 = ax.plot(time/3600, SEI_thickness*1e9, 'm:', linewidth=2,
                           label='SEI Thickness (nm)', marker='d', markersize=3)
        
        # Configure axes
        ax.set_xlabel('Time (hours)', fontsize=9)
        ax.set_ylabel('Capacity/SEI (%)/nm', fontsize=9)
        ax2.set_ylabel('Power Loss (%)', fontsize=9)
        ax3.set_ylabel('Crack Length (nm)', fontsize=9)
        
        # Add legend
        lines = []
        labels = []
        if 'line1' in locals():
            lines.extend(line1)
            labels.append(line1[0].get_label())
        if 'line2' in locals():
            lines.extend(line2)
            labels.append(line2[0].get_label())
        if 'line3' in locals():
            lines.extend(line3)
            labels.append(line3[0].get_label())
        if 'line4' in locals():
            lines.extend(line4)
            labels.append(line4[0].get_label())
        
        if lines:
            ax.legend(lines, labels, loc='upper left', fontsize=8,
                     frameon=True, framealpha=0.9)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        return ax
    
    def plot_phase_evolution(self, history, ax=None):
        """Plot phase fraction evolution"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        
        if 'time' not in history or 'phase_alpha' not in history or 'phase_beta' not in history:
            return ax
        
        time = np.array(history['time'])
        phase_alpha = np.array(history['phase_alpha'])
        phase_beta = np.array(history['phase_beta'])
        
        ax.fill_between(time/3600, 0, phase_alpha*100, 
                       alpha=0.6, color='#FF6B6B', label='FePO$_4$-rich')
        ax.fill_between(time/3600, phase_alpha*100, 100, 
                       alpha=0.6, color='#4ECDC4', label='LiFePO$_4$-rich')
        
        ax.set_xlabel('Time (hours)', fontsize=9)
        ax.set_ylabel('Phase Fraction (%)', fontsize=9)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8, frameon=True, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        return ax
    
    def plot_capacity_fade(self, history, ax=None):
        """Plot capacity fade with different mechanisms"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        
        if 'time' not in history or 'capacity_loss' not in history:
            return ax
        
        time = np.array(history['time'])
        capacity_loss = np.array(history['capacity_loss'])
        
        # Simulated contributions (in reality would come from separate tracking)
        if len(capacity_loss) > 0:
            SEI_contribution = 0.6 * capacity_loss
            mechanical_contribution = 0.3 * capacity_loss
            calendar_contribution = 0.1 * capacity_loss
            
            ax.stackplot(time/3600, 
                        [SEI_contribution, mechanical_contribution, calendar_contribution],
                        colors=['#FFA726', '#EF5350', '#7E57C2'],
                        labels=['SEI Growth', 'Mechanical', 'Calendar Aging'],
                        alpha=0.8)
            
            ax.plot(time/3600, capacity_loss, 'k--', linewidth=1.5, 
                   label='Total Loss')
        
        ax.set_xlabel('Time (hours)', fontsize=9)
        ax.set_ylabel('Capacity Loss (%)', fontsize=9)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=7, frameon=True, framealpha=0.9, loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        return ax
    
    def create_3D_surface_plot(self, field, title=None, save_path=None):
        """Create 3D surface plot for concentration or stress"""
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        X, Y = np.meshgrid(np.arange(field.shape[1]), 
                          np.arange(field.shape[0]))
        
        # Create surface plot
        surf = ax.plot_surface(X, Y, field.T, 
                              cmap=self.cmaps['concentration'],
                              linewidth=0, 
                              antialiased=True,
                              alpha=0.8,
                              rstride=2, cstride=2)
        
        ax.set_xlabel('X position', labelpad=10)
        ax.set_ylabel('Y position', labelpad=10)
        ax.set_zlabel('Value', labelpad=10)
        
        if title:
            ax.set_title(title, fontweight='bold', pad=20)
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_animation(self, field_sequence, output_path, fps=10, dpi=150):
        """Create animation from field sequence"""
        try:
            import matplotlib.animation as animation
            
            fig, ax = plt.subplots(figsize=(6, 5))
            
            # Initial frame
            im = ax.imshow(field_sequence[0].T, 
                          cmap=self.cmaps['concentration'],
                          animated=True,
                          vmin=np.min(field_sequence),
                          vmax=np.max(field_sequence))
            ax.set_xlabel('X position')
            ax.set_ylabel('Y position')
            plt.colorbar(im, ax=ax, label='Concentration')
            
            def update(frame):
                im.set_array(field_sequence[frame].T)
                ax.set_title(f'Time step: {frame}', fontweight='bold')
                return [im]
            
            anim = animation.FuncAnimation(fig, update, 
                                          frames=min(len(field_sequence), 50),  # Limit frames
                                          interval=1000/fps, 
                                          blit=True)
            
            # Save animation
            anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
            plt.close()
            
            return anim
        except ImportError:
            st.warning("Animation requires matplotlib.animation module")
            return None
    
    def export_figure(self, fig, filename, formats=None):
        """Export figure in multiple publication formats"""
        if formats is None:
            formats = ['pdf', 'png', 'svg', 'eps']
        
        saved_files = []
        for fmt in formats:
            save_path = f"{filename}.{fmt}"
            try:
                fig.savefig(save_path, 
                           dpi=self.formats[self.style]['dpi'],
                           bbox_inches='tight',
                           pad_inches=0.1)
                saved_files.append(save_path)
            except Exception as e:
                st.warning(f"Could not save {fmt} format: {e}")
        
        return saved_files

# =====================================================
# 4. DATA EXPORT MODULE FOR PUBLICATIONS
# =====================================================

class PublicationDataExporter:
    """Export data in formats suitable for publications"""
    
    def __init__(self):
        self.export_formats = {
            'csv': self.export_to_csv,
            'excel': self.export_to_excel,
            'json': self.export_to_json,
            'hdf5': self.export_to_hdf5,
            'mat': self.export_to_mat,
            'pickle': self.export_to_pickle
        }
    
    def export_simulation_data(self, model, output_dir='./exports'):
        """Export complete simulation data for publication"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        export_info = {
            'metadata': self._create_metadata(model),
            'time_series': self._export_time_series(model),
            'snapshots': self._export_snapshots(model),
            'statistics': self._export_statistics(model),
            'degradation': self._export_degradation_data(model)
        }
        
        # Export in multiple formats
        exported_files = []
        for fmt, func in self.export_formats.items():
            try:
                if fmt in ['hdf5', 'mat']:  # Special formats
                    files = func(model, output_dir)
                else:
                    files = func(export_info, output_dir)
                
                if files:
                    exported_files.extend(files)
            except Exception as e:
                st.warning(f"Could not export {fmt} format: {e}")
        
        # Create README file
        readme_file = self._create_readme(model, output_dir, exported_files)
        if readme_file:
            exported_files.append(readme_file)
        
        # Create zip archive
        zip_path = self._create_zip_archive(output_dir, exported_files)
        
        return exported_files, zip_path
    
    def _create_metadata(self, model):
        """Create comprehensive metadata"""
        metadata = {
            'simulation_type': 'LiFePO4_PhaseField_Degradation',
            'timestamp': datetime.now().isoformat(),
            'grid_dimensions': {'nx': model.nx, 'ny': model.ny},
            'domain_size': {'Lx': model.Lx, 'Ly': model.Ly},
            'time_parameters': {
                'dt': model.dt,
                'total_time': model.time,
                'total_steps': model.step
            },
            'physical_parameters': model.params,
            'degradation_parameters': model.degradation_params,
            'simulation_conditions': {
                'initial_condition': getattr(model, 'initial_mode', 'lithiation')
            },
            'software_info': {
                'python_version': sys.version,
                'numpy_version': np.__version__,
                'simulation_version': '1.0.0'
            }
        }
        return metadata
    
    def _export_time_series(self, model):
        """Export time series data"""
        time_series = {}
        
        # Basic time series
        for key in model.history:
            if key.endswith('_fields'):  # Skip large arrays
                continue
            if len(model.history[key]) > 0:
                time_series[key] = np.array(model.history[key]).tolist()
        
        # Add derived quantities
        if 'time' in time_series and len(time_series['time']) > 0:
            time = np.array(time_series['time'])
            time_series['hours'] = (time / 3600).tolist()
            
            # Rate of change
            if 'capacity_loss' in time_series and len(time_series['capacity_loss']) > 1:
                capacity = np.array(time_series['capacity_loss'])
                time_series['capacity_fade_rate'] = np.gradient(capacity, time).tolist()
        
        return time_series
    
    def _export_snapshots(self, model):
        """Export field snapshots"""
        snapshots = {}
        
        if 'c_fields' in model.history and len(model.history['c_fields']) > 0:
            # Convert to list of lists for JSON serialization
            snapshots['concentration'] = [field.tolist() for field in model.history['c_fields']]
            snapshots['time_points'] = model.history['time'][::100][:len(snapshots['concentration'])]
        
        return snapshots
    
    def _export_statistics(self, model):
        """Export statistical analysis"""
        stats = {}
        
        if 'mean_c' in model.history and len(model.history['mean_c']) > 0:
            c_data = np.array(model.history['mean_c'])
            stats['concentration'] = {
                'mean': float(np.mean(c_data)),
                'std': float(np.std(c_data)),
                'min': float(np.min(c_data)),
                'max': float(np.max(c_data)),
                'final': float(c_data[-1])
            }
        
        if 'capacity_loss' in model.history and len(model.history['capacity_loss']) > 0:
            cap_data = np.array(model.history['capacity_loss'])
            stats['capacity'] = {
                'total_loss': float(cap_data[-1]),
                'projected_80_percent': self._project_lifetime(cap_data, model.history['time'])
            }
        
        # Phase statistics
        if ('phase_alpha' in model.history and 'phase_beta' in model.history and
            len(model.history['phase_alpha']) > 0 and len(model.history['phase_beta']) > 0):
            alpha = np.array(model.history['phase_alpha'])
            beta = np.array(model.history['phase_beta'])
            stats['phases'] = {
                'alpha_final': float(alpha[-1]),
                'beta_final': float(beta[-1]),
                'interface_area': float(self._calculate_interface_area(model))
            }
        
        return stats
    
    def _export_degradation_data(self, model):
        """Export detailed degradation data"""
        degradation = {}
        
        degradation['capacity_fade'] = {
            'current': float(model.capacity_loss),
            'SEI_contribution': float(model.capacity_loss * 0.6),  # Estimated
            'mechanical_contribution': float(model.capacity_loss * 0.3),
            'calendar_contribution': float(model.capacity_loss * 0.1)
        }
        
        degradation['mechanical'] = {
            'crack_length': float(model.crack_length),
            'damage_density': float(np.mean(model.damage))
        }
        
        degradation['SEI'] = {
            'mean_thickness': float(np.mean(model.SEI_thickness)),
            'max_thickness': float(np.max(model.SEI_thickness))
        }
        
        return degradation
    
    def _project_lifetime(self, capacity_data, time_data):
        """Project time to 80% capacity retention"""
        if len(capacity_data) < 2:
            return float('inf')
        
        # Simple linear projection
        time = np.array(time_data)
        capacity = np.array(capacity_data)
        
        if capacity[-1] >= 20:  # Haven't reached 80% yet
            return float('inf')
        
        # Fit linear model to last 5 points
        n_points = min(5, len(capacity))
        x = time[-n_points:] / 3600  # Convert to hours
        y = capacity[-n_points:]
        
        try:
            slope, intercept = np.polyfit(x, y, 1)
        except:
            return float('inf')
        
        # Time to reach 20% loss (80% retention)
        if abs(slope) < 1e-10:
            return float('inf')
        
        time_to_80 = (20 - intercept) / slope
        
        return max(0, float(time_to_80))
    
    def _calculate_interface_area(self, model):
        """Calculate phase interface area"""
        if not hasattr(model, 'c'):
            return 0
        
        c = model.c
        if model.nx <= 2 or model.ny <= 2:
            return 0
        
        # Find gradient magnitude
        try:
            grad_x, grad_y = np.gradient(c, model.dx, model.dy)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # Interface defined where gradient exceeds threshold
            threshold = 0.1 * grad_mag.max() if grad_mag.max() > 0 else 0.1
            interface_mask = grad_mag > threshold
            
            # Calculate interface length
            interface_pixels = np.sum(interface_mask)
            interface_length = interface_pixels * model.dx  # In physical units
            
            return interface_length
        except:
            return 0
    
    def export_to_csv(self, data, output_dir):
        """Export data to CSV files"""
        files = []
        
        try:
            # Export metadata
            metadata_df = pd.json_normalize(data['metadata'], sep='_')
            metadata_file = f"{output_dir}/metadata.csv"
            metadata_df.to_csv(metadata_file, index=False)
            files.append(metadata_file)
            
            # Export time series
            if data['time_series']:
                time_series_df = pd.DataFrame(data['time_series'])
                time_series_file = f"{output_dir}/time_series.csv"
                time_series_df.to_csv(time_series_file, index=False)
                files.append(time_series_file)
            
            # Export statistics
            if data['statistics']:
                # Flatten nested dictionary
                flat_stats = {}
                for category, values in data['statistics'].items():
                    for key, value in values.items():
                        if isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                flat_stats[f"{category}_{key}_{subkey}"] = subvalue
                        else:
                            flat_stats[f"{category}_{key}"] = value
                
                stats_df = pd.DataFrame([flat_stats])
                stats_file = f"{output_dir}/statistics.csv"
                stats_df.to_csv(stats_file, index=False)
                files.append(stats_file)
            
            # Export degradation data
            if data['degradation']:
                degradation_df = pd.json_normalize(data['degradation'], sep='_')
                degradation_file = f"{output_dir}/degradation.csv"
                degradation_df.to_csv(degradation_file, index=False)
                files.append(degradation_file)
        except Exception as e:
            st.warning(f"CSV export failed: {e}")
        
        return files
    
    def export_to_excel(self, data, output_dir):
        """Export data to Excel with multiple sheets"""
        try:
            excel_file = f"{output_dir}/simulation_data.xlsx"
            
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # Metadata sheet
                metadata_df = pd.json_normalize(data['metadata'], sep='_')
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
                
                # Time series sheet
                if data['time_series']:
                    time_series_df = pd.DataFrame(data['time_series'])
                    time_series_df.to_excel(writer, sheet_name='Time_Series', index=False)
                
                # Statistics sheet
                if data['statistics']:
                    flat_stats = {}
                    for category, values in data['statistics'].items():
                        for key, value in values.items():
                            if isinstance(value, dict):
                                for subkey, subvalue in value.items():
                                    flat_stats[f"{category}_{key}_{subkey}"] = subvalue
                            else:
                                flat_stats[f"{category}_{key}"] = value
                    
                    stats_df = pd.DataFrame([flat_stats])
                    stats_df.to_excel(writer, sheet_name='Statistics', index=False)
                
                # Degradation sheet
                if data['degradation']:
                    degradation_df = pd.json_normalize(data['degradation'], sep='_')
                    degradation_df.to_excel(writer, sheet_name='Degradation', index=False)
            
            return [excel_file]
        except Exception as e:
            st.warning(f"Excel export failed: {e}")
            return []
    
    def export_to_json(self, data, output_dir):
        """Export data to JSON format"""
        try:
            json_file = f"{output_dir}/simulation_data.json"
            
            # Convert numpy arrays to lists for JSON serialization
            def convert_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.generic):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(v) for v in obj]
                else:
                    return obj
            
            json_data = convert_for_json(data)
            
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            return [json_file]
        except Exception as e:
            st.warning(f"JSON export failed: {e}")
            return []
    
    def export_to_hdf5(self, model, output_dir):
        """Export to HDF5 format for large datasets"""
        try:
            import h5py
            hdf5_file = f"{output_dir}/simulation_data.h5"
            
            with h5py.File(hdf5_file, 'w') as f:
                # Create groups
                metadata = f.create_group('metadata')
                time_series = f.create_group('time_series')
                fields = f.create_group('fields')
                
                # Store metadata as attributes
                meta_data = self._create_metadata(model)
                for key, value in meta_data.items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata.attrs[key] = value
                
                # Store time series data
                ts_data = self._export_time_series(model)
                for key, values in ts_data.items():
                    if values:  # Only create dataset if there's data
                        time_series.create_dataset(key, data=np.array(values))
                
                # Store current fields
                current = f.create_group('current_fields')
                current.create_dataset('concentration', data=model.c)
                current.create_dataset('potential', data=model.phi)
                current.create_dataset('damage', data=model.damage)
                current.create_dataset('SEI', data=model.SEI_thickness)
            
            return [hdf5_file]
        except ImportError:
            st.warning("HDF5 export requires h5py package")
            return []
        except Exception as e:
            st.warning(f"HDF5 export failed: {e}")
            return []
    
    def export_to_mat(self, data, output_dir):
        """Export to MATLAB .mat format"""
        try:
            import scipy.io
            mat_file = f"{output_dir}/simulation_data.mat"
            
            # Prepare data for MATLAB
            mat_data = {}
            
            # Flatten metadata
            for key, value in data['metadata'].items():
                if isinstance(value, (str, int, float, bool)):
                    mat_data[f'metadata_{key}'] = value
            
            # Add time series
            for key, values in data['time_series'].items():
                if values:
                    mat_data[key] = np.array(values)
            
            # Add statistics
            for category, values in data['statistics'].items():
                for key, value in values.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            mat_data[f'stats_{category}_{key}_{subkey}'] = subvalue
                    else:
                        mat_data[f'stats_{category}_{key}'] = value
            
            scipy.io.savemat(mat_file, mat_data)
            return [mat_file]
        except ImportError:
            st.warning("MATLAB export requires scipy package")
            return []
        except Exception as e:
            st.warning(f"MATLAB export failed: {e}")
            return []
    
    def export_to_pickle(self, data, output_dir):
        """Export complete Python object"""
        try:
            pickle_file = f"{output_dir}/simulation_data.pkl"
            
            with open(pickle_file, 'wb') as f:
                pickle.dump(data, f)
            
            return [pickle_file]
        except Exception as e:
            st.warning(f"Pickle export failed: {e}")
            return []
    
    def _create_readme(self, model, output_dir, files):
        """Create README file describing exported data"""
        try:
            readme_content = f"""# LiFePO₄ Phase Field Simulation Data Export
## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Simulation Parameters
- Grid size: {model.nx} x {model.ny}
- Domain size: {model.Lx:.2e} m x {model.Ly:.2e} m
- Simulation time: {model.time:.2e} s ({model.time/3600:.2f} hours)
- Time steps: {model.step}
- Time step size: {model.dt:.2e} s

## Physical Parameters
- Regular solution parameter (Ω): {model.params.get('A', 'N/A')}
- Gradient coefficient (κ): {model.params.get('kappa', 'N/A')}
- Mobility (M): {model.params.get('M', 'N/A')}
- Young's modulus: {model.params.get('E', 'N/A')} Pa
- Poisson's ratio: {model.params.get('nu', 'N/A')}

## Degradation Parameters
- SEI growth rate: {model.degradation_params.get('k_SEI', 'N/A')} m/s
- Fracture toughness: {model.degradation_params.get('K_IC', 'N/A')} Pa·√m
- Calendar aging rate: {model.degradation_params.get('k_calendar', 'N/A')} 1/s

## Exported Files
"""
            
            for file in files:
                file_path = Path(file)
                if file_path.exists():
                    file_size = file_path.stat().st_size / 1024  # KB
                    readme_content += f"- `{file_path.name}` ({file_size:.1f} KB)\n"
            
            readme_content += """
## Data Structure

### 1. Time Series Data
Contains evolution of scalar quantities over time:
- `time`: Simulation time (s)
- `mean_c`: Mean lithium concentration
- `capacity_loss`: Total capacity loss (%)
- `power_loss`: Power loss (%)
- `crack_length`: Total crack length (m)
- `SEI_thickness`: Average SEI thickness (m)

### 2. Field Snapshots
Contains 2D field data at specific time points:
- `concentration`: Lithium concentration field
- `stress`: Von Mises stress field
- `damage`: Damage parameter field (0-1)
- `SEI`: SEI thickness field

### 3. Statistics
Summary statistics including:
- Mean, standard deviation, min, max values
- Capacity fade rates
- Phase fractions
- Interface properties

### 4. Degradation Analysis
Detailed breakdown of degradation mechanisms:
- Contribution of SEI growth, mechanical damage, calendar aging
- Current degradation state
- Projected lifetime estimates

## Usage Notes
- All spatial coordinates are in meters
- Time is in seconds unless otherwise noted
- Concentration is dimensionless (x in Li_xFePO₄)
- Stress values are in Pascals
- Damage parameter ranges from 0 (pristine) to 1 (fully damaged)

## Citation
If you use this data in a publication, please cite:
> LiFePO₄ Phase Field Degradation Model, [Your Institution], [Year]

## Contact
For questions about this data, contact: [Your Email]
"""
            
            readme_file = f"{output_dir}/README.txt"
            with open(readme_file, 'w') as f:
                f.write(readme_content)
            
            return readme_file
        except Exception as e:
            st.warning(f"README creation failed: {e}")
            return None
    
    def _create_zip_archive(self, output_dir, files):
        """Create zip archive of all exported files"""
        try:
            zip_path = f"{output_dir}/simulation_export.zip"
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file in files:
                    file_path = Path(file)
                    if file_path.exists():
                        arcname = file_path.relative_to(output_dir)
                        zipf.write(file, arcname)
            
            return zip_path
        except Exception as e:
            st.warning(f"ZIP archive creation failed: {e}")
            return None

# =====================================================
# 5. ENHANCED PINN FOR DATA ASSIMILATION
# =====================================================

class EnhancedLiFePO4AssimilationPINN(nn.Module):
    """Enhanced PINN with uncertainty quantification"""
    
    def __init__(self, input_dim=2, hidden_dims=[64, 64, 64], output_dim=1, dropout_rate=0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Tanh())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # Output between 0 and 1
        
        self.net = nn.Sequential(*layers)
        
        # Uncertainty estimation layers
        self.log_var = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, return_uncertainty=False):
        """Forward pass with optional uncertainty estimation"""
        mean = self.net(x)
        
        if return_uncertainty:
            # Simple uncertainty estimation
            variance = torch.exp(self.log_var)
            return mean, variance
        else:
            return mean
    
    def physics_loss(self, x, c_pred, params):
        """Physics-informed loss function"""
        # Enable gradient tracking
        x.requires_grad_(True)
        
        # Compute gradients
        grad_outputs = torch.ones_like(c_pred)
        grad_c = torch.autograd.grad(c_pred, x, grad_outputs=grad_outputs, 
                                    create_graph=True)[0]
        
        # Compute Laplacian
        grad_grad = torch.autograd.grad(grad_c, x, grad_outputs=torch.ones_like(grad_c),
                                       create_graph=True)[0]
        laplacian = grad_grad.sum(dim=1, keepdim=True)
        
        # Chemical potential from double-well
        A, B, C, kappa = params['A'], params['B'], params['C'], params['kappa']
        mu_chem = 2*A*c_pred + 3*B*c_pred**2 + 4*C*c_pred**3 - kappa*laplacian
        
        # Electrostatic contribution
        if 'phi' in params and 'z' in params and 'F' in params:
            mu_total = mu_chem + params['z'] * params['F'] * params['phi']
        else:
            mu_total = mu_chem
        
        # Physics loss: minimize variation in chemical potential
        loss_physics = torch.var(mu_total)
        
        # Interface sharpness regularization
        loss_interface = torch.mean(grad_c**2)
        
        return loss_physics + 0.1 * loss_interface

# =====================================================
# 6. MAIN STREAMLIT APPLICATION
# =====================================================

def main():
    """Main Streamlit application with publication features"""
    
    # Title
    st.markdown('<h1 class="main-header">LiFePO₄ Phase Field Degradation Model</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">High-fidelity simulation of phase separation, electrochemistry, and degradation mechanisms in battery electrodes</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = PublicationVisualizer(style='nature')
    
    if 'exporter' not in st.session_state:
        st.session_state.exporter = PublicationDataExporter()
    
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    
    # Sidebar with enhanced controls
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/battery--v1.png", width=80)
        st.markdown("### ⚙️ Simulation Controls")
        
        # Simulation setup
        with st.expander("📊 Simulation Setup", expanded=True):
            grid_option = st.selectbox(
                "Grid Resolution",
                ["64×64 (Fast, Draft)", "128×128 (Balanced, Standard)", 
                 "256×256 (Detailed, Publication)", "512×512 (High-res, Computational)"],
                index=1
            )
            
            if grid_option == "64×64 (Fast, Draft)":
                nx, ny = 64, 64
            elif grid_option == "128×128 (Balanced, Standard)":
                nx, ny = 128, 128
            elif grid_option == "256×256 (Detailed, Publication)":
                nx, ny = 256, 256
            else:
                nx, ny = 512, 512
            
            domain_size = st.slider(
                "Domain Size (nm)", 
                50, 1000, 200, 50,
                help="Physical size of simulation domain"
            )
            
            Lx = Ly = domain_size * 1e-9  # Convert to meters
            
            init_mode = st.selectbox(
                "Initial State",
                ["lithiation", "delithiation", "spinodal", "two_phase"],
                format_func=lambda x: x.replace("_", " ").title(),
                index=0
            )
            
            noise_level = st.slider(
                "Initial Noise Level",
                0.0, 0.2, 0.05, 0.01
            )
        
        # Physical parameters
        with st.expander("🔬 Physical Parameters", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                kappa = st.number_input(
                    "κ (Gradient energy)",
                    0.01, 10.0, 0.1, 0.01,
                    format="%.3f"
                )
                mobility = st.number_input(
                    "M (Mobility)",
                    0.01, 5.0, 1.0, 0.01,
                    format="%.3f"
                )
                omega = st.number_input(
                    "Ω (Partial molar vol, m³/mol)",
                    1e-7, 1e-5, 3e-6, 1e-7,
                    format="%.2e"
                )
            
            with col2:
                E_modulus = st.number_input(
                    "E (Young's modulus, Pa)",
                    50e9, 300e9, 150e9, 10e9,
                    format="%.1e"
                )
                poisson = st.slider(
                    "ν (Poisson's ratio)",
                    0.1, 0.4, 0.3, 0.05
                )
                k_SEI = st.number_input(
                    "k_SEI (Growth rate, m/s)",
                    1e-13, 1e-11, 1e-12, 1e-13,
                    format="%.2e"
                )
        
        # Degradation parameters
        with st.expander("⚠️ Degradation Parameters", expanded=False):
            degradation_mode = st.selectbox(
                "Degradation Mode",
                ["standard", "accelerated", "conservative", "mechanical_focus", "SEI_focus"],
                format_func=lambda x: x.replace("_", " ").title(),
                index=0
            )
            
            Q_loss_SEI = st.number_input(
                "Q_loss_SEI (% per μm SEI)",
                0.01, 1.0, 0.1, 0.01,
                format="%.2f",
                help="Capacity loss per micron of SEI thickness"
            )
            
            Q_loss_mech = st.number_input(
                "Q_loss_mech (% per crack)",
                0.01, 0.5, 0.05, 0.01,
                format="%.2f",
                help="Capacity loss per unit crack length"
            )
        
        # Simulation control
        with st.expander("⏱️ Run Simulation", expanded=True):
            dt = st.number_input(
                "Time Step (s)",
                1e-5, 0.1, 0.01, 1e-5,
                format="%.4f"
            )
            
            simulation_time = st.number_input(
                "Total Simulation Time (s)",
                1.0, 1e6, 3600.0, 100.0,
                format="%.1f"
            )
            
            steps_per_batch = st.slider(
                "Steps per Batch",
                10, 1000, 100, 10
            )
            
            col_run1, col_run2 = st.columns(2)
            with col_run1:
                if st.button("🔄 Initialize Model", use_container_width=True):
                    with st.spinner("Initializing model..."):
                        # Create physics parameters
                        physics_params = {
                            'kappa': kappa,
                            'M': mobility,
                            'omega': omega,
                            'E': E_modulus,
                            'nu': poisson,
                        }
                        
                        # Create degradation parameters
                        degradation_params = {
                            'k_SEI': k_SEI,
                            'Q_loss_SEI': Q_loss_SEI,
                            'Q_loss_mech': Q_loss_mech,
                            'K_IC': 1.5e6,
                            'sigma_max': 500e6,
                        }
                        
                        # Initialize model
                        st.session_state.model = DegradationPhaseFieldModel(
                            nx=nx, ny=ny, Lx=Lx, Ly=Ly,
                            physics_params=physics_params,
                            degradation_params=degradation_params
                        )
                        
                        # Set dt
                        st.session_state.model.dt = dt
                        
                        # Initialize fields
                        st.session_state.model.initialize_fields(
                            mode=init_mode, 
                            noise_level=noise_level
                        )
                        
                        st.success("✅ Model initialized!")
                        st.rerun()
            
            with col_run2:
                if st.button("▶️ Run Simulation", use_container_width=True, type="primary"):
                    if st.session_state.model:
                        st.session_state.simulation_running = True
                        st.rerun()
        
        # Export section
        st.divider()
        st.markdown("### 💾 Export Options")
        
        if st.button("📥 Export All Data", use_container_width=True):
            if st.session_state.model:
                with st.spinner("Exporting data..."):
                    try:
                        exported_files, zip_path = st.session_state.exporter.export_simulation_data(
                            st.session_state.model,
                            output_dir='./exports'
                        )
                        
                        if zip_path and Path(zip_path).exists():
                            with open(zip_path, 'rb') as f:
                                bytes_data = f.read()
                            
                            st.download_button(
                                label="Download Export Package",
                                data=bytes_data,
                                file_name="lifepo4_simulation_export.zip",
                                mime="application/zip",
                                use_container_width=True
                            )
                    except Exception as e:
                        st.error(f"Export failed: {e}")
        
        # Current status
        st.divider()
        st.markdown("### 📈 Current Status")
        
        if st.session_state.model:
            model = st.session_state.model
            st.metric("Simulation Time", f"{model.time:.1f} s")
            st.metric("Steps", f"{model.step:,}")
            st.metric("Capacity Loss", f"{model.capacity_loss:.1f}%")
            st.metric("Crack Length", f"{model.crack_length*1e9:.1f} nm")
        else:
            st.info("Model not initialized")
    
    # Main content area
    if not st.session_state.model:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class="paper-card">
            <h3>Welcome to the LiFePO₄ Phase Field Degradation Simulator</h3>
            
            <p>This advanced simulation platform enables:</p>
            
            <h4>🎯 Key Features:</h4>
            <ul>
            <li><strong>Phase Field Modeling</strong>: Cahn-Hilliard with electrostatics</li>
            <li><strong>Degradation Mechanisms</strong>: SEI growth, mechanical cracking, calendar aging</li>
            <li><strong>Publication-Quality Outputs</strong>: High-resolution figures, animations, data exports</li>
            <li><strong>Advanced Visualization</strong>: Contour plots, 3D surfaces, vector fields</li>
            <li><strong>Comprehensive Data Export</strong>: CSV, Excel, JSON, HDF5, MATLAB formats</li>
            </ul>
            
            <h4>🚀 Quick Start:</h4>
            <ol>
            <li>Configure simulation parameters in the sidebar</li>
            <li>Click "Initialize Model"</li>
            <li>Run simulation for desired time</li>
            <li>Generate publication-quality figures</li>
            <li>Export data for analysis</li>
            </ol>
            
            <div class="info-box">
            <strong>Publication Ready:</strong> All outputs are formatted for scientific publications
            with proper labeling, color schemes, and high resolution.
            </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Example outputs
        with st.expander("📊 Example Publication Outputs", expanded=True):
            col_ex1, col_ex2, col_ex3 = st.columns(3)
            
            with col_ex1:
                st.markdown("**Multi-panel Figures**")
                st.image("https://via.placeholder.com/300x200/3949ab/ffffff?text=Figure+1", 
                        caption="Comprehensive multi-panel analysis")
            
            with col_ex2:
                st.markdown("**3D Visualizations**")
                st.image("https://via.placeholder.com/300x200/5c6bc0/ffffff?text=3D+Surface", 
                        caption="3D surface plots of concentration")
            
            with col_ex3:
                st.markdown("**Time Evolution**")
                st.image("https://via.placeholder.com/300x200/7e57c2/ffffff?text=Animation", 
                        caption="Time-lapse animations")
        
        return
    
    # Main simulation interface
    model = st.session_state.model
    visualizer = st.session_state.visualizer
    
    # Run simulation if requested
    if st.session_state.simulation_running:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_steps = int(simulation_time / dt)
        completed_steps = 0
        
        while completed_steps < total_steps and st.session_state.simulation_running:
            steps_to_run = min(steps_per_batch, total_steps - completed_steps)
            model.step_time(steps_to_run)
            completed_steps += steps_to_run
            
            progress = completed_steps / total_steps
            progress_bar.progress(progress)
            status_text.text(f"Running... {completed_steps}/{total_steps} steps ({model.time:.1f} s)")
        
        if completed_steps >= total_steps:
            st.session_state.simulation_running = False
            st.success(f"✅ Simulation complete! {total_steps} steps run in {model.time:.1f} seconds")
        
        st.rerun()
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Simulation Dashboard", 
        "🎨 Visualization Gallery", 
        "📊 Data Analysis", 
        "💾 Export Center",
        "📚 Publication Tools"
    ])
    
    with tab1:
        # Simulation dashboard
        st.markdown('<h2 class="section-header">Simulation Dashboard</h2>', unsafe_allow_html=True)
        
        # Quick metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown('<div class="metric-box">Time<br><h3>{:.1f} s</h3></div>'.format(model.time), 
                       unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-box">Mean x<br><h3>{:.3f}</h3></div>'.format(np.mean(model.c)), 
                       unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-box">Capacity<br><h3>{:.1f}%</h3></div>'.format(model.capacity_loss), 
                       unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-box">Cracks<br><h3>{:.1f} nm</h3></div>'.format(model.crack_length*1e9), 
                       unsafe_allow_html=True)
        with col5:
            st.markdown('<div class="metric-box">SEI<br><h3>{:.1f} nm</h3></div>'.format(np.mean(model.SEI_thickness)*1e9), 
                       unsafe_allow_html=True)
        
        # Current fields
        st.markdown('<h3 class="subsection-header">Current Field Distributions</h3>', unsafe_allow_html=True)
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            # Concentration field
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            visualizer.plot_concentration_field(model.c, ax=ax1)
            st.pyplot(fig1)
            plt.close(fig1)
        
        with col_f2:
            # Damage field
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            visualizer.plot_damage_field(model.damage, ax=ax2)
            st.pyplot(fig2)
            plt.close(fig2)
        
        # Additional controls
        with st.expander("⚙️ Additional Controls"):
            col_c1, col_c2, col_c3 = st.columns(3)
            with col_c1:
                if st.button("Reset Simulation", use_container_width=True):
                    model.initialize_fields(mode=init_mode, noise_level=noise_level)
                    st.success("Simulation reset!")
                    st.rerun()
            
            with col_c2:
                if st.button("Stop Simulation", use_container_width=True):
                    st.session_state.simulation_running = False
                    st.info("Simulation stopped")
                    st.rerun()
            
            with col_c3:
                if st.button("Clear History", use_container_width=True):
                    model.history = defaultdict(list)
                    st.success("History cleared!")
                    st.rerun()
    
    with tab2:
        # Visualization gallery
        st.markdown('<h2 class="section-header">Visualization Gallery</h2>', unsafe_allow_html=True)
        
        # Visualization options
        viz_option = st.selectbox(
            "Select Visualization Type",
            ["Multi-panel Figure", "3D Surface Plot", "Time Evolution", "Comparison Plot"],
            index=0
        )
        
        if viz_option == "Multi-panel Figure":
            journal_style = st.selectbox(
                "Journal Style",
                ["nature", "science", "acs", "elsevier", "thesis"],
                index=0
            )
            
            visualizer.set_style(journal_style)
            
            with st.spinner("Generating multi-panel figure..."):
                fig = visualizer.create_multi_panel_figure(
                    model, 
                    title="LiFePO₄ Phase Field Simulation Results"
                )
                st.pyplot(fig)
                
                # Export options
                col_exp1, col_exp2, col_exp3 = st.columns(3)
                with col_exp1:
                    if st.button("📥 Download as PDF", use_container_width=True):
                        saved_files = visualizer.export_figure(fig, "multi_panel_figure", ['pdf'])
                        if saved_files:
                            with open(saved_files[0], 'rb') as f:
                                pdf_data = f.read()
                            st.download_button(
                                label="Download PDF",
                                data=pdf_data,
                                file_name="publication_figure.pdf",
                                mime="application/pdf"
                            )
                
                with col_exp2:
                    if st.button("📥 Download as PNG", use_container_width=True):
                        saved_files = visualizer.export_figure(fig, "multi_panel_figure", ['png'])
                        if saved_files:
                            st.image(saved_files[0], caption="High-resolution PNG")
                
                with col_exp3:
                    if st.button("📥 Download as SVG", use_container_width=True):
                        saved_files = visualizer.export_figure(fig, "multi_panel_figure", ['svg'])
                        if saved_files:
                            with open(saved_files[0], 'rb') as f:
                                svg_data = f.read()
                            st.download_button(
                                label="Download SVG",
                                data=svg_data,
                                file_name="publication_figure.svg",
                                mime="image/svg+xml"
                            )
                
                plt.close(fig)
        
        elif viz_option == "3D Surface Plot":
            field_choice = st.selectbox(
                "Select Field for 3D Visualization",
                ["Concentration", "Damage", "SEI Thickness"]
            )
            
            if field_choice == "Concentration":
                field = model.c
                title = "Li Concentration Field - 3D Visualization"
            elif field_choice == "Damage":
                field = model.damage
                title = "Damage Field - 3D Visualization"
            elif field_choice == "SEI Thickness":
                field = model.SEI_thickness
                title = "SEI Thickness - 3D Visualization"
            
            with st.spinner("Generating 3D surface plot..."):
                fig = visualizer.create_3D_surface_plot(field, title=title)
                st.pyplot(fig)
                plt.close(fig)
        
        elif viz_option == "Time Evolution":
            if model.history and 'time' in model.history and len(model.history['time']) > 1:
                st.markdown("### Time Evolution Plots")
                
                col_ev1, col_ev2 = st.columns(2)
                
                with col_ev1:
                    fig_phase, ax_phase = plt.subplots(figsize=(6, 4))
                    visualizer.plot_phase_evolution(model.history, ax=ax_phase)
                    st.pyplot(fig_phase)
                    plt.close(fig_phase)
                
                with col_ev2:
                    fig_cap, ax_cap = plt.subplots(figsize=(6, 4))
                    visualizer.plot_capacity_fade(model.history, ax=ax_cap)
                    st.pyplot(fig_cap)
                    plt.close(fig_cap)
            else:
                st.info("Run simulation to see time evolution plots.")
    
    with tab3:
        # Data analysis
        st.markdown('<h2 class="section-header">Data Analysis</h2>', unsafe_allow_html=True)
        
        if not model.history or 'time' not in model.history or len(model.history['time']) < 2:
            st.info("Run simulation to see analysis results.")
        else:
            # Time series analysis
            st.markdown("### 📈 Time Series Analysis")
            
            # Create interactive time series plot
            if 'capacity_loss' in model.history and len(model.history['capacity_loss']) > 0:
                time_data = np.array(model.history['time'])
                capacity_data = np.array(model.history['capacity_loss'])
                
                fig_ts, ax_ts = plt.subplots(figsize=(8, 4))
                ax_ts.plot(time_data/3600, capacity_data, 'b-', linewidth=2, label='Capacity Loss')
                ax_ts.set_xlabel('Time (hours)', fontsize=10)
                ax_ts.set_ylabel('Capacity Loss (%)', fontsize=10)
                ax_ts.set_title('Capacity Fade Over Time', fontsize=12, fontweight='bold')
                ax_ts.grid(True, alpha=0.3)
                ax_ts.legend(fontsize=9)
                
                # Add trend line if enough data
                if len(capacity_data) > 5:
                    z = np.polyfit(time_data/3600, capacity_data, 1)
                    p = np.poly1d(z)
                    ax_ts.plot(time_data/3600, p(time_data/3600), 'r--', 
                              linewidth=1.5, label=f'Trend: {z[0]:.3f} %/hour')
                    ax_ts.legend(fontsize=9)
                
                st.pyplot(fig_ts)
                plt.close(fig_ts)
            
            # Statistical analysis
            st.markdown("### 📊 Statistical Summary")
            
            # Create summary table
            summary_data = []
            
            # Time statistics
            if 'time' in model.history and model.history['time']:
                total_time = model.history['time'][-1]
                summary_data.append(('Total Simulation Time', f"{total_time:.1f} s", 
                                   f"{total_time/3600:.2f} hours"))
            
            # Concentration statistics
            if 'mean_c' in model.history and model.history['mean_c']:
                mean_c = np.mean(model.history['mean_c'])
                std_c = np.std(model.history['mean_c'])
                summary_data.append(('Mean Concentration', f"{mean_c:.4f}", "±"))
                summary_data.append(('Concentration Std', f"{std_c:.4f}", ""))
            
            # Capacity statistics
            if 'capacity_loss' in model.history and model.history['capacity_loss']:
                final_capacity_loss = model.history['capacity_loss'][-1]
                summary_data.append(('Final Capacity Loss', f"{final_capacity_loss:.2f}", "%"))
            
            # Crack statistics
            if 'crack_length' in model.history and model.history['crack_length']:
                final_crack_length = model.history['crack_length'][-1]
                summary_data.append(('Crack Length', f"{final_crack_length*1e9:.1f}", "nm"))
            
            # SEI statistics
            if 'SEI_thickness' in model.history and model.history['SEI_thickness']:
                mean_SEI = np.mean(model.history['SEI_thickness'])
                summary_data.append(('Mean SEI Thickness', f"{mean_SEI*1e9:.1f}", "nm"))
            
            # Display table
            if summary_data:
                df_summary = pd.DataFrame(summary_data, columns=['Metric', 'Value', 'Unit'])
                st.dataframe(df_summary, use_container_width=True, hide_index=True)
            
            # Advanced analysis
            with st.expander("🔬 Advanced Statistical Analysis"):
                if 'capacity_loss' in model.history and len(model.history['capacity_loss']) > 10:
                    st.markdown("#### Capacity Fade Analysis")
                    
                    time_data = np.array(model.history['time'])
                    capacity_data = np.array(model.history['capacity_loss'])
                    
                    col_adv1, col_adv2 = st.columns(2)
                    
                    with col_adv1:
                        st.markdown("**Linear Regression**")
                        try:
                            slope, intercept = np.polyfit(time_data/3600, capacity_data, 1)
                            r_squared = np.corrcoef(time_data/3600, capacity_data)[0,1]**2
                            
                            st.write(f"Fade Rate: {slope:.4f} %/hour")
                            st.write(f"Intercept: {intercept:.3f} %")
                            st.write(f"R²: {r_squared:.4f}")
                            
                            # Project lifetime
                            if slope > 0:
                                time_to_80 = (20 - intercept) / slope
                                st.metric("Time to 80% capacity", f"{time_to_80:.1f} hours")
                        except:
                            st.write("Insufficient data for regression")
                    
                    with col_adv2:
                        st.markdown("**Exponential Fit**")
                        try:
                            from scipy.optimize import curve_fit
                            
                            def exp_func(t, a, b, c):
                                return a * (1 - np.exp(-b * t)) + c
                            
                            popt, pcov = curve_fit(exp_func, time_data/3600, capacity_data, 
                                                  p0=[100, 0.01, 0], maxfev=5000)
                            
                            st.write(f"Time constant: {1/popt[1]:.1f} hours")
                            st.write(f"Asymptote: {popt[0]:.1f}%")
                        except Exception as e:
                            st.write(f"Exponential fit not available: {e}")
    
    with tab4:
        # Export center
        st.markdown('<h2 class="section-header">Export Center</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="paper-card">
        <h3>Comprehensive Data Export</h3>
        <p>Export simulation data in multiple formats suitable for:</p>
        <ul>
        <li>📄 <strong>Publications</strong>: High-resolution figures, formatted tables</li>
        <li>📊 <strong>Analysis</strong>: Raw data in CSV, Excel, JSON formats</li>
        <li>🔬 <strong>Collaboration</strong>: MATLAB, HDF5 for cross-platform use</li>
        <li>💾 <strong>Archiving</strong>: Complete simulation state for reproducibility</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Export options
        col_exp_opt1, col_exp_opt2, col_exp_opt3 = st.columns(3)
        
        with col_exp_opt1:
            st.markdown("**📈 Time Series Data**")
            if st.button("Export Time Series", use_container_width=True):
                if model.history and 'time' in model.history:
                    time_series = st.session_state.exporter._export_time_series(model)
                    df = pd.DataFrame(time_series)
                    
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="time_series.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        with col_exp_opt2:
            st.markdown("**🖼️ Field Data**")
            if st.button("Export Current Fields", use_container_width=True):
                # Export current fields as CSV
                field_data = {
                    'concentration': model.c.flatten(),
                    'damage': model.damage.flatten(),
                    'SEI_thickness': model.SEI_thickness.flatten()
                }
                
                # Create coordinate arrays
                nx, ny = model.c.shape
                x_coords = np.tile(np.arange(nx), ny)
                y_coords = np.repeat(np.arange(ny), nx)
                
                df = pd.DataFrame({
                    'x': x_coords,
                    'y': y_coords,
                    'concentration': field_data['concentration'],
                    'damage': field_data['damage'],
                    'SEI_thickness': field_data['SEI_thickness']
                })
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="field_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col_exp_opt3:
            st.markdown("**📊 Statistics**")
            if st.button("Export Statistics", use_container_width=True):
                stats = st.session_state.exporter._export_statistics(model)
                
                # Flatten nested dictionary
                flat_stats = {}
                for category, values in stats.items():
                    for key, value in values.items():
                        if isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                flat_stats[f"{category}_{key}_{subkey}"] = subvalue
                        else:
                            flat_stats[f"{category}_{key}"] = value
                
                df = pd.DataFrame([flat_stats])
                csv = df.to_csv(index=False)
                
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="statistics.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        # Complete export
        st.divider()
        st.markdown("### 💾 Complete Export Package")
        
        if st.button("Generate Complete Export Package", type="primary", use_container_width=True):
            with st.spinner("Preparing export package..."):
                try:
                    exported_files, zip_path = st.session_state.exporter.export_simulation_data(
                        model,
                        output_dir='./exports'
                    )
                    
                    if zip_path and Path(zip_path).exists():
                        # Read zip file
                        with open(zip_path, 'rb') as f:
                            zip_data = f.read()
                        
                        # Display export summary
                        st.success(f"✅ Export package created with {len(exported_files)} files")
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Complete Package (ZIP)",
                            data=zip_data,
                            file_name="lifepo4_complete_export.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
                        
                        # Show file list
                        with st.expander("📋 View Export Contents"):
                            for file in exported_files:
                                file_path = Path(file)
                                if file_path.exists():
                                    file_size = file_path.stat().st_size / 1024
                                    st.write(f"- `{file_path.name}` ({file_size:.1f} KB)")
                    else:
                        st.error("Failed to create export package")
                except Exception as e:
                    st.error(f"Export failed: {e}")
    
    with tab5:
        # Publication tools
        st.markdown('<h2 class="section-header">Publication Tools</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="paper-card">
        <h3>Publication-Ready Outputs</h3>
        <p>Generate materials directly suitable for scientific publications:</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Journal-specific templates
        st.markdown("### 📝 Journal Templates")
        
        journal_choice = st.selectbox(
            "Select Journal Format",
            ["Nature", "Science", "ACS Nano", "Advanced Materials", 
             "Journal of Power Sources", "Custom"],
            index=0
        )
        
        if journal_choice != "Custom":
            # Apply journal style
            visualizer.set_style(journal_choice.lower())
            st.success(f"Applied {journal_choice} formatting style")
        
        # Generate publication figure
        st.markdown("### 🎨 Generate Publication Figure")
        
        col_pub1, col_pub2 = st.columns(2)
        
        with col_pub1:
            figure_type = st.selectbox(
                "Figure Type",
                ["Multi-panel", "Single panel", "Time series", "Comparison"],
                index=0
            )
        
        with col_pub2:
            output_format = st.multiselect(
                "Output Formats",
                ["PDF", "PNG", "SVG", "EPS", "TIFF"],
                default=["PDF", "PNG"]
            )
        
        if st.button("Generate Publication Figure", use_container_width=True):
            with st.spinner("Creating publication-quality figure..."):
                # Create figure based on selection
                if figure_type == "Multi-panel":
                    fig = visualizer.create_multi_panel_figure(
                        model, 
                        title="LiFePO₄ Phase Field Simulation"
                    )
                elif figure_type == "Single panel":
                    fig, ax = plt.subplots(figsize=(6, 4))
                    visualizer.plot_concentration_field(model.c, ax=ax)
                    ax.set_title("Li Concentration Field", fontweight='bold')
                elif figure_type == "Time series":
                    fig, ax = plt.subplots(figsize=(8, 4))
                    if model.history and 'time' in model.history and len(model.history['time']) > 1:
                        visualizer.plot_degradation_summary(model, ax=ax)
                        ax.set_title("Degradation Evolution", fontweight='bold')
                    else:
                        ax.text(0.5, 0.5, 'Run simulation to see time series', 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.set_title("Degradation Evolution", fontweight='bold')
                else:  # Comparison
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                    visualizer.plot_concentration_field(model.c, ax=ax1)
                    visualizer.plot_damage_field(model.damage, ax=ax2)
                    ax1.set_title("Concentration Field", fontweight='bold')
                    ax2.set_title("Damage Field", fontweight='bold')
                
                # Display figure
                st.pyplot(fig)
                
                # Export options
                for fmt in output_format:
                    if fmt == "PDF":
                        try:
                            fig.savefig("publication_figure.pdf", dpi=600, bbox_inches='tight')
                            with open("publication_figure.pdf", 'rb') as f:
                                pdf_data = f.read()
                            
                            st.download_button(
                                label="📥 Download PDF",
                                data=pdf_data,
                                file_name="publication_figure.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.warning(f"Could not save PDF: {e}")
                    
                    elif fmt == "PNG":
                        try:
                            fig.savefig("publication_figure.png", dpi=600, bbox_inches='tight')
                            st.image("publication_figure.png", caption="High-resolution PNG")
                        except Exception as e:
                            st.warning(f"Could not save PNG: {e}")
                
                plt.close(fig)
        
        # Citation generator
        st.markdown("### 📚 Citation Generator")
        
        col_cite1, col_cite2, col_cite3 = st.columns(3)
        
        with col_cite1:
            author = st.text_input("First Author", "Smith, J.")
        
        with col_cite2:
            year = st.number_input("Year", 2020, 2030, 2024)
        
        with col_cite3:
            journal = st.text_input("Journal", "Nature Energy")
        
        citation_style = st.selectbox(
            "Citation Style",
            ["APA", "MLA", "Chicago", "Harvard", "IEEE"],
            index=0
        )
        
        if st.button("Generate Citation"):
            if citation_style == "APA":
                citation = f"{author} ({year}). LiFePO₄ phase field degradation modeling. {journal}."
            elif citation_style == "MLA":
                citation = f"{author}. \"LiFePO₄ phase field degradation modeling.\" {journal} {year}."
            elif citation_style == "Chicago":
                citation = f"{author}. {year}. \"LiFePO₄ phase field degradation modeling.\" {journal}."
            elif citation_style == "Harvard":
                citation = f"{author} ({year}) LiFePO₄ phase field degradation modeling, {journal}."
            else:  # IEEE
                citation = f"J. Smith, \"LiFePO₄ phase field degradation modeling,\" {journal}, 2024."
            
            st.code(citation, language=None)
            
            # Download button
            st.download_button(
                label="📋 Copy Citation",
                data=citation,
                file_name="citation.txt",
                mime="text/plain",
                use_container_width=True
            )

# =====================================================
# 7. UTILITY FUNCTIONS AND SETUP
# =====================================================

def create_requirements_file():
    """Create requirements.txt for the application"""
    requirements = """streamlit>=1.28.0
numpy>=1.24.0
torch>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.17.0
pandas>=2.0.0
scipy>=1.11.0
scikit-learn>=1.3.0
h5py>=3.9.0
openpyxl>=3.1.0
Pillow>=10.0.0
tqdm>=4.65.0
typing-extensions>=4.8.0
"""
    return requirements

def check_dependencies():
    """Check if all required dependencies are installed"""
    missing_deps = []
    
    required_packages = [
        'streamlit', 'numpy', 'torch', 'matplotlib', 'seaborn',
        'plotly', 'pandas', 'scipy', 'sklearn', 'h5py',
        'openpyxl', 'PIL', 'tqdm'
    ]
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'PIL':
                import PIL
            else:
                __import__(package)
        except ImportError:
            missing_deps.append(package)
    
    return missing_deps

# =====================================================
# 8. MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    # Check dependencies
    missing = check_dependencies()
    if missing:
        st.error(f"Missing required packages: {', '.join(missing)}")
        st.info("Please install with: pip install " + " ".join(missing))
    else:
        # Run the application
        try:
            main()
        except Exception as e:
            st.error(f"Application error: {e}")
            st.info("Please check the console for details")
            
            # Display error details in expander
            with st.expander("Error Details"):
                import traceback
                st.code(traceback.format_exc())
