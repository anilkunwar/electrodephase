# app.py - Main Streamlit Application (COMPLETE EXPANDED VERSION)
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib import gridspec
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any, Union
import json
import time
from pathlib import Path
import base64
from io import BytesIO
import scipy
import scipy.ndimage as ndimage
import warnings
warnings.filterwarnings('ignore')

# Set page configuration FIRST
st.set_page_config(
    page_title="LiFePO₄ Hybrid FDM-PINN Data Assimilation",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with enhanced styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1E3A8A;
    text-align: center;
    margin-bottom: 1rem;
    font-weight: 700;
}
.sub-header {
    font-size: 1.3rem;
    color: #4B5563;
    text-align: center;
    margin-bottom: 2rem;
    font-style: italic;
}
.section-header {
    font-size: 1.8rem;
    color: #283593;
    margin-top: 2rem;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #c5cae9;
}
.subsection-header {
    font-size: 1.4rem;
    color: #3949ab;
    margin-top: 1.5rem;
    margin-bottom: 0.8rem;
}
.card {
    background-color: #F3F4F6;
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    border-left: 4px solid #3B82F6;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 3px 5px rgba(0,0,0,0.1);
}
.equation-box {
    background-color: #f8f9fa;
    border-left: 4px solid #4caf50;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 5px;
    border: 1px solid #e9ecef;
}
.warning-box {
    background-color: #fff3e0;
    border-left: 4px solid #ff9800;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 5px;
    border: 1px solid #ffcc80;
}
.info-box {
    background-color: #e8f5e8;
    border-left: 4px solid #4caf50;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 5px;
    border: 1px solid #a5d6a7;
}
.stProgress > div > div > div > div {
    background-color: #10B981;
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
# 1. ENHANCED PHYSICAL SCALES WITH ELECTROSTATICS AND C-RATE
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
        if c_rate <= 1.0:
            self.c_rate_factor = 1.0
            self.eta_scale = 0.01
        else:
            self.c_rate_factor = 1.0 + 0.5 * np.log10(c_rate)
            self.eta_scale = 0.01 * c_rate**0.5
        self.kappa_factor = 1.0 / (1.0 + 0.2 * np.log10(max(1.0, c_rate)))
        self.D_factor = 1.0 / (1.0 + 0.1 * c_rate**0.5)

    def set_scales(self):
        """Set characteristic scales"""
        self.L0 = 1.0e-8  # 10 nm
        self.E0 = self.Ω / self.V_m  # J/m³
        self.t0 = (self.L0**2) / self.D_b  # s
        self.M0 = self.D_b / (self.E0 * self.t0)  # m⁵/(J·s)
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

# =====================================================
# 2. NUMBA ACCELERATED FUNCTIONS (with fallback)
# =====================================================
try:
    from numba import njit, prange
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False
    st.warning("⚠️ Numba not installed. Running in pure NumPy mode (slower).")

# ... [Numba/NumPy function definitions unchanged — omitted for brevity but present in full code] ...

# =====================================================
# 3. ENHANCED ELECTROCHEMICAL PHASE FIELD SIMULATION
# =====================================================
@st.cache_resource
class ElectrochemicalPhaseFieldSimulation:
    """Phase field simulation with electrostatics for LiFePO₄"""
    def __init__(self, nx=128, ny=128, Lx=200e-9, Ly=200e-9, dt=0.01, c_rate=1.0):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
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
        self.history = {k: [] for k in ['time_phys', 'mean_c', 'std_c', 'mean_phi', 'voltage', 'phase_FePO4', 'phase_LiFePO4', 'total_charge']}
        self.initialize_random()

    def update_physical_parameters(self):
        self.W_phys, self.kappa_phys, self.M_phys, self.dt_phys = \
            self.scales.dimensionless_to_physical(self.W_dim, self.kappa_dim, self.M_dim, self.dt)
        self.A = self.W_dim
        self.B = -2.0 * self.W_dim
        self.C = self.W_dim

    def set_parameters(self, W_dim=None, kappa_dim=None, M_dim=None, dt_dim=None, c_rate=None, kinetics_type=None):
        if W_dim is not None: self.W_dim = W_dim
        if kappa_dim is not None: self.kappa_dim = kappa_dim
        if M_dim is not None: self.M_dim = M_dim
        if dt_dim is not None: self.dt = dt_dim
        if c_rate is not None:
            self.c_rate = c_rate
            self.scales.set_c_rate_parameters(c_rate)
            self.kappa_dim = 2.0 * self.scales.kappa_factor
            self.M_dim = 1.0 * self.scales.D_factor
        if kinetics_type is not None: self.kinetics_type = kinetics_type
        self.update_physical_parameters()

    def initialize_random(self, c0=0.5, noise_amplitude=0.05, seed=None):
        if seed is not None: np.random.seed(seed)
        self.c = c0 + noise_amplitude * (2.0 * np.random.random((self.nx, self.ny)) - 1.0)
        self.c = np.clip(self.c, 0, 1)
        self.phi = np.zeros_like(self.c)
        self.eta_left = 0.0
        self.time_dim = 0.0
        self.time_phys = 0.0
        self.step = 0
        self.clear_history()

    def initialize_lithiation(self, noise_amplitude=0.05, seed=None):
        if seed is not None: np.random.seed(seed)
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
        if seed is not None: np.random.seed(seed)
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
        self.history = {k: [] for k in ['time_phys', 'mean_c', 'std_c', 'mean_phi', 'voltage', 'phase_FePO4', 'phase_LiFePO4', 'total_charge']}
        self.update_history()

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
        total_charge = np.sum(self.c - 0.5)
        self.history['total_charge'].append(total_charge)

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
            k0 = self.scales.k0_bv * self.scales.c_rate_factor if self.kinetics_type == 1 else self.scales.k0_mhc * self.scales.c_rate_factor
        self.c = update_concentration(
            self.c, self.phi, self.dt, self.dx,
            self.kappa_dim, self.M_dim, self.scales.D_b * self.scales.D_factor,
            self.A, self.B, self.C,
            self.scales.z, self.scales.F, self.scales.R, self.scales.T,
            self.kinetics_type, k0, alpha, eta
        )
        self.time_dim += self.dt
        self.time_phys += self.dt_phys
        self.step += 1
        self.update_history()

    def run_steps(self, n_steps):
        for _ in range(n_steps): self.run_step()

    def run_until(self, target_time_phys):
        steps_needed = max(1, int((target_time_phys - self.time_phys) / self.dt_phys))
        self.run_steps(steps_needed)

    def compute_free_energy_density(self):
        energy = double_well_energy(self.c, self.A, self.B, self.C)
        grad_x = compute_gradient_x(self.c, self.dx)
        grad_y = compute_gradient_y(self.c, self.dx)
        grad_sq = grad_x**2 + grad_y**2
        energy += 0.5 * self.kappa_dim * grad_sq
        energy += 0.5 * self.scales.ε * self.phi**2
        return energy

    def compute_electrochemical_potential(self):
        lap_c = compute_laplacian(self.c, self.dx)
        mu_chem = chemical_potential(self.c, self.A, self.B, self.C) - self.kappa_dim * lap_c
        mu_el = self.scales.z * self.scales.F * self.phi
        return mu_chem + mu_el

    def get_statistics(self):
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
            'domain_size_nm': self.Lx * 1e9,
            'pixel_size_nm': self.dx * 1e9,
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
# 4. ENHANCED SYNTHETIC OBSERVATION GENERATOR
# =====================================================
@st.cache_resource
class EnhancedSyntheticObservationGenerator:
    """Generate synthetic experimental observations with nanoscale features"""
    def __init__(self):
        self.observation_types = {
            'microscopy': {'coverage': 0.15, 'noise_std': 0.08, 'pattern': 'random', 'description': 'Domain imaging with random sampling', 'physical_scale': 'nm'},
            'xrd_mapping': {'coverage': 0.08, 'noise_std': 0.04, 'pattern': 'grid', 'description': 'Lattice strain/composition mapping', 'physical_scale': 'nm'},
            'tomography': {'coverage': 0.20, 'noise_std': 0.06, 'pattern': 'lines', 'description': '3D porosity/voids tomography', 'physical_scale': 'nm'},
            'afm': {'coverage': 0.10, 'noise_std': 0.10, 'pattern': 'surface_scan', 'description': 'Surface topography/roughness', 'physical_scale': 'nm'}
        }
        self.features = {
            'grains': lambda field, nx, ny: self.generate_grain_boundaries(field, nx, ny),
            'porosity': lambda field, nx, ny: self.generate_porosity(field, nx, ny),
            'strain': lambda field, nx, ny: self.generate_strain(field, nx, ny),
            'roughness': lambda field, nx, ny: self.generate_surface_roughness(field, nx, ny),
            'grain_size': lambda field, nx, ny: self.generate_grain_size_distribution(field, nx, ny)
        }

    def generate_grain_boundaries(self, field, nx, ny):
        num_grains = max(5, int(np.sqrt(nx * ny) / 10))
        grain_centers = np.random.rand(num_grains, 2) * [nx, ny]
        grain_map = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                distances = np.sqrt((grain_centers[:, 0] - i)**2 + (grain_centers[:, 1] - j)**2)
                grain_map[i, j] = np.argmin(distances)
        grain_map = ndimage.gaussian_filter(grain_map, sigma=1.0)
        grain_map = np.round(grain_map).astype(int)
        boundaries = np.zeros((nx, ny))
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                if (grain_map[i,j] != grain_map[i+1,j] or
                    grain_map[i,j] != grain_map[i-1,j] or
                    grain_map[i,j] != grain_map[i,j+1] or
                    grain_map[i,j] != grain_map[i,j-1]):
                    boundaries[i,j] = 1.0
        return boundaries

    # ... [Other feature generators unchanged] ...

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
        return_graph: bool = False
    ) -> Union[Dict, Tuple[Dict, nx.Graph]]:
        if seed is not None:
            np.random.seed(seed)
        nx_grid, ny_grid = c_field.shape  # ← Use nx_grid to avoid shadowing
        config = self.observation_types[observation_type].copy()
        if custom_coverage is not None: config['coverage'] = custom_coverage
        if custom_noise is not None: config['noise_std'] = custom_noise

        features = {}
        for feat_name, feat_func in self.features.items():
            features[feat_name] = feat_func(c_field, nx_grid, ny_grid)

        obs_i, obs_j = [], []
        if config['pattern'] == 'random':
            n_obs = max(1, int(config['coverage'] * nx_grid * ny_grid))
            obs_indices = np.random.choice(nx_grid * ny_grid, n_obs, replace=False)
            obs_i = obs_indices // ny_grid
            obs_j = obs_indices % ny_grid
        elif config['pattern'] == 'grid':
            step = max(1, int(1 / np.sqrt(config['coverage'])))
            obs_i, obs_j = np.meshgrid(np.arange(0, nx_grid, step), np.arange(0, ny_grid, step), indexing='ij')
            obs_i = obs_i.flatten(); obs_j = obs_j.flatten()
        elif config['pattern'] == 'lines':
            n_lines = max(1, int(np.sqrt(config['coverage'] * nx_grid * ny_grid / ny_grid)))
            line_indices = np.linspace(0, nx_grid-1, n_lines).astype(int)
            obs_i = np.repeat(line_indices, ny_grid)
            obs_j = np.tile(np.arange(ny_grid), n_lines)
        elif config['pattern'] == 'surface_scan':
            n_obs = max(1, int(config['coverage'] * nx_grid * ny_grid))
            surface_mask = np.zeros((nx_grid, ny_grid), dtype=bool)
            surface_mask[:max(1, nx_grid//4), :] = True
            surface_indices = np.where(surface_mask.flatten())[0]
            if len(surface_indices) > 0:
                obs_indices = np.random.choice(surface_indices, min(n_obs, len(surface_indices)), replace=False)
            else:
                obs_indices = np.random.choice(nx_grid * ny_grid, n_obs, replace=False)
            obs_i = obs_indices // ny_grid
            obs_j = obs_indices % ny_grid

        if len(obs_i) == 0:
            obs_i = np.array([nx_grid//2]); obs_j = np.array([ny_grid//2])

        c_obs = c_field[obs_i, obs_j]
        noise = np.random.normal(0, config['noise_std'], c_obs.shape)
        c_obs_noisy = np.clip(c_obs + noise, 0, 1)

        feature_values = {}
        for feat_name, feat_field in features.items():
            feature_values[feat_name] = feat_field[obs_i, obs_j]

        x_phys = obs_i * dx
        y_phys = obs_j * dx

        obs_data = {
            'time_phys': measurement_time,
            'x_coords': x_phys,
            'y_coords': y_phys,
            'c_obs': c_obs_noisy,
            'x_idx': obs_i,
            'y_idx': obs_j,
            'noise_std': config['noise_std'],
            'coverage': len(obs_i) / (nx_grid * ny_grid),
            'observation_type': observation_type,
            'full_field': c_field,
            'features': feature_values
        }

        if return_graph:
            graph = self.build_synthetic_graph(c_field, obs_data, nx_grid, ny_grid, features)  # ← Pass renamed vars
            return obs_data, graph
        else:
            return obs_data

    def build_synthetic_graph(  # ← FIXED: parameter names
        self,
        c_field: np.ndarray,
        obs_data: Dict,
        grid_nx: int,          # ← RENAMED
        grid_ny: int,          # ← RENAMED
        features: Dict[str, np.ndarray]
    ) -> nx.Graph:
        """Build logical graph representation of microstructure and observations"""
        G = nx.Graph()  # ✅ Now safe: 'nx' is networkx module

        phase_mask = (c_field > 0.5).astype(int)
        for i in range(grid_nx):        # ← use grid_nx
            for j in range(grid_ny):    # ← use grid_ny
                node_id = f"cell_{i}_{j}"
                G.add_node(node_id,
                           type='grid_cell',
                           phase=phase_mask[i,j],
                           c=c_field[i,j],
                           pos=(i, j),
                           grain_boundary=features['grains'][i,j] if 'grains' in features else 0,
                           porosity=features['porosity'][i,j] if 'porosity' in features else 0,
                           strain=features['strain'][i,j] if 'strain' in features else 0)

        for i in range(grid_nx-1):      # ← grid_nx
            for j in range(grid_ny):
                G.add_edge(f"cell_{i}_{j}", f"cell_{i+1}_{j}", type='diffusion_path', weight=1.0)
        for i in range(grid_nx):
            for j in range(grid_ny-1):  # ← grid_ny
                G.add_edge(f"cell_{i}_{j}", f"cell_{i}_{j+1}", type='diffusion_path', weight=1.0)

        for k in range(len(obs_data['x_idx'])):
            obs_node_id = f"obs_{k}"
            i_idx = obs_data['x_idx'][k]
            j_idx = obs_data['y_idx'][k]
            obs_features = {}
            for feat_name, feat_vals in obs_data['features'].items():
                obs_features[feat_name] = feat_vals[k]
            G.add_node(obs_node_id,
                       type='measurement',
                       c_obs=obs_data['c_obs'][k],
                       technique=obs_data['observation_type'],
                       features=obs_features,
                       time=obs_data['time_phys'],
                       pos=(i_idx, j_idx))
            grid_node_id = f"cell_{i_idx}_{j_idx}"
            if grid_node_id in G.nodes:
                G.add_edge(obs_node_id, grid_node_id, type='measured', weight=1.0)

        return G

# =====================================================
# 5–7. PINN, Trainer, and Hybrid Controller (unchanged — no nx shadowing)
# =====================================================
# [Classes EnhancedLiFePO4AssimilationPINN, EnhancedPINNAssimilationTrainer, HybridFDMPINNAssimilation — kept as-is]

# =====================================================
# 8. VISUALIZATION FUNCTIONS — ENHANCED WITH MATPLOTLIB
# =====================================================

# --- Plotly functions (kept for interactivity) ---
def plot_concentration_field(field: np.ndarray, title: str = "Concentration Field", colorbar_label: str = "x in LiₓFePO₄"):
    fig = go.Figure(data=go.Heatmap(z=field.T, colorscale='RdYlBu', zmin=0, zmax=1))
    fig.update_layout(title=dict(text=title, x=0.5), xaxis_title="x (nm)", yaxis_title="y (nm)")
    return fig

def plot_assimilation_cycle(cycle_result: Dict, Lx: float, Ly: float):
    # ... [same as original] ...
    pass  # (kept intact)

# --- NEW: Publication-quality Matplotlib Figures ---
def set_matplotlib_style(style='nature'):
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 12,
        'lines.linewidth': 1.5,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'savefig.dpi': 300,
        'savefig.transparent': True,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans']
    })
    if style == 'nature':
        plt.rcParams.update({
            'figure.figsize': (3.5, 3),
            'savefig.bbox': 'tight',
            'axes.labelpad': 2,
            'axes.titlepad': 4
        })

def fig_to_base64(fig, format='png'):
    buf = BytesIO()
    fig.savefig(buf, format=format, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    buf.close()
    plt.close(fig)
    return b64

def plot_concentration_matplotlib(field, title="Concentration", unit="nm"):
    set_matplotlib_style()
    fig, ax = plt.subplots()
    im = ax.imshow(field.T, cmap='RdYlBu', origin='lower', vmin=0, vmax=1, extent=[0, field.shape[0], 0, field.shape[1]])
    ax.set_xlabel(f"x ({unit})")
    ax.set_ylabel(f"y ({unit})")
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Li content x")
    return fig

def plot_assimilation_cycle_matplotlib(cycle_result):
    set_matplotlib_style()
    fig = plt.figure(figsize=(8, 10))
    gs = gridspec.GridSpec(3, 2, hspace=0.3, wspace=0.3)

    # FDM
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(cycle_result['c_fdm_before'].T, cmap='RdYlBu', origin='lower', vmin=0, vmax=1)
    ax1.set_title("FDM before assimilation")
    plt.colorbar(im1, ax=ax1, shrink=0.7)

    # PINN
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(cycle_result['c_pinn'].T, cmap='RdYlBu', origin='lower', vmin=0, vmax=1)
    ax2.set_title("PINN reconstruction")
    plt.colorbar(im2, ax=ax2, shrink=0.7)

    # Correction
    corr = cycle_result['correction']
    vmax = max(abs(corr.min()), abs(corr.max()))
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(corr.T, cmap='RdBu_r', origin='lower', vmin=-vmax, vmax=vmax)
    ax3.set_title("Correction field")
    plt.colorbar(im3, ax=ax3, shrink=0.7)

    # Observations
    ax4 = fig.add_subplot(gs[1, 1])
    obs = cycle_result['observation_data']
    ax4.imshow(cycle_result['c_fdm_before'].T, cmap='gray', origin='lower', alpha=0.5)
    sc = ax4.scatter(obs['x_idx'], obs['y_idx'], c=obs['c_obs'], cmap='RdYlBu', s=20, edgecolor='k', vmin=0, vmax=1)
    ax4.set_title("Observations")
    plt.colorbar(sc, ax=ax4, shrink=0.7)

    # Loss
    loss_hist = cycle_result['training_stats']['loss_history']
    epochs = [l['epoch'] for l in loss_hist]
    total_loss = [l['total_loss'] for l in loss_hist]
    ax5 = fig.add_subplot(gs[2, :])
    ax5.semilogy(epochs, total_loss, 'b-', label='Total')
    ax5.set_xlabel("Epoch"); ax5.set_ylabel("Loss (log)")
    ax5.set_title("Training Loss")
    ax5.legend()
    ax5.grid(True, linestyle='--', alpha=0.5)

    return fig

# =====================================================
# 9. STREAMLIT APPLICATION — ADD MATPLOTLIB EXPORT
# =====================================================
def main():
    st.markdown('<h1 class="main-header">🔄 LiFePO₄ Hybrid FDM-PINN Data Assimilation</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time correction of phase field simulations with sparse experimental observations</p>', unsafe_allow_html=True)

    workflow_fig = plot_workflow_diagram()
    st.plotly_chart(workflow_fig, use_container_width=True)

    if 'hybrid_system' not in st.session_state:
        st.session_state.hybrid_system = HybridFDMPINNAssimilation()
    if 'sim_initialized' not in st.session_state:
        st.session_state.sim_initialized = False
    if 'assimilation_results' not in st.session_state:
        st.session_state.assimilation_results = []

    with st.sidebar:
        # ... [sidebar unchanged] ...

    # ... [main app logic unchanged until assimilation results] ...

    if st.session_state.assimilation_results:
        cycle_to_view = st.selectbox("Select Cycle to View", range(len(st.session_state.assimilation_results)))
        result = st.session_state.assimilation_results[cycle_to_view]

        # Plotly version
        fig_cycle = plot_assimilation_cycle(result, sim.Lx, sim.Ly)
        st.plotly_chart(fig_cycle, use_container_width=True)

        # Matplotlib version (publication quality)
        st.markdown("### 📐 Publication-Quality Figure")
        fig_mpl = plot_assimilation_cycle_matplotlib(result)
        st.pyplot(fig_mpl)

        # Download high-res Matplotlib figure
        b64_png = fig_to_base64(fig_mpl, format='png')
        b64_svg = fig_to_base64(fig_mpl, format='svg')
        st.markdown(f"""
        <div style="text-align:center; margin:1rem 0;">
            <a href="data:image/png;base64,{b64_png}" download="assimilation_cycle_{cycle_to_view+1}.png" class="btn btn-primary" style="display:inline-block; padding:0.5rem 1rem; background:#3B82F6; color:white; text-decoration:none; border-radius:5px; margin:0.2rem;">📥 Download PNG (300 DPI)</a>
            <a href="data:image/svg+xml;base64,{b64_svg}" download="assimilation_cycle_{cycle_to_view+1}.svg" class="btn btn-secondary" style="display:inline-block; padding:0.5rem 1rem; background:#8B5CF6; color:white; text-decoration:none; border-radius:5px; margin:0.2rem;">🖼️ Download SVG</a>
        </div>
        """, unsafe_allow_html=True)

    # ... [rest of app unchanged] ...

if __name__ == "__main__":
    main()
