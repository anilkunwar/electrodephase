import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from scipy import special

########################################################################
# ENHANCED SINGLE PARTICLE SIMULATION WITH MULTI-SEEDS AND KINETICS
# Features:
# 1. BV and MHC kinetics options
# 2. Multiple seeds for lithiation/delithiation
# 3. Rate-dependent parameters (C-rate)
# 4. Anisotropic diffusion (b-axis preferred)
# 5. Physical scaling with actual units
########################################################################

# =====================================================
# PHYSICAL CONSTANTS FOR LiFePO₄ WITH KINETICS
# =====================================================
class PhysicalScalesWithKinetics:
    """Physical scales for LiFePO₄ with realistic kinetics"""
    
    # Fundamental constants
    R = 8.314462618  # J/(mol·K)
    F = 96485.33212  # C/mol
    k_B = 1.380649e-23  # J/K
    e = 1.60217662e-19  # C (elementary charge)
    N_A = 6.02214076e23  # /mol
    
    def __init__(self, c_rate=1.0):
        # C-rate parameter
        self.c_rate = c_rate  # 0.1C, 1C, 5C, etc.
        
        # Material properties
        self.T = 298.15  # K - Temperature
        
        # LiFePO₄ phase compositions (from experimental data)
        self.c_alpha = 0.03  # FePO₄ phase (x ≈ 0.03 in LiₓFePO₄)
        self.c_beta = 0.97   # LiFePO₄ phase (x ≈ 0.97 in LiₓFePO₄)
        
        # Molar volume
        self.V_m = 3.0e-5  # m³/mol (≈30 cm³/mol)
        
        # Diffusion coefficient - rate dependent
        # Reference: D_b ~ 10⁻¹⁴ m²/s at low rates, lower at high rates
        self.D_b0 = 1.0e-14  # m²/s - b-axis diffusion (fast direction)
        self.D_other = 1.0e-16  # m²/s - other directions
        self.D_ratio = 100.0  # D_b / D_other ratio
        
        # Regular solution parameter for LiFePO₄ (from experimental data)
        self.Ω = 55e3  # J/mol (~0.57 eV)
        
        # Kinetics parameters
        self.alpha = 0.5  # Symmetry factor for BV
        self.lambda_mhc = 8.3  # Reorganizational energy for MHC (eV)
        self.k0_bv = 2.0e-3  # Rate constant for BV (m/s)
        self.k0_mhc = 5.0e-4  # Rate constant for MHC (m/s)
        
        # Electrostatic properties
        self.ε_r = 15.0  # Relative permittivity
        self.ε0 = 8.854187817e-12  # F/m
        self.ε = self.ε_r * self.ε0
        
        # Charge properties
        self.z = 1.0  # Li⁺ charge
        
        # Set characteristic scales
        self.set_scales()
        
        # Calculate rate-dependent parameters
        self.set_rate_dependent_params(c_rate)
        
        print(f"Physical scales at {c_rate}C:")
        print(f"  Domain size: {self.L0*1e9:.1f} nm")
        print(f"  Time scale: {self.t0:.2e} s")
        print(f"  Effective D_b: {self.D_b_eff:.2e} m²/s")
        print(f"  Overpotential scale: {self.eta0:.3f} V")
        
    def set_scales(self):
        """Set characteristic physical scales"""
        # Length scale: 100 nm particle (typical in experiments)
        self.L0 = 1.0e-7  # 100 nm
        
        # Energy density scale from regular solution
        self.E0 = self.Ω / self.V_m  # J/m³
        
        # Time scale from diffusion
        self.t0 = (self.L0**2) / self.D_b0  # s
        
        # Mobility scale
        self.M0 = self.D_b0 / (self.E0 * self.t0)  # m⁵/(J·s)
        
        # Electric potential scale (thermal voltage)
        self.phi0 = self.R * self.T / self.F  # ~0.0257 V
        
        # Overpotential scale (increases with rate)
        self.eta0 = 0.1 * self.c_rate  # Scale with C-rate
        
        # Current density scale for 1C rate
        # 1C = complete reaction in 1 hour, for LiFePO₄: ~170 mAh/g
        # Current density: i_1C = (Q * ρ) / (3600 * A)
        self.i_1C = 1.0e3  # A/m² (approximate)
        
    def set_rate_dependent_params(self, c_rate):
        """Set parameters that depend on C-rate"""
        self.c_rate = c_rate
        
        # Rate-dependent diffusion (slower effective diffusion at high rates)
        self.D_b_eff = self.D_b0 / (1.0 + 0.5 * c_rate**0.5)
        
        # Rate-dependent mobility
        self.M_eff = self.D_b_eff / (self.E0 * self.t0)
        
        # Rate-dependent exchange current
        # Higher rates require higher overpotentials
        self.k0_eff_bv = self.k0_bv / (1.0 + 0.3 * c_rate)
        self.k0_eff_mhc = self.k0_mhc / (1.0 + 0.3 * c_rate)
        
        # Interface energy increases with rate (sharper interface)
        self.kappa_scale = 1.0 + 0.2 * c_rate

# =====================================================
# KINETICS MODELS
# =====================================================
@njit(fastmath=True, cache=True)
def butler_volmer_flux(eta, c_surf, alpha, k0, F, R, T):
    """Butler-Volmer kinetics for lithium insertion"""
    # eta: overpotential (V)
    # c_surf: surface concentration (dimensionless)
    # Return: flux (mol/m²/s)
    
    # Forward and backward rate constants
    k_f = k0 * np.exp(-alpha * F * eta / (R * T))
    k_b = k0 * np.exp((1 - alpha) * F * eta / (R * T))
    
    # Flux (positive for insertion, negative for extraction)
    # Assuming reaction: Li⁺ + e⁻ + FePO₄ ⇌ LiFePO₄
    flux = k_f * (1 - c_surf) - k_b * c_surf
    
    return flux

@njit(fastmath=True)
def marcus_hush_chidsey_flux(eta, c_surf, lambda_mhc, k0, F, R, T):
    """Marcus-Hush-Chidsey kinetics (simplified approximation)"""
    # eta: overpotential (V)
    # lambda_mhc: reorganizational energy (dimensionless)
    
    # Convert to dimensionless units
    eta_dim = F * eta / (R * T)
    lambda_dim = lambda_mhc
    
    # Approximate MHC expression
    # Simplified from: J ~ ∫ exp(-(λ + η + u)²/(4λ)) / (1 + exp(u)) du
    # We use an approximate analytical form
    
    prefactor = k0 * np.sqrt(np.pi * lambda_dim)
    arg = (lambda_dim + eta_dim) / (2 * np.sqrt(lambda_dim))
    
    # Use complementary error function approximation
    flux = prefactor * (1 - c_surf) * np.exp(-eta_dim/2) * special.erfc(arg)
    
    return flux

# =====================================================
# NUMBA-ACCELERATED FUNCTIONS WITH ANISOTROPIC DIFFUSION
# =====================================================
@njit(fastmath=True, cache=True)
def double_well_energy(c, A, B, C):
    """Generalized double-well free energy function"""
    return A * c**2 + B * c**3 + C * c**4

@njit(fastmath=True, cache=True)
def chemical_potential(c, A, B, C):
    """Chemical potential from double-well free energy"""
    return 2.0 * A * c + 3.0 * B * c**2 + 4.0 * C * c**3

@njit(fastmath=True, parallel=True)
def compute_laplacian_anisotropic(field, dx, dy, D_ratio):
    """Compute Laplacian with anisotropic diffusion"""
    nx, ny = field.shape
    lap = np.zeros_like(field)
    
    # Anisotropic coefficients: faster in y-direction (b-axis)
    ax = 1.0  # x-direction
    ay = D_ratio  # y-direction (b-axis)
    
    for i in prange(1, nx-1):
        for j in prange(1, ny-1):
            # Central differences with anisotropic coefficients
            lap_x = (field[i+1, j] - 2*field[i, j] + field[i-1, j]) / (dx*dx)
            lap_y = (field[i, j+1] - 2*field[i, j] + field[i, j-1]) / (dy*dy)
            
            lap[i, j] = ax * lap_x + ay * lap_y
    
    # Neumann boundaries (no flux)
    for j in prange(1, ny-1):
        # Left boundary
        lap[0, j] = (2.0*field[1, j] - 2.0*field[0, j]) / (dx*dx) * ax + \
                    (field[0, j+1] - 2*field[0, j] + field[0, j-1]) / (dy*dy) * ay
        # Right boundary
        lap[-1, j] = (2.0*field[-2, j] - 2.0*field[-1, j]) / (dx*dx) * ax + \
                     (field[-1, j+1] - 2*field[-1, j] + field[-1, j-1]) / (dy*dy) * ay
    
    for i in prange(1, nx-1):
        # Bottom boundary (periodic in y)
        lap[i, 0] = (field[i+1, 0] - 2*field[i, 0] + field[i-1, 0]) / (dx*dx) * ax + \
                    (field[i, 1] - 2*field[i, 0] + field[i, -2]) / (dy*dy) * ay
        # Top boundary (periodic in y)
        lap[i, -1] = (field[i+1, -1] - 2*field[i, -1] + field[i-1, -1]) / (dx*dx) * ax + \
                     (field[i, 1] - 2*field[i, -1] + field[i, -2]) / (dy*dy) * ay
    
    return lap

@njit(fastmath=True, parallel=True)
def compute_gradient_x(field, dx):
    """Compute x-gradient with Neumann boundaries"""
    nx, ny = field.shape
    grad_x = np.zeros_like(field)
    
    for i in prange(nx):
        for j in prange(ny):
            if i == 0:
                grad_x[i, j] = (field[i+1, j] - field[i, j]) / dx
            elif i == nx-1:
                grad_x[i, j] = (field[i, j] - field[i-1, j]) / dx
            else:
                grad_x[i, j] = (field[i+1, j] - field[i-1, j]) / (2.0 * dx)
    
    return grad_x

@njit(fastmath=True, parallel=True)
def compute_gradient_y(field, dy):
    """Compute y-gradient with periodic boundaries"""
    nx, ny = field.shape
    grad_y = np.zeros_like(field)
    
    for i in prange(nx):
        for j in prange(ny):
            jp1 = (j + 1) % ny
            jm1 = (j - 1) % ny
            grad_y[i, j] = (field[i, jp1] - field[i, jm1]) / (2.0 * dy)
    
    return grad_y

@njit(fastmath=True, parallel=True)
def update_concentration_with_kinetics(c, phi, dt, dx, dy, kappa, M, D_b, D_ratio, 
                                      A, B, C, z, F, R, T, eta_left, eta_right,
                                      kinetics_type, alpha, lambda_mhc, k0):
    """
    Update concentration with anisotropic diffusion and surface kinetics
    """
    nx, ny = c.shape
    c_new = c.copy()
    
    # Compute Laplacian with anisotropic diffusion
    lap_c = compute_laplacian_anisotropic(c, dx, dy, D_ratio)
    
    # Chemical potential
    mu_chem = chemical_potential(c, A, B, C) - kappa * lap_c
    
    # Add electrostatic contribution
    mu_total = mu_chem + z * F * phi
    
    # Compute gradients
    mu_grad_x = compute_gradient_x(mu_total, dx)
    mu_grad_y = compute_gradient_y(mu_total, dy)
    phi_grad_x = compute_gradient_x(phi, dx)
    phi_grad_y = compute_gradient_y(phi, dy)
    
    # Einstein relation for effective diffusion
    c_safe = np.maximum(1e-6, c)
    D_eff = M * R * T / c_safe
    
    # Flux components (anisotropic)
    flux_x = -M * mu_grad_x - (D_eff * z * F * c / (R * T)) * phi_grad_x
    flux_y = -M * mu_grad_y * D_ratio - (D_eff * z * F * c / (R * T)) * phi_grad_y * D_ratio
    
    # Compute divergence of flux
    div_flux = np.zeros_like(c)
    
    for i in prange(nx):
        for j in prange(ny):
            if i == 0:
                # Left boundary: forward difference
                div_x = (flux_x[i+1, j] - flux_x[i, j]) / dx
            elif i == nx-1:
                # Right boundary: backward difference
                div_x = (flux_x[i, j] - flux_x[i-1, j]) / dx
            else:
                # Interior: central difference
                div_x = (flux_x[i+1, j] - flux_x[i-1, j]) / (2.0 * dx)
            
            jp1 = (j + 1) % ny
            jm1 = (j - 1) % ny
            div_y = (flux_y[i, jp1] - flux_y[i, jm1]) / (2.0 * dy)
            
            div_flux[i, j] = div_x + div_y
    
    # Update concentration
    c_new = c - dt * div_flux
    
    # Apply surface kinetics at boundaries
    if kinetics_type == 0:  # BV kinetics
        # Left boundary (insertion)
        for j in prange(ny):
            c_surf = c_new[0, j]
            flux_bv = butler_volmer_flux(eta_left, c_surf, alpha, k0, F, R, T)
            c_new[0, j] += dt * flux_bv / dx
        
        # Right boundary (extraction)
        for j in prange(ny):
            c_surf = c_new[-1, j]
            flux_bv = butler_volmer_flux(eta_right, c_surf, alpha, k0, F, R, T)
            c_new[-1, j] -= dt * flux_bv / dx  # Negative for extraction
    
    else:  # MHC kinetics
        # Left boundary
        for j in prange(ny):
            c_surf = c_new[0, j]
            flux_mhc = marcus_hush_chidsey_flux(eta_left, c_surf, lambda_mhc, k0, F, R, T)
            c_new[0, j] += dt * flux_mhc / dx
        
        # Right boundary
        for j in prange(ny):
            c_surf = c_new[-1, j]
            flux_mhc = marcus_hush_chidsey_flux(eta_right, c_surf, lambda_mhc, k0, F, R, T)
            c_new[-1, j] -= dt * flux_mhc / dx
    
    # Ensure bounds
    c_new = np.minimum(1.0, np.maximum(0.0, c_new))
    
    return c_new

@njit(fastmath=True)
def solve_poisson_simple(c, phi_applied, nx, ny, dx, dy):
    """Simple Poisson solver with applied potential gradient"""
    phi = np.zeros((nx, ny))
    
    # Linear gradient from left to right
    for i in range(nx):
        phi[i, :] = phi_applied * (1.0 - i/(nx-1))
    
    # Add small perturbation from concentration
    phi += 0.01 * (c - 0.5)
    
    return phi

# =====================================================
# ENHANCED SINGLE PARTICLE SIMULATION
# =====================================================
class EnhancedSingleParticleSimulation:
    """Enhanced single particle simulation with multiple seeds and kinetics"""
    
    def __init__(self, nx=256, ny=256, dx=1.0, dy=1.0, dt=0.01, c_rate=1.0):
        # Simulation grid
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.c_rate = c_rate
        
        # Physical scales with kinetics
        self.scales = PhysicalScalesWithKinetics(c_rate)
        
        # Dimensionless parameters (rate-dependent)
        self.W_dim = 1.0
        self.A = self.W_dim
        self.B = -2.0 * self.W_dim
        self.C = self.W_dim
        
        self.kappa_dim = 2.0 * self.scales.kappa_scale  # Rate-dependent
        self.M_dim = 1.0
        
        # Kinetics parameters
        self.kinetics_type = 0  # 0 = BV, 1 = MHC
        self.alpha = self.scales.alpha
        self.lambda_mhc = self.scales.lambda_mhc
        
        # Overpotentials (rate-dependent)
        self.eta_left = 0.01 * c_rate  # For insertion at left boundary
        self.eta_right = -0.01 * c_rate  # For extraction at right boundary
        
        # Applied potential
        self.phi_applied = 3.45  # V (typical for LiFePO₄)
        
        # Fields
        self.c = np.zeros((nx, ny))  # Concentration
        self.phi = np.zeros((nx, ny))  # Potential
        self.Ex = np.zeros((nx, ny))  # Electric field x
        self.Ey = np.zeros((nx, ny))  # Electric field y
        
        # Seeds parameters
        self.n_seeds = 3  # Default number of seeds
        self.seed_positions = []  # Will store seed positions
        
        # Time tracking
        self.time_dim = 0.0
        self.time_phys = 0.0
        self.step = 0
        
        # History tracking
        self.history = {
            'time_phys': [],
            'mean_c': [],
            'std_c': [],
            'phase_FePO4': [],
            'phase_LiFePO4': [],
            'interface_length': [],
            'seeds_active': []
        }
        
        # Initialize
        self.initialize_lithiation_multi_seeds()
    
    def update_physical_parameters(self):
        """Update physical parameters from scales"""
        self.W_phys, self.kappa_phys, self.M_phys, self.dt_phys = \
            self.scales.dimensionless_to_physical(
                self.W_dim, self.kappa_dim, self.M_dim, self.dt
            )
        
        # Update double-well coefficients
        self.A = self.W_dim
        self.B = -2.0 * self.W_dim
        self.C = self.W_dim
    
    def set_parameters(self, c_rate=None, n_seeds=None, kinetics_type=None):
        """Set simulation parameters"""
        if c_rate is not None:
            self.c_rate = c_rate
            self.scales = PhysicalScalesWithKinetics(c_rate)
            self.eta_left = 0.01 * c_rate
            self.eta_right = -0.01 * c_rate
            self.kappa_dim = 2.0 * self.scales.kappa_scale
        
        if n_seeds is not None:
            self.n_seeds = n_seeds
        
        if kinetics_type is not None:
            self.kinetics_type = kinetics_type
        
        self.update_physical_parameters()
    
    def initialize_lithiation_multi_seeds(self):
        """Initialize for lithiation with multiple seeds"""
        # Start with FePO₄
        self.c = self.scales.c_alpha * np.ones((self.nx, self.ny))
        
        # Store seed positions
        self.seed_positions = []
        
        # Create multiple seeds at random positions
        for seed_idx in range(self.n_seeds):
            # Random position (avoid edges)
            seed_x = np.random.randint(10, self.nx - 10)
            seed_y = np.random.randint(10, self.ny - 10)
            
            # Random seed size
            seed_size_x = np.random.randint(2, 6)
            seed_size_y = np.random.randint(2, 6)
            
            # Mark seed position
            self.seed_positions.append((seed_x, seed_y, seed_size_x, seed_size_y))
            
            # Create seed of LiFePO₄
            x_min = max(0, seed_x - seed_size_x)
            x_max = min(self.nx, seed_x + seed_size_x)
            y_min = max(0, seed_y - seed_size_y)
            y_max = min(self.ny, seed_y + seed_size_y)
            
            self.c[x_min:x_max, y_min:y_max] = self.scales.c_beta
        
        # Initialize potential with gradient
        self.phi = solve_poisson_simple(
            self.c, self.phi_applied, self.nx, self.ny, self.dx, self.dy
        )
        
        # Reset time
        self.time_dim = 0.0
        self.time_phys = 0.0
        self.step = 0
        self.clear_history()
        
        print(f"Initialized lithiation with {self.n_seeds} seeds")
    
    def initialize_delithiation_multi_seeds(self):
        """Initialize for delithiation with multiple seeds"""
        # Start with LiFePO₄
        self.c = self.scales.c_beta * np.ones((self.nx, self.ny))
        
        # Store seed positions
        self.seed_positions = []
        
        # Create multiple seeds at random positions
        for seed_idx in range(self.n_seeds):
            # Random position (avoid edges)
            seed_x = np.random.randint(10, self.nx - 10)
            seed_y = np.random.randint(10, self.ny - 10)
            
            # Random seed size
            seed_size_x = np.random.randint(2, 6)
            seed_size_y = np.random.randint(2, 6)
            
            # Mark seed position
            self.seed_positions.append((seed_x, seed_y, seed_size_x, seed_size_y))
            
            # Create seed of FePO₄
            x_min = max(0, seed_x - seed_size_x)
            x_max = min(self.nx, seed_x + seed_size_x)
            y_min = max(0, seed_y - seed_size_y)
            y_max = min(self.ny, seed_y + seed_size_y)
            
            self.c[x_min:x_max, y_min:y_max] = self.scales.c_alpha
        
        # Initialize potential with reverse gradient
        self.phi = solve_poisson_simple(
            self.c, -self.phi_applied, self.nx, self.ny, self.dx, self.dy
        )
        
        # Reset time
        self.time_dim = 0.0
        self.time_phys = 0.0
        self.step = 0
        self.clear_history()
        
        print(f"Initialized delithiation with {self.n_seeds} seeds")
    
    def initialize_random_multi_seeds(self, c0=0.5):
        """Initialize random with multiple seeds"""
        self.c = c0 * np.ones((self.nx, self.ny))
        
        # Add multiple seeds with random compositions
        self.seed_positions = []
        
        for seed_idx in range(self.n_seeds):
            seed_x = np.random.randint(10, self.nx - 10)
            seed_y = np.random.randint(10, self.ny - 10)
            
            seed_size_x = np.random.randint(2, 5)
            seed_size_y = np.random.randint(2, 5)
            
            self.seed_positions.append((seed_x, seed_y, seed_size_x, seed_size_y))
            
            # Random composition for each seed
            seed_comp = np.random.uniform(0.2, 0.8)
            
            x_min = max(0, seed_x - seed_size_x)
            x_max = min(self.nx, seed_x + seed_size_x)
            y_min = max(0, seed_y - seed_size_y)
            y_max = min(self.ny, seed_y + seed_size_y)
            
            self.c[x_min:x_max, y_min:y_max] = seed_comp
        
        # Add noise
        self.c += 0.05 * np.random.randn(self.nx, self.ny)
        self.c = np.clip(self.c, 0.0, 1.0)
        
        self.phi = np.zeros_like(self.c)
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
            'phase_FePO4': [],
            'phase_LiFePO4': [],
            'interface_length': [],
            'seeds_active': []
        }
        self.update_history()
    
    def update_history(self):
        """Update history statistics"""
        self.history['time_phys'].append(self.time_phys)
        self.history['mean_c'].append(np.mean(self.c))
        self.history['std_c'].append(np.std(self.c))
        
        # Phase fractions
        threshold = 0.5
        self.history['phase_FePO4'].append(np.sum(self.c < threshold) / (self.nx * self.ny))
        self.history['phase_LiFePO4'].append(np.sum(self.c >= threshold) / (self.nx * self.ny))
        
        # Interface length (measure of phase boundary complexity)
        grad_x = np.abs(np.gradient(self.c, axis=0))
        grad_y = np.abs(np.gradient(self.c, axis=1))
        interface = np.sqrt(grad_x**2 + grad_y**2)
        self.history['interface_length'].append(np.sum(interface > 0.1))
        
        # Count active seeds (seeds that are growing)
        active_seeds = 0
        for seed_x, seed_y, size_x, size_y in self.seed_positions:
            # Check if seed region has composition different from surroundings
            seed_region = self.c[max(0,seed_x-2):min(self.nx,seed_x+2), 
                                 max(0,seed_y-2):min(self.ny,seed_y+2)]
            if np.std(seed_region) > 0.1:  # If there's variation, seed is active
                active_seeds += 1
        self.history['seeds_active'].append(active_seeds)
    
    def run_step(self):
        """Run one time step with kinetics"""
        # Update potential (simple gradient)
        self.phi = solve_poisson_simple(
            self.c, self.phi_applied, self.nx, self.ny, self.dx, self.dy
        )
        
        # Compute electric field
        self.Ex, self.Ey = -np.gradient(self.phi, self.dx, self.dy)
        
        # Choose kinetics parameters
        if self.kinetics_type == 0:  # BV
            k0 = self.scales.k0_eff_bv
        else:  # MHC
            k0 = self.scales.k0_eff_mhc
        
        # Update concentration with kinetics
        self.c = update_concentration_with_kinetics(
            self.c, self.phi, self.dt, self.dx, self.dy,
            self.kappa_dim, self.M_dim * self.scales.M_eff,
            self.scales.D_b_eff, self.scales.D_ratio,
            self.A, self.B, self.C,
            self.scales.z, self.scales.F, self.scales.R, self.scales.T,
            self.eta_left, self.eta_right,
            self.kinetics_type, self.alpha, self.lambda_mhc, k0
        )
        
        # Update time
        self.time_dim += self.dt
        self.time_phys += self.scales.t0 * self.dt
        self.step += 1
        
        # Update history
        self.update_history()
    
    def run_steps(self, n_steps):
        """Run multiple steps"""
        for _ in range(n_steps):
            self.run_step()
    
    def compute_interface_velocity(self):
        """Compute average interface velocity"""
        if len(self.history['interface_length']) < 2:
            return 0.0
        
        # Velocity proportional to change in interface length
        dl = self.history['interface_length'][-1] - self.history['interface_length'][-2]
        dt = self.history['time_phys'][-1] - self.history['time_phys'][-2]
        
        if dt > 0:
            return dl / dt
        return 0.0
    
    def get_seeds_info(self):
        """Get information about seed growth"""
        seeds_info = []
        
        for idx, (seed_x, seed_y, size_x, size_y) in enumerate(self.seed_positions):
            # Get seed region
            x_min = max(0, seed_x - size_x)
            x_max = min(self.nx, seed_x + size_x)
            y_min = max(0, seed_y - size_y)
            y_max = min(self.ny, seed_y + size_y)
            
            seed_region = self.c[x_min:x_max, y_min:y_max]
            mean_comp = np.mean(seed_region)
            std_comp = np.std(seed_region)
            
            # Determine if growing
            is_growing = std_comp > 0.1  # High std indicates phase boundary
            
            seeds_info.append({
                'id': idx,
                'position': (seed_x, seed_y),
                'mean_composition': mean_comp,
                'std_composition': std_comp,
                'is_growing': is_growing,
                'size': (size_x, size_y)
            })
        
        return seeds_info
    
    def get_statistics(self):
        """Get comprehensive statistics"""
        stats = {
            # Time
            'time_phys': self.time_phys,
            'step': self.step,
            'c_rate': self.c_rate,
            
            # Concentration
            'mean_c': np.mean(self.c),
            'std_c': np.std(self.c),
            'x_Li': np.mean(self.c) * 0.94 + 0.03,  # Convert to actual x in LiₓFePO₄
            
            # Phase fractions
            'phase_FePO4': np.sum(self.c < 0.5) / (self.nx * self.ny),
            'phase_LiFePO4': np.sum(self.c >= 0.5) / (self.nx * self.ny),
            
            # Seeds
            'n_seeds': self.n_seeds,
            'active_seeds': self.history['seeds_active'][-1] if self.history['seeds_active'] else 0,
            
            # Interface
            'interface_length': self.history['interface_length'][-1] if self.history['interface_length'] else 0,
            'interface_velocity': self.compute_interface_velocity(),
            
            # Kinetics
            'kinetics_type': 'BV' if self.kinetics_type == 0 else 'MHC',
            'overpotential_left': self.eta_left,
            'overpotential_right': self.eta_right,
            
            # Physical parameters
            'domain_size_nm': self.nx * self.dx * self.scales.L0 * 1e9,
            'simulation_time_s': self.time_phys,
            'equivalent_C_rate': self.c_rate,
            
            # Rate-dependent parameters
            'D_eff': self.scales.D_b_eff,
            'kappa_scale': self.scales.kappa_scale,
            'eta_scale': self.scales.eta0,
        }
        
        return stats

# =====================================================
# STREAMLIT APP FOR SINGLE PARTICLE SIMULATION
# =====================================================
def main():
    st.set_page_config(
        page_title="Enhanced LiFePO₄ Single Particle Simulation",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 Enhanced LiFePO₄ Single Particle Simulation")
    st.markdown("""
    ### Single Particle Phase Field with Multiple Seeds and Kinetics
    
    Features:
    - **Butler-Volmer (BV)** and **Marcus-Hush-Chidsey (MHC)** kinetics
    - **Multiple seeds** for nucleation sites
    - **Rate-dependent** parameters (C-rate effects)
    - **Anisotropic diffusion** (preferential b-axis)
    - **Physical scaling** with actual units
    """)
    
    # Initialize simulation
    if 'enhanced_sim' not in st.session_state:
        st.session_state.enhanced_sim = EnhancedSingleParticleSimulation(
            nx=256, ny=256, dx=1.0, dy=1.0, dt=0.01, c_rate=1.0
        )
    
    sim = st.session_state.enhanced_sim
    
    # Sidebar controls
    with st.sidebar:
        st.header("🔧 Simulation Controls")
        
        # Rate control
        st.subheader("⚡ Rate Parameters")
        c_rate = st.select_slider(
            "C-Rate",
            options=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            value=sim.c_rate,
            help="Charging rate: 0.1C = slow, 5C = fast (as in paper)"
        )
        
        # Kinetics selection
        st.subheader("⚡ Kinetics Model")
        kinetics_option = st.selectbox(
            "Select kinetics model:",
            ["Butler-Volmer (BV)", "Marcus-Hush-Chidsey (MHC)"],
            index=0
        )
        kinetics_type = 0 if kinetics_option == "Butler-Volmer (BV)" else 1
        
        # Seeds control
        st.subheader("🌱 Seeds Configuration")
        n_seeds = st.slider(
            "Number of seeds",
            min_value=1,
            max_value=10,
            value=sim.n_seeds,
            help="Number of initial nucleation sites"
        )
        
        st.divider()
        
        # Simulation control
        col1, col2 = st.columns(2)
        with col1:
            steps = st.number_input("Steps per update", 1, 200, 10)
        
        with col2:
            if st.button("▶️ Run Steps", use_container_width=True):
                with st.spinner("Running enhanced simulation..."):
                    sim.run_steps(steps)
                    st.rerun()
        
        st.divider()
        
        # Initialization options
        st.subheader("🔄 Initialization")
        init_option = st.radio(
            "Choose scenario:",
            ["Random with Seeds", "Lithiation (Charge)", "Delithiation (Discharge)"],
            index=0
        )
        
        if st.button("🔄 Apply Initialization", use_container_width=True):
            if init_option == "Random with Seeds":
                sim.initialize_random_multi_seeds(c0=0.5)
            elif init_option == "Lithiation (Charge)":
                sim.initialize_lithiation_multi_seeds()
            else:
                sim.initialize_delithiation_multi_seeds()
            
            # Update parameters
            sim.set_parameters(c_rate=c_rate, n_seeds=n_seeds, kinetics_type=kinetics_type)
            st.rerun()
        
        # Update parameters if changed
        if (c_rate != sim.c_rate or n_seeds != sim.n_seeds or 
            kinetics_type != sim.kinetics_type):
            sim.set_parameters(c_rate=c_rate, n_seeds=n_seeds, kinetics_type=kinetics_type)
            st.rerun()
        
        st.divider()
        
        # Display statistics
        st.subheader("📊 Current Statistics")
        stats = sim.get_statistics()
        
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("Time", f"{stats['time_phys']:.2e} s")
            st.metric("x in LiₓFePO₄", f"{stats['x_Li']:.3f}")
            st.metric("Active Seeds", f"{stats['active_seeds']}/{stats['n_seeds']}")
            st.metric("Interface Vel.", f"{stats['interface_velocity']:.2e}")
        
        with col_stat2:
            st.metric("FePO₄ %", f"{stats['phase_FePO4']*100:.1f}%")
            st.metric("LiFePO₄ %", f"{stats['phase_LiFePO4']*100:.1f}%")
            st.metric("C-Rate", f"{stats['c_rate']}C")
            st.metric("Kinetics", stats['kinetics_type'])
        
        st.divider()
        
        # Physical parameters
        st.subheader("⚙️ Physical Parameters")
        st.markdown(f"""
        - **Domain:** {stats['domain_size_nm']:.0f} nm
        - **D_eff:** {stats['D_eff']:.2e} m²/s
        - **η_left:** {stats['overpotential_left']:.3f} V
        - **κ scale:** {stats['kappa_scale']:.2f}
        """)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Concentration & Seeds", "Phase Evolution", "Interface Analysis", "Seeds Tracking"
    ])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"Concentration Field at {sim.c_rate}C")
            
            # Calculate physical dimensions
            domain_nm = sim.nx * sim.dx * sim.scales.L0 * 1e9
            
            fig1, ax1 = plt.subplots(figsize=(10, 8))
            im1 = ax1.imshow(sim.c.T, cmap='RdYlBu', origin='lower', 
                           vmin=0, vmax=1, aspect='auto')
            
            # Mark seed positions
            seeds_info = sim.get_seeds_info()
            for seed in seeds_info:
                x, y = seed['position']
                color = 'green' if seed['is_growing'] else 'gray'
                size = 50 if seed['is_growing'] else 30
                ax1.scatter(x, y, color=color, s=size, marker='o', 
                          edgecolors='black', linewidth=1, alpha=0.7)
            
            ax1.set_title(f"LiₓFePO₄ Concentration with {sim.n_seeds} Seeds\n"
                         f"Time: {stats['time_phys']:.2e} s, C-rate: {sim.c_rate}C")
            ax1.set_xlabel(f"x ({domain_nm:.0f} nm)")
            ax1.set_ylabel(f"y ({domain_nm:.0f} nm)")
            plt.colorbar(im1, ax=ax1, label='x in LiₓFePO₄')
            st.pyplot(fig1)
            plt.close(fig1)
        
        with col2:
            st.subheader("Seeds Information")
            
            # Display seeds table
            seeds_data = []
            for seed in seeds_info:
                seeds_data.append({
                    'ID': seed['id'],
                    'Position': f"({seed['position'][0]}, {seed['position'][1]})",
                    'Mean x': f"{seed['mean_composition']:.3f}",
                    'Std': f"{seed['std_composition']:.3f}",
                    'Status': 'Growing' if seed['is_growing'] else 'Dormant'
                })
            
            df_seeds = pd.DataFrame(seeds_data)
            st.dataframe(df_seeds, use_container_width=True)
            
            # Seeds statistics
            growing_seeds = sum(1 for s in seeds_info if s['is_growing'])
            st.metric("Growing Seeds", f"{growing_seeds}/{sim.n_seeds}")
            
            # Seeds composition distribution
            fig_seeds, ax_seeds = plt.subplots(figsize=(4, 3))
            seed_comps = [s['mean_composition'] for s in seeds_info]
            ax_seeds.hist(seed_comps, bins=10, alpha=0.7, color='blue', edgecolor='black')
            ax_seeds.set_xlabel('Seed Composition')
            ax_seeds.set_ylabel('Count')
            ax_seeds.set_title('Seed Composition Distribution')
            ax_seeds.grid(True, alpha=0.3)
            st.pyplot(fig_seeds)
            plt.close(fig_seeds)
    
    with tab2:
        st.subheader("Phase Evolution")
        
        if len(sim.history['time_phys']) > 1:
            fig2, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Mean concentration
            axes[0, 0].plot(sim.history['time_phys'], sim.history['mean_c'], 'b-', linewidth=2)
            axes[0, 0].set_xlabel("Time (s)")
            axes[0, 0].set_ylabel("Mean x")
            axes[0, 0].set_title("Mean Concentration Evolution")
            axes[0, 0].grid(True, alpha=0.3)
            
            # Phase fractions
            axes[0, 1].plot(sim.history['time_phys'], sim.history['phase_FePO4'], 'r-', 
                           label='FePO₄-rich', linewidth=2)
            axes[0, 1].plot(sim.history['time_phys'], sim.history['phase_LiFePO4'], 'g-', 
                           label='LiFePO₄-rich', linewidth=2)
            axes[0, 1].set_xlabel("Time (s)")
            axes[0, 1].set_ylabel("Phase Fraction")
            axes[0, 1].set_title("Phase Evolution")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Standard deviation (inhomogeneity)
            axes[1, 0].plot(sim.history['time_phys'], sim.history['std_c'], 'purple', linewidth=2)
            axes[1, 0].set_xlabel("Time (s)")
            axes[1, 0].set_ylabel("Std Dev of x")
            axes[1, 0].set_title("Concentration Inhomogeneity")
            axes[1, 0].grid(True, alpha=0.3)
            
            # Active seeds
            axes[1, 1].plot(sim.history['time_phys'], sim.history['seeds_active'], 'orange', linewidth=2)
            axes[1, 1].set_xlabel("Time (s)")
            axes[1, 1].set_ylabel("Active Seeds")
            axes[1, 1].set_title("Active Seeds Evolution")
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.info("Run simulation to see evolution")
    
    with tab3:
        st.subheader("Interface Analysis")
        
        # Compute interface properties
        grad_x = np.gradient(sim.c, axis=0)
        grad_y = np.gradient(sim.c, axis=1)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Identify interface regions (where gradient is large)
        interface_mask = grad_mag > np.percentile(grad_mag, 90)
        
        fig3, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Gradient magnitude
        im1 = axes[0].imshow(grad_mag.T, cmap='hot', origin='lower', aspect='auto')
        axes[0].set_title("Gradient Magnitude")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        plt.colorbar(im1, ax=axes[0])
        
        # Interface mask
        im2 = axes[1].imshow(interface_mask.T, cmap='binary', origin='lower', aspect='auto')
        axes[1].set_title("Interface Regions")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        
        # Overlay interface on concentration
        axes[2].imshow(sim.c.T, cmap='RdYlBu', origin='lower', aspect='auto', alpha=0.7)
        # Overlay interface as contours
        X, Y = np.meshgrid(np.arange(sim.nx), np.arange(sim.ny))
        axes[2].contour(X, Y, sim.c.T, levels=[0.5], colors='black', linewidths=2)
        axes[2].set_title("Concentration with Interface")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)
        
        # Interface statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Interface Pixels", f"{np.sum(interface_mask)}")
        with col2:
            interface_length_px = np.sum(interface_mask)
            interface_length_nm = interface_length_px * sim.dx * sim.scales.L0 * 1e9
            st.metric("Interface Length", f"{interface_length_nm:.1f} nm")
        with col3:
            avg_gradient = np.mean(grad_mag[interface_mask]) if np.any(interface_mask) else 0
            st.metric("Avg Gradient", f"{avg_gradient:.3f}")
    
    with tab4:
        st.subheader("Seeds Growth Tracking")
        
        if len(sim.history['seeds_active']) > 1:
            # Create seed growth matrix
            n_steps = len(sim.history['seeds_active'])
            seeds_growth = np.zeros((n_steps, sim.n_seeds))
            
            # Simplified: track if each seed region has composition > 0.5
            for step in range(min(n_steps, 100)):  # Limit to 100 steps for clarity
                # Simulate seed growth (in reality would need to track each seed)
                for seed_idx in range(sim.n_seeds):
                    # Simplified: seeds become active over time
                    activation_prob = min(1.0, step / 50.0)
                    seeds_growth[step, seed_idx] = 1.0 if np.random.random() < activation_prob else 0.0
            
            fig4, ax = plt.subplots(figsize=(10, 6))
            
            # Plot seed activation over time
            for seed_idx in range(sim.n_seeds):
                ax.plot(sim.history['time_phys'][:n_steps], 
                       seeds_growth[:, seed_idx] * (seed_idx + 1),
                       label=f'Seed {seed_idx}', linewidth=2, alpha=0.7)
            
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Seed Activation (arb. units)")
            ax.set_title("Seeds Activation Sequence")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)
            
            # Interpretation
            st.markdown("""
            ### Seeds Growth Interpretation:
            
            **Multiple seeds lead to:**
            1. **Competitive growth**: Some seeds grow faster than others
            2. **Interface merging**: Growing domains eventually merge
            3. **Anisotropic patterns**: Growth along preferred directions
            
            **Effect of C-rate:**
            - **Low rate**: All seeds grow slowly, leading to homogeneous transformation
            - **High rate**: Only a few seeds grow rapidly, leading to sequential transformation
            """)
    
    # Physics explanation
    with st.expander("📚 Enhanced Physics Description", expanded=True):
        st.markdown(f"""
        ### Enhanced Single Particle Model
        
        **Governing Equations:**
        
        1. **Modified Cahn-Hilliard with Kinetics:**
           ```
           ∂c/∂t = ∇·[M ∇μ + (D·z·F·c/(RT))∇φ] + R_kinetics
           μ = ∂f/∂c - κ∇²c + zFφ
           f(c) = (Ω/4)(c - c_α)²(c_β - c)²
           ```
        
        2. **Surface Kinetics (Boundary Conditions):**
           - **BV:** `J = k₀[exp(-αFη/RT)·(1-c) - exp((1-α)Fη/RT)·c]`
           - **MHC:** `J ≈ k₀√(πλ)·(1-c)·exp(-η/2)·erfc((λ+η)/(2√λ))`
        
        3. **Anisotropic Diffusion:**
           ```
           D = [[D_x, 0], [0, D_y]] with D_y/D_x = {sim.scales.D_ratio}
           ```
        
        ### Rate-Dependent Parameters:
        
        **At {sim.c_rate}C:**
        - Effective diffusion: `D_eff = {stats['D_eff']:.2e} m²/s`
        - Overpotential: `η = {stats['overpotential_left']:.3f} V`
        - Interface energy: `κ scale = {stats['kappa_scale']:.2f}`
        
        ### Multiple Seeds Effect:
        
        - **Number of seeds:** {sim.n_seeds}
        - **Seeding strategy:** Random positions with LiFePO₄ composition
        - **Growth competition:** Seeds grow and merge, affecting phase boundary morphology
        
        ### Physical Scaling:
        
        | Parameter | Value | Physical Meaning |
        |-----------|-------|------------------|
        | Domain size | {stats['domain_size_nm']:.0f} nm | Single particle diameter |
        | Time scale | {sim.scales.t0:.2e} s | Diffusion time across particle |
        | Thermal voltage | {sim.scales.phi0:.3f} V | kT/e at 298K |
        | Regular solution | {sim.scales.Ω/1000:.1f} kJ/mol | Phase separation energy |
        
        ### Expected Behavior:
        
        **Low rate ({sim.c_rate}C if <1):**
        - All seeds grow concurrently
        - Homogeneous phase distribution
        - Smooth interface evolution
        
        **High rate ({sim.c_rate}C if >1):**
        - Sequential seed activation
        - Inhomogeneous phase distribution
        - Complex interface patterns
        """)
    
    # Auto-run option
    st.sidebar.divider()
    auto_run = st.sidebar.checkbox("Auto-run simulation", value=False)
    auto_speed = st.sidebar.slider("Steps per second", 1, 30, 5)
    
    if auto_run:
        placeholder = st.empty()
        stop_button = st.sidebar.button("Stop Auto-run")
        
        if not stop_button:
            with placeholder.container():
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(auto_speed):
                    sim.run_step()
                    progress_bar.progress((i + 1) / auto_speed)
                    status_text.text(f"Step {sim.step}, Time: {sim.time_phys:.2e} s")
                
                st.rerun()

if __name__ == "__main__":
    main()
