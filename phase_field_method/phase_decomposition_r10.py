import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

########################################################################
# ENHANCED SINGLE PARTICLE SIMULATION WITH C-RATE CONTROL
# Key features:
# 1. C-rate parameter that scales physical processes
# 2. Multiple seeds for nucleation
# 3. Proper dimensionless scaling matching original code
# 4. BV and MHC kinetics options
########################################################################

# =====================================================
# PHYSICAL CONSTANTS WITH C-RATE SCALING
# =====================================================
class PhysicalScalesWithC_Rate:
    """Physical scales with C-rate parameter for single particle"""
    
    # Fundamental constants (same as original code)
    R = 8.314462618  # J/(mol·K)
    F = 96485.33212  # C/mol
    k_B = 1.380649e-23  # J/K
    ε0 = 8.854187817e-12  # F/m
    
    def __init__(self, c_rate=1.0):
        """Initialize with C-rate parameter"""
        self.c_rate = c_rate  # C-rate (0.1C, 1C, 5C, etc.)
        
        # Material properties (same as original)
        self.T = 298.15  # K
        
        # LiFePO₄ phase compositions
        self.c_alpha = 0.03  # FePO₄ phase
        self.c_beta = 0.97   # LiFePO₄ phase
        
        # Molar volume
        self.V_m = 3.0e-5  # m³/mol
        
        # Diffusion coefficient (same as original)
        self.D_b = 1.0e-14  # m²/s
        
        # Electrostatic properties (same as original)
        self.ε_r = 15.0
        self.ε = self.ε_r * self.ε0
        
        # Charge properties
        self.z = 1.0
        self.ρ0 = 1.0e6
        
        # Regular solution parameter
        self.Ω = 55e3  # J/mol
        
        # Kinetics parameters (rate-dependent)
        self.k0_bv = 1.0e-6  # BV rate constant (m/s)
        self.k0_mhc = 5.0e-7  # MHC rate constant (m/s)
        self.alpha = 0.5  # BV symmetry factor
        
        # C-rate dependent parameters
        self.set_c_rate_parameters(c_rate)
        
        # Set characteristic scales (same scaling as original)
        self.set_scales()
        
        print(f"Physical scales at {c_rate}C:")
        print(f"  Domain: {self.L0*1e9:.1f} nm")
        print(f"  Time scale: {self.t0:.2e} s")
        print(f"  Overpotential scale: {self.eta_scale:.3f} V")
        print(f"  C-rate multiplier: {self.c_rate_factor:.2f}")
    
    def set_c_rate_parameters(self, c_rate):
        """Set C-rate dependent parameters"""
        self.c_rate = c_rate
        
        # C-rate scaling factor (1.0 for 1C)
        # Higher rates require higher overpotentials
        if c_rate <= 1.0:
            self.c_rate_factor = 1.0
            self.eta_scale = 0.01  # Small overpotential for slow rates
            self.nucleation_probability = 0.8  # High nucleation probability
        else:
            # Exponential scaling for high rates
            self.c_rate_factor = 1.0 + 0.5 * np.log10(c_rate)
            self.eta_scale = 0.01 * c_rate**0.5  # Larger overpotential
            self.nucleation_probability = 0.3  # Lower nucleation probability at high rates
        
        # Rate-dependent interface sharpness
        # At high rates, interface is sharper (more two-phase)
        self.kappa_factor = 1.0 + 0.2 * np.log10(max(1.0, c_rate))
        
        # Rate-dependent mobility (effective diffusion)
        self.D_factor = 1.0 / (1.0 + 0.1 * c_rate**0.5)
    
    def set_scales(self):
        """Set characteristic scales (SAME as original code)"""
        # Length scale: 10 nm domain (same as original)
        self.L0 = 1.0e-8  # 10 nm
        
        # Energy density scale from regular solution
        self.E0 = self.Ω / self.V_m  # J/m³
        
        # Time scale from diffusion (same as original)
        self.t0 = (self.L0**2) / self.D_b  # s
        
        # Mobility scale (same as original)
        self.M0 = self.D_b / (self.E0 * self.t0)  # m⁵/(J·s)
        
        # Electric potential scale (same as original)
        self.φ0 = self.R * self.T / self.F  # ~0.0257 V
    
    def dimensionless_to_physical(self, W_dim, κ_dim, M_dim, dt_dim):
        """Convert dimensionless to physical (same as original)"""
        W_phys = W_dim * self.E0
        κ_phys = κ_dim * self.E0 * self.L0**2
        M_phys = M_dim * self.M0
        dt_phys = dt_dim * self.t0
        return W_phys, κ_phys, M_phys, dt_phys

# =====================================================
# KINETICS FUNCTIONS (compatible with original code)
# =====================================================
@njit(fastmath=True, cache=True)
def double_well_energy(c, A, B, C):
    """Generalized double-well free energy function"""
    return A * c**2 + B * c**3 + C * c**4

@njit(fastmath=True, cache=True)
def chemical_potential(c, A, B, C):
    """Chemical potential from double-well free energy"""
    return 2.0 * A * c + 3.0 * B * c**2 + 4.0 * C * c**3

@njit(fastmath=True)
def butler_volmer_flux(eta, c_surf, alpha, k0, F, R, T):
    """Butler-Volmer kinetics"""
    # Simplified BV: J = k0 * [exp(-alpha*eta) - exp((1-alpha)*eta)] * (c_ref - c_surf)
    k_f = k0 * np.exp(-alpha * F * eta / (R * T))
    k_b = k0 * np.exp((1 - alpha) * F * eta / (R * T))
    
    # Assuming reaction limited by available sites
    flux = k_f * (1.0 - c_surf) - k_b * c_surf
    
    return flux

@njit(fastmath=True)
def marcus_hush_chidsey_flux(eta, c_surf, k0, F, R, T):
    """Simplified MHC kinetics"""
    # Simplified approximation: J ~ k0 * tanh(eta/2) * (1-c_surf)
    eta_dim = F * eta / (R * T)
    flux = k0 * np.tanh(eta_dim / 2.0) * (1.0 - c_surf)
    
    return flux

# =====================================================
# NUMBA-ACCELERATED FUNCTIONS (compatible with original)
# =====================================================
@njit(fastmath=True, parallel=True)
def compute_laplacian(field, dx):
    """Compute 5-point stencil Laplacian with periodic BCs"""
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
    """Compute x-gradient with periodic BCs"""
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
    """Compute y-gradient with periodic BCs"""
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
    """
    Solve Poisson equation: ∇²φ = -F/ε * (c - c_ref)
    Using Jacobi iteration with periodic BCs
    """
    nx, ny = c.shape
    phi_new = phi.copy()
    
    # Precompute constant
    kappa = F_const / epsilon * dx**2
    
    for it in range(max_iter):
        phi_old = phi_new.copy()
        
        for i in prange(nx):
            for j in prange(ny):
                # Periodic indices
                ip1 = (i + 1) % nx
                im1 = (i - 1) % nx
                jp1 = (j + 1) % ny
                jm1 = (j - 1) % ny
                
                # Jacobi update
                phi_new[i, j] = 0.25 * (phi_old[ip1, j] + phi_old[im1, j] + 
                                        phi_old[i, jp1] + phi_old[i, jm1] + 
                                        kappa * (c[i, j] - c_ref))
        
        # Check convergence
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
def update_concentration_with_kinetics(c, phi, dt, dx, kappa, M, D, A, B, C, 
                                      z, F, R, T, kinetics_type, k0, eta_left):
    """
    Update concentration with electrostatics AND surface kinetics
    Kinetics applied at left boundary (x=0) only
    """
    nx, ny = c.shape
    
    # Compute Laplacian of concentration for gradient energy
    lap_c = compute_laplacian(c, dx)
    
    # Chemical potential from free energy
    mu_chem = chemical_potential(c, A, B, C) - kappa * lap_c
    
    # Add electrostatic contribution to chemical potential
    mu_total = mu_chem + z * F * phi
    
    # Compute gradients
    mu_grad_x = compute_gradient_x(mu_total, dx)
    mu_grad_y = compute_gradient_y(mu_total, dx)
    phi_grad_x = compute_gradient_x(phi, dx)
    phi_grad_y = compute_gradient_y(phi, dx)
    
    # Einstein relation
    c_safe = np.maximum(1e-6, c)
    D_eff = M * R * T / c_safe
    
    # Flux components
    flux_diff_x = -M * mu_grad_x
    flux_diff_y = -M * mu_grad_y
    
    flux_mig_x = -(D_eff * z * F * c / (R * T)) * phi_grad_x
    flux_mig_y = -(D_eff * z * F * c / (R * T)) * phi_grad_y
    
    # Total flux
    flux_x = flux_diff_x + flux_mig_x
    flux_y = flux_diff_y + flux_mig_y
    
    # Compute divergence of flux
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
    
    # Update concentration in bulk
    c_new = c - dt * div_flux
    
    # Apply surface kinetics at left boundary (x=0)
    for j in prange(ny):
        c_surf = c_new[0, j]
        
        if kinetics_type == 0:  # BV kinetics
            flux = butler_volmer_flux(eta_left, c_surf, 0.5, k0, F, R, T)
        else:  # MHC kinetics
            flux = marcus_hush_chidsey_flux(eta_left, c_surf, k0, F, R, T)
        
        # Apply flux at boundary (positive for insertion)
        c_new[0, j] += dt * flux / dx
    
    # Ensure concentration stays in [0, 1]
    c_new = np.minimum(1.0, np.maximum(0.0, c_new))
    
    return c_new

@njit(fastmath=True, parallel=True)
def compute_electric_field(phi, dx):
    """Compute electric field E = -∇φ"""
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

# =====================================================
# ENHANCED SINGLE PARTICLE SIMULATION WITH C-RATE
# =====================================================
class EnhancedSingleParticleSimulation:
    """Enhanced single particle simulation with C-rate control"""
    
    def __init__(self, nx=256, ny=256, dx=1.0, dt=0.01, c_rate=1.0):
        # Simulation grid (same as original)
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dt = dt
        self.c_rate = c_rate
        
        # Physical scales with C-rate
        self.scales = PhysicalScalesWithC_Rate(c_rate)
        
        # Dimensionless parameters (C-rate adjusted)
        self.W_dim = 1.0
        self.A = self.W_dim
        self.B = -2.0 * self.W_dim
        self.C = self.W_dim
        
        # C-rate adjusted kappa (sharper interface at high rates)
        self.kappa_dim = 2.0 * self.scales.kappa_factor
        
        # C-rate adjusted mobility
        self.M_dim = 1.0 * self.scales.D_factor
        
        # Update physical parameters
        self.update_physical_parameters()
        
        # Kinetics parameters
        self.kinetics_type = 0  # 0 = BV, 1 = MHC
        self.eta_left = self.scales.eta_scale  # Overpotential at left boundary
        
        # Seeds parameters
        self.n_seeds = 5  # Default number of seeds
        self.seed_positions = []
        
        # Fields (same as original)
        self.c = np.zeros((nx, ny))
        self.phi = np.zeros((nx, ny))
        self.Ex = np.zeros((nx, ny))
        self.Ey = np.zeros((nx, ny))
        
        # Time tracking
        self.time_dim = 0.0
        self.time_phys = 0.0
        self.step = 0
        
        # History tracking (enhanced)
        self.history = {
            'time_phys': [],
            'mean_c': [],
            'std_c': [],
            'mean_phi': [],
            'voltage': [],
            'phase_FePO4': [],
            'phase_LiFePO4': [],
            'nuclei_active': [],
            'interface_length': []
        }
        
        # Initialize with multiple seeds
        self.initialize_lithiation_multi_seeds()
    
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
                      c_rate=None, n_seeds=None, kinetics_type=None):
        """Set simulation parameters"""
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
            self.scales = PhysicalScalesWithC_Rate(c_rate)
            self.eta_left = self.scales.eta_scale
            self.kappa_dim = 2.0 * self.scales.kappa_factor
            self.M_dim = 1.0 * self.scales.D_factor
        if n_seeds is not None:
            self.n_seeds = n_seeds
        if kinetics_type is not None:
            self.kinetics_type = 0 if kinetics_type == "BV" else 1
        
        self.update_physical_parameters()
    
    def create_multiple_seeds(self, target_composition):
        """Create multiple seeds for nucleation"""
        self.seed_positions = []
        
        # Determine seed count based on C-rate
        # High rates: fewer active seeds, low rates: more seeds
        active_seeds = max(1, int(self.n_seeds * self.scales.nucleation_probability))
        
        for seed_idx in range(active_seeds):
            # Random position (avoid very edges)
            seed_x = np.random.randint(5, self.nx - 5)
            seed_y = np.random.randint(5, self.ny - 5)
            
            # Random seed size (C-rate dependent: smaller at high rates)
            if self.c_rate > 1.0:
                seed_size = np.random.randint(1, 3)
            else:
                seed_size = np.random.randint(2, 5)
            
            self.seed_positions.append((seed_x, seed_y, seed_size))
            
            # Create seed
            x_min = max(0, seed_x - seed_size)
            x_max = min(self.nx, seed_x + seed_size)
            y_min = max(0, seed_y - seed_size)
            y_max = min(self.ny, seed_y + seed_size)
            
            self.c[x_min:x_max, y_min:y_max] = target_composition
        
        return active_seeds
    
    def initialize_lithiation_multi_seeds(self):
        """Initialize for lithiation with multiple seeds"""
        # Start with FePO₄
        self.c = self.scales.c_alpha * np.ones((self.nx, self.ny))
        
        # Create multiple LiFePO₄ seeds
        active_seeds = self.create_multiple_seeds(self.scales.c_beta)
        
        # Apply electric potential gradient
        self.phi = np.zeros_like(self.c)
        for i in range(self.nx):
            self.phi[i, :] = -0.1 * (i / self.nx)  # Negative gradient for insertion
        
        # Reset time
        self.time_dim = 0.0
        self.time_phys = 0.0
        self.step = 0
        self.clear_history()
        
        print(f"Lithiation initialized with {active_seeds}/{self.n_seeds} active seeds at {self.c_rate}C")
    
    def initialize_delithiation_multi_seeds(self):
        """Initialize for delithiation with multiple seeds"""
        # Start with LiFePO₄
        self.c = self.scales.c_beta * np.ones((self.nx, self.ny))
        
        # Create multiple FePO₄ seeds
        active_seeds = self.create_multiple_seeds(self.scales.c_alpha)
        
        # Apply reverse electric potential gradient
        self.phi = np.zeros_like(self.c)
        for i in range(self.nx):
            self.phi[i, :] = 0.1 * (i / self.nx)  # Positive gradient for extraction
        
        # Reset time
        self.time_dim = 0.0
        self.time_phys = 0.0
        self.step = 0
        self.clear_history()
        
        print(f"Delithiation initialized with {active_seeds}/{self.n_seeds} active seeds at {self.c_rate}C")
    
    def initialize_random_multi_seeds(self, c0=0.5):
        """Initialize random with multiple seeds"""
        self.c = c0 * np.ones((self.nx, self.ny))
        
        # Add random composition seeds
        for seed_idx in range(self.n_seeds):
            seed_x = np.random.randint(10, self.nx - 10)
            seed_y = np.random.randint(10, self.ny - 10)
            seed_size = np.random.randint(2, 5)
            
            self.seed_positions.append((seed_x, seed_y, seed_size))
            
            # Random composition for each seed
            seed_comp = np.random.uniform(0.2, 0.8)
            
            x_min = max(0, seed_x - seed_size)
            x_max = min(self.nx, seed_x + seed_size)
            y_min = max(0, seed_y - seed_size)
            y_max = min(self.ny, seed_y + seed_size)
            
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
            'mean_phi': [],
            'voltage': [],
            'phase_FePO4': [],
            'phase_LiFePO4': [],
            'nuclei_active': [],
            'interface_length': []
        }
        self.update_history()
    
    def update_history(self):
        """Update history statistics"""
        self.history['time_phys'].append(self.time_phys)
        self.history['mean_c'].append(np.mean(self.c))
        self.history['std_c'].append(np.std(self.c))
        self.history['mean_phi'].append(np.mean(self.phi))
        
        # Voltage (same as original)
        voltage = np.mean(self.phi[-1, :]) - np.mean(self.phi[0, :])
        self.history['voltage'].append(voltage)
        
        # Phase fractions (same as original)
        threshold = 0.5
        self.history['phase_FePO4'].append(np.sum(self.c < threshold) / (self.nx * self.ny))
        self.history['phase_LiFePO4'].append(np.sum(self.c >= threshold) / (self.nx * self.ny))
        
        # Count active nuclei (seeds that are growing)
        active_nuclei = 0
        for seed_x, seed_y, size in self.seed_positions:
            # Check if seed region has composition different from surroundings
            seed_region = self.c[max(0,seed_x-2):min(self.nx,seed_x+2), 
                                 max(0,seed_y-2):min(self.ny,seed_y+2)]
            if np.std(seed_region) > 0.1:  # Active if there's variation
                active_nuclei += 1
        self.history['nuclei_active'].append(active_nuclei)
        
        # Interface length (measure of phase boundary complexity)
        grad_x = np.abs(np.gradient(self.c, axis=0))
        grad_y = np.abs(np.gradient(self.c, axis=1))
        interface = np.sqrt(grad_x**2 + grad_y**2)
        self.history['interface_length'].append(np.sum(interface > 0.1))
    
    def run_step(self):
        """Run one time step with electrostatics and kinetics"""
        # Reference concentration for Poisson equation
        c_ref = np.mean(self.c)
        
        # Solve Poisson equation (same as original)
        self.phi = solve_poisson_periodic(
            self.phi, self.c, c_ref, self.dx,
            self.scales.ε, self.scales.F,
            max_iter=50, tol=1e-4
        )
        
        # Compute electric field (same as original)
        self.Ex, self.Ey = compute_electric_field(self.phi, self.dx)
        
        # Choose kinetics parameters based on C-rate
        if self.kinetics_type == 0:  # BV
            k0 = self.scales.k0_bv * self.scales.c_rate_factor
        else:  # MHC
            k0 = self.scales.k0_mhc * self.scales.c_rate_factor
        
        # Update concentration with kinetics
        self.c = update_concentration_with_kinetics(
            self.c, self.phi, self.dt, self.dx,
            self.kappa_dim, self.M_dim, self.scales.D_b * self.scales.D_factor,
            self.A, self.B, self.C,
            self.scales.z, self.scales.F, self.scales.R, self.scales.T,
            self.kinetics_type, k0, self.eta_left
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
    
    def get_seeds_info(self):
        """Get information about seeds"""
        seeds_info = []
        
        for idx, (seed_x, seed_y, size) in enumerate(self.seed_positions):
            # Get seed region
            x_min = max(0, seed_x - size)
            x_max = min(self.nx, seed_x + size)
            y_min = max(0, seed_y - size)
            y_max = min(self.ny, seed_y + size)
            
            seed_region = self.c[x_min:x_max, y_min:y_max]
            mean_comp = np.mean(seed_region)
            std_comp = np.std(seed_region)
            
            # Determine if growing
            is_growing = std_comp > 0.1
            
            seeds_info.append({
                'id': idx,
                'position': (seed_x, seed_y),
                'mean_composition': mean_comp,
                'std_composition': std_comp,
                'is_growing': is_growing,
                'size': size
            })
        
        return seeds_info
    
    def get_statistics(self):
        """Get comprehensive statistics"""
        seeds_info = self.get_seeds_info()
        active_seeds = sum(1 for s in seeds_info if s['is_growing'])
        
        stats = {
            # Time
            'time_phys': self.time_phys,
            'step': self.step,
            'c_rate': self.c_rate,
            
            # Concentration
            'mean_c': np.mean(self.c),
            'std_c': np.std(self.c),
            'x_Li': np.mean(self.c),  # Same as original
            
            # Electric field (same as original)
            'mean_phi': np.mean(self.phi),
            'max_phi': np.max(self.phi),
            'min_phi': np.min(self.phi),
            'mean_E': np.mean(np.sqrt(self.Ex**2 + self.Ey**2)),
            
            # Voltage (same as original)
            'voltage': np.mean(self.phi[-1, :]) - np.mean(self.phi[0, :]),
            
            # Phase fractions (same as original)
            'phase_FePO4': np.sum(self.c < 0.5) / (self.nx * self.ny),
            'phase_LiFePO4': np.sum(self.c >= 0.5) / (self.nx * self.ny),
            
            # Seeds
            'n_seeds': len(self.seed_positions),
            'active_seeds': active_seeds,
            
            # Interface
            'interface_length': self.history['interface_length'][-1] if self.history['interface_length'] else 0,
            
            # Kinetics
            'kinetics_type': 'BV' if self.kinetics_type == 0 else 'MHC',
            'overpotential': self.eta_left,
            
            # Physical parameters (same as original)
            'domain_size_nm': self.nx * self.dx * self.scales.L0 * 1e9,
            'interface_width_nm': np.sqrt(self.kappa_phys / self.W_phys) * 1e9,
            'debye_length_nm': self.scales.λ_D * 1e9,
            
            # Dimensionless parameters (same as original)
            'W_dim': self.W_dim,
            'kappa_dim': self.kappa_dim,
            'M_dim': self.M_dim,
            
            # Physical parameters (same as original)
            'W_phys': self.W_phys,
            'kappa_phys': self.kappa_phys,
            'M_phys': self.M_phys,
            'dt_phys': self.dt_phys,
        }
        
        return stats

# =====================================================
# STREAMLIT APP WITH C-RATE CONTROL
# =====================================================
def main():
    st.set_page_config(
        page_title="LiFePO₄ Single Particle with C-Rate",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 LiFePO₄ Single Particle Simulation with C-Rate Control")
    st.markdown("""
    ### Enhanced Single Particle Model with:
    - **C-rate parameter** (0.1C to 10C)
    - **Multiple nucleation seeds**
    - **BV and MHC kinetics options**
    - **Rate-dependent phase transformation**
    
    *Maintains original code's numerical scaling*
    """)
    
    # Initialize simulation
    if 'enhanced_sim' not in st.session_state:
        st.session_state.enhanced_sim = EnhancedSingleParticleSimulation(
            nx=256, ny=256, dx=1.0, dt=0.01, c_rate=1.0
        )
    
    sim = st.session_state.enhanced_sim
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚡ C-Rate Control")
        
        # C-rate selection
        c_rate = st.select_slider(
            "Charging Rate (C)",
            options=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            value=sim.c_rate,
            help="0.1C = slow (10 hours), 5C = fast (12 minutes)"
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
        st.subheader("🌱 Nucleation Seeds")
        n_seeds = st.slider(
            "Number of seeds",
            min_value=1,
            max_value=10,
            value=sim.n_seeds,
            help="Initial nucleation sites"
        )
        
        st.divider()
        
        # Simulation control
        col1, col2 = st.columns(2)
        with col1:
            steps = st.number_input("Steps/update", 1, 200, 10)
        
        with col2:
            if st.button("▶️ Run Steps", use_container_width=True):
                with st.spinner("Running simulation..."):
                    sim.run_steps(steps)
                    st.rerun()
        
        st.divider()
        
        # Initialization
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
        
        # Original parameters (same as original code)
        st.subheader("🎛️ Model Parameters")
        W_dim = st.slider("W (Double-well)", 0.1, 5.0, float(sim.W_dim), 0.1)
        kappa_dim = st.slider("κ (Gradient)", 0.1, 10.0, float(sim.kappa_dim), 0.1)
        M_dim = st.slider("M (Mobility)", 0.01, 5.0, float(sim.M_dim), 0.01)
        dt_dim = st.slider("Δt (Time step)", 0.001, 0.1, float(sim.dt), 0.001)
        
        # Update original parameters
        if (W_dim != sim.W_dim or kappa_dim != sim.kappa_dim or 
            M_dim != sim.M_dim or dt_dim != sim.dt):
            sim.set_parameters(W_dim=W_dim, kappa_dim=kappa_dim, M_dim=M_dim, dt_dim=dt_dim)
        
        st.divider()
        
        # Display statistics
        st.subheader("📊 Current State")
        stats = sim.get_statistics()
        
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("Time", f"{stats['time_phys']:.2e} s")
            st.metric("x in LiₓFePO₄", f"{stats['mean_c']:.3f}")
            st.metric("Voltage", f"{stats['voltage']:.3f} V")
            st.metric("Active Seeds", f"{stats['active_seeds']}/{stats['n_seeds']}")
        
        with col_stat2:
            st.metric("FePO₄", f"{stats['phase_FePO4']:.3f}")
            st.metric("LiFePO₄", f"{stats['phase_LiFePO4']:.3f}")
            st.metric("C-Rate", f"{stats['c_rate']}C")
            st.metric("Kinetics", stats['kinetics_type'])
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Concentration", "Potential & Field", "Statistics", "Nucleation Analysis"
    ])
    
    with tab1:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(f"Concentration Field at {sim.c_rate}C")
            
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
            
            ax1.set_title(f"LiₓFePO₄ Concentration\n"
                         f"Time: {stats['time_phys']:.2e} s, "
                         f"C-rate: {sim.c_rate}C, "
                         f"Seeds: {stats['active_seeds']}/{stats['n_seeds']}")
            ax1.set_xlabel(f"x ({domain_nm:.0f} nm)")
            ax1.set_ylabel(f"y ({domain_nm:.0f} nm)")
            plt.colorbar(im1, ax=ax1, label='x in LiₓFePO₄')
            st.pyplot(fig1)
            plt.close(fig1)
        
        with col2:
            st.subheader("Phase Fractions")
            
            fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
            labels = ['FePO₄-rich', 'LiFePO₄-rich']
            sizes = [stats['phase_FePO4'], stats['phase_LiFePO4']]
            colors = ['#ff6b6b', '#4ecdc4']
            ax_pie.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax_pie.axis('equal')
            st.pyplot(fig_pie)
            plt.close(fig_pie)
            
            st.subheader("Seeds Status")
            seeds_data = []
            for seed in seeds_info:
                seeds_data.append({
                    'ID': seed['id'],
                    'Position': f"({seed['position'][0]}, {seed['position'][1]})",
                    'Mean x': f"{seed['mean_composition']:.3f}",
                    'Status': 'Growing' if seed['is_growing'] else 'Dormant'
                })
            
            df_seeds = pd.DataFrame(seeds_data)
            st.dataframe(df_seeds, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Electric Potential")
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            im2 = ax2.imshow(sim.phi.T, cmap='RdBu_r', origin='lower', aspect='auto')
            ax2.set_title(f"Electric Potential φ")
            ax2.set_xlabel("x position")
            ax2.set_ylabel("y position")
            plt.colorbar(im2, ax=ax2, label='Potential φ (V)')
            st.pyplot(fig2)
            plt.close(fig2)
        
        with col2:
            st.subheader("Electric Field Magnitude")
            E_mag = np.sqrt(sim.Ex**2 + sim.Ey**2)
            fig3, ax3 = plt.subplots(figsize=(6, 5))
            im3 = ax3.imshow(E_mag.T, cmap='plasma', origin='lower', aspect='auto')
            ax3.set_title("Electric Field |E|")
            ax3.set_xlabel("x position")
            ax3.set_ylabel("y position")
            plt.colorbar(im3, ax=ax3, label='|E| (V/m)')
            st.pyplot(fig3)
            plt.close(fig3)
    
    with tab3:
        st.subheader("Time Evolution")
        
        if len(sim.history['time_phys']) > 1:
            fig4, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            axes[0, 0].plot(sim.history['time_phys'], sim.history['mean_c'], 'b-', linewidth=2)
            axes[0, 0].set_xlabel("Time (s)")
            axes[0, 0].set_ylabel("Mean x")
            axes[0, 0].set_title("Concentration Evolution")
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].plot(sim.history['time_phys'], sim.history['voltage'], color='orange', linewidth=2)
            axes[0, 1].set_xlabel("Time (s)")
            axes[0, 1].set_ylabel("Voltage (V)")
            axes[0, 1].set_title("Voltage Profile")
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[1, 0].plot(sim.history['time_phys'], sim.history['phase_FePO4'], 'r-', 
                           label='FePO₄', linewidth=2)
            axes[1, 0].plot(sim.history['time_phys'], sim.history['phase_LiFePO4'], 'g-', 
                           label='LiFePO₄', linewidth=2)
            axes[1, 0].set_xlabel("Time (s)")
            axes[1, 0].set_ylabel("Phase Fraction")
            axes[1, 0].set_title("Phase Evolution")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].plot(sim.history['time_phys'], sim.history['nuclei_active'], 'purple', linewidth=2)
            axes[1, 1].set_xlabel("Time (s)")
            axes[1, 1].set_ylabel("Active Nuclei")
            axes[1, 1].set_title("Active Nuclei Evolution")
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)
        else:
            st.info("Run simulation to see evolution")
    
    with tab4:
        st.subheader("C-Rate Effects Analysis")
        
        # Create comparison for different C-rates
        st.markdown(f"""
        ### Expected C-Rate Effects at {sim.c_rate}C:
        
        **Low C-rate ({sim.c_rate}C if ≤1):**
        - Multiple seeds grow simultaneously
        - Gradual phase transformation
        - Lower overpotential required
        - More homogeneous transformation
        
        **High C-rate ({sim.c_rate}C if >1):**
        - Fewer active seeds
        - Rapid phase boundary movement
        - Higher overpotential required
        - More sequential/heterogeneous transformation
        
        **Current Parameters:**
        - Overpotential: {sim.eta_left:.4f} V
        - Interface sharpness factor: {sim.scales.kappa_factor:.2f}
        - Diffusion factor: {sim.scales.D_factor:.2f}
        - Nucleation probability: {sim.scales.nucleation_probability:.1f}
        """)
        
        # Interface analysis
        grad_x = np.gradient(sim.c, axis=0)
        grad_y = np.gradient(sim.c, axis=1)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        fig5, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(grad_mag.T, cmap='hot', origin='lower', aspect='auto')
        axes[0].set_title("Gradient Magnitude (Interface)")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        
        # Histogram of concentration
        axes[1].hist(sim.c.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[1].axvline(sim.scales.c_alpha, color='red', linestyle='--', label='FePO₄')
        axes[1].axvline(sim.scales.c_beta, color='green', linestyle='--', label='LiFePO₄')
        axes[1].set_xlabel('x in LiₓFePO₄')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Concentration Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)
    
    # Physics explanation
    with st.expander("📚 C-Rate in Single Particle Simulation", expanded=True):
        st.markdown(f"""
        ### How C-Rate Affects Single Particle Behavior:
        
        **1. Overpotential Scaling:**
        ```
        η ∝ log(C-rate)  (Butler-Volmer equation)
        ```
        - Low C-rate (0.1C): η ≈ 10 mV
        - High C-rate (5C): η ≈ 50-100 mV
        
        **2. Nucleation Probability:**
        - **High C-rate**: Fewer active seeds (competition limited)
        - **Low C-rate**: More simultaneous nucleation
        
        **3. Interface Dynamics:**
        - **High C-rate**: Sharper interface (κ factor = {sim.scales.kappa_factor:.2f})
        - **Low C-rate**: Diffuser interface (κ factor = 1.0)
        
        **4. Effective Diffusion:**
        ```
        D_eff = D₀ / (1 + α·√C-rate)
        ```
        - Accounts for kinetic limitations at high rates
        
        **5. Transformation Time:**
        - **1C**: Complete in ~1 hour (theoretical)
        - **5C**: Complete in ~12 minutes
        - **0.1C**: Complete in ~10 hours
        
        ### Numerical Implementation:
        - **Same dimensionless scaling** as original code
        - **C-rate multiplies** key parameters
        - **Physical time scaling** preserved
        - **Consistent with** experimental paper observations
        """)

if __name__ == "__main__":
    main()
