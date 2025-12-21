import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO
import pandas as pd

########################################################################
# Involved models for lithiation and delithiation procedure
################################################################################

# =====================================================
# PHYSICAL CONSTANTS AND SCALES FOR LiFePO₄ WITH ELECTROSTATICS AND C-RATE
# =====================================================

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
       
        print(f"Physical scales with electrostatics at {c_rate}C:")
        print(f" Length scale L0 = {self.L0:.2e} m ({self.L0*1e9:.1f} nm)")
        print(f" Time scale t0 = {self.t0:.2e} s")
        print(f" Debye length λ_D = {self.λ_D*1e9:.2f} nm")
        print(f" Permittivity ε = {self.ε:.2e} F/m")
        print(f" Overpotential scale: {self.eta_scale:.3f} V")
        print(f" C-rate multiplier: {self.c_rate_factor:.2f}")
       
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


# =====================================================
# NUMBA-ACCELERATED FUNCTIONS WITH ELECTROSTATICS
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
    k_f = k0 * np.exp(-alpha * F * eta / (R * T))
    k_b = k0 * np.exp((1 - alpha) * F * eta / (R * T))
    flux = k_f * (1.0 - c_surf) - k_b * c_surf
    return flux

@njit(fastmath=True)
def marcus_hush_chidsey_flux(eta, c_surf, k0, F, R, T):
    """Simplified MHC kinetics"""
    eta_dim = F * eta / (R * T)
    flux = k0 * np.tanh(eta_dim / 2.0) * (1.0 - c_surf)
    return flux

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
    """
    Update concentration: base is extended Cahn-Hilliard with optional electrostatics.
    For BV/MHC, remove migration and add boundary kinetics.
    """
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
# ELECTROCHEMICAL PHASE FIELD SIMULATION
# =====================================================

class ElectrochemicalPhaseFieldSimulation:
    """Phase field simulation with electrostatics for LiFePO₄"""
   
    def __init__(self, nx=256, ny=256, dx=1.0, dt=0.01, c_rate=1.0):
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
# STREAMLIT APP WITH ELECTROSTATICS
# =====================================================

def main():
    st.set_page_config(
        page_title="LiFePO₄ Electrochemical Phase Field",
        page_icon="⚡",
        layout="wide"
    )
   
    # Title
    st.title("⚡ LiFePO₄ Electrochemical Phase Field with Electrostatics and C-Rate")
    st.markdown("""
    ### Cahn-Hilliard + Poisson-Nernst-Planck Model for Battery Electrodes
   
    This simulation adds **electrostatics** to the phase field model, solving coupled
    Poisson equation and extended Cahn-Hilliard equation for realistic electrochemistry.
    Extended with C-rate control and optional BV/MHC kinetics instead of PNP migration.
    """)
   
    # Initialize simulation
    if 'sim' not in st.session_state:
        st.session_state.sim = ElectrochemicalPhaseFieldSimulation(nx=256, ny=256, dx=1.0, dt=0.01, c_rate=1.0)
   
    sim = st.session_state.sim
    
    # Store initial concentration for random case
    if 'initial_c0' not in st.session_state:
        st.session_state.initial_c0 = 0.5
   
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Simulation Controls")
       
        # Display physical parameters
        with st.expander("⚡ Electrostatic Parameters", expanded=False):
            stats = sim.get_statistics()
            st.markdown(f"""
            **Debye Length:** {stats['debye_length_nm']:.2f} nm
            **Permittivity ε:** {sim.scales.ε:.2e} F/m
            **Thermal Voltage:** {sim.scales.φ0:.3f} V
            **Li⁺ Charge:** z = {sim.scales.z}+
            **Diffusion Coef.:** {sim.scales.D_b:.2e} m²/s
            """)
       
        st.divider()
       
        # C-rate selection
        c_rate = st.select_slider(
            "C-Rate",
            options=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            value=sim.c_rate
        )
       
        # Kinetics selection
        kinetics_option = st.selectbox(
            "Kinetics Type",
            ['Poisson-Nernst-Planck (PNP)', 'Butler-Volmer (BV)', 'Marcus-Hush-Chidsey (MHC)'],
            index=0
        )
        kinetics_type = 0 if kinetics_option == 'Poisson-Nernst-Planck (PNP)' else \
                       1 if kinetics_option == 'Butler-Volmer (BV)' else 2
       
        # Random seed
        random_seed = st.number_input("Random Seed (optional)", value=None, step=1)
        
        # Initial concentration slider for random case
        initial_c0 = st.slider(
            "Initial x in LiₓFePO₄ (for Random case)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.initial_c0,
            step=0.01,
            help="Initial lithium concentration for the random initialization"
        )
        st.session_state.initial_c0 = initial_c0
       
        st.divider()
        
        # Simulation time control
        st.subheader("⏱️ Simulation Time")
        
        col_time1, col_time2 = st.columns(2)
        with col_time1:
            steps = st.number_input("Steps per run", 1, 2000, 100)
        
        with col_time2:
            if st.button("▶️ Run Steps", use_container_width=True):
                with st.spinner("Running..."):
                    sim.run_steps(steps)
                    st.rerun()
        
        # Run to specific time
        target_time = st.number_input(
            "Run to time (s)",
            min_value=0.0,
            max_value=1e6,
            value=min(1e5, sim.time_phys * 10),
            step=1000.0,
            format="%.1e"
        )
        
        if st.button("⏱️ Run to Time", use_container_width=True):
            with st.spinner(f"Running to time {target_time:.2e} s..."):
                sim.run_until(target_time)
                st.rerun()
       
        st.divider()
       
        # Initialization
        st.subheader("🔄 Initialization")
        init_option = st.radio(
            "Choose scenario:",
            ["Random (No Bias)", "Lithiation (Charge)", "Delithiation (Discharge)"],
            index=0
        )
        
        # Noise amplitude control
        noise_amplitude = st.slider(
            "Noise Amplitude",
            min_value=0.0,
            max_value=0.2,
            value=0.05,
            step=0.01,
            help="Amplitude of random noise added to initial condition"
        )
       
        if st.button("🔄 Apply Initialization", use_container_width=True):
            if init_option == "Random (No Bias)":
                sim.initialize_random(c0=initial_c0, noise_amplitude=noise_amplitude, seed=random_seed)
            elif init_option == "Lithiation (Charge)":
                sim.initialize_lithiation(noise_amplitude=noise_amplitude, seed=random_seed)
            else:
                sim.initialize_delithiation(noise_amplitude=noise_amplitude, seed=random_seed)
            sim.set_parameters(c_rate=c_rate, kinetics_type=kinetics_type)
            st.rerun()
       
        st.divider()
       
        # Parameter controls
        st.subheader("🎛️ Model Parameters")
       
        W_dim = st.slider("W (Double-well)", 0.1, 5.0, float(sim.W_dim), 0.1)
        kappa_dim = st.slider("κ (Gradient)", 0.1, 10.0, float(sim.kappa_dim), 0.1)
        M_dim = st.slider("M (Mobility)", 0.01, 5.0, float(sim.M_dim), 0.01)
        dt_dim = st.slider("Δt (Time step)", 0.001, 0.1, float(sim.dt), 0.001)
       
        # Electric field strength
        phi_scale = st.slider("Electric Field Scale", 0.0, 0.5, 0.1, 0.01,
                             help="Controls strength of applied electric potential")
       
        # Update parameters
        if (W_dim != sim.W_dim or kappa_dim != sim.kappa_dim or
            M_dim != sim.M_dim or dt_dim != sim.dt or
            c_rate != sim.c_rate or kinetics_type != sim.kinetics_type):
            sim.set_parameters(W_dim=W_dim, kappa_dim=kappa_dim, M_dim=M_dim, 
                             dt_dim=dt_dim, c_rate=c_rate, kinetics_type=kinetics_type)
       
        st.divider()
       
        # Statistics
        st.subheader("📊 Current State")
        stats = sim.get_statistics()
       
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("Time", f"{stats['time_phys']:.2e} s")
            st.metric("x in LiₓFePO₄", f"{stats['mean_c']:.3f}")
            st.metric("Voltage", f"{stats['voltage']:.3f} V")
            st.metric("Mean φ", f"{stats['mean_phi']:.3f} V")
       
        with col_stat2:
            st.metric("Mean |E|", f"{stats['mean_E']:.2e} V/m")
            st.metric("FePO₄", f"{stats['phase_FePO4']:.3f}")
            st.metric("LiFePO₄", f"{stats['phase_LiFePO4']:.3f}")
            st.metric("Debye Length", f"{stats['debye_length_nm']:.2f} nm")
            st.metric("C-Rate", f"{stats['c_rate']}C")
            st.metric("Kinetics", ['PNP', 'BV', 'MHC'][stats['kinetics_type']])
            
        # Progress bar for simulation
        max_sim_time = 1e6  # Maximum simulation time in seconds
        progress = min(1.0, stats['time_phys'] / max_sim_time)
        st.progress(progress, text=f"Simulation Progress: {progress*100:.1f}%")
   
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Concentration", "Potential", "Electric Field", "Statistics"])
   
    with tab1:
        col1, col2 = st.columns([3, 1])
       
        with col1:
            st.subheader("Lithium Concentration Field")
           
            # Calculate physical dimensions
            domain_nm = sim.nx * sim.dx * sim.scales.L0 * 1e9
           
            fig1, ax1 = plt.subplots(figsize=(8, 7))
            im1 = ax1.imshow(sim.c.T, cmap='RdYlBu', origin='lower',
                           vmin=0, vmax=1, aspect='auto')
            ax1.set_title(f"LiₓFePO₄ Concentration\n"
                         f"Time = {stats['time_phys']:.2e} s, "
                         f"x = {stats['mean_c']:.3f}")
            ax1.set_xlabel(f"x ({domain_nm:.0f} nm)")
            ax1.set_ylabel(f"y ({domain_nm:.0f} nm)")
            plt.colorbar(im1, ax=ax1, label='x in LiₓFePO₄')
            st.pyplot(fig1)
            plt.close(fig1)
       
        with col2:
            st.subheader("Phase Fractions")
           
            # Phase fraction pie chart
            fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
            labels = ['FePO₄-rich', 'LiFePO₄-rich']
            sizes = [stats['phase_FePO4'], stats['phase_LiFePO4']]
            colors = ['#ff6b6b', '#4ecdc4']
            ax_pie.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax_pie.axis('equal')
            st.pyplot(fig_pie)
            plt.close(fig_pie)
           
            # Histogram
            st.subheader("Concentration Distribution")
            fig_hist, ax_hist = plt.subplots(figsize=(4, 3))
            ax_hist.hist(sim.c.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax_hist.axvline(sim.scales.c_alpha, color='red', linestyle='--', alpha=0.7, label='FePO₄')
            ax_hist.axvline(sim.scales.c_beta, color='green', linestyle='--', alpha=0.7, label='LiFePO₄')
            ax_hist.set_xlim(0, 1)
            ax_hist.set_xlabel('x in LiₓFePO₄')
            ax_hist.set_ylabel('Frequency')
            ax_hist.legend()
            ax_hist.grid(True, alpha=0.3)
            st.pyplot(fig_hist)
            plt.close(fig_hist)
   
    with tab2:
        st.subheader("Electric Potential Field")
       
        fig2, ax2 = plt.subplots(figsize=(8, 7))
        im2 = ax2.imshow(sim.phi.T, cmap='RdBu_r', origin='lower', aspect='auto')
        ax2.set_title(f"Electric Potential φ\n"
                     f"Mean = {stats['mean_phi']:.3f} V, "
                     f"Range = [{stats['min_phi']:.3f}, {stats['max_phi']:.3f}] V")
        ax2.set_xlabel("x position")
        ax2.set_ylabel("y position")
        plt.colorbar(im2, ax=ax2, label='Potential φ (V)')
        st.pyplot(fig2)
        plt.close(fig2)
   
    with tab3:
        col1, col2 = st.columns(2)
       
        with col1:
            st.subheader("Electric Field Magnitude")
           
            E_mag = np.sqrt(sim.Ex**2 + sim.Ey**2)
           
            fig3a, ax3a = plt.subplots(figsize=(6, 5))
            im3a = ax3a.imshow(E_mag.T, cmap='plasma', origin='lower', aspect='auto')
            ax3a.set_title("Electric Field Magnitude |E|")
            ax3a.set_xlabel("x position")
            ax3a.set_ylabel("y position")
            plt.colorbar(im3a, ax=ax3a, label='|E| (V/m)')
            st.pyplot(fig3a)
            plt.close(fig3a)
       
        with col2:
            st.subheader("Electric Field Vectors")
           
            # Subsample for clearer visualization
            subsample = 8
            X, Y = np.meshgrid(np.arange(0, sim.nx, subsample),
                             np.arange(0, sim.ny, subsample))
            Ex_sub = sim.Ex[::subsample, ::subsample].T
            Ey_sub = sim.Ey[::subsample, ::subsample].T
           
            fig3b, ax3b = plt.subplots(figsize=(6, 5))
            ax3b.quiver(X, Y, Ex_sub, Ey_sub,
                       np.sqrt(Ex_sub**2 + Ey_sub**2),
                       cmap='viridis', scale=50, scale_units='inches')
            ax3b.set_title("Electric Field Vectors")
            ax3b.set_xlabel("x position")
            ax3b.set_ylabel("y position")
            ax3b.set_xlim(0, sim.nx)
            ax3b.set_ylim(0, sim.ny)
            plt.colorbar(ax3b.collections[0], ax=ax3b, label='|E| (V/m)')
            st.pyplot(fig3b)
            plt.close(fig3b)
   
    with tab4:
        st.subheader("Time Evolution Statistics")
       
        if len(sim.history['time_phys']) > 1:
            fig4, axes = plt.subplots(2, 2, figsize=(12, 8))
           
            # Concentration evolution
            axes[0, 0].plot(sim.history['time_phys'], sim.history['mean_c'], 'b-', linewidth=2)
            axes[0, 0].set_xlabel("Time (s)")
            axes[0, 0].set_ylabel("Mean x in LiₓFePO₄")
            axes[0, 0].set_title("Lithium Content Evolution")
            axes[0, 0].grid(True, alpha=0.3)
           
            # Voltage evolution
            axes[0, 1].plot(sim.history['time_phys'], sim.history['voltage'], color='orange', linewidth=2)
            axes[0, 1].set_xlabel("Time (s)")
            axes[0, 1].set_ylabel("Voltage (V)")
            axes[0, 1].set_title("Voltage Evolution")
            axes[0, 1].grid(True, alpha=0.3)
           
            # Phase fractions
            axes[1, 0].plot(sim.history['time_phys'], sim.history['phase_FePO4'], 'r-',
                           label='FePO₄-rich', linewidth=2)
            axes[1, 0].plot(sim.history['time_phys'], sim.history['phase_LiFePO4'], 'g-',
                           label='LiFePO₄-rich', linewidth=2)
            axes[1, 0].set_xlabel("Time (s)")
            axes[1, 0].set_ylabel("Phase Fraction")
            axes[1, 0].set_title("Phase Evolution")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
           
            # Electric potential
            axes[1, 1].plot(sim.history['time_phys'], sim.history['mean_phi'], color='purple', linewidth=2)
            axes[1, 1].set_xlabel("Time (s)")
            axes[1, 1].set_ylabel("Mean φ (V)")
            axes[1, 1].set_title("Electric Potential Evolution")
            axes[1, 1].grid(True, alpha=0.3)
           
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)
           
            # Data table
            st.subheader("History Data")
            df = pd.DataFrame({
                'Time (s)': sim.history['time_phys'],
                'Mean x': sim.history['mean_c'],
                'Std x': sim.history['std_c'],
                'Voltage (V)': sim.history['voltage'],
                'Mean φ (V)': sim.history['mean_phi'],
                'FePO₄ fraction': sim.history['phase_FePO4'],
                'LiFePO₄ fraction': sim.history['phase_LiFePO4']
            })
            st.dataframe(df.tail(10))
           
            # Export
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="electrochemical_phase_field.csv",
                mime="text/csv"
            )
        else:
            st.info("Run simulation to see time evolution statistics.")
   
    # Physics explanation
    with st.expander("📚 Governing Equations with Electrostatics", expanded=False):
        st.markdown("""
        ### Extended Cahn-Hilliard Equation with Electrostatics or Kinetics
       
        The model couples **phase separation** with **electrostatics** or **kinetics**:
       
        1. **Modified Chemical Potential**:
           μ = ∂f/∂c - κ∇²c + zFφ (for PNP)
       
        2. **Flux for PNP**:
           J = -M∇μ - (DzF*c/(RT))∇φ
       
        3. **Flux for BV/MHC**:
           J = -M∇μ (internal), with boundary flux from BV or MHC
       
        4. **Conservation**:
           ∂c/∂t = -∇·J
       
        5. **Poisson Equation** (always):
           ∇²φ = -F/ε * (c - c_ref)
       
        ### Initial Conditions:
        - **Random (No Bias)**: Uniform x with random noise (x controlled by slider)
        - **Lithiation (Charge)**: Uniform FePO₄ (x≈0.03) + noise, negative electric field at left
        - **Delithiation (Discharge)**: Uniform LiFePO₄ (x≈0.97) + noise, positive electric field at left
       
        ### Key Improvements:
        - **No artificial seeds**: Lithiation/delithiation start with truly uniform compositions
        - **Exponential potential**: Stronger electric field at left boundary decays exponentially
        - **Proper nucleation**: Phase boundaries form naturally from noise under electric field
        - **Extended simulation time**: Run up to 1e6 seconds (≈11.5 days)
       
        ### C-Rate Effects:
        - **Low C-rate (≤1C)**: Gradual transformation, lower overpotential
        - **High C-rate (>1C)**: Faster transformation, higher overpotential
        """)

    # Auto-run option
    st.sidebar.divider()
    auto_run = st.sidebar.checkbox("Auto-run simulation", value=False)
    auto_speed = st.sidebar.slider("Auto-run speed (steps/sec)", 1, 100, 20)
   
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
                    status_text.text(f"Step {sim.step}, Time: {sim.time_phys:.2e} s, x: {np.mean(sim.c):.3f}")
               
                st.rerun()

if __name__ == "__main__":
    main()
