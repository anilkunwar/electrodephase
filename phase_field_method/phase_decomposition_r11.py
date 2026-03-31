import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO
import time
import sys

# =====================================================
# Physical Scales Class for LiFePO₄
# =====================================================

class PhysicalScalesLiFePO4:
    """
    Physical unit conversion for LiFePO₄ phase-field simulations.
    All scales based on literature values for LiFePO₄ cathode materials.
    
    References:
    - Malik et al., J. Electrochem. Soc. (2011) - LiFePO4 thermodynamics
    - Bai et al., J. Electrochem. Soc. (2011) - Phase-field modeling
    - Cogswell & Bazant, ACS Nano (2013) - Coherency strain effects
    """
    def __init__(self, L0_nm=10.0, D_b=1e-14, Omega_kJmol=12.0, T=298.15):
        # Fundamental constants
        self.R = 8.314462618          # J/(mol·K) - Universal gas constant
        self.T = T                    # K - Temperature (default: room temp)
        self.V_m = 4.38e-5            # m³/mol - Molar volume of LiFePO₄
        
        # Material parameters (user-configurable)
        self.D_b = D_b                # m²/s - Diffusion coefficient along b-axis
        self.Omega = Omega_kJmol * 1e3  # J/mol - Regular solution parameter
        
        # Characteristic scales (derived)
        self.L0 = L0_nm * 1e-9        # m - Reference length scale
        self.E0 = self.Omega / self.V_m   # J/m³ - Energy density scale
        self.t0 = self.L0**2 / self.D_b   # s - Diffusion time scale
        
        # Mobility scale: M₀ = D_b / E₀ for regular solution model
        self.M0 = self.D_b / self.E0      # m⁵/(J·s)
        
    def dim_to_phys(self, W_dim, kappa_dim, M_dim, dt_dim, dx_dim=1.0):
        """
        Convert dimensionless parameters to physical SI units.
        
        Parameters:
        -----------
        W_dim : float - Dimensionless double-well barrier height
        kappa_dim : float - Dimensionless gradient coefficient
        M_dim : float - Dimensionless mobility
        dt_dim : float - Dimensionless time step
        dx_dim : float - Dimensionless grid spacing (default: 1.0)
        
        Returns:
        --------
        tuple : (W_phys, kappa_phys, M_phys, dt_phys, dx_phys) in SI units
        """
        W_phys = W_dim * self.E0                      # J/m³
        kappa_phys = kappa_dim * self.E0 * self.L0**2 # J/m
        M_phys = M_dim * self.M0                       # m⁵/(J·s)
        dt_phys = dt_dim * self.t0                     # s
        dx_phys = dx_dim * self.L0                     # m
        return W_phys, kappa_phys, M_phys, dt_phys, dx_phys
    
    def phys_to_interface_width(self, kappa_phys, W_phys):
        """
        Estimate interface width from gradient energy and barrier height.
        ξ ≈ √(κ/W) - characteristic width of diffuse interface.
        
        Parameters:
        -----------
        kappa_phys : float - Gradient coefficient in J/m
        W_phys : float - Double-well barrier in J/m³
        
        Returns:
        --------
        float : Interface width in meters
        """
        if W_phys <= 0 or kappa_phys <= 0:
            return 1e-9  # Fallback: 1 nm
        return np.sqrt(kappa_phys / W_phys)
    
    def format_time(self, t_seconds):
        """Format physical time with appropriate SI prefix."""
        if not np.isfinite(t_seconds) or t_seconds < 0:
            return "0 s"
        if t_seconds < 1e-9:
            return f"{t_seconds*1e12:.2f} ps"
        elif t_seconds < 1e-6:
            return f"{t_seconds*1e9:.2f} ns"
        elif t_seconds < 1e-3:
            return f"{t_seconds*1e6:.2f} μs"
        elif t_seconds < 1.0:
            return f"{t_seconds*1e3:.2f} ms"
        elif t_seconds < 3600:
            return f"{t_seconds:.3f} s"
        elif t_seconds < 86400:
            return f"{t_seconds/3600:.3f} h"
        else:
            return f"{t_seconds/86400:.3f} d"
    
    def format_length(self, L_meters):
        """Format length with appropriate SI prefix."""
        if not np.isfinite(L_meters) or L_meters < 0:
            return "0 nm"
        if L_meters < 1e-10:
            return f"{L_meters*1e12:.2f} pm"
        elif L_meters < 1e-9:
            return f"{L_meters*1e10:.2f} Å"
        elif L_meters < 1e-6:
            return f"{L_meters*1e9:.2f} nm"
        elif L_meters < 1e-3:
            return f"{L_meters*1e6:.2f} μm"
        elif L_meters < 1.0:
            return f"{L_meters*1e3:.2f} mm"
        else:
            return f"{L_meters:.3f} m"
    
    def format_energy_density(self, E_Jm3):
        """Format energy density with appropriate units."""
        if not np.isfinite(E_Jm3):
            return "0 J/m³"
        if abs(E_Jm3) < 1e3:
            return f"{E_Jm3:.2e} J/m³"
        elif abs(E_Jm3) < 1e6:
            return f"{E_Jm3/1e3:.2f} kJ/m³"
        else:
            return f"{E_Jm3/1e6:.2f} MJ/m³"


# =====================================================
# Numba-accelerated Phase Field Functions (Physical Units)
# =====================================================

@njit(fastmath=True, cache=True)
def double_well_energy(c, A, B, C):
    """
    Generalized double-well free energy density.
    f(c) = A·c² + B·c³ + C·c⁴
    
    For standard symmetric double-well: A=W, B=-2W, C=W
    giving f(c) = W·c²(1-c)² with minima at c=0 and c=1.
    
    Parameters:
    -----------
    c : float or array - Concentration (0 ≤ c ≤ 1)
    A, B, C : float - Free energy coefficients in J/m³
    
    Returns:
    --------
    float or array - Free energy density in J/m³
    """
    return A * c**2 + B * c**3 + C * c**4


@njit(fastmath=True, cache=True)
def chemical_potential(c, A, B, C):
    """
    Chemical potential: μ = ∂f/∂c (variational derivative of free energy).
    
    μ(c) = 2A·c + 3B·c² + 4C·c³
    
    Parameters:
    -----------
    c : float or array - Concentration
    A, B, C : float - Free energy coefficients in J/m³
    
    Returns:
    --------
    float or array - Chemical potential in J/m³
    """
    return 2.0 * A * c + 3.0 * B * c**2 + 4.0 * C * c**3


@njit(fastmath=True, parallel=True)
def compute_laplacian(field, dx):
    """
    Compute 5-point stencil Laplacian with periodic boundary conditions.
    ∇²f ≈ [f(i+1,j) + f(i-1,j) + f(i,j+1) + f(i,j-1) - 4f(i,j)] / dx²
    
    Parameters:
    -----------
    field : 2D array - Scalar field to differentiate
    dx : float - Grid spacing in meters
    
    Returns:
    --------
    2D array - Laplacian of field (same units as field/m²)
    """
    nx, ny = field.shape
    lap = np.zeros_like(field)
    
    for i in prange(nx):
        for j in prange(ny):
            # Periodic boundary conditions via modulo
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
    """
    Compute x-component of gradient using central differences.
    ∂f/∂x ≈ [f(i+1,j) - f(i-1,j)] / (2·dx)
    
    Parameters:
    -----------
    field : 2D array - Scalar field
    dx : float - Grid spacing in meters
    
    Returns:
    --------
    2D array - x-gradient of field (units: field/m)
    """
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
    """
    Compute y-component of gradient using central differences.
    ∂f/∂y ≈ [f(i,j+1) - f(i,j-1)] / (2·dx)
    
    Parameters:
    -----------
    field : 2D array - Scalar field
    dx : float - Grid spacing in meters
    
    Returns:
    --------
    2D array - y-gradient of field (units: field/m)
    """
    nx, ny = field.shape
    grad_y = np.zeros_like(field)
    
    for i in prange(nx):
        for j in prange(ny):
            jp1 = (j + 1) % ny
            jm1 = (j - 1) % ny
            grad_y[i, j] = (field[i, jp1] - field[i, jm1]) / (2.0 * dx)
    
    return grad_y


@njit(fastmath=True, parallel=True)
def update_concentration_physical(c, dt, dx, kappa, M, A, B, C):
    """
    Update concentration field using Cahn-Hilliard equation in physical units.
    
    Governing equation:
    ∂c/∂t = ∇·[M ∇(∂f/∂c - κ∇²c)]
    
    where:
    - c: Li concentration (dimensionless, 0 ≤ c ≤ 1)
    - M: Mobility [m⁵/(J·s)]
    - κ: Gradient energy coefficient [J/m]
    - f(c): Free energy density [J/m³]
    - ∂f/∂c: Chemical potential contribution [J/m³]
    
    Algorithm:
    1. Compute ∇²c (Laplacian of concentration)
    2. Compute μ = ∂f/∂c - κ∇²c (chemical potential)
    3. Compute ∇μ (gradient of chemical potential)
    4. Compute flux J = -M∇μ (Fick's law with gradient energy)
    5. Compute ∇·J (divergence of flux)
    6. Update: c_new = c + dt·(∇·J)
    
    Parameters:
    -----------
    c : 2D array - Current concentration field
    dt : float - Time step in seconds
    dx : float - Grid spacing in meters
    kappa : float - Gradient coefficient in J/m
    M : float - Mobility in m⁵/(J·s)
    A, B, C : float - Free energy coefficients in J/m³
    
    Returns:
    --------
    2D array - Updated concentration field
    """
    nx, ny = c.shape
    
    # Step 1: Compute Laplacian of concentration
    lap_c = compute_laplacian(c, dx)
    
    # Step 2: Compute local chemical potential ∂f/∂c
    mu_local = chemical_potential(c, A, B, C)
    
    # Step 3: Full chemical potential including gradient term
    mu = mu_local - kappa * lap_c
    
    # Step 4: Compute gradient of chemical potential
    mu_x = compute_gradient_x(mu, dx)
    mu_y = compute_gradient_y(mu, dx)
    
    # Step 5: Compute flux (negative sign: diffusion downhill)
    flux_x = -M * mu_x
    flux_y = -M * mu_y
    
    # Step 6: Compute divergence of flux
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
    
    # Step 7: Forward Euler time integration
    return c + dt * div_flux


# =====================================================
# PhaseFieldSimulation Class (Physical Units)
# =====================================================

class PhaseFieldSimulation:
    """
    2D Phase-field simulation of LiₓFePO₄ spinodal decomposition.
    
    Implements the Cahn-Hilliard equation with physical units for:
    - Length: meters (displayed as nm/μm)
    - Time: seconds (auto-formatted)
    - Energy: J/m³ for free energy density
    - Transport: m²/s diffusivity, m⁵/(J·s) mobility
    
    Key features:
    - Numba JIT acceleration for performance
    - Periodic boundary conditions
    - Two initialization modes: spinodal (random) and nucleation (seed)
    - Real-time statistics and history tracking
    - Export functionality for figures and data
    """
    
    def __init__(self, nx=256, ny=256, dx_dim=1.0, dt_dim=0.01, 
                 L0_nm=10.0, D_b=1e-14, Omega_kJmol=12.0):
        """
        Initialize simulation with physical parameters.
        
        Parameters:
        -----------
        nx, ny : int - Grid dimensions (pixels)
        dx_dim : float - Dimensionless grid spacing (internal use)
        dt_dim : float - Dimensionless time step (internal use)
        L0_nm : float - Reference length scale in nanometers
        D_b : float - Diffusion coefficient in m²/s
        Omega_kJmol : float - Regular solution parameter in kJ/mol
        """
        # Grid parameters (dimensionless internally for numerical stability)
        self.nx = nx
        self.ny = ny
        self.dx_dim = dx_dim
        
        # Initialize physical scales first
        self.scales = PhysicalScalesLiFePO4(
            L0_nm=L0_nm, 
            D_b=D_b, 
            Omega_kJmol=Omega_kJmol
        )
        
        # Dimensionless model parameters (for numerical stability)
        self.W_dim = 1.0        # Double-well barrier (dimensionless)
        self.kappa_dim = 2.0    # Gradient coefficient (dimensionless)
        self.M_dim = 1.0        # Mobility (dimensionless)
        self.dt_dim = dt_dim    # Time step (dimensionless)
        
        # ⚠️ CRITICAL: Define free energy coefficients BEFORE calling _update_physical_params
        # This fixes the AttributeError from the original code
        self.A_dim = self.W_dim
        self.B_dim = -2.0 * self.W_dim
        self.C_dim = self.W_dim
        
        # Convert to physical parameters
        self._update_physical_params()
        
        # Initialize concentration field
        self.c = np.zeros((nx, ny), dtype=np.float64)
        
        # Time tracking (both dimensionless and physical)
        self.time_dim = 0.0
        self.time_phys = 0.0
        self.step = 0
        
        # History for plotting and analysis
        self.history = {
            'time_dim': [],
            'time_phys': [],
            'mean': [],
            'std': [],
            'phase_high': [],
            'phase_low': [],
            'energy': []
        }
        
    def _update_physical_params(self):
        """
        Convert dimensionless parameters to physical SI units.
        Called whenever model or material parameters change.
        """
        # Defensive: ensure free energy coefficients exist
        if not hasattr(self, 'A_dim'):
            self.A_dim = getattr(self, 'W_dim', 1.0)
        if not hasattr(self, 'B_dim'):
            self.B_dim = -2.0 * getattr(self, 'W_dim', 1.0)
        if not hasattr(self, 'C_dim'):
            self.C_dim = getattr(self, 'W_dim', 1.0)
        
        # Convert main parameters
        (self.W_phys, self.kappa_phys, self.M_phys, 
         self.dt_phys, self.dx_phys) = self.scales.dim_to_phys(
            self.W_dim, self.kappa_dim, self.M_dim, 
            self.dt_dim, self.dx_dim
        )
        
        # Convert free energy coefficients to physical units
        self.A_phys = self.A_dim * self.scales.E0
        self.B_phys = self.B_dim * self.scales.E0
        self.C_phys = self.C_dim * self.scales.E0
        
        # Update physical time if dimensionless time is set
        self.time_phys = self.time_dim * self.scales.t0
        
    def set_physical_parameters(self, W_Jm3=None, kappa_Jm=None, M_m5Js=None, dt_s=None,
                                L0_nm=None, D_b=None, Omega_kJmol=None):
        """
        Set physical parameters directly (converts to dimensionless internally).
        
        Parameters:
        -----------
        W_Jm3 : float - Double-well barrier in J/m³
        kappa_Jm : float - Gradient coefficient in J/m
        M_m5Js : float - Mobility in m⁵/(J·s)
        dt_s : float - Time step in seconds
        L0_nm : float - Reference length in nm (triggers scale recalculation)
        D_b : float - Diffusion coefficient in m²/s
        Omega_kJmol : float - Regular solution parameter in kJ/mol
        """
        # Update fundamental scales if material parameters changed
        scales_changed = False
        new_L0_nm = L0_nm if L0_nm is not None else self.scales.L0 * 1e9
        new_D_b = D_b if D_b is not None else self.scales.D_b
        new_Omega = Omega_kJmol if Omega_kJmol is not None else self.scales.Omega / 1e3
        
        if (L0_nm is not None and abs(L0_nm - self.scales.L0*1e9) > 1e-6) or \
           (D_b is not None and abs(D_b - self.scales.D_b) > 1e-20) or \
           (Omega_kJmol is not None and abs(Omega_kJmol - self.scales.Omega/1e3) > 1e-6):
            scales_changed = True
            old_L0 = self.scales.L0
            self.scales = PhysicalScalesLiFePO4(
                L0_nm=new_L0_nm,
                D_b=new_D_b,
                Omega_kJmol=new_Omega
            )
            # Adjust dimensionless kappa if L0 changed (κ scales with L0²)
            if L0_nm is not None and L0_nm != old_L0 * 1e9:
                ratio = (self.scales.L0 / old_L0)**2
                self.kappa_dim = self.kappa_dim / ratio
        
        # Convert physical → dimensionless for model parameters
        if W_Jm3 is not None and self.scales.E0 > 0:
            self.W_dim = W_Jm3 / self.scales.E0
            self.A_dim = self.W_dim
            self.B_dim = -2.0 * self.W_dim
            self.C_dim = self.W_dim
            
        if kappa_Jm is not None and self.scales.E0 > 0 and self.scales.L0 > 0:
            self.kappa_dim = kappa_Jm / (self.scales.E0 * self.scales.L0**2)
            
        if M_m5Js is not None and self.scales.M0 > 0:
            self.M_dim = M_m5Js / self.scales.M0
            
        if dt_s is not None and self.scales.t0 > 0:
            self.dt_dim = dt_s / self.scales.t0
            
        # Update all physical parameters
        self._update_physical_params()
    
    def set_dimensionless_parameters(self, W_dim=None, kappa_dim=None, M_dim=None, dt_dim=None):
        """
        Set dimensionless parameters directly (for advanced users/testing).
        
        Parameters:
        -----------
        W_dim : float - Dimensionless double-well barrier
        kappa_dim : float - Dimensionless gradient coefficient
        M_dim : float - Dimensionless mobility
        dt_dim : float - Dimensionless time step
        """
        if W_dim is not None:
            self.W_dim = W_dim
            self.A_dim = W_dim
            self.B_dim = -2.0 * W_dim
            self.C_dim = W_dim
        if kappa_dim is not None:
            self.kappa_dim = kappa_dim
        if M_dim is not None:
            self.M_dim = M_dim
        if dt_dim is not None:
            self.dt_dim = dt_dim
        self._update_physical_params()
    
    def initialize_random(self, c0=0.5, noise_amplitude=0.01, seed=None):
        """
        Initialize with random fluctuations for spinodal decomposition.
        
        Parameters:
        -----------
        c0 : float - Average initial concentration (0 ≤ c0 ≤ 1)
        noise_amplitude : float - Amplitude of random fluctuations
        seed : int or None - Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random initial condition, clipped to [0, 1]
        noise = noise_amplitude * (2.0 * np.random.random((self.nx, self.ny)) - 1.0)
        self.c = np.clip(c0 + noise, 0.0, 1.0)
        
        # Reset time and history
        self.time_dim = 0.0
        self.time_phys = 0.0
        self.step = 0
        self.clear_history()
        
    def initialize_seed(self, c0=0.3, seed_value=0.7, radius_grid=15, 
                      center=None, seed=None):
        """
        Initialize with circular seed for nucleation and growth.
        
        Parameters:
        -----------
        c0 : float - Background concentration
        seed_value : float - Concentration inside seed region
        radius_grid : float - Seed radius in grid units
        center : tuple or None - Seed center (x, y) in grid indices
        seed : int or None - Random seed
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Start with uniform background
        self.c = c0 * np.ones((self.nx, self.ny), dtype=np.float64)
        
        # Set seed center to domain center if not specified
        if center is None:
            center_x, center_y = self.nx // 2, self.ny // 2
        else:
            center_x, center_y = center
        
        # Create circular seed region
        for i in range(self.nx):
            for j in range(self.ny):
                dist_sq = (i - center_x)**2 + (j - center_y)**2
                if dist_sq < radius_grid**2:
                    # Smooth transition at seed boundary
                    weight = min(1.0, np.sqrt(dist_sq) / radius_grid)
                    self.c[i, j] = seed_value * (1 - weight) + c0 * weight
        
        # Reset time and history
        self.time_dim = 0.0
        self.time_phys = 0.0
        self.step = 0
        self.clear_history()
    
    def initialize_from_array(self, c_array, reset_time=True):
        """
        Initialize concentration from external array.
        
        Parameters:
        -----------
        c_array : 2D array - Concentration field (will be clipped to [0,1])
        reset_time : bool - Whether to reset time counter
        """
        self.c = np.clip(np.array(c_array, dtype=np.float64), 0.0, 1.0)
        if reset_time:
            self.time_dim = 0.0
            self.time_phys = 0.0
            self.step = 0
            self.clear_history()
    
    def clear_history(self):
        """Clear all history tracking arrays."""
        self.history = {
            'time_dim': [],
            'time_phys': [],
            'mean': [],
            'std': [],
            'phase_high': [],
            'phase_low': [],
            'energy': []
        }
        self.update_history()
    
    def update_history(self):
        """Record current state to history arrays."""
        self.history['time_dim'].append(self.time_dim)
        self.history['time_phys'].append(self.time_phys)
        self.history['mean'].append(float(np.mean(self.c)))
        self.history['std'].append(float(np.std(self.c)))
        self.history['phase_high'].append(float(np.sum(self.c > 0.5) / (self.nx * self.ny)))
        self.history['phase_low'].append(float(np.sum(self.c < 0.5) / (self.nx * self.ny)))
        
        # Compute and store average free energy density
        energy_density = double_well_energy(self.c, self.A_phys, self.B_phys, self.C_phys)
        self.history['energy'].append(float(np.mean(energy_density)))
    
    def run_step(self):
        """
        Execute one time step of the Cahn-Hilliard dynamics.
        
        Uses explicit Euler integration with Numba-accelerated stencil.
        Concentration is clipped to [0, 1] after each step for stability.
        """
        # Update concentration field
        self.c = update_concentration_physical(
            self.c, 
            self.dt_phys, 
            self.dx_phys,
            self.kappa_phys, 
            self.M_phys,
            self.A_phys, 
            self.B_phys, 
            self.C_phys
        )
        
        # Clip concentration to physical bounds [0, 1]
        self.c = np.clip(self.c, 0.0, 1.0)
        
        # Update time counters
        self.time_dim += self.dt_dim
        self.time_phys = self.time_dim * self.scales.t0
        self.step += 1
        
        # Record to history
        self.update_history()
    
    def run_steps(self, n_steps, progress_callback=None):
        """
        Execute multiple time steps with optional progress reporting.
        
        Parameters:
        -----------
        n_steps : int - Number of steps to run
        progress_callback : callable or None - Function(step, total) for progress
        """
        for step_idx in range(n_steps):
            self.run_step()
            if progress_callback is not None:
                progress_callback(step_idx + 1, n_steps)
    
    def compute_free_energy_density(self):
        """
        Compute free energy density field in J/m³.
        
        Returns:
        --------
        2D array - Free energy density at each grid point
        """
        return double_well_energy(self.c, self.A_phys, self.B_phys, self.C_phys)
    
    def compute_total_free_energy(self):
        """
        Compute total free energy of the system (integral over domain).
        
        F = ∫[f(c) + (κ/2)|∇c|²] dV
        
        Returns:
        --------
        float - Total free energy in Joules
        """
        # Bulk free energy contribution
        f_bulk = double_well_energy(self.c, self.A_phys, self.B_phys, self.C_phys)
        
        # Gradient energy contribution: (κ/2)|∇c|²
        grad_x = compute_gradient_x(self.c, self.dx_phys)
        grad_y = compute_gradient_y(self.c, self.dx_phys)
        grad_sq = grad_x**2 + grad_y**2
        f_gradient = 0.5 * self.kappa_phys * grad_sq
        
        # Integrate over domain (dx² is area per pixel in 2D)
        total_energy = np.sum(f_bulk + f_gradient) * (self.dx_phys**2)
        return float(total_energy)
    
    def get_statistics(self):
        """
        Compute comprehensive simulation statistics.
        
        Returns:
        --------
        dict - Statistics with physical units and formatted strings
        """
        # Geometric quantities
        domain_size_m = self.nx * self.dx_phys
        interface_width_m = self.scales.phys_to_interface_width(self.kappa_phys, self.W_phys)
        diffusion_length_m = np.sqrt(self.scales.D_b * self.time_phys)
        
        # Concentration statistics
        c_mean = float(np.mean(self.c))
        c_std = float(np.std(self.c))
        c_min = float(np.min(self.c))
        c_max = float(np.max(self.c))
        
        # Phase fractions
        n_pixels = self.nx * self.ny
        phase_high = float(np.sum(self.c > 0.5) / n_pixels)
        phase_low = float(np.sum(self.c < 0.5) / n_pixels)
        
        return {
            # Time
            'time_dim': self.time_dim,
            'time_phys': self.time_phys,
            'time_formatted': self.scales.format_time(self.time_phys),
            
            # Step counter
            'step': self.step,
            
            # Length scales
            'domain_size_m': domain_size_m,
            'domain_size_formatted': self.scales.format_length(domain_size_m),
            'interface_width_m': interface_width_m,
            'interface_width_formatted': self.scales.format_length(interface_width_m),
            'diffusion_length_m': diffusion_length_m,
            'diffusion_length_formatted': self.scales.format_length(diffusion_length_m),
            
            # Concentration statistics
            'mean_concentration': c_mean,
            'std_concentration': c_std,
            'min_concentration': c_min,
            'max_concentration': c_max,
            
            # Phase fractions
            'phase_fraction_high': phase_high,
            'phase_fraction_low': phase_low,
            'phase_fraction_interface': 1.0 - phase_high - phase_low,
            
            # Model parameters (physical)
            'W_phys': self.W_phys,
            'W_formatted': self.scales.format_energy_density(self.W_phys),
            'kappa_phys': self.kappa_phys,
            'M_phys': self.M_phys,
            'dt_phys': self.dt_phys,
            
            # Material parameters
            'D_b': self.scales.D_b,
            'Omega': self.scales.Omega,
            'L0': self.scales.L0,
        }
    
    def get_concentration_field(self):
        """Return copy of current concentration field."""
        return self.c.copy()
    
    def get_time_series(self, key):
        """
        Retrieve time series data from history.
        
        Parameters:
        -----------
        key : str - History key ('mean', 'std', 'phase_high', etc.)
        
        Returns:
        --------
        tuple : (time_array, value_array)
        """
        if key not in self.history:
            raise ValueError(f"Unknown history key: {key}")
        return np.array(self.history['time_phys']), np.array(self.history[key])


# =====================================================
# Streamlit App: Main Application Logic
# =====================================================

def main():
    """Main Streamlit application entry point."""
    
    # Page configuration
    st.set_page_config(
        page_title="LiFePO₄ Phase-Field Simulation",
        page_icon="🔋",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .metric-card {background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin: 5px 0;}
    .stButton>button {width: 100%;}
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🔋 LiₓFePO₄ Phase-Field Simulation")
    st.markdown("""
    **Physically realistic 2D simulation** of phase separation in lithium iron phosphate 
    cathode particles using the Cahn-Hilliard equation. All parameters in **real SI units**.
    
    *No electrostatics included - pure chemical spinodal decomposition and nucleation.*
    """)
    
    # Initialize simulation in session state (persists across reruns)
    if 'sim' not in st.session_state:
        st.session_state.sim = PhaseFieldSimulation(
            nx=256, ny=256, 
            dx_dim=1.0, dt_dim=0.01,
            L0_nm=10.0, D_b=1e-14, Omega_kJmol=12.0
        )
        st.session_state.sim.initialize_random(c0=0.5, noise_amplitude=0.05, seed=42)
        st.session_state.last_params_hash = None
    
    sim = st.session_state.sim
    
    # =====================================================
    # Sidebar: Control Panel
    # =====================================================
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # --- Run Controls ---
        st.subheader("⏱️ Time Stepping")
        
        col_run1, col_run2 = st.columns(2)
        with col_run1:
            steps_input = st.number_input(
                "Steps per update", 
                min_value=1, max_value=5000, value=50,
                help="Number of time steps to compute per button click"
            )
        with col_run2:
            if st.button("▶️ Run", type="primary", use_container_width=True):
                with st.spinner(f"Computing {steps_input} steps..."):
                    sim.run_steps(steps_input)
                st.rerun()
        
        col_stop1, col_stop2 = st.columns(2)
        with col_stop1:
            if st.button("⏸️ Pause", use_container_width=True):
                st.rerun()
        with col_stop2:
            if st.button("⏭️ Run 1 Step", use_container_width=True):
                sim.run_step()
                st.rerun()
        
        st.divider()
        
        # --- Initialization ---
        st.subheader("🎲 Initial Conditions")
        
        init_mode = st.radio(
            "Initialization mode",
            ["Spinodal (random)", "Nucleation (seed)"],
            index=0,
            help="Spinodal: random fluctuations; Nucleation: circular seed"
        )
        
        if init_mode == "Spinodal (random)":
            c0_spinodal = st.slider("Average concentration c₀", 0.1, 0.9, 0.5, 0.01)
            noise_spinodal = st.slider("Fluctuation amplitude", 0.001, 0.15, 0.05, 0.001)
            
            if st.button("🔄 Reset: Spinodal", use_container_width=True):
                sim.initialize_random(c0=c0_spinodal, noise_amplitude=noise_spinodal, seed=42)
                st.rerun()
        else:
            c0_nuc = st.slider("Background concentration", 0.1, 0.9, 0.3, 0.01)
            seed_val = st.slider("Seed concentration", 0.5, 1.0, 0.7, 0.01)
            seed_radius = st.slider("Seed radius (grid units)", 5, 50, 15, 1)
            
            if st.button("🌱 Reset: Nucleation", use_container_width=True):
                sim.initialize_seed(c0=c0_nuc, seed_value=seed_val, radius_grid=seed_radius, seed=42)
                st.rerun()
        
        st.divider()
        
        # --- Material Parameters ---
        st.subheader("🧪 Material Properties")
        st.caption("Fundamental parameters for LiFePO₄")
        
        # Reference length
        L0_nm = st.slider(
            "Reference length L₀ (nm)", 
            2.0, 100.0, 10.0, 1.0,
            help="Characteristic length scale; interface width ~2-10 nm typical"
        )
        
        # Diffusion coefficient (log scale)
        D_b_exp = st.slider(
            "log₁₀(D_b) [m²/s]", 
            -18, -10, -14, 1,
            help="Li⁺ diffusion coefficient along b-axis of LiFePO₄ crystal"
        )
        D_b_val = 10**D_b_exp
        
        # Regular solution parameter
        Omega_kJmol = st.slider(
            "Mixing enthalpy Ω (kJ/mol)", 
            5.0, 60.0, 12.0, 1.0,
            help="Regular solution parameter; controls phase separation driving force"
        )
        
        # Temperature (for future extensions)
        T_K = st.slider("Temperature T (K)", 200.0, 400.0, 298.15, 5.0)
        
        apply_material = st.button("Apply Material Parameters", use_container_width=True)
        if apply_material:
            sim.set_physical_parameters(
                L0_nm=L0_nm, 
                D_b=D_b_val, 
                Omega_kJmol=Omega_kJmol
            )
            # Update scales object temperature if needed in future
            sim.scales.T = T_K
            st.rerun()
        
        st.divider()
        
        # --- Model Parameters (Physical Units) ---
        st.subheader("⚙️ Model Parameters")
        st.caption("Cahn-Hilliard equation parameters in SI units")
        
        # Compute sensible slider ranges based on current scales
        E0 = sim.scales.E0
        L0 = sim.scales.L0
        M0 = sim.scales.M0
        t0 = sim.scales.t0
        
        # Double-well barrier W [J/m³]
        W_min = E0 * 0.01
        W_max = E0 * 100
        W_default = sim.W_phys
        W_phys = st.number_input(
            "W: Barrier height (J/m³)",
            min_value=float(W_min), max_value=float(W_max),
            value=float(W_default), format="%.2e",
            help="Controls energy barrier between Li-rich and Li-poor phases"
        )
        
        # Gradient coefficient κ [J/m]
        kappa_min = E0 * L0**2 * 0.01
        kappa_max = E0 * L0**2 * 100
        kappa_default = sim.kappa_phys
        kappa_phys = st.number_input(
            "κ: Gradient coefficient (J/m)",
            min_value=float(kappa_min), max_value=float(kappa_max),
            value=float(kappa_default), format="%.2e",
            help="Controls interface energy and width; ξ ≈ √(κ/W)"
        )
        
        # Mobility M [m⁵/(J·s)]
        M_min = M0 * 0.01
        M_max = M0 * 100
        M_default = sim.M_phys
        M_phys = st.number_input(
            "M: Mobility (m⁵/J·s)",
            min_value=float(M_min), max_value=float(M_max),
            value=float(M_default), format="%.2e",
            help="Controls kinetics; higher M = faster phase separation"
        )
        
        # Time step Δt [s] - with stability warning
        dt_max_safe = 0.05 * t0  # Conservative CFL-like limit for 4th-order PDE
        dt_default = min(sim.dt_phys, dt_max_safe)
        dt_phys = st.number_input(
            "Δt: Time step (s)",
            min_value=1e-15, max_value=float(dt_max_safe),
            value=float(dt_default), format="%.2e",
            help="Numerical time step; too large causes instability"
        )
        
        apply_model = st.button("Apply Model Parameters", use_container_width=True)
        if apply_model:
            sim.set_physical_parameters(
                W_Jm3=W_phys,
                kappa_Jm=kappa_phys,
                M_m5Js=M_phys,
                dt_s=dt_phys
            )
            st.rerun()
        
        # Stability info box
        interface_width_nm = sim.scales.phys_to_interface_width(kappa_phys, W_phys) * 1e9
        if interface_width_nm < 2.0:
            st.warning(f"⚠️ Interface width ({interface_width_nm:.2f} nm) < 2 nm: may be under-resolved")
        elif interface_width_nm > 50.0:
            st.info(f"ℹ️ Interface width ({interface_width_nm:.1f} nm) is quite diffuse")
        
        st.divider()
        
        # --- Live Statistics ---
        stats = sim.get_statistics()
        st.subheader("📊 Live Statistics")
        
        # Key metrics in cards
        st.markdown(f"""
        <div class="metric-card">
        <b>⏱️ Physical Time:</b> {stats['time_formatted']}
        </div>
        <div class="metric-card">
        <b>🔢 Simulation Step:</b> {stats['step']:,}
        </div>
        <div class="metric-card">
        <b>📐 Domain Size:</b> {stats['domain_size_formatted']}
        </div>
        <div class="metric-card">
        <b>🔲 Interface Width:</b> {stats['interface_width_formatted']}
        </div>
        <div class="metric-card">
        <b>📏 Diffusion Length:</b> {stats['diffusion_length_formatted']}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Concentration metrics
        st.markdown(f"**Concentration Statistics**")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.metric("⟨c⟩", f"{stats['mean_concentration']:.3f}")
            st.metric("min(c)", f"{stats['min_concentration']:.3f}")
        with col_c2:
            st.metric("σ(c)", f"{stats['std_concentration']:.3f}")
            st.metric("max(c)", f"{stats['max_concentration']:.3f}")
        
        # Phase fractions
        st.markdown(f"**Phase Distribution**")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.metric("Li-rich (x>0.5)", f"{stats['phase_fraction_high']*100:.1f}%")
        with col_p2:
            st.metric("Li-poor (x<0.5)", f"{stats['phase_fraction_low']*100:.1f}%")
        with col_p3:
            st.metric("Interface", f"{stats['phase_fraction_interface']*100:.1f}%")
    
    # =====================================================
    # Main Content Area: Visualizations
    # =====================================================
    
    # Row 1: Concentration field + Free energy
    col_viz1, col_viz2 = st.columns([2, 1])
    
    with col_viz1:
        st.subheader(f"Concentration Field")
        st.caption(f"Li fraction x in LiₓFePO₄ at t = {stats['time_formatted']}")
        
        # Create concentration plot with physical axis labels
        fig_c, ax_c = plt.subplots(figsize=(8, 7), dpi=100)
        
        # Physical extent for axis labels (convert grid to nm)
        extent_nm = [
            0, sim.nx * sim.scales.L0 * 1e9,  # x: 0 to L_x in nm
            0, sim.ny * sim.scales.L0 * 1e9   # y: 0 to L_y in nm
        ]
        
        im_c = ax_c.imshow(
            sim.c, 
            cmap='RdBu_r', 
            origin='lower', 
            vmin=0, vmax=1,
            extent=extent_nm,
            interpolation='bilinear'
        )
        ax_c.set_xlabel("x (nm)")
        ax_c.set_ylabel("y (nm)")
        ax_c.set_title(f"Li Concentration in LiₓFePO₄")
        ax_c.set_aspect('equal')
        
        cbar_c = plt.colorbar(im_c, ax=ax_c, label="Li fraction x")
        cbar_c.ax.tick_params(labelsize=9)
        
        # Add interface width indicator
        xi_nm = stats['interface_width_m'] * 1e9
        if 1 <= xi_nm <= 100:
            ax_c.text(0.02, 0.98, f"ξ ≈ {xi_nm:.2f} nm", 
                     transform=ax_c.transAxes, fontsize=9,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        st.pyplot(fig_c, use_container_width=True)
        plt.close(fig_c)
    
    with col_viz2:
        st.subheader("Free Energy Density")
        st.caption("f(c) = W·c²(1-c)² + gradient terms")
        
        # Compute free energy density
        energy = sim.compute_free_energy_density()
        
        fig_e, ax_e = plt.subplots(figsize=(5, 4.5), dpi=100)
        im_e = ax_e.imshow(
            energy,
            cmap='viridis',
            origin='lower',
            extent=extent_nm,
            interpolation='bilinear'
        )
        ax_e.set_xlabel("x (nm)")
        ax_e.set_ylabel("y (nm)")
        ax_e.set_title("Free Energy Density")
        ax_e.set_aspect('equal')
        
        cbar_e = plt.colorbar(im_e, ax=ax_e, label="J/m³")
        cbar_e.ax.tick_params(labelsize=8)
        cbar_e.formatter.set_powerlimits((0, 0))
        
        st.pyplot(fig_e, use_container_width=True)
        plt.close(fig_e)
        
        # Concentration histogram
        st.subheader("Concentration Distribution")
        fig_h, ax_h = plt.subplots(figsize=(5, 3), dpi=100)
        
        # Histogram with phase boundaries
        counts, bins, _ = ax_h.hist(
            sim.c.flatten(), 
            bins=50, 
            range=[0, 1],
            alpha=0.7, 
            color='steelblue', 
            edgecolor='black',
            density=False
        )
        
        # Mark phase boundaries and mean
        ax_h.axvline(0.5, color='gray', linestyle='--', linewidth=1, label='Phase boundary (c=0.5)')
        ax_h.axvline(stats['mean_concentration'], color='red', linestyle='-', 
                    linewidth=2, label=f"⟨c⟩ = {stats['mean_concentration']:.2f}")
        
        ax_h.set_xlim(0, 1)
        ax_h.set_xlabel("Li concentration x")
        ax_h.set_ylabel("Frequency")
        ax_h.legend(fontsize=8)
        ax_h.grid(True, alpha=0.3, linestyle=':')
        
        st.pyplot(fig_h, use_container_width=True)
        plt.close(fig_h)
    
    # Row 2: Time evolution plots
    st.divider()
    st.subheader("📈 Kinetics & Evolution")
    
    if len(sim.history['time_phys']) > 2:
        # Convert time to hours for battery-relevant plotting
        times_h = np.array(sim.history['time_phys']) / 3600
        
        fig_kin, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=100)
        
        # Plot 1: Mean concentration vs time
        axes[0].plot(times_h, sim.history['mean'], 'b-', linewidth=2, label='⟨c⟩')
        axes[0].axhline(0.5, color='gray', linestyle=':', linewidth=0.5)
        axes[0].set_xlabel("Time (hours)")
        axes[0].set_ylabel("Mean Li concentration")
        axes[0].set_title("Average Composition")
        axes[0].grid(True, alpha=0.3, linestyle=':')
        axes[0].legend(fontsize=9)
        
        # Plot 2: Order parameter (std dev) vs time
        axes[1].plot(times_h, sim.history['std'], 'r-', linewidth=2, label='σ(c)')
        axes[1].set_xlabel("Time (hours)")
        axes[1].set_ylabel("Standard deviation")
        axes[1].set_title("Phase Separation Progress")
        axes[1].grid(True, alpha=0.3, linestyle=':')
        axes[1].legend(fontsize=9)
        
        # Plot 3: Phase fractions vs time
        ph_high = np.array(sim.history['phase_high']) * 100
        ph_low = np.array(sim.history['phase_low']) * 100
        axes[2].plot(times_h, ph_high, 'g-', linewidth=2, label='Li-rich (x>0.5)')
        axes[2].plot(times_h, ph_low, 'orange', linewidth=2, label='Li-poor (x<0.5)')
        axes[2].set_xlabel("Time (hours)")
        axes[2].set_ylabel("Phase fraction (%)")
        axes[2].set_title("Phase Evolution")
        axes[2].legend(fontsize=9)
        axes[2].grid(True, alpha=0.3, linestyle=':')
        
        plt.tight_layout()
        st.pyplot(fig_kin, use_container_width=True)
        plt.close(fig_kin)
        
        # Optional: Free energy vs time
        with st.expander("🔋 Free Energy Evolution (click to expand)"):
            fig_fe, ax_fe = plt.subplots(figsize=(6, 4), dpi=100)
            ax_fe.plot(times_h, sim.history['energy'], 'purple-', linewidth=2)
            ax_fe.set_xlabel("Time (hours)")
            ax_fe.set_ylabel("Avg. free energy density (J/m³)")
            ax_fe.set_title("Free Energy Minimization")
            ax_fe.grid(True, alpha=0.3, linestyle=':')
            ax_fe.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.tight_layout()
            st.pyplot(fig_fe, use_container_width=True)
            plt.close(fig_fe)
    else:
        st.info("📊 Run simulation for at least 3 steps to display kinetics plots.")
    
    # =====================================================
    # Export Section
    # =====================================================
    st.divider()
    st.subheader("💾 Export Results")
    
    col_exp1, col_exp2, col_exp3, col_exp4 = st.columns(4)
    
    with col_exp1:
        if st.button("📸 Save Snapshot", use_container_width=True):
            fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
            im = ax.imshow(sim.c, cmap='RdBu_r', origin='lower', vmin=0, vmax=1,
                          extent=extent_nm, interpolation='bilinear')
            ax.set_xlabel("x (nm)")
            ax.set_ylabel("y (nm)")
            ax.set_title(f"LiₓFePO₄ Phase Field\n t = {stats['time_formatted']}")
            ax.set_aspect('equal')
            plt.colorbar(im, ax=ax, label="Li fraction x")
            
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            plt.close(fig)
            buf.seek(0)
            
            filename = f"LiFePO4_snapshot_t{sim.time_phys:.1e}s.png"
            st.download_button(
                label="⬇️ Download PNG",
                data=buf.getvalue(),
                file_name=filename,
                mime="image/png",
                use_container_width=True
            )
    
    with col_exp2:
        if st.button("📊 Save Statistics", use_container_width=True):
            # Build CSV content
            csv_lines = ["time_s,time_h,mean_c,std_c,phase_high,phase_low,energy_Jm3"]
            for i in range(len(sim.history['time_phys'])):
                line = f"{sim.history['time_phys'][i]:.6e},"
                line += f"{sim.history['time_phys'][i]/3600:.6e},"
                line += f"{sim.history['mean'][i]:.6f},"
                line += f"{sim.history['std'][i]:.6f},"
                line += f"{sim.history['phase_high'][i]:.6f},"
                line += f"{sim.history['phase_low'][i]:.6f},"
                line += f"{sim.history['energy'][i]:.6e}"
                csv_lines.append(line)
            
            csv_content = "\n".join(csv_lines)
            
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_content,
                file_name="phase_field_statistics.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col_exp3:
        if st.button("⚙️ Save Parameters", use_container_width=True):
            # Build parameter file
            params = f"""# LiFePO4 Phase-Field Simulation Parameters
# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

[MATERIAL_PROPERTIES]
L0_nm = {sim.scales.L0 * 1e9:.3f}
D_b_m2s = {sim.scales.D_b:.3e}
Omega_Jmol = {sim.scales.Omega:.3e}
Omega_kJmol = {sim.scales.Omega / 1e3:.3f}
T_K = {sim.scales.T:.2f}
V_m_m3mol = {sim.scales.V_m:.3e}

[MODEL_PARAMETERS_PHYSICAL]
W_Jm3 = {sim.W_phys:.3e}
kappa_Jm = {sim.kappa_phys:.3e}
M_m5Js = {sim.M_phys:.3e}
dt_s = {sim.dt_phys:.3e}
dx_m = {sim.dx_phys:.3e}

[DERIVED_QUANTITIES]
E0_Jm3 = {sim.scales.E0:.3e}
t0_s = {sim.scales.t0:.3e}
M0_m5Js = {sim.scales.M0:.3e}
interface_width_m = {sim.scales.phys_to_interface_width(sim.kappa_phys, sim.W_phys):.3e}
interface_width_nm = {sim.scales.phys_to_interface_width(sim.kappa_phys, sim.W_phys) * 1e9:.3f}

[SIMULATION_STATE]
nx = {sim.nx}
ny = {sim.ny}
step = {sim.step}
time_dim = {sim.time_dim:.6e}
time_phys_s = {sim.time_phys:.6e}
mean_concentration = {np.mean(sim.c):.6f}
            """
            
            st.download_button(
                label="⬇️ Download .txt",
                data=params,
                file_name="simulation_parameters.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    with col_exp4:
        if st.button("🗃️ Save Full State", use_container_width=True):
            # Save NumPy array of concentration field
            npz_buf = BytesIO()
            np.savez_compressed(
                npz_buf,
                concentration=sim.c,
                time_phys=sim.time_phys,
                step=sim.step,
                params={
                    'L0_nm': sim.scales.L0*1e9,
                    'D_b': sim.scales.D_b,
                    'Omega': sim.scales.Omega,
                    'W': sim.W_phys,
                    'kappa': sim.kappa_phys,
                    'M': sim.M_phys,
                    'dt': sim.dt_phys
                }
            )
            npz_buf.seek(0)
            
            st.download_button(
                label="⬇️ Download NPZ",
                data=npz_buf.getvalue(),
                file_name=f"phase_field_state_t{sim.time_phys:.1e}s.npz",
                mime="application/octet-stream",
                use_container_width=True
            )
    
    # =====================================================
    # Information & Documentation
    # =====================================================
    with st.expander("ℹ️ Physics Guide & Parameter Reference", expanded=False):
        st.markdown("""
        ## 🔋 LiₓFePO₄ Phase-Field Model
        
        This simulation implements the **Cahn-Hilliard equation** to model phase separation 
        in lithium iron phosphate (LiFePO₄) cathode materials during battery operation.
        
        ### Governing Equation
        
        ```
        ∂c/∂t = ∇·[M ∇(∂f/∂c - κ∇²c)]
        
        where:
        • c(x,t) : Li concentration (0 ≤ c ≤ 1, dimensionless)
        • M        : Mobility [m⁵/(J·s)]
        • f(c)     : Free energy density [J/m³]
        • κ        : Gradient energy coefficient [J/m]
        ```
        
        ### Free Energy Model
        
        Regular solution (double-well) free energy:
        ```
        f(c) = W · c² · (1-c)² = A·c² + B·c³ + C·c⁴
        with: A = W, B = -2W, C = W
        ```
        
        This creates two stable phases:
        - **Li-poor phase**: c ≈ 0 (FePO₄)
        - **Li-rich phase**: c ≈ 1 (LiFePO₄)
        
        ### Key Physical Parameters
        
        | Parameter | Symbol | Typical Range | Physical Meaning |
        |-----------|--------|--------------|------------------|
        | Reference length | L₀ | 2–100 nm | Interface/grid scale |
        | Diffusion coeff. | D_b | 10⁻¹⁸–10⁻¹⁰ m²/s | Li⁺ mobility in crystal |
        | Mixing enthalpy | Ω | 5–60 kJ/mol | Phase separation driving force |
        | Barrier height | W | 10⁴–10⁸ J/m³ | Energy barrier between phases |
        | Gradient coeff. | κ | 10⁻¹²–10⁻⁸ J/m | Interface energy penalty |
        | Mobility | M | 10⁻²²–10⁻¹⁴ m⁵/J·s | Kinetic coefficient |
        
        ### Interpreting Results
        
        #### Length Scales
        - **Interface width**: ξ ≈ √(κ/W) — typical 2–10 nm for LiFePO₄
        - **Domain size**: L = N·dx — simulation box size
        - **Diffusion length**: ℓ_D = √(D·t) — how far Li diffuses in time t
        
        #### Time Scales  
        - **Phase separation onset**: τ_spinodal ≈ (M·W)⁻¹
        - **Coarsening time**: τ_coarsen ≈ L⁴/(M·κ)
        - **Diffusion time**: τ_diff ≈ L²/D
        
        #### Phase Behavior
        - **Spinodal decomposition**: Random initial fluctuations → interconnected patterns
        - **Nucleation & growth**: Local seed → expanding Li-rich domain
        - **Coarsening**: Small domains merge → fewer, larger domains (Ostwald ripening)
        
        ### Stability Guidelines ⚠️
        
        For explicit time integration of the 4th-order Cahn-Hilliard equation:
        
        ```
        Δt ≲ 0.05 · (Δx)⁴ / (M · κ)   [stability criterion]
        ```
        
        Practical recommendations:
        1. Ensure interface spans ≥3 grid points: ξ/Δx ≥ 3
        2. Start with small Δt and increase gradually
        3. Monitor concentration bounds: c should stay in [0, 1]
        4. If simulation diverges: reduce Δt or increase κ
        
        ### Applications
        
        ✓ Study effect of **particle size** on phase separation kinetics  
        ✓ Optimize **charging rates** by tuning D_b and M  
        ✓ Compare **spinodal vs nucleation** mechanisms  
        ✓ Investigate **temperature dependence** via D_b(T) and Ω(T)  
        ✓ Educational tool for **phase-field methods** in materials science  
        
        ### References
        
        1. Cahn, J.W. & Hilliard, J.E. (1958). *J. Chem. Phys.* **28**, 258.
        2. Malik, R. et al. (2011). *J. Electrochem. Soc.* **158**, A530.
        3. Bai, P. et al. (2011). *J. Electrochem. Soc.* **158**, A1115.
        4. Cogswell, D.A. & Bazant, M.Z. (2013). *ACS Nano* **7**, 3036.
        """)
    
    # =====================================================
    # Auto-run Feature
    # =====================================================
    st.sidebar.divider()
    
    with st.sidebar.expander("🔄 Auto-run Settings"):
        auto_run = st.checkbox("Enable auto-run", value=False)
        auto_speed = st.slider("Speed (steps/second)", 1, 200, 20)
        auto_max_steps = st.number_input("Max steps (0 = unlimited)", 0, 100000, 0)
        
        if auto_run:
            stop_auto = st.button("⏹️ Stop Auto-run", type="secondary")
            
            if not stop_auto:
                # Run specified number of steps
                steps_this_frame = min(auto_speed, auto_max_steps - sim.step if auto_max_steps > 0 else auto_speed)
                
                if steps_this_frame > 0 and (auto_max_steps == 0 or sim.step < auto_max_steps):
                    with st.spinner(f"Auto-running {steps_this_frame} steps..."):
                        sim.run_steps(steps_this_frame)
                    st.rerun()
                elif auto_max_steps > 0 and sim.step >= auto_max_steps:
                    st.success(f"✓ Reached max steps: {sim.step}")
    
    # Footer
    st.markdown("---")
    st.caption(
        "LiₓFePO₄ Phase-Field Simulation | Cahn-Hilliard Equation | "
        "Physical Units: m, s, J/m³ | No electrostatics included"
    )


# =====================================================
# Application Entry Point
# =====================================================

if __name__ == "__main__":
    # Optional: Print startup info to console
    print("🔋 Starting LiFePO₄ Phase-Field Simulation...")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   NumPy: {np.__version__}")
    print(f"   Numba: available")
    print(f"   Streamlit: launching app")
    
    # Run the Streamlit app
    main()
