import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numba
from numba import njit, prange, float64, int32, types
import time
import pandas as pd
import base64
from io import BytesIO
import warnings
import hashlib
import sys

warnings.filterwarnings('ignore')

# ============================================
# STREAMLIT CONFIGURATION
# ============================================
st.set_page_config(
    page_title="LiFePO₄ Phase-Field Simulator",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E0E0E0;
        padding-bottom: 0.5rem;
    }
    .metric-box {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
    .warning-box {
        background-color: #FFF3CD;
        border: 1px solid #FFEAA7;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .debug-info {
        font-family: monospace;
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        font-size: 0.9em;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🔋 LiFePO₄ Phase-Field Simulator</h1>', unsafe_allow_html=True)
st.markdown("**Comprehensive simulation of spinodal decomposition with anisotropic elasticity & plasticity**")

# Initialize session state
for key in ['simulation_results', 'current_params_hash', 'debug_info', 'numba_cache_cleared']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'numba_cache_cleared' else False

# ============================================
# DEBUG UTILITIES
# ============================================
def print_debug_info(var_name, var_value):
    """Print debug information about variables"""
    info = f"{var_name}: "
    if hasattr(var_value, 'dtype'):
        info += f"type={type(var_value).__name__}, dtype={var_value.dtype}, shape={var_value.shape}"
    elif isinstance(var_value, (int, float, np.number)):
        info += f"type={type(var_value).__name__}, value={var_value}"
    else:
        info += f"type={type(var_value).__name__}"
    
    st.session_state.debug_info = st.session_state.debug_info or []
    st.session_state.debug_info.append(info)

def clear_numba_cache():
    """Clear Numba's JIT cache to force recompilation"""
    from numba.core.dispatcher import Dispatcher
    for mod in list(sys.modules.values()):
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if isinstance(attr, Dispatcher):
                attr.clear()
    st.session_state.numba_cache_cleared = True
    return True

# ============================================
# SIDEBAR - SIMULATION PARAMETERS
# ============================================
with st.sidebar:
    st.markdown('<div class="section-header">Simulation Parameters</div>', unsafe_allow_html=True)
    
    # Debug mode
    debug_mode = st.checkbox("Enable Debug Mode", value=False, 
                           help="Print detailed type information for debugging")
    
    # Simulation Type
    sim_type = st.selectbox(
        "Simulation Type",
        ["2D Cross-section (ac-plane)", "2D Cross-section (ab-plane)"],
        index=0,
        help="ac-plane: a-axis (x) vs c-axis (y), ab-plane: a-axis (x) vs b-axis (y, fast diffusion)"
    )
    
    # Grid size
    grid_size = st.slider("Grid Size (N×N)", 64, 512, 128, step=64,
                         help="Larger grids are more accurate but slower")
    nx, ny = grid_size, grid_size
    
    # Time parameters
    total_steps = st.number_input("Total Time Steps", 1000, 50000, 5000, step=500)
    save_every = st.number_input("Save Frame Every N Steps", 50, 2000, 100, step=50)
    
    # Thermodynamic Parameters
    st.markdown("#### Thermodynamics")
    Omega_RT = st.slider("Ω / RT (Miscibility Gap)", 8.0, 20.0, 13.0, step=0.5,
                        help="Ω > 2 for spinodal decomposition. Typical LiFePO₄: Ω ≈ 13RT")
    T = st.slider("Temperature (°C)", 25, 200, 25, step=25)
    T_K = T + 273.15
    
    # Gradient Energy
    kappa = st.slider("Gradient Coefficient κ (J/m)", 1e-12, 1e-9, 1e-11, format="%.1e",
                     help="Interface energy coefficient. Controls interface width")
    
    # Mobility & Anisotropy
    st.markdown("#### Kinetics")
    M0 = st.number_input("Baseline Mobility M₀ (m²/s)", 1e-20, 1e-14, 1e-16, format="%.1e",
                        help="Li diffusivity in slow direction")
    
    # Anisotropy
    if "ac-plane" in sim_type:
        anisotropy = st.slider("Mobility Anisotropy (a:c)", 1, 100, 10, 
                              help="M_a : M_c ratio (moderately fast along a-axis)")
        Mx, My = M0 * anisotropy, M0
    else:  # ab-plane
        anisotropy = st.slider("Mobility Anisotropy (a:b)", 100, 10000, 1000,
                              help="M_b : M_a ratio (very fast along b-axis)")
        Mx, My = M0, M0 * anisotropy
    
    # Elasticity Parameters
    st.markdown("#### Elasticity")
    use_elasticity = st.checkbox("Enable Anisotropic Elasticity", value=True)
    
    if use_elasticity:
        st.markdown("**Elastic Constants (Orthorhombic)**")
        col1, col2 = st.columns(2)
        with col1:
            C11 = st.number_input("C₁₁ (GPa)", 100.0, 300.0, 200.0, step=10.0)
            C12 = st.number_input("C₁₂ (GPa)", 50.0, 150.0, 70.0, step=5.0)
        with col2:
            C22 = st.number_input("C₂₂ (GPa)", 100.0, 300.0, 180.0, step=10.0)
            C44 = st.number_input("C₄₄ (GPa)", 30.0, 100.0, 60.0, step=5.0)
        
        # Eigenstrain
        st.markdown("**Eigenstrain (Total Volume Change: 6.8%)**")
        col1, col2 = st.columns(2)
        with col1:
            eps_xx = st.number_input("εₓₓ (a-axis, %)", 1.0, 5.0, 2.5, step=0.1) / 100
            eps_yy = st.number_input("εᵧᵧ (b-axis, %)", -0.5, 1.5, 0.15, step=0.1) / 100
        with col2:
            eps_zz = st.number_input("ε₂₂ (c-axis, %)", -5.0, -1.0, -2.5, step=0.1) / 100
        
        epsilon0 = np.array([eps_xx, eps_yy], dtype=np.float64)  # 2D simplification
    
    # Plasticity Parameters
    st.markdown("#### Plasticity")
    use_plasticity = st.checkbox("Enable J2 Plasticity", value=False)
    
    if use_plasticity:
        sigma_y0 = st.number_input("Yield Stress σ_y₀ (GPa)", 0.5, 5.0, 2.0, step=0.1)
        hardening = st.number_input("Hardening h (GPa)", 0.0, 200.0, 50.0, step=10.0)
    
    # Electrochemical BCs
    st.markdown("#### Electrochemistry")
    bc_type = st.selectbox(
        "Boundary Condition",
        ["Potentiostatic (Butler-Volmer)", "Galvanostatic (Constant Flux)", "Open System (No Flux)"],
        index=0
    )
    
    if bc_type == "Potentiostatic (Butler-Volmer)":
        eta = st.slider("Overpotential η (mV)", -100, 100, 30, step=10)
        k0 = st.number_input("Exchange Current k₀ (mol/m²s)", 0.01, 10.0, 1.0, step=0.1)
        alpha = st.slider("Charge Transfer α", 0.3, 0.7, 0.5, step=0.05)
    elif bc_type == "Galvanostatic (Constant Flux)":
        current = st.number_input("Current Density (A/m²)", 0.1, 100.0, 10.0, step=1.0)
    
    # Initial Conditions
    st.markdown("#### Initial Conditions")
    init_type = st.selectbox(
        "Initial Composition",
        ["Homogeneous (x=0.5)", "Central Nucleus", "Random Fluctuations", 
         "Graded Profile", "Two Phase Mixture", "Spinodal Fluctuations"],
        index=0
    )
    
    noise_level = st.slider("Initial Noise (%)", 0.0, 10.0, 2.0, step=0.5) / 100
    
    # Numerical stability
    st.markdown("#### Numerical Settings")
    safety_factor = st.slider("Stability Safety Factor", 0.01, 0.5, 0.1, step=0.01,
                            help="Smaller = more stable but slower")
    enable_numba = st.checkbox("Enable Numba JIT", value=True,
                             help="Disable for debugging Numba issues")
    
    # Run Buttons
    col1, col2 = st.columns(2)
    with col1:
        run_simulation = st.button("🚀 Run Simulation", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear Cache", use_container_width=True):
            st.session_state.simulation_results = None
            st.session_state.current_params_hash = None
            st.session_state.debug_info = None
            if clear_numba_cache():
                st.success("Numba cache cleared!")
            st.rerun()

# ============================================
# NUMBA-OPTIMIZED KERNELS WITH TYPE SAFETY
# ============================================

# Define Numba signatures for type safety
chemical_potential_signature = float64[:,:](float64[:,:], float64, float64, float64)
elastic_potential_signature = float64[:,:](float64[:,:], float64[:], float64, float64, float64, float64, float64)
update_concentration_signature = float64[:,:](float64[:,:], float64[:,:], float64, float64, float64, float64, float64, float64[:,:])

@njit(chemical_potential_signature, parallel=True, fastmath=True, cache=True)
def chemical_potential_numba(c, Omega_RT, kappa, dx):
    """
    Calculate chemical potential with explicit type annotations.
    
    Args:
        c: 2D array of float64, concentration field
        Omega_RT: float64, regular solution parameter divided by RT
        kappa: float64, gradient energy coefficient
        dx: float64, grid spacing
    
    Returns:
        2D array of float64, chemical potential field
    """
    nx, ny = c.shape
    mu = np.zeros((nx, ny), dtype=np.float64)
    
    inv_dx2 = 1.0 / (dx * dx)
    
    for i in prange(nx):
        for j in range(ny):
            # Periodic boundary conditions
            im = (i - 1) % nx
            ip = (i + 1) % nx
            jm = (j - 1) % ny
            jp = (j + 1) % ny
            
            # Current concentration value
            ci = c[i, j]
            
            # Clamp to avoid log(0) or log(1)
            ci_clipped = ci
            if ci_clipped < 1e-12:
                ci_clipped = 1e-12
            elif ci_clipped > 1.0 - 1e-12:
                ci_clipped = 1.0 - 1e-12
            
            # Homogeneous chemical potential
            mu_hom = Omega_RT * (1.0 - 2.0 * ci_clipped) + np.log(ci_clipped) - np.log(1.0 - ci_clipped)
            
            # Laplacian (central difference)
            lap = (c[im, j] + c[ip, j] + c[i, jm] + c[i, jp] - 4.0 * ci) * inv_dx2
            
            # Total chemical potential
            mu[i, j] = mu_hom - kappa * lap
    
    return mu

@njit(elastic_potential_signature, parallel=True, fastmath=True, cache=True)
def elastic_potential_numba(c, epsilon0, C11, C12, C22, C44, V_m):
    """
    Calculate elastic contribution to chemical potential.
    
    Args:
        c: 2D array of float64, concentration field
        epsilon0: 1D array of float64, eigenstrain components
        C11, C12, C22, C44: float64, elastic constants
        V_m: float64, molar volume
    
    Returns:
        2D array of float64, elastic potential field
    """
    nx, ny = c.shape
    mu_el = np.zeros((nx, ny), dtype=np.float64)
    
    # Mean composition
    c_mean = np.mean(c)
    
    for i in prange(nx):
        for j in range(ny):
            # Composition deviation from mean
            dc = c[i, j] - c_mean
            
            # Simplified elastic contribution
            # For 2D plane stress with anisotropic eigenstrain
            eps_mag = np.sqrt(epsilon0[0]**2 + epsilon0[1]**2)
            
            # Average stiffness
            C_avg = 0.5 * (C11 + C22)
            
            # Stress magnitude (simplified)
            stress_mag = C_avg * eps_mag * dc
            
            # Elastic contribution to chemical potential
            mu_el[i, j] = -V_m * stress_mag
    
    return mu_el

@njit(update_concentration_signature, parallel=True, fastmath=True, cache=True)
def update_concentration_numba(c, mu, Mx, My, dt, dx, flux_rate, flux_mask):
    """
    Update concentration field using Cahn-Hilliard equation.
    
    Args:
        c: 2D array of float64, current concentration
        mu: 2D array of float64, chemical potential
        Mx, My: float64, mobilities in x and y directions
        dt: float64, time step
        dx: float64, grid spacing
        flux_rate: float64, boundary flux rate
        flux_mask: 2D array of float64, mask for flux application
    
    Returns:
        2D array of float64, updated concentration
    """
    nx, ny = c.shape
    c_new = np.zeros((nx, ny), dtype=np.float64)
    
    inv_dx2 = 1.0 / (dx * dx)
    coeff_x = Mx * dt * inv_dx2
    coeff_y = My * dt * inv_dx2
    
    for i in prange(nx):
        for j in range(ny):
            # Periodic boundary conditions
            im = (i - 1) % nx
            ip = (i + 1) % nx
            jm = (j - 1) % ny
            jp = (j + 1) % ny
            
            # Divergence of M*grad(mu)
            div_x = coeff_x * (mu[im, j] - 2.0 * mu[i, j] + mu[ip, j])
            div_y = coeff_y * (mu[i, jm] - 2.0 * mu[i, j] + mu[i, jp])
            
            # Update concentration
            c_new[i, j] = c[i, j] + div_x + div_y
    
    # Apply boundary flux if any
    if flux_rate != 0.0:
        c_new += flux_mask * flux_rate * dt
    
    # Clamp to physical range
    for i in prange(nx):
        for j in range(ny):
            val = c_new[i, j]
            if val < 0.0:
                c_new[i, j] = 0.0
            elif val > 1.0:
                c_new[i, j] = 1.0
    
    return c_new

# Python fallback functions (for debugging)
def chemical_potential_python(c, Omega_RT, kappa, dx=1.0):
    """Python implementation for debugging"""
    nx, ny = c.shape
    mu = np.zeros_like(c)
    
    inv_dx2 = 1.0 / (dx * dx)
    
    for i in range(nx):
        for j in range(ny):
            im = (i - 1) % nx
            ip = (i + 1) % nx
            jm = (j - 1) % ny
            jp = (j + 1) % ny
            
            ci = c[i, j]
            ci_clipped = max(min(ci, 1.0 - 1e-12), 1e-12)
            
            mu_hom = Omega_RT * (1.0 - 2.0 * ci_clipped) + np.log(ci_clipped) - np.log(1.0 - ci_clipped)
            lap = (c[im, j] + c[ip, j] + c[i, jm] + c[i, jp] - 4.0 * ci) * inv_dx2
            mu[i, j] = mu_hom - kappa * lap
    
    return mu

def update_concentration_python(c, mu, Mx, My, dt, dx=1.0, flux_rate=0.0, flux_mask=None):
    """Python implementation for debugging"""
    nx, ny = c.shape
    c_new = np.zeros_like(c)
    
    inv_dx2 = 1.0 / (dx * dx)
    coeff_x = Mx * dt * inv_dx2
    coeff_y = My * dt * inv_dx2
    
    for i in range(nx):
        for j in range(ny):
            im = (i - 1) % nx
            ip = (i + 1) % nx
            jm = (j - 1) % ny
            jp = (j + 1) % ny
            
            div_x = coeff_x * (mu[im, j] - 2.0 * mu[i, j] + mu[ip, j])
            div_y = coeff_y * (mu[i, jm] - 2.0 * mu[i, j] + mu[i, jp])
            c_new[i, j] = c[i, j] + div_x + div_y
    
    if flux_rate != 0.0 and flux_mask is not None:
        c_new += flux_mask * flux_rate * dt
    
    return np.clip(c_new, 0.0, 1.0)

# Select which implementation to use based on debug mode
if enable_numba:
    chemical_potential_func = chemical_potential_numba
    elastic_potential_func = elastic_potential_numba if use_elasticity else None
    update_concentration_func = update_concentration_numba
else:
    chemical_potential_func = chemical_potential_python
    elastic_potential_func = None
    update_concentration_func = update_concentration_python

# ============================================
# UTILITY FUNCTIONS
# ============================================
def calculate_stable_dt(Mx, My, kappa, dx=1.0, safety_factor=0.1):
    """Calculate stable time step using CFL condition"""
    M_max = max(Mx, My)
    if M_max == 0 or kappa == 0:
        return 0.01  # Default safe value
    
    dt_max = safety_factor * dx * dx / (M_max * kappa)
    return min(dt_max, 0.1)  # Upper bound

def create_initial_condition(init_type, nx, ny, noise_level=0.02):
    """Create initial concentration field with type safety"""
    c = np.full((nx, ny), 0.5, dtype=np.float64)
    
    if init_type == "Homogeneous (x=0.5)":
        pass  # Already set to 0.5
    
    elif init_type == "Central Nucleus":
        center_x, center_y = nx // 2, ny // 2
        radius = min(nx, ny) // 6
        
        Y, X = np.ogrid[:nx, :ny]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        mask = dist_from_center <= radius
        c[mask] = 0.9
        c[~mask] = 0.1
        
        # Smooth transition
        from scipy.ndimage import gaussian_filter
        c = gaussian_filter(c, sigma=1.0, mode='wrap').astype(np.float64)
    
    elif init_type == "Random Fluctuations":
        fluctuations = np.random.normal(0, noise_level, (nx, ny)).astype(np.float64)
        c += fluctuations
    
    elif init_type == "Graded Profile":
        for i in range(nx):
            c[i, :] = 0.1 + 0.8 * (i / nx)
        c = c.astype(np.float64)
    
    elif init_type == "Two Phase Mixture":
        # Checkerboard pattern
        for i in range(nx):
            for j in range(ny):
                if (i // 10 + j // 10) % 2 == 0:
                    c[i, j] = 0.9
                else:
                    c[i, j] = 0.1
        
        from scipy.ndimage import gaussian_filter
        c = gaussian_filter(c, sigma=1.0, mode='wrap').astype(np.float64)
    
    elif init_type == "Spinodal Fluctuations":
        # Create random fluctuations with specific wavelength
        x = np.linspace(0, 2*np.pi, nx)
        y = np.linspace(0, 2*np.pi, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Multiple sinusoidal modes
        c = 0.5 + 0.1 * (np.sin(3*X) * np.sin(3*Y) + 
                        0.5 * np.sin(5*X) * np.cos(5*Y) +
                        0.3 * np.cos(7*X) * np.sin(7*Y))
        
        # Add random noise
        c += np.random.normal(0, noise_level, (nx, ny))
        c = c.astype(np.float64)
    
    # Add final noise to all configurations
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * 0.5, (nx, ny)).astype(np.float64)
        c += noise
    
    # Ensure proper dtype and bounds
    c = np.clip(c, 0.01, 0.99).astype(np.float64)
    return c

def create_flux_mask(nx, ny, flux_rows=5, location='bottom'):
    """Create mask for flux boundary condition"""
    mask = np.zeros((nx, ny), dtype=np.float64)
    
    if location == 'bottom':
        mask[:flux_rows, :] = 1.0
    elif location == 'top':
        mask[-flux_rows:, :] = 1.0
    elif location == 'left':
        mask[:, :flux_rows] = 1.0
    elif location == 'right':
        mask[:, -flux_rows:] = 1.0
    
    return mask

# ============================================
# SIMULATION ENGINE WITH TYPE SAFETY
# ============================================
class LFPPhaseFieldSimulator:
    def __init__(self, params):
        self.params = params
        self.frames = []
        self.stats = []
        self.debug_log = []
        
        # Initialize concentration field with proper dtype
        self.c = create_initial_condition(
            params['init_type'],
            params['nx'],
            params['ny'],
            params['noise_level']
        )
        
        # Ensure proper dtype
        self.c = self.c.astype(np.float64)
        
        # Create flux mask if needed
        self.flux_mask = None
        if params['bc_type'] in ['Potentiostatic (Butler-Volmer)', 'Galvanostatic (Constant Flux)']:
            self.flux_mask = create_flux_mask(params['nx'], params['ny'], flux_rows=5, location='bottom')
            self.flux_mask = self.flux_mask.astype(np.float64)
        
        # Debug initialization
        if debug_mode:
            self.debug_log.append(f"Initialized: c shape={self.c.shape}, dtype={self.c.dtype}")
            self.debug_log.append(f"c min={np.min(self.c):.4f}, max={np.max(self.c):.4f}, mean={np.mean(self.c):.4f}")
    
    def calculate_flux_rate(self, step, c_surface=None):
        """Calculate flux rate based on boundary condition"""
        params = self.params
        
        if params['bc_type'] == "Open System (No Flux)":
            return 0.0
        
        elif params['bc_type'] == "Galvanostatic (Constant Flux)":
            # Constant current
            flux_rate = params.get('current', 0.01) * 1e-4  # Scale for stability
            return float(flux_rate)
        
        elif params['bc_type'] == "Potentiostatic (Butler-Volmer)":
            # Butler-Volmer kinetics
            if c_surface is None:
                c_surface = np.mean(self.c[:5, :]) if self.c.shape[0] > 5 else np.mean(self.c)
            
            eta = params.get('eta', 0.03)  # V
            k0 = params.get('k0', 1.0)
            alpha = params.get('alpha', 0.5)
            
            # F/RT at given temperature
            F = 96485.0  # C/mol
            R = 8.314    # J/(mol·K)
            F_over_RT = F / (R * params['T_K'])
            
            # Avoid overflow in exp
            exponent1 = alpha * F_over_RT * eta
            exponent2 = -(1 - alpha) * F_over_RT * eta
            
            exponent1 = np.clip(exponent1, -50.0, 50.0)
            exponent2 = np.clip(exponent2, -50.0, 50.0)
            
            # Current density
            i = k0 * (np.exp(exponent1) - np.exp(exponent2))
            
            # Convert to flux and scale
            flux_rate = i * 1e-5  # Scale for numerical stability
            
            return float(flux_rate)
        
        return 0.0
    
    def run(self):
        """Main simulation loop with type safety"""
        params = self.params
        
        # Calculate stable time step
        dt = calculate_stable_dt(
            params['Mx'], params['My'], 
            params['kappa'], 
            dx=1.0, safety_factor=safety_factor
        )
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_container = st.container()
        
        # Debug information
        if debug_mode:
            with st.expander("🔍 Debug Information", expanded=True):
                st.markdown("### Initial State Debug")
                st.write(f"c dtype: {self.c.dtype}")
                st.write(f"c shape: {self.c.shape}")
                st.write(f"c range: [{np.min(self.c):.6f}, {np.max(self.c):.6f}]")
                st.write(f"Omega_RT: {params['Omega_RT']}, type: {type(params['Omega_RT'])}")
                st.write(f"kappa: {params['kappa']}, type: {type(params['kappa'])}")
                st.write(f"dt: {dt}")
                
                # Test chemical potential function
                try:
                    test_mu = chemical_potential_func(self.c, params['Omega_RT'], params['kappa'], 1.0)
                    st.success(f"✅ Chemical potential function test passed")
                    st.write(f"mu dtype: {test_mu.dtype}, shape: {test_mu.shape}")
                except Exception as e:
                    st.error(f"❌ Chemical potential function test failed: {e}")
        
        start_time = time.time()
        
        for step in range(params['total_steps']):
            try:
                # Calculate chemical potential
                mu_chem = chemical_potential_func(
                    self.c, 
                    float(params['Omega_RT']), 
                    float(params['kappa']), 
                    1.0  # dx
                )
                
                # Add elastic contribution if enabled
                mu = mu_chem.copy()
                if (params.get('use_elasticity', False) and 
                    elastic_potential_func is not None and 
                    'epsilon0' in params):
                    
                    mu_el = elastic_potential_func(
                        self.c,
                        params['epsilon0'].astype(np.float64),
                        float(params.get('C11', 200e9)),
                        float(params.get('C12', 70e9)),
                        float(params.get('C22', 180e9)),
                        float(params.get('C44', 60e9)),
                        float(3.02e-5)  # V_m
                    )
                    mu += mu_el
                
                # Calculate flux rate
                flux_rate = self.calculate_flux_rate(step)
                
                # Update concentration
                self.c = update_concentration_func(
                    self.c, mu, 
                    float(params['Mx']), float(params['My']), 
                    dt, 1.0, float(flux_rate),
                    self.flux_mask if self.flux_mask is not None else np.zeros_like(self.c)
                )
                
                # Ensure proper dtype
                self.c = self.c.astype(np.float64)
                
                # Save frame periodically
                if step % params['save_every'] == 0 or step == params['total_steps'] - 1:
                    self.frames.append(self.c.copy())
                    
                    # Calculate statistics
                    stats = {
                        'step': int(step),
                        'time': float(step * dt),
                        'mean_c': float(np.mean(self.c)),
                        'std_c': float(np.std(self.c)),
                        'min_c': float(np.min(self.c)),
                        'max_c': float(np.max(self.c)),
                        'gradient': float(np.mean(np.abs(np.gradient(self.c)))),
                    }
                    self.stats.append(stats)
                
                # Update progress
                if step % 100 == 0:
                    progress = step / params['total_steps']
                    progress_bar.progress(progress)
                    
                    if step % 1000 == 0:
                        with status_container:
                            status_text.text(f"Step {step:,} | Time = {step*dt:.3f} | Mean c = {np.mean(self.c):.4f}")
            
            except Exception as e:
                # Capture and display error
                error_msg = f"Error at step {step}: {str(e)}"
                self.debug_log.append(error_msg)
                
                if debug_mode:
                    st.error(f"❌ Simulation stopped at step {step}: {e}")
                    with st.expander("Error Details", expanded=True):
                        st.code(f"""
                        Step: {step}
                        c shape: {self.c.shape if 'c' in locals() else 'N/A'}
                        c dtype: {self.c.dtype if 'c' in locals() else 'N/A'}
                        Error: {e}
                        """)
                
                # Try to save partial results
                if len(self.frames) == 0:
                    self.frames.append(self.c.copy())
                
                break
        
        elapsed = time.time() - start_time
        
        # Final progress update
        progress_bar.progress(1.0)
        
        return self.frames, self.stats, elapsed, self.debug_log
    
    def analyze_morphology(self):
        """Analyze domain morphology from final frame"""
        try:
            c = self.frames[-1] if self.frames else self.c
            
            # Simple thresholding
            c_binary = c > 0.5
            
            # Count domains using connected components
            from scipy import ndimage
            labeled, num_features = ndimage.label(c_binary)
            
            # Calculate domain sizes
            domain_sizes = []
            for i in range(1, num_features + 1):
                size = np.sum(labeled == i)
                if size > 4:  # Ignore very small domains
                    domain_sizes.append(size)
            
            return {
                'num_domains': int(num_features),
                'mean_size': float(np.mean(domain_sizes)) if domain_sizes else 0.0,
                'size_std': float(np.std(domain_sizes)) if domain_sizes else 0.0,
                'phase_fraction': float(np.mean(c_binary)),
                'domain_size_distribution': domain_sizes[:20]  # First 20 domains
            }
        except Exception as e:
            # Fallback if scipy not available or error
            c_binary = (self.c > 0.5) if hasattr(self, 'c') else np.array([False])
            return {
                'num_domains': 0,
                'mean_size': 0.0,
                'size_std': 0.0,
                'phase_fraction': float(np.mean(c_binary)) if c_binary.size > 0 else 0.0,
                'domain_size_distribution': []
            }

# ============================================
# VISUALIZATION FUNCTIONS
# ============================================
def create_animation(frames, steps, sim_type):
    """Create interactive animation of concentration evolution"""
    if not frames:
        return go.Figure()
    
    # Convert frames to list if needed
    if isinstance(frames, np.ndarray):
        frames = [frames]
    
    # Create frames for animation
    fig_frames = []
    for i, frame in enumerate(frames):
        fig_frames.append(
            go.Frame(
                data=[go.Heatmap(
                    z=frame,
                    zmin=0, zmax=1,
                    colorscale='RdBu_r',
                    showscale=True,
                    colorbar=dict(title="Li Fraction x")
                )],
                name=f"Step {steps[i] if i < len(steps) else i}"
            )
        )
    
    # Create figure
    fig = go.Figure(
        data=[go.Heatmap(
            z=frames[0],
            zmin=0, zmax=1,
            colorscale='RdBu_r',
            showscale=True,
            colorbar=dict(title="Li Fraction x")
        )],
        frames=fig_frames
    )
    
    # Set axis labels based on simulation type
    if "ac-plane" in sim_type:
        xaxis_title = "a-axis"
        yaxis_title = "c-axis"
    else:
        xaxis_title = "a-axis"
        yaxis_title = "b-axis (fast diffusion)"
    
    # Update layout
    fig.update_layout(
        title="LiₓFePO₄ Phase Evolution",
        height=600,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {
                    "label": "▶️ Play",
                    "method": "animate",
                    "args": [None, {
                        "frame": {"duration": 100, "redraw": True},
                        "fromcurrent": True,
                        "mode": "immediate"
                    }]
                },
                {
                    "label": "⏸️ Pause",
                    "method": "animate",
                    "args": [[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate"
                    }]
                },
                {
                    "label": "⏭️ Next",
                    "method": "animate",
                    "args": [[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "next"
                    }]
                }
            ]
        }]
    )
    
    return fig

def plot_statistics(stats):
    """Plot evolution of statistical measures"""
    if not stats:
        return go.Figure()
    
    steps = [s['step'] for s in stats]
    time = [s['time'] for s in stats]
    mean_c = [s['mean_c'] for s in stats]
    std_c = [s['std_c'] for s in stats]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Mean Concentration",
            "Phase Separation (Std Dev)",
            "Concentration Range",
            "Gradient Evolution"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # Mean concentration
    fig.add_trace(
        go.Scatter(x=time, y=mean_c, mode='lines', name='Mean c',
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    # Standard deviation (measure of phase separation)
    fig.add_trace(
        go.Scatter(x=time, y=std_c, mode='lines', name='Std Dev',
                  line=dict(color='red', width=2)),
        row=1, col=2
    )
    
    # Concentration range
    min_c = [s['min_c'] for s in stats]
    max_c = [s['max_c'] for s in stats]
    fig.add_trace(
        go.Scatter(x=time, y=min_c, mode='lines', name='Min c',
                  line=dict(color='green', width=1, dash='dash')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=time, y=max_c, mode='lines', name='Max c',
                  line=dict(color='orange', width=1, dash='dash')),
        row=2, col=1
    )
    
    # Gradient evolution
    if 'gradient' in stats[0]:
        gradient = [s['gradient'] for s in stats]
        fig.add_trace(
            go.Scatter(x=time, y=gradient, mode='lines', name='Gradient',
                      line=dict(color='purple', width=2)),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Time (arb. units)", row=2, col=1)
    fig.update_xaxes(title_text="Time (arb. units)", row=2, col=2)
    fig.update_yaxes(title_text="Concentration", row=1, col=1)
    fig.update_yaxes(title_text="Std Dev", row=1, col=2)
    fig.update_yaxes(title_text="Concentration", row=2, col=1)
    fig.update_yaxes(title_text="Gradient", row=2, col=2)
    
    return fig

# ============================================
# MAIN SIMULATION EXECUTION
# ============================================
if run_simulation:
    # Clear previous debug info
    st.session_state.debug_info = []
    
    # Create parameters dictionary with explicit type conversion
    params = {
        'nx': int(nx),
        'ny': int(ny),
        'total_steps': int(total_steps),
        'save_every': int(save_every),
        'Omega_RT': float(Omega_RT),
        'T_K': float(T_K),
        'kappa': float(kappa),
        'Mx': float(Mx),
        'My': float(My),
        'use_elasticity': bool(use_elasticity),
        'init_type': str(init_type),
        'noise_level': float(noise_level),
        'bc_type': str(bc_type),
        'sim_type': str(sim_type),
        'debug_mode': bool(debug_mode)
    }
    
    # Add elasticity parameters if enabled
    if use_elasticity:
        epsilon0_array = np.array([float(eps_xx), float(eps_yy)], dtype=np.float64)
        params.update({
            'C11': float(C11 * 1e9),
            'C22': float(C22 * 1e9),
            'C12': float(C12 * 1e9),
            'C44': float(C44 * 1e9),
            'epsilon0': epsilon0_array
        })
    
    # Add BC-specific parameters
    if bc_type == "Potentiostatic (Butler-Volmer)":
        params.update({
            'eta': float(eta * 0.001),  # Convert mV to V
            'k0': float(k0),
            'alpha': float(alpha)
        })
    elif bc_type == "Galvanostatic (Constant Flux)":
        params.update({
            'current': float(current)
        })
    
    # Add plasticity parameters if enabled
    if use_plasticity:
        params.update({
            'sigma_y0': float(sigma_y0 * 1e9),
            'hardening': float(hardening * 1e9)
        })
    
    # Create hash for caching
    params_str = str(sorted([(k, str(v) if isinstance(v, np.ndarray) else v) 
                            for k, v in params.items()]))
    params_hash = hashlib.md5(params_str.encode()).hexdigest()
    
    # Check if we can use cached results
    use_cache = False
    if (st.session_state.simulation_results is not None and 
        st.session_state.current_params_hash == params_hash and
        not debug_mode):  # Don't use cache in debug mode
        use_cache = True
        st.info("📊 Using cached results from previous run with same parameters.")
    
    # Run simulation or use cache
    if not use_cache:
        # Show simulation info
        st.info(f"""
        **Simulation Configuration:**
        - Grid: {nx} × {ny} ({sim_type})
        - Time steps: {total_steps:,}
        - Ω/RT: {Omega_RT:.1f} ({'Spinodal regime' if Omega_RT > 2.0 else 'No spinodal'})
        - Anisotropy: {anisotropy}:1
        - Elasticity: {'Enabled' if use_elasticity else 'Disabled'}
        - Boundary: {bc_type}
        - Numba JIT: {'Enabled' if enable_numba else 'Disabled'}
        """)
        
        # Create simulator
        simulator = LFPPhaseFieldSimulator(params)
        
        # Run simulation
        with st.spinner("Running phase-field simulation..."):
            frames, stats, elapsed, debug_log = simulator.run()
        
        # Store results
        st.session_state.simulation_results = {
            'frames': frames,
            'stats': stats,
            'morphology': simulator.analyze_morphology(),
            'final_state': frames[-1] if frames else simulator.c,
            'params': params,
            'elapsed': elapsed,
            'debug_log': debug_log
        }
        st.session_state.current_params_hash = params_hash
        
        st.success(f"✅ Simulation completed in {elapsed:.1f} seconds!")
    
    else:
        # Use cached results
        frames = st.session_state.simulation_results['frames']
        stats = st.session_state.simulation_results['stats']
        elapsed = st.session_state.simulation_results['elapsed']
        debug_log = st.session_state.simulation_results.get('debug_log', [])
    
    # ============================================
    # DISPLAY RESULTS
    # ============================================
    
    # Main visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="section-header">Phase Evolution Animation</div>', unsafe_allow_html=True)
        
        # Create animation
        if frames:
            steps = [s['step'] for s in stats]
            fig_anim = create_animation(frames, steps, sim_type)
            st.plotly_chart(fig_anim, use_container_width=True)
        else:
            st.warning("No frames were generated during simulation.")
    
    with col2:
        st.markdown('<div class="section-header">Simulation Metrics</div>', unsafe_allow_html=True)
        
        if stats:
            final_stats = stats[-1]
            morphology = st.session_state.simulation_results['morphology']
            
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                st.metric("Total Time", f"{final_stats['time']:.3f}")
                st.metric("Final Mean x", f"{final_stats['mean_c']:.4f}")
                st.metric("Phase Separation", f"{final_stats['std_c']:.4f}")
            
            with metrics_col2:
                st.metric("Domains", morphology['num_domains'])
                st.metric("Li-rich Phase", f"{morphology['phase_fraction']*100:.1f}%")
                st.metric("Run Time", f"{elapsed:.1f}s")
            
            # Phase separation indicator
            st.markdown("#### Phase Separation Status")
            if final_stats['std_c'] > 0.2:
                st.success("✅ **Strong Spinodal Decomposition**")
                st.progress(min(final_stats['std_c'], 1.0))
            elif final_stats['std_c'] > 0.05:
                st.warning("⚠️ **Moderate Phase Separation**")
                st.progress(final_stats['std_c'] / 0.2)
            else:
                st.info("🔵 **Homogeneous / Solid Solution**")
                st.progress(final_stats['std_c'] / 0.2)
        else:
            st.warning("No statistics available.")
    
    # Statistics plot
    if stats:
        st.markdown('<div class="section-header">Evolution Statistics</div>', unsafe_allow_html=True)
        fig_stats = plot_statistics(stats)
        st.plotly_chart(fig_stats, use_container_width=True)
    
    # ============================================
    # DEBUG INFORMATION
    # ============================================
    if debug_mode and debug_log:
        with st.expander("📋 Debug Log", expanded=False):
            for log_entry in debug_log:
                st.text(log_entry)
    
    # ============================================
    # EXPORT AND ANALYSIS
    # ============================================
    st.markdown('<div class="section-header">Export & Analysis</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if stats:
            df_stats = pd.DataFrame(stats)
            csv = df_stats.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="lfp_stats.csv">📊 Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    with col2:
        if frames:
            final_state = frames[-1]
            np_bytes = BytesIO()
            np.save(np_bytes, final_state, allow_pickle=False)
            b64 = base64.b64encode(np_bytes.getvalue()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="final_state.npy">💾 Download Final State</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    with col3:
        params_df = pd.DataFrame([{k: str(v) for k, v in params.items()}])
        csv = params_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="parameters.csv">⚙️ Download Parameters</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    # ============================================
    # PHYSICS INSIGHTS
    # ============================================
    if stats:
        st.markdown('<div class="section-header">Physics Insights</div>', unsafe_allow_html=True)
        
        final_stats = stats[-1]
        morphology = st.session_state.simulation_results['morphology']
        
        insights = []
        
        # Analyze results
        if final_stats['std_c'] > 0.25:
            insights.append("✅ **Strong spinodal decomposition**: Clear phase separation into Li-rich and Li-poor domains")
        elif final_stats['std_c'] > 0.1:
            insights.append("⚠️ **Moderate phase separation**: Beginning of domain formation")
        else:
            insights.append("🔵 **Homogeneous/Solid solution**: System remains mixed or phase separation suppressed")
        
        if use_elasticity:
            if morphology['num_domains'] > 10 and morphology['mean_size'] < 100:
                insights.append("📐 **Elasticity-mediated patterning**: Multiple small domains indicating elastic strain effects")
            else:
                insights.append("📏 **Elastic coarsening suppression**: Limited domain growth due to strain energy")
        
        if "ab-plane" in sim_type and anisotropy > 100:
            insights.append("➡️ **Anisotropic domain growth**: Elongated domains along fast diffusion direction (b-axis)")
        
        if bc_type == "Potentiostatic (Butler-Volmer)":
            if np.abs(params.get('eta', 0)) > 0.05:
                insights.append("⚡ **High overpotential driving**: Rapid phase transformation")
            else:
                insights.append("🔋 **Moderate electrochemical driving**: Controlled phase evolution")
        
        # Display insights
        for insight in insights:
            st.info(insight)
        
        # Theoretical summary
        with st.expander("🧪 Theoretical Summary", expanded=False):
            st.markdown(f"""
            **Spinodal Analysis:**
            
            - Critical point: Ω/RT = 2.0
            - Current: Ω/RT = {Omega_RT:.1f} ({'Above' if Omega_RT > 2.0 else 'Below'} critical)
            - Spinodal region: x ∈ [{max(0, 0.5 - np.sqrt(1 - 2/Omega_RT)/2):.3f}, {min(1, 0.5 + np.sqrt(1 - 2/Omega_RT)/2):.3f}]
            
            **Characteristic Length Scale:**
            
            - Interface width: ξ ≈ √(κ/Ω) ≈ {np.sqrt(params['kappa'] / (params['Omega_RT'] * 8.314 * params['T_K'])):.2e} m
            - Characteristic wavelength: λ ≈ 2πξ ≈ {2*np.pi*np.sqrt(params['kappa'] / (params['Omega_RT'] * 8.314 * params['T_K'])):.2e} m
            
            **Kinetics:**
            
            - Characteristic time: τ ≈ ξ²/D ≈ {params['kappa']/(params['Mx'] * params['Omega_RT'] * 8.314 * params['T_K']):.2e} s
            - Fast/slow diffusion ratio: {anisotropy}:1
            """)

else:
    # ============================================
    # WELCOME / INSTRUCTIONS
    # ============================================
    st.markdown("""
    ## 📖 Welcome to the LiFePO₄ Phase-Field Simulator
    
    This tool simulates **spinodal decomposition** in LiFePO₄ nanoparticles during battery charging/discharging.
    
    ### Key Features:
    
    1. **Anisotropic Elasticity**: Models 6.8% anisotropic lattice mismatch between LiFePO₄ and FePO₄
    2. **Anisotropic Diffusion**: Fast Li transport along b-axis (1D channels)
    3. **Electrochemical BCs**: Butler-Volmer kinetics with stress coupling
    4. **Real-time Visualization**: Interactive animations and comprehensive analysis
    5. **Numba JIT Acceleration**: High-performance computation with type safety
    
    ### How to Use:
    
    1. **Configure parameters** in the sidebar
    2. **Click "Run Simulation"** to start
    3. **Analyze results** in the main panel
    
    ### Troubleshooting Numba Errors:
    
    If you encounter Numba TypingError:
    
    1. **Enable Debug Mode** in sidebar
    2. **Disable Numba JIT** temporarily
    3. **Clear Cache** to force recompilation
    4. Check parameter types and ranges
    
    ### Quick Start:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⚡ Load Example Parameters", type="secondary"):
            # This would set parameters in a full implementation
            st.info("Example parameters loaded! Adjust as needed and click 'Run Simulation'")
            st.rerun()
    
    with col2:
        if st.button("🔧 Test Numba Compilation", type="secondary"):
            # Test Numba compilation
            st.info("Testing Numba compilation...")
            try:
                # Create test data
                test_c = np.random.rand(10, 10).astype(np.float64)
                test_result = chemical_potential_numba(test_c, 13.0, 1e-11, 1.0)
                st.success(f"✅ Numba compilation successful! Result shape: {test_result.shape}")
            except Exception as e:
                st.error(f"❌ Numba compilation failed: {e}")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p><strong>LiFePO₄ Phase-Field Simulator</strong> | Based on Cahn-Hilliard theory with anisotropic elasticity</p>
    <p>Simulates spinodal decomposition in battery cathodes • Supports academic research</p>
    <p style="font-size: 0.8rem; color: #999;">
        Version 2.0 | Fixed Numba TypingError | Added type safety and debugging tools
    </p>
</div>
""", unsafe_allow_html=True)
